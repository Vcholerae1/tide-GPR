import os
import warnings
from collections.abc import Sequence
from typing import Any

import torch

from .. import staggered
from ..callbacks import Callback, CallbackState
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_2d, _normalize_pml_width_2d
from ..padding import create_or_pad, zero_interior
from ..utils import C0, EP0, compile_material_coefficients
from .common import _init_polarization_state, _make_compute_stream, _pad_dispersion_for_model
from .tm2d_autograd import MaxwellTMForwardFunc
from .tm2d_helpers import (
    _init_tm_wavefield,
    _physical_tm2d_callback_wavefields,
    _prepare_tm2d_source_injection,
    _resolve_tm2d_storage_spec,
    _unscale_tm2d_outputs,
)
from .tm2d_python import maxwell_python


def maxwell_c_cuda(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    stencil: int,
    pml_width: int | Sequence[int],
    max_vel: float | None,
    Ey_0: torch.Tensor | None,
    Hx_0: torch.Tensor | None,
    Hz_0: torch.Tensor | None,
    m_Ey_x_0: torch.Tensor | None,
    m_Ey_z_0: torch.Tensor | None,
    m_Hx_z_0: torch.Tensor | None,
    m_Hz_x_0: torch.Tensor | None,
    nt: int | None,
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: bool | None,
    forward_callback: Callback | None,
    backward_callback: Callback | None,
    callback_frequency: int,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    storage_chunk_steps: int = 0,
    n_threads: int | None = None,
    dispersion: DebyeDispersion | None = None,
    execution_backend: str = "standard",
):
    """Performs Maxwell propagation using the native C/CUDA backend."""
    from .. import backend_utils

    if epsilon.ndim not in {2, 3}:
        raise RuntimeError("epsilon must be 2D or batched 3D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    device = epsilon.device
    dtype = epsilon.dtype
    original_dtype = dtype
    original_device = device
    model_batched = epsilon.ndim == 3
    model_ny, model_nx = epsilon.shape[-2:]
    execution_backend_str = str(execution_backend).lower()
    if execution_backend_str not in {
        "standard",
        "direct_epsilon_grad",
        "direct_material_grad",
        "direct_material_endpoint_grad",
        "direct_material_grad_ecurl",
    }:
        raise ValueError(
            "execution_backend must be 'standard', 'direct_epsilon_grad', "
            "'direct_material_grad', 'direct_material_endpoint_grad', or "
            "'direct_material_grad_ecurl', "
            f"but got {execution_backend!r}."
        )

    backend_device = device
    if device.type not in {"cpu", "cuda"}:
        raise NotImplementedError("C/CUDA backend supports only cpu and cuda devices.")
    grid_spacing = _normalize_grid_spacing_2d(grid_spacing)
    dy, dx = grid_spacing

    n_threads_val = 0
    if n_threads is not None:
        n_threads_val = int(n_threads)
        if n_threads_val < 0:
            raise ValueError("n_threads must be >= 0 when provided.")

    pml_width_list = _normalize_pml_width_2d(pml_width)
    physical_snapshot_storage_requested = (
        os.environ.get("TIDE_TM2D_PHYSICAL_SNAPSHOT_STORAGE", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = source_amplitude.shape[-1]
    nt_steps = int(nt)

    gradient_sampling_interval = int(model_gradient_sampling_interval)
    if gradient_sampling_interval < 1:
        gradient_sampling_interval = 1
    if nt_steps > 0:
        gradient_sampling_interval = min(gradient_sampling_interval, nt_steps)

    if source_amplitude is not None and source_amplitude.numel() > 0:
        n_shots = source_amplitude.shape[0]
    elif source_location is not None and source_location.numel() > 0:
        n_shots = source_location.shape[0]
    elif receiver_location is not None and receiver_location.numel() > 0:
        n_shots = receiver_location.shape[0]
    else:
        n_shots = 1

    if model_batched and int(epsilon.shape[0]) != n_shots:
        raise RuntimeError(
            "Batched model count must match the effective shot count after normalization."
        )

    if max_vel is None:
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    pml_freq = 0.5 / dt

    fd_pad = stencil // 2
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]

    padded_ny = model_ny + total_pad[0] + total_pad[1]
    padded_nx = model_nx + total_pad[2] + total_pad[3]
    padded_size = (
        (int(epsilon.shape[0]), padded_ny, padded_nx)
        if model_batched
        else (padded_ny, padded_nx)
    )

    epsilon_padded = create_or_pad(
        epsilon, total_pad, device, dtype, padded_size, mode="replicate"
    )
    sigma_padded = create_or_pad(
        sigma, total_pad, device, dtype, padded_size, mode="replicate"
    )
    mu_padded = create_or_pad(
        mu, total_pad, device, dtype, padded_size, mode="replicate"
    )

    dispersion_padded = _pad_dispersion_for_model(
        dispersion,
        model_shape=tuple(epsilon.shape[-2:]),
        total_pad=total_pad,
        padded_size=(padded_ny, padded_nx),
        device=device,
        dtype=dtype,
    )
    material = compile_material_coefficients(
        epsilon_padded,
        sigma_padded,
        mu_padded,
        dt,
        dispersion=dispersion_padded,
    )
    ca = (
        material["ca"].contiguous()
        if model_batched
        else material["ca"][None, :, :].contiguous()
    )
    cb = (
        material["cb"].contiguous()
        if model_batched
        else material["cb"][None, :, :].contiguous()
    )
    cq = (
        material["cq"].contiguous()
        if model_batched
        else material["cq"][None, :, :].contiguous()
    )
    ca_phys = ca
    cb_phys = cb
    cq_phys = cq
    has_dispersion = bool(material["has_dispersion"])
    debye = material.get("debye")

    flat_model_shape = padded_ny * padded_nx
    if source_location is not None and source_location.numel() > 0:
        source_y = source_location[..., 0] + total_pad[0]
        source_x = source_location[..., 1] + total_pad[2]
        sources_i = (source_y * padded_nx + source_x).long().contiguous()
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        receiver_y = receiver_location[..., 0] + total_pad[0]
        receiver_x = receiver_location[..., 1] + total_pad[2]
        receivers_i = (receiver_y * padded_nx + receiver_x).long().contiguous()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    source_coeff = -1.0 / (dx * dy)
    cb_at_src: torch.Tensor | None = None
    if n_sources > 0:
        if model_batched:
            cb_flat = cb_phys.reshape(n_shots, flat_model_shape)
        else:
            cb_flat = cb_phys.expand(n_shots, -1, -1).reshape(n_shots, flat_model_shape)
        cb_at_src = cb_flat.gather(1, sources_i)

    source_injection, _f_shot = _prepare_tm2d_source_injection(
        source_amplitude=source_amplitude,
        cb_at_src=cb_at_src,
        source_coeff=source_coeff,
        dtype=dtype,
        n_shots=n_shots,
        n_sources=n_sources,
        nt_steps=nt_steps,
    )
    scale_ctx: dict[str, Any] | None = None

    size_with_batch = (n_shots, padded_ny, padded_nx)
    Ey = _init_tm_wavefield(
        Ey_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    Hx = _init_tm_wavefield(
        Hx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    Hz = _init_tm_wavefield(
        Hz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_Ey_x = _init_tm_wavefield(
        m_Ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_Ey_z = _init_tm_wavefield(
        m_Ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_Hx_z = _init_tm_wavefield(
        m_Hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_Hz_x = _init_tm_wavefield(
        m_Hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )

    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], [1, 0, 0, 1]):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)

    (
        ay,
        ay_h,
        ax,
        ax_h,
        by,
        by_h,
        bx,
        bx_h,
        ky,
        ky_h,
        kx,
        kx_h,
    ) = staggered.set_pml_profiles(
        pml_width=pml_width_list,
        accuracy=stencil,
        fd_pad=fd_pad_list,
        dt=dt,
        grid_spacing=grid_spacing,
        max_vel=max_vel,
        dtype=dtype,
        device=device,
        pml_freq=pml_freq,
        ny=padded_ny,
        nx=padded_nx,
        eps_scale=EP0,
    )

    ay_flat = ay.squeeze().contiguous()
    ay_h_flat = ay_h.squeeze().contiguous()
    ax_flat = ax.squeeze().contiguous()
    ax_h_flat = ax_h.squeeze().contiguous()
    by_flat = by.squeeze().contiguous()
    by_h_flat = by_h.squeeze().contiguous()
    bx_flat = bx.squeeze().contiguous()
    bx_h_flat = bx_h.squeeze().contiguous()
    ky_flat = ky.squeeze().contiguous()
    ky_h_flat = ky_h.squeeze().contiguous()
    kx_flat = kx.squeeze().contiguous()
    kx_h_flat = kx_h.squeeze().contiguous()
    f = source_injection.contiguous()

    pml_y0 = fd_pad_list[0] + pml_width_list[0]
    pml_y1 = padded_ny - fd_pad_list[1] - pml_width_list[1]
    pml_x0 = fd_pad_list[2] + pml_width_list[2]
    pml_x1 = padded_nx - fd_pad_list[3] - pml_width_list[3]

    requires_grad = epsilon.requires_grad or sigma.requires_grad
    direct_epsilon_grad_requested = execution_backend_str == "direct_epsilon_grad"
    direct_material_grad_requested = execution_backend_str == "direct_material_grad"
    direct_material_endpoint_requested = (
        execution_backend_str == "direct_material_endpoint_grad"
    )
    direct_material_ecurl_requested = (
        execution_backend_str == "direct_material_grad_ecurl"
    )
    direct_epsilon_grad = direct_epsilon_grad_requested and requires_grad
    direct_material_grad = direct_material_grad_requested and requires_grad
    direct_material_endpoint = direct_material_endpoint_requested and requires_grad
    direct_material_ecurl = direct_material_ecurl_requested and requires_grad
    if direct_epsilon_grad:
        if device.type != "cuda":
            raise NotImplementedError(
                "execution_backend='direct_epsilon_grad' is CUDA-only."
            )
        if model_batched:
            raise NotImplementedError(
                "execution_backend='direct_epsilon_grad' does not support batched models yet."
            )
        if dispersion is not None:
            raise NotImplementedError(
                "execution_backend='direct_epsilon_grad' does not support Debye dispersion."
            )
        if not epsilon.requires_grad or sigma.requires_grad:
            raise NotImplementedError(
                "execution_backend='direct_epsilon_grad' currently requires "
                "epsilon.requires_grad=True and sigma.requires_grad=False."
            )
        if backward_callback is not None:
            raise NotImplementedError(
                "execution_backend='direct_epsilon_grad' does not support backward_callback yet."
            )
    if direct_material_grad or direct_material_endpoint or direct_material_ecurl:
        if device.type != "cuda":
            raise NotImplementedError(
                f"execution_backend={execution_backend_str!r} is CUDA-only."
            )
        if model_batched:
            raise NotImplementedError(
                f"execution_backend={execution_backend_str!r} does not support "
                "batched models yet."
            )
        if dispersion is not None:
            raise NotImplementedError(
                f"execution_backend={execution_backend_str!r} does not support "
                "Debye dispersion."
            )
        if direct_material_grad and gradient_sampling_interval != 1:
            raise NotImplementedError(
                "execution_backend='direct_material_grad' currently requires "
                "model_gradient_sampling_interval=1."
            )
        if forward_callback is not None or backward_callback is not None:
            raise NotImplementedError(
                f"execution_backend={execution_backend_str!r} does not support "
                "callbacks yet."
            )
        if (
            source_amplitude is not None
            and isinstance(source_amplitude, torch.Tensor)
            and source_amplitude.requires_grad
        ):
            raise NotImplementedError(
                f"execution_backend={execution_backend_str!r} currently supports "
                "model gradients only, not source-amplitude gradients."
            )
        if not direct_material_ecurl:
            f = f.detach()
    if has_dispersion and (requires_grad or (save_snapshots is True)):
        warnings.warn(
            "Debye native backend currently supports forward inference only; "
            "falling back to the Python backend for gradients or snapshot storage.",
            RuntimeWarning,
        )
        return maxwell_python(
            epsilon,
            sigma,
            mu,
            grid_spacing,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            stencil,
            pml_width,
            max_vel,
            Ey_0,
            Hx_0,
            Hz_0,
            m_Ey_x_0,
            m_Ey_z_0,
            m_Hx_z_0,
            m_Hz_x_0,
            nt,
            model_gradient_sampling_interval,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            save_snapshots,
            forward_callback,
            backward_callback,
            callback_frequency,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            n_threads,
            dispersion,
            execution_backend,
            validate_material_inputs=False,
        )

    if torch._C._are_functorch_transforms_active():
        raise NotImplementedError(
            "torch.func transforms are not supported for the C/CUDA backend."
        )

    _, _, storage_bytes_per_elem, storage_format = _resolve_tm2d_storage_spec(
        storage_compression=storage_compression,
        dtype=dtype,
        device=device,
        context="storage_compression",
    )

    do_save_snapshots = requires_grad if save_snapshots is None else save_snapshots
    if requires_grad and save_snapshots is False:
        warnings.warn(
            "save_snapshots=False but model parameters require gradients. "
            "Backward pass will fail.",
            UserWarning,
        )

    storage_mode_str = storage_mode.lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )
    if device.type == "cpu" and storage_mode_str == "cpu":
        storage_mode_str = "device"

    needs_storage = do_save_snapshots and requires_grad
    effective_storage_mode_str = storage_mode_str
    if not needs_storage:
        if effective_storage_mode_str == "auto":
            effective_storage_mode_str = "none"
    else:
        if effective_storage_mode_str == "none":
            raise ValueError(
                "storage_mode='none' is not compatible with gradient computation "
                "when model parameters require gradients."
            )
        if effective_storage_mode_str == "auto":
            num_elements_per_shot = padded_ny * padded_nx
            shot_bytes_uncomp = num_elements_per_shot * storage_bytes_per_elem
            n_stored = (
                nt_steps + gradient_sampling_interval - 1
            ) // gradient_sampling_interval
            total_bytes = n_stored * n_shots * shot_bytes_uncomp * 2

            limit_device = (
                storage_bytes_limit_device
                if storage_bytes_limit_device is not None
                else float("inf")
            )
            limit_host = (
                storage_bytes_limit_host
                if storage_bytes_limit_host is not None
                else float("inf")
            )
            if device.type == "cuda" and total_bytes <= limit_device:
                effective_storage_mode_str = "device"
            elif total_bytes <= limit_host:
                effective_storage_mode_str = "cpu"
            else:
                effective_storage_mode_str = "disk"

            warnings.warn(
                f"storage_mode='auto' selected storage_mode='{effective_storage_mode_str}' "
                f"for estimated storage size {total_bytes / 1e9:.2f} GB.",
                RuntimeWarning,
            )
    if (
        (
            direct_epsilon_grad
            or direct_material_grad
            or direct_material_endpoint
            or direct_material_ecurl
        )
        and effective_storage_mode_str != "device"
    ):
        raise NotImplementedError(
            f"execution_backend={execution_backend_str!r} currently supports only "
            "storage_mode='device'."
        )
    if (
        physical_snapshot_storage_requested
        and not (
            requires_grad
            and do_save_snapshots
            and device.type == "cuda"
            and effective_storage_mode_str == "device"
            and not direct_epsilon_grad
            and not direct_material_grad
            and not direct_material_endpoint
            and not direct_material_ecurl
            and forward_callback is None
            and backward_callback is None
        )
    ):
        warnings.warn(
            "TIDE_TM2D_PHYSICAL_SNAPSHOT_STORAGE=1 requires standard CUDA "
            "gradients, storage_mode='device', and no callbacks; using full-domain "
            "snapshot storage.",
            RuntimeWarning,
        )

    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca_phys,
        "cb": cb_phys,
        "cq": cq_phys,
    }
    if dispersion is not None:
        callback_models["dispersion"] = dispersion

    if requires_grad and do_save_snapshots:
        direct_model_grad = (
            direct_epsilon_grad
            or direct_material_grad
            or direct_material_endpoint
            or direct_material_ecurl
        )
        ca_arg = ca.detach() if direct_model_grad else ca
        cb_arg = cb.detach() if direct_model_grad else cb
        cq_arg = cq.detach() if direct_model_grad else cq
        epsilon_direct = (
            epsilon_padded
            if direct_model_grad
            else torch.empty(0, device=device, dtype=dtype)
        )
        sigma_direct = (
            sigma_padded
            if direct_material_grad
            or direct_material_endpoint
            or direct_material_ecurl
            else torch.empty(0, device=device, dtype=dtype)
        )
        result = MaxwellTMForwardFunc.apply(
            ca_arg,
            cb_arg,
            cq_arg,
            f,
            ay_flat,
            by_flat,
            ay_h_flat,
            by_h_flat,
            ax_flat,
            bx_flat,
            ax_h_flat,
            bx_h_flat,
            ky_flat,
            ky_h_flat,
            kx_flat,
            kx_h_flat,
            sources_i,
            receivers_i,
            1.0 / dy,
            1.0 / dx,
            dt,
            nt_steps,
            n_shots,
            padded_ny,
            padded_nx,
            n_sources,
            n_receivers,
            gradient_sampling_interval,
            stencil,
            model_batched,
            model_batched,
            model_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            tuple(fd_pad_list),
            tuple(pml_width_list),
            callback_models,
            forward_callback,
            backward_callback,
            callback_frequency,
            scale_ctx,
            effective_storage_mode_str,
            storage_format,
            storage_path,
            storage_compression,
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            n_threads_val,
            backend_device,
            3
            if direct_material_endpoint
            else 4
            if direct_material_ecurl
            else 2
            if direct_material_grad
            else 1
            if direct_epsilon_grad
            else 0,
            epsilon_direct,
            sigma_direct,
        )
        if len(result) == 9:
            (
                Ey_out,
                Hx_out,
                Hz_out,
                m_Ey_x_out,
                m_Ey_z_out,
                m_Hx_z_out,
                m_Hz_x_out,
                receiver_amplitudes,
                _ctx_handle,
            ) = result
        else:
            (
                Ey_out,
                Hx_out,
                Hz_out,
                m_Ey_x_out,
                m_Ey_z_out,
                m_Hx_z_out,
                m_Hz_x_out,
                receiver_amplitudes,
            ) = result

        (
            Ey_out,
            Hx_out,
            Hz_out,
            m_Ey_x_out,
            m_Ey_z_out,
            m_Hx_z_out,
            m_Hz_x_out,
            receiver_amplitudes,
        ) = _unscale_tm2d_outputs(
            scale_ctx=scale_ctx,
            Ey=Ey_out,
            Hx=Hx_out,
            Hz=Hz_out,
            m_Ey_x=m_Ey_x_out,
            m_Ey_z=m_Ey_z_out,
            m_Hx_z=m_Hx_z_out,
            m_Hz_x=m_Hz_x_out,
            receiver_amplitudes=receiver_amplitudes,
        )

        s = (
            slice(None),
            slice(
                fd_pad_list[0],
                padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None,
            ),
            slice(
                fd_pad_list[2],
                padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None,
            ),
        )
        result = (
            Ey_out[s],
            Hx_out[s],
            Hz_out[s],
            m_Ey_x_out[s],
            m_Ey_z_out[s],
            m_Hx_z_out[s],
            m_Hz_x_out[s],
            receiver_amplitudes,
        )
        if original_device != device or original_dtype != dtype:
            result = tuple(
                t.to(device=original_device, dtype=original_dtype)
                if t.is_floating_point()
                else t.to(device=original_device)
                for t in result
            )
        return result

    try:
        forward_func = backend_utils.get_backend_function(
            "maxwell_tm",
            "forward",
            stencil,
            dtype,
            backend_device,
        )
    except AttributeError as exc:
        raise RuntimeError(
            f"C/CUDA backend function not available for accuracy={stencil}, "
            f"dtype={dtype}, device={device}. Error: {exc}"
        ) from exc

    device_idx = (
        device.index if device.type == "cuda" and device.index is not None else 0
    )
    if n_receivers > 0:
        receiver_amplitudes = torch.zeros(
            nt_steps,
            n_shots,
            n_receivers,
            device=device,
            dtype=dtype,
        )
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

    polarization = torch.empty(0, device=device, dtype=dtype)
    ey_prev = torch.empty(0, device=device, dtype=dtype)
    debye_a = torch.empty(0, device=device, dtype=dtype)
    debye_b = torch.empty(0, device=device, dtype=dtype)
    debye_cp = torch.empty(0, device=device, dtype=dtype)
    n_poles = 0
    if has_dispersion and debye is not None:
        n_poles = int(debye["n_poles"])
        polarization = _init_polarization_state(
            n_shots=n_shots,
            n_poles=n_poles,
            spatial_shape=(padded_ny, padded_nx),
            device=device,
            dtype=dtype,
        ).contiguous()
        ey_prev = torch.empty_like(Ey, dtype=dtype)
        debye_a = debye["a"].contiguous()
        debye_b = debye["b"].contiguous()
        debye_cp = debye["cp"].contiguous()

    effective_callback_freq = nt_steps if forward_callback is None else callback_frequency
    compute_stream_handle, _keepalive = _make_compute_stream(device)

    for step in range(0, nt_steps, effective_callback_freq):
        step_nt = min(nt_steps - step, effective_callback_freq)
        forward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(f),
            backend_utils.tensor_to_ptr(Ey),
            backend_utils.tensor_to_ptr(Hx),
            backend_utils.tensor_to_ptr(Hz),
            backend_utils.tensor_to_ptr(m_Ey_x),
            backend_utils.tensor_to_ptr(m_Ey_z),
            backend_utils.tensor_to_ptr(m_Hx_z),
            backend_utils.tensor_to_ptr(m_Hz_x),
            backend_utils.tensor_to_ptr(debye_a),
            backend_utils.tensor_to_ptr(debye_b),
            backend_utils.tensor_to_ptr(debye_cp),
            backend_utils.tensor_to_ptr(polarization),
            backend_utils.tensor_to_ptr(ey_prev),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
            n_poles,
            backend_utils.tensor_to_ptr(ay_flat),
            backend_utils.tensor_to_ptr(by_flat),
            backend_utils.tensor_to_ptr(ay_h_flat),
            backend_utils.tensor_to_ptr(by_h_flat),
            backend_utils.tensor_to_ptr(ax_flat),
            backend_utils.tensor_to_ptr(bx_flat),
            backend_utils.tensor_to_ptr(ax_h_flat),
            backend_utils.tensor_to_ptr(bx_h_flat),
            backend_utils.tensor_to_ptr(ky_flat),
            backend_utils.tensor_to_ptr(ky_h_flat),
            backend_utils.tensor_to_ptr(kx_flat),
            backend_utils.tensor_to_ptr(kx_h_flat),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            1.0 / dy,
            1.0 / dx,
            dt,
            step_nt,
            n_shots,
            padded_ny,
            padded_nx,
            n_sources,
            n_receivers,
            gradient_sampling_interval,
            has_dispersion,
            model_batched,
            model_batched,
            model_batched,
            step,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            n_threads_val,
            device_idx,
            compute_stream_handle,
        )

        if forward_callback is not None:
            callback_wavefields = _physical_tm2d_callback_wavefields(
                {
                    "Ey": Ey,
                    "Hx": Hx,
                    "Hz": Hz,
                    "m_Ey_x": m_Ey_x,
                    "m_Ey_z": m_Ey_z,
                    "m_Hx_z": m_Hx_z,
                    "m_Hz_x": m_Hz_x,
                },
                scale_ctx=scale_ctx,
            )
            if has_dispersion:
                callback_wavefields["polarization"] = polarization.sum(dim=1)
            forward_callback(
                CallbackState(
                    dt=dt,
                    step=step + step_nt,
                    nt=nt_steps,
                    wavefields=callback_wavefields,
                    models=callback_models,
                    gradients=None,
                    fd_pad=fd_pad_list,
                    pml_width=pml_width_list,
                    is_backward=False,
                    grid_spacing=[dy, dx],
                )
            )

    (
        Ey,
        Hx,
        Hz,
        m_Ey_x,
        m_Ey_z,
        m_Hx_z,
        m_Hz_x,
        receiver_amplitudes,
    ) = _unscale_tm2d_outputs(
        scale_ctx=scale_ctx,
        Ey=Ey,
        Hx=Hx,
        Hz=Hz,
        m_Ey_x=m_Ey_x,
        m_Ey_z=m_Ey_z,
        m_Hx_z=m_Hx_z,
        m_Hz_x=m_Hz_x,
        receiver_amplitudes=receiver_amplitudes,
        inplace_float_outputs=True,
    )

    s = (
        slice(None),
        slice(
            fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None
        ),
        slice(
            fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None
        ),
    )
    result = (
        Ey[s],
        Hx[s],
        Hz[s],
        m_Ey_x[s],
        m_Ey_z[s],
        m_Hx_z[s],
        m_Hz_x[s],
        receiver_amplitudes,
    )
    if original_device != device or original_dtype != dtype:
        result = tuple(
            t.to(device=original_device, dtype=original_dtype)
            if t.is_floating_point()
            else t.to(device=original_device)
            for t in result
        )
    return result


__all__ = ["maxwell_c_cuda"]
