import warnings
from collections.abc import Sequence

import torch

from ..callbacks import Callback, CallbackState
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_3d, _normalize_pml_width_3d
from ..storage import _normalize_storage_compression, _resolve_storage_compression
from ..utils import C0, compile_material_coefficients
from .common import (
    _init_polarization_state,
    _make_compute_stream,
    _pad_dispersion_for_model,
)
from .maxwell3d_autograd import Maxwell3DForwardFunc
from .maxwell3d_python import maxwell3d_python
from .validation_internal import _COMPONENT_TO_INDEX_3D


def maxwell3d_c_cuda(
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
    Ex_0: torch.Tensor | None,
    Ey_0: torch.Tensor | None,
    Ez_0: torch.Tensor | None,
    Hx_0: torch.Tensor | None,
    Hy_0: torch.Tensor | None,
    Hz_0: torch.Tensor | None,
    m_hz_y_0: torch.Tensor | None,
    m_hy_z_0: torch.Tensor | None,
    m_hx_z_0: torch.Tensor | None,
    m_hz_x_0: torch.Tensor | None,
    m_hy_x_0: torch.Tensor | None,
    m_hx_y_0: torch.Tensor | None,
    m_ey_z_0: torch.Tensor | None,
    m_ez_y_0: torch.Tensor | None,
    m_ez_x_0: torch.Tensor | None,
    m_ex_z_0: torch.Tensor | None,
    m_ex_y_0: torch.Tensor | None,
    m_ey_x_0: torch.Tensor | None,
    nt: int | None,
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: bool | None,
    forward_callback: Callback | None,
    backward_callback: Callback | None,
    callback_frequency: int,
    source_component: str,
    receiver_component: str,
    execution_backend: str = "standard",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    storage_chunk_steps: int = 0,
    n_threads: int | None = None,
    dispersion: DebyeDispersion | None = None,
):
    """3D C/CUDA forward propagation path with Python fallback for gradients."""
    from .. import backend_utils, staggered
    from ..padding import create_or_pad, zero_interior
    del (
        storage_chunk_steps,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    if epsilon.ndim not in {3, 4}:
        raise RuntimeError("epsilon must be 3D or batched 4D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    storage_mode_str = str(storage_mode).lower()
    execution_backend_str = str(execution_backend).lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )
    if execution_backend_str != "standard":
        raise ValueError(
            "execution_backend must be 'standard', "
            f"but got {execution_backend!r}"
        )
    execution_backend_id = 0

    n_threads_val = 0
    if n_threads is not None:
        n_threads_val = int(n_threads)
        if n_threads_val < 0:
            raise ValueError("n_threads must be >= 0 when provided.")

    storage_kind = _normalize_storage_compression(storage_compression)
    requires_grad = epsilon.requires_grad or sigma.requires_grad
    functorch_active = torch._C._are_functorch_transforms_active()
    device = epsilon.device
    storage_bytes_per_elem = epsilon.element_size()
    def _fallback_reason(reason: str):
        fallback_storage_mode = storage_mode
        if str(fallback_storage_mode).lower() in {"cpu", "disk", "auto"}:
            fallback_storage_mode = "device"
        warnings.warn(
            f"{reason}; falling back to Python backend.",
            RuntimeWarning,
        )
        return maxwell3d_python(
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
            Ex_0,
            Ey_0,
            Ez_0,
            Hx_0,
            Hy_0,
            Hz_0,
            m_hz_y_0,
            m_hy_z_0,
            m_hx_z_0,
            m_hz_x_0,
            m_hy_x_0,
            m_hx_y_0,
            m_ey_z_0,
            m_ez_y_0,
            m_ez_x_0,
            m_ex_z_0,
            m_ex_y_0,
            m_ey_x_0,
            nt,
            model_gradient_sampling_interval,
            0.0,
            0.0,
            False,
            save_snapshots,
            forward_callback,
            backward_callback,
            callback_frequency,
            source_component,
            receiver_component,
            execution_backend_str,
            storage_mode=fallback_storage_mode,
            storage_compression=storage_compression,
            n_threads=n_threads,
            dispersion=dispersion,
        )

    if not backend_utils.is_backend_available():
        return _fallback_reason("C/CUDA backend library is unavailable")

    if functorch_active:
        return _fallback_reason(
            "torch.func transforms are not supported for 3D C/CUDA backend"
        )

    if requires_grad:
        if storage_kind != "none":
            if device.type != "cuda":
                return _fallback_reason(
                    "3D C backend does not support compressed snapshot storage"
                )
            _, _, storage_bytes_per_elem, _ = _resolve_storage_compression(
                storage_compression,
                epsilon.dtype,
                device,
                context="storage_compression",
            )
        if storage_mode_str == "none":
            return _fallback_reason(
                "storage_mode='none' is incompatible with 3D gradient computation"
            )
    else:
        if storage_kind != "none":
            warnings.warn(
                "3D C/CUDA forward path ignores storage_compression when gradients are not requested.",
                RuntimeWarning,
            )
        if save_snapshots:
            warnings.warn(
                "save_snapshots is ignored in 3D C/CUDA forward-only path.",
                RuntimeWarning,
            )
        if backward_callback is not None:
            warnings.warn(
                "backward_callback is ignored when model parameters do not require gradients.",
                RuntimeWarning,
            )

    dtype = epsilon.dtype
    model_batched = epsilon.ndim == 4
    model_nz, model_ny, model_nx = epsilon.shape[-3:]

    grid_spacing_list = _normalize_grid_spacing_3d(grid_spacing)
    dz, dy, dx = grid_spacing_list
    pml_width_list = _normalize_pml_width_3d(pml_width)

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

    effective_storage_mode_str = storage_mode_str
    if requires_grad:
        if device.type == "cpu" and effective_storage_mode_str in {"cpu", "disk"}:
            effective_storage_mode_str = "device"
        if effective_storage_mode_str == "auto":
            if device.type == "cpu":
                effective_storage_mode_str = "device"
            else:
                n_stored = (
                    nt_steps + gradient_sampling_interval - 1
                ) // gradient_sampling_interval
                shot_numel_est = model_nz * model_ny * model_nx
                shot_bytes_uncomp_est = shot_numel_est * storage_bytes_per_elem
                total_bytes = n_stored * n_shots * shot_bytes_uncomp_est * 6
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
                if total_bytes <= limit_device:
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
    else:
        if effective_storage_mode_str == "auto":
            effective_storage_mode_str = "device"

    if max_vel is None:
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    pml_freq = 0.5 / dt

    fd_pad = stencil // 2
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]

    padded_nz = model_nz + total_pad[0] + total_pad[1]
    padded_ny = model_ny + total_pad[2] + total_pad[3]
    padded_nx = model_nx + total_pad[4] + total_pad[5]

    padded_size = (
        (int(epsilon.shape[0]), padded_nz, padded_ny, padded_nx)
        if model_batched
        else (padded_nz, padded_ny, padded_nx)
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
        model_shape=tuple(epsilon.shape[-3:]),
        total_pad=total_pad,
        padded_size=(padded_nz, padded_ny, padded_nx),
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
    ca = material["ca"]
    cb = material["cb"]
    cq = material["cq"]
    has_dispersion = bool(material["has_dispersion"])
    debye = material.get("debye")
    size_with_batch = (n_shots, padded_nz, padded_ny, padded_nx)

    def init_wavefield(field_0: torch.Tensor | None) -> torch.Tensor:
        if field_0 is not None:
            if field_0.ndim == 3:
                field_0 = field_0[None, :, :, :].expand(n_shots, -1, -1, -1)
            return create_or_pad(
                field_0,
                fd_pad_list,
                device,
                dtype,
                size_with_batch,
                mode="constant",
            ).contiguous()
        return torch.zeros(size_with_batch, device=device, dtype=dtype)

    Ex = init_wavefield(Ex_0)
    Ey = init_wavefield(Ey_0)
    Ez = init_wavefield(Ez_0)
    Hx = init_wavefield(Hx_0)
    Hy = init_wavefield(Hy_0)
    Hz = init_wavefield(Hz_0)

    m_hz_y = init_wavefield(m_hz_y_0)
    m_hy_z = init_wavefield(m_hy_z_0)
    m_hx_z = init_wavefield(m_hx_z_0)
    m_hz_x = init_wavefield(m_hz_x_0)
    m_hy_x = init_wavefield(m_hy_x_0)
    m_hx_y = init_wavefield(m_hx_y_0)
    m_ey_z = init_wavefield(m_ey_z_0)
    m_ez_y = init_wavefield(m_ez_y_0)
    m_ez_x = init_wavefield(m_ez_x_0)
    m_ex_z = init_wavefield(m_ex_z_0)
    m_ex_y = init_wavefield(m_ex_y_0)
    m_ey_x = init_wavefield(m_ey_x_0)

    pml_aux = [
        (m_hz_y, 1),
        (m_hy_z, 0),
        (m_hx_z, 0),
        (m_hz_x, 2),
        (m_hy_x, 2),
        (m_hx_y, 1),
        (m_ey_z, 0),
        (m_ez_y, 1),
        (m_ez_x, 2),
        (m_ex_z, 0),
        (m_ex_y, 1),
        (m_ey_x, 2),
    ]
    for wf, dim in pml_aux:
        zero_interior(wf, fd_pad_list, pml_width_list, dim)

    pml_ab_profiles, pml_k_profiles = staggered.set_pml_profiles_3d(
        pml_width=pml_width_list,
        accuracy=stencil,
        fd_pad=fd_pad_list,
        dt=dt,
        grid_spacing=grid_spacing_list,
        max_vel=max_vel,
        dtype=dtype,
        device=device,
        pml_freq=pml_freq,
        nz=padded_nz,
        ny=padded_ny,
        nx=padded_nx,
    )
    (
        az,
        az_h,
        ay,
        ay_h,
        ax,
        ax_h,
        bz,
        bz_h,
        by,
        by_h,
        bx,
        bx_h,
    ) = pml_ab_profiles
    kz, kz_h, ky, ky_h, kx, kx_h = pml_k_profiles

    az_flat = az.reshape(-1).contiguous()
    bz_flat = bz.reshape(-1).contiguous()
    az_h_flat = az_h.reshape(-1).contiguous()
    bz_h_flat = bz_h.reshape(-1).contiguous()
    ay_flat = ay.reshape(-1).contiguous()
    by_flat = by.reshape(-1).contiguous()
    ay_h_flat = ay_h.reshape(-1).contiguous()
    by_h_flat = by_h.reshape(-1).contiguous()
    ax_flat = ax.reshape(-1).contiguous()
    bx_flat = bx.reshape(-1).contiguous()
    ax_h_flat = ax_h.reshape(-1).contiguous()
    bx_h_flat = bx_h.reshape(-1).contiguous()

    kz_flat = kz.reshape(-1).contiguous()
    kz_h_flat = kz_h.reshape(-1).contiguous()
    ky_flat = ky.reshape(-1).contiguous()
    ky_h_flat = ky_h.reshape(-1).contiguous()
    kx_flat = kx.reshape(-1).contiguous()
    kx_h_flat = kx_h.reshape(-1).contiguous()

    flat_model_shape = padded_nz * padded_ny * padded_nx

    if source_location is not None and source_location.numel() > 0:
        source_z = source_location[..., 0] + total_pad[0]
        source_y = source_location[..., 1] + total_pad[2]
        source_x = source_location[..., 2] + total_pad[4]
        sources_i = ((source_z * padded_ny + source_y) * padded_nx + source_x).long()
        sources_i = sources_i.contiguous()
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        receiver_z = receiver_location[..., 0] + total_pad[0]
        receiver_y = receiver_location[..., 1] + total_pad[2]
        receiver_x = receiver_location[..., 2] + total_pad[4]
        receivers_i = (
            (receiver_z * padded_ny + receiver_y) * padded_nx + receiver_x
        ).long()
        receivers_i = receivers_i.contiguous()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    if n_sources > 0 and source_amplitude is not None and source_amplitude.numel() > 0:
        source_coeff = -1.0 / (dx * dy * dz)
        if model_batched:
            cb_flat = cb.reshape(n_shots, flat_model_shape)
        else:
            cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, sources_i)
        f = source_amplitude.permute(2, 0, 1).contiguous()
        f = (f * cb_at_src[None, :, :] * source_coeff).reshape(
            nt_steps * n_shots * n_sources
        )
        f = f.contiguous()
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    if n_receivers > 0:
        receiver_amplitudes = torch.zeros(
            nt_steps, n_shots, n_receivers, device=device, dtype=dtype
        )
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

    ca = ca.contiguous() if model_batched else ca[None, :, :, :].contiguous()
    cb = cb.contiguous() if model_batched else cb[None, :, :, :].contiguous()
    cq = cq.contiguous() if model_batched else cq[None, :, :, :].contiguous()

    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca,
        "cb": cb,
        "cq": cq,
    }
    if dispersion is not None:
        callback_models["dispersion"] = dispersion

    pml_z0 = fd_pad_list[0] + pml_width_list[0]
    pml_z1 = padded_nz - fd_pad_list[1] - pml_width_list[1]
    pml_y0 = fd_pad_list[2] + pml_width_list[2]
    pml_y1 = padded_ny - fd_pad_list[3] - pml_width_list[3]
    pml_x0 = fd_pad_list[4] + pml_width_list[4]
    pml_x1 = padded_nx - fd_pad_list[5] - pml_width_list[5]

    source_component_idx = _COMPONENT_TO_INDEX_3D[source_component]
    receiver_component_idx = _COMPONENT_TO_INDEX_3D[receiver_component]
    if has_dispersion and requires_grad:
        return _fallback_reason(
            "3D Debye C/CUDA path currently supports forward inference only"
        )
    if has_dispersion and device.type == "cpu":
        return _fallback_reason(
            "3D Debye CPU backend is not enabled yet"
        )
    if requires_grad:
        try:
            _ = backend_utils.get_backend_function(
                "maxwell_3d", "forward_with_storage", stencil, dtype, device
            )
            _ = backend_utils.get_backend_function(
                "maxwell_3d", "backward", stencil, dtype, device
            )
        except (RuntimeError, AttributeError, TypeError) as e:
            return _fallback_reason(f"3D C/CUDA backward symbols are unavailable ({e})")

        meta = {
            "dt": dt,
            "nt": nt_steps,
            "n_shots": n_shots,
            "nz": padded_nz,
            "ny": padded_ny,
            "nx": padded_nx,
            "n_sources": n_sources,
            "n_receivers": n_receivers,
            "step_ratio": gradient_sampling_interval,
            "accuracy": stencil,
            "pml_z0": pml_z0,
            "pml_y0": pml_y0,
            "pml_x0": pml_x0,
            "pml_z1": pml_z1,
            "pml_y1": pml_y1,
            "pml_x1": pml_x1,
            "source_component_idx": source_component_idx,
            "receiver_component_idx": receiver_component_idx,
            "fd_pad": tuple(fd_pad_list),
            "pml_width": tuple(pml_width_list),
            "models": callback_models,
            "forward_callback": forward_callback,
            "backward_callback": backward_callback,
            "callback_frequency": callback_frequency,
            "n_threads": n_threads_val,
            "grid_spacing": (dz, dy, dx),
            "rdz": 1.0 / dz,
            "rdy": 1.0 / dy,
            "rdx": 1.0 / dx,
            "storage_mode_str": effective_storage_mode_str,
            "storage_path": storage_path,
            "storage_compression": storage_compression,
            "execution_backend_id": execution_backend_id,
            "ca_batched": model_batched,
            "cb_batched": model_batched,
            "cq_batched": model_batched,
        }

        outputs = Maxwell3DForwardFunc.apply(
            ca,
            cb,
            cq,
            f,
            (
                az_flat,
                bz_flat,
                az_h_flat,
                bz_h_flat,
                ay_flat,
                by_flat,
                ay_h_flat,
                by_h_flat,
                ax_flat,
                bx_flat,
                ax_h_flat,
                bx_h_flat,
                kz_flat,
                kz_h_flat,
                ky_flat,
                ky_h_flat,
                kx_flat,
                kx_h_flat,
            ),
            (sources_i, receivers_i),
            (
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                m_hz_y,
                m_hy_z,
                m_hx_z,
                m_hz_x,
                m_hy_x,
                m_hx_y,
                m_ey_z,
                m_ez_y,
                m_ez_x,
                m_ex_z,
                m_ex_y,
                m_ey_x,
            ),
            meta,
        )
        (
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            m_hz_y,
            m_hy_z,
            m_hx_z,
            m_hz_x,
            m_hy_x,
            m_hx_y,
            m_ey_z,
            m_ez_y,
            m_ez_x,
            m_ex_z,
            m_ex_y,
            m_ey_x,
            receiver_amplitudes,
        ) = outputs
    else:
        try:
            forward_func = backend_utils.get_backend_function(
                "maxwell_3d", "forward", stencil, dtype, device
            )
        except (RuntimeError, AttributeError, TypeError) as e:
            return _fallback_reason(
                f"3D C/CUDA forward symbol is unavailable ({e})"
            )

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        callback_window = nt_steps if forward_callback is None else callback_frequency
        if callback_window <= 0:
            callback_window = nt_steps if nt_steps > 0 else 1
        compute_stream_handle, compute_stream_keepalive = _make_compute_stream(device)
        del compute_stream_keepalive

        debye_a = torch.empty(0, device=device, dtype=dtype)
        debye_b = torch.empty(0, device=device, dtype=dtype)
        debye_cp = torch.empty(0, device=device, dtype=dtype)
        pol_ex = torch.empty(0, device=device, dtype=dtype)
        pol_ey = torch.empty(0, device=device, dtype=dtype)
        pol_ez = torch.empty(0, device=device, dtype=dtype)
        ex_prev = torch.empty(0, device=device, dtype=dtype)
        ey_prev = torch.empty(0, device=device, dtype=dtype)
        ez_prev = torch.empty(0, device=device, dtype=dtype)
        n_poles = 0
        if has_dispersion and debye is not None:
            n_poles = int(debye["n_poles"])
            debye_a = debye["a"].contiguous()
            debye_b = debye["b"].contiguous()
            debye_cp = debye["cp"].contiguous()
            pol_ex = _init_polarization_state(
                n_shots=n_shots,
                n_poles=n_poles,
                spatial_shape=(padded_nz, padded_ny, padded_nx),
                device=device,
                dtype=dtype,
            ).contiguous()
            pol_ey = torch.zeros_like(pol_ex)
            pol_ez = torch.zeros_like(pol_ex)
            ex_prev = torch.empty_like(Ex)
            ey_prev = torch.empty_like(Ey)
            ez_prev = torch.empty_like(Ez)

        def _launch_forward(
            source_buffer: torch.Tensor,
            receiver_buffer: torch.Tensor,
            *,
            step_nt_local: int,
            start_step: int,
            stream_handle: int,
        ) -> None:
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(source_buffer),
                backend_utils.tensor_to_ptr(Ex),
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Ez),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hy),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_hz_y),
                backend_utils.tensor_to_ptr(m_hy_z),
                backend_utils.tensor_to_ptr(m_hx_z),
                backend_utils.tensor_to_ptr(m_hz_x),
                backend_utils.tensor_to_ptr(m_hy_x),
                backend_utils.tensor_to_ptr(m_hx_y),
                backend_utils.tensor_to_ptr(m_ey_z),
                backend_utils.tensor_to_ptr(m_ez_y),
                backend_utils.tensor_to_ptr(m_ez_x),
                backend_utils.tensor_to_ptr(m_ex_z),
                backend_utils.tensor_to_ptr(m_ex_y),
                backend_utils.tensor_to_ptr(m_ey_x),
                backend_utils.tensor_to_ptr(debye_a),
                backend_utils.tensor_to_ptr(debye_b),
                backend_utils.tensor_to_ptr(debye_cp),
                backend_utils.tensor_to_ptr(pol_ex),
                backend_utils.tensor_to_ptr(pol_ey),
                backend_utils.tensor_to_ptr(pol_ez),
                backend_utils.tensor_to_ptr(ex_prev),
                backend_utils.tensor_to_ptr(ey_prev),
                backend_utils.tensor_to_ptr(ez_prev),
                backend_utils.tensor_to_ptr(receiver_buffer),
                n_poles,
                backend_utils.tensor_to_ptr(az_flat),
                backend_utils.tensor_to_ptr(bz_flat),
                backend_utils.tensor_to_ptr(az_h_flat),
                backend_utils.tensor_to_ptr(bz_h_flat),
                backend_utils.tensor_to_ptr(ay_flat),
                backend_utils.tensor_to_ptr(by_flat),
                backend_utils.tensor_to_ptr(ay_h_flat),
                backend_utils.tensor_to_ptr(by_h_flat),
                backend_utils.tensor_to_ptr(ax_flat),
                backend_utils.tensor_to_ptr(bx_flat),
                backend_utils.tensor_to_ptr(ax_h_flat),
                backend_utils.tensor_to_ptr(bx_h_flat),
                backend_utils.tensor_to_ptr(kz_flat),
                backend_utils.tensor_to_ptr(kz_h_flat),
                backend_utils.tensor_to_ptr(ky_flat),
                backend_utils.tensor_to_ptr(ky_h_flat),
                backend_utils.tensor_to_ptr(kx_flat),
                backend_utils.tensor_to_ptr(kx_h_flat),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                1.0 / dz,
                1.0 / dy,
                1.0 / dx,
                dt,
                step_nt_local,
                n_shots,
                padded_nz,
                padded_ny,
                padded_nx,
                n_sources,
                n_receivers,
                gradient_sampling_interval,
                has_dispersion,
                model_batched,
                model_batched,
                model_batched,
                start_step,
                pml_z0,
                pml_y0,
                pml_x0,
                pml_z1,
                pml_y1,
                pml_x1,
                source_component_idx,
                receiver_component_idx,
                n_threads_val,
                device_idx,
                execution_backend_id,
                stream_handle,
            )

        for window_start in range(0, nt_steps, callback_window):
            if forward_callback is not None:
                callback_wavefields = {
                    "Ex": Ex,
                    "Ey": Ey,
                    "Ez": Ez,
                    "Hx": Hx,
                    "Hy": Hy,
                    "Hz": Hz,
                    "m_hz_y": m_hz_y,
                    "m_hy_z": m_hy_z,
                    "m_hx_z": m_hx_z,
                    "m_hz_x": m_hz_x,
                    "m_hy_x": m_hy_x,
                    "m_hx_y": m_hx_y,
                    "m_ey_z": m_ey_z,
                    "m_ez_y": m_ez_y,
                    "m_ez_x": m_ez_x,
                    "m_ex_z": m_ex_z,
                    "m_ex_y": m_ex_y,
                    "m_ey_x": m_ey_x,
                }
                if has_dispersion:
                    callback_wavefields["polarization"] = torch.stack(
                        (pol_ex.sum(dim=1), pol_ey.sum(dim=1), pol_ez.sum(dim=1)),
                        dim=1,
                    )
                forward_callback(
                    CallbackState(
                        dt=dt,
                        step=window_start,
                        nt=nt_steps,
                        wavefields=callback_wavefields,
                        models=callback_models,
                        gradients=None,
                        fd_pad=fd_pad_list,
                        pml_width=pml_width_list,
                        is_backward=False,
                        grid_spacing=[dz, dy, dx],
                    )
                )

            window_end = min(nt_steps, window_start + callback_window)
            _launch_forward(
                f,
                receiver_amplitudes,
                step_nt_local=window_end - window_start,
                start_step=window_start,
                stream_handle=compute_stream_handle,
            )

    s = (
        slice(None),
        slice(
            fd_pad_list[0], padded_nz - fd_pad_list[1] if fd_pad_list[1] > 0 else None
        ),
        slice(
            fd_pad_list[2], padded_ny - fd_pad_list[3] if fd_pad_list[3] > 0 else None
        ),
        slice(
            fd_pad_list[4], padded_nx - fd_pad_list[5] if fd_pad_list[5] > 0 else None
        ),
    )

    outputs = (
        Ex[s],
        Ey[s],
        Ez[s],
        Hx[s],
        Hy[s],
        Hz[s],
        m_hz_y[s],
        m_hy_z[s],
        m_hx_z[s],
        m_hz_x[s],
        m_hy_x[s],
        m_hx_y[s],
        m_ey_z[s],
        m_ez_y[s],
        m_ez_x[s],
        m_ex_z[s],
        m_ex_y[s],
        m_ey_x[s],
        receiver_amplitudes,
    )
    return outputs

__all__ = ["maxwell3d_c_cuda"]
