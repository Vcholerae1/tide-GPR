import warnings
from collections.abc import Sequence
from typing import Any

import torch

from .. import staggered
from ..grid_utils import _normalize_grid_spacing_2d, _normalize_pml_width_2d
from ..padding import create_or_pad, zero_interior
from ..storage import STORAGE_DEVICE
from ..utils import (
    C0,
    EP0,
    compile_material_coefficients,
    linearize_material_coefficients,
)
from ..validation import validate_model_gradient_sampling_interval
from .tm2d_born_autograd import BornTMForwardFunc
from .tm2d_born_python import borntm_python
from .tm2d_helpers import (
    _init_tm_wavefield,
    _prepare_tm2d_source_injection,
    _resolve_tm2d_storage_spec,
)


def borntm_c_cuda(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    depsilon: torch.Tensor | None,
    dsigma: torch.Tensor | None,
    dca: torch.Tensor | None,
    dcb: torch.Tensor | None,
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
    dEy_0: torch.Tensor | None,
    dHx_0: torch.Tensor | None,
    dHz_0: torch.Tensor | None,
    dm_Ey_x_0: torch.Tensor | None,
    dm_Ey_z_0: torch.Tensor | None,
    dm_Hx_z_0: torch.Tensor | None,
    dm_Hz_x_0: torch.Tensor | None,
    nt: int | None,
    parameterization: str,
    model_gradient_sampling_interval: int,
    linearize_source: bool,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    n_threads: int | None = None,
    return_background_receiver_amplitudes: bool = False,
    background_cache: dict[str, Any] | None = None,
    capture_background_cache: bool = False,
    capture_background_adjoint: bool = False,
):
    from .. import backend_utils

    if epsilon.ndim not in {2, 3}:
        raise NotImplementedError(
            "Native borntm requires a 2D model or one 2D model per shot."
        )
    if sigma.shape != epsilon.shape or mu.shape != epsilon.shape:
        raise RuntimeError("sigma and mu must have the same shape as epsilon")
    if parameterization not in {"epsilon_sigma", "ca_cb"}:
        raise ValueError(
            "parameterization must be 'epsilon_sigma' or 'ca_cb', "
            f"got {parameterization!r}."
        )
    device = epsilon.device

    source_requires_grad = bool(
        source_amplitude is not None and source_amplitude.requires_grad
    )
    state_requires_grad = any(
        tensor is not None and tensor.requires_grad
        for tensor in (
            Ey_0,
            Hx_0,
            Hz_0,
            m_Ey_x_0,
            m_Ey_z_0,
            m_Hx_z_0,
            m_Hz_x_0,
            dEy_0,
            dHx_0,
            dHz_0,
            dm_Ey_x_0,
            dm_Ey_z_0,
            dm_Hx_z_0,
            dm_Hz_x_0,
        )
    )

    grid_spacing = _normalize_grid_spacing_2d(grid_spacing)
    dy, dx = grid_spacing
    pml_width_list = _normalize_pml_width_2d(pml_width)

    if mu.requires_grad:
        warnings.warn(
            "Native borntm does not yet support gradients with respect to mu. "
            "Falling back to the Python reference path.",
            RuntimeWarning,
        )
        return borntm_python(
            epsilon,
            sigma,
            mu,
            depsilon,
            dsigma,
            dca,
            dcb,
            grid_spacing,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            None,
            stencil=stencil,
            pml_width=pml_width,
            max_vel=max_vel,
            Ey_0=Ey_0,
            Hx_0=Hx_0,
            Hz_0=Hz_0,
            m_Ey_x_0=m_Ey_x_0,
            m_Ey_z_0=m_Ey_z_0,
            m_Hx_z_0=m_Hx_z_0,
            m_Hz_x_0=m_Hz_x_0,
            dEy_0=dEy_0,
            dHx_0=dHx_0,
            dHz_0=dHz_0,
            dm_Ey_x_0=dm_Ey_x_0,
            dm_Ey_z_0=dm_Ey_z_0,
            dm_Hx_z_0=dm_Hx_z_0,
            dm_Hz_x_0=dm_Hz_x_0,
            nt=nt,
            parameterization=parameterization,
            linearize_source=linearize_source,
        )

    if source_requires_grad or state_requires_grad:
        warnings.warn(
            "Native borntm does not support gradients with respect to source "
            "amplitudes or initial wavefields. Falling back to the Python "
            "reference path.",
            RuntimeWarning,
        )
        return borntm_python(
            epsilon,
            sigma,
            mu,
            depsilon,
            dsigma,
            dca,
            dcb,
            grid_spacing,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            None,
            stencil=stencil,
            pml_width=pml_width,
            max_vel=max_vel,
            Ey_0=Ey_0,
            Hx_0=Hx_0,
            Hz_0=Hz_0,
            m_Ey_x_0=m_Ey_x_0,
            m_Ey_z_0=m_Ey_z_0,
            m_Hx_z_0=m_Hx_z_0,
            m_Hz_x_0=m_Hz_x_0,
            dEy_0=dEy_0,
            dHx_0=dHx_0,
            dHz_0=dHz_0,
            dm_Ey_x_0=dm_Ey_x_0,
            dm_Ey_z_0=dm_Ey_z_0,
            dm_Hx_z_0=dm_Hx_z_0,
            dm_Hz_x_0=dm_Hz_x_0,
            nt=nt,
            parameterization=parameterization,
            linearize_source=linearize_source,
        )

    if torch._C._are_functorch_transforms_active():
        raise NotImplementedError(
            "torch.func transforms are not supported for the native born backend."
        )

    dtype = epsilon.dtype
    backend_device = device
    if device.type not in {"cpu", "cuda"}:
        raise NotImplementedError("C/CUDA backend supports only cpu and cuda devices.")

    n_threads_val = 0
    if n_threads is not None:
        n_threads_val = int(n_threads)
        if n_threads_val < 0:
            raise ValueError("n_threads must be >= 0 when provided.")

    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = int(source_amplitude.shape[-1])
    nt_steps = int(nt)
    gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    if gradient_sampling_interval < 1:
        gradient_sampling_interval = 1
    if nt_steps > 0:
        gradient_sampling_interval = min(gradient_sampling_interval, nt_steps)

    if source_amplitude is not None and source_amplitude.numel() > 0:
        n_shots = int(source_amplitude.shape[0])
    elif source_location is not None and source_location.numel() > 0:
        n_shots = int(source_location.shape[0])
    elif receiver_location is not None and receiver_location.numel() > 0:
        n_shots = int(receiver_location.shape[0])
    else:
        n_shots = 1

    if max_vel is None:
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    pml_freq = 0.5 / dt

    fd_pad = stencil // 2
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]

    model_ny, model_nx = epsilon.shape[-2:]
    padded_ny = model_ny + total_pad[0] + total_pad[1]
    padded_nx = model_nx + total_pad[2] + total_pad[3]
    padded_size = (padded_ny, padded_nx)

    epsilon_padded = create_or_pad(
        epsilon, total_pad, device, dtype, padded_size, mode="replicate"
    )
    sigma_padded = create_or_pad(
        sigma, total_pad, device, dtype, padded_size, mode="replicate"
    )
    mu_padded = create_or_pad(
        mu, total_pad, device, dtype, padded_size, mode="replicate"
    )

    material = compile_material_coefficients(
        epsilon_padded,
        sigma_padded,
        mu_padded,
        dt,
    )
    model_batched = epsilon.ndim == 3
    if model_batched and int(epsilon.shape[0]) != n_shots:
        raise ValueError("Native batched borntm currently requires one model per shot.")
    ca = material["ca"].contiguous()
    cb = material["cb"].contiguous()
    cq = material["cq"].contiguous()
    if not model_batched:
        ca = ca[None, :, :]
        cb = cb[None, :, :]
        cq = cq[None, :, :]

    if parameterization == "epsilon_sigma":
        depsilon_padded = create_or_pad(
            torch.empty(0, device=device, dtype=dtype)
            if depsilon is None
            else depsilon,
            total_pad,
            device,
            dtype,
            padded_size,
            mode="constant",
        )
        dsigma_padded = create_or_pad(
            torch.empty(0, device=device, dtype=dtype) if dsigma is None else dsigma,
            total_pad,
            device,
            dtype,
            padded_size,
            mode="constant",
        )
        dca_padded, dcb_padded = linearize_material_coefficients(
            epsilon_padded,
            sigma_padded,
            material["ca"],
            material["cb"],
            dt,
            depsilon_r=depsilon_padded,
            dsigma=dsigma_padded,
        )
    else:
        dca_padded = create_or_pad(
            torch.empty(0, device=device, dtype=dtype) if dca is None else dca,
            total_pad,
            device,
            dtype,
            padded_size,
            mode="constant",
        )
        dcb_padded = create_or_pad(
            torch.empty(0, device=device, dtype=dtype) if dcb is None else dcb,
            total_pad,
            device,
            dtype,
            padded_size,
            mode="constant",
        )

    dca_native = dca_padded.contiguous()
    dcb_native = dcb_padded.contiguous()
    if not model_batched:
        dca_native = dca_native[None, :, :]
        dcb_native = dcb_native[None, :, :]

    flat_model_shape = padded_ny * padded_nx
    if source_location is not None and source_location.numel() > 0:
        source_y = source_location[..., 0] + total_pad[0]
        source_x = source_location[..., 1] + total_pad[2]
        sources_i = (source_y * padded_nx + source_x).long().contiguous()
        n_sources = int(source_location.shape[1])
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        receiver_y = receiver_location[..., 0] + total_pad[0]
        receiver_x = receiver_location[..., 1] + total_pad[2]
        receivers_i = (receiver_y * padded_nx + receiver_x).long().contiguous()
        n_receivers = int(receiver_location.shape[1])
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    source_coeff = -1.0 / (dx * dy)
    cb_at_src: torch.Tensor | None = None
    dcb_at_src: torch.Tensor | None = None
    if n_sources > 0:
        cb_flat = cb.reshape(-1, flat_model_shape)
        if not model_batched:
            cb_flat = cb_flat.expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, sources_i)
        if linearize_source:
            dcb_flat = dcb_native.reshape(-1, flat_model_shape)
            if not model_batched:
                dcb_flat = dcb_flat.expand(n_shots, -1)
            dcb_at_src = dcb_flat.gather(1, sources_i)

    f0 = _prepare_tm2d_source_injection(
        source_amplitude=source_amplitude,
        cb_at_src=cb_at_src,
        source_coeff=source_coeff,
        dtype=dtype,
        n_shots=n_shots,
        n_sources=n_sources,
        nt_steps=nt_steps,
    )[0].contiguous()
    if linearize_source:
        df = _prepare_tm2d_source_injection(
            source_amplitude=source_amplitude,
            cb_at_src=dcb_at_src,
            source_coeff=source_coeff,
            dtype=dtype,
            n_shots=n_shots,
            n_sources=n_sources,
            nt_steps=nt_steps,
        )[0].contiguous()
    elif n_sources > 0:
        df = torch.zeros(
            nt_steps * n_shots * n_sources,
            device=device,
            dtype=dtype,
        )
    else:
        df = torch.empty(0, device=device, dtype=dtype)

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
    dEy = _init_tm_wavefield(
        dEy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dHx = _init_tm_wavefield(
        dHx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dHz = _init_tm_wavefield(
        dHz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_Ey_x = _init_tm_wavefield(
        dm_Ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_Ey_z = _init_tm_wavefield(
        dm_Ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_Hx_z = _init_tm_wavefield(
        dm_Hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_Hz_x = _init_tm_wavefield(
        dm_Hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )

    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], [1, 0, 0, 1]):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)
    for wf, dim in zip([dm_Ey_x, dm_Ey_z, dm_Hx_z, dm_Hz_x], [1, 0, 0, 1]):
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

    pml_y0 = fd_pad_list[0] + pml_width_list[0]
    pml_y1 = padded_ny - fd_pad_list[1] - pml_width_list[1]
    pml_x0 = fd_pad_list[2] + pml_width_list[2]
    pml_x1 = padded_nx - fd_pad_list[3] - pml_width_list[3]

    needs_storage = dca_native.requires_grad or dcb_native.requires_grad
    needs_autograd = needs_storage or df.requires_grad
    if device.type == "cpu" and needs_autograd and gradient_sampling_interval != 1:
        raise NotImplementedError(
            "Native TM2D Born model/background gradients on CPU currently require "
            "model_gradient_sampling_interval in {0, 1}."
        )
    _, _, storage_bytes_per_elem, storage_format = _resolve_tm2d_storage_spec(
        storage_compression=storage_compression,
        dtype=dtype,
        device=device,
        context="storage_compression",
    )

    storage_mode_str = storage_mode.lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )
    if device.type == "cpu" and storage_mode_str == "cpu":
        storage_mode_str = "device"
    if needs_storage:
        if storage_mode_str == "none":
            raise ValueError(
                "storage_mode='none' is not compatible with gradient computation "
                "for native born forward."
            )
        if storage_mode_str == "auto":
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
                storage_mode_str = "device"
            elif total_bytes <= limit_host:
                storage_mode_str = "cpu"
            else:
                storage_mode_str = "disk"
    elif storage_mode_str == "auto":
        storage_mode_str = "none"

    capture_background_receivers = return_background_receiver_amplitudes
    empty_cached = torch.empty(0, device=device, dtype=dtype)
    cached_ey_store = (
        background_cache["ey_store"] if background_cache is not None else empty_cached
    )
    cached_curl_store = (
        background_cache["curl_store"] if background_cache is not None else empty_cached
    )
    cached_background_receiver = (
        background_cache["predicted_data"]
        if background_cache is not None
        else empty_cached
    )
    cached_lambda_store = (
        background_cache.get("lambda_store", empty_cached)
        if background_cache is not None
        else empty_cached
    )
    background_n_shots = (
        int(background_cache.get("n_shots", n_shots))
        if background_cache is not None
        else n_shots
    )

    if needs_autograd:
        result = BornTMForwardFunc.apply(
            dca_native,
            dcb_native,
            ca,
            cb,
            cq,
            f0,
            df,
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
            storage_mode_str,
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
            dEy,
            dHx,
            dHz,
            dm_Ey_x,
            dm_Ey_z,
            dm_Hx_z,
            dm_Hz_x,
            n_threads_val,
            backend_device,
            cached_ey_store,
            cached_curl_store,
            cached_background_receiver,
            background_cache is not None,
            cached_lambda_store,
            capture_background_adjoint,
            background_n_shots,
        )
        (
            dEy_out,
            dHx_out,
            dHz_out,
            dm_Ey_x_out,
            dm_Ey_z_out,
            dm_Hx_z_out,
            dm_Hz_x_out,
            receiver_amplitudes,
        ) = result[:8]
        background_receiver_amplitudes = result[8]
        captured_cache: dict[str, Any] | None = None
        if capture_background_cache:
            from .common import _get_ctx_handle

            ctx_handle_id = int(result[-1].item())
            ctx_data = _get_ctx_handle(ctx_handle_id)
            if ctx_data["storage_mode"] != STORAGE_DEVICE:
                raise RuntimeError(
                    "Reusable TM2D background cache requires device storage."
                )
            captured_cache = {
                "ey_store": ctx_data["backward_storage_tensors"][0],
                "curl_store": ctx_data["backward_storage_tensors"][2],
                "lambda_store": ctx_data["lambda_store"],
                "predicted_data": background_receiver_amplitudes.detach(),
                "storage_format": ctx_data["storage_format"],
                "shot_bytes_uncomp": ctx_data["shot_bytes_uncomp"],
                "n_shots": n_shots,
            }
    else:
        forward_func = backend_utils.get_backend_function(
            "maxwell_tm",
            "born_forward",
            stencil,
            dtype,
            backend_device,
        )
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt_steps, n_shots, n_receivers, device=device, dtype=dtype
            )
            background_receiver_amplitudes = (
                torch.zeros_like(receiver_amplitudes)
                if capture_background_receivers
                else torch.empty(0, device=device, dtype=dtype)
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
            background_receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        compute_stream_handle = 0
        if device.type == "cuda":
            compute_stream_handle = int(
                getattr(torch.cuda.current_stream(device=device), "cuda_stream", 0) or 0
            )

        forward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(dca_native),
            backend_utils.tensor_to_ptr(dcb_native),
            backend_utils.tensor_to_ptr(f0),
            backend_utils.tensor_to_ptr(df),
            backend_utils.tensor_to_ptr(Ey),
            backend_utils.tensor_to_ptr(Hx),
            backend_utils.tensor_to_ptr(Hz),
            backend_utils.tensor_to_ptr(m_Ey_x),
            backend_utils.tensor_to_ptr(m_Ey_z),
            backend_utils.tensor_to_ptr(m_Hx_z),
            backend_utils.tensor_to_ptr(m_Hz_x),
            backend_utils.tensor_to_ptr(dEy),
            backend_utils.tensor_to_ptr(dHx),
            backend_utils.tensor_to_ptr(dHz),
            backend_utils.tensor_to_ptr(dm_Ey_x),
            backend_utils.tensor_to_ptr(dm_Ey_z),
            backend_utils.tensor_to_ptr(dm_Hx_z),
            backend_utils.tensor_to_ptr(dm_Hz_x),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
            backend_utils.tensor_to_ptr(background_receiver_amplitudes),
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
            nt_steps,
            n_shots,
            padded_ny,
            padded_nx,
            n_sources,
            n_receivers,
            gradient_sampling_interval,
            model_batched,
            model_batched,
            model_batched,
            0,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            n_threads_val,
            device_idx,
            compute_stream_handle,
        )
        dEy_out, dHx_out, dHz_out = dEy, dHx, dHz
        dm_Ey_x_out, dm_Ey_z_out, dm_Hx_z_out, dm_Hz_x_out = (
            dm_Ey_x,
            dm_Ey_z,
            dm_Hx_z,
            dm_Hz_x,
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
    outputs = (
        Ey[s],
        Hx[s],
        Hz[s],
        m_Ey_x[s],
        m_Ey_z[s],
        m_Hx_z[s],
        m_Hz_x[s],
        dEy_out[s],
        dHx_out[s],
        dHz_out[s],
        dm_Ey_x_out[s],
        dm_Ey_z_out[s],
        dm_Hx_z_out[s],
        dm_Hz_x_out[s],
        receiver_amplitudes,
    ) + (
        (background_receiver_amplitudes,)
        if return_background_receiver_amplitudes
        else ()
    )
    if capture_background_cache:
        return outputs + (captured_cache,)
    return outputs


__all__ = ["borntm_c_cuda"]
