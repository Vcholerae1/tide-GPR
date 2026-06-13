import warnings
from collections.abc import Sequence

import torch

from ..storage import _normalize_storage_compression, _resolve_storage_compression
from .. import staggered
from ..grid_utils import _normalize_grid_spacing_3d, _normalize_pml_width_3d
from ..padding import create_or_pad, zero_interior
from ..utils import (
    C0,
    compile_material_coefficients,
    linearize_material_coefficients,
)
from ..validation import validate_model_gradient_sampling_interval
from .maxwell3d_born_autograd import Born3DForwardFunc
from .maxwell3d_born_python import born3d_python
from .validation_internal import _COMPONENT_TO_INDEX_3D


_HYBRID_TENSOR_ARG_NAMES = (
    "epsilon",
    "sigma",
    "mu",
    "depsilon",
    "dsigma",
    "dca",
    "dcb",
    "source_amplitude",
    "source_location",
    "receiver_location",
    "Ex_0",
    "Ey_0",
    "Ez_0",
    "Hx_0",
    "Hy_0",
    "Hz_0",
    "m_hz_y_0",
    "m_hy_z_0",
    "m_hx_z_0",
    "m_hz_x_0",
    "m_hy_x_0",
    "m_hx_y_0",
    "m_ey_z_0",
    "m_ez_y_0",
    "m_ez_x_0",
    "m_ex_z_0",
    "m_ex_y_0",
    "m_ey_x_0",
    "dEx_0",
    "dEy_0",
    "dEz_0",
    "dHx_0",
    "dHy_0",
    "dHz_0",
    "dm_hz_y_0",
    "dm_hy_z_0",
    "dm_hx_z_0",
    "dm_hz_x_0",
    "dm_hy_x_0",
    "dm_hx_y_0",
    "dm_ey_z_0",
    "dm_ez_y_0",
    "dm_ez_x_0",
    "dm_ex_z_0",
    "dm_ex_y_0",
    "dm_ey_x_0",
)


class _Born3DNativeAutogradFallbackFunc(torch.autograd.Function):
    """Native 3D Born forward with Python reference gradients for unsupported grads."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        *args,
    ) -> tuple[torch.Tensor, ...]:
        meta = args[-1]
        values = dict(zip(_HYBRID_TENSOR_ARG_NAMES, args[:-1], strict=True))
        ctx.meta = meta
        ctx.values = values

        def detached(name: str) -> torch.Tensor | None:
            value = values[name]
            return value.detach() if isinstance(value, torch.Tensor) else None

        outputs = born3d_c_cuda(
            detached("epsilon"),
            detached("sigma"),
            detached("mu"),
            detached("depsilon"),
            detached("dsigma"),
            detached("dca"),
            detached("dcb"),
            meta["grid_spacing"],
            meta["dt"],
            detached("source_amplitude"),
            detached("source_location"),
            detached("receiver_location"),
            meta["stencil"],
            meta["pml_width"],
            meta["max_vel"],
            detached("Ex_0"),
            detached("Ey_0"),
            detached("Ez_0"),
            detached("Hx_0"),
            detached("Hy_0"),
            detached("Hz_0"),
            detached("m_hz_y_0"),
            detached("m_hy_z_0"),
            detached("m_hx_z_0"),
            detached("m_hz_x_0"),
            detached("m_hy_x_0"),
            detached("m_hx_y_0"),
            detached("m_ey_z_0"),
            detached("m_ez_y_0"),
            detached("m_ez_x_0"),
            detached("m_ex_z_0"),
            detached("m_ex_y_0"),
            detached("m_ey_x_0"),
            detached("dEx_0"),
            detached("dEy_0"),
            detached("dEz_0"),
            detached("dHx_0"),
            detached("dHy_0"),
            detached("dHz_0"),
            detached("dm_hz_y_0"),
            detached("dm_hy_z_0"),
            detached("dm_hx_z_0"),
            detached("dm_hz_x_0"),
            detached("dm_hy_x_0"),
            detached("dm_hx_y_0"),
            detached("dm_ey_z_0"),
            detached("dm_ez_y_0"),
            detached("dm_ez_x_0"),
            detached("dm_ex_z_0"),
            detached("dm_ex_y_0"),
            detached("dm_ey_x_0"),
            meta["nt"],
            meta["parameterization"],
            meta["model_gradient_sampling_interval"],
            meta["linearize_source"],
            meta["source_component"],
            meta["receiver_component"],
            storage_mode=meta["storage_mode"],
            storage_path=meta["storage_path"],
            storage_compression=meta["storage_compression"],
            storage_bytes_limit_device=meta["storage_bytes_limit_device"],
            storage_bytes_limit_host=meta["storage_bytes_limit_host"],
            n_threads=meta["n_threads"],
            _force_no_gradient_fallback=True,
        )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        meta = ctx.meta
        values = ctx.values

        differentiable_names: list[str] = []
        differentiable_values: list[torch.Tensor] = []
        ref_values: dict[str, torch.Tensor | None] = {}

        for idx, name in enumerate(_HYBRID_TENSOR_ARG_NAMES):
            value = values[name]
            if not isinstance(value, torch.Tensor):
                ref_values[name] = None
                continue
            if ctx.needs_input_grad[idx] and torch.is_floating_point(value):
                ref = value.detach().requires_grad_(True)
                differentiable_names.append(name)
                differentiable_values.append(ref)
            else:
                ref = value.detach()
            ref_values[name] = ref

        if not differentiable_values:
            return (None,) * (len(_HYBRID_TENSOR_ARG_NAMES) + 1)

        with torch.enable_grad():
            ref_outputs = born3d_python(
                ref_values["epsilon"],
                ref_values["sigma"],
                ref_values["mu"],
                ref_values["depsilon"],
                ref_values["dsigma"],
                ref_values["dca"],
                ref_values["dcb"],
                meta["grid_spacing"],
                meta["dt"],
                ref_values["source_amplitude"],
                ref_values["source_location"],
                ref_values["receiver_location"],
                None,
                stencil=meta["stencil"],
                pml_width=meta["pml_width"],
                max_vel=meta["max_vel"],
                Ex_0=ref_values["Ex_0"],
                Ey_0=ref_values["Ey_0"],
                Ez_0=ref_values["Ez_0"],
                Hx_0=ref_values["Hx_0"],
                Hy_0=ref_values["Hy_0"],
                Hz_0=ref_values["Hz_0"],
                m_hz_y_0=ref_values["m_hz_y_0"],
                m_hy_z_0=ref_values["m_hy_z_0"],
                m_hx_z_0=ref_values["m_hx_z_0"],
                m_hz_x_0=ref_values["m_hz_x_0"],
                m_hy_x_0=ref_values["m_hy_x_0"],
                m_hx_y_0=ref_values["m_hx_y_0"],
                m_ey_z_0=ref_values["m_ey_z_0"],
                m_ez_y_0=ref_values["m_ez_y_0"],
                m_ez_x_0=ref_values["m_ez_x_0"],
                m_ex_z_0=ref_values["m_ex_z_0"],
                m_ex_y_0=ref_values["m_ex_y_0"],
                m_ey_x_0=ref_values["m_ey_x_0"],
                dEx_0=ref_values["dEx_0"],
                dEy_0=ref_values["dEy_0"],
                dEz_0=ref_values["dEz_0"],
                dHx_0=ref_values["dHx_0"],
                dHy_0=ref_values["dHy_0"],
                dHz_0=ref_values["dHz_0"],
                dm_hz_y_0=ref_values["dm_hz_y_0"],
                dm_hy_z_0=ref_values["dm_hy_z_0"],
                dm_hx_z_0=ref_values["dm_hx_z_0"],
                dm_hz_x_0=ref_values["dm_hz_x_0"],
                dm_hy_x_0=ref_values["dm_hy_x_0"],
                dm_hx_y_0=ref_values["dm_hx_y_0"],
                dm_ey_z_0=ref_values["dm_ey_z_0"],
                dm_ez_y_0=ref_values["dm_ez_y_0"],
                dm_ez_x_0=ref_values["dm_ez_x_0"],
                dm_ex_z_0=ref_values["dm_ex_z_0"],
                dm_ex_y_0=ref_values["dm_ex_y_0"],
                dm_ey_x_0=ref_values["dm_ey_x_0"],
                nt=meta["nt"],
                parameterization=meta["parameterization"],
                linearize_source=meta["linearize_source"],
                source_component=meta["source_component"],
                receiver_component=meta["receiver_component"],
            )
            ref_outputs = (*ref_outputs[:-2], ref_outputs[-1])
            active_outputs: list[torch.Tensor] = []
            active_grad_outputs: list[torch.Tensor] = []
            for out, grad in zip(ref_outputs, grad_outputs, strict=True):
                if grad is None or not out.requires_grad:
                    continue
                active_outputs.append(out)
                active_grad_outputs.append(grad)
            if not active_outputs:
                return (None,) * (len(_HYBRID_TENSOR_ARG_NAMES) + 1)
            computed_grads = torch.autograd.grad(
                active_outputs,
                differentiable_values,
                grad_outputs=active_grad_outputs,
                allow_unused=True,
            )

        grads_by_name = dict(zip(differentiable_names, computed_grads, strict=True))
        grads: list[torch.Tensor | None] = []
        for name in _HYBRID_TENSOR_ARG_NAMES:
            grads.append(grads_by_name.get(name))
        grads.append(None)
        return tuple(grads)


def _to_native_output_layout(
    result: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Drop the Python-only background receiver output from Born fallback results."""
    return (*result[:-2], result[-1])


def _init_wavefield_3d(
    field_0: torch.Tensor | None,
    *,
    n_shots: int,
    size_with_batch: tuple[int, int, int, int],
    fd_pad_list: list[int],
    device: torch.device,
    dtype: torch.dtype,
    contiguous: bool = False,
) -> torch.Tensor:
    if field_0 is not None:
        if field_0.ndim == 3:
            field_0 = field_0[None, :, :, :].expand(n_shots, -1, -1, -1)
        field = create_or_pad(
            field_0,
            fd_pad_list,
            device,
            dtype,
            size_with_batch,
            mode="constant",
        )
    else:
        field = torch.zeros(size_with_batch, device=device, dtype=dtype)
    return field.contiguous() if contiguous else field


def _prepare_source_term(
    source_amplitude: torch.Tensor | None,
    coeff_at_src: torch.Tensor | None,
    *,
    source_coeff: float,
    dtype: torch.dtype,
    device: torch.device,
    nt_steps: int,
    n_shots: int,
    n_sources: int,
) -> torch.Tensor:
    if (
        n_sources == 0
        or source_amplitude is None
        or source_amplitude.numel() == 0
        or coeff_at_src is None
    ):
        return torch.empty(0, device=device, dtype=dtype)
    source_flat = source_amplitude.permute(2, 0, 1).contiguous()
    source_flat = (source_flat * coeff_at_src[None, :, :] * source_coeff).reshape(
        nt_steps * n_shots * n_sources
    )
    return source_flat.contiguous()


def born3d_c_cuda(
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
    dEx_0: torch.Tensor | None,
    dEy_0: torch.Tensor | None,
    dEz_0: torch.Tensor | None,
    dHx_0: torch.Tensor | None,
    dHy_0: torch.Tensor | None,
    dHz_0: torch.Tensor | None,
    dm_hz_y_0: torch.Tensor | None,
    dm_hy_z_0: torch.Tensor | None,
    dm_hx_z_0: torch.Tensor | None,
    dm_hz_x_0: torch.Tensor | None,
    dm_hy_x_0: torch.Tensor | None,
    dm_hx_y_0: torch.Tensor | None,
    dm_ey_z_0: torch.Tensor | None,
    dm_ez_y_0: torch.Tensor | None,
    dm_ez_x_0: torch.Tensor | None,
    dm_ex_z_0: torch.Tensor | None,
    dm_ex_y_0: torch.Tensor | None,
    dm_ey_x_0: torch.Tensor | None,
    nt: int | None,
    parameterization: str,
    model_gradient_sampling_interval: int,
    linearize_source: bool,
    source_component: str,
    receiver_component: str,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    n_threads: int | None = None,
    _force_no_gradient_fallback: bool = False,
):
    from .. import backend_utils

    if epsilon.ndim != 3:
        raise NotImplementedError(
            "Native born3d currently supports a single 3D model only."
        )
    if sigma.shape != epsilon.shape or mu.shape != epsilon.shape:
        raise RuntimeError("sigma and mu must have the same shape as epsilon")
    if parameterization not in {"epsilon_sigma", "ca_cb"}:
        raise ValueError(
            "parameterization must be 'epsilon_sigma' or 'ca_cb', "
            f"got {parameterization!r}."
        )

    source_requires_grad = bool(
        source_amplitude is not None and source_amplitude.requires_grad
    )
    state_requires_grad = any(
        tensor is not None and tensor.requires_grad
        for tensor in (
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
            dEx_0,
            dEy_0,
            dEz_0,
            dHx_0,
            dHy_0,
            dHz_0,
            dm_hz_y_0,
            dm_hy_z_0,
            dm_hx_z_0,
            dm_hz_x_0,
            dm_hy_x_0,
            dm_hx_y_0,
            dm_ey_z_0,
            dm_ez_y_0,
            dm_ez_x_0,
            dm_ex_z_0,
            dm_ex_y_0,
            dm_ey_x_0,
        )
    )

    unsupported_gradient_fallback = (
        mu.requires_grad or source_requires_grad or state_requires_grad
    )
    if unsupported_gradient_fallback and not _force_no_gradient_fallback:
        meta = {
            "grid_spacing": grid_spacing,
            "dt": dt,
            "stencil": stencil,
            "pml_width": pml_width,
            "max_vel": max_vel,
            "nt": nt,
            "parameterization": parameterization,
            "model_gradient_sampling_interval": model_gradient_sampling_interval,
            "linearize_source": linearize_source,
            "source_component": source_component,
            "receiver_component": receiver_component,
            "storage_mode": storage_mode,
            "storage_path": storage_path,
            "storage_compression": storage_compression,
            "storage_bytes_limit_device": storage_bytes_limit_device,
            "storage_bytes_limit_host": storage_bytes_limit_host,
            "n_threads": n_threads,
        }
        return _Born3DNativeAutogradFallbackFunc.apply(
            epsilon,
            sigma,
            mu,
            depsilon,
            dsigma,
            dca,
            dcb,
            source_amplitude,
            source_location,
            receiver_location,
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
            dEx_0,
            dEy_0,
            dEz_0,
            dHx_0,
            dHy_0,
            dHz_0,
            dm_hz_y_0,
            dm_hy_z_0,
            dm_hx_z_0,
            dm_hz_x_0,
            dm_hy_x_0,
            dm_hx_y_0,
            dm_ey_z_0,
            dm_ez_y_0,
            dm_ez_x_0,
            dm_ex_z_0,
            dm_ex_y_0,
            dm_ey_x_0,
            meta,
        )

    if torch._C._are_functorch_transforms_active():
        raise NotImplementedError(
            "torch.func transforms are not supported for the native 3D born backend."
        )

    device = epsilon.device
    dtype = epsilon.dtype
    backend_device = device
    if device.type not in {"cpu", "cuda"}:
        raise NotImplementedError("C/CUDA backend supports only cpu and cuda devices.")

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

    n_threads_val = 0
    if n_threads is not None:
        n_threads_val = int(n_threads)
        if n_threads_val < 0:
            raise ValueError("n_threads must be >= 0 when provided.")

    grid_spacing_list = _normalize_grid_spacing_3d(grid_spacing)
    dz, dy, dx = grid_spacing_list
    pml_width_list = _normalize_pml_width_3d(pml_width)

    fd_pad = stencil // 2
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]

    model_nz, model_ny, model_nx = epsilon.shape
    padded_nz = model_nz + total_pad[0] + total_pad[1]
    padded_ny = model_ny + total_pad[2] + total_pad[3]
    padded_nx = model_nx + total_pad[4] + total_pad[5]
    padded_size = (padded_nz, padded_ny, padded_nx)

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
    ca = material["ca"][None, :, :, :].contiguous()
    cb = material["cb"][None, :, :, :].contiguous()
    cq = material["cq"][None, :, :, :].contiguous()

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

    dca_native = dca_padded[None, :, :, :].contiguous()
    dcb_native = dcb_padded[None, :, :, :].contiguous()

    flat_model_shape = padded_nz * padded_ny * padded_nx
    if source_location is not None and source_location.numel() > 0:
        source_z = source_location[..., 0] + total_pad[0]
        source_y = source_location[..., 1] + total_pad[2]
        source_x = source_location[..., 2] + total_pad[4]
        sources_i = ((source_z * padded_ny + source_y) * padded_nx + source_x).long()
        sources_i = sources_i.contiguous()
        n_sources = int(source_location.shape[1])
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
        n_receivers = int(receiver_location.shape[1])
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    source_coeff = -1.0 / (dx * dy * dz)
    cb_at_src: torch.Tensor | None = None
    dcb_at_src: torch.Tensor | None = None
    if n_sources > 0:
        cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, sources_i)
        if linearize_source:
            dcb_flat = dcb_native.reshape(1, flat_model_shape).expand(n_shots, -1)
            dcb_at_src = dcb_flat.gather(1, sources_i)

    f0 = _prepare_source_term(
        source_amplitude,
        cb_at_src,
        source_coeff=source_coeff,
        dtype=dtype,
        device=device,
        nt_steps=nt_steps,
        n_shots=n_shots,
        n_sources=n_sources,
    )
    if linearize_source:
        df = _prepare_source_term(
            source_amplitude,
            dcb_at_src,
            source_coeff=source_coeff,
            dtype=dtype,
            device=device,
            nt_steps=nt_steps,
            n_shots=n_shots,
            n_sources=n_sources,
        )
    elif n_sources > 0:
        df = torch.zeros(nt_steps * n_shots * n_sources, device=device, dtype=dtype)
    else:
        df = torch.empty(0, device=device, dtype=dtype)

    size_with_batch = (n_shots, padded_nz, padded_ny, padded_nx)
    Ex = _init_wavefield_3d(
        Ex_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    Ey = _init_wavefield_3d(
        Ey_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    Ez = _init_wavefield_3d(
        Ez_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    Hx = _init_wavefield_3d(
        Hx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    Hy = _init_wavefield_3d(
        Hy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    Hz = _init_wavefield_3d(
        Hz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )

    m_hz_y = _init_wavefield_3d(
        m_hz_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_hy_z = _init_wavefield_3d(
        m_hy_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_hx_z = _init_wavefield_3d(
        m_hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_hz_x = _init_wavefield_3d(
        m_hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_hy_x = _init_wavefield_3d(
        m_hy_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_hx_y = _init_wavefield_3d(
        m_hx_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_ey_z = _init_wavefield_3d(
        m_ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_ez_y = _init_wavefield_3d(
        m_ez_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_ez_x = _init_wavefield_3d(
        m_ez_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_ex_z = _init_wavefield_3d(
        m_ex_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_ex_y = _init_wavefield_3d(
        m_ex_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    m_ey_x = _init_wavefield_3d(
        m_ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )

    dEx = _init_wavefield_3d(
        dEx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dEy = _init_wavefield_3d(
        dEy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dEz = _init_wavefield_3d(
        dEz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dHx = _init_wavefield_3d(
        dHx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dHy = _init_wavefield_3d(
        dHy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dHz = _init_wavefield_3d(
        dHz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_hz_y = _init_wavefield_3d(
        dm_hz_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_hy_z = _init_wavefield_3d(
        dm_hy_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_hx_z = _init_wavefield_3d(
        dm_hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_hz_x = _init_wavefield_3d(
        dm_hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_hy_x = _init_wavefield_3d(
        dm_hy_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_hx_y = _init_wavefield_3d(
        dm_hx_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_ey_z = _init_wavefield_3d(
        dm_ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_ez_y = _init_wavefield_3d(
        dm_ez_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_ez_x = _init_wavefield_3d(
        dm_ez_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_ex_z = _init_wavefield_3d(
        dm_ex_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_ex_y = _init_wavefield_3d(
        dm_ex_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )
    dm_ey_x = _init_wavefield_3d(
        dm_ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
        contiguous=True,
    )

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
    for wf, dim in [
        (dm_hz_y, 1),
        (dm_hy_z, 0),
        (dm_hx_z, 0),
        (dm_hz_x, 2),
        (dm_hy_x, 2),
        (dm_hx_y, 1),
        (dm_ey_z, 0),
        (dm_ez_y, 1),
        (dm_ez_x, 2),
        (dm_ex_z, 0),
        (dm_ex_y, 1),
        (dm_ey_x, 2),
    ]:
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

    profiles = (
        az.reshape(-1).contiguous(),
        bz.reshape(-1).contiguous(),
        az_h.reshape(-1).contiguous(),
        bz_h.reshape(-1).contiguous(),
        ay.reshape(-1).contiguous(),
        by.reshape(-1).contiguous(),
        ay_h.reshape(-1).contiguous(),
        by_h.reshape(-1).contiguous(),
        ax.reshape(-1).contiguous(),
        bx.reshape(-1).contiguous(),
        ax_h.reshape(-1).contiguous(),
        bx_h.reshape(-1).contiguous(),
        kz.reshape(-1).contiguous(),
        kz_h.reshape(-1).contiguous(),
        ky.reshape(-1).contiguous(),
        ky_h.reshape(-1).contiguous(),
        kx.reshape(-1).contiguous(),
        kx_h.reshape(-1).contiguous(),
    )

    pml_z0 = fd_pad_list[0] + pml_width_list[0]
    pml_y0 = fd_pad_list[2] + pml_width_list[2]
    pml_x0 = fd_pad_list[4] + pml_width_list[4]
    pml_z1 = padded_nz - fd_pad_list[1] - pml_width_list[1]
    pml_y1 = padded_ny - fd_pad_list[3] - pml_width_list[3]
    pml_x1 = padded_nx - fd_pad_list[5] - pml_width_list[5]

    background_coeff_requires_grad = (
        ca.requires_grad or cb.requires_grad or f0.requires_grad
    )
    needs_storage = (
        dca_native.requires_grad
        or dcb_native.requires_grad
        or background_coeff_requires_grad
    )
    needs_autograd = needs_storage or df.requires_grad

    storage_mode_str = str(storage_mode).lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )
    if needs_storage and storage_mode_str == "none":
        raise ValueError(
            "storage_mode='none' is not compatible with gradient computation "
            "for native born3d."
        )
    storage_kind = _normalize_storage_compression(storage_compression)
    if needs_autograd and storage_kind == "bf16" and device.type != "cuda":
        raise NotImplementedError(
            "Native born3d BF16 snapshot storage is currently supported only on CUDA."
        )
    _, _, storage_bytes_per_elem, _ = _resolve_storage_compression(
        storage_compression,
        dtype,
        device,
        context="storage_compression",
    )
    effective_storage_mode_str = storage_mode_str
    if needs_storage:
        if device.type == "cpu" and effective_storage_mode_str in {
            "cpu",
            "disk",
            "auto",
        }:
            effective_storage_mode_str = "device"
        elif effective_storage_mode_str == "auto":
            num_steps_stored = (
                nt_steps + gradient_sampling_interval - 1
            ) // gradient_sampling_interval
            snapshot_components = 0
            if dca_native.requires_grad or background_coeff_requires_grad:
                snapshot_components += 3
            if dcb_native.requires_grad or background_coeff_requires_grad:
                snapshot_components += 3
            if ca.requires_grad:
                snapshot_components += 3
            if cb.requires_grad:
                snapshot_components += 3
            total_bytes = (
                n_shots
                * padded_nz
                * padded_ny
                * padded_nx
                * storage_bytes_per_elem
                * num_steps_stored
                * snapshot_components
            )
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
        if storage_kind == "bf16" and effective_storage_mode_str != "device":
            raise NotImplementedError(
                "Native born3d BF16 snapshot storage currently requires "
                "storage_mode='device'."
            )
    elif effective_storage_mode_str == "auto":
        effective_storage_mode_str = "none"

    source_component_idx = _COMPONENT_TO_INDEX_3D[source_component]
    receiver_component_idx = _COMPONENT_TO_INDEX_3D[receiver_component]

    try:
        _ = backend_utils.get_backend_function(
            "maxwell_3d",
            "born_forward_with_storage" if needs_autograd else "born_forward",
            stencil,
            dtype,
            backend_device,
        )
        if needs_autograd:
            _ = backend_utils.get_backend_function(
                "maxwell_3d",
                "born_backward",
                stencil,
                dtype,
                backend_device,
            )
        if background_coeff_requires_grad:
            _ = backend_utils.get_backend_function(
                "maxwell_3d",
                "born_backward_bggrad",
                stencil,
                dtype,
                backend_device,
            )
    except (RuntimeError, AttributeError, TypeError) as exc:
        warnings.warn(
            f"3D native born symbols are unavailable ({exc}); falling back to Python reference path.",
            RuntimeWarning,
        )
        return _to_native_output_layout(
            born3d_python(
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
                stencil=stencil,
                pml_width=pml_width,
                max_vel=max_vel,
                dEx_0=dEx_0,
                dEy_0=dEy_0,
                dEz_0=dEz_0,
                dHx_0=dHx_0,
                dHy_0=dHy_0,
                dHz_0=dHz_0,
                dm_hz_y_0=dm_hz_y_0,
                dm_hy_z_0=dm_hy_z_0,
                dm_hx_z_0=dm_hx_z_0,
                dm_hz_x_0=dm_hz_x_0,
                dm_hy_x_0=dm_hy_x_0,
                dm_hx_y_0=dm_hx_y_0,
                dm_ey_z_0=dm_ey_z_0,
                dm_ez_y_0=dm_ez_y_0,
                dm_ez_x_0=dm_ez_x_0,
                dm_ex_z_0=dm_ex_z_0,
                dm_ex_y_0=dm_ex_y_0,
                dm_ey_x_0=dm_ey_x_0,
                nt=nt,
                parameterization=parameterization,
                linearize_source=linearize_source,
                source_component=source_component,
                receiver_component=receiver_component,
            )
        )

    if needs_autograd:
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
            "n_threads": n_threads_val,
            "rdz": 1.0 / dz,
            "rdy": 1.0 / dy,
            "rdx": 1.0 / dx,
            "backend_device": backend_device,
            "storage_compression": storage_compression,
            "storage_mode_str": effective_storage_mode_str,
            "storage_path": storage_path,
        }
        outputs = Born3DForwardFunc.apply(
            dca_native,
            dcb_native,
            ca,
            cb,
            cq,
            f0,
            df,
            profiles,
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
            (
                dEx,
                dEy,
                dEz,
                dHx,
                dHy,
                dHz,
                dm_hz_y,
                dm_hy_z,
                dm_hx_z,
                dm_hz_x,
                dm_hy_x,
                dm_hx_y,
                dm_ey_z,
                dm_ez_y,
                dm_ez_x,
                dm_ex_z,
                dm_ex_y,
                dm_ey_x,
            ),
            meta,
        )
        (
            dEx_out,
            dEy_out,
            dEz_out,
            dHx_out,
            dHy_out,
            dHz_out,
            dm_hz_y_out,
            dm_hy_z_out,
            dm_hx_z_out,
            dm_hz_x_out,
            dm_hy_x_out,
            dm_hx_y_out,
            dm_ey_z_out,
            dm_ez_y_out,
            dm_ez_x_out,
            dm_ex_z_out,
            dm_ex_y_out,
            dm_ey_x_out,
            receiver_amplitudes,
        ) = outputs
    else:
        forward_func = backend_utils.get_backend_function(
            "maxwell_3d",
            "born_forward",
            stencil,
            dtype,
            backend_device,
        )
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt_steps, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        compute_stream_handle = 0
        if device.type == "cuda":
            compute_stream_handle = int(
                getattr(torch.cuda.current_stream(device=device), "cuda_stream", 0) or 0
            )
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        forward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(dca_native),
            backend_utils.tensor_to_ptr(dcb_native),
            backend_utils.tensor_to_ptr(f0),
            backend_utils.tensor_to_ptr(df),
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
            backend_utils.tensor_to_ptr(dEx),
            backend_utils.tensor_to_ptr(dEy),
            backend_utils.tensor_to_ptr(dEz),
            backend_utils.tensor_to_ptr(dHx),
            backend_utils.tensor_to_ptr(dHy),
            backend_utils.tensor_to_ptr(dHz),
            backend_utils.tensor_to_ptr(dm_hz_y),
            backend_utils.tensor_to_ptr(dm_hy_z),
            backend_utils.tensor_to_ptr(dm_hx_z),
            backend_utils.tensor_to_ptr(dm_hz_x),
            backend_utils.tensor_to_ptr(dm_hy_x),
            backend_utils.tensor_to_ptr(dm_hx_y),
            backend_utils.tensor_to_ptr(dm_ey_z),
            backend_utils.tensor_to_ptr(dm_ez_y),
            backend_utils.tensor_to_ptr(dm_ez_x),
            backend_utils.tensor_to_ptr(dm_ex_z),
            backend_utils.tensor_to_ptr(dm_ex_y),
            backend_utils.tensor_to_ptr(dm_ey_x),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
            *(backend_utils.tensor_to_ptr(p) for p in profiles),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            1.0 / dz,
            1.0 / dy,
            1.0 / dx,
            dt,
            nt_steps,
            n_shots,
            padded_nz,
            padded_ny,
            padded_nx,
            n_sources,
            n_receivers,
            gradient_sampling_interval,
            False,
            False,
            False,
            0,
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
            0,
            compute_stream_handle,
        )
        (
            dEx_out,
            dEy_out,
            dEz_out,
            dHx_out,
            dHy_out,
            dHz_out,
            dm_hz_y_out,
            dm_hy_z_out,
            dm_hx_z_out,
            dm_hz_x_out,
            dm_hy_x_out,
            dm_hx_y_out,
            dm_ey_z_out,
            dm_ez_y_out,
            dm_ez_x_out,
            dm_ex_z_out,
            dm_ex_y_out,
            dm_ey_x_out,
        ) = (
            dEx,
            dEy,
            dEz,
            dHx,
            dHy,
            dHz,
            dm_hz_y,
            dm_hy_z,
            dm_hx_z,
            dm_hz_x,
            dm_hy_x,
            dm_hx_y,
            dm_ey_z,
            dm_ez_y,
            dm_ez_x,
            dm_ex_z,
            dm_ex_y,
            dm_ey_x,
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
    return (
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
        dEx_out[s],
        dEy_out[s],
        dEz_out[s],
        dHx_out[s],
        dHy_out[s],
        dHz_out[s],
        dm_hz_y_out[s],
        dm_hy_z_out[s],
        dm_hx_z_out[s],
        dm_hz_x_out[s],
        dm_hy_x_out[s],
        dm_hx_y_out[s],
        dm_ey_z_out[s],
        dm_ez_y_out[s],
        dm_ez_x_out[s],
        dm_ex_z_out[s],
        dm_ex_y_out[s],
        dm_ey_x_out[s],
        receiver_amplitudes,
    )


__all__ = ["born3d_c_cuda"]
