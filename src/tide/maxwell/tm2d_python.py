import warnings
from collections.abc import Callable, Sequence

import torch

from .. import staggered
from ..callbacks import Callback, CallbackState
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_2d, _normalize_pml_width_2d
from ..padding import create_or_pad, zero_interior
from ..storage import _normalize_storage_compression
from ..utils import C0, EP0, compile_material_coefficients
from .common import (
    _debye_polarization_term,
    _init_polarization_state,
    _pad_dispersion_for_model,
)
from .tm2d_helpers import _init_tm_wavefield

_update_E_jit: Callable | None = None
_update_E_compile: Callable | None = None
_update_H_jit: Callable | None = None
_update_H_compile: Callable | None = None

_update_E_opt: Callable | None = None
_update_H_opt: Callable | None = None


def maxwell_func(
    python_backend: bool | str,
    *args,
    validate_material_inputs: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Dispatch to Python or C/CUDA backend for Maxwell propagation."""
    global _update_E_jit, _update_E_compile, _update_E_opt
    global _update_H_jit, _update_H_compile, _update_H_opt

    device_type = (
        args[0].device.type
        if len(args) > 0 and isinstance(args[0], torch.Tensor)
        else "cpu"
    )

    use_python = python_backend
    if device_type not in {"cpu", "cuda"} and not use_python:
        use_python = True

    if not use_python:
        try:
            from .. import backend_utils

            if not backend_utils.is_backend_available():
                warnings.warn(
                    "C/CUDA backend not available, falling back to Python backend. "
                    "To use the C/CUDA backend, compile the library first.",
                    RuntimeWarning,
                )
                use_python = True
        except ImportError:
            warnings.warn(
                "backend_utils not available, falling back to Python backend.",
                RuntimeWarning,
            )
            use_python = True

    if use_python:
        if python_backend is True or python_backend is False:
            mode = "eager"
        elif isinstance(python_backend, str):
            mode = python_backend.lower()
        else:
            raise TypeError(
                f"python_backend must be bool or str, but got {type(python_backend)}"
            )

        if mode == "jit":
            if _update_E_jit is None:
                _update_E_jit = torch.jit.script(update_E)
            _update_E_opt = _update_E_jit
            if _update_H_jit is None:
                _update_H_jit = torch.jit.script(update_H)
            _update_H_opt = _update_H_jit
        elif mode == "compile":
            if _update_E_compile is None:
                _update_E_compile = torch.compile(update_E, fullgraph=True)
            _update_E_opt = _update_E_compile
            if _update_H_compile is None:
                _update_H_compile = torch.compile(update_H, fullgraph=True)
            _update_H_opt = _update_H_compile
        elif mode == "eager":
            _update_E_opt = update_E
            _update_H_opt = update_H
        else:
            raise ValueError(f"Unknown python_backend value {mode!r}.")

        return maxwell_python(*args, validate_material_inputs=validate_material_inputs)

    from .tm2d_cuda import maxwell_c_cuda

    return maxwell_c_cuda(*args)


def maxwell_python(
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
    *,
    validate_material_inputs: bool = True,
):
    """Performs the forward propagation of the 2D TM Maxwell equations."""
    del backward_callback
    del save_snapshots
    del storage_path
    del storage_bytes_limit_device
    del storage_bytes_limit_host
    del storage_chunk_steps
    del n_threads
    del model_gradient_sampling_interval

    assert _update_E_opt is not None, "_update_E_opt must be set by maxwell_func"
    assert _update_H_opt is not None, "_update_H_opt must be set by maxwell_func"

    if epsilon.ndim == 3:
        raise NotImplementedError(
            "Batched models are supported only on the native C/CUDA backend in v1."
        )
    if epsilon.ndim != 2:
        raise RuntimeError("epsilon must be 2D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    device = epsilon.device
    dtype = epsilon.dtype
    model_ny, model_nx = epsilon.shape

    if dispersion is not None and any(
        state is not None
        for state in (Ey_0, Hx_0, Hz_0, m_Ey_x_0, m_Ey_z_0, m_Hx_z_0, m_Hz_x_0)
    ):
        warnings.warn(
            "Debye v1 does not support persisting polarization state across calls; "
            "field initial conditions are applied, but polarization restarts from zero.",
            RuntimeWarning,
        )

    storage_mode_str = storage_mode.lower()
    if storage_mode_str in {"cpu", "disk"}:
        raise ValueError(
            "python_backend does not support storage_mode='cpu' or 'disk'. "
            "Use the C/CUDA backend or storage_mode='device'/'none'."
        )
    storage_kind = _normalize_storage_compression(storage_compression)
    if storage_kind != "none":
        raise NotImplementedError(
            "storage_compression is not implemented yet; set storage_compression=False."
        )

    grid_spacing = _normalize_grid_spacing_2d(grid_spacing)
    dy, dx = grid_spacing
    pml_width_list = _normalize_pml_width_2d(pml_width)

    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = source_amplitude.shape[-1]
    nt_steps = int(nt)

    if source_amplitude is not None and source_amplitude.numel() > 0:
        n_shots = source_amplitude.shape[0]
    elif source_location is not None and source_location.numel() > 0:
        n_shots = source_location.shape[0]
    elif receiver_location is not None and receiver_location.numel() > 0:
        n_shots = receiver_location.shape[0]
    else:
        n_shots = 1

    if max_vel is None:
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    pml_freq = 0.5 / dt

    fd_pad = stencil // 2
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]
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
    dispersion_padded = _pad_dispersion_for_model(
        dispersion,
        model_shape=tuple(epsilon.shape),
        total_pad=total_pad,
        padded_size=padded_size,
        device=device,
        dtype=dtype,
    )

    material = compile_material_coefficients(
        epsilon_padded,
        sigma_padded,
        mu_padded,
        dt,
        dispersion=dispersion_padded,
        validate_inputs=validate_material_inputs,
    )
    ca = material["ca"][None, :, :]
    cb = material["cb"][None, :, :]
    cq = material["cq"][None, :, :]
    has_dispersion = bool(material["has_dispersion"])
    debye = material.get("debye")

    size_with_batch = (n_shots, padded_ny, padded_nx)
    Ey = _init_tm_wavefield(
        Ey_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    Hx = _init_tm_wavefield(
        Hx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    Hz = _init_tm_wavefield(
        Hz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_Ey_x = _init_tm_wavefield(
        m_Ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_Ey_z = _init_tm_wavefield(
        m_Ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_Hx_z = _init_tm_wavefield(
        m_Hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_Hz_x = _init_tm_wavefield(
        m_Hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )

    polarization = None
    if has_dispersion and debye is not None:
        polarization = _init_polarization_state(
            n_shots=n_shots,
            n_poles=debye["n_poles"],
            spatial_shape=(padded_ny, padded_nx),
            device=device,
            dtype=dtype,
        )

    pml_aux_dims = [1, 0, 0, 1]
    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], pml_aux_dims):
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
        kappa_y,
        kappa_y_h,
        kappa_x,
        kappa_x_h,
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

    rdy = torch.tensor(1.0 / dy, device=device, dtype=dtype)
    rdx = torch.tensor(1.0 / dx, device=device, dtype=dtype)
    dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

    flat_model_shape = padded_ny * padded_nx
    if source_location is not None and source_location.numel() > 0:
        source_y = source_location[..., 0] + total_pad[0]
        source_x = source_location[..., 1] + total_pad[2]
        sources_i = (source_y * padded_nx + source_x).long()
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        receiver_y = receiver_location[..., 0] + total_pad[0]
        receiver_x = receiver_location[..., 1] + total_pad[2]
        receivers_i = (receiver_y * padded_nx + receiver_x).long()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    receiver_samples: list[torch.Tensor] = []

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

    source_coeff = -1.0 / (dx * dy)

    for step in range(nt_steps):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = {
                "Ey": Ey,
                "Hx": Hx,
                "Hz": Hz,
                "m_Ey_x": m_Ey_x,
                "m_Ey_z": m_Ey_z,
                "m_Hx_z": m_Hx_z,
                "m_Hz_x": m_Hz_x,
            }
            if polarization is not None:
                callback_wavefields["polarization"] = polarization.sum(dim=1)
            forward_callback(
                CallbackState(
                    dt=dt,
                    step=step,
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

        Hx, Hz, m_Ey_x, m_Ey_z = _update_H_opt(
            cq,
            Hx,
            Hz,
            Ey,
            m_Ey_x,
            m_Ey_z,
            kappa_y,
            kappa_y_h,
            kappa_x,
            kappa_x_h,
            ay,
            ay_h,
            ax,
            ax_h,
            by,
            by_h,
            bx,
            bx_h,
            rdy,
            rdx,
            dt_tensor,
            stencil,
        )

        ey_prev = Ey
        Ey, m_Hx_z, m_Hz_x = _update_E_opt(
            ca,
            cb,
            Hx,
            Hz,
            Ey,
            m_Hx_z,
            m_Hz_x,
            kappa_y,
            kappa_y_h,
            kappa_x,
            kappa_x_h,
            ay,
            ay_h,
            ax,
            ax_h,
            by,
            by_h,
            bx,
            bx_h,
            rdy,
            rdx,
            dt_tensor,
            stencil,
        )
        if polarization is not None and debye is not None:
            Ey = Ey + _debye_polarization_term(debye["cp"], polarization)

        if (
            source_amplitude is not None
            and source_amplitude.numel() > 0
            and n_sources > 0
        ):
            src_amp = source_amplitude[:, :, step]
            cb_flat = ca.new_empty(0)
            cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
            cb_at_src = cb_flat.gather(1, sources_i)
            scaled_src = cb_at_src * src_amp * source_coeff
            Ey = (
                Ey.reshape(n_shots, flat_model_shape)
                .scatter_add(1, sources_i, scaled_src)
                .reshape(size_with_batch)
            )
        if polarization is not None and debye is not None:
            polarization = (
                debye["a"].unsqueeze(0) * polarization
                + debye["b"].unsqueeze(0) * (Ey + ey_prev).unsqueeze(1)
            )

        if n_receivers > 0:
            receiver_samples.append(
                Ey.reshape(n_shots, flat_model_shape).gather(1, receivers_i)
            )

    if n_receivers > 0:
        receiver_amplitudes = torch.stack(receiver_samples, dim=0)
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

    s = (
        slice(None),
        slice(
            fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None
        ),
        slice(
            fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None
        ),
    )

    return (
        Ey[s],
        Hx[s],
        Hz[s],
        m_Ey_x[s],
        m_Ey_z[s],
        m_Hx_z[s],
        m_Hz_x[s],
        receiver_amplitudes,
    )


def update_E(
    ca: torch.Tensor,
    cb: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    Ey: torch.Tensor,
    m_Hx_z: torch.Tensor,
    m_Hz_x: torch.Tensor,
    kappa_y: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x: torch.Tensor,
    kappa_x_h: torch.Tensor,
    ay: torch.Tensor,
    ay_h: torch.Tensor,
    ax: torch.Tensor,
    ax_h: torch.Tensor,
    by: torch.Tensor,
    by_h: torch.Tensor,
    bx: torch.Tensor,
    bx_h: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    stencil: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del dt

    dHz_dx = staggered.diffx1(Hz, stencil, rdx)
    dHx_dz = staggered.diffy1(Hx, stencil, rdy)

    m_Hz_x = bx * m_Hz_x + ax * dHz_dx
    m_Hx_z = by * m_Hx_z + ay * dHx_dz

    dHz_dx_pml = dHz_dx / kappa_x + m_Hz_x
    dHx_dz_pml = dHx_dz / kappa_y + m_Hx_z

    Ey = ca * Ey + cb * (dHz_dx_pml - dHx_dz_pml)
    return Ey, m_Hx_z, m_Hz_x


def update_H(
    cq: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    Ey: torch.Tensor,
    m_Ey_x: torch.Tensor,
    m_Ey_z: torch.Tensor,
    kappa_y: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x: torch.Tensor,
    kappa_x_h: torch.Tensor,
    ay: torch.Tensor,
    ay_h: torch.Tensor,
    ax: torch.Tensor,
    ax_h: torch.Tensor,
    by: torch.Tensor,
    by_h: torch.Tensor,
    bx: torch.Tensor,
    bx_h: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    stencil: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del kappa_y
    del kappa_x
    del ay
    del ax
    del by
    del bx
    del dt

    dEy_dz = staggered.diffyh1(Ey, stencil, rdy)
    dEy_dx = staggered.diffxh1(Ey, stencil, rdx)

    m_Ey_z = by_h * m_Ey_z + ay_h * dEy_dz
    m_Ey_x = bx_h * m_Ey_x + ax_h * dEy_dx

    dEy_dz_pml = dEy_dz / kappa_y_h + m_Ey_z
    dEy_dx_pml = dEy_dx / kappa_x_h + m_Ey_x

    Hx = Hx - cq * dEy_dz_pml
    Hz = Hz + cq * dEy_dx_pml
    return Hx, Hz, m_Ey_x, m_Ey_z


_update_E_opt = update_E
_update_H_opt = update_H


__all__ = ["maxwell_func", "maxwell_python", "update_E", "update_H"]
