import warnings
from collections.abc import Sequence

import torch

from ..callbacks import Callback, CallbackState
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_3d, _normalize_pml_width_3d
from ..padding import create_or_pad, zero_interior
from ..storage import _normalize_storage_compression
from ..utils import C0, compile_material_coefficients
from .common import (
    _debye_polarization_term,
    _init_polarization_state,
    _pad_dispersion_for_model,
)

def _select_e_component(
    component: str,
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
) -> torch.Tensor:
    if component == "ex":
        return ex
    if component == "ey":
        return ey
    return ez


def _inject_component(
    field: torch.Tensor,
    flat_shape: int,
    indices: torch.Tensor,
    values: torch.Tensor,
    output_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    return (
        field.reshape(output_shape[0], flat_shape)
        .scatter_add(1, indices, values)
        .reshape(output_shape)
    )


def maxwell3d_python(
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
    *,
    validate_material_inputs: bool = True,
):
    """3D Python backend propagation with autograd support."""
    del (
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        backward_callback,
        execution_backend,
        storage_path,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        n_threads,
    )
    if epsilon.ndim == 4:
        raise NotImplementedError(
            "Batched models are supported only on the native C/CUDA backend in v1."
        )
    if epsilon.ndim != 3:
        raise RuntimeError("epsilon must be 3D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

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
    if dispersion is not None and any(
        state is not None
        for state in (
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
        )
    ):
        warnings.warn(
            "Debye v1 does not support persisting polarization state across calls; "
            "field initial conditions are applied, but polarization restarts from zero.",
            RuntimeWarning,
        )

    device = epsilon.device
    dtype = epsilon.dtype
    model_nz, model_ny, model_nx = epsilon.shape

    grid_spacing_list = _normalize_grid_spacing_3d(grid_spacing)
    dz, dy, dx = grid_spacing_list
    pml_width_list = _normalize_pml_width_3d(pml_width)

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
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]

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
    ca = material["ca"]
    cb = material["cb"]
    cq = material["cq"]
    has_dispersion = bool(material["has_dispersion"])
    debye = material.get("debye")
    ca = ca[None, :, :, :]
    cb = cb[None, :, :, :]
    cq = cq[None, :, :, :]

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
            )
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
    pol_ex = pol_ey = pol_ez = None
    if has_dispersion and debye is not None:
        pol_ex = _init_polarization_state(
            n_shots=n_shots,
            n_poles=debye["n_poles"],
            spatial_shape=(padded_nz, padded_ny, padded_nx),
            device=device,
            dtype=dtype,
        )
        pol_ey = torch.zeros_like(pol_ex)
        pol_ez = torch.zeros_like(pol_ex)

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

    from .. import staggered as _staggered

    pml_ab_profiles, pml_k_profiles = _staggered.set_pml_profiles_3d(
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

    rdz = torch.tensor(1.0 / dz, device=device, dtype=dtype)
    rdy = torch.tensor(1.0 / dy, device=device, dtype=dtype)
    rdx = torch.tensor(1.0 / dx, device=device, dtype=dtype)

    flat_model_shape = padded_nz * padded_ny * padded_nx
    if source_location is not None and source_location.numel() > 0:
        source_z = source_location[..., 0] + total_pad[0]
        source_y = source_location[..., 1] + total_pad[2]
        source_x = source_location[..., 2] + total_pad[4]
        sources_i = ((source_z * padded_ny + source_y) * padded_nx + source_x).long()
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
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    receiver_samples: list[torch.Tensor] = []

    source_coeff = -1.0 / (dx * dy * dz)
    if n_sources > 0 and source_amplitude is not None and source_amplitude.numel() > 0:
        cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, sources_i)
    else:
        cb_at_src = torch.empty(0, device=device, dtype=dtype)

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

    for step in range(nt_steps):
        if forward_callback is not None and step % callback_frequency == 0:
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
            if pol_ex is not None and pol_ey is not None and pol_ez is not None:
                callback_wavefields["polarization"] = torch.stack(
                    (pol_ex.sum(dim=1), pol_ey.sum(dim=1), pol_ez.sum(dim=1)),
                    dim=1,
                )
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
                    grid_spacing=[dz, dy, dx],
                )
            )

        # H update using half-grid derivatives of E
        dEy_dz = _staggered.diffzh1(Ey, stencil, rdz)
        dEz_dy = _staggered.diffyh1(Ez, stencil, rdy)
        dEz_dx = _staggered.diffxh1(Ez, stencil, rdx)
        dEx_dz = _staggered.diffzh1(Ex, stencil, rdz)
        dEx_dy = _staggered.diffyh1(Ex, stencil, rdy)
        dEy_dx = _staggered.diffxh1(Ey, stencil, rdx)

        m_ey_z = bz_h * m_ey_z + az_h * dEy_dz
        m_ez_y = by_h * m_ez_y + ay_h * dEz_dy
        m_ez_x = bx_h * m_ez_x + ax_h * dEz_dx
        m_ex_z = bz_h * m_ex_z + az_h * dEx_dz
        m_ex_y = by_h * m_ex_y + ay_h * dEx_dy
        m_ey_x = bx_h * m_ey_x + ax_h * dEy_dx

        dEy_dz_pml = dEy_dz / kz_h + m_ey_z
        dEz_dy_pml = dEz_dy / ky_h + m_ez_y
        dEz_dx_pml = dEz_dx / kx_h + m_ez_x
        dEx_dz_pml = dEx_dz / kz_h + m_ex_z
        dEx_dy_pml = dEx_dy / ky_h + m_ex_y
        dEy_dx_pml = dEy_dx / kx_h + m_ey_x

        Hx = Hx - cq * (dEy_dz_pml - dEz_dy_pml)
        Hy = Hy - cq * (dEz_dx_pml - dEx_dz_pml)
        Hz = Hz - cq * (dEx_dy_pml - dEy_dx_pml)

        # E update using integer-grid derivatives of H
        dHy_dz = _staggered.diffz1(Hy, stencil, rdz)
        dHz_dy = _staggered.diffy1(Hz, stencil, rdy)
        dHz_dx = _staggered.diffx1(Hz, stencil, rdx)
        dHx_dz = _staggered.diffz1(Hx, stencil, rdz)
        dHx_dy = _staggered.diffy1(Hx, stencil, rdy)
        dHy_dx = _staggered.diffx1(Hy, stencil, rdx)

        m_hy_z = bz * m_hy_z + az * dHy_dz
        m_hz_y = by * m_hz_y + ay * dHz_dy
        m_hz_x = bx * m_hz_x + ax * dHz_dx
        m_hx_z = bz * m_hx_z + az * dHx_dz
        m_hx_y = by * m_hx_y + ay * dHx_dy
        m_hy_x = bx * m_hy_x + ax * dHy_dx

        dHy_dz_pml = dHy_dz / kz + m_hy_z
        dHz_dy_pml = dHz_dy / ky + m_hz_y
        dHz_dx_pml = dHz_dx / kx + m_hz_x
        dHx_dz_pml = dHx_dz / kz + m_hx_z
        dHx_dy_pml = dHx_dy / ky + m_hx_y
        dHy_dx_pml = dHy_dx / kx + m_hy_x

        ex_prev = Ex
        ey_prev = Ey
        ez_prev = Ez
        Ex = ca * Ex + cb * (dHy_dz_pml - dHz_dy_pml)
        Ey = ca * Ey + cb * (dHz_dx_pml - dHx_dz_pml)
        Ez = ca * Ez + cb * (dHx_dy_pml - dHy_dx_pml)
        if pol_ex is not None and pol_ey is not None and pol_ez is not None and debye is not None:
            Ex = Ex + _debye_polarization_term(debye["cp"], pol_ex)
            Ey = Ey + _debye_polarization_term(debye["cp"], pol_ey)
            Ez = Ez + _debye_polarization_term(debye["cp"], pol_ez)

        if (
            source_amplitude is not None
            and source_amplitude.numel() > 0
            and n_sources > 0
        ):
            src_amp = source_amplitude[:, :, step]
            scaled_src = cb_at_src * src_amp * source_coeff
            if source_component == "ex":
                Ex = _inject_component(
                    Ex, flat_model_shape, sources_i, scaled_src, size_with_batch
                )
            elif source_component == "ey":
                Ey = _inject_component(
                    Ey, flat_model_shape, sources_i, scaled_src, size_with_batch
                )
            else:
                Ez = _inject_component(
                    Ez, flat_model_shape, sources_i, scaled_src, size_with_batch
                )
        if pol_ex is not None and pol_ey is not None and pol_ez is not None and debye is not None:
            a = debye["a"].unsqueeze(0)
            b = debye["b"].unsqueeze(0)
            pol_ex = a * pol_ex + b * (Ex + ex_prev).unsqueeze(1)
            pol_ey = a * pol_ey + b * (Ey + ey_prev).unsqueeze(1)
            pol_ez = a * pol_ez + b * (Ez + ez_prev).unsqueeze(1)

        if n_receivers > 0:
            rec_field = _select_e_component(receiver_component, Ex, Ey, Ez)
            receiver_samples.append(
                rec_field.reshape(n_shots, flat_model_shape).gather(1, receivers_i)
            )

    if n_receivers > 0:
        receiver_amplitudes = torch.stack(receiver_samples, dim=0)
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

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

__all__ = ["maxwell3d_python"]
