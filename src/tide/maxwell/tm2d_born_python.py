from collections.abc import Sequence

import torch

from .. import staggered
from ..grid_utils import _normalize_grid_spacing_2d, _normalize_pml_width_2d
from ..padding import create_or_pad, zero_interior
from ..utils import (
    C0,
    EP0,
    compile_material_coefficients,
    linearize_material_coefficients,
)
from .tm2d_helpers import _init_tm_wavefield
from .validation_internal import _validate_location_bounds


def update_H_born(
    cq: torch.Tensor,
    dHx: torch.Tensor,
    dHz: torch.Tensor,
    dEy: torch.Tensor,
    dm_Ey_x: torch.Tensor,
    dm_Ey_z: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x_h: torch.Tensor,
    ay_h: torch.Tensor,
    ax_h: torch.Tensor,
    by_h: torch.Tensor,
    bx_h: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    stencil: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_dEy_dz = staggered.diffyh1(dEy, stencil, rdy)
    d_dEy_dx = staggered.diffxh1(dEy, stencil, rdx)

    dm_Ey_z = by_h * dm_Ey_z + ay_h * d_dEy_dz
    dm_Ey_x = bx_h * dm_Ey_x + ax_h * d_dEy_dx

    d_dEy_dz_pml = d_dEy_dz / kappa_y_h + dm_Ey_z
    d_dEy_dx_pml = d_dEy_dx / kappa_x_h + dm_Ey_x

    dHx = dHx - cq * d_dEy_dz_pml
    dHz = dHz + cq * d_dEy_dx_pml
    return dHx, dHz, dm_Ey_x, dm_Ey_z


def update_E_born(
    ca: torch.Tensor,
    cb: torch.Tensor,
    dca: torch.Tensor,
    dcb: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    Ey: torch.Tensor,
    dHx: torch.Tensor,
    dHz: torch.Tensor,
    dEy: torch.Tensor,
    m_Hx_z: torch.Tensor,
    m_Hz_x: torch.Tensor,
    dm_Hx_z: torch.Tensor,
    dm_Hz_x: torch.Tensor,
    kappa_y: torch.Tensor,
    kappa_x: torch.Tensor,
    ay: torch.Tensor,
    ax: torch.Tensor,
    by: torch.Tensor,
    bx: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    stencil: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    dHz_dx = staggered.diffx1(Hz, stencil, rdx)
    dHx_dz = staggered.diffy1(Hx, stencil, rdy)

    m_Hz_x = bx * m_Hz_x + ax * dHz_dx
    m_Hx_z = by * m_Hx_z + ay * dHx_dz

    dHz_dx_pml = dHz_dx / kappa_x + m_Hz_x
    dHx_dz_pml = dHx_dz / kappa_y + m_Hx_z
    curl_h = dHz_dx_pml - dHx_dz_pml

    ddHz_dx = staggered.diffx1(dHz, stencil, rdx)
    ddHx_dz = staggered.diffy1(dHx, stencil, rdy)

    dm_Hz_x = bx * dm_Hz_x + ax * ddHz_dx
    dm_Hx_z = by * dm_Hx_z + ay * ddHx_dz

    ddHz_dx_pml = ddHz_dx / kappa_x + dm_Hz_x
    ddHx_dz_pml = ddHx_dz / kappa_y + dm_Hx_z
    dcurl_h = ddHz_dx_pml - ddHx_dz_pml

    Ey_old = Ey
    dEy = ca * dEy + cb * dcurl_h + dca * Ey_old + dcb * curl_h
    Ey = ca * Ey + cb * curl_h

    return Ey, dEy, m_Hx_z, m_Hz_x, dm_Hx_z, dm_Hz_x


def borntm_python(
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
    bg_receiver_location: torch.Tensor | None = None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    Ey_0: torch.Tensor | None = None,
    Hx_0: torch.Tensor | None = None,
    Hz_0: torch.Tensor | None = None,
    m_Ey_x_0: torch.Tensor | None = None,
    m_Ey_z_0: torch.Tensor | None = None,
    m_Hx_z_0: torch.Tensor | None = None,
    m_Hz_x_0: torch.Tensor | None = None,
    dEy_0: torch.Tensor | None = None,
    dHx_0: torch.Tensor | None = None,
    dHz_0: torch.Tensor | None = None,
    dm_Ey_x_0: torch.Tensor | None = None,
    dm_Ey_z_0: torch.Tensor | None = None,
    dm_Hx_z_0: torch.Tensor | None = None,
    dm_Hz_x_0: torch.Tensor | None = None,
    nt: int | None = None,
    parameterization: str = "epsilon_sigma",
    linearize_source: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if epsilon.ndim != 2:
        raise RuntimeError("epsilon must be 2D for borntm_python")
    if sigma.shape != epsilon.shape or mu.shape != epsilon.shape:
        raise RuntimeError("sigma and mu must have the same shape as epsilon")
    if parameterization not in {"epsilon_sigma", "ca_cb"}:
        raise ValueError(
            "parameterization must be 'epsilon_sigma' or 'ca_cb', "
            f"got {parameterization!r}."
        )

    device = epsilon.device
    dtype = epsilon.dtype
    model_ny, model_nx = epsilon.shape

    _validate_location_bounds(
        source_location,
        shape=(model_ny, model_nx),
        name="Source location",
        check_lower_bound=False,
    )
    _validate_location_bounds(
        receiver_location,
        shape=(model_ny, model_nx),
        name="Receiver location",
        check_lower_bound=False,
    )
    _validate_location_bounds(
        bg_receiver_location,
        shape=(model_ny, model_nx),
        name="Background receiver location",
        check_lower_bound=False,
    )

    grid_spacing_list = _normalize_grid_spacing_2d(grid_spacing)
    dy, dx = grid_spacing_list
    pml_width_list = _normalize_pml_width_2d(pml_width)

    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = int(source_amplitude.shape[-1])
    nt_steps = int(nt)

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
        validate_inputs=True,
    )
    ca = material["ca"][None, :, :]
    cb = material["cb"][None, :, :]
    cq = material["cq"][None, :, :]

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

    dca_b = dca_padded[None, :, :]
    dcb_b = dcb_padded[None, :, :]

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

    dEy = _init_tm_wavefield(
        dEy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dHx = _init_tm_wavefield(
        dHx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dHz = _init_tm_wavefield(
        dHz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_Ey_x = _init_tm_wavefield(
        dm_Ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_Ey_z = _init_tm_wavefield(
        dm_Ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_Hx_z = _init_tm_wavefield(
        dm_Hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_Hz_x = _init_tm_wavefield(
        dm_Hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )

    pml_aux_dims = [1, 0, 0, 1]
    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], pml_aux_dims):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)
    for wf, dim in zip([dm_Ey_x, dm_Ey_z, dm_Hx_z, dm_Hz_x], pml_aux_dims):
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
        grid_spacing=grid_spacing_list,
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

    flat_model_shape = padded_ny * padded_nx
    if source_location is not None and source_location.numel() > 0:
        source_y = source_location[..., 0] + total_pad[0]
        source_x = source_location[..., 1] + total_pad[2]
        sources_i = (source_y * padded_nx + source_x).long()
        n_sources = int(source_location.shape[1])
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        receiver_y = receiver_location[..., 0] + total_pad[0]
        receiver_x = receiver_location[..., 1] + total_pad[2]
        receivers_i = (receiver_y * padded_nx + receiver_x).long()
        n_receivers = int(receiver_location.shape[1])
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    if bg_receiver_location is not None and bg_receiver_location.numel() > 0:
        bg_receiver_y = bg_receiver_location[..., 0] + total_pad[0]
        bg_receiver_x = bg_receiver_location[..., 1] + total_pad[2]
        bg_receivers_i = (bg_receiver_y * padded_nx + bg_receiver_x).long()
        n_bg_receivers = int(bg_receiver_location.shape[1])
    else:
        bg_receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_bg_receivers = 0

    source_coeff = -1.0 / (dx * dy)
    if source_amplitude is not None and source_amplitude.numel() > 0 and n_sources > 0:
        cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, sources_i)

        if linearize_source:
            dcb_flat = dcb_b.reshape(1, flat_model_shape).expand(n_shots, -1)
            dcb_at_src = dcb_flat.gather(1, sources_i)
        else:
            dcb_at_src = torch.zeros_like(cb_at_src)
    else:
        cb_at_src = torch.empty(0, device=device, dtype=dtype)
        dcb_at_src = torch.empty(0, device=device, dtype=dtype)

    bg_receiver_samples: list[torch.Tensor] = []
    receiver_samples: list[torch.Tensor] = []

    for step in range(nt_steps):
        Hx, Hz, m_Ey_x, m_Ey_z = _update_background_H(
            cq,
            Hx,
            Hz,
            Ey,
            m_Ey_x,
            m_Ey_z,
            kappa_y_h,
            kappa_x_h,
            ay_h,
            ax_h,
            by_h,
            bx_h,
            rdy,
            rdx,
            stencil,
        )
        dHx, dHz, dm_Ey_x, dm_Ey_z = update_H_born(
            cq,
            dHx,
            dHz,
            dEy,
            dm_Ey_x,
            dm_Ey_z,
            kappa_y_h,
            kappa_x_h,
            ay_h,
            ax_h,
            by_h,
            bx_h,
            rdy,
            rdx,
            stencil,
        )

        Ey, dEy, m_Hx_z, m_Hz_x, dm_Hx_z, dm_Hz_x = update_E_born(
            ca,
            cb,
            dca_b,
            dcb_b,
            Hx,
            Hz,
            Ey,
            dHx,
            dHz,
            dEy,
            m_Hx_z,
            m_Hz_x,
            dm_Hx_z,
            dm_Hz_x,
            kappa_y,
            kappa_x,
            ay,
            ax,
            by,
            bx,
            rdy,
            rdx,
            stencil,
        )

        if (
            source_amplitude is not None
            and source_amplitude.numel() > 0
            and n_sources > 0
        ):
            src_amp = source_amplitude[:, :, step]
            scaled_src = cb_at_src * src_amp * source_coeff
            Ey = (
                Ey.reshape(n_shots, flat_model_shape)
                .scatter_add(1, sources_i, scaled_src)
                .reshape(size_with_batch)
            )

            if linearize_source:
                dscaled_src = dcb_at_src * src_amp * source_coeff
                dEy = (
                    dEy.reshape(n_shots, flat_model_shape)
                    .scatter_add(1, sources_i, dscaled_src)
                    .reshape(size_with_batch)
                )

        if n_bg_receivers > 0:
            bg_receiver_samples.append(
                Ey.reshape(n_shots, flat_model_shape).gather(1, bg_receivers_i)
            )

        if n_receivers > 0:
            receiver_samples.append(
                dEy.reshape(n_shots, flat_model_shape).gather(1, receivers_i)
            )

    if n_bg_receivers > 0:
        bg_receiver_amplitudes = torch.stack(bg_receiver_samples, dim=0)
    else:
        bg_receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

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
        dEy[s],
        dHx[s],
        dHz[s],
        dm_Ey_x[s],
        dm_Ey_z[s],
        dm_Hx_z[s],
        dm_Hz_x[s],
        bg_receiver_amplitudes,
        receiver_amplitudes,
    )


def _update_background_H(
    cq: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    Ey: torch.Tensor,
    m_Ey_x: torch.Tensor,
    m_Ey_z: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x_h: torch.Tensor,
    ay_h: torch.Tensor,
    ax_h: torch.Tensor,
    by_h: torch.Tensor,
    bx_h: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    stencil: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dEy_dz = staggered.diffyh1(Ey, stencil, rdy)
    dEy_dx = staggered.diffxh1(Ey, stencil, rdx)

    m_Ey_z = by_h * m_Ey_z + ay_h * dEy_dz
    m_Ey_x = bx_h * m_Ey_x + ax_h * dEy_dx

    dEy_dz_pml = dEy_dz / kappa_y_h + m_Ey_z
    dEy_dx_pml = dEy_dx / kappa_x_h + m_Ey_x

    Hx = Hx - cq * dEy_dz_pml
    Hz = Hz + cq * dEy_dx_pml
    return Hx, Hz, m_Ey_x, m_Ey_z


__all__ = [
    "borntm_python",
    "update_E_born",
    "update_H_born",
]
