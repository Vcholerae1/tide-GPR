from collections.abc import Sequence

import torch

from .. import staggered
from ..grid_utils import _normalize_grid_spacing_3d, _normalize_pml_width_3d
from ..padding import create_or_pad, zero_interior
from ..utils import (
    C0,
    compile_material_coefficients,
    linearize_material_coefficients,
)
from .validation_internal import _validate_location_bounds


def _select_component(
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


def _init_wavefield_3d(
    field_0: torch.Tensor | None,
    *,
    n_shots: int,
    size_with_batch: tuple[int, int, int, int],
    fd_pad_list: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
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


def update_H_born_3d(
    cq: torch.Tensor,
    dEx: torch.Tensor,
    dEy: torch.Tensor,
    dEz: torch.Tensor,
    dHx: torch.Tensor,
    dHy: torch.Tensor,
    dHz: torch.Tensor,
    dm_ey_z: torch.Tensor,
    dm_ez_y: torch.Tensor,
    dm_ez_x: torch.Tensor,
    dm_ex_z: torch.Tensor,
    dm_ex_y: torch.Tensor,
    dm_ey_x: torch.Tensor,
    kz_h: torch.Tensor,
    ky_h: torch.Tensor,
    kx_h: torch.Tensor,
    az_h: torch.Tensor,
    ay_h: torch.Tensor,
    ax_h: torch.Tensor,
    bz_h: torch.Tensor,
    by_h: torch.Tensor,
    bx_h: torch.Tensor,
    rdz: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    stencil: int,
) -> tuple[torch.Tensor, ...]:
    d_dEy_dz = staggered.diffzh1(dEy, stencil, rdz)
    d_dEz_dy = staggered.diffyh1(dEz, stencil, rdy)
    d_dEz_dx = staggered.diffxh1(dEz, stencil, rdx)
    d_dEx_dz = staggered.diffzh1(dEx, stencil, rdz)
    d_dEx_dy = staggered.diffyh1(dEx, stencil, rdy)
    d_dEy_dx = staggered.diffxh1(dEy, stencil, rdx)

    dm_ey_z = bz_h * dm_ey_z + az_h * d_dEy_dz
    dm_ez_y = by_h * dm_ez_y + ay_h * d_dEz_dy
    dm_ez_x = bx_h * dm_ez_x + ax_h * d_dEz_dx
    dm_ex_z = bz_h * dm_ex_z + az_h * d_dEx_dz
    dm_ex_y = by_h * dm_ex_y + ay_h * d_dEx_dy
    dm_ey_x = bx_h * dm_ey_x + ax_h * d_dEy_dx

    d_dEy_dz_pml = d_dEy_dz / kz_h + dm_ey_z
    d_dEz_dy_pml = d_dEz_dy / ky_h + dm_ez_y
    d_dEz_dx_pml = d_dEz_dx / kx_h + dm_ez_x
    d_dEx_dz_pml = d_dEx_dz / kz_h + dm_ex_z
    d_dEx_dy_pml = d_dEx_dy / ky_h + dm_ex_y
    d_dEy_dx_pml = d_dEy_dx / kx_h + dm_ey_x

    dHx = dHx - cq * (d_dEy_dz_pml - d_dEz_dy_pml)
    dHy = dHy - cq * (d_dEz_dx_pml - d_dEx_dz_pml)
    dHz = dHz - cq * (d_dEx_dy_pml - d_dEy_dx_pml)

    return (
        dHx,
        dHy,
        dHz,
        dm_ey_z,
        dm_ez_y,
        dm_ez_x,
        dm_ex_z,
        dm_ex_y,
        dm_ey_x,
    )


def update_E_born_3d(
    ca: torch.Tensor,
    cb: torch.Tensor,
    dca: torch.Tensor,
    dcb: torch.Tensor,
    Hx: torch.Tensor,
    Hy: torch.Tensor,
    Hz: torch.Tensor,
    Ex: torch.Tensor,
    Ey: torch.Tensor,
    Ez: torch.Tensor,
    dHx: torch.Tensor,
    dHy: torch.Tensor,
    dHz: torch.Tensor,
    dEx: torch.Tensor,
    dEy: torch.Tensor,
    dEz: torch.Tensor,
    m_hz_y: torch.Tensor,
    m_hy_z: torch.Tensor,
    m_hx_z: torch.Tensor,
    m_hz_x: torch.Tensor,
    m_hy_x: torch.Tensor,
    m_hx_y: torch.Tensor,
    dm_hz_y: torch.Tensor,
    dm_hy_z: torch.Tensor,
    dm_hx_z: torch.Tensor,
    dm_hz_x: torch.Tensor,
    dm_hy_x: torch.Tensor,
    dm_hx_y: torch.Tensor,
    kz: torch.Tensor,
    ky: torch.Tensor,
    kx: torch.Tensor,
    az: torch.Tensor,
    ay: torch.Tensor,
    ax: torch.Tensor,
    bz: torch.Tensor,
    by: torch.Tensor,
    bx: torch.Tensor,
    rdz: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    stencil: int,
) -> tuple[torch.Tensor, ...]:
    dHy_dz = staggered.diffz1(Hy, stencil, rdz)
    dHz_dy = staggered.diffy1(Hz, stencil, rdy)
    dHz_dx = staggered.diffx1(Hz, stencil, rdx)
    dHx_dz = staggered.diffz1(Hx, stencil, rdz)
    dHx_dy = staggered.diffy1(Hx, stencil, rdy)
    dHy_dx = staggered.diffx1(Hy, stencil, rdx)

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

    curl_x = dHy_dz_pml - dHz_dy_pml
    curl_y = dHz_dx_pml - dHx_dz_pml
    curl_z = dHx_dy_pml - dHy_dx_pml

    ddHy_dz = staggered.diffz1(dHy, stencil, rdz)
    ddHz_dy = staggered.diffy1(dHz, stencil, rdy)
    ddHz_dx = staggered.diffx1(dHz, stencil, rdx)
    ddHx_dz = staggered.diffz1(dHx, stencil, rdz)
    ddHx_dy = staggered.diffy1(dHx, stencil, rdy)
    ddHy_dx = staggered.diffx1(dHy, stencil, rdx)

    dm_hy_z = bz * dm_hy_z + az * ddHy_dz
    dm_hz_y = by * dm_hz_y + ay * ddHz_dy
    dm_hz_x = bx * dm_hz_x + ax * ddHz_dx
    dm_hx_z = bz * dm_hx_z + az * ddHx_dz
    dm_hx_y = by * dm_hx_y + ay * ddHx_dy
    dm_hy_x = bx * dm_hy_x + ax * ddHy_dx

    ddHy_dz_pml = ddHy_dz / kz + dm_hy_z
    ddHz_dy_pml = ddHz_dy / ky + dm_hz_y
    ddHz_dx_pml = ddHz_dx / kx + dm_hz_x
    ddHx_dz_pml = ddHx_dz / kz + dm_hx_z
    ddHx_dy_pml = ddHx_dy / ky + dm_hx_y
    ddHy_dx_pml = ddHy_dx / kx + dm_hy_x

    dcurl_x = ddHy_dz_pml - ddHz_dy_pml
    dcurl_y = ddHz_dx_pml - ddHx_dz_pml
    dcurl_z = ddHx_dy_pml - ddHy_dx_pml

    ex_old = Ex
    ey_old = Ey
    ez_old = Ez

    dEx = ca * dEx + cb * dcurl_x + dca * ex_old + dcb * curl_x
    dEy = ca * dEy + cb * dcurl_y + dca * ey_old + dcb * curl_y
    dEz = ca * dEz + cb * dcurl_z + dca * ez_old + dcb * curl_z

    Ex = ca * Ex + cb * curl_x
    Ey = ca * Ey + cb * curl_y
    Ez = ca * Ez + cb * curl_z

    return (
        Ex,
        Ey,
        Ez,
        dEx,
        dEy,
        dEz,
        m_hz_y,
        m_hy_z,
        m_hx_z,
        m_hz_x,
        m_hy_x,
        m_hx_y,
        dm_hz_y,
        dm_hy_z,
        dm_hx_z,
        dm_hz_x,
        dm_hy_x,
        dm_hx_y,
    )


def born3d_python(
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
    *,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    Ex_0: torch.Tensor | None = None,
    Ey_0: torch.Tensor | None = None,
    Ez_0: torch.Tensor | None = None,
    Hx_0: torch.Tensor | None = None,
    Hy_0: torch.Tensor | None = None,
    Hz_0: torch.Tensor | None = None,
    m_hz_y_0: torch.Tensor | None = None,
    m_hy_z_0: torch.Tensor | None = None,
    m_hx_z_0: torch.Tensor | None = None,
    m_hz_x_0: torch.Tensor | None = None,
    m_hy_x_0: torch.Tensor | None = None,
    m_hx_y_0: torch.Tensor | None = None,
    m_ey_z_0: torch.Tensor | None = None,
    m_ez_y_0: torch.Tensor | None = None,
    m_ez_x_0: torch.Tensor | None = None,
    m_ex_z_0: torch.Tensor | None = None,
    m_ex_y_0: torch.Tensor | None = None,
    m_ey_x_0: torch.Tensor | None = None,
    dEx_0: torch.Tensor | None = None,
    dEy_0: torch.Tensor | None = None,
    dEz_0: torch.Tensor | None = None,
    dHx_0: torch.Tensor | None = None,
    dHy_0: torch.Tensor | None = None,
    dHz_0: torch.Tensor | None = None,
    dm_hz_y_0: torch.Tensor | None = None,
    dm_hy_z_0: torch.Tensor | None = None,
    dm_hx_z_0: torch.Tensor | None = None,
    dm_hz_x_0: torch.Tensor | None = None,
    dm_hy_x_0: torch.Tensor | None = None,
    dm_hx_y_0: torch.Tensor | None = None,
    dm_ey_z_0: torch.Tensor | None = None,
    dm_ez_y_0: torch.Tensor | None = None,
    dm_ez_x_0: torch.Tensor | None = None,
    dm_ex_z_0: torch.Tensor | None = None,
    dm_ex_y_0: torch.Tensor | None = None,
    dm_ey_x_0: torch.Tensor | None = None,
    nt: int | None = None,
    parameterization: str = "epsilon_sigma",
    linearize_source: bool = True,
    source_component: str = "ey",
    receiver_component: str = "ey",
) -> tuple[torch.Tensor, ...]:
    if epsilon.ndim != 3:
        raise RuntimeError("epsilon must be 3D for born3d_python")
    if sigma.shape != epsilon.shape or mu.shape != epsilon.shape:
        raise RuntimeError("sigma and mu must have the same shape as epsilon")
    if parameterization not in {"epsilon_sigma", "ca_cb"}:
        raise ValueError(
            "parameterization must be 'epsilon_sigma' or 'ca_cb', "
            f"got {parameterization!r}."
        )

    device = epsilon.device
    dtype = epsilon.dtype
    model_nz, model_ny, model_nx = epsilon.shape

    _validate_location_bounds(
        source_location,
        shape=(model_nz, model_ny, model_nx),
        name="Source location",
        check_lower_bound=False,
    )
    _validate_location_bounds(
        receiver_location,
        shape=(model_nz, model_ny, model_nx),
        name="Receiver location",
        check_lower_bound=False,
    )
    _validate_location_bounds(
        bg_receiver_location,
        shape=(model_nz, model_ny, model_nx),
        name="Background receiver location",
        check_lower_bound=False,
    )

    grid_spacing_list = _normalize_grid_spacing_3d(grid_spacing)
    dz, dy, dx = grid_spacing_list
    pml_width_list = _normalize_pml_width_3d(pml_width)

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

    material = compile_material_coefficients(
        epsilon_padded,
        sigma_padded,
        mu_padded,
        dt,
        validate_inputs=True,
    )
    ca = material["ca"][None, :, :, :]
    cb = material["cb"][None, :, :, :]
    cq = material["cq"][None, :, :, :]

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

    dca_b = dca_padded[None, :, :, :]
    dcb_b = dcb_padded[None, :, :, :]

    size_with_batch = (n_shots, padded_nz, padded_ny, padded_nx)
    Ex = _init_wavefield_3d(
        Ex_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    Ey = _init_wavefield_3d(
        Ey_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    Ez = _init_wavefield_3d(
        Ez_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    Hx = _init_wavefield_3d(
        Hx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    Hy = _init_wavefield_3d(
        Hy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    Hz = _init_wavefield_3d(
        Hz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )

    m_hz_y = _init_wavefield_3d(
        m_hz_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_hy_z = _init_wavefield_3d(
        m_hy_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_hx_z = _init_wavefield_3d(
        m_hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_hz_x = _init_wavefield_3d(
        m_hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_hy_x = _init_wavefield_3d(
        m_hy_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_hx_y = _init_wavefield_3d(
        m_hx_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_ey_z = _init_wavefield_3d(
        m_ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_ez_y = _init_wavefield_3d(
        m_ez_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_ez_x = _init_wavefield_3d(
        m_ez_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_ex_z = _init_wavefield_3d(
        m_ex_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_ex_y = _init_wavefield_3d(
        m_ex_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    m_ey_x = _init_wavefield_3d(
        m_ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )

    dEx = _init_wavefield_3d(
        dEx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dEy = _init_wavefield_3d(
        dEy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dEz = _init_wavefield_3d(
        dEz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dHx = _init_wavefield_3d(
        dHx_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dHy = _init_wavefield_3d(
        dHy_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dHz = _init_wavefield_3d(
        dHz_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_hz_y = _init_wavefield_3d(
        dm_hz_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_hy_z = _init_wavefield_3d(
        dm_hy_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_hx_z = _init_wavefield_3d(
        dm_hx_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_hz_x = _init_wavefield_3d(
        dm_hz_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_hy_x = _init_wavefield_3d(
        dm_hy_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_hx_y = _init_wavefield_3d(
        dm_hx_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_ey_z = _init_wavefield_3d(
        dm_ey_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_ez_y = _init_wavefield_3d(
        dm_ez_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_ez_x = _init_wavefield_3d(
        dm_ez_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_ex_z = _init_wavefield_3d(
        dm_ex_z_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_ex_y = _init_wavefield_3d(
        dm_ex_y_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
    )
    dm_ey_x = _init_wavefield_3d(
        dm_ey_x_0,
        n_shots=n_shots,
        size_with_batch=size_with_batch,
        fd_pad_list=fd_pad_list,
        device=device,
        dtype=dtype,
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

    rdz = torch.tensor(1.0 / dz, device=device, dtype=dtype)
    rdy = torch.tensor(1.0 / dy, device=device, dtype=dtype)
    rdx = torch.tensor(1.0 / dx, device=device, dtype=dtype)

    flat_model_shape = padded_nz * padded_ny * padded_nx
    if source_location is not None and source_location.numel() > 0:
        source_z = source_location[..., 0] + total_pad[0]
        source_y = source_location[..., 1] + total_pad[2]
        source_x = source_location[..., 2] + total_pad[4]
        sources_i = ((source_z * padded_ny + source_y) * padded_nx + source_x).long()
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
        n_receivers = int(receiver_location.shape[1])
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    if bg_receiver_location is not None and bg_receiver_location.numel() > 0:
        bg_receiver_z = bg_receiver_location[..., 0] + total_pad[0]
        bg_receiver_y = bg_receiver_location[..., 1] + total_pad[2]
        bg_receiver_x = bg_receiver_location[..., 2] + total_pad[4]
        bg_receivers_i = (
            (bg_receiver_z * padded_ny + bg_receiver_y) * padded_nx + bg_receiver_x
        ).long()
        n_bg_receivers = int(bg_receiver_location.shape[1])
    else:
        bg_receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_bg_receivers = 0

    source_coeff = -1.0 / (dx * dy * dz)
    if n_sources > 0 and source_amplitude is not None and source_amplitude.numel() > 0:
        cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
        cb_at_src = cb_flat.gather(1, sources_i)
        if linearize_source:
            dcb_flat = dcb_b.reshape(1, flat_model_shape).expand(n_shots, -1)
            dcb_at_src = dcb_flat.gather(1, sources_i)
        else:
            dcb_at_src = None
    else:
        cb_at_src = torch.empty(0, device=device, dtype=dtype)
        dcb_at_src = None

    bg_receiver_samples: list[torch.Tensor] = []
    receiver_samples: list[torch.Tensor] = []

    for step in range(nt_steps):
        dEy_dz = staggered.diffzh1(Ey, stencil, rdz)
        dEz_dy = staggered.diffyh1(Ez, stencil, rdy)
        dEz_dx = staggered.diffxh1(Ez, stencil, rdx)
        dEx_dz = staggered.diffzh1(Ex, stencil, rdz)
        dEx_dy = staggered.diffyh1(Ex, stencil, rdy)
        dEy_dx = staggered.diffxh1(Ey, stencil, rdx)

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

        (
            dHx,
            dHy,
            dHz,
            dm_ey_z,
            dm_ez_y,
            dm_ez_x,
            dm_ex_z,
            dm_ex_y,
            dm_ey_x,
        ) = update_H_born_3d(
            cq,
            dEx,
            dEy,
            dEz,
            dHx,
            dHy,
            dHz,
            dm_ey_z,
            dm_ez_y,
            dm_ez_x,
            dm_ex_z,
            dm_ex_y,
            dm_ey_x,
            kz_h,
            ky_h,
            kx_h,
            az_h,
            ay_h,
            ax_h,
            bz_h,
            by_h,
            bx_h,
            rdz,
            rdy,
            rdx,
            stencil,
        )

        (
            Ex,
            Ey,
            Ez,
            dEx,
            dEy,
            dEz,
            m_hz_y,
            m_hy_z,
            m_hx_z,
            m_hz_x,
            m_hy_x,
            m_hx_y,
            dm_hz_y,
            dm_hy_z,
            dm_hx_z,
            dm_hz_x,
            dm_hy_x,
            dm_hx_y,
        ) = update_E_born_3d(
            ca,
            cb,
            dca_b,
            dcb_b,
            Hx,
            Hy,
            Hz,
            Ex,
            Ey,
            Ez,
            dHx,
            dHy,
            dHz,
            dEx,
            dEy,
            dEz,
            m_hz_y,
            m_hy_z,
            m_hx_z,
            m_hz_x,
            m_hy_x,
            m_hx_y,
            dm_hz_y,
            dm_hy_z,
            dm_hx_z,
            dm_hz_x,
            dm_hy_x,
            dm_hx_y,
            kz,
            ky,
            kx,
            az,
            ay,
            ax,
            bz,
            by,
            bx,
            rdz,
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
            scaled_dsrc = (
                dcb_at_src * src_amp * source_coeff
                if linearize_source and dcb_at_src is not None
                else None
            )

            if source_component == "ex":
                Ex = _inject_component(
                    Ex, flat_model_shape, sources_i, scaled_src, size_with_batch
                )
                if scaled_dsrc is not None:
                    dEx = _inject_component(
                        dEx,
                        flat_model_shape,
                        sources_i,
                        scaled_dsrc,
                        size_with_batch,
                    )
            elif source_component == "ez":
                Ez = _inject_component(
                    Ez, flat_model_shape, sources_i, scaled_src, size_with_batch
                )
                if scaled_dsrc is not None:
                    dEz = _inject_component(
                        dEz,
                        flat_model_shape,
                        sources_i,
                        scaled_dsrc,
                        size_with_batch,
                    )
            else:
                Ey = _inject_component(
                    Ey, flat_model_shape, sources_i, scaled_src, size_with_batch
                )
                if scaled_dsrc is not None:
                    dEy = _inject_component(
                        dEy,
                        flat_model_shape,
                        sources_i,
                        scaled_dsrc,
                        size_with_batch,
                    )

        if n_bg_receivers > 0:
            bg_rec_field = _select_component(receiver_component, Ex, Ey, Ez)
            bg_receiver_samples.append(
                bg_rec_field.reshape(n_shots, flat_model_shape).gather(
                    1, bg_receivers_i
                )
            )

        if n_receivers > 0:
            rec_field = _select_component(receiver_component, dEx, dEy, dEz)
            receiver_samples.append(
                rec_field.reshape(n_shots, flat_model_shape).gather(1, receivers_i)
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
        dEx[s],
        dEy[s],
        dEz[s],
        dHx[s],
        dHy[s],
        dHz[s],
        dm_hz_y[s],
        dm_hy_z[s],
        dm_hx_z[s],
        dm_hz_x[s],
        dm_hy_x[s],
        dm_hx_y[s],
        dm_ey_z[s],
        dm_ez_y[s],
        dm_ez_x[s],
        dm_ex_z[s],
        dm_ex_y[s],
        dm_ey_x[s],
        bg_receiver_amplitudes,
        receiver_amplitudes,
    )


__all__ = ["born3d_python"]
