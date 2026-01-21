from typing import Tuple

import torch

from . import utils


def set_pml_profiles(
    pml_width: list[int],
    accuracy: int,
    fd_pad: list[int],
    dt: float,
    grid_spacing: list[float],
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    ny: int,
    nx: int,
) -> list[torch.Tensor]:
    """Sets up PML profiles for a staggered grid.

    Args:
        pml_width: A list of integers specifying the width of the PML
            on each side (top, bottom, left, right).
        accuracy: The finite-difference accuracy order.
        fd_pad: A list of integers specifying the padding for finite-difference.
        dt: The time step.
        grid_spacing: A list of floats specifying the grid spacing in
            y and x directions.
        max_vel: The maximum velocity in the model.
        dtype: The data type of the tensors (e.g., torch.float32).
        device: The device on which the tensors will be (e.g., 'cuda', 'cpu').
        pml_freq: The PML frequency.
        ny: The number of grid points in the y direction.
        nx: The number of grid points in the x direction.

    Returns:
        A list containing:
        - a, b profiles: [ay, ayh, ax, axh, by, byh, bx, bxh]
        - k profiles: [ky, kyh, kx, kxh]
        Total 12 tensors.

    """
    pml_start: list[float] = [
        fd_pad[0] + pml_width[0],
        ny - 1 - fd_pad[1] - pml_width[1],
        fd_pad[2] + pml_width[2],
        nx - 1 - fd_pad[3] - pml_width[3],
    ]
    max_pml = max(
        [
            pml_width[0] * grid_spacing[0],
            pml_width[1] * grid_spacing[0],
            pml_width[2] * grid_spacing[1],
            pml_width[3] * grid_spacing[1],
        ],
    )

    # Integer grid PML profiles
    ay, by, ky = utils.setup_pml(
        pml_width[:2],
        pml_start[:2],
        max_pml,
        dt,
        ny,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[0],
    )
    ax, bx, kx = utils.setup_pml(
        pml_width[2:],
        pml_start[2:],
        max_pml,
        dt,
        nx,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[1],
    )

    # Half grid PML profiles
    ayh, byh, kyh = utils.setup_pml_half(
        pml_width[:2],
        pml_start[:2],
        max_pml,
        dt,
        ny,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[0],
    )
    axh, bxh, kxh = utils.setup_pml_half(
        pml_width[2:],
        pml_start[2:],
        max_pml,
        dt,
        nx,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=grid_spacing[1],
    )

    # Reshape for broadcasting: [batch, ny, nx]
    ay = ay[None, :, None]
    ayh = ayh[None, :, None]
    ax = ax[None, None, :]
    axh = axh[None, None, :]
    by = by[None, :, None]
    byh = byh[None, :, None]
    bx = bx[None, None, :]
    bxh = bxh[None, None, :]

    ky = ky[None, :, None]
    kyh = kyh[None, :, None]
    kx = kx[None, None, :]
    kxh = kxh[None, None, :]

    return [ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh]


def setup_pml_profiles_1d(
    n: int,
    pml_width0: int,
    pml_width1: int,
    sigma_max: float,
    dt: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Create 1D CPML profiles (a, b, k) for integer and half-grid points."""
    eps = 1e-9
    n_power = 2

    if pml_width0 == 0 and pml_width1 == 0:
        zeros = torch.zeros(n, device=device, dtype=dtype)
        ones = torch.ones(n, device=device, dtype=dtype)
        return zeros, zeros, zeros, zeros, ones, ones

    def _profiles(start: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.arange(n, device=device, dtype=dtype) + start
        left_start = float(pml_width0)
        right_start = float(n - 1 - pml_width1)

        if pml_width0 == 0:
            frac_left = torch.zeros_like(x)
        else:
            frac_left = (left_start - x) / float(pml_width0)
        if pml_width1 == 0:
            frac_right = torch.zeros_like(x)
        else:
            frac_right = (x - right_start) / float(pml_width1)

        pml_frac = torch.clamp(torch.maximum(frac_left, frac_right), 0.0, 1.0)
        sigma = sigma_max * pml_frac.pow(n_power)
        kappa = torch.ones_like(sigma)

        sigma_alpha = sigma
        b = torch.exp(-sigma_alpha * abs(dt))
        denom = sigma_alpha + eps
        a = torch.where(
            sigma_alpha > 0.0, sigma * (b - 1.0) / denom, torch.zeros_like(b)
        )
        return a, b, kappa

    ay, by, ky = _profiles(0.0)
    ayh, byh, kyh = _profiles(0.5)
    return ay, ayh, by, byh, ky, kyh


def set_pml_profiles_3d(
    pml_width: list[int],
    accuracy: int,
    fd_pad: list[int],
    dt: float,
    grid_spacing: list[float],
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    nz: int,
    ny: int,
    nx: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Sets up 3D PML profiles for a staggered grid.

    Args:
        pml_width: Widths [z0, z1, y0, y1, x0, x1].
        accuracy: Finite-difference accuracy order (unused, kept for API parity).
        fd_pad: FD padding [z0, z1, y0, y1, x0, x1].
        dt: Time step.
        grid_spacing: Grid spacing [dz, dy, dx].
        max_vel: Maximum velocity (unused in EM formulation, kept for API parity).
        dtype: Tensor dtype.
        device: Tensor device.
        pml_freq: PML frequency.
        nz, ny, nx: Padded grid sizes.

    Returns:
        - PML a/b profiles: [az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh]
        - PML kappa profiles: [kz, kzh, ky, kyh, kx, kxh]
    """
    _ = accuracy
    dz, dy, dx = grid_spacing

    pml_start: list[float] = [
        fd_pad[0] + pml_width[0],
        nz - 1 - fd_pad[1] - pml_width[1],
        fd_pad[2] + pml_width[2],
        ny - 1 - fd_pad[3] - pml_width[3],
        fd_pad[4] + pml_width[4],
        nx - 1 - fd_pad[5] - pml_width[5],
    ]

    max_pml = max(
        [
            pml_width[0] * dz,
            pml_width[1] * dz,
            pml_width[2] * dy,
            pml_width[3] * dy,
            pml_width[4] * dx,
            pml_width[5] * dx,
        ]
    )

    az, bz, kz = utils.setup_pml(
        pml_width[:2],
        pml_start[:2],
        max_pml,
        dt,
        nz,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=dz,
    )
    ay, by, ky = utils.setup_pml(
        pml_width[2:4],
        pml_start[2:4],
        max_pml,
        dt,
        ny,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=dy,
    )
    ax, bx, kx = utils.setup_pml(
        pml_width[4:],
        pml_start[4:],
        max_pml,
        dt,
        nx,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=dx,
    )

    azh, bzh, kzh = utils.setup_pml_half(
        pml_width[:2],
        pml_start[:2],
        max_pml,
        dt,
        nz,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=dz,
    )
    ayh, byh, kyh = utils.setup_pml_half(
        pml_width[2:4],
        pml_start[2:4],
        max_pml,
        dt,
        ny,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=dy,
    )
    axh, bxh, kxh = utils.setup_pml_half(
        pml_width[4:],
        pml_start[4:],
        max_pml,
        dt,
        nx,
        max_vel,
        dtype,
        device,
        pml_freq,
        start=0.0,
        grid_spacing=dx,
    )

    az = az[None, :, None, None]
    azh = azh[None, :, None, None]
    bz = bz[None, :, None, None]
    bzh = bzh[None, :, None, None]
    kz = kz[None, :, None, None]
    kzh = kzh[None, :, None, None]

    ay = ay[None, None, :, None]
    ayh = ayh[None, None, :, None]
    by = by[None, None, :, None]
    byh = byh[None, None, :, None]
    ky = ky[None, None, :, None]
    kyh = kyh[None, None, :, None]

    ax = ax[None, None, None, :]
    axh = axh[None, None, None, :]
    bx = bx[None, None, None, :]
    bxh = bxh[None, None, None, :]
    kx = kx[None, None, None, :]
    kxh = kxh[None, None, None, :]

    return (
        [az, azh, ay, ayh, ax, axh, bz, bzh, by, byh, bx, bxh],
        [kz, kzh, ky, kyh, kx, kxh],
    )


def diffy1(a: torch.Tensor, stencil: int, rdy: torch.Tensor) -> torch.Tensor:
    """Calculates the first y derivative at integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad(
            (a[..., 1:, :] - a[..., :-1, :]) * rdy, (0, 0, 1, 0)
        )
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1, :] - a[..., 1:-2, :])
                + -1 / 24 * (a[..., 3:, :] - a[..., :-3, :])
            )
            * rdy,
            (0, 0, 2, 1),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2, :] - a[..., 2:-3, :])
                + -25 / 384 * (a[..., 4:-1, :] - a[..., 1:-4, :])
                + 3 / 640 * (a[..., 5:, :] - a[..., :-5, :])
            )
            * rdy,
            (0, 0, 3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3, :] - a[..., 3:-4, :])
            + -245 / 3072 * (a[..., 5:-2, :] - a[..., 2:-5, :])
            + 49 / 5120 * (a[..., 6:-1, :] - a[..., 1:-6, :])
            + -5 / 7168 * (a[..., 7:, :] - a[..., :-7, :])
        )
        * rdy,
        (0, 0, 4, 3),
    )


def diffx1(a: torch.Tensor, stencil: int, rdx: torch.Tensor) -> torch.Tensor:
    """Calculates the first x derivative at integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad((a[..., 1:] - a[..., :-1]) * rdx, (1, 0))
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1] - a[..., 1:-2])
                + -1 / 24 * (a[..., 3:] - a[..., :-3])
            )
            * rdx,
            (2, 1),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2] - a[..., 2:-3])
                + -25 / 384 * (a[..., 4:-1] - a[..., 1:-4])
                + 3 / 640 * (a[..., 5:] - a[..., :-5])
            )
            * rdx,
            (3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3] - a[..., 3:-4])
            + -245 / 3072 * (a[..., 5:-2] - a[..., 2:-5])
            + 49 / 5120 * (a[..., 6:-1] - a[..., 1:-6])
            + -5 / 7168 * (a[..., 7:] - a[..., :-7])
        )
        * rdx,
        (4, 3),
    )


def diffz1(a: torch.Tensor, stencil: int, rdz: torch.Tensor) -> torch.Tensor:
    """Calculates the first z derivative at integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad(
            (a[..., 1:, :, :] - a[..., :-1, :, :]) * rdz,
            (0, 0, 0, 0, 1, 0),
        )
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 2:-1, :, :] - a[..., 1:-2, :, :])
                + -1 / 24 * (a[..., 3:, :, :] - a[..., :-3, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 1),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 3:-2, :, :] - a[..., 2:-3, :, :])
                + -25 / 384 * (a[..., 4:-1, :, :] - a[..., 1:-4, :, :])
                + 3 / 640 * (a[..., 5:, :, :] - a[..., :-5, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 2),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 4:-3, :, :] - a[..., 3:-4, :, :])
            + -245 / 3072 * (a[..., 5:-2, :, :] - a[..., 2:-5, :, :])
            + 49 / 5120 * (a[..., 6:-1, :, :] - a[..., 1:-6, :, :])
            + -5 / 7168 * (a[..., 7:, :, :] - a[..., :-7, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 3),
    )


def diffyh1(a: torch.Tensor, stencil: int, rdy: torch.Tensor) -> torch.Tensor:
    """Calculates the first y derivative at half integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :] - a[..., 1:-1, :]) * rdy, (0, 0, 1, 1)
        )
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :] - a[..., 2:-2, :])
                + -1 / 24 * (a[..., 4:, :] - a[..., 1:-3, :])
            )
            * rdy,
            (0, 0, 2, 2),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :] - a[..., 3:-3, :])
                + -25 / 384 * (a[..., 5:-1, :] - a[..., 2:-4, :])
                + 3 / 640 * (a[..., 6:, :] - a[..., 1:-5, :])
            )
            * rdy,
            (0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :] - a[..., 4:-4, :])
            + -245 / 3072 * (a[..., 6:-2, :] - a[..., 3:-5, :])
            + 49 / 5120 * (a[..., 7:-1, :] - a[..., 2:-6, :])
            + -5 / 7168 * (a[..., 8:, :] - a[..., 1:-7, :])
        )
        * rdy,
        (0, 0, 4, 4),
    )


def diffzh1(a: torch.Tensor, stencil: int, rdz: torch.Tensor) -> torch.Tensor:
    """Calculates the first z derivative at half integer grid points.

    For a tensor with shape [..., nz, ny, nx], the derivative is taken along
    the z dimension at half-grid locations.
    """
    if stencil == 2:
        return torch.nn.functional.pad(
            (a[..., 2:, :, :] - a[..., 1:-1, :, :]) * rdz, (0, 0, 0, 0, 1, 1)
        )
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1, :, :] - a[..., 2:-2, :, :])
                + -1 / 24 * (a[..., 4:, :, :] - a[..., 1:-3, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 2, 2),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2, :, :] - a[..., 3:-3, :, :])
                + -25 / 384 * (a[..., 5:-1, :, :] - a[..., 2:-4, :, :])
                + 3 / 640 * (a[..., 6:, :, :] - a[..., 1:-5, :, :])
            )
            * rdz,
            (0, 0, 0, 0, 3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3, :, :] - a[..., 4:-4, :, :])
            + -245 / 3072 * (a[..., 6:-2, :, :] - a[..., 3:-5, :, :])
            + 49 / 5120 * (a[..., 7:-1, :, :] - a[..., 2:-6, :, :])
            + -5 / 7168 * (a[..., 8:, :, :] - a[..., 1:-7, :, :])
        )
        * rdz,
        (0, 0, 0, 0, 4, 4),
    )


def diffxh1(a: torch.Tensor, stencil: int, rdx: torch.Tensor) -> torch.Tensor:
    """Calculates the first x derivative at half integer grid points."""
    if stencil == 2:
        return torch.nn.functional.pad((a[..., 2:] - a[..., 1:-1]) * rdx, (1, 1))
    if stencil == 4:
        return torch.nn.functional.pad(
            (
                9 / 8 * (a[..., 3:-1] - a[..., 2:-2])
                + -1 / 24 * (a[..., 4:] - a[..., 1:-3])
            )
            * rdx,
            (2, 2),
        )
    if stencil == 6:
        return torch.nn.functional.pad(
            (
                75 / 64 * (a[..., 4:-2] - a[..., 3:-3])
                + -25 / 384 * (a[..., 5:-1] - a[..., 2:-4])
                + 3 / 640 * (a[..., 6:] - a[..., 1:-5])
            )
            * rdx,
            (3, 3),
        )
    return torch.nn.functional.pad(
        (
            1225 / 1024 * (a[..., 5:-3] - a[..., 4:-4])
            + -245 / 3072 * (a[..., 6:-2] - a[..., 3:-5])
            + 49 / 5120 * (a[..., 7:-1] - a[..., 2:-6])
            + -5 / 7168 * (a[..., 8:] - a[..., 1:-7])
        )
        * rdx,
        (4, 4),
    )
