import math
from typing import Sequence

import torch

# Physical constants
EP0 = 8.8541878128e-12  # vacuum permittivity
MU0 = 1.2566370614359173e-06  # vacuum permeability
C0 = 1.0 / math.sqrt(EP0 * MU0)  # speed of light in vacuum


def prepare_parameters(
    epsilon_r: torch.Tensor, sigma: torch.Tensor, mu_r: torch.Tensor, dt: float
) -> tuple:
    """Prepare update coefficients for Maxwell equations.

    Args:
        epsilon_r: Relative permittivity.
        sigma: Conductivity (S/m).
        mu_r: Relative permeability.
        dt: Time step (s).

    Returns:
        Tuple of (ca, cb, cq) coefficients.
    """
    # Convert to absolute values
    epsilon = epsilon_r * EP0
    mu = mu_r * MU0

    # Ca and Cb for E-field update: E^{n+1} = Ca*E^n + Cb*(curl H)
    # Ca = (1 - sigma*dt/(2*epsilon)) / (1 + sigma*dt/(2*epsilon))
    # Cb = dt/epsilon / (1 + sigma*dt/(2*epsilon))
    denom = 1.0 + sigma * dt / (2.0 * epsilon)
    ca = (1.0 - sigma * dt / (2.0 * epsilon)) / denom
    cb = (dt / epsilon) / denom

    # Cq for H-field update: H^{n+1/2} = H^{n-1/2} - Cq*(curl E)
    cq = dt / mu

    return ca, cb, cq


def setup_pml(
    pml_width: Sequence[int],
    pml_start: Sequence[float],
    max_pml: float,
    dt: float,
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    start: float = 0.0,
    r_val: float = 1e-8,
    n_power: int = 4,
    eps: float = 1e-9,
    grid_spacing: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates a, b, and k profiles for electromagnetic C-PML.

    This implementation follows the standard CPML formulation for electromagnetic
    wave simulation with proper physical parameters.

    Only the first fd_pad[0]+pml_width[0] and last fd_pad[1]+pml_width[1]
    elements of the profiles will be non-zero.

    Args:
        pml_width: list of two integers specifying the width of the PML
            region [left/bottom, right/top].
        pml_start: list of two floats specifying the coordinates (in grid
            cells) of the start of the PML regions.
        max_pml: Float specifying the length (in distance units) of the
            longest PML over all sides and dimensions.
        dt: Time step interval.
        n: Integer specifying desired profile length, including fd_pad and
           pml_width.
        max_vel: Maximum wave speed (not used in EM formulation, kept for API).
        dtype: PyTorch datatype to use.
        device: PyTorch device to use.
        pml_freq: The frequency value to use for the profile (not used in EM,
            kept for API compatibility).
        start: Float specifying the coordinate (in grid cells) of the first
            element. Optional, default 0.
        r_val: The reflection coefficient. Optional, default 1e-8.
        n_power: The power for the profile. Optional, default 4.
        eps: A small number to prevent division by zero. Optional,
            default 1e-9.
        grid_spacing: The grid spacing (dx or dy). Optional, default 1.0.

    Returns:
        A tuple containing the (a, b, k) profiles as Tensors.
        - a: CPML 'a' coefficient for recursive convolution
        - b: CPML 'b' coefficient for recursive convolution
        - k: CPML stretching factor

    """
    # CPML parameters for electromagnetic waves
    k_max_cpml = 5.0  # maximum stretching factor
    alpha_max_cpml = 0.008  # maximum frequency shift

    # Create output tensors
    a = torch.zeros(n, device=device, dtype=dtype)
    b = torch.zeros(n, device=device, dtype=dtype)
    k = torch.ones(n, device=device, dtype=dtype)

    if max_pml == 0 or (pml_width[0] == 0 and pml_width[1] == 0):
        return a, b, k

    # Standard CPML sigma_max: sig0 = (Npower+1) / (150 * pi * dx)
    sigma0 = (n_power + 1) / (150.0 * math.pi * grid_spacing)

    # Calculate profiles for each grid point
    x = torch.arange(start, start + n, device=device, dtype=dtype)

    # Left/bottom PML region
    if pml_width[0] > 0:
        origin_left = pml_start[0]  # This is the inner edge of PML (in grid cells)
        abscissa_left = origin_left - x  # Distance from inner edge (in grid cells)
        mask_left = abscissa_left >= 0

        # Normalized distance into PML (0 at inner edge, 1 at outer edge)
        # abscissa_left is in grid cells, pml_width[0] is also in grid cells
        abscissa_norm_left = torch.clamp(abscissa_left / pml_width[0], 0, 1)

        # Sigma, k, alpha profiles with polynomial grading
        sigma_left = sigma0 * (abscissa_norm_left**n_power)
        k_left = 1.0 + (k_max_cpml - 1.0) * (abscissa_norm_left**n_power)
        alpha_left = alpha_max_cpml * (1.0 - abscissa_norm_left) + 0.1 * alpha_max_cpml

        # Apply to left region
        k = torch.where(mask_left, k_left, k)

        # Calculate b = exp(-(sigma/k + alpha) * dt / epsilon0)
        b_left = torch.exp(-(sigma_left / k_left + alpha_left) * dt / EP0)
        b = torch.where(mask_left, b_left, b)

        # Calculate a = sigma * (b - 1) / (k * (sigma + k * alpha))
        denom_left = k_left * (sigma_left + k_left * alpha_left) + eps
        a_left = sigma_left * (b_left - 1.0) / denom_left
        # Only apply where sigma is significant
        a_left = torch.where(sigma_left > 1e-6, a_left, torch.zeros_like(a_left))
        a = torch.where(mask_left, a_left, a)

    # Right/top PML region
    if pml_width[1] > 0:
        origin_right = pml_start[1]  # This is the inner edge of PML (in grid cells)
        abscissa_right = x - origin_right  # Distance from inner edge (in grid cells)
        mask_right = abscissa_right >= 0

        # Normalized distance into PML (0 at inner edge, 1 at outer edge)
        # abscissa_right is in grid cells, pml_width[1] is also in grid cells
        abscissa_norm_right = torch.clamp(abscissa_right / pml_width[1], 0, 1)

        # Sigma, k, alpha profiles
        sigma_right = sigma0 * (abscissa_norm_right**n_power)
        k_right = 1.0 + (k_max_cpml - 1.0) * (abscissa_norm_right**n_power)
        alpha_right = (
            alpha_max_cpml * (1.0 - abscissa_norm_right) + 0.1 * alpha_max_cpml
        )

        # Apply to right region
        k = torch.where(mask_right, k_right, k)

        # Calculate b = exp(-(sigma/k + alpha) * dt / epsilon0)
        b_right = torch.exp(-(sigma_right / k_right + alpha_right) * dt / EP0)
        b = torch.where(mask_right, b_right, b)

        # Calculate a = sigma * (b - 1) / (k * (sigma + k * alpha))
        denom_right = k_right * (sigma_right + k_right * alpha_right) + eps
        a_right = sigma_right * (b_right - 1.0) / denom_right
        a_right = torch.where(sigma_right > 1e-6, a_right, torch.zeros_like(a_right))
        a = torch.where(mask_right, a_right, a)

    return a, b, k


def setup_pml_half(
    pml_width: Sequence[int],
    pml_start: Sequence[float],
    max_pml: float,
    dt: float,
    n: int,
    max_vel: float,
    dtype: torch.dtype,
    device: torch.device,
    pml_freq: float,
    start: float = 0.0,
    r_val: float = 1e-8,
    n_power: int = 4,
    eps: float = 1e-9,
    grid_spacing: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates a, b, and k profiles for C-PML at half grid points.

    This is used for staggered grid implementations where some field components
    are located at half grid points (e.g., H fields in Yee grid).

    Args:
        Same as setup_pml.

    Returns:
        A tuple containing the (a_half, b_half, k_half) profiles as Tensors.

    """
    # CPML parameters for electromagnetic waves
    k_max_cpml = 5.0
    alpha_max_cpml = 0.008

    a = torch.zeros(n, device=device, dtype=dtype)
    b = torch.zeros(n, device=device, dtype=dtype)
    k = torch.ones(n, device=device, dtype=dtype)

    if max_pml == 0 or (pml_width[0] == 0 and pml_width[1] == 0):
        return a, b, k

    # Standard CPML sigma_max: sig0 = (Npower+1) / (150 * pi * dx)
    sigma0 = (n_power + 1) / (150.0 * math.pi * grid_spacing)

    # Half grid positions (shifted by dx/2 or dy/2)
    x = torch.arange(start, start + n, device=device, dtype=dtype) + 0.5

    # Left/bottom PML region (half grid)
    if pml_width[0] > 0:
        origin_left = pml_start[0]  # Inner edge of PML (in grid cells)
        abscissa_left = origin_left - x  # Distance in grid cells
        mask_left = abscissa_left >= 0

        # Normalized distance (both in grid cells)
        abscissa_norm_left = torch.clamp(abscissa_left / pml_width[0], 0, 1)

        sigma_left = sigma0 * (abscissa_norm_left**n_power)
        k_left = 1.0 + (k_max_cpml - 1.0) * (abscissa_norm_left**n_power)
        alpha_left = alpha_max_cpml * (1.0 - abscissa_norm_left) + 0.1 * alpha_max_cpml

        k = torch.where(mask_left, k_left, k)

        # b = exp(-(sigma/k + alpha) * dt / epsilon0)
        b_left = torch.exp(-(sigma_left / k_left + alpha_left) * dt / EP0)
        b = torch.where(mask_left, b_left, b)

        # a = sigma * (b - 1) / (k * (sigma + k * alpha))
        denom_left = k_left * (sigma_left + k_left * alpha_left) + eps
        a_left = sigma_left * (b_left - 1.0) / denom_left
        a_left = torch.where(sigma_left > 1e-6, a_left, torch.zeros_like(a_left))
        a = torch.where(mask_left, a_left, a)

    # Right/top PML region (half grid)
    if pml_width[1] > 0:
        origin_right = pml_start[1]  # Inner edge of PML (in grid cells)
        abscissa_right = x - origin_right  # Distance in grid cells
        mask_right = abscissa_right >= 0

        # Normalized distance (both in grid cells)
        abscissa_norm_right = torch.clamp(abscissa_right / pml_width[1], 0, 1)

        sigma_right = sigma0 * (abscissa_norm_right**n_power)
        k_right = 1.0 + (k_max_cpml - 1.0) * (abscissa_norm_right**n_power)
        alpha_right = (
            alpha_max_cpml * (1.0 - abscissa_norm_right) + 0.1 * alpha_max_cpml
        )

        k = torch.where(mask_right, k_right, k)

        # b = exp(-(sigma/k + alpha) * dt / epsilon0)
        b_right = torch.exp(-(sigma_right / k_right + alpha_right) * dt / EP0)
        b = torch.where(mask_right, b_right, b)

        # a = sigma * (b - 1) / (k * (sigma + k * alpha))
        denom_right = k_right * (sigma_right + k_right * alpha_right) + eps
        a_right = sigma_right * (b_right - 1.0) / denom_right
        a_right = torch.where(sigma_right > 1e-6, a_right, torch.zeros_like(a_right))
        a = torch.where(mask_right, a_right, a)

    return a, b, k
