"""
Maxwell Born approximation module for 2D TM mode.

This module implements the Born approximation for Maxwell's equations in 2D TM mode.
It simultaneously propagates background and scattered wavefields to model weak
perturbations in material properties (epsilon, sigma, mu).

The Born approximation assumes:
    δε << ε₀, δσ << σ₀, δμ << μ₀

where the subscript 0 denotes background (reference) values.

Usage:
    from tide.maxwell_born import MaxwellBornTM

    # Define background and perturbation models
    epsilon0 = torch.ones(ny, nx) * 4.0  # Background permittivity
    d_epsilon = torch.zeros(ny, nx)      # Perturbation
    d_epsilon[50:60, 50:60] = 0.1        # Small anomaly

    # Create propagator
    prop = MaxwellBornTM(
        epsilon0=epsilon0,
        sigma0=torch.zeros_like(epsilon0),
        mu0=torch.ones_like(epsilon0),
        d_epsilon=d_epsilon,
        d_sigma=torch.zeros_like(epsilon0),
        d_mu=torch.zeros_like(epsilon0),
        grid_spacing=0.01,
    )

    # Forward propagation
    r_bg, r_sc = prop(dt, source_amplitude, source_location, receiver_location)
"""

import math
from typing import Optional, Sequence, Tuple, Union

import torch

from . import staggered
from .cfl import cfl_condition
from .utils import EP0, MU0, prepare_parameters


def prepare_born_parameters(
    epsilon0: torch.Tensor,
    sigma0: torch.Tensor,
    mu0: torch.Tensor,
    d_epsilon: torch.Tensor,
    d_sigma: torch.Tensor,
    d_mu: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, ...]:
    """Prepare background and perturbation coefficients for Maxwell Born.

    Args:
        epsilon0: Background relative permittivity.
        sigma0: Background conductivity (S/m).
        mu0: Background relative permeability.
        d_epsilon: Perturbation in relative permittivity.
        d_sigma: Perturbation in conductivity (S/m).
        d_mu: Perturbation in relative permeability.
        dt: Time step (s).

    Returns:
        Tuple of (Ca0, Cb0, Cq0, dCa, dCb, dCq) coefficients.

    Note:
        The perturbation coefficients are computed via first-order Taylor expansion:

        Ca(ε+δε, σ+δσ) ≈ Ca(ε,σ) + ∂Ca/∂ε·δε + ∂Ca/∂σ·δσ

        For Ca = (1 - σdt/2ε) / (1 + σdt/2ε):
            ∂Ca/∂ε = σdt / [ε²(1 + σdt/2ε)²]
            ∂Ca/∂σ = -dt/ε / (1 + σdt/2ε)²

        For Cb = (dt/ε) / (1 + σdt/2ε):
            ∂Cb/∂ε = -dt/ε² / (1 + σdt/2ε) - dt/ε · σdt/2ε² / (1 + σdt/2ε)²
            ∂Cb/∂σ = -(dt/ε) · dt/2ε / (1 + σdt/2ε)²

        For Cq = dt/μ:
            ∂Cq/∂μ = -dt/μ²
    """
    # Compute background coefficients
    ca0, cb0, cq0 = prepare_parameters(epsilon0, sigma0, mu0, dt)

    # Convert to absolute values
    eps0_abs = epsilon0 * EP0
    eps_pert_abs = d_epsilon * EP0
    mu0_abs = mu0 * MU0
    mu_pert_abs = d_mu * MU0

    # Compute denominators (for numerical stability, reuse from background)
    denom = 1.0 + sigma0 * dt / (2.0 * eps0_abs)
    denom_sq = denom * denom

    # Partial derivatives of Ca with respect to ε and σ
    # ∂Ca/∂ε = σdt / [ε²(1 + σdt/2ε)²]
    dCa_deps = sigma0 * dt / (eps0_abs * eps0_abs * denom_sq)

    # ∂Ca/∂σ = -dt/ε / (1 + σdt/2ε)²
    dCa_dsigma = -dt / (eps0_abs * denom_sq)

    # dCa = ∂Ca/∂ε · δε + ∂Ca/∂σ · δσ
    dCa = dCa_deps * eps_pert_abs + dCa_dsigma * d_sigma

    # Partial derivatives of Cb with respect to ε and σ
    # ∂Cb/∂ε = -dt/ε² / (1 + σdt/2ε) - (dt/ε) · (σdt/2ε²) / (1 + σdt/2ε)²
    # Simplified: ∂Cb/∂ε = -dt/ε² · [1/(1+σdt/2ε) + (σdt/2ε)/(1+σdt/2ε)²]
    term1 = 1.0 / denom
    term2 = (sigma0 * dt / (2.0 * eps0_abs)) / denom_sq
    dCb_deps = -(dt / (eps0_abs * eps0_abs)) * (term1 + term2)

    # ∂Cb/∂σ = -(dt/ε) · (dt/2ε) / (1 + σdt/2ε)²
    dCb_dsigma = -(dt / eps0_abs) * (dt / (2.0 * eps0_abs)) / denom_sq

    # dCb = ∂Cb/∂ε · δε + ∂Cb/∂σ · δσ
    dCb = dCb_deps * eps_pert_abs + dCb_dsigma * d_sigma

    # Partial derivative of Cq with respect to μ
    # ∂Cq/∂μ = -dt/μ²
    dCq_dmu = -dt / (mu0_abs * mu0_abs)

    # dCq = ∂Cq/∂μ · δμ
    dCq = dCq_dmu * mu_pert_abs

    return ca0, cb0, cq0, dCa, dCb, dCq


class MaxwellBornTM(torch.nn.Module):
    """Maxwell Born approximation solver for 2D TM mode.

    This module propagates both background and scattered wavefields to model
    linear perturbations in electromagnetic material properties.

    Args:
        epsilon0: Background relative permittivity [ny, nx].
        sigma0: Background conductivity [ny, nx] in S/m.
        mu0: Background relative permeability [ny, nx].
        d_epsilon: Perturbation in relative permittivity [ny, nx].
        d_sigma: Perturbation in conductivity [ny, nx] in S/m.
        d_mu: Perturbation in relative permeability [ny, nx].
        grid_spacing: Grid spacing in meters.
        d_epsilon_requires_grad: Whether to compute gradients for d_epsilon.
        d_sigma_requires_grad: Whether to compute gradients for d_sigma.
        d_mu_requires_grad: Whether to compute gradients for d_mu.

    Note:
        The Born approximation is valid when perturbations are small:
        |δε/ε₀| << 1, |δσ/σ₀| << 1, |δμ/μ₀| << 1

        For strong contrasts or multiple scattering, full nonlinear modeling
        should be used instead.
    """

    def __init__(
        self,
        epsilon0: torch.Tensor,
        sigma0: torch.Tensor,
        mu0: torch.Tensor,
        d_epsilon: torch.Tensor,
        d_sigma: torch.Tensor,
        d_mu: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        d_epsilon_requires_grad: Optional[bool] = None,
        d_sigma_requires_grad: Optional[bool] = None,
        d_mu_requires_grad: Optional[bool] = None,
    ) -> None:
        super().__init__()

        # Validate inputs
        if not isinstance(epsilon0, torch.Tensor):
            raise TypeError(
                f"epsilon0 must be torch.Tensor, got {type(epsilon0).__name__}"
            )
        if not isinstance(sigma0, torch.Tensor):
            raise TypeError(f"sigma0 must be torch.Tensor, got {type(sigma0).__name__}")
        if not isinstance(mu0, torch.Tensor):
            raise TypeError(f"mu0 must be torch.Tensor, got {type(mu0).__name__}")
        if not isinstance(d_epsilon, torch.Tensor):
            raise TypeError(
                f"d_epsilon must be torch.Tensor, got {type(d_epsilon).__name__}"
            )
        if not isinstance(d_sigma, torch.Tensor):
            raise TypeError(
                f"d_sigma must be torch.Tensor, got {type(d_sigma).__name__}"
            )
        if not isinstance(d_mu, torch.Tensor):
            raise TypeError(f"d_mu must be torch.Tensor, got {type(d_mu).__name__}")

        # Register background parameters as buffers (non-trainable)
        self.register_buffer("epsilon0", epsilon0)
        self.register_buffer("sigma0", sigma0)
        self.register_buffer("mu0", mu0)

        # Register perturbation parameters (potentially trainable)
        if d_epsilon_requires_grad is None:
            d_epsilon_requires_grad = d_epsilon.requires_grad
        if d_sigma_requires_grad is None:
            d_sigma_requires_grad = d_sigma.requires_grad
        if d_mu_requires_grad is None:
            d_mu_requires_grad = d_mu.requires_grad

        self.d_epsilon = torch.nn.Parameter(
            d_epsilon, requires_grad=d_epsilon_requires_grad
        )
        self.d_sigma = torch.nn.Parameter(d_sigma, requires_grad=d_sigma_requires_grad)
        self.d_mu = torch.nn.Parameter(d_mu, requires_grad=d_mu_requires_grad)

        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitude: Optional[torch.Tensor],
        source_location: Optional[torch.Tensor],
        receiver_location: Optional[torch.Tensor],
        stencil: int = 2,
        pml_width: Union[int, Sequence[int]] = 20,
        max_vel: Optional[float] = None,
        Ey_bg_0: Optional[torch.Tensor] = None,
        Hx_bg_0: Optional[torch.Tensor] = None,
        Hz_bg_0: Optional[torch.Tensor] = None,
        Ey_sc_0: Optional[torch.Tensor] = None,
        Hx_sc_0: Optional[torch.Tensor] = None,
        Hz_sc_0: Optional[torch.Tensor] = None,
        m_Ey_x_bg: Optional[torch.Tensor] = None,
        m_Ey_z_bg: Optional[torch.Tensor] = None,
        m_Hx_z_bg: Optional[torch.Tensor] = None,
        m_Hz_x_bg: Optional[torch.Tensor] = None,
        m_Ey_x_sc: Optional[torch.Tensor] = None,
        m_Ey_z_sc: Optional[torch.Tensor] = None,
        m_Hx_z_sc: Optional[torch.Tensor] = None,
        m_Hz_x_sc: Optional[torch.Tensor] = None,
        nt: Optional[int] = None,
        source_amplitude_sc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation of Born scattered wavefield.

        Args:
            dt: Time step (seconds).
            source_amplitude: Source amplitudes [n_shots, n_sources, nt].
            source_location: Source locations [n_shots, n_sources, 2] (y, x).
            receiver_location: Receiver locations [n_shots, n_receivers, 2] (y, x).
            stencil: Finite difference stencil order (2, 4, 6, or 8).
            pml_width: PML absorbing boundary width.
            max_vel: Maximum velocity for CFL condition (default: computed from epsilon0).
            Ey_bg_0: Initial Ey background field.
            Hx_bg_0: Initial Hx background field.
            Hz_bg_0: Initial Hz background field.
            Ey_sc_0: Initial Ey scattered field.
            Hx_sc_0: Initial Hx scattered field.
            Hz_sc_0: Initial Hz scattered field.
            m_Ey_x_bg: PML memory variable for background Ey (x-direction).
            m_Ey_z_bg: PML memory variable for background Ey (z-direction).
            m_Hx_z_bg: PML memory variable for background Hx.
            m_Hz_x_bg: PML memory variable for background Hz.
            m_Ey_x_sc: PML memory variable for scattered Ey (x-direction).
            m_Ey_z_sc: PML memory variable for scattered Ey (z-direction).
            m_Hx_z_sc: PML memory variable for scattered Hx.
            m_Hz_x_sc: PML memory variable for scattered Hz.
            nt: Number of time steps (default: inferred from source_amplitude).
            source_amplitude_sc: Optional separate source for scattered field.
                If None, uses same source as background.

        Returns:
            Tuple of (r_bg, r_sc) where:
                r_bg: Background receiver data [n_shots, n_receivers, nt].
                r_sc: Scattered receiver data [n_shots, n_receivers, nt].
        """
        assert isinstance(self.epsilon0, torch.Tensor)
        assert isinstance(self.sigma0, torch.Tensor)
        assert isinstance(self.mu0, torch.Tensor)

        return maxwell_born_tm(
            self.epsilon0,
            self.sigma0,
            self.mu0,
            self.d_epsilon,
            self.d_sigma,
            self.d_mu,
            self.grid_spacing,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            stencil,
            pml_width,
            max_vel,
            Ey_bg_0,
            Hx_bg_0,
            Hz_bg_0,
            Ey_sc_0,
            Hx_sc_0,
            Hz_sc_0,
            m_Ey_x_bg,
            m_Ey_z_bg,
            m_Hx_z_bg,
            m_Hz_x_bg,
            m_Ey_x_sc,
            m_Ey_z_sc,
            m_Hx_z_sc,
            m_Hz_x_sc,
            nt,
            source_amplitude_sc,
        )


def maxwell_born_tm(
    epsilon0: torch.Tensor,
    sigma0: torch.Tensor,
    mu0: torch.Tensor,
    d_epsilon: torch.Tensor,
    d_sigma: torch.Tensor,
    d_mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int = 2,
    pml_width: Union[int, Sequence[int]] = 20,
    max_vel: Optional[float] = None,
    Ey_bg_0: Optional[torch.Tensor] = None,
    Hx_bg_0: Optional[torch.Tensor] = None,
    Hz_bg_0: Optional[torch.Tensor] = None,
    Ey_sc_0: Optional[torch.Tensor] = None,
    Hx_sc_0: Optional[torch.Tensor] = None,
    Hz_sc_0: Optional[torch.Tensor] = None,
    m_Ey_x_bg: Optional[torch.Tensor] = None,
    m_Ey_z_bg: Optional[torch.Tensor] = None,
    m_Hx_z_bg: Optional[torch.Tensor] = None,
    m_Hz_x_bg: Optional[torch.Tensor] = None,
    m_Ey_x_sc: Optional[torch.Tensor] = None,
    m_Ey_z_sc: Optional[torch.Tensor] = None,
    m_Hx_z_sc: Optional[torch.Tensor] = None,
    m_Hz_x_sc: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    source_amplitude_sc: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maxwell Born TM mode forward propagation.

    This is the functional interface to Maxwell Born modeling.
    See MaxwellBornTM class for parameter descriptions.
    """

    # Validate dimensions
    if epsilon0.ndim != 2:
        raise ValueError(f"epsilon0 must be 2D, got shape {epsilon0.shape}")
    if sigma0.shape != epsilon0.shape:
        raise ValueError(
            f"sigma0 shape {sigma0.shape} != epsilon0 shape {epsilon0.shape}"
        )
    if mu0.shape != epsilon0.shape:
        raise ValueError(f"mu0 shape {mu0.shape} != epsilon0 shape {epsilon0.shape}")
    if d_epsilon.shape != epsilon0.shape:
        raise ValueError(
            f"d_epsilon shape {d_epsilon.shape} != epsilon0 shape {epsilon0.shape}"
        )
    if d_sigma.shape != epsilon0.shape:
        raise ValueError(
            f"d_sigma shape {d_sigma.shape} != epsilon0 shape {epsilon0.shape}"
        )
    if d_mu.shape != epsilon0.shape:
        raise ValueError(f"d_mu shape {d_mu.shape} != epsilon0 shape {epsilon0.shape}")

    ny, nx = epsilon0.shape
    device = epsilon0.device
    dtype = epsilon0.dtype

    # Parse grid spacing
    if isinstance(grid_spacing, (int, float)):
        dy = dx = float(grid_spacing)
    else:
        dy, dx = float(grid_spacing[0]), float(grid_spacing[1])

    # Parse PML width
    if isinstance(pml_width, int):
        pml_y = pml_x = pml_width
    else:
        pml_y, pml_x = pml_width[0], pml_width[1]

    # Compute background and perturbation coefficients
    ca0, cb0, cq0, dca, dcb, dcq = prepare_born_parameters(
        epsilon0, sigma0, mu0, d_epsilon, d_sigma, d_mu, dt
    )

    # Determine number of time steps
    if source_amplitude is not None:
        if nt is None:
            nt = source_amplitude.shape[2]
        n_shots = source_amplitude.shape[0]
        n_sources = source_amplitude.shape[1]
    else:
        if nt is None:
            raise ValueError("Either source_amplitude or nt must be provided")
        n_shots = 1
        n_sources = 0

    # Use same source for scattered field if not specified
    if source_amplitude_sc is None and source_amplitude is not None:
        source_amplitude_sc = torch.zeros_like(source_amplitude)

    # Determine number of receivers
    if receiver_location is not None:
        n_receivers = receiver_location.shape[1]
    else:
        n_receivers = 0

    # CFL condition check
    if max_vel is None:
        # Estimate max velocity from epsilon and mu
        c = 1.0 / torch.sqrt(epsilon0 * EP0 * mu0 * MU0)
        max_vel = float(c.max())

    cfl_dt = cfl_condition(dx, max_vel, stencil)
    if dt > cfl_dt:
        raise ValueError(
            f"CFL condition violated: dt={dt:.2e} > {cfl_dt:.2e}. "
            f"Reduce dt or increase grid_spacing."
        )

    # Initialize fields if not provided
    if Ey_bg_0 is None:
        Ey_bg_0 = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if Hx_bg_0 is None:
        Hx_bg_0 = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if Hz_bg_0 is None:
        Hz_bg_0 = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if Ey_sc_0 is None:
        Ey_sc_0 = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if Hx_sc_0 is None:
        Hx_sc_0 = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if Hz_sc_0 is None:
        Hz_sc_0 = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

    # Initialize PML memory variables
    if m_Ey_x_bg is None:
        m_Ey_x_bg = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if m_Ey_z_bg is None:
        m_Ey_z_bg = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if m_Hx_z_bg is None:
        m_Hx_z_bg = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if m_Hz_x_bg is None:
        m_Hz_x_bg = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if m_Ey_x_sc is None:
        m_Ey_x_sc = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if m_Ey_z_sc is None:
        m_Ey_z_sc = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if m_Hx_z_sc is None:
        m_Hx_z_sc = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
    if m_Hz_x_sc is None:
        m_Hz_x_sc = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

    # Setup PML profiles
    if pml_y == 0:
        sigma_max_y = 0.0
    else:
        sigma_max_y = (
            -(stencil + 1) * max_vel * math.log(1e-5) / (2.0 * pml_y * dy)
        )
    if pml_x == 0:
        sigma_max_x = 0.0
    else:
        sigma_max_x = (
            -(stencil + 1) * max_vel * math.log(1e-5) / (2.0 * pml_x * dx)
        )

    ay, ayh, by, byh, ky, kyh = staggered.setup_pml_profiles_1d(
        ny, pml_y, pml_y, sigma_max_y, dt, device, dtype
    )
    ax, axh, bx, bxh, kx, kxh = staggered.setup_pml_profiles_1d(
        nx, pml_x, pml_x, sigma_max_x, dt, device, dtype
    )

    # Convert source/receiver locations to flat indices
    if source_location is not None:
        sources_i = (
            source_location[..., 0].long() * nx + source_location[..., 1].long()
        ).flatten()
    else:
        sources_i = torch.tensor([], dtype=torch.int64, device=device)

    if receiver_location is not None:
        receivers_i = (
            receiver_location[..., 0].long() * nx + receiver_location[..., 1].long()
        ).flatten()
    else:
        receivers_i = torch.tensor([], dtype=torch.int64, device=device)

    # Check if CUDA implementation is available
    try:
        import tide._C

        if dtype == torch.float32:
            dtype_str = "float"
        elif dtype == torch.float64:
            dtype_str = "double"
        else:
            dtype_str = None

        use_cuda = (
            device.type == "cuda"
            and dtype_str is not None
            and hasattr(
                tide._C, f"maxwell_born_tm_{stencil}_{dtype_str}_forward_cuda"
            )
        )
    except (ImportError, AttributeError):
        use_cuda = False

    if use_cuda:
        # Call CUDA kernel
        func_name = f"maxwell_born_tm_{stencil}_{dtype_str}_forward_cuda"
        func = getattr(tide._C, func_name)

        r_bg_storage = torch.zeros(
            nt, n_shots, n_receivers, device=device, dtype=dtype
        )
        r_sc_storage = torch.zeros(
            nt, n_shots, n_receivers, device=device, dtype=dtype
        )

        func(
            ca0,
            cb0,
            cq0,
            dca,
            dcb,
            dcq,
            Ey_bg_0,
            Hx_bg_0,
            Hz_bg_0,
            Ey_sc_0,
            Hx_sc_0,
            Hz_sc_0,
            m_Ey_x_bg,
            m_Ey_z_bg,
            m_Hx_z_bg,
            m_Hz_x_bg,
            m_Ey_x_sc,
            m_Ey_z_sc,
            m_Hx_z_sc,
            m_Hz_x_sc,
            ay,
            ayh,
            ax,
            axh,
            by,
            byh,
            bx,
            bxh,
            ky,
            kyh,
            kx,
            kxh,
            r_bg_storage,
            r_sc_storage,
            source_amplitude if source_amplitude is not None else torch.empty(0),
            source_amplitude_sc if source_amplitude_sc is not None else torch.empty(0),
            sources_i,
            receivers_i,
            1.0 / dy,
            1.0 / dx,
            ny,
            nx,
            n_shots,
            n_sources,
            n_receivers,
            pml_y,
            ny - pml_y,
            pml_x,
            nx - pml_x,
            nt,
            ca0.ndim == 3,
            cb0.ndim == 3,
            cq0.ndim == 3,
            dca.ndim == 3,
            dcb.ndim == 3,
            dcq.ndim == 3,
        )
        r_bg = r_bg_storage.permute(1, 2, 0)
        r_sc = r_sc_storage.permute(1, 2, 0)
    else:
        r_bg = torch.zeros(n_shots, n_receivers, nt, device=device, dtype=dtype)
        r_sc = torch.zeros(n_shots, n_receivers, nt, device=device, dtype=dtype)

        fd_pad = stencil // 2
        rdy = torch.tensor(1.0 / dy, device=device, dtype=dtype)
        rdx = torch.tensor(1.0 / dx, device=device, dtype=dtype)

        y_idx = torch.arange(ny, device=device)
        x_idx = torch.arange(nx, device=device)

        mask_y_e = (y_idx >= fd_pad) & (y_idx <= ny - fd_pad)
        mask_x_e = (x_idx >= fd_pad) & (x_idx <= nx - fd_pad)
        mask_e = (mask_y_e[:, None] & mask_x_e[None, :]).unsqueeze(0)

        mask_y_hx = (y_idx >= fd_pad) & (y_idx < ny - fd_pad)
        mask_x_hx = (x_idx >= fd_pad) & (x_idx <= nx - fd_pad)
        mask_hx = (mask_y_hx[:, None] & mask_x_hx[None, :]).unsqueeze(0)

        mask_y_hz = (y_idx >= fd_pad) & (y_idx <= ny - fd_pad)
        mask_x_hz = (x_idx >= fd_pad) & (x_idx < nx - fd_pad)
        mask_hz = (mask_y_hz[:, None] & mask_x_hz[None, :]).unsqueeze(0)

        pml_y0 = pml_y
        pml_y1 = ny - pml_y
        pml_x0 = pml_x
        pml_x1 = nx - pml_x

        pml_y0h = pml_y0
        pml_y1h = max(pml_y0, pml_y1 - 1)
        pml_x0h = pml_x0
        pml_x1h = max(pml_x0, pml_x1 - 1)

        pml_y_mask = ((y_idx < pml_y0) | (y_idx >= pml_y1))
        pml_x_mask = ((x_idx < pml_x0) | (x_idx >= pml_x1))
        pml_y_mask_h = ((y_idx < pml_y0h) | (y_idx >= pml_y1h))
        pml_x_mask_h = ((x_idx < pml_x0h) | (x_idx >= pml_x1h))

        pml_y_mask_e = (pml_y_mask[:, None] & mask_e[0]).unsqueeze(0)
        pml_x_mask_e = (pml_x_mask[None, :] & mask_e[0]).unsqueeze(0)
        pml_y_mask_h = (pml_y_mask_h[:, None] & mask_hx[0]).unsqueeze(0)
        pml_x_mask_h = (pml_x_mask_h[None, :] & mask_hz[0]).unsqueeze(0)

        ay_y = ay.view(1, ny, 1)
        by_y = by.view(1, ny, 1)
        ky_y = ky.view(1, ny, 1)
        ayh_y = ayh.view(1, ny, 1)
        byh_y = byh.view(1, ny, 1)
        kyh_y = kyh.view(1, ny, 1)

        ax_x = ax.view(1, 1, nx)
        bx_x = bx.view(1, 1, nx)
        kx_x = kx.view(1, 1, nx)
        axh_x = axh.view(1, 1, nx)
        bxh_x = bxh.view(1, 1, nx)
        kxh_x = kxh.view(1, 1, nx)

        if n_sources > 0:
            sources_i = sources_i.view(n_shots, n_sources)
            sources_i_safe = sources_i.clamp(min=0)
            sources_mask = sources_i >= 0
            sources_need_mask = not torch.all(sources_mask).item()
        if n_receivers > 0:
            receivers_i = receivers_i.view(n_shots, n_receivers)
            receivers_i_safe = receivers_i.clamp(min=0)
            receivers_mask = receivers_i >= 0
            receivers_need_mask = not torch.all(receivers_mask).item()

        for t in range(nt):
            if n_sources > 0 and source_amplitude is not None:
                src_amp = source_amplitude[:, :, t]
                if sources_need_mask:
                    src_amp = torch.where(sources_mask, src_amp, torch.zeros_like(src_amp))
                Ey_bg = (
                    Ey_bg_0.reshape(n_shots, ny * nx)
                    .scatter_add(1, sources_i_safe, src_amp)
                    .reshape(n_shots, ny, nx)
                )

                if source_amplitude_sc is not None:
                    src_amp_sc = source_amplitude_sc[:, :, t]
                    if sources_need_mask:
                        src_amp_sc = torch.where(
                            sources_mask, src_amp_sc, torch.zeros_like(src_amp_sc)
                        )
                    Ey_sc = (
                        Ey_sc_0.reshape(n_shots, ny * nx)
                        .scatter_add(1, sources_i_safe, src_amp_sc)
                        .reshape(n_shots, ny, nx)
                    )
                else:
                    Ey_sc = Ey_sc_0
            else:
                Ey_bg = Ey_bg_0
                Ey_sc = Ey_sc_0

            dEy_bg_dz = staggered.diffyh1(Ey_bg, stencil, rdy)
            dEy_bg_dx = staggered.diffxh1(Ey_bg, stencil, rdx)

            m_Ey_z_bg = torch.where(
                pml_y_mask_h, byh_y * m_Ey_z_bg + ayh_y * dEy_bg_dz, m_Ey_z_bg
            )
            dEy_bg_dz_cpml = torch.where(
                pml_y_mask_h, dEy_bg_dz / kyh_y + m_Ey_z_bg, dEy_bg_dz
            )

            m_Ey_x_bg = torch.where(
                pml_x_mask_h, bxh_x * m_Ey_x_bg + axh_x * dEy_bg_dx, m_Ey_x_bg
            )
            dEy_bg_dx_cpml = torch.where(
                pml_x_mask_h, dEy_bg_dx / kxh_x + m_Ey_x_bg, dEy_bg_dx
            )

            Hx_bg = torch.where(mask_hx, Hx_bg_0 - cq0 * dEy_bg_dz_cpml, Hx_bg_0)
            Hz_bg = torch.where(mask_hz, Hz_bg_0 + cq0 * dEy_bg_dx_cpml, Hz_bg_0)

            dEy_sc_dz = staggered.diffyh1(Ey_sc, stencil, rdy)
            dEy_sc_dx = staggered.diffxh1(Ey_sc, stencil, rdx)

            m_Ey_z_sc = torch.where(
                pml_y_mask_h, byh_y * m_Ey_z_sc + ayh_y * dEy_sc_dz, m_Ey_z_sc
            )
            dEy_sc_dz_cpml = torch.where(
                pml_y_mask_h, dEy_sc_dz / kyh_y + m_Ey_z_sc, dEy_sc_dz
            )

            m_Ey_x_sc = torch.where(
                pml_x_mask_h, bxh_x * m_Ey_x_sc + axh_x * dEy_sc_dx, m_Ey_x_sc
            )
            dEy_sc_dx_cpml = torch.where(
                pml_x_mask_h, dEy_sc_dx / kxh_x + m_Ey_x_sc, dEy_sc_dx
            )

            Hx_sc = torch.where(
                mask_hx,
                Hx_sc_0 - cq0 * dEy_sc_dz_cpml - dcq * dEy_bg_dz_cpml,
                Hx_sc_0,
            )
            Hz_sc = torch.where(
                mask_hz,
                Hz_sc_0 + cq0 * dEy_sc_dx_cpml + dcq * dEy_bg_dx_cpml,
                Hz_sc_0,
            )

            dHz_bg_dx = staggered.diffx1(Hz_bg, stencil, rdx)
            dHx_bg_dz = staggered.diffy1(Hx_bg, stencil, rdy)

            m_Hz_x_bg = torch.where(
                pml_x_mask_e, bx_x * m_Hz_x_bg + ax_x * dHz_bg_dx, m_Hz_x_bg
            )
            dHz_bg_dx_cpml = torch.where(
                pml_x_mask_e, dHz_bg_dx / kx_x + m_Hz_x_bg, dHz_bg_dx
            )

            m_Hx_z_bg = torch.where(
                pml_y_mask_e, by_y * m_Hx_z_bg + ay_y * dHx_bg_dz, m_Hx_z_bg
            )
            dHx_bg_dz_cpml = torch.where(
                pml_y_mask_e, dHx_bg_dz / ky_y + m_Hx_z_bg, dHx_bg_dz
            )

            curl_h_bg = dHz_bg_dx_cpml - dHx_bg_dz_cpml
            Ey_bg = torch.where(mask_e, ca0 * Ey_bg + cb0 * curl_h_bg, Ey_bg)

            dHz_sc_dx = staggered.diffx1(Hz_sc, stencil, rdx)
            dHx_sc_dz = staggered.diffy1(Hx_sc, stencil, rdy)

            m_Hz_x_sc = torch.where(
                pml_x_mask_e, bx_x * m_Hz_x_sc + ax_x * dHz_sc_dx, m_Hz_x_sc
            )
            dHz_sc_dx_cpml = torch.where(
                pml_x_mask_e, dHz_sc_dx / kx_x + m_Hz_x_sc, dHz_sc_dx
            )

            m_Hx_z_sc = torch.where(
                pml_y_mask_e, by_y * m_Hx_z_sc + ay_y * dHx_sc_dz, m_Hx_z_sc
            )
            dHx_sc_dz_cpml = torch.where(
                pml_y_mask_e, dHx_sc_dz / ky_y + m_Hx_z_sc, dHx_sc_dz
            )

            curl_h_sc = dHz_sc_dx_cpml - dHx_sc_dz_cpml
            Ey_sc = torch.where(
                mask_e,
                ca0 * Ey_sc + cb0 * curl_h_sc + dca * Ey_bg + dcb * curl_h_bg,
                Ey_sc,
            )

            if n_receivers > 0:
                Ey_bg_flat = Ey_bg.reshape(n_shots, ny * nx)
                Ey_sc_flat = Ey_sc.reshape(n_shots, ny * nx)
                bg_vals = Ey_bg_flat.gather(1, receivers_i_safe)
                sc_vals = Ey_sc_flat.gather(1, receivers_i_safe)
                if receivers_need_mask:
                    bg_vals = torch.where(
                        receivers_mask, bg_vals, torch.zeros_like(bg_vals)
                    )
                    sc_vals = torch.where(
                        receivers_mask, sc_vals, torch.zeros_like(sc_vals)
                    )
                r_bg[:, :, t] = bg_vals
                r_sc[:, :, t] = sc_vals

            Ey_bg_0 = Ey_bg
            Ey_sc_0 = Ey_sc
            Hx_bg_0 = Hx_bg
            Hz_bg_0 = Hz_bg
            Hx_sc_0 = Hx_sc
            Hz_sc_0 = Hz_sc

    return r_bg, r_sc
