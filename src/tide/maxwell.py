import itertools
import warnings
from typing import Any, Callable, Literal, Optional, Sequence, Union

import torch

from . import staggered
from .callbacks import Callback, CallbackState
from .cfl import cfl_condition
from .grid_utils import (
    _normalize_grid_spacing_2d,
    _normalize_grid_spacing_3d,
    _normalize_pml_width_2d,
    _normalize_pml_width_3d,
)
from .padding import create_or_pad, zero_interior
from .resampling import downsample_and_movedim, upsample
from .storage import (
    _CPU_STORAGE_BUFFERS,
    STORAGE_CPU,
    STORAGE_DEVICE,
    STORAGE_DISK,
    STORAGE_NONE,
    TemporaryStorage,
    _normalize_storage_compression,
    _resolve_storage_compression,
    storage_mode_to_int,
)
from .utils import (
    C0,
    EP0,
    NondimContext,
    build_nondim_context,
    prepare_parameters,
    prepare_parameters_nondim,
)
from .validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)

_CTX_HANDLE_COUNTER = itertools.count()
_CTX_HANDLE_REGISTRY: dict[int, dict[str, Any]] = {}


def _register_ctx_handle(ctx_data: dict[str, Any]) -> torch.Tensor:
    handle = next(_CTX_HANDLE_COUNTER)
    _CTX_HANDLE_REGISTRY[handle] = ctx_data
    return torch.tensor(handle, dtype=torch.int64)


def _get_ctx_handle(handle: int) -> dict[str, Any]:
    try:
        return _CTX_HANDLE_REGISTRY[handle]
    except KeyError as exc:
        raise RuntimeError(f"Unknown context handle: {handle}") from exc


def _release_ctx_handle(handle: Optional[int]) -> None:
    if handle is None:
        return
    _CTX_HANDLE_REGISTRY.pop(handle, None)


def _init_tm_wavefield(
    field_0: Optional[torch.Tensor],
    *,
    n_shots: int,
    size_with_batch: tuple[int, int, int],
    fd_pad_list: list[int],
    device: torch.device,
    dtype: torch.dtype,
    contiguous: bool = False,
) -> torch.Tensor:
    """Initialize TM wavefield tensors with asymmetric FD padding."""
    if field_0 is not None:
        if field_0.ndim == 2:
            field_0 = field_0[None, :, :].expand(n_shots, -1, -1)
        wavefield = create_or_pad(
            field_0,
            fd_pad_list,
            device,
            dtype,
            size_with_batch,
            mode="constant",
        )
    else:
        wavefield = torch.zeros(size_with_batch, device=device, dtype=dtype)
    return wavefield.contiguous() if contiguous else wavefield


class MaxwellTM(torch.nn.Module):
    """2D TM mode Maxwell equations solver using FDTD method.

    This module solves the TM (Transverse Magnetic) mode Maxwell equations
    in 2D with fields (Ey, Hx, Hz) using the FDTD method with CPML absorbing
    boundary conditions.

    Args:
        epsilon: Relative permittivity tensor [ny, nx].
            For vacuum/air, use 1.0. For common materials:
            - Water: ~80
            - Glass: ~4-7
            - Soil (dry): ~3-5
            - Concrete: ~4-8
        sigma: Electrical conductivity tensor [ny, nx] in S/m.
            For lossless media, use 0.0.
        mu: Relative permeability tensor [ny, nx].
            For most non-magnetic materials, use 1.0.
        grid_spacing: Grid spacing in meters. Can be a single value (same for
            both directions) or a sequence [dy, dx].
        epsilon_requires_grad: Whether to compute gradients for permittivity.
        sigma_requires_grad: Whether to compute gradients for conductivity.

    Note:
        The input parameters are RELATIVE values (dimensionless). They will be
        multiplied internally by the vacuum permittivity (ε₀ = 8.854e-12 F/m)
        and vacuum permeability (μ₀ = 1.257e-6 H/m) respectively.
    """

    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        epsilon_requires_grad: Optional[bool] = None,
        sigma_requires_grad: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if epsilon_requires_grad is not None and not isinstance(
            epsilon_requires_grad, bool
        ):
            raise TypeError(
                f"epsilon_requires_grad must be bool or None, "
                f"got {type(epsilon_requires_grad).__name__}",
            )
        if not isinstance(epsilon, torch.Tensor):
            raise TypeError(
                f"epsilon must be torch.Tensor, got {type(epsilon).__name__}",
            )
        if sigma_requires_grad is not None and not isinstance(
            sigma_requires_grad, bool
        ):
            raise TypeError(
                f"sigma_requires_grad must be bool or None, "
                f"got {type(sigma_requires_grad).__name__}",
            )
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(
                f"sigma must be torch.Tensor, got {type(sigma).__name__}",
            )
        if not isinstance(mu, torch.Tensor):
            raise TypeError(
                f"mu must be torch.Tensor, got {type(mu).__name__}",
            )

        # If requires_grad not specified, preserve the input tensor's setting
        if epsilon_requires_grad is None:
            epsilon_requires_grad = epsilon.requires_grad
        if sigma_requires_grad is None:
            sigma_requires_grad = sigma.requires_grad

        self.epsilon = torch.nn.Parameter(epsilon, requires_grad=epsilon_requires_grad)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=sigma_requires_grad)
        self.register_buffer("mu", mu)  # In normal we don't optimize mu
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitude: Optional[torch.Tensor],  # [shot,source,time]
        source_location: Optional[torch.Tensor],  # [shot,source,2]
        receiver_location: Optional[torch.Tensor],  # [shot,receiver,2]
        stencil: int = 2,
        pml_width: Union[int, Sequence[int]] = 20,
        max_vel: Optional[float] = None,
        Ey_0: Optional[torch.Tensor] = None,
        Hx_0: Optional[torch.Tensor] = None,
        Hz_0: Optional[torch.Tensor] = None,
        m_Ey_x: Optional[torch.Tensor] = None,
        m_Ey_z: Optional[torch.Tensor] = None,
        m_Hx_z: Optional[torch.Tensor] = None,
        m_Hz_x: Optional[torch.Tensor] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        save_snapshots: Optional[bool] = None,
        forward_callback: Optional[Callback] = None,
        backward_callback: Optional[Callback] = None,
        callback_frequency: int = 1,
        python_backend: Union[bool, str] = False,
        storage_mode: str = "device",
        storage_path: str = ".",
        storage_compression: Union[bool, str] = False,
        storage_bytes_limit_device: Optional[int] = None,
        storage_bytes_limit_host: Optional[int] = None,
        storage_chunk_steps: int = 0,
        compute_dtype: str = "fp32",
        mp_mode: str = "throughput",
    ):
        # Type assertions for buffer and parameter tensors
        assert isinstance(self.epsilon, torch.Tensor)
        assert isinstance(self.sigma, torch.Tensor)
        assert isinstance(self.mu, torch.Tensor)
        return maxwelltm(
            self.epsilon,
            self.sigma,
            self.mu,
            self.grid_spacing,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            stencil,
            pml_width,
            max_vel,
            Ey_0,
            Hx_0,
            Hz_0,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            nt,
            model_gradient_sampling_interval,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            save_snapshots,
            forward_callback,
            backward_callback,
            callback_frequency,
            python_backend,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            compute_dtype=compute_dtype,
            mp_mode=mp_mode,
        )


def _normalize_compute_dtype(value: str) -> Literal["fp32", "fp16"]:
    if not isinstance(value, str):
        raise TypeError("compute_dtype must be a string.")
    dtype_norm = value.lower()
    if dtype_norm not in {"fp32", "fp16"}:
        raise ValueError("compute_dtype must be 'fp32' or 'fp16'.")
    return dtype_norm  # type: ignore[return-value]


def _normalize_mp_mode(value: str) -> Literal["throughput", "balanced", "robust"]:
    if not isinstance(value, str):
        raise TypeError("mp_mode must be a string.")
    mode_norm = value.lower()
    if mode_norm not in {"throughput", "balanced", "robust"}:
        raise ValueError("mp_mode must be 'throughput', 'balanced', or 'robust'.")
    return mode_norm  # type: ignore[return-value]


def _scale_tm_states_to_nondim(
    Ey_0: Optional[torch.Tensor],
    Hx_0: Optional[torch.Tensor],
    Hz_0: Optional[torch.Tensor],
    m_Ey_x: Optional[torch.Tensor],
    m_Ey_z: Optional[torch.Tensor],
    m_Hx_z: Optional[torch.Tensor],
    m_Hz_x: Optional[torch.Tensor],
    ctx: NondimContext,
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    return (
        Ey_0,
        None if Hx_0 is None else Hx_0 * ctx.Zref,
        None if Hz_0 is None else Hz_0 * ctx.Zref,
        None if m_Ey_x is None else m_Ey_x * ctx.L0,
        None if m_Ey_z is None else m_Ey_z * ctx.L0,
        None if m_Hx_z is None else m_Hx_z * (ctx.Zref * ctx.L0),
        None if m_Hz_x is None else m_Hz_x * (ctx.Zref * ctx.L0),
    )


def _restore_tm_states_from_nondim(
    Ey: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    m_Ey_x: torch.Tensor,
    m_Ey_z: torch.Tensor,
    m_Hx_z: torch.Tensor,
    m_Hz_x: torch.Tensor,
    ctx: NondimContext,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        Ey,
        Hx / ctx.Zref,
        Hz / ctx.Zref,
        m_Ey_x / ctx.L0,
        m_Ey_z / ctx.L0,
        m_Hx_z / (ctx.Zref * ctx.L0),
        m_Hz_x / (ctx.Zref * ctx.L0),
    )


def _warn_nondim_ranges(
    epsilon_hat: torch.Tensor,
    mu_hat: torch.Tensor,
    sigma_hat: torch.Tensor,
    dt_hat: float,
) -> None:
    with torch.no_grad():
        eps_min = float(epsilon_hat.min().item())
        eps_max = float(epsilon_hat.max().item())
        mu_min = float(mu_hat.min().item())
        mu_max = float(mu_hat.max().item())
        ratio = sigma_hat * dt_hat / torch.clamp(epsilon_hat, min=1e-12)
        ratio_min = float(ratio.min().item())
        ratio_max = float(ratio.max().item())

    if eps_min < 1.0 - 1e-5 or mu_min < 1.0 - 1e-5:
        warnings.warn(
            f"Nondim parameter range check: expected eps_hat, mu_hat >= 1, got "
            f"eps_hat_min={eps_min:.4e}, mu_hat_min={mu_min:.4e}.",
            RuntimeWarning,
        )
    if abs(dt_hat) > 1.0:
        warnings.warn(
            f"Nondim dt_hat={dt_hat:.4e} is large; verify CFL stability.",
            RuntimeWarning,
        )
    if ratio_max < 1e-5:
        warnings.warn(
            "sigma_hat * dt_hat / epsilon_hat is very small; attenuation may be "
            "weak in reduced precision.",
            RuntimeWarning,
        )
    if ratio_min < 0.0:
        warnings.warn(
            "Detected negative sigma_hat * dt_hat / epsilon_hat; verify conductivity input.",
            RuntimeWarning,
        )
    if eps_max > 1e4 or mu_max > 1e4:
        warnings.warn(
            f"Large nondim contrasts detected (eps_hat_max={eps_max:.4e}, "
            f"mu_hat_max={mu_max:.4e}); reduced precision may degrade.",
            RuntimeWarning,
        )


def maxwelltm(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int = 2,
    pml_width: Union[int, Sequence[int]] = 20,
    max_vel: Optional[float] = None,
    Ey_0: Optional[torch.Tensor] = None,
    Hx_0: Optional[torch.Tensor] = None,
    Hz_0: Optional[torch.Tensor] = None,
    m_Ey_x: Optional[torch.Tensor] = None,
    m_Ey_z: Optional[torch.Tensor] = None,
    m_Hx_z: Optional[torch.Tensor] = None,
    m_Hz_x: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    save_snapshots: Optional[bool] = None,
    forward_callback: Optional[Callback] = None,
    backward_callback: Optional[Callback] = None,
    callback_frequency: int = 1,
    python_backend: Union[bool, str] = False,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    n_threads: Optional[int] = None,
    compute_dtype: str = "fp32",
    mp_mode: str = "throughput",
):
    """2D TM mode Maxwell equations solver.

    This is the main entry point for Maxwell TM propagation. It automatically
    handles CFL condition checking and time step resampling when needed.

    If the user-provided time step (dt) is too large for numerical stability,
    the source signal will be upsampled internally and receiver data will be
    downsampled back to the original sampling rate.

    Args:
        epsilon: Relative permittivity tensor [ny, nx].
        sigma: Electrical conductivity tensor [ny, nx] in S/m.
        mu: Relative permeability tensor [ny, nx].
        grid_spacing: Grid spacing in meters. Single value or [dy, dx].
        dt: Time step in seconds.
        source_amplitude: Source waveform [n_shots, n_sources, nt].
        source_location: Source locations [n_shots, n_sources, 2].
        receiver_location: Receiver locations [n_shots, n_receivers, 2].
        stencil: FD stencil order (2, 4, 6, or 8).
        pml_width: PML width (single int or [top, bottom, left, right]).
        max_vel: Maximum wave velocity. If None, computed from model.
        Ey_0, Hx_0, Hz_0: Initial field values.
        m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x: Initial CPML memory variables.
        nt: Number of time steps (required if source_amplitude is None).
        model_gradient_sampling_interval: Interval for storing gradient snapshots.
            Values > 1 reduce memory usage during backpropagation.
        freq_taper_frac: Fraction of frequency spectrum to taper (0.0-1.0).
            Helps reduce ringing artifacts during resampling.
        time_pad_frac: Fraction for zero padding before FFT (0.0-1.0).
            Helps reduce wraparound artifacts during resampling.
        time_taper: Whether to apply Hann window (mainly for testing).
        save_snapshots: Whether to save wavefield snapshots for gradient computation.
            If None (default), snapshots are saved only when model parameters
            require gradients. Set to False to disable snapshot saving even
            when gradients are needed. Set to True to force snapshot saving
            even without gradients.
        forward_callback: Callback function called during forward propagation.
        backward_callback: Callback function called during backward (adjoint)
            propagation. Receives the same CallbackState as forward_callback,
            but with is_backward=True and gradients available.
        callback_frequency: How often to call the callback.
        python_backend: False for C/CUDA, True or 'eager'/'jit'/'compile' for Python.
        storage_mode: Where to store intermediate snapshots for the ASM
            backward pass. One of "device", "cpu", "disk", "none", or "auto".
        storage_path: Base path for disk storage when storage_mode="disk".
        storage_compression: Compression for stored snapshots. Use False/True
            (True == BF16), or one of "none" / "bf16".
        storage_bytes_limit_device: Soft limit in bytes for device snapshot
            storage when storage_mode="auto".
        storage_bytes_limit_host: Soft limit in bytes for host snapshot
            storage when storage_mode="auto".
        storage_chunk_steps: Optional chunk size (in stored steps) for
            CPU/disk modes. Currently unused.
        n_threads: OpenMP thread count for CPU backend. None uses the OpenMP default.
        compute_dtype: Compute precision mode. One of "fp32" (default) or
            "fp16" (CUDA only).
        mp_mode: Mixed precision policy used when compute_dtype="fp16".
            One of "throughput", "balanced", or "robust".

    Returns:
        Tuple of (Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x, receiver_amplitudes).
    """
    # Validate resampling parameters
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)
    compute_dtype_norm = _normalize_compute_dtype(compute_dtype)
    mp_mode_norm = _normalize_mp_mode(mp_mode)

    # MPS (Apple GPU) dtype guard: Metal does not support float64
    if epsilon.device.type == "mps" and epsilon.dtype == torch.float64:
        raise TypeError(
            "Apple MPS backend does not support float64 (double precision). "
            "Please use float32 tensors, e.g.: epsilon.float()"
        )
    if compute_dtype_norm == "fp16" and epsilon.device.type != "cuda":
        raise TypeError(
            "compute_dtype='fp16' is currently supported only on CUDA devices."
        )
    if compute_dtype_norm == "fp16":
        storage_kind = _normalize_storage_compression(storage_compression)
        if storage_kind != "none":
            raise ValueError(
                "compute_dtype='fp16' currently requires raw snapshot storage "
                "(storage_compression=False/'none')."
            )

    # Check inputs
    if source_location is not None and source_location.numel() > 0:
        if source_location[..., 0].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Source location dim 0 must be less than {epsilon.shape[-2]}"
            )
        if source_location[..., 1].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Source location dim 1 must be less than {epsilon.shape[-1]}"
            )

    if receiver_location is not None and receiver_location.numel() > 0:
        if receiver_location[..., 0].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Receiver location dim 0 must be less than {epsilon.shape[-2]}"
            )
        if receiver_location[..., 1].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Receiver location dim 1 must be less than {epsilon.shape[-1]}"
            )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    # Normalize grid_spacing to list
    grid_spacing_list = _normalize_grid_spacing_2d(grid_spacing)

    # Compute maximum velocity if not provided
    if max_vel is None:
        # For EM waves: v = c0 / sqrt(epsilon_r * mu_r)
        max_vel_computed = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    else:
        max_vel_computed = max_vel

    # Check CFL condition and compute step_ratio
    inner_dt, step_ratio = cfl_condition(grid_spacing_list, dt, max_vel_computed)

    # Upsample source if needed for CFL
    source_amplitude_internal = source_amplitude
    if step_ratio > 1 and source_amplitude is not None and source_amplitude.numel() > 0:
        source_amplitude_internal = upsample(
            source_amplitude,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )

    # Compute internal number of time steps
    nt_internal = None
    if nt is not None:
        nt_internal = nt * step_ratio
    elif source_amplitude_internal is not None:
        nt_internal = source_amplitude_internal.shape[-1]

    # Optional nondimensional scaling for mixed precision path
    nondim_ctx: Optional[NondimContext] = None
    epsilon_internal = epsilon
    sigma_internal = sigma
    mu_internal = mu
    source_amplitude_solver = source_amplitude_internal
    Ey_0_solver = Ey_0
    Hx_0_solver = Hx_0
    Hz_0_solver = Hz_0
    m_Ey_x_solver = m_Ey_x
    m_Ey_z_solver = m_Ey_z
    m_Hx_z_solver = m_Hx_z
    m_Hz_x_solver = m_Hz_x
    grid_spacing_solver: Union[float, Sequence[float]] = grid_spacing
    dt_solver = inner_dt
    max_vel_solver = max_vel_computed
    pml_eps_scale = None

    if compute_dtype_norm == "fp16":
        nondim_ctx = build_nondim_context(epsilon, mu, grid_spacing_list, inner_dt)
        epsilon_internal = epsilon / nondim_ctx.eps_ref_r
        mu_internal = mu / nondim_ctx.mu_ref_r
        sigma_internal = sigma * (nondim_ctx.T0 / nondim_ctx.eps_ref_abs)

        _warn_nondim_ranges(
            epsilon_internal,
            mu_internal,
            sigma_internal,
            nondim_ctx.dt_hat,
        )

        if source_amplitude_internal is not None and source_amplitude_internal.numel() > 0:
            source_amplitude_solver = source_amplitude_internal * nondim_ctx.source_scale

        (
            Ey_0_solver,
            Hx_0_solver,
            Hz_0_solver,
            m_Ey_x_solver,
            m_Ey_z_solver,
            m_Hx_z_solver,
            m_Hz_x_solver,
        ) = _scale_tm_states_to_nondim(
            Ey_0,
            Hx_0,
            Hz_0,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            nondim_ctx,
        )

        grid_spacing_solver = [nondim_ctx.dy_hat, nondim_ctx.dx_hat]
        dt_solver = nondim_ctx.dt_hat
        cmax = nondim_ctx.L0 / nondim_ctx.T0
        max_vel_solver = max_vel_computed / cmax
        pml_eps_scale = 1.0

        solver_dtype: torch.dtype
        if mp_mode_norm in {"throughput", "balanced"}:
            solver_dtype = torch.float16
        else:
            solver_dtype = torch.float32

        epsilon_internal = epsilon_internal.to(solver_dtype)
        sigma_internal = sigma_internal.to(solver_dtype)
        mu_internal = mu_internal.to(solver_dtype)
        if source_amplitude_solver is not None and source_amplitude_solver.numel() > 0:
            source_amplitude_solver = source_amplitude_solver.to(solver_dtype)
        if Ey_0_solver is not None:
            Ey_0_solver = Ey_0_solver.to(solver_dtype)
        if Hx_0_solver is not None:
            Hx_0_solver = Hx_0_solver.to(solver_dtype)
        if Hz_0_solver is not None:
            Hz_0_solver = Hz_0_solver.to(solver_dtype)
        if m_Ey_x_solver is not None:
            m_Ey_x_solver = m_Ey_x_solver.to(solver_dtype)
        if m_Ey_z_solver is not None:
            m_Ey_z_solver = m_Ey_z_solver.to(solver_dtype)
        if m_Hx_z_solver is not None:
            m_Hx_z_solver = m_Hx_z_solver.to(solver_dtype)
        if m_Hz_x_solver is not None:
            m_Hz_x_solver = m_Hz_x_solver.to(solver_dtype)

    # Call the propagation function with internal dt and upsampled source
    result = maxwell_func(
        python_backend,
        epsilon_internal,
        sigma_internal,
        mu_internal,
        grid_spacing_solver,
        dt_solver,
        source_amplitude_solver,
        source_location,
        receiver_location,
        stencil,
        pml_width,
        max_vel_solver,
        Ey_0_solver,
        Hx_0_solver,
        Hz_0_solver,
        m_Ey_x_solver,
        m_Ey_z_solver,
        m_Hx_z_solver,
        m_Hz_x_solver,
        nt_internal,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        forward_callback,
        backward_callback,
        callback_frequency,
        storage_mode,
        storage_path,
        storage_compression,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        n_threads,
        compute_dtype=compute_dtype_norm,
        mp_mode=mp_mode_norm,
        pml_eps_scale=pml_eps_scale,
    )

    # Unpack result
    (
        Ey_out,
        Hx_out,
        Hz_out,
        m_Ey_x_out,
        m_Ey_z_out,
        m_Hx_z_out,
        m_Hz_x_out,
        receiver_amplitudes,
    ) = result

    # Downsample receiver data if we upsampled
    if step_ratio > 1 and receiver_amplitudes.numel() > 0:
        receiver_amplitudes = downsample_and_movedim(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
        # Move time back to first dimension to match expected output format
        receiver_amplitudes = torch.movedim(receiver_amplitudes, -1, 0)

    if nondim_ctx is not None:
        (
            Ey_out,
            Hx_out,
            Hz_out,
            m_Ey_x_out,
            m_Ey_z_out,
            m_Hx_z_out,
            m_Hz_x_out,
        ) = _restore_tm_states_from_nondim(
            Ey_out,
            Hx_out,
            Hz_out,
            m_Ey_x_out,
            m_Ey_z_out,
            m_Hx_z_out,
            m_Hz_x_out,
            nondim_ctx,
        )

    return (
        Ey_out,
        Hx_out,
        Hz_out,
        m_Ey_x_out,
        m_Ey_z_out,
        m_Hx_z_out,
        m_Hz_x_out,
        receiver_amplitudes,
    )


_update_E_jit: Optional[Callable] = None
_update_E_compile: Optional[Callable] = None
_update_H_jit: Optional[Callable] = None
_update_H_compile: Optional[Callable] = None

# These will be set after the functions are defined
_update_E_opt: Optional[Callable] = None
_update_H_opt: Optional[Callable] = None


def maxwell_func(
    python_backend: Union[bool, str],
    *args,
    compute_dtype: str = "fp32",
    mp_mode: str = "throughput",
    pml_eps_scale: Optional[float] = None,
) -> tuple[
    torch.Tensor,  # Ey
    torch.Tensor,  # Hx
    torch.Tensor,  # Hz
    torch.Tensor,  # m_Ey_x
    torch.Tensor,  # m_Ey_z
    torch.Tensor,  # m_Hx_z
    torch.Tensor,  # m_Hz_x
    torch.Tensor,  # receiver_amplitudes
]:
    """Dispatch to Python or C/CUDA backend for Maxwell propagation."""
    global _update_E_jit, _update_E_compile, _update_E_opt
    global _update_H_jit, _update_H_compile, _update_H_opt

    # Detect device from the first positional arg (epsilon tensor)
    _device_type = args[0].device.type if len(args) > 0 and isinstance(args[0], torch.Tensor) else "cpu"

    # Check if we should use Python backend or C/CUDA backend
    use_python = python_backend

    # MPS (Apple GPU): prefer the compiled Metal backend when available.
    # Falls back to the Python/PyTorch backend only when Metal kernels are
    # not compiled into the shared library.
    if _device_type == "mps" and not use_python:
        try:
            from . import backend_utils

            if not backend_utils.metal_available():
                use_python = True
        except ImportError:
            use_python = True

    if not use_python:
        # Try to use C/CUDA backend
        try:
            from . import backend_utils

            if not backend_utils.is_backend_available():
                import warnings

                warnings.warn(
                    "C/CUDA backend not available, falling back to Python backend. "
                    "To use the C/CUDA backend, compile the library first.",
                    RuntimeWarning,
                )
                use_python = True
        except ImportError:
            import warnings

            warnings.warn(
                "backend_utils not available, falling back to Python backend.",
                RuntimeWarning,
            )
            use_python = True

    if use_python:
        if python_backend is True or python_backend is False:
            mode = "eager"  # Default to eager
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

        return maxwell_python(
            *args,
            compute_dtype=compute_dtype,
            mp_mode=mp_mode,
            pml_eps_scale=pml_eps_scale,
        )
    else:
        # Use C/CUDA backend
        return maxwell_c_cuda(
            *args,
            compute_dtype=compute_dtype,
            mp_mode=mp_mode,
            pml_eps_scale=pml_eps_scale,
        )


def maxwell_python(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int,
    pml_width: Union[int, Sequence[int]],
    max_vel: Optional[float],
    Ey_0: Optional[torch.Tensor],
    Hx_0: Optional[torch.Tensor],
    Hz_0: Optional[torch.Tensor],
    m_Ey_x_0: Optional[torch.Tensor],
    m_Ey_z_0: Optional[torch.Tensor],
    m_Hx_z_0: Optional[torch.Tensor],
    m_Hz_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    n_threads: Optional[int] = None,
    compute_dtype: str = "fp32",
    mp_mode: str = "throughput",
    pml_eps_scale: Optional[float] = None,
):
    """Performs the forward propagation of the 2D TM Maxwell equations.

    This function implements the FDTD time-stepping loop for the TM mode
    (Ey, Hx, Hz) with CPML absorbing boundary conditions.

    - Models are padded by fd_pad + pml_width with replicate mode
    - Wavefields are padded by fd_pad only with zero padding
    - Output wavefields are cropped by fd_pad only (PML region is preserved)

    Args:
        epsilon: Permittivity model [ny, nx].
        sigma: Conductivity model [ny, nx].
        mu: Permeability model [ny, nx].
        grid_spacing: Grid spacing (dy, dx) or single value for both.
        dt: Time step.
        source_amplitude: Source amplitudes [n_shots, n_sources, nt].
        source_location: Source locations [n_shots, n_sources, 2].
        receiver_location: Receiver locations [n_shots, n_receivers, 2].
        stencil: Finite difference stencil order (2, 4, 6, or 8).
        pml_width: PML width on each side [top, bottom, left, right] or single value.
        max_vel: Maximum velocity for PML (if None, computed from model).
        Ey_0, Hx_0, Hz_0: Initial field values.
        m_Ey_x_0, m_Ey_z_0, m_Hx_z_0, m_Hz_x_0: Initial CPML memory variables.
        nt: Number of time steps (required if source_amplitude is None).
        model_gradient_sampling_interval: Interval for storing gradients.
        freq_taper_frac: Frequency taper fraction.
        time_pad_frac: Time padding fraction.
        time_taper: Whether to apply time taper.
        save_snapshots: Whether to save wavefield snapshots for backward pass.
            If None, determined by requires_grad on model parameters.
        forward_callback: Callback function called during propagation.
        callback_frequency: Frequency of callback calls.
    Returns:
        Tuple containing:
            - Ey: Final electric field [n_shots, ny + pml, nx + pml]
            - Hx, Hz: Final magnetic fields
            - m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x: Final CPML memory variables
            - receiver_amplitudes: Recorded data at receivers [nt, n_shots, n_receivers]
    """

    # These should be set by maxwell_func before calling this function
    assert _update_E_opt is not None, "_update_E_opt must be set by maxwell_func"
    assert _update_H_opt is not None, "_update_H_opt must be set by maxwell_func"
    # Validate inputs
    if epsilon.ndim != 2:
        raise RuntimeError("epsilon must be 2D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    device = epsilon.device
    dtype = epsilon.dtype
    model_ny, model_nx = epsilon.shape  # Original model dimensions

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

    # Normalize grid_spacing to list
    grid_spacing = _normalize_grid_spacing_2d(grid_spacing)
    dy, dx = grid_spacing

    # Normalize pml_width to list [top, bottom, left, right]
    pml_width_list = _normalize_pml_width_2d(pml_width)

    # Determine number of time steps
    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = source_amplitude.shape[-1]

    # Type cast to ensure nt is int for type checker
    nt_steps: int = int(nt)

    # Determine number of shots
    if source_amplitude is not None and source_amplitude.numel() > 0:
        n_shots = source_amplitude.shape[0]
    elif source_location is not None and source_location.numel() > 0:
        n_shots = source_location.shape[0]
    elif receiver_location is not None and receiver_location.numel() > 0:
        n_shots = receiver_location.shape[0]
    else:
        n_shots = 1

    # Compute maximum velocity for PML if not provided
    if max_vel is None:
        # For EM waves: v = c0 / sqrt(epsilon_r * mu_r)
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0

    # Compute PML frequency (dominant frequency estimate)
    pml_freq = 0.5 / dt  # Nyquist as default

    # =========================================================================
    # Padding strategy:
    # - fd_pad: padding for finite difference stencil accuracy
    # - pml_width: padding for PML absorbing layers
    # - Total model padding = fd_pad + pml_width
    # - Wavefield padding = fd_pad only (wavefields include PML region)
    # =========================================================================

    # FD padding based on stencil: accuracy // 2
    fd_pad = stencil // 2
    # fd_pad_list: [y0, y1, x0, x1] - for 2D staggered grid, asymmetric because
    # staggered diff a[1:] - a[:-1] reduces array size by 1, so we need fd_pad-1 at end
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]

    # Total padding for models = fd_pad + pml_width
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]

    # Calculate padded dimensions
    # Model is padded by total_pad on each side
    padded_ny = model_ny + total_pad[0] + total_pad[1]
    padded_nx = model_nx + total_pad[2] + total_pad[3]

    # Pad model tensors with replicate mode (extend boundary values)
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

    # Prepare update coefficients using padded models
    if compute_dtype == "fp16":
        ca, cb, cq = prepare_parameters_nondim(
            epsilon_padded, sigma_padded, mu_padded, dt
        )
    else:
        ca, cb, cq = prepare_parameters(epsilon_padded, sigma_padded, mu_padded, dt)

    # Expand coefficients for batch dimension
    ca = ca[None, :, :]  # [1, padded_ny, padded_nx]
    cb = cb[None, :, :]
    cq = cq[None, :, :]

    # =========================================================================
    # Initialize wavefields
    # Wavefields are padded by fd_pad only (they include the PML region)
    # Size = [n_shots, model_ny + pml_width*2 + fd_pad*2, model_nx + ...]
    # Which equals [n_shots, padded_ny, padded_nx]
    # =========================================================================
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

    # Zero out interior of PML auxiliary variables (optimization)
    # PML memory variables should only be non-zero in PML regions.
    # This works correctly even with user-provided initial states because:
    # 1. The output preserves PML region (only fd_pad is cropped)
    # 2. zero_interior only zeros the interior, preserving PML boundary values
    # 3. Interior values are already zero in correctly propagated wavefields
    # Dimension mapping for zero_interior:
    # - m_Ey_x: x-direction auxiliary -> dim=1 (zero y-interior, keep x-boundaries)
    # - m_Ey_z: y/z-direction auxiliary -> dim=0 (zero x-interior, keep y-boundaries)
    # - m_Hx_z: y/z-direction auxiliary -> dim=0
    # - m_Hz_x: x-direction auxiliary -> dim=1
    pml_aux_dims = [1, 0, 0, 1]  # [m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x]
    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], pml_aux_dims):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)

    # Set up PML profiles for the padded domain
    pml_profiles_list = staggered.set_pml_profiles(
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
        eps_scale=(
            pml_eps_scale
            if pml_eps_scale is not None
            else (1.0 if compute_dtype == "fp16" else EP0)
        ),
    )
    # pml_profiles_list = [ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh]
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
    ) = pml_profiles_list

    if compute_dtype == "fp16":
        b_min = min(float(by.min().item()), float(bx.min().item()))
        b_max = max(float(by.max().item()), float(bx.max().item()))
        if b_max >= 1.0:
            warnings.warn(
                f"CPML b profile max is {b_max:.6f} (expected < 1 for damping).",
                RuntimeWarning,
            )
        if b_min <= 0.0:
            warnings.warn(
                f"CPML b profile min is {b_min:.6f} (expected > 0).",
                RuntimeWarning,
            )

    # Reciprocal grid spacing
    rdy = torch.tensor(1.0 / dy, device=device, dtype=dtype)
    rdx = torch.tensor(1.0 / dx, device=device, dtype=dtype)
    dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

    # =========================================================================
    # Prepare source and receiver indices
    # Original positions are in the un-padded model coordinate system.
    # We need to offset by total_pad (fd_pad + pml_width) to get padded coords.
    # =========================================================================
    flat_model_shape = padded_ny * padded_nx

    if source_location is not None and source_location.numel() > 0:
        # Adjust source positions by total padding offset
        source_y = source_location[..., 0] + total_pad[0]  # Add top offset
        source_x = source_location[..., 1] + total_pad[2]  # Add left offset
        sources_i = (source_y * padded_nx + source_x).long()  # [n_shots, n_sources]
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        # Adjust receiver positions by total padding offset
        receiver_y = receiver_location[..., 0] + total_pad[0]  # Add top offset
        receiver_x = receiver_location[..., 1] + total_pad[2]  # Add left offset
        receivers_i = (receiver_y * padded_nx + receiver_x).long()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    # Initialize receiver amplitudes
    if n_receivers > 0:
        receiver_amplitudes = torch.zeros(
            nt_steps, n_shots, n_receivers, device=device, dtype=dtype
        )
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

    # Prepare callback data - models dict uses the padded models
    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca,
        "cb": cb,
        "cq": cq,
    }

    # Callback fd_pad is the actual fd_pad used for wavefields
    callback_fd_pad = fd_pad_list

    # Source injection coefficient: -cb * dt / (dx * dy)
    # Since our cb already contains dt/epsilon, we need: -cb / (dx * dy)
    # This normalizes the source by cell volume for correct amplitude
    source_coeff = -1.0 / (dx * dy)

    # Time stepping loop
    for step in range(nt_steps):
        # Callback at specified frequency
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
            # Create CallbackState for standardized interface
            callback_state = CallbackState(
                dt=dt,
                step=step,
                nt=nt_steps,
                wavefields=callback_wavefields,
                models=callback_models,
                gradients=None,  # No gradients during forward pass
                fd_pad=callback_fd_pad,
                pml_width=pml_width_list,
                is_backward=False,
                grid_spacing=[dy, dx],
            )
            forward_callback(callback_state)

        # Update H fields: H^{n+1/2} = H^{n-1/2} + ...
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

        # Update E field: E^{n+1} = E^n + ...
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

        # Inject source into Ey field (after E update, following reference implementation)
        # Source term: Ey += -cb * f * dt / (dx * dz) = -cb * f / (dx * dz) since cb contains dt
        if (
            source_amplitude is not None
            and source_amplitude.numel() > 0
            and n_sources > 0
        ):
            # source_amplitude: [n_shots, n_sources, nt]
            src_amp = source_amplitude[:, :, step]  # [n_shots, n_sources]
            # Get cb at source locations for proper scaling
            cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
            cb_at_src = cb_flat.gather(1, sources_i)  # [n_shots, n_sources]
            # Apply source with coefficient: -cb * f / (dx * dy)
            scaled_src = cb_at_src * src_amp * source_coeff
            Ey = (
                Ey.reshape(n_shots, flat_model_shape)
                .scatter_add(1, sources_i, scaled_src)
                .reshape(size_with_batch)
            )

        # Record at receivers (after source injection)
        if n_receivers > 0:
            receiver_amplitudes[step] = Ey.reshape(n_shots, flat_model_shape).gather(
                1, receivers_i
            )

    # =========================================================================
    # Output cropping:
    # Only remove fd_pad, keep the PML region in the output wavefields.
    # Output shape: [n_shots, model_ny + pml_width_y, model_nx + pml_width_x]
    # =========================================================================
    s = (
        slice(None),  # batch dimension
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Update electric field Ey with CPML absorbing boundary conditions.

    For TM mode, the update equation is:
        Ey^{n+1} = Ca * Ey^n + Cb * (dHz/dx - dHx/dz)

    With CPML, we split the derivatives and apply auxiliary variables:
        dHz/dx -> dHz/dx / kappa_x + m_Hz_x
        dHx/dz -> dHx/dz / kappa_y + m_Hx_z

    Args:
        ca, cb: Update coefficients from material parameters
        Hx, Hz: Magnetic field components
        Ey: Electric field component to update
        m_Hx_z, m_Hz_x: CPML auxiliary memory variables
        kappa_y, kappa_y_h: CPML kappa profiles in y direction
        kappa_x, kappa_x_h: CPML kappa profiles in x direction
        ay, ay_h, ax, ax_h: CPML a coefficients
        by, by_h, bx, bx_h: CPML b coefficients
        rdy, rdx: Reciprocal of grid spacing (1/dy, 1/dx)
        dt: Time step
        stencil: Finite difference stencil order (2, 4, 6, or 8)

    Returns:
        Updated Ey, m_Hx_z, m_Hz_x
    """

    # Compute spatial derivatives using staggered grid operators
    # dHz/dx at integer grid points (where Ey lives)
    dHz_dx = staggered.diffx1(Hz, stencil, rdx)
    # dHx/dz at integer grid points (where Ey lives)
    dHx_dz = staggered.diffy1(Hx, stencil, rdy)

    # Update CPML auxiliary variables using standard CPML recursion:
    # psi_new = b * psi_old + a * derivative
    # m_Hz_x stores the x-direction PML memory for Hz derivative
    m_Hz_x = bx * m_Hz_x + ax * dHz_dx
    # m_Hx_z stores the z-direction PML memory for Hx derivative
    m_Hx_z = by * m_Hx_z + ay * dHx_dz

    # Apply CPML correction to derivatives
    # In CPML: d/dx -> (1/kappa) * d/dx + m
    dHz_dx_pml = dHz_dx / kappa_x + m_Hz_x
    dHx_dz_pml = dHx_dz / kappa_y + m_Hx_z

    # Update Ey using the FDTD update equation
    # Ey^{n+1} = Ca * Ey^n + Cb * (dHz/dx - dHx/dz)
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Update magnetic fields Hx and Hz with CPML absorbing boundary conditions.

    For TM mode, the update equations are:
        Hx^{n+1} = Hx^n - Cq * dEy/dz
        Hz^{n+1} = Hz^n + Cq * dEy/dx

    With CPML, we use half-grid derivatives and auxiliary variables:
        dEy/dz -> dEy/dz / kappa_y_h + m_Ey_z
        dEy/dx -> dEy/dx / kappa_x_h + m_Ey_x

    Args:
        cq: Update coefficient (dt/mu)
        Hx, Hz: Magnetic field components to update
        Ey: Electric field component
        m_Ey_x, m_Ey_z: CPML auxiliary memory variables
        kappa_y, kappa_y_h: CPML kappa profiles in y direction (integer and half grid)
        kappa_x, kappa_x_h: CPML kappa profiles in x direction (integer and half grid)
        ay, ay_h, ax, ax_h: CPML a coefficients
        by, by_h, bx, bx_h: CPML b coefficients
        rdy, rdx: Reciprocal of grid spacing (1/dy, 1/dx)
        dt: Time step
        stencil: Finite difference stencil order (2, 4, 6, or 8)

    Returns:
        Updated Hx, Hz, m_Ey_x, m_Ey_z
    """

    # Compute spatial derivatives at half grid points (where H fields live)
    # dEy/dz at half grid points in z (for Hx update)
    dEy_dz = staggered.diffyh1(Ey, stencil, rdy)
    # dEy/dx at half grid points in x (for Hz update)
    dEy_dx = staggered.diffxh1(Ey, stencil, rdx)

    # Update CPML auxiliary variables using standard CPML recursion:
    # psi_new = b * psi_old + a * derivative
    # m_Ey_z stores the z-direction PML memory for Ey derivative (used in Hx update)
    m_Ey_z = by_h * m_Ey_z + ay_h * dEy_dz
    # m_Ey_x stores the x-direction PML memory for Ey derivative (used in Hz update)
    m_Ey_x = bx_h * m_Ey_x + ax_h * dEy_dx

    # Apply CPML correction to derivatives
    # In CPML: d/dz -> (1/kappa_h) * d/dz + m
    dEy_dz_pml = dEy_dz / kappa_y_h + m_Ey_z
    dEy_dx_pml = dEy_dx / kappa_x_h + m_Ey_x

    # Update Hx using the FDTD update equation
    # Hx^{n+1} = Hx^n - Cq * dEy/dz
    Hx = Hx - cq * dEy_dz_pml

    # Update Hz using the FDTD update equation
    # Hz^{n+1} = Hz^n + Cq * dEy/dx
    Hz = Hz + cq * dEy_dx_pml

    return Hx, Hz, m_Ey_x, m_Ey_z


# Initialize the optimized function pointers to the default implementations
_update_E_opt = update_E
_update_H_opt = update_H


def maxwell_c_cuda(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int,
    pml_width: Union[int, Sequence[int]],
    max_vel: Optional[float],
    Ey_0: Optional[torch.Tensor],
    Hx_0: Optional[torch.Tensor],
    Hz_0: Optional[torch.Tensor],
    m_Ey_x_0: Optional[torch.Tensor],
    m_Ey_z_0: Optional[torch.Tensor],
    m_Hx_z_0: Optional[torch.Tensor],
    m_Hz_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    n_threads: Optional[int] = None,
    compute_dtype: str = "fp32",
    mp_mode: str = "throughput",
    pml_eps_scale: Optional[float] = None,
):
    """Performs Maxwell propagation using C/CUDA backend.

    This function provides the interface to the compiled C/CUDA implementations
    for high-performance wave propagation.

    Padding strategy:
    - Models are padded by fd_pad + pml_width with replicate mode
    - Wavefields are padded by fd_pad only with zero padding
    - Output wavefields are cropped by fd_pad only (PML region is preserved)

    Args:
        Same as maxwell_python.

    Returns:
        Same as maxwell_python.
    """
    from . import backend_utils

    # Validate inputs
    if epsilon.ndim != 2:
        raise RuntimeError("epsilon must be 2D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    device = epsilon.device
    dtype = epsilon.dtype
    _original_dtype = dtype
    model_ny, model_nx = epsilon.shape  # Original model dimensions

    # MPS (Apple GPU) bridging: The Metal C bridge creates its own MTLBuffers
    # internally, so we need tensors with valid CPU data_ptr() values.
    # MPS tensor data_ptr() returns Metal GPU buffer handles that cannot be
    # read from host code. Moving to CPU on Apple Silicon unified memory is
    # near-zero-cost (metadata change only, no data copy).
    _original_device = device
    # _backend_device: used for get_backend_function() symbol lookup.
    # For MPS, tensors live on CPU but we dispatch to the _mps symbols.
    _backend_device = device
    if device.type == "mps":
        device = torch.device("cpu")
        epsilon = epsilon.to(device)
        sigma = sigma.to(device)
        mu = mu.to(device)
        if source_amplitude is not None:
            source_amplitude = source_amplitude.to(device)
        if source_location is not None:
            source_location = source_location.to(device)
        if receiver_location is not None:
            receiver_location = receiver_location.to(device)
        if Ey_0 is not None:
            Ey_0 = Ey_0.to(device)
        if Hx_0 is not None:
            Hx_0 = Hx_0.to(device)
        if Hz_0 is not None:
            Hz_0 = Hz_0.to(device)
        if m_Ey_x_0 is not None:
            m_Ey_x_0 = m_Ey_x_0.to(device)
        if m_Ey_z_0 is not None:
            m_Ey_z_0 = m_Ey_z_0.to(device)
        if m_Hx_z_0 is not None:
            m_Hx_z_0 = m_Hx_z_0.to(device)
        if m_Hz_x_0 is not None:
            m_Hz_x_0 = m_Hz_x_0.to(device)

    _ = mp_mode

    # Normalize grid_spacing to list
    grid_spacing = _normalize_grid_spacing_2d(grid_spacing)
    dy, dx = grid_spacing

    n_threads_val = 0
    if n_threads is not None:
        n_threads_val = int(n_threads)
        if n_threads_val < 0:
            raise ValueError("n_threads must be >= 0 when provided.")

    # Normalize pml_width to list [top, bottom, left, right]
    pml_width_list = _normalize_pml_width_2d(pml_width)

    # Determine number of time steps
    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided")
        nt = source_amplitude.shape[-1]

    # Ensure nt is an integer for iteration
    nt_steps: int = int(nt)
    # Clamp gradient sampling interval to a sensible range for storage/backprop
    gradient_sampling_interval = int(model_gradient_sampling_interval)
    if gradient_sampling_interval < 1:
        gradient_sampling_interval = 1
    if nt_steps > 0:
        gradient_sampling_interval = min(gradient_sampling_interval, nt_steps)

    # Determine number of shots
    if source_amplitude is not None and source_amplitude.numel() > 0:
        n_shots = source_amplitude.shape[0]
    elif source_location is not None and source_location.numel() > 0:
        n_shots = source_location.shape[0]
    elif receiver_location is not None and receiver_location.numel() > 0:
        n_shots = receiver_location.shape[0]
    else:
        n_shots = 1

    # Compute maximum velocity for PML if not provided
    if max_vel is None:
        max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0

    # Compute PML frequency
    pml_freq = 0.5 / dt

    # =========================================================================
    # Padding strategy:
    # - fd_pad: padding for finite difference stencil accuracy
    # - pml_width: padding for PML absorbing layers
    # - Total model padding = fd_pad + pml_width
    # - Wavefield padding = fd_pad only (wavefields include PML region)
    # =========================================================================

    # FD padding based on stencil: accuracy // 2
    fd_pad = stencil // 2
    # fd_pad_list: [y0, y1, x0, x1] - asymmetric for staggered grid
    fd_pad_list = [fd_pad, fd_pad - 1, fd_pad, fd_pad - 1]

    # Total padding for models = fd_pad + pml_width
    total_pad = [fd + pml for fd, pml in zip(fd_pad_list, pml_width_list)]

    # Calculate padded dimensions
    padded_ny = model_ny + total_pad[0] + total_pad[1]
    padded_nx = model_nx + total_pad[2] + total_pad[3]

    # Pad model tensors with replicate mode (extend boundary values)
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

    # Prepare update coefficients using padded models
    if compute_dtype == "fp16":
        ca, cb, cq = prepare_parameters_nondim(
            epsilon_padded, sigma_padded, mu_padded, dt
        )
    else:
        ca, cb, cq = prepare_parameters(epsilon_padded, sigma_padded, mu_padded, dt)

    # Initialize fields with padded dimensions
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

    # Zero out interior of PML auxiliary variables (optimization)
    # This works correctly with user-provided states (see forward pass comments)
    pml_aux_dims = [1, 0, 0, 1]  # [m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x]
    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], pml_aux_dims):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)

    # Set up PML profiles for the padded domain
    pml_profiles_list = staggered.set_pml_profiles(
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
        eps_scale=(
            pml_eps_scale
            if pml_eps_scale is not None
            else (1.0 if compute_dtype == "fp16" else EP0)
        ),
    )
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
    ) = pml_profiles_list

    if compute_dtype == "fp16":
        b_min = min(float(by.min().item()), float(bx.min().item()))
        b_max = max(float(by.max().item()), float(bx.max().item()))
        if b_max >= 1.0:
            warnings.warn(
                f"CPML b profile max is {b_max:.6f} (expected < 1 for damping).",
                RuntimeWarning,
            )
        if b_min <= 0.0:
            warnings.warn(
                f"CPML b profile min is {b_min:.6f} (expected > 0).",
                RuntimeWarning,
            )

    # Flatten PML profiles for C backend (remove batch dimensions)
    ay_flat = ay.squeeze().contiguous()
    ay_h_flat = ay_h.squeeze().contiguous()
    ax_flat = ax.squeeze().contiguous()
    ax_h_flat = ax_h.squeeze().contiguous()
    by_flat = by.squeeze().contiguous()
    by_h_flat = by_h.squeeze().contiguous()
    bx_flat = bx.squeeze().contiguous()
    bx_h_flat = bx_h.squeeze().contiguous()

    # Flatten kappa profiles for C backend
    ky_flat = ky.squeeze().contiguous()
    ky_h_flat = ky_h.squeeze().contiguous()
    kx_flat = kx.squeeze().contiguous()
    kx_h_flat = kx_h.squeeze().contiguous()

    # =========================================================================
    # Prepare source and receiver indices
    # Original positions are in the un-padded model coordinate system.
    # We need to offset by total_pad (fd_pad + pml_width) to get padded coords.
    # =========================================================================
    flat_model_shape = padded_ny * padded_nx

    if source_location is not None and source_location.numel() > 0:
        # Adjust source positions by total padding offset
        source_y = source_location[..., 0] + total_pad[0]
        source_x = source_location[..., 1] + total_pad[2]
        sources_i = (source_y * padded_nx + source_x).long().contiguous()
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        # Adjust receiver positions by total padding offset
        receiver_y = receiver_location[..., 0] + total_pad[0]
        receiver_x = receiver_location[..., 1] + total_pad[2]
        receivers_i = (receiver_y * padded_nx + receiver_x).long().contiguous()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    # Prepare source amplitudes with proper scaling
    if source_amplitude is not None and source_amplitude.numel() > 0:
        source_coeff = -1.0 / (dx * dy)
        # Expand cb to batch dimension for gather
        cb_expanded = cb[None, :, :].expand(n_shots, -1, -1)
        cb_flat = cb_expanded.reshape(n_shots, flat_model_shape)
        cb_at_src = cb_flat.gather(1, sources_i)
        # Reshape source amplitude: [shot, source, time] -> [time, shot, source]
        f = source_amplitude.permute(2, 0, 1).contiguous()
        # Scale by cb and source coefficient
        f = f * cb_at_src[None, :, :] * source_coeff
        f = f.reshape(nt_steps * n_shots * n_sources)
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    # Flatten fields for C backend
    Ey = Ey.contiguous()
    Hx = Hx.contiguous()
    Hz = Hz.contiguous()
    m_Ey_x = m_Ey_x.contiguous()
    m_Ey_z = m_Ey_z.contiguous()
    m_Hx_z = m_Hx_z.contiguous()
    m_Hz_x = m_Hz_x.contiguous()

    # Flatten coefficients (add batch dimension for consistency)
    ca = ca[None, :, :].contiguous()
    cb = cb[None, :, :].contiguous()
    cq = cq[None, :, :].contiguous()

    # PML boundaries (where PML starts in the padded domain)
    pml_y0 = fd_pad_list[0] + pml_width_list[0]
    pml_y1 = padded_ny - fd_pad_list[1] - pml_width_list[1]
    pml_x0 = fd_pad_list[2] + pml_width_list[2]
    pml_x1 = padded_nx - fd_pad_list[3] - pml_width_list[3]

    # Determine if any input requires gradients
    requires_grad = epsilon.requires_grad or sigma.requires_grad

    functorch_active = torch._C._are_functorch_transforms_active()
    if functorch_active:
        raise NotImplementedError(
            "torch.func transforms are not supported for the C/CUDA backend."
        )

    storage_kind, _, storage_bytes_per_elem = _resolve_storage_compression(
        storage_compression,
        dtype,
        device,
        context="storage_compression",
    )

    # Determine if we should save snapshots for backward pass
    if save_snapshots is None:
        do_save_snapshots = requires_grad
    else:
        do_save_snapshots = save_snapshots

    # If save_snapshots is False but requires_grad is True, warn user
    if requires_grad and save_snapshots is False:
        import warnings

        warnings.warn(
            "save_snapshots=False but model parameters require gradients. "
            "Backward pass will fail.",
            UserWarning,
        )

    storage_mode_str = storage_mode.lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )
    if device.type == "cpu" and storage_mode_str == "cpu":
        storage_mode_str = "device"

    needs_storage = do_save_snapshots and requires_grad
    effective_storage_mode_str = storage_mode_str
    if not needs_storage:
        if effective_storage_mode_str == "auto":
            effective_storage_mode_str = "none"
    else:
        if effective_storage_mode_str == "none":
            raise ValueError(
                "storage_mode='none' is not compatible with gradient computation "
                "when model parameters require gradients."
            )
        if effective_storage_mode_str == "auto":
            dtype_size = storage_bytes_per_elem
            # Estimate required bytes for storing Ey and curl_H.
            num_elements_per_shot = padded_ny * padded_nx
            shot_bytes_uncomp = num_elements_per_shot * dtype_size
            n_stored = (
                nt_steps + gradient_sampling_interval - 1
            ) // gradient_sampling_interval
            total_bytes = n_stored * n_shots * shot_bytes_uncomp * 2  # Ey + curl_H

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
            import warnings

            if device.type == "cuda" and total_bytes <= limit_device:
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

    # Callback fd_pad is the actual fd_pad used for wavefields
    callback_fd_pad = fd_pad_list

    # Callback models dict
    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca,
        "cb": cb,
        "cq": cq,
        "compute_dtype": compute_dtype,
    }

    use_autograd_fn = (
        requires_grad
        and do_save_snapshots
    ) or functorch_active
    if use_autograd_fn:
        # Use autograd Function for gradient computation
        result = MaxwellTMForwardFunc.apply(
            ca,
            cb,
            cq,
            f,
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
            1.0 / dy,  # rdy
            1.0 / dx,  # rdx
            dt,
            nt_steps,
            n_shots,
            padded_ny,
            padded_nx,
            n_sources,
            n_receivers,
            gradient_sampling_interval,  # step_ratio
            stencil,  # accuracy
            False,  # ca_batched
            False,  # cb_batched
            False,  # cq_batched
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            tuple(fd_pad_list),  # fd_pad for callback
            tuple(pml_width_list),  # pml_width for callback
            callback_models,  # models dict for callback
            forward_callback,
            backward_callback,
            callback_frequency,
            effective_storage_mode_str,
            storage_path,
            storage_compression,
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            n_threads_val,
            _backend_device,
        )
        # Unpack result (drop context handle if present)
        if len(result) == 9:
            (
                Ey_out,
                Hx_out,
                Hz_out,
                m_Ey_x_out,
                m_Ey_z_out,
                m_Hx_z_out,
                m_Hz_x_out,
                receiver_amplitudes,
                _ctx_handle,
            ) = result
        else:
            (
                Ey_out,
                Hx_out,
                Hz_out,
                m_Ey_x_out,
                m_Ey_z_out,
                m_Hx_z_out,
                m_Hz_x_out,
                receiver_amplitudes,
            ) = result
        # Output cropping: only remove fd_pad, keep PML region
        s = (
            slice(None),  # batch dimension
            slice(
                fd_pad_list[0],
                padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None,
            ),
            slice(
                fd_pad_list[2],
                padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None,
            ),
        )

        result = (
            Ey_out[s],
            Hx_out[s],
            Hz_out[s],
            m_Ey_x_out[s],
            m_Ey_z_out[s],
            m_Hx_z_out[s],
            m_Hz_x_out[s],
            receiver_amplitudes,
        )
        if _original_device != device or _original_dtype != dtype:
            result = tuple(
                t.to(device=_original_device, dtype=_original_dtype)
                if t.is_floating_point()
                else t.to(device=_original_device)
                for t in result
            )
        return result
    else:
        # Direct call without autograd for inference
        # Get the backend function
        try:
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward", stencil, dtype, _backend_device
            )
        except AttributeError as e:
            raise RuntimeError(
                f"C/CUDA backend function not available for accuracy={stencil}, "
                f"dtype={dtype}, device={device}. Error: {e}"
            )

        # Get device index for CUDA
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        # Initialize receiver amplitudes
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt_steps, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        # If no callback is provided, run entire propagation in single call
        # Otherwise, chunk into callback_frequency steps
        if forward_callback is None:
            effective_callback_freq = nt_steps
        else:
            effective_callback_freq = callback_frequency

        # Main time-stepping loop with chunked calls for callback support
        for step in range(0, nt_steps, effective_callback_freq):
            # Call callback at the start of each chunk
            if forward_callback is not None:
                callback_wavefields = {
                    "Ey": Ey,
                    "Hx": Hx,
                    "Hz": Hz,
                    "m_Ey_x": m_Ey_x,
                    "m_Ey_z": m_Ey_z,
                    "m_Hx_z": m_Hx_z,
                    "m_Hz_x": m_Hz_x,
                }
                callback_state = CallbackState(
                    dt=dt,
                    step=step,
                    nt=nt_steps,
                    wavefields=callback_wavefields,
                    models=callback_models,
                    gradients=None,
                    fd_pad=callback_fd_pad,
                    pml_width=pml_width_list,
                    is_backward=False,
                    grid_spacing=[dy, dx],
                )
                forward_callback(callback_state)

            # Number of steps to propagate in this chunk
            step_nt = min(nt_steps - step, effective_callback_freq)

            # Call the C/CUDA function for this chunk
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(f),
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_Ey_x),
                backend_utils.tensor_to_ptr(m_Ey_z),
                backend_utils.tensor_to_ptr(m_Hx_z),
                backend_utils.tensor_to_ptr(m_Hz_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
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
                1.0 / dy,  # rdy
                1.0 / dx,  # rdx
                dt,
                step_nt,  # nt for this chunk
                n_shots,
                padded_ny,
                padded_nx,
                n_sources,
                n_receivers,
                gradient_sampling_interval,  # step_ratio
                False,  # ca_batched
                False,  # cb_batched
                False,  # cq_batched
                step,  # start_t - crucial for correct source injection timing
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                n_threads_val,
                device_idx,
            )

        # Output cropping: only remove fd_pad, keep PML region
        s = (
            slice(None),  # batch dimension
            slice(
                fd_pad_list[0],
                padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None,
            ),
            slice(
                fd_pad_list[2],
                padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None,
            ),
        )

        result = (
            Ey[s],
            Hx[s],
            Hz[s],
            m_Ey_x[s],
            m_Ey_z[s],
            m_Hx_z[s],
            m_Hz_x[s],
            receiver_amplitudes,
        )
        if _original_device != device or _original_dtype != dtype:
            result = tuple(
                t.to(device=_original_device, dtype=_original_dtype)
                if t.is_floating_point()
                else t.to(device=_original_device)
                for t in result
            )
        return result


class MaxwellTMForwardFunc(torch.autograd.Function):
    """Autograd function for the forward pass of Maxwell TM wave propagation.

    This class defines the forward and backward passes for the 2D TM mode
    Maxwell equations, allowing PyTorch to compute gradients through the wave
    propagation operation. It interfaces directly with the C/CUDA backend.

    The backward pass uses the Adjoint State Method (ASM) which requires
    storing forward wavefield values at each step_ratio interval for
    gradient computation.
    """

    @staticmethod
    def forward(
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        ay: torch.Tensor,
        by: torch.Tensor,
        ay_h: torch.Tensor,
        by_h: torch.Tensor,
        ax: torch.Tensor,
        bx: torch.Tensor,
        ax_h: torch.Tensor,
        bx_h: torch.Tensor,
        ky: torch.Tensor,
        ky_h: torch.Tensor,
        kx: torch.Tensor,
        kx_h: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        rdy: float,
        rdx: float,
        dt: float,
        nt: int,
        n_shots: int,
        ny: int,
        nx: int,
        n_sources: int,
        n_receivers: int,
        step_ratio: int,
        accuracy: int,
        ca_batched: bool,
        cb_batched: bool,
        cq_batched: bool,
        pml_y0: int,
        pml_x0: int,
        pml_y1: int,
        pml_x1: int,
        fd_pad: tuple[int, int, int, int],
        pml_width: tuple[int, int, int, int],
        models: dict,
        forward_callback: Optional[Callback],
        backward_callback: Optional[Callback],
        callback_frequency: int,
        storage_mode_str: str,
        storage_path: str,
        storage_compression: Union[bool, str],
        Ey: torch.Tensor,
        Hx: torch.Tensor,
        Hz: torch.Tensor,
        m_Ey_x: torch.Tensor,
        m_Ey_z: torch.Tensor,
        m_Hx_z: torch.Tensor,
        m_Hz_x: torch.Tensor,
        n_threads: int,
        backend_device: torch.device,
    ) -> tuple[Any, ...]:
        """Performs the forward propagation of the Maxwell TM equations."""
        from . import backend_utils

        device = Ey.device
        dtype = Ey.dtype
        _backend_device = backend_device

        ca_requires_grad = ca.requires_grad
        cb_requires_grad = cb.requires_grad
        needs_grad = ca_requires_grad or cb_requires_grad

        # Initialize receiver amplitudes
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        # Get device index for CUDA
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        backward_storage_tensors: list[torch.Tensor] = []
        backward_storage_objects: list[Any] = []
        backward_storage_filename_arrays: list[Any] = []
        storage_mode = STORAGE_NONE
        shot_bytes_uncomp = 0

        if needs_grad:
            import ctypes

            # Resolve storage mode and sizes
            if str(device) == "cpu" and storage_mode_str == "cpu":
                storage_mode_str = "device"
            storage_mode = storage_mode_to_int(storage_mode_str)

            num_elements_per_shot = ny * nx
            _, store_dtype, _ = _resolve_storage_compression(
                storage_compression,
                dtype,
                device,
                context="storage_compression",
            )

            shot_bytes_uncomp = num_elements_per_shot * store_dtype.itemsize

            num_steps_stored = (nt + step_ratio - 1) // step_ratio

            # Storage buffers and filename arrays (mirrors Deepwave)
            char_ptr_type = ctypes.c_char_p
            is_cuda = device.type == "cuda"

            def alloc_storage(requires_grad_cond: bool):
                store_1 = torch.empty(0)
                store_3 = torch.empty(0)
                filenames_arr = (char_ptr_type * 0)()

                if requires_grad_cond and storage_mode != STORAGE_NONE:
                    if storage_mode == STORAGE_DEVICE:
                        store_1 = torch.empty(
                            num_steps_stored,
                            n_shots,
                            ny,
                            nx,
                            device=device,
                            dtype=store_dtype,
                        )
                    elif storage_mode == STORAGE_CPU:
                        # Multi-buffer device staging to overlap D2H copies.
                        store_1 = torch.empty(
                            _CPU_STORAGE_BUFFERS,
                            n_shots,
                            ny,
                            nx,
                            device=device,
                            dtype=store_dtype,
                        )
                        store_3 = torch.empty(
                            num_steps_stored,
                            n_shots,
                            shot_bytes_uncomp // store_dtype.itemsize,
                            device="cpu",
                            pin_memory=True,
                            dtype=store_dtype,
                        )
                    elif storage_mode == STORAGE_DISK:
                        storage_obj = TemporaryStorage(
                            storage_path, 1 if is_cuda else n_shots
                        )
                        backward_storage_objects.append(storage_obj)
                        filenames_list = [
                            f.encode("utf-8") for f in storage_obj.get_filenames()
                        ]
                        filenames_arr = (char_ptr_type * len(filenames_list))()
                        for i_file, f_name in enumerate(filenames_list):
                            filenames_arr[i_file] = ctypes.cast(
                                ctypes.create_string_buffer(f_name), char_ptr_type
                            )

                        store_1 = torch.empty(
                            n_shots, ny, nx, device=device, dtype=store_dtype
                        )
                        if is_cuda:
                            store_3 = torch.empty(
                                n_shots,
                                shot_bytes_uncomp // store_dtype.itemsize,
                                device="cpu",
                                pin_memory=True,
                                dtype=store_dtype,
                            )

                backward_storage_tensors.extend([store_1, store_3])
                backward_storage_filename_arrays.append(filenames_arr)

                filenames_ptr = (
                    ctypes.cast(filenames_arr, ctypes.c_void_p)
                    if storage_mode == STORAGE_DISK
                    else 0
                )

                return store_1, store_3, filenames_ptr

            ey_store_1, ey_store_3, ey_filenames_ptr = alloc_storage(ca_requires_grad)
            curl_store_1, curl_store_3, curl_filenames_ptr = alloc_storage(
                cb_requires_grad
            )

            # Get the backend function with storage
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward_with_storage", accuracy, dtype, _backend_device
            )

            # Determine effective callback frequency
            if forward_callback is None:
                effective_callback_freq = nt // step_ratio
            else:
                effective_callback_freq = callback_frequency

            # Chunked forward propagation with callback support
            for step in range(0, nt // step_ratio, effective_callback_freq):
                step_nt = (
                    min(effective_callback_freq, nt // step_ratio - step) * step_ratio
                )
                start_t = step * step_ratio

                # Call the C/CUDA function with storage for this chunk
                forward_func(
                    backend_utils.tensor_to_ptr(ca),
                    backend_utils.tensor_to_ptr(cb),
                    backend_utils.tensor_to_ptr(cq),
                    backend_utils.tensor_to_ptr(source_amplitudes_scaled),
                    backend_utils.tensor_to_ptr(Ey),
                    backend_utils.tensor_to_ptr(Hx),
                    backend_utils.tensor_to_ptr(Hz),
                    backend_utils.tensor_to_ptr(m_Ey_x),
                    backend_utils.tensor_to_ptr(m_Ey_z),
                    backend_utils.tensor_to_ptr(m_Hx_z),
                    backend_utils.tensor_to_ptr(m_Hz_x),
                    backend_utils.tensor_to_ptr(receiver_amplitudes),
                    backend_utils.tensor_to_ptr(ey_store_1),
                    backend_utils.tensor_to_ptr(ey_store_3),
                    ey_filenames_ptr,
                    backend_utils.tensor_to_ptr(curl_store_1),
                    backend_utils.tensor_to_ptr(curl_store_3),
                    curl_filenames_ptr,
                    backend_utils.tensor_to_ptr(ay),
                    backend_utils.tensor_to_ptr(by),
                    backend_utils.tensor_to_ptr(ay_h),
                    backend_utils.tensor_to_ptr(by_h),
                    backend_utils.tensor_to_ptr(ax),
                    backend_utils.tensor_to_ptr(bx),
                    backend_utils.tensor_to_ptr(ax_h),
                    backend_utils.tensor_to_ptr(bx_h),
                    backend_utils.tensor_to_ptr(ky),
                    backend_utils.tensor_to_ptr(ky_h),
                    backend_utils.tensor_to_ptr(kx),
                    backend_utils.tensor_to_ptr(kx_h),
                    backend_utils.tensor_to_ptr(sources_i),
                    backend_utils.tensor_to_ptr(receivers_i),
                    rdy,
                    rdx,
                    dt,
                    step_nt,  # number of steps in this chunk
                    n_shots,
                    ny,
                    nx,
                    n_sources,
                    n_receivers,
                    step_ratio,
                    storage_mode,
                    shot_bytes_uncomp,
                    ca_requires_grad,
                    cb_requires_grad,
                    ca_batched,
                    cb_batched,
                    cq_batched,
                    start_t,  # starting time step
                    pml_y0,
                    pml_x0,
                    pml_y1,
                    pml_x1,
                    n_threads,
                    device_idx,
                )

                # Call forward callback after each chunk
                if forward_callback is not None:
                    callback_wavefields = {
                        "Ey": Ey,
                        "Hx": Hx,
                        "Hz": Hz,
                        "m_Ey_x": m_Ey_x,
                        "m_Ey_z": m_Ey_z,
                        "m_Hx_z": m_Hx_z,
                        "m_Hz_x": m_Hz_x,
                    }
                    forward_callback(
                        CallbackState(
                            dt=dt,
                            step=step + step_nt // step_ratio,
                            nt=nt // step_ratio,
                            wavefields=callback_wavefields,
                            models=models,
                            gradients={},
                            fd_pad=list(fd_pad),
                            pml_width=list(pml_width),
                            is_backward=False,
                        )
                    )
        else:
            # Use regular forward without storage
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward", accuracy, dtype, _backend_device
            )

            # Call the C/CUDA function
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(source_amplitudes_scaled),
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_Ey_x),
                backend_utils.tensor_to_ptr(m_Ey_z),
                backend_utils.tensor_to_ptr(m_Hx_z),
                backend_utils.tensor_to_ptr(m_Hz_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                rdy,
                rdx,
                dt,
                nt,
                n_shots,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                ca_batched,
                cb_batched,
                cq_batched,
                0,  # start_t
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                n_threads,
                device_idx,
            )

        ctx_data = {
            "backward_storage_tensors": backward_storage_tensors,
            "backward_storage_objects": backward_storage_objects,
            "backward_storage_filename_arrays": backward_storage_filename_arrays,
            "storage_mode": storage_mode,
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "source_amplitudes_scaled": source_amplitudes_scaled,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
        }
        ctx_handle = _register_ctx_handle(ctx_data)

        return (
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            receiver_amplitudes,
            ctx_handle,
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        (
            ca,
            cb,
            cq,
            _source_amplitudes_scaled,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            rdy,
            rdx,
            dt,
            nt,
            n_shots,
            ny,
            nx,
            n_sources,
            n_receivers,
            step_ratio,
            accuracy,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            fd_pad,
            pml_width,
            models,
            _forward_callback,
            backward_callback,
            callback_frequency,
            _storage_mode_str,
            _storage_path,
            _storage_compression,
            _Ey,
            _Hx,
            _Hz,
            _m_Ey_x,
            _m_Ey_z,
            _m_Hx_z,
            _m_Hz_x,
            n_threads,
            _backend_device,
        ) = inputs

        outputs = output if isinstance(output, tuple) else (output,)
        if len(outputs) != 9:
            raise RuntimeError(
                "MaxwellTMForwardFunc expected a context handle output for setup_context."
            )
        ctx_handle = outputs[-1]
        if not isinstance(ctx_handle, torch.Tensor):
            raise RuntimeError("MaxwellTMForwardFunc context handle must be a Tensor.")

        ctx_handle_id = int(ctx_handle.item())
        ctx_data = _get_ctx_handle(ctx_handle_id)
        ctx._ctx_handle_id = ctx_handle_id
        backward_storage_tensors = ctx_data["backward_storage_tensors"]

        ctx.save_for_backward(
            ca,
            cb,
            cq,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            *backward_storage_tensors,
        )
        ctx.save_for_forward(
            ca,
            cb,
            cq,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
        )
        ctx.backward_storage_objects = ctx_data["backward_storage_objects"]
        ctx.backward_storage_filename_arrays = ctx_data[
            "backward_storage_filename_arrays"
        ]
        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.ny = ny
        ctx.nx = nx
        ctx.n_sources = n_sources
        ctx.n_receivers = n_receivers
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.pml_y0 = pml_y0
        ctx.pml_x0 = pml_x0
        ctx.pml_y1 = pml_y1
        ctx.pml_x1 = pml_x1
        ctx.ca_requires_grad = ctx_data["ca_requires_grad"]
        ctx.cb_requires_grad = ctx_data["cb_requires_grad"]
        ctx.storage_mode = ctx_data["storage_mode"]
        ctx.shot_bytes_uncomp = ctx_data["shot_bytes_uncomp"]
        ctx.fd_pad = fd_pad
        ctx.pml_width = pml_width
        ctx.models = models
        ctx.backward_callback = backward_callback
        ctx.callback_frequency = callback_frequency
        ctx.source_amplitudes_scaled = ctx_data["source_amplitudes_scaled"]
        ctx.n_threads = n_threads
        ctx._backend_device = _backend_device

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Computes the gradients during the backward pass using ASM.

        Uses the Adjoint State Method (ASM) to compute gradients:
        - grad_ca = sum_t (E_y^n * lambda_Ey^{n+1})
        - grad_cb = sum_t (curl_H^n * lambda_Ey^{n+1})

        Args:
            ctx: A context object containing information saved during forward.
            grad_outputs: Gradients of the loss with respect to the outputs.

        Returns:
            Gradients with respect to the inputs of the forward pass.
        """
        from . import backend_utils

        grad_outputs_list = list(grad_outputs)
        if len(grad_outputs_list) == 9:
            grad_outputs_list.pop()  # drop context handle grad

        # Unpack grad_outputs
        (
            grad_Ey,
            grad_Hx,
            grad_Hz,
            grad_m_Ey_x,
            grad_m_Ey_z,
            grad_m_Hx_z,
            grad_m_Hz_x,
            grad_r,
        ) = grad_outputs_list

        # Retrieve saved tensors
        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        ay, by, ay_h, by_h = saved[3], saved[4], saved[5], saved[6]
        ax, bx, ax_h, bx_h = saved[7], saved[8], saved[9], saved[10]
        ky, ky_h, kx, kx_h = saved[11], saved[12], saved[13], saved[14]
        sources_i, receivers_i = saved[15], saved[16]
        ey_store_1, ey_store_3 = saved[17], saved[18]
        curl_store_1, curl_store_3 = saved[19], saved[20]

        device = ca.device
        dtype = ca.dtype
        _backend_device = ctx._backend_device

        rdy = ctx.rdy
        rdx = ctx.rdx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        ny = ctx.ny
        nx = ctx.nx
        n_sources = ctx.n_sources
        n_receivers = ctx.n_receivers
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        ca_batched = ctx.ca_batched
        cb_batched = ctx.cb_batched
        cq_batched = ctx.cq_batched
        pml_y0_fwd = ctx.pml_y0
        pml_x0_fwd = ctx.pml_x0
        pml_y1_fwd = ctx.pml_y1
        pml_x1_fwd = ctx.pml_x1
        ca_requires_grad = ctx.ca_requires_grad
        cb_requires_grad = ctx.cb_requires_grad
        pml_width = ctx.pml_width
        storage_mode = ctx.storage_mode
        shot_bytes_uncomp = ctx.shot_bytes_uncomp

        import ctypes

        if storage_mode == STORAGE_DISK:
            ey_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[0], ctypes.c_void_p
            )
            curl_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[1], ctypes.c_void_p
            )
        else:
            ey_filenames_ptr = 0
            curl_filenames_ptr = 0

        # Ensure grad_r is contiguous
        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            grad_r = grad_r.contiguous()

        # Initialize adjoint fields (lambda fields)
        lambda_ey = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hx = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hz = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        # Initialize adjoint PML memory variables
        m_lambda_ey_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_ey_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hx_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hz_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        # Allocate gradient outputs
        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            if ca_batched:
                grad_ca = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_ca = torch.zeros(ny, nx, device=device, dtype=dtype)
            # Per-shot workspace for gradient accumulation (needed for CUDA)
            grad_ca_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            if cb_batched:
                grad_cb = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_cb = torch.zeros(ny, nx, device=device, dtype=dtype)
            # Per-shot workspace for gradient accumulation (needed for CUDA)
            grad_cb_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        # Get device index for CUDA
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        # Get callback-related context
        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        fd_pad_ctx = ctx.fd_pad
        models = ctx.models
        n_threads = ctx.n_threads

        pml_y0, pml_y1 = pml_y0_fwd, pml_y1_fwd
        pml_x0, pml_x1 = pml_x0_fwd, pml_x1_fwd

        # Get the backend function
        backward_func = backend_utils.get_backend_function(
            "maxwell_tm", "backward", accuracy, dtype, _backend_device
        )

        # Determine effective callback frequency
        if backward_callback is None:
            effective_callback_freq = nt // step_ratio
        else:
            effective_callback_freq = callback_frequency

        # Chunked backward propagation with callback support
        # Backward propagation goes from nt to 0
        for step in range(nt // step_ratio, 0, -effective_callback_freq):
            step_nt = min(step, effective_callback_freq) * step_ratio
            start_t = step * step_ratio

            # Call the C/CUDA backward function for this chunk
            backward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(grad_r),
                backend_utils.tensor_to_ptr(lambda_ey),
                backend_utils.tensor_to_ptr(lambda_hx),
                backend_utils.tensor_to_ptr(lambda_hz),
                backend_utils.tensor_to_ptr(m_lambda_ey_x),
                backend_utils.tensor_to_ptr(m_lambda_ey_z),
                backend_utils.tensor_to_ptr(m_lambda_hx_z),
                backend_utils.tensor_to_ptr(m_lambda_hz_x),
                backend_utils.tensor_to_ptr(ey_store_1),
                backend_utils.tensor_to_ptr(ey_store_3),
                ey_filenames_ptr,
                backend_utils.tensor_to_ptr(curl_store_1),
                backend_utils.tensor_to_ptr(curl_store_3),
                curl_filenames_ptr,
                backend_utils.tensor_to_ptr(grad_f),
                backend_utils.tensor_to_ptr(grad_ca),
                backend_utils.tensor_to_ptr(grad_cb),
                backend_utils.tensor_to_ptr(grad_ca_shot),
                backend_utils.tensor_to_ptr(grad_cb_shot),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                rdy,
                rdx,
                dt,
                step_nt,  # number of steps to run in this chunk
                n_shots,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                storage_mode,
                shot_bytes_uncomp,
                ca_requires_grad,
                cb_requires_grad,
                ca_batched,
                cb_batched,
                cq_batched,
                start_t,  # starting time step for this chunk
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                n_threads,
                device_idx,
            )

            # Call backward callback after each chunk
            if backward_callback is not None:
                # The time step index is step - 1 because the callback is
                # executed after the calculations for the current backward
                # step are complete
                callback_wavefields = {
                    "lambda_Ey": lambda_ey,
                    "lambda_Hx": lambda_hx,
                    "lambda_Hz": lambda_hz,
                    "m_lambda_Ey_x": m_lambda_ey_x,
                    "m_lambda_Ey_z": m_lambda_ey_z,
                    "m_lambda_Hx_z": m_lambda_hx_z,
                    "m_lambda_Hz_x": m_lambda_hz_x,
                }
                callback_gradients = {}
                if ca_requires_grad:
                    callback_gradients["ca"] = grad_ca
                if cb_requires_grad:
                    callback_gradients["cb"] = grad_cb
                if ca_requires_grad or cb_requires_grad:
                    # Calculate physical gradients (epsilon and sigma) using VJP
                    with torch.enable_grad():
                        eps_req = models["epsilon"].detach().requires_grad_(True)
                        sig_req = models["sigma"].detach().requires_grad_(True)
                        mu_req = models["mu"]

                        if models.get("compute_dtype", "fp32") == "fp16":
                            ca_v, cb_v, _ = prepare_parameters_nondim(eps_req, sig_req, mu_req, dt)
                        else:
                            ca_v, cb_v, _ = prepare_parameters(eps_req, sig_req, mu_req, dt)

                        vjp_tensors = []
                        vjp_grads = []
                        if ca_requires_grad:
                            vjp_tensors.append(ca_v)
                            vjp_grads.append(grad_ca)
                        if cb_requires_grad:
                            vjp_tensors.append(cb_v)
                            vjp_grads.append(grad_cb)

                        torch.autograd.backward(vjp_tensors, vjp_grads)

                        callback_gradients["epsilon"] = eps_req.grad
                        callback_gradients["sigma"] = sig_req.grad

                backward_callback(
                    CallbackState(
                        dt=dt,
                        step=step - 1,
                        nt=nt // step_ratio,
                        wavefields=callback_wavefields,
                        models=models,
                        gradients=callback_gradients,
                        fd_pad=list(fd_pad_ctx),
                        pml_width=list(pml_width),
                        is_backward=True,
                    )
                )

        # Return gradients for all inputs
        # Order: ca, cb, cq, source_amplitudes_scaled,
        #        ay, by, ay_h, by_h, ax, bx, ax_h, bx_h,
        #        ky, ky_h, kx, kx_h,
        #        sources_i, receivers_i,
        #        rdy, rdx, dt, nt, n_shots, ny, nx, n_sources, n_receivers,
        #        step_ratio, accuracy, ca_batched, cb_batched, cq_batched,
        #        pml_y0, pml_x0, pml_y1, pml_x1,
        #        fd_pad, pml_width, models,
        #        forward_callback, backward_callback, callback_frequency,
        #        Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x,
        #        n_threads, _backend_device

        # Flatten grad_f to match input shape [nt * n_shots * n_sources]
        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None

        # Match gradient shapes to input shapes
        # Input ca, cb are [1, ny, nx] but grad_ca, grad_cb are [ny, nx] when not batched
        if ca_requires_grad and not ca_batched:
            grad_ca = grad_ca.unsqueeze(0)  # [ny, nx] -> [1, ny, nx]
        if cb_requires_grad and not cb_batched:
            grad_cb = grad_cb.unsqueeze(0)  # [ny, nx] -> [1, ny, nx]

        _release_ctx_handle(getattr(ctx, "_ctx_handle_id", None))
        return (
            grad_ca if ca_requires_grad else None,  # ca
            grad_cb if cb_requires_grad else None,  # cb
            None,  # cq
            grad_f_flat,  # source_amplitudes_scaled
            None,
            None,
            None,
            None,  # ay, by, ay_h, by_h
            None,
            None,
            None,
            None,  # ax, bx, ax_h, bx_h
            None,
            None,
            None,
            None,  # ky, ky_h, kx, kx_h
            None,
            None,  # sources_i, receivers_i
            None,
            None,
            None,  # rdy, rdx, dt
            None,
            None,
            None,
            None,  # nt, n_shots, ny, nx
            None,
            None,  # n_sources, n_receivers
            None,  # step_ratio
            None,  # accuracy
            None,
            None,
            None,  # ca_batched, cb_batched, cq_batched
            None,
            None,
            None,
            None,  # pml_y0, pml_x0, pml_y1, pml_x1
            None,
            None,
            None,  # fd_pad, pml_width, models
            None,
            None,
            None,  # forward_callback, backward_callback, callback_frequency
            None,
            None,
            None,  # storage_mode_str, storage_path, storage_compression
            None,
            None,
            None,  # Ey, Hx, Hz
            None,
            None,
            None,
            None,  # m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
            None,  # n_threads
            None,  # _backend_device
        )


_COMPONENT_TO_INDEX_3D = {"ex": 0, "ey": 1, "ez": 2}


def _normalize_component_3d(component: str, *, name: str) -> str:
    if not isinstance(component, str):
        raise TypeError(f"{name} must be a string, got {type(component).__name__}.")
    value = component.strip().lower()
    if value not in _COMPONENT_TO_INDEX_3D:
        raise ValueError(f"{name} must be one of 'ex', 'ey', or 'ez', got {component!r}.")
    return value


class Maxwell3D(torch.nn.Module):
    """3D Maxwell equations solver using FDTD + CPML.

    This class is the 3D counterpart to `MaxwellTM`. It supports forward modeling
    and inversion through PyTorch autograd on `(epsilon, sigma)`.
    """

    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
        grid_spacing: Union[float, Sequence[float]],
        epsilon_requires_grad: Optional[bool] = None,
        sigma_requires_grad: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if epsilon_requires_grad is not None and not isinstance(
            epsilon_requires_grad, bool
        ):
            raise TypeError(
                f"epsilon_requires_grad must be bool or None, got {type(epsilon_requires_grad).__name__}",
            )
        if sigma_requires_grad is not None and not isinstance(
            sigma_requires_grad, bool
        ):
            raise TypeError(
                f"sigma_requires_grad must be bool or None, got {type(sigma_requires_grad).__name__}",
            )
        if not isinstance(epsilon, torch.Tensor):
            raise TypeError(
                f"epsilon must be torch.Tensor, got {type(epsilon).__name__}",
            )
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(
                f"sigma must be torch.Tensor, got {type(sigma).__name__}",
            )
        if not isinstance(mu, torch.Tensor):
            raise TypeError(
                f"mu must be torch.Tensor, got {type(mu).__name__}",
            )
        if epsilon_requires_grad is None:
            epsilon_requires_grad = epsilon.requires_grad
        if sigma_requires_grad is None:
            sigma_requires_grad = sigma.requires_grad

        self.epsilon = torch.nn.Parameter(epsilon, requires_grad=epsilon_requires_grad)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=sigma_requires_grad)
        self.register_buffer("mu", mu)
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
        Ex_0: Optional[torch.Tensor] = None,
        Ey_0: Optional[torch.Tensor] = None,
        Ez_0: Optional[torch.Tensor] = None,
        Hx_0: Optional[torch.Tensor] = None,
        Hy_0: Optional[torch.Tensor] = None,
        Hz_0: Optional[torch.Tensor] = None,
        m_hz_y: Optional[torch.Tensor] = None,
        m_hy_z: Optional[torch.Tensor] = None,
        m_hx_z: Optional[torch.Tensor] = None,
        m_hz_x: Optional[torch.Tensor] = None,
        m_hy_x: Optional[torch.Tensor] = None,
        m_hx_y: Optional[torch.Tensor] = None,
        m_ey_z: Optional[torch.Tensor] = None,
        m_ez_y: Optional[torch.Tensor] = None,
        m_ez_x: Optional[torch.Tensor] = None,
        m_ex_z: Optional[torch.Tensor] = None,
        m_ex_y: Optional[torch.Tensor] = None,
        m_ey_x: Optional[torch.Tensor] = None,
        nt: Optional[int] = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        save_snapshots: Optional[bool] = None,
        forward_callback: Optional[Callback] = None,
        backward_callback: Optional[Callback] = None,
        callback_frequency: int = 1,
        source_component: str = "ey",
        receiver_component: str = "ey",
        python_backend: Union[bool, str] = False,
        storage_mode: str = "device",
        storage_path: str = ".",
        storage_compression: Union[bool, str] = False,
        storage_bytes_limit_device: Optional[int] = None,
        storage_bytes_limit_host: Optional[int] = None,
        storage_chunk_steps: int = 0,
        n_threads: Optional[int] = None,
    ):
        assert isinstance(self.epsilon, torch.Tensor)
        assert isinstance(self.sigma, torch.Tensor)
        assert isinstance(self.mu, torch.Tensor)
        return maxwell3d(
            self.epsilon,
            self.sigma,
            self.mu,
            self.grid_spacing,
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
            nt,
            model_gradient_sampling_interval,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            save_snapshots,
            forward_callback,
            backward_callback,
            callback_frequency,
            source_component,
            receiver_component,
            python_backend,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            n_threads,
        )


def maxwell3d(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int = 2,
    pml_width: Union[int, Sequence[int]] = 20,
    max_vel: Optional[float] = None,
    Ex_0: Optional[torch.Tensor] = None,
    Ey_0: Optional[torch.Tensor] = None,
    Ez_0: Optional[torch.Tensor] = None,
    Hx_0: Optional[torch.Tensor] = None,
    Hy_0: Optional[torch.Tensor] = None,
    Hz_0: Optional[torch.Tensor] = None,
    m_hz_y: Optional[torch.Tensor] = None,
    m_hy_z: Optional[torch.Tensor] = None,
    m_hx_z: Optional[torch.Tensor] = None,
    m_hz_x: Optional[torch.Tensor] = None,
    m_hy_x: Optional[torch.Tensor] = None,
    m_hx_y: Optional[torch.Tensor] = None,
    m_ey_z: Optional[torch.Tensor] = None,
    m_ez_y: Optional[torch.Tensor] = None,
    m_ez_x: Optional[torch.Tensor] = None,
    m_ex_z: Optional[torch.Tensor] = None,
    m_ex_y: Optional[torch.Tensor] = None,
    m_ey_x: Optional[torch.Tensor] = None,
    nt: Optional[int] = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    save_snapshots: Optional[bool] = None,
    forward_callback: Optional[Callback] = None,
    backward_callback: Optional[Callback] = None,
    callback_frequency: int = 1,
    source_component: str = "ey",
    receiver_component: str = "ey",
    python_backend: Union[bool, str] = False,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    n_threads: Optional[int] = None,
):
    """3D Maxwell equations solver.

    Coordinate convention is `[z, y, x]`.
    """
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)

    if epsilon.ndim != 3:
        raise RuntimeError("epsilon must be 3D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    source_component = _normalize_component_3d(source_component, name="source_component")
    receiver_component = _normalize_component_3d(
        receiver_component, name="receiver_component"
    )

    if source_location is not None and source_location.numel() > 0:
        if source_location[..., 0].min() < 0 or source_location[..., 0].max() >= epsilon.shape[-3]:
            raise RuntimeError(
                f"Source location dim 0 must be in [0, {epsilon.shape[-3] - 1}]"
            )
        if source_location[..., 1].min() < 0 or source_location[..., 1].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Source location dim 1 must be in [0, {epsilon.shape[-2] - 1}]"
            )
        if source_location[..., 2].min() < 0 or source_location[..., 2].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Source location dim 2 must be in [0, {epsilon.shape[-1] - 1}]"
            )

    if receiver_location is not None and receiver_location.numel() > 0:
        if receiver_location[..., 0].min() < 0 or receiver_location[..., 0].max() >= epsilon.shape[-3]:
            raise RuntimeError(
                f"Receiver location dim 0 must be in [0, {epsilon.shape[-3] - 1}]"
            )
        if receiver_location[..., 1].min() < 0 or receiver_location[..., 1].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Receiver location dim 1 must be in [0, {epsilon.shape[-2] - 1}]"
            )
        if receiver_location[..., 2].min() < 0 or receiver_location[..., 2].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Receiver location dim 2 must be in [0, {epsilon.shape[-1] - 1}]"
            )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    grid_spacing_list = _normalize_grid_spacing_3d(grid_spacing)

    if max_vel is None:
        max_vel_computed = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    else:
        max_vel_computed = max_vel

    inner_dt, step_ratio = cfl_condition(grid_spacing_list, dt, max_vel_computed)

    source_amplitude_internal = source_amplitude
    if step_ratio > 1 and source_amplitude is not None and source_amplitude.numel() > 0:
        source_amplitude_internal = upsample(
            source_amplitude,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )

    nt_internal = None
    if nt is not None:
        nt_internal = nt * step_ratio
    elif source_amplitude_internal is not None:
        nt_internal = source_amplitude_internal.shape[-1]

    if isinstance(python_backend, bool):
        use_python = python_backend
    elif isinstance(python_backend, str):
        use_python = True
    else:
        raise TypeError(
            f"python_backend must be bool or str, but got {type(python_backend).__name__}"
        )

    result = (maxwell3d_python if use_python else maxwell3d_c_cuda)(
        epsilon,
        sigma,
        mu,
        grid_spacing,
        inner_dt,
        source_amplitude_internal,
        source_location,
        receiver_location,
        stencil,
        pml_width,
        max_vel_computed,
        Ex_0,
        Ey_0,
        Ez_0,
        Hx_0,
        Hy_0,
        Hz_0,
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
        nt_internal,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        forward_callback,
        backward_callback,
        callback_frequency,
        source_component,
        receiver_component,
        storage_mode,
        storage_path,
        storage_compression,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        n_threads,
    )

    (
        Ex_out,
        Ey_out,
        Ez_out,
        Hx_out,
        Hy_out,
        Hz_out,
        m_hz_y_out,
        m_hy_z_out,
        m_hx_z_out,
        m_hz_x_out,
        m_hy_x_out,
        m_hx_y_out,
        m_ey_z_out,
        m_ez_y_out,
        m_ez_x_out,
        m_ex_z_out,
        m_ex_y_out,
        m_ey_x_out,
        receiver_amplitudes,
    ) = result

    if step_ratio > 1 and receiver_amplitudes.numel() > 0:
        receiver_amplitudes = downsample_and_movedim(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
        receiver_amplitudes = torch.movedim(receiver_amplitudes, -1, 0)

    return (
        Ex_out,
        Ey_out,
        Ez_out,
        Hx_out,
        Hy_out,
        Hz_out,
        m_hz_y_out,
        m_hy_z_out,
        m_hx_z_out,
        m_hz_x_out,
        m_hy_x_out,
        m_hx_y_out,
        m_ey_z_out,
        m_ez_y_out,
        m_ez_x_out,
        m_ex_z_out,
        m_ex_y_out,
        m_ey_x_out,
        receiver_amplitudes,
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
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int,
    pml_width: Union[int, Sequence[int]],
    max_vel: Optional[float],
    Ex_0: Optional[torch.Tensor],
    Ey_0: Optional[torch.Tensor],
    Ez_0: Optional[torch.Tensor],
    Hx_0: Optional[torch.Tensor],
    Hy_0: Optional[torch.Tensor],
    Hz_0: Optional[torch.Tensor],
    m_hz_y_0: Optional[torch.Tensor],
    m_hy_z_0: Optional[torch.Tensor],
    m_hx_z_0: Optional[torch.Tensor],
    m_hz_x_0: Optional[torch.Tensor],
    m_hy_x_0: Optional[torch.Tensor],
    m_hx_y_0: Optional[torch.Tensor],
    m_ey_z_0: Optional[torch.Tensor],
    m_ez_y_0: Optional[torch.Tensor],
    m_ez_x_0: Optional[torch.Tensor],
    m_ex_z_0: Optional[torch.Tensor],
    m_ex_y_0: Optional[torch.Tensor],
    m_ey_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    source_component: str,
    receiver_component: str,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    n_threads: Optional[int] = None,
):
    """3D Python backend propagation with autograd support."""
    del (
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        backward_callback,
        storage_path,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        n_threads,
    )
    from .padding import create_or_pad, zero_interior

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
    ca, cb, cq = prepare_parameters(epsilon_padded, sigma_padded, mu_padded, dt)
    ca = ca[None, :, :, :]
    cb = cb[None, :, :, :]
    cq = cq[None, :, :, :]

    size_with_batch = (n_shots, padded_nz, padded_ny, padded_nx)

    def init_wavefield(field_0: Optional[torch.Tensor]) -> torch.Tensor:
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

    from . import staggered as _staggered

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

    if n_receivers > 0:
        receiver_amplitudes = torch.zeros(
            nt_steps,
            n_shots,
            n_receivers,
            device=device,
            dtype=dtype,
        )
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

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

        Ex = ca * Ex + cb * (dHy_dz_pml - dHz_dy_pml)
        Ey = ca * Ey + cb * (dHz_dx_pml - dHx_dz_pml)
        Ez = ca * Ez + cb * (dHx_dy_pml - dHy_dx_pml)

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

        if n_receivers > 0:
            rec_field = _select_e_component(receiver_component, Ex, Ey, Ez)
            receiver_amplitudes[step] = rec_field.reshape(
                n_shots, flat_model_shape
            ).gather(1, receivers_i)

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
        receiver_amplitudes,
    )


class Maxwell3DForwardFunc(torch.autograd.Function):
    """Autograd function for 3D C/CUDA backend propagation."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        profiles: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, torch.Tensor],
        wavefields: tuple[torch.Tensor, ...],
        meta: dict[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        from . import backend_utils

        (
            az,
            bz,
            az_h,
            bz_h,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            kz,
            kz_h,
            ky,
            ky_h,
            kx,
            kx_h,
        ) = profiles
        sources_i, receivers_i = indices
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
        ) = wavefields

        device = Ex.device
        dtype = Ex.dtype

        nt = int(meta["nt"])
        n_shots = int(meta["n_shots"])
        nz = int(meta["nz"])
        ny = int(meta["ny"])
        nx = int(meta["nx"])
        n_sources = int(meta["n_sources"])
        n_receivers = int(meta["n_receivers"])
        step_ratio = int(meta["step_ratio"])
        accuracy = int(meta["accuracy"])
        pml_z0 = int(meta["pml_z0"])
        pml_y0 = int(meta["pml_y0"])
        pml_x0 = int(meta["pml_x0"])
        pml_z1 = int(meta["pml_z1"])
        pml_y1 = int(meta["pml_y1"])
        pml_x1 = int(meta["pml_x1"])
        source_component_idx = int(meta["source_component_idx"])
        receiver_component_idx = int(meta["receiver_component_idx"])
        n_threads = int(meta["n_threads"])
        callback_frequency = int(meta["callback_frequency"])
        forward_callback = meta["forward_callback"]
        models = meta["models"]
        fd_pad = meta["fd_pad"]
        pml_width = meta["pml_width"]
        grid_spacing = meta["grid_spacing"]
        dt = float(meta["dt"])

        ca_requires_grad = bool(ca.requires_grad)
        cb_requires_grad = bool(cb.requires_grad)
        requires_grad = ca_requires_grad or cb_requires_grad
        if not requires_grad:
            raise RuntimeError("Maxwell3DForwardFunc should only be used when gradients are required.")

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        step_ratio = max(1, step_ratio)
        num_steps_stored = (nt + step_ratio - 1) // step_ratio
        shot_numel = nz * ny * nx
        shot_bytes_uncomp = shot_numel * dtype.itemsize
        storage_mode = STORAGE_DEVICE

        # Store E and curl(H) components for adjoint gradients.
        store_ex = torch.empty(
            num_steps_stored, n_shots, nz, ny, nx, device=device, dtype=dtype
        )
        store_ey = torch.empty(
            num_steps_stored, n_shots, nz, ny, nx, device=device, dtype=dtype
        )
        store_ez = torch.empty(
            num_steps_stored, n_shots, nz, ny, nx, device=device, dtype=dtype
        )
        store_curl_x = torch.empty(
            num_steps_stored, n_shots, nz, ny, nx, device=device, dtype=dtype
        )
        store_curl_y = torch.empty(
            num_steps_stored, n_shots, nz, ny, nx, device=device, dtype=dtype
        )
        store_curl_z = torch.empty(
            num_steps_stored, n_shots, nz, ny, nx, device=device, dtype=dtype
        )
        empty_store = torch.empty(0, device=device, dtype=dtype)

        forward_func = backend_utils.get_backend_function(
            "maxwell_3d", "forward_with_storage", accuracy, dtype, device
        )
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        effective_callback_freq = nt if forward_callback is None else callback_frequency
        if effective_callback_freq <= 0:
            effective_callback_freq = nt if nt > 0 else 1

        for step in range(0, nt, effective_callback_freq):
            if forward_callback is not None:
                forward_callback(
                    CallbackState(
                        dt=dt,
                        step=step,
                        nt=nt,
                        wavefields={
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
                        },
                        models=models,
                        gradients={},
                        fd_pad=list(fd_pad),
                        pml_width=list(pml_width),
                        is_backward=False,
                        grid_spacing=list(grid_spacing),
                    )
                )

            step_nt = min(nt - step, effective_callback_freq)
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(source_amplitudes_scaled),
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
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(store_ex),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_ey),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_ez),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_curl_x),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_curl_y),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_curl_z),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(az),
                backend_utils.tensor_to_ptr(bz),
                backend_utils.tensor_to_ptr(az_h),
                backend_utils.tensor_to_ptr(bz_h),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(kz),
                backend_utils.tensor_to_ptr(kz_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                float(meta["rdz"]),
                float(meta["rdy"]),
                float(meta["rdx"]),
                dt,
                step_nt,
                n_shots,
                nz,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                storage_mode,
                shot_bytes_uncomp,
                ca_requires_grad,
                cb_requires_grad,
                False,
                False,
                False,
                step,
                pml_z0,
                pml_y0,
                pml_x0,
                pml_z1,
                pml_y1,
                pml_x1,
                source_component_idx,
                receiver_component_idx,
                n_threads,
                device_idx,
            )

        ctx.save_for_backward(
            ca,
            cb,
            cq,
            az,
            bz,
            az_h,
            bz_h,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            kz,
            kz_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            store_ex,
            store_ey,
            store_ez,
            store_curl_x,
            store_curl_y,
            store_curl_z,
        )
        ctx.meta = {
            "dt": dt,
            "nt": nt,
            "n_shots": n_shots,
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "n_sources": n_sources,
            "n_receivers": n_receivers,
            "step_ratio": step_ratio,
            "accuracy": accuracy,
            "pml_z0": pml_z0,
            "pml_y0": pml_y0,
            "pml_x0": pml_x0,
            "pml_z1": pml_z1,
            "pml_y1": pml_y1,
            "pml_x1": pml_x1,
            "source_component_idx": source_component_idx,
            "receiver_component_idx": receiver_component_idx,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
            "models": models,
            "fd_pad": fd_pad,
            "pml_width": pml_width,
            "backward_callback": meta["backward_callback"],
            "callback_frequency": callback_frequency,
            "n_threads": n_threads,
            "rdz": float(meta["rdz"]),
            "rdy": float(meta["rdy"]),
            "rdx": float(meta["rdx"]),
            "shot_bytes_uncomp": shot_bytes_uncomp,
        }

        return (
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
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        from . import backend_utils

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        az, bz, az_h, bz_h = saved[3], saved[4], saved[5], saved[6]
        ay, by, ay_h, by_h = saved[7], saved[8], saved[9], saved[10]
        ax, bx, ax_h, bx_h = saved[11], saved[12], saved[13], saved[14]
        kz, kz_h, ky, ky_h, kx, kx_h = saved[15], saved[16], saved[17], saved[18], saved[19], saved[20]
        sources_i, receivers_i = saved[21], saved[22]
        store_ex, store_ey, store_ez = saved[23], saved[24], saved[25]
        store_curl_x, store_curl_y, store_curl_z = saved[26], saved[27], saved[28]

        meta = ctx.meta
        device = ca.device
        dtype = ca.dtype

        nt = int(meta["nt"])
        n_shots = int(meta["n_shots"])
        nz = int(meta["nz"])
        ny = int(meta["ny"])
        nx = int(meta["nx"])
        n_sources = int(meta["n_sources"])
        n_receivers = int(meta["n_receivers"])
        step_ratio = int(meta["step_ratio"])
        accuracy = int(meta["accuracy"])
        pml_z0 = int(meta["pml_z0"])
        pml_y0 = int(meta["pml_y0"])
        pml_x0 = int(meta["pml_x0"])
        pml_z1 = int(meta["pml_z1"])
        pml_y1 = int(meta["pml_y1"])
        pml_x1 = int(meta["pml_x1"])
        source_component_idx = int(meta["source_component_idx"])
        receiver_component_idx = int(meta["receiver_component_idx"])
        ca_requires_grad = bool(meta["ca_requires_grad"])
        cb_requires_grad = bool(meta["cb_requires_grad"])
        backward_callback = meta["backward_callback"]
        callback_frequency = int(meta["callback_frequency"])
        models = meta["models"]
        fd_pad = meta["fd_pad"]
        pml_width = meta["pml_width"]
        n_threads = int(meta["n_threads"])
        shot_bytes_uncomp = int(meta["shot_bytes_uncomp"])
        dt = float(meta["dt"])

        grad_r = grad_outputs[-1]
        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            grad_r = grad_r.contiguous()

        lambda_ex = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_ey = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_ez = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_hx = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_hy = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_hz = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)

        m_lambda_ey_z = torch.zeros_like(lambda_ex)
        m_lambda_ez_y = torch.zeros_like(lambda_ex)
        m_lambda_ez_x = torch.zeros_like(lambda_ex)
        m_lambda_ex_z = torch.zeros_like(lambda_ex)
        m_lambda_ex_y = torch.zeros_like(lambda_ex)
        m_lambda_ey_x = torch.zeros_like(lambda_ex)
        m_lambda_hz_y = torch.zeros_like(lambda_ex)
        m_lambda_hy_z = torch.zeros_like(lambda_ex)
        m_lambda_hx_z = torch.zeros_like(lambda_ex)
        m_lambda_hz_x = torch.zeros_like(lambda_ex)
        m_lambda_hy_x = torch.zeros_like(lambda_ex)
        m_lambda_hx_y = torch.zeros_like(lambda_ex)

        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            grad_ca = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            grad_ca_shot = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            grad_cb = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            grad_cb_shot = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad or cb_requires_grad:
            grad_eps = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            grad_sigma = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        empty_store = torch.empty(0, device=device, dtype=dtype)
        backward_func = backend_utils.get_backend_function(
            "maxwell_3d", "backward", accuracy, dtype, device
        )
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        effective_callback_freq = nt if backward_callback is None else callback_frequency
        if effective_callback_freq <= 0:
            effective_callback_freq = nt if nt > 0 else 1

        for step in range(nt, 0, -effective_callback_freq):
            step_nt = min(step, effective_callback_freq)
            backward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(grad_r),
                backend_utils.tensor_to_ptr(lambda_ex),
                backend_utils.tensor_to_ptr(lambda_ey),
                backend_utils.tensor_to_ptr(lambda_ez),
                backend_utils.tensor_to_ptr(lambda_hx),
                backend_utils.tensor_to_ptr(lambda_hy),
                backend_utils.tensor_to_ptr(lambda_hz),
                backend_utils.tensor_to_ptr(m_lambda_ey_z),
                backend_utils.tensor_to_ptr(m_lambda_ez_y),
                backend_utils.tensor_to_ptr(m_lambda_ez_x),
                backend_utils.tensor_to_ptr(m_lambda_ex_z),
                backend_utils.tensor_to_ptr(m_lambda_ex_y),
                backend_utils.tensor_to_ptr(m_lambda_ey_x),
                backend_utils.tensor_to_ptr(m_lambda_hz_y),
                backend_utils.tensor_to_ptr(m_lambda_hy_z),
                backend_utils.tensor_to_ptr(m_lambda_hx_z),
                backend_utils.tensor_to_ptr(m_lambda_hz_x),
                backend_utils.tensor_to_ptr(m_lambda_hy_x),
                backend_utils.tensor_to_ptr(m_lambda_hx_y),
                backend_utils.tensor_to_ptr(store_ex),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_ey),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_ez),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_curl_x),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_curl_y),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(store_curl_z),
                backend_utils.tensor_to_ptr(empty_store),
                0,
                backend_utils.tensor_to_ptr(grad_f),
                backend_utils.tensor_to_ptr(grad_ca),
                backend_utils.tensor_to_ptr(grad_cb),
                backend_utils.tensor_to_ptr(grad_eps),
                backend_utils.tensor_to_ptr(grad_sigma),
                backend_utils.tensor_to_ptr(grad_ca_shot),
                backend_utils.tensor_to_ptr(grad_cb_shot),
                backend_utils.tensor_to_ptr(az),
                backend_utils.tensor_to_ptr(bz),
                backend_utils.tensor_to_ptr(az_h),
                backend_utils.tensor_to_ptr(bz_h),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(kz),
                backend_utils.tensor_to_ptr(kz_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                float(meta["rdz"]),
                float(meta["rdy"]),
                float(meta["rdx"]),
                dt,
                step_nt,
                n_shots,
                nz,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                STORAGE_DEVICE,
                shot_bytes_uncomp,
                ca_requires_grad,
                cb_requires_grad,
                False,
                False,
                False,
                step,
                pml_z0,
                pml_y0,
                pml_x0,
                pml_z1,
                pml_y1,
                pml_x1,
                source_component_idx,
                receiver_component_idx,
                n_threads,
                device_idx,
            )

            if backward_callback is not None:
                callback_gradients = {}
                if ca_requires_grad:
                    callback_gradients["ca"] = grad_ca
                if cb_requires_grad:
                    callback_gradients["cb"] = grad_cb
                if ca_requires_grad or cb_requires_grad:
                    callback_gradients["epsilon"] = grad_eps
                    callback_gradients["sigma"] = grad_sigma
                backward_callback(
                    CallbackState(
                        dt=dt,
                        step=step - 1,
                        nt=nt,
                        wavefields={
                            "lambda_Ex": lambda_ex,
                            "lambda_Ey": lambda_ey,
                            "lambda_Ez": lambda_ez,
                            "lambda_Hx": lambda_hx,
                            "lambda_Hy": lambda_hy,
                            "lambda_Hz": lambda_hz,
                        },
                        models=models,
                        gradients=callback_gradients,
                        fd_pad=list(fd_pad),
                        pml_width=list(pml_width),
                        is_backward=True,
                    )
                )

        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None

        grad_ca_out = grad_ca.unsqueeze(0) if ca_requires_grad else None
        grad_cb_out = grad_cb.unsqueeze(0) if cb_requires_grad else None
        return (
            grad_ca_out,
            grad_cb_out,
            None,
            grad_f_flat,
            None,
            None,
            None,
            None,
        )


def maxwell3d_c_cuda(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Union[float, Sequence[float]],
    dt: float,
    source_amplitude: Optional[torch.Tensor],
    source_location: Optional[torch.Tensor],
    receiver_location: Optional[torch.Tensor],
    stencil: int,
    pml_width: Union[int, Sequence[int]],
    max_vel: Optional[float],
    Ex_0: Optional[torch.Tensor],
    Ey_0: Optional[torch.Tensor],
    Ez_0: Optional[torch.Tensor],
    Hx_0: Optional[torch.Tensor],
    Hy_0: Optional[torch.Tensor],
    Hz_0: Optional[torch.Tensor],
    m_hz_y_0: Optional[torch.Tensor],
    m_hy_z_0: Optional[torch.Tensor],
    m_hx_z_0: Optional[torch.Tensor],
    m_hz_x_0: Optional[torch.Tensor],
    m_hy_x_0: Optional[torch.Tensor],
    m_hx_y_0: Optional[torch.Tensor],
    m_ey_z_0: Optional[torch.Tensor],
    m_ez_y_0: Optional[torch.Tensor],
    m_ez_x_0: Optional[torch.Tensor],
    m_ex_z_0: Optional[torch.Tensor],
    m_ex_y_0: Optional[torch.Tensor],
    m_ey_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    source_component: str,
    receiver_component: str,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    n_threads: Optional[int] = None,
):
    """3D C/CUDA forward propagation path with Python fallback for gradients."""
    from . import backend_utils, staggered
    from .padding import create_or_pad, zero_interior
    import warnings

    del (
        storage_path,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
    )

    if epsilon.ndim != 3:
        raise RuntimeError("epsilon must be 3D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    storage_mode_str = str(storage_mode).lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )

    n_threads_val = 0
    if n_threads is not None:
        n_threads_val = int(n_threads)
        if n_threads_val < 0:
            raise ValueError("n_threads must be >= 0 when provided.")

    storage_kind = _normalize_storage_compression(storage_compression)
    requires_grad = epsilon.requires_grad or sigma.requires_grad
    functorch_active = torch._C._are_functorch_transforms_active()

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
            storage_mode=fallback_storage_mode,
            storage_compression=storage_compression,
            n_threads=n_threads,
        )

    if not backend_utils.is_backend_available():
        return _fallback_reason("C/CUDA backend library is unavailable")

    if functorch_active:
        return _fallback_reason(
            "torch.func transforms are not supported for 3D C/CUDA backend"
        )

    if requires_grad:
        if storage_kind != "none":
            return _fallback_reason(
                "3D C/CUDA gradient path currently requires storage_compression=False"
            )
        if storage_mode_str == "none":
            return _fallback_reason(
                "storage_mode='none' is incompatible with 3D gradient computation"
            )
        if storage_mode_str in {"cpu", "disk"}:
            return _fallback_reason(
                "3D C/CUDA gradient path currently supports storage_mode='device' or 'auto' only"
            )
        if storage_mode_str == "auto":
            storage_mode_str = "device"
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
    mu_padded = create_or_pad(mu, total_pad, device, dtype, padded_size, mode="replicate")

    ca, cb, cq = prepare_parameters(epsilon_padded, sigma_padded, mu_padded, dt)
    size_with_batch = (n_shots, padded_nz, padded_ny, padded_nx)

    def init_wavefield(field_0: Optional[torch.Tensor]) -> torch.Tensor:
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

    ca = ca[None, :, :, :].contiguous()
    cb = cb[None, :, :, :].contiguous()
    cq = cq[None, :, :, :].contiguous()

    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca,
        "cb": cb,
        "cq": cq,
    }

    pml_z0 = fd_pad_list[0] + pml_width_list[0]
    pml_z1 = padded_nz - fd_pad_list[1] - pml_width_list[1]
    pml_y0 = fd_pad_list[2] + pml_width_list[2]
    pml_y1 = padded_ny - fd_pad_list[3] - pml_width_list[3]
    pml_x0 = fd_pad_list[4] + pml_width_list[4]
    pml_x1 = padded_nx - fd_pad_list[5] - pml_width_list[5]

    source_component_idx = _COMPONENT_TO_INDEX_3D[source_component]
    receiver_component_idx = _COMPONENT_TO_INDEX_3D[receiver_component]
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
            return _fallback_reason(f"3D C/CUDA forward symbol is unavailable ({e})")

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        effective_callback_freq = nt_steps if forward_callback is None else callback_frequency
        if effective_callback_freq <= 0:
            effective_callback_freq = nt_steps if nt_steps > 0 else 1

        for step in range(0, nt_steps, effective_callback_freq):
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

            step_nt = min(nt_steps - step, effective_callback_freq)
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(f),
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
                backend_utils.tensor_to_ptr(receiver_amplitudes),
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
                step_nt,
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
                step,
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
        receiver_amplitudes,
    )
