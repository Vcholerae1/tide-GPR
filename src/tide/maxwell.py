from typing import Any, Callable, Optional, Sequence, Union

import torch

from .autograd_utils import (
    _get_ctx_handle,
    _register_ctx_handle,
    _release_ctx_handle,
)
from .callbacks import CallbackState, Callback
from .cfl import cfl_condition
from .grid_utils import (
    _compute_boundary_indices_flat,
    _normalize_grid_spacing_2d,
    _normalize_pml_width_2d,
)
from .resampling import upsample, downsample_and_movedim
from .validation import (
    validate_model_gradient_sampling_interval,
    validate_freq_taper_frac,
    validate_time_pad_frac,
)
from .storage import (
    TemporaryStorage,
    storage_mode_to_int,
    STORAGE_DEVICE,
    STORAGE_CPU,
    STORAGE_DISK,
    STORAGE_NONE,
    _CPU_STORAGE_BUFFERS,
    _normalize_storage_compression,
    _resolve_storage_compression,
)
from .utils import C0, prepare_parameters
from . import staggered


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
        if epsilon_requires_grad is not None and not isinstance(epsilon_requires_grad, bool):
            raise TypeError(
                f"epsilon_requires_grad must be bool or None, "
                f"got {type(epsilon_requires_grad).__name__}",
            )
        if not isinstance(epsilon, torch.Tensor):
            raise TypeError(
                f"epsilon must be torch.Tensor, got {type(epsilon).__name__}",
            )
        if sigma_requires_grad is not None and not isinstance(sigma_requires_grad, bool):
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

        self.epsilon = torch.nn.Parameter(
            epsilon, requires_grad=epsilon_requires_grad
        )
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
        gradient_mode: str = "snapshot",
        storage_mode: str = "device",
        storage_path: str = ".",
        storage_compression: Union[bool, str] = False,
        storage_bytes_limit_device: Optional[int] = None,
        storage_bytes_limit_host: Optional[int] = None,
        storage_chunk_steps: int = 0,
        boundary_width: int = 0,
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
            gradient_mode,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            boundary_width,
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
    gradient_mode: str = "snapshot",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    boundary_width: int = 0,
    n_threads: Optional[int] = None,
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
        gradient_mode: Gradient computation mode:
            - "snapshot": store Ey/curl(H) snapshots for ASM (CPU/CUDA).
            - "boundary": store a boundary ring and reconstruct during backward (C/CUDA backend).
        storage_mode: Where to store intermediate snapshots for the ASM
            backward pass. One of "device", "cpu", "disk", "none", or "auto".
        storage_path: Base path for disk storage when storage_mode="disk".
        storage_compression: Compression for stored snapshots. Use False/True
            (True == BF16), or one of "bf16" / "fp8".
        storage_bytes_limit_device: Soft limit in bytes for device snapshot
            storage when storage_mode="auto".
        storage_bytes_limit_host: Soft limit in bytes for host snapshot
            storage when storage_mode="auto".
        storage_chunk_steps: Optional chunk size (in stored steps) for
            CPU/disk modes. Currently unused.
        boundary_width: Width of boundary storage region (stage 2 only).
        n_threads: OpenMP thread count for CPU backend. None uses the OpenMP default.

    Returns:
        Tuple of (Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x, receiver_amplitudes).
    """
    # Validate resampling parameters
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)
    
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

    device = epsilon.device

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

    # Call the propagation function with internal dt and upsampled source
    result = maxwell_func(
        python_backend,
        epsilon,
        sigma,
        mu,
        grid_spacing,
        inner_dt,  # Use internal time step for CFL compliance
        source_amplitude_internal,
        source_location,
        receiver_location,
        stencil,
        pml_width,
        max_vel_computed,  # Pass computed max_vel so it's not recomputed
        Ey_0,
        Hx_0,
        Hz_0,
        m_Ey_x,
        m_Ey_z,
        m_Hx_z,
        m_Hz_x,
        nt_internal,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        forward_callback,
        backward_callback,
        callback_frequency,
        gradient_mode,
        storage_mode,
        storage_path,
        storage_compression,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        boundary_width,
        n_threads,
    )
    
    # Unpack result
    Ey_out, Hx_out, Hz_out, m_Ey_x_out, m_Ey_z_out, m_Hx_z_out, m_Hz_x_out, receiver_amplitudes = result
    
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

    # Check if we should use Python backend or C/CUDA backend
    use_python = python_backend
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

        return maxwell_python(*args)
    else:
        # Use C/CUDA backend
        return maxwell_c_cuda(*args)


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
    gradient_mode: str = "snapshot",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    boundary_width: int = 0,
    n_threads: Optional[int] = None,
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

    from .padding import create_or_pad, zero_interior

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

    gradient_mode_str = gradient_mode.lower()
    if gradient_mode_str != "snapshot":
        raise NotImplementedError(
            f"gradient_mode={gradient_mode!r} is not implemented yet; "
            "only 'snapshot' is supported for the python backend."
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
    epsilon_padded = create_or_pad(epsilon, total_pad, device, dtype, padded_size, mode='replicate')
    sigma_padded = create_or_pad(sigma, total_pad, device, dtype, padded_size, mode='replicate')
    mu_padded = create_or_pad(mu, total_pad, device, dtype, padded_size, mode='replicate')

    # Prepare update coefficients using padded models
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
    
    # Helper function to initialize wavefields with fd_pad padding
    def init_wavefield(field_0: Optional[torch.Tensor]) -> torch.Tensor:
        """Initialize wavefield with fd_pad zero padding.
        
        Zero padding is used for wavefields because the fd_pad region should
        always be zero after output cropping and re-padding. The staggered grid
        operators only read from this region but don't need non-zero values there
        for correct propagation.
        """
        if field_0 is not None:
            # User provides [n_shots, ny, nx] or [ny, nx]
            if field_0.ndim == 2:
                field_0 = field_0[None, :, :].expand(n_shots, -1, -1)
            # Pad with asymmetric fd_pad_list for staggered grid (zero padding)
            return create_or_pad(field_0, fd_pad_list, device, dtype, size_with_batch, mode='constant')
        return torch.zeros(size_with_batch, device=device, dtype=dtype)

    Ey = init_wavefield(Ey_0)
    Hx = init_wavefield(Hx_0)
    Hz = init_wavefield(Hz_0)
    m_Ey_x = init_wavefield(m_Ey_x_0)
    m_Ey_z = init_wavefield(m_Ey_z_0)
    m_Hx_z = init_wavefield(m_Hx_z_0)
    m_Hz_x = init_wavefield(m_Hz_x_0)
    
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
    pml_profiles, kappa_profiles = staggered.set_pml_profiles(
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
    )
    # pml_profiles = [ay, ayh, ax, axh, by, byh, bx, bxh]
    ay, ay_h, ax, ax_h, by, by_h, bx, bx_h = pml_profiles
    # kappa_profiles = [ky, kyh, kx, kxh]
    kappa_y, kappa_y_h, kappa_x, kappa_x_h = kappa_profiles

    # PML boundaries for interior gradient accumulation
    pml_y0 = fd_pad_list[0] + pml_width_list[0]
    pml_y1 = padded_ny - fd_pad_list[1] - pml_width_list[1]
    pml_x0 = fd_pad_list[2] + pml_width_list[2]
    pml_x1 = padded_nx - fd_pad_list[3] - pml_width_list[3]

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
        receiver_amplitudes = torch.zeros(nt_steps, n_shots, n_receivers, device=device, dtype=dtype)
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
        if source_amplitude is not None and source_amplitude.numel() > 0 and n_sources > 0:
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
            receiver_amplitudes[step] = (
                Ey.reshape(n_shots, flat_model_shape)
                .gather(1, receivers_i)
            )

    # =========================================================================
    # Output cropping:
    # Only remove fd_pad, keep the PML region in the output wavefields.
    # Output shape: [n_shots, model_ny + pml_width_y, model_nx + pml_width_x]
    # =========================================================================
    s = (
        slice(None),  # batch dimension
        slice(fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
        slice(fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
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
    gradient_mode: str = "snapshot",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    boundary_width: int = 0,
    n_threads: Optional[int] = None,
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
    from . import staggered
    from .padding import create_or_pad, zero_interior

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
    epsilon_padded = create_or_pad(epsilon, total_pad, device, dtype, padded_size, mode='replicate')
    sigma_padded = create_or_pad(sigma, total_pad, device, dtype, padded_size, mode='replicate')
    mu_padded = create_or_pad(mu, total_pad, device, dtype, padded_size, mode='replicate')

    # Prepare update coefficients using padded models
    ca, cb, cq = prepare_parameters(epsilon_padded, sigma_padded, mu_padded, dt)

    # Initialize fields with padded dimensions
    size_with_batch = (n_shots, padded_ny, padded_nx)

    def init_wavefield(field_0: Optional[torch.Tensor]) -> torch.Tensor:
        """Initialize wavefield with fd_pad zero padding."""
        if field_0 is not None:
            if field_0.ndim == 2:
                field_0 = field_0[None, :, :].expand(n_shots, -1, -1)
            # Pad with asymmetric fd_pad_list for staggered grid
            return create_or_pad(field_0, fd_pad_list, device, dtype, size_with_batch).contiguous()
        return torch.zeros(size_with_batch, device=device, dtype=dtype)

    Ey = init_wavefield(Ey_0)
    Hx = init_wavefield(Hx_0)
    Hz = init_wavefield(Hz_0)
    m_Ey_x = init_wavefield(m_Ey_x_0)
    m_Ey_z = init_wavefield(m_Ey_z_0)
    m_Hx_z = init_wavefield(m_Hx_z_0)
    m_Hz_x = init_wavefield(m_Hz_x_0)
    
    # Zero out interior of PML auxiliary variables (optimization)
    # This works correctly with user-provided states (see forward pass comments)
    pml_aux_dims = [1, 0, 0, 1]  # [m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x]
    for wf, dim in zip([m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x], pml_aux_dims):
        zero_interior(wf, fd_pad_list, pml_width_list, dim)

    # Set up PML profiles for the padded domain
    pml_profiles, kappa_profiles = staggered.set_pml_profiles(
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
    )
    ay, ay_h, ax, ax_h, by, by_h, bx, bx_h = pml_profiles
    ky, ky_h, kx, kx_h = kappa_profiles

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

    gradient_mode_str = gradient_mode.lower()
    if gradient_mode_str not in {"snapshot", "boundary"}:
        raise ValueError(
            "gradient_mode must be 'snapshot' or 'boundary', "
            f"but got {gradient_mode!r}"
        )
    functorch_active = torch._C._are_functorch_transforms_active()
    if functorch_active:
        raise NotImplementedError(
            "torch.func transforms are not supported for the C/CUDA backend."
        )

    boundary_indices: Optional[torch.Tensor] = None

    storage_kind, _, storage_bytes_per_elem = _resolve_storage_compression(
        storage_compression,
        dtype,
        device,
        context="storage_compression",
    )
    storage_bf16 = storage_kind == "bf16"
    
    # Determine if we should save snapshots for backward pass
    if save_snapshots is None:
        do_save_snapshots = requires_grad
    else:
        do_save_snapshots = save_snapshots

    # If save_snapshots is False but requires_grad is True, warn user
    if (
        requires_grad
        and save_snapshots is False
    ):
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
                "storage_mode='none' is not compatible with "
                f"gradient_mode={gradient_mode!r} when gradients are required."
            )
        if effective_storage_mode_str == "auto":
            dtype_size = storage_bytes_per_elem
            if gradient_mode_str == "snapshot":
                # Estimate required bytes for storing Ey and curl_H.
                num_elements_per_shot = padded_ny * padded_nx
                shot_bytes_uncomp = num_elements_per_shot * dtype_size
                n_stored = (nt_steps + gradient_sampling_interval - 1) // gradient_sampling_interval
                total_bytes = n_stored * n_shots * shot_bytes_uncomp * 2  # Ey + curl_H
            else:
                # Boundary storage: store Ey/Hx/Hz boundary ring for every step (nt+1).
                if boundary_width <= 0:
                    boundary_width = fd_pad
                if boundary_width < fd_pad:
                    raise ValueError(
                        f"boundary_width must be >= {fd_pad} for stencil={stencil}."
                    )
                boundary_indices = _compute_boundary_indices_flat(
                    ny=padded_ny,
                    nx=padded_nx,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    boundary_width=boundary_width,
                    device=device,
                )
                boundary_numel = int(boundary_indices.numel())
                shot_bytes_uncomp = boundary_numel * dtype_size
                total_bytes = (nt_steps + 1) * n_shots * shot_bytes_uncomp * 3  # Ey/Hx/Hz

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
    }

    use_autograd_fn = (
        (requires_grad and do_save_snapshots and gradient_mode_str in {"snapshot", "boundary"})
        or functorch_active
    )
    if use_autograd_fn:
        # Use autograd Function for gradient computation
        if gradient_mode_str == "snapshot":
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
            )
        elif gradient_mode_str == "boundary":
            import warnings

            if forward_callback is not None or backward_callback is not None:
                raise NotImplementedError(
                    "Callbacks are not supported yet for gradient_mode='boundary'."
                )

            if boundary_width <= 0:
                boundary_width = fd_pad
            if boundary_width < fd_pad:
                raise ValueError(
                    f"boundary_width must be >= {fd_pad} for stencil={stencil}."
                )
            if gradient_sampling_interval != 1:
                warnings.warn(
                    "gradient_mode='boundary' requires model_gradient_sampling_interval=1; "
                    f"got {gradient_sampling_interval}, forcing to 1.",
                    RuntimeWarning,
                )
                gradient_sampling_interval = 1

            if boundary_indices is None:
                boundary_indices = _compute_boundary_indices_flat(
                    ny=padded_ny,
                    nx=padded_nx,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    boundary_width=boundary_width,
                    device=device,
                )

            result = MaxwellTMForwardBoundaryFunc.apply(
                ca,
                cb,
                cq,
                f,
                boundary_indices,
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
                stencil,  # accuracy
                False,  # ca_batched
                False,  # cb_batched
                False,  # cq_batched
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
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
            ) = result  # type: ignore
        # Output cropping: only remove fd_pad, keep PML region
        s = (
            slice(None),  # batch dimension
            slice(fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
            slice(fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
        )
        
        return (
            Ey_out[s],
            Hx_out[s],
            Hz_out[s],
            m_Ey_x_out[s],
            m_Ey_z_out[s],
            m_Hx_z_out[s],
            m_Hz_x_out[s],
            receiver_amplitudes,
        )
    else:
        # Direct call without autograd for inference
        # Get the backend function
        try:
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward", stencil, dtype, device
            )
        except AttributeError as e:
            raise RuntimeError(
                f"C/CUDA backend function not available for accuracy={stencil}, "
                f"dtype={dtype}, device={device}. Error: {e}"
            )

        # Get device index for CUDA
        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        # Initialize receiver amplitudes
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(nt_steps, n_shots, n_receivers, device=device, dtype=dtype)
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
            slice(fd_pad_list[0], padded_ny - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
            slice(fd_pad_list[2], padded_nx - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
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
    ) -> tuple[Any, ...]:
        """Performs the forward propagation of the Maxwell TM equations."""
        from . import backend_utils

        device = Ey.device
        dtype = Ey.dtype

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
        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

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
            curl_store_1, curl_store_3, curl_filenames_ptr = alloc_storage(cb_requires_grad)

            # Get the backend function with storage
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm", "forward_with_storage", accuracy, dtype, device
            )

            # Determine effective callback frequency
            if forward_callback is None:
                effective_callback_freq = nt // step_ratio
            else:
                effective_callback_freq = callback_frequency

            # Chunked forward propagation with callback support
            for step in range(0, nt // step_ratio, effective_callback_freq):
                step_nt = min(effective_callback_freq, nt // step_ratio - step) * step_ratio
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
                "maxwell_tm", "forward", accuracy, dtype, device
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
    def setup_context(ctx: Any, inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> None:
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
        ) = inputs

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
        ctx.backward_storage_filename_arrays = ctx_data["backward_storage_filename_arrays"]
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

        grad_outputs = list(grad_outputs)
        if len(grad_outputs) == 9:
            grad_outputs.pop()  # drop context handle grad

        # Unpack grad_outputs
        (
            grad_Ey, grad_Hx, grad_Hz,
            grad_m_Ey_x, grad_m_Ey_z, grad_m_Hx_z, grad_m_Hz_x,
            grad_r,
        ) = grad_outputs

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
        pml_y0 = ctx.pml_y0
        pml_x0 = ctx.pml_x0
        pml_y1 = ctx.pml_y1
        pml_x1 = ctx.pml_x1
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

        # Recalculate PML boundaries for gradient accumulation
        # 
        # For staggered grid schemes, the backward pass uses an extended PML region
        # compared to forward. This is because backward calculations
        # involve spatial derivatives of terms that are themselves spatial derivatives.
        #
        # In tide, the padded domain includes both fd_pad and pml_width:
        #   - pml_y0 = fd_pad + pml_width (start of interior, from forward)
        #   - pml_y1 = ny - (fd_pad-1) - pml_width (end of interior, from forward)
        # 
        # The gradient accumulation region is controlled by loop bounds in C/CUDA
        # with pml_bounds array and 3-region loop.

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

        if ca_requires_grad or cb_requires_grad:
            if ca_batched:
                grad_eps = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_eps = torch.zeros(ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(ny, nx, device=device, dtype=dtype)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        # Get device index for CUDA
        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        # Get callback-related context
        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        fd_pad_ctx = ctx.fd_pad
        models = ctx.models
        n_threads = ctx.n_threads

        # Get the backend function
        backward_func = backend_utils.get_backend_function(
            "maxwell_tm", "backward", accuracy, dtype, device
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
                backend_utils.tensor_to_ptr(grad_eps),
                backend_utils.tensor_to_ptr(grad_sigma),
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
                pml_y0,  # Use original PML boundaries for adjoint propagation
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
                    callback_gradients["epsilon"] = grad_eps
                    callback_gradients["sigma"] = grad_sigma
                
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
        #        fd_pad, pml_width, models, backward_callback, callback_frequency,
        #        Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
        
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
            None, None, None, None,  # ay, by, ay_h, by_h
            None, None, None, None,  # ax, bx, ax_h, bx_h
            None, None, None, None,  # ky, ky_h, kx, kx_h
            None, None,  # sources_i, receivers_i
            None, None, None,  # rdy, rdx, dt
            None, None, None, None,  # nt, n_shots, ny, nx
            None, None,  # n_sources, n_receivers
            None,  # step_ratio
            None,  # accuracy
            None, None, None,  # ca_batched, cb_batched, cq_batched
            None, None, None, None,  # pml_y0, pml_x0, pml_y1, pml_x1
            None, None, None,  # fd_pad, pml_width, models
            None, None, None,  # forward_callback, backward_callback, callback_frequency
            None, None, None,  # storage_mode_str, storage_path, storage_compression
            None, None, None,  # Ey, Hx, Hz
            None, None, None, None,  # m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
            None,  # n_threads
        )



class Maxwell3DForwardFunc(torch.autograd.Function):
    """Autograd function for 3D Maxwell forward with ASM snapshot storage."""

    @staticmethod
    def forward(
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        az: torch.Tensor,
        bz: torch.Tensor,
        az_h: torch.Tensor,
        bz_h: torch.Tensor,
        ay: torch.Tensor,
        by: torch.Tensor,
        ay_h: torch.Tensor,
        by_h: torch.Tensor,
        ax: torch.Tensor,
        bx: torch.Tensor,
        ax_h: torch.Tensor,
        bx_h: torch.Tensor,
        kz: torch.Tensor,
        kzh: torch.Tensor,
        ky: torch.Tensor,
        kyh: torch.Tensor,
        kx: torch.Tensor,
        kxh: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        rdz: float,
        rdy: float,
        rdx: float,
        dt: float,
        nt: int,
        n_shots: int,
        nz: int,
        ny: int,
        nx: int,
        n_sources: int,
        n_receivers: int,
        step_ratio: int,
        accuracy: int,
        ca_batched: bool,
        cb_batched: bool,
        cq_batched: bool,
        pml_z0: int,
        pml_y0: int,
        pml_x0: int,
        pml_z1: int,
        pml_y1: int,
        pml_x1: int,
        storage_mode_str: str,
        storage_path: str,
        storage_compression: Union[bool, str],
        source_component: int,
        receiver_component: int,
        Ex: torch.Tensor,
        Ey: torch.Tensor,
        Ez: torch.Tensor,
        Hx: torch.Tensor,
        Hy: torch.Tensor,
        Hz: torch.Tensor,
        m_Hz_y: torch.Tensor,
        m_Hy_z: torch.Tensor,
        m_Hx_z: torch.Tensor,
        m_Hz_x: torch.Tensor,
        m_Hy_x: torch.Tensor,
        m_Hx_y: torch.Tensor,
        m_Ey_z: torch.Tensor,
        m_Ez_y: torch.Tensor,
        m_Ez_x: torch.Tensor,
        m_Ex_z: torch.Tensor,
        m_Ex_y: torch.Tensor,
        m_Ey_x: torch.Tensor,
        n_threads: int,
    ) -> tuple[Any, ...]:
        from . import backend_utils

        import ctypes

        device = Ex.device
        dtype = Ex.dtype

        if device.type != "cuda":
            raise NotImplementedError(
                "Maxwell3DForwardFunc is only supported on CUDA."
            )

        ca_requires_grad = ca.requires_grad
        cb_requires_grad = cb.requires_grad
        needs_grad = ca_requires_grad or cb_requires_grad

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        backward_storage_tensors: list[torch.Tensor] = []
        backward_storage_objects: list[Any] = []
        backward_storage_filename_arrays: list[Any] = []
        storage_mode = STORAGE_NONE
        shot_bytes_uncomp = 0

        if needs_grad:
            storage_mode = storage_mode_to_int(storage_mode_str)
            if storage_mode == STORAGE_NONE:
                raise ValueError(
                    "storage_mode='none' is not compatible with gradient_mode='snapshot'."
                )

            _, store_dtype, _ = _resolve_storage_compression(
                storage_compression,
                dtype,
                device,
                context="storage_compression",
                allow_fp8=False,
            )

            shot_numel = nz * ny * nx
            shot_bytes_uncomp = shot_numel * store_dtype.itemsize
            num_steps_stored = (nt + step_ratio - 1) // step_ratio

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
                            nz,
                            ny,
                            nx,
                            device=device,
                            dtype=store_dtype,
                        )
                    elif storage_mode == STORAGE_CPU:
                        store_1 = torch.empty(
                            _CPU_STORAGE_BUFFERS,
                            n_shots,
                            nz,
                            ny,
                            nx,
                            device=device,
                            dtype=store_dtype,
                        )
                        store_3 = torch.empty(
                            num_steps_stored,
                            n_shots,
                            shot_numel,
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
                            n_shots, nz, ny, nx, device=device, dtype=store_dtype
                        )
                        if is_cuda:
                            store_3 = torch.empty(
                                n_shots,
                                shot_numel,
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

            ex_store_1, ex_store_3, ex_filenames_ptr = alloc_storage(ca_requires_grad)
            ey_store_1, ey_store_3, ey_filenames_ptr = alloc_storage(ca_requires_grad)
            ez_store_1, ez_store_3, ez_filenames_ptr = alloc_storage(ca_requires_grad)
            curlx_store_1, curlx_store_3, curlx_filenames_ptr = alloc_storage(
                cb_requires_grad
            )
            curly_store_1, curly_store_3, curly_filenames_ptr = alloc_storage(
                cb_requires_grad
            )
            curlz_store_1, curlz_store_3, curlz_filenames_ptr = alloc_storage(
                cb_requires_grad
            )
        else:
            ex_store_1 = ey_store_1 = ez_store_1 = torch.empty(0)
            ex_store_3 = ey_store_3 = ez_store_3 = torch.empty(0)
            curlx_store_1 = curly_store_1 = curlz_store_1 = torch.empty(0)
            curlx_store_3 = curly_store_3 = curlz_store_3 = torch.empty(0)
            ex_filenames_ptr = ey_filenames_ptr = ez_filenames_ptr = 0
            curlx_filenames_ptr = curly_filenames_ptr = curlz_filenames_ptr = 0

        device_idx = device.index if device.index is not None else 0

        forward_func = backend_utils.get_backend_function(
            "maxwell_3d", "forward_with_storage", accuracy, dtype, device
        )

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
            backend_utils.tensor_to_ptr(m_Hz_y),
            backend_utils.tensor_to_ptr(m_Hy_z),
            backend_utils.tensor_to_ptr(m_Hx_z),
            backend_utils.tensor_to_ptr(m_Hz_x),
            backend_utils.tensor_to_ptr(m_Hy_x),
            backend_utils.tensor_to_ptr(m_Hx_y),
            backend_utils.tensor_to_ptr(m_Ey_z),
            backend_utils.tensor_to_ptr(m_Ez_y),
            backend_utils.tensor_to_ptr(m_Ez_x),
            backend_utils.tensor_to_ptr(m_Ex_z),
            backend_utils.tensor_to_ptr(m_Ex_y),
            backend_utils.tensor_to_ptr(m_Ey_x),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
            backend_utils.tensor_to_ptr(ex_store_1),
            backend_utils.tensor_to_ptr(ex_store_3),
            ex_filenames_ptr,
            backend_utils.tensor_to_ptr(ey_store_1),
            backend_utils.tensor_to_ptr(ey_store_3),
            ey_filenames_ptr,
            backend_utils.tensor_to_ptr(ez_store_1),
            backend_utils.tensor_to_ptr(ez_store_3),
            ez_filenames_ptr,
            backend_utils.tensor_to_ptr(curlx_store_1),
            backend_utils.tensor_to_ptr(curlx_store_3),
            curlx_filenames_ptr,
            backend_utils.tensor_to_ptr(curly_store_1),
            backend_utils.tensor_to_ptr(curly_store_3),
            curly_filenames_ptr,
            backend_utils.tensor_to_ptr(curlz_store_1),
            backend_utils.tensor_to_ptr(curlz_store_3),
            curlz_filenames_ptr,
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
            backend_utils.tensor_to_ptr(kzh),
            backend_utils.tensor_to_ptr(ky),
            backend_utils.tensor_to_ptr(kyh),
            backend_utils.tensor_to_ptr(kx),
            backend_utils.tensor_to_ptr(kxh),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            rdz,
            rdy,
            rdx,
            dt,
            nt,
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
            ca_batched,
            cb_batched,
            cq_batched,
            0,
            pml_z0,
            pml_y0,
            pml_x0,
            pml_z1,
            pml_y1,
            pml_x1,
            source_component,
            receiver_component,
            n_threads,
            device_idx,
        )

        ctx_data = {
            "backward_storage_tensors": backward_storage_tensors,
            "backward_storage_objects": backward_storage_objects,
            "backward_storage_filename_arrays": backward_storage_filename_arrays,
            "storage_mode": storage_mode,
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
        }
        ctx_handle = _register_ctx_handle(ctx_data)

        return (
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            m_Hz_y,
            m_Hy_z,
            m_Hx_z,
            m_Hz_x,
            m_Hy_x,
            m_Hx_y,
            m_Ey_z,
            m_Ez_y,
            m_Ez_x,
            m_Ex_z,
            m_Ex_y,
            m_Ey_x,
            receiver_amplitudes,
            ctx_handle,
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], outputs: tuple[Any, ...]) -> None:
        (
            ca,
            cb,
            cq,
            _source_amplitudes_scaled,
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
            kzh,
            ky,
            kyh,
            kx,
            kxh,
            sources_i,
            receivers_i,
            rdz,
            rdy,
            rdx,
            dt,
            nt,
            n_shots,
            nz,
            ny,
            nx,
            n_sources,
            n_receivers,
            step_ratio,
            accuracy,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_z0,
            pml_y0,
            pml_x0,
            pml_z1,
            pml_y1,
            pml_x1,
            _storage_mode_str,
            _storage_path,
            _storage_compression,
            source_component,
            receiver_component,
            _Ex,
            _Ey,
            _Ez,
            _Hx,
            _Hy,
            _Hz,
            _m_Hz_y,
            _m_Hy_z,
            _m_Hx_z,
            _m_Hz_x,
            _m_Hy_x,
            _m_Hx_y,
            _m_Ey_z,
            _m_Ez_y,
            _m_Ez_x,
            _m_Ex_z,
            _m_Ex_y,
            _m_Ey_x,
            n_threads,
        ) = inputs

        if len(outputs) != 20:
            raise RuntimeError(
                "Maxwell3DForwardFunc expected a context handle output for setup_context."
            )
        ctx_handle = outputs[-1]
        if not isinstance(ctx_handle, torch.Tensor):
            raise RuntimeError("Maxwell3DForwardFunc context handle must be a Tensor.")

        ctx_handle_id = int(ctx_handle.item())
        ctx_data = _get_ctx_handle(ctx_handle_id)
        ctx._ctx_handle_id = ctx_handle_id
        backward_storage_tensors = ctx_data["backward_storage_tensors"]

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
            kzh,
            ky,
            kyh,
            kx,
            kxh,
            sources_i,
            receivers_i,
            *backward_storage_tensors,
        )
        ctx.backward_storage_objects = ctx_data["backward_storage_objects"]
        ctx.backward_storage_filename_arrays = ctx_data["backward_storage_filename_arrays"]
        ctx.rdz = rdz
        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.nz = nz
        ctx.ny = ny
        ctx.nx = nx
        ctx.n_sources = n_sources
        ctx.n_receivers = n_receivers
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.pml_z0 = pml_z0
        ctx.pml_y0 = pml_y0
        ctx.pml_x0 = pml_x0
        ctx.pml_z1 = pml_z1
        ctx.pml_y1 = pml_y1
        ctx.pml_x1 = pml_x1
        ctx.ca_requires_grad = ctx_data["ca_requires_grad"]
        ctx.cb_requires_grad = ctx_data["cb_requires_grad"]
        ctx.storage_mode = ctx_data["storage_mode"]
        ctx.shot_bytes_uncomp = ctx_data["shot_bytes_uncomp"]
        ctx.source_component = int(source_component)
        ctx.receiver_component = int(receiver_component)
        ctx.n_threads = n_threads

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Optional[torch.Tensor], ...]:
        from . import backend_utils

        grad_outputs = list(grad_outputs)
        if len(grad_outputs) == 20:
            grad_outputs.pop()  # drop context handle grad

        (
            _grad_ex,
            _grad_ey,
            _grad_ez,
            _grad_hx,
            _grad_hy,
            _grad_hz,
            _grad_m_hz_y,
            _grad_m_hy_z,
            _grad_m_hx_z,
            _grad_m_hz_x,
            _grad_m_hy_x,
            _grad_m_hx_y,
            _grad_m_ey_z,
            _grad_m_ez_y,
            _grad_m_ez_x,
            _grad_m_ex_z,
            _grad_m_ex_y,
            _grad_m_ey_x,
            grad_r,
        ) = grad_outputs

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        az, bz, az_h, bz_h = saved[3], saved[4], saved[5], saved[6]
        ay, by, ay_h, by_h = saved[7], saved[8], saved[9], saved[10]
        ax, bx, ax_h, bx_h = saved[11], saved[12], saved[13], saved[14]
        kz, kzh, ky, kyh, kx, kxh = (
            saved[15],
            saved[16],
            saved[17],
            saved[18],
            saved[19],
            saved[20],
        )
        sources_i, receivers_i = saved[21], saved[22]
        storage_tensors = list(saved[23:])

        (
            ex_store_1,
            ex_store_3,
            ey_store_1,
            ey_store_3,
            ez_store_1,
            ez_store_3,
            curlx_store_1,
            curlx_store_3,
            curly_store_1,
            curly_store_3,
            curlz_store_1,
            curlz_store_3,
        ) = storage_tensors

        device = ca.device
        dtype = ca.dtype

        rdz = ctx.rdz
        rdy = ctx.rdy
        rdx = ctx.rdx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        nz = ctx.nz
        ny = ctx.ny
        nx = ctx.nx
        n_sources = ctx.n_sources
        n_receivers = ctx.n_receivers
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        ca_batched = ctx.ca_batched
        cb_batched = ctx.cb_batched
        cq_batched = ctx.cq_batched
        pml_z0 = ctx.pml_z0
        pml_y0 = ctx.pml_y0
        pml_x0 = ctx.pml_x0
        pml_z1 = ctx.pml_z1
        pml_y1 = ctx.pml_y1
        pml_x1 = ctx.pml_x1
        ca_requires_grad = ctx.ca_requires_grad
        cb_requires_grad = ctx.cb_requires_grad
        storage_mode = ctx.storage_mode
        shot_bytes_uncomp = ctx.shot_bytes_uncomp
        source_component = ctx.source_component
        receiver_component = ctx.receiver_component
        n_threads = ctx.n_threads

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

        m_lambda_ey_z = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_ez_y = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_ez_x = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_ex_z = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_ex_y = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_ey_x = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)

        m_lambda_hz_y = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_hy_z = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_hx_z = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_hz_x = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_hy_x = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        m_lambda_hx_y = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)

        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            if ca_batched:
                grad_ca = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            else:
                grad_ca = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            grad_ca_shot = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            if cb_batched:
                grad_cb = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            else:
                grad_cb = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            grad_cb_shot = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad or cb_requires_grad:
            if ca_batched or cb_batched:
                grad_eps = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            else:
                grad_eps = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(nz, ny, nx, device=device, dtype=dtype)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        if storage_mode == STORAGE_DISK:
            import ctypes

            ex_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[0], ctypes.c_void_p
            )
            ey_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[1], ctypes.c_void_p
            )
            ez_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[2], ctypes.c_void_p
            )
            curlx_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[3], ctypes.c_void_p
            )
            curly_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[4], ctypes.c_void_p
            )
            curlz_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[5], ctypes.c_void_p
            )
        else:
            ex_filenames_ptr = 0
            ey_filenames_ptr = 0
            ez_filenames_ptr = 0
            curlx_filenames_ptr = 0
            curly_filenames_ptr = 0
            curlz_filenames_ptr = 0

        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        backward_func = backend_utils.get_backend_function(
            "maxwell_3d", "backward", accuracy, dtype, device
        )

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
            backend_utils.tensor_to_ptr(ex_store_1),
            backend_utils.tensor_to_ptr(ex_store_3),
            ex_filenames_ptr,
            backend_utils.tensor_to_ptr(ey_store_1),
            backend_utils.tensor_to_ptr(ey_store_3),
            ey_filenames_ptr,
            backend_utils.tensor_to_ptr(ez_store_1),
            backend_utils.tensor_to_ptr(ez_store_3),
            ez_filenames_ptr,
            backend_utils.tensor_to_ptr(curlx_store_1),
            backend_utils.tensor_to_ptr(curlx_store_3),
            curlx_filenames_ptr,
            backend_utils.tensor_to_ptr(curly_store_1),
            backend_utils.tensor_to_ptr(curly_store_3),
            curly_filenames_ptr,
            backend_utils.tensor_to_ptr(curlz_store_1),
            backend_utils.tensor_to_ptr(curlz_store_3),
            curlz_filenames_ptr,
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
            backend_utils.tensor_to_ptr(kzh),
            backend_utils.tensor_to_ptr(ky),
            backend_utils.tensor_to_ptr(kyh),
            backend_utils.tensor_to_ptr(kx),
            backend_utils.tensor_to_ptr(kxh),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            rdz,
            rdy,
            rdx,
            dt,
            nt,
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
            ca_batched,
            cb_batched,
            cq_batched,
            nt,
            pml_z0,
            pml_y0,
            pml_x0,
            pml_z1,
            pml_y1,
            pml_x1,
            source_component,
            receiver_component,
            n_threads,
            device_idx,
        )

        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None

        if ca_requires_grad and not ca_batched:
            grad_ca = grad_ca.unsqueeze(0)
        if cb_requires_grad and not cb_batched:
            grad_cb = grad_cb.unsqueeze(0)

        _release_ctx_handle(getattr(ctx, "_ctx_handle_id", None))
        return (
            grad_ca if ca_requires_grad else None,  # ca
            grad_cb if cb_requires_grad else None,  # cb
            None,  # cq
            grad_f_flat,  # source_amplitudes_scaled
            None, None, None, None,  # az, bz, az_h, bz_h
            None, None, None, None,  # ay, by, ay_h, by_h
            None, None, None, None,  # ax, bx, ax_h, bx_h
            None, None, None, None, None, None,  # kz, kzh, ky, kyh, kx, kxh
            None, None,  # sources_i, receivers_i
            None, None, None,  # rdz, rdy, rdx
            None, None, None, None, None, None, None, None,  # dt, nt, n_shots, nz, ny, nx, n_sources, n_receivers
            None, None,  # step_ratio, accuracy
            None, None, None,  # ca_batched, cb_batched, cq_batched
            None, None, None, None, None, None,  # pml_z0, pml_y0, pml_x0, pml_z1, pml_y1, pml_x1
            None, None, None,  # storage_mode_str, storage_path, storage_compression
            None, None,  # source_component, receiver_component
            None, None, None, None, None, None,  # Ex, Ey, Ez, Hx, Hy, Hz
            None, None, None, None, None, None,  # m_Hz_y, m_Hy_z, m_Hx_z, m_Hz_x, m_Hy_x, m_Hx_y
            None, None, None, None, None, None,  # m_Ey_z, m_Ez_y, m_Ez_x, m_Ex_z, m_Ex_y, m_Ey_x
            None,  # n_threads
        )


# =============================================================================
# 3D Maxwell FDTD (Python backend)
# =============================================================================


class Maxwell3D(torch.nn.Module):
    """3D Maxwell equations solver (Ex, Ey, Ez, Hx, Hy, Hz) using FDTD."""

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
        if epsilon_requires_grad is not None and not isinstance(epsilon_requires_grad, bool):
            raise TypeError(
                f"epsilon_requires_grad must be bool or None, "
                f"got {type(epsilon_requires_grad).__name__}",
            )
        if not isinstance(epsilon, torch.Tensor):
            raise TypeError(
                f"epsilon must be torch.Tensor, got {type(epsilon).__name__}",
            )
        if sigma_requires_grad is not None and not isinstance(sigma_requires_grad, bool):
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

        if epsilon.ndim != 3:
            raise ValueError("epsilon must be a 3D tensor [nz, ny, nx].")
        if sigma.shape != epsilon.shape or mu.shape != epsilon.shape:
            raise ValueError("sigma and mu must have the same shape as epsilon.")

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
        m_Hz_y_0: Optional[torch.Tensor] = None,
        m_Hy_z_0: Optional[torch.Tensor] = None,
        m_Hx_z_0: Optional[torch.Tensor] = None,
        m_Hz_x_0: Optional[torch.Tensor] = None,
        m_Hy_x_0: Optional[torch.Tensor] = None,
        m_Hx_y_0: Optional[torch.Tensor] = None,
        m_Ey_z_0: Optional[torch.Tensor] = None,
        m_Ez_y_0: Optional[torch.Tensor] = None,
        m_Ez_x_0: Optional[torch.Tensor] = None,
        m_Ex_z_0: Optional[torch.Tensor] = None,
        m_Ex_y_0: Optional[torch.Tensor] = None,
        m_Ey_x_0: Optional[torch.Tensor] = None,
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
        gradient_mode: str = "snapshot",
        storage_mode: str = "device",
        storage_path: str = ".",
        storage_compression: Union[bool, str] = False,
        storage_bytes_limit_device: Optional[int] = None,
        storage_bytes_limit_host: Optional[int] = None,
        storage_chunk_steps: int = 0,
        boundary_width: int = 0,
        source_component: str = "Ez",
        receiver_component: Optional[str] = None,
        n_threads: Optional[int] = None,
    ):
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
            m_Hz_y_0,
            m_Hy_z_0,
            m_Hx_z_0,
            m_Hz_x_0,
            m_Hy_x_0,
            m_Hx_y_0,
            m_Ey_z_0,
            m_Ez_y_0,
            m_Ez_x_0,
            m_Ex_z_0,
            m_Ex_y_0,
            m_Ey_x_0,
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
            gradient_mode,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            boundary_width,
            source_component,
            receiver_component,
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
    m_Hz_y_0: Optional[torch.Tensor] = None,
    m_Hy_z_0: Optional[torch.Tensor] = None,
    m_Hx_z_0: Optional[torch.Tensor] = None,
    m_Hz_x_0: Optional[torch.Tensor] = None,
    m_Hy_x_0: Optional[torch.Tensor] = None,
    m_Hx_y_0: Optional[torch.Tensor] = None,
    m_Ey_z_0: Optional[torch.Tensor] = None,
    m_Ez_y_0: Optional[torch.Tensor] = None,
    m_Ez_x_0: Optional[torch.Tensor] = None,
    m_Ex_z_0: Optional[torch.Tensor] = None,
    m_Ex_y_0: Optional[torch.Tensor] = None,
    m_Ey_x_0: Optional[torch.Tensor] = None,
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
    gradient_mode: str = "snapshot",
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: Union[bool, str] = False,
    storage_bytes_limit_device: Optional[int] = None,
    storage_bytes_limit_host: Optional[int] = None,
    storage_chunk_steps: int = 0,
    boundary_width: int = 0,
    source_component: str = "Ez",
    receiver_component: Optional[str] = None,
    n_threads: Optional[int] = None,
):
    """3D Maxwell equations solver (Python backend reference).

    Field and model tensors use [nz, ny, nx] ordering with z as the slowest
    dimension, matching the single-pass z-scan layout described in GPU 3DFD.
    """
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)

    if source_location is not None and source_location.numel() > 0:
        if source_location.shape[-1] != 3:
            raise RuntimeError("source_location must have shape [..., 3] for [z, y, x].")
        if source_location[..., 0].max() >= epsilon.shape[-3]:
            raise RuntimeError(
                f"Source location dim 0 must be less than {epsilon.shape[-3]}"
            )
        if source_location[..., 1].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Source location dim 1 must be less than {epsilon.shape[-2]}"
            )
        if source_location[..., 2].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Source location dim 2 must be less than {epsilon.shape[-1]}"
            )

    if receiver_location is not None and receiver_location.numel() > 0:
        if receiver_location.shape[-1] != 3:
            raise RuntimeError(
                "receiver_location must have shape [..., 3] for [z, y, x]."
            )
        if receiver_location[..., 0].max() >= epsilon.shape[-3]:
            raise RuntimeError(
                f"Receiver location dim 0 must be less than {epsilon.shape[-3]}"
            )
        if receiver_location[..., 1].max() >= epsilon.shape[-2]:
            raise RuntimeError(
                f"Receiver location dim 1 must be less than {epsilon.shape[-2]}"
            )
        if receiver_location[..., 2].max() >= epsilon.shape[-1]:
            raise RuntimeError(
                f"Receiver location dim 2 must be less than {epsilon.shape[-1]}"
            )

    if not isinstance(callback_frequency, int):
        raise TypeError("callback_frequency must be an int.")
    if callback_frequency <= 0:
        raise ValueError("callback_frequency must be positive.")

    if isinstance(grid_spacing, (int, float)):
        grid_spacing_list = [float(grid_spacing)] * 3
    else:
        grid_spacing_list = list(grid_spacing)
    if len(grid_spacing_list) != 3:
        raise ValueError("grid_spacing must be a float or sequence [dz, dy, dx].")

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

    result = maxwell3d_func(
        python_backend,
        epsilon,
        sigma,
        mu,
        grid_spacing_list,
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
        m_Hz_y_0,
        m_Hy_z_0,
        m_Hx_z_0,
        m_Hz_x_0,
        m_Hy_x_0,
        m_Hx_y_0,
        m_Ey_z_0,
        m_Ez_y_0,
        m_Ez_x_0,
        m_Ex_z_0,
        m_Ex_y_0,
        m_Ey_x_0,
        nt_internal,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        forward_callback,
        backward_callback,
        callback_frequency,
        gradient_mode,
        storage_mode,
        storage_path,
        storage_compression,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        boundary_width,
        source_component,
        receiver_component,
        n_threads,
    )

    (
        Ex_out,
        Ey_out,
        Ez_out,
        Hx_out,
        Hy_out,
        Hz_out,
        m_Hz_y_out,
        m_Hy_z_out,
        m_Hx_z_out,
        m_Hz_x_out,
        m_Hy_x_out,
        m_Hx_y_out,
        m_Ey_z_out,
        m_Ez_y_out,
        m_Ez_x_out,
        m_Ex_z_out,
        m_Ex_y_out,
        m_Ey_x_out,
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
        m_Hz_y_out,
        m_Hy_z_out,
        m_Hx_z_out,
        m_Hz_x_out,
        m_Hy_x_out,
        m_Hx_y_out,
        m_Ey_z_out,
        m_Ez_y_out,
        m_Ez_x_out,
        m_Ex_z_out,
        m_Ex_y_out,
        m_Ey_x_out,
        receiver_amplitudes,
    )


_update_E3d_jit: Optional[Callable] = None
_update_E3d_compile: Optional[Callable] = None
_update_H3d_jit: Optional[Callable] = None
_update_H3d_compile: Optional[Callable] = None
_update_E3d_opt: Optional[Callable] = None
_update_H3d_opt: Optional[Callable] = None


def maxwell3d_func(python_backend: Union[bool, str], *args):
    """Dispatch to Python or C/CUDA backend for 3D Maxwell propagation."""
    global _update_E3d_jit, _update_E3d_compile, _update_E3d_opt
    global _update_H3d_jit, _update_H3d_compile, _update_H3d_opt

    use_python = python_backend
    if not use_python:
        try:
            from . import backend_utils
            if not backend_utils.is_backend_available():
                import warnings

                warnings.warn(
                    "C/CUDA backend not available, falling back to Python backend.",
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
            mode = "eager"
        elif isinstance(python_backend, str):
            mode = python_backend.lower()
        else:
            raise TypeError(
                f"python_backend must be bool or str, but got {type(python_backend)}"
            )

        if mode == "jit":
            
            if _update_E3d_jit is None:
                _update_E3d_jit = torch.jit.script(update_E_3d)
            _update_E3d_opt = _update_E3d_jit
            if _update_H3d_jit is None:
                _update_H3d_jit = torch.jit.script(update_H_3d)
            _update_H3d_opt = _update_H3d_jit
        elif mode == "compile":
            if _update_E3d_compile is None:
                _update_E3d_compile = torch.compile(update_E_3d, fullgraph=True)
            _update_E3d_opt = _update_E3d_compile
            if _update_H3d_compile is None:
                _update_H3d_compile = torch.compile(update_H_3d, fullgraph=True)
            _update_H3d_opt = _update_H3d_compile
        elif mode == "eager":
            _update_E3d_opt = update_E_3d
            _update_H3d_opt = update_H_3d
        else:
            raise ValueError(f"Unknown python_backend value {mode!r}.")

        return maxwell3d_python(*args)
    else:
        return maxwell3d_c_cuda(*args)


def maxwell3d_python(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Sequence[float],
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
    m_Hz_y_0: Optional[torch.Tensor],
    m_Hy_z_0: Optional[torch.Tensor],
    m_Hx_z_0: Optional[torch.Tensor],
    m_Hz_x_0: Optional[torch.Tensor],
    m_Hy_x_0: Optional[torch.Tensor],
    m_Hx_y_0: Optional[torch.Tensor],
    m_Ey_z_0: Optional[torch.Tensor],
    m_Ez_y_0: Optional[torch.Tensor],
    m_Ez_x_0: Optional[torch.Tensor],
    m_Ex_z_0: Optional[torch.Tensor],
    m_Ex_y_0: Optional[torch.Tensor],
    m_Ey_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    gradient_mode: str,
    storage_mode: str,
    storage_path: str,
    storage_compression: Union[bool, str],
    storage_bytes_limit_device: Optional[int],
    storage_bytes_limit_host: Optional[int],
    storage_chunk_steps: int,
    boundary_width: int,
    source_component: str,
    receiver_component: Optional[str],
    n_threads: Optional[int] = None,
):
    """Reference Python backend for 3D Maxwell propagation (forward only)."""
    from .padding import create_or_pad

    _ = (
        model_gradient_sampling_interval,
        save_snapshots,
        backward_callback,
        storage_path,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        boundary_width,
        n_threads,
    )

    if epsilon.ndim != 3:
        raise RuntimeError("epsilon must be 3D [nz, ny, nx].")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon.")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon.")

    gradient_mode_str = gradient_mode.lower()
    if gradient_mode_str != "snapshot":
        raise NotImplementedError(
            f"gradient_mode={gradient_mode!r} is not implemented yet; "
            "only 'snapshot' is supported."
        )

    storage_mode_str = storage_mode.lower()
    if storage_mode_str in {"cpu", "disk"}:
        raise ValueError(
            "python_backend does not support storage_mode='cpu' or 'disk'. "
            "Use storage_mode='device' or 'none'."
        )
    storage_kind = _normalize_storage_compression(storage_compression)
    if storage_kind != "none":
        raise NotImplementedError(
            "storage_compression is not implemented yet; set storage_compression=False."
        )

    if isinstance(pml_width, int):
        pml_width_list = [pml_width] * 6
    else:
        pml_width_list = list(pml_width)
        if len(pml_width_list) == 1:
            pml_width_list = pml_width_list * 6
        elif len(pml_width_list) == 3:
            pml_width_list = [
                pml_width_list[0],
                pml_width_list[0],
                pml_width_list[1],
                pml_width_list[1],
                pml_width_list[2],
                pml_width_list[2],
            ]
        elif len(pml_width_list) != 6:
            raise ValueError(
                "pml_width must be int or sequence of length 1, 3, or 6."
            )

    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided.")
        nt = source_amplitude.shape[-1]
    nt_steps: int = int(nt)

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

    model_nz, model_ny, model_nx = epsilon.shape
    padded_nz = model_nz + total_pad[0] + total_pad[1]
    padded_ny = model_ny + total_pad[2] + total_pad[3]
    padded_nx = model_nx + total_pad[4] + total_pad[5]

    device = epsilon.device
    dtype = epsilon.dtype

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
                field_0, fd_pad_list, device, dtype, size_with_batch, mode="constant"
            )
        return torch.zeros(size_with_batch, device=device, dtype=dtype)

    Ex = init_wavefield(Ex_0)
    Ey = init_wavefield(Ey_0)
    Ez = init_wavefield(Ez_0)
    Hx = init_wavefield(Hx_0)
    Hy = init_wavefield(Hy_0)
    Hz = init_wavefield(Hz_0)

    m_Hz_y = init_wavefield(m_Hz_y_0)
    m_Hy_z = init_wavefield(m_Hy_z_0)
    m_Hx_z = init_wavefield(m_Hx_z_0)
    m_Hz_x = init_wavefield(m_Hz_x_0)
    m_Hy_x = init_wavefield(m_Hy_x_0)
    m_Hx_y = init_wavefield(m_Hx_y_0)

    m_Ey_z = init_wavefield(m_Ey_z_0)
    m_Ez_y = init_wavefield(m_Ez_y_0)
    m_Ez_x = init_wavefield(m_Ez_x_0)
    m_Ex_z = init_wavefield(m_Ex_z_0)
    m_Ex_y = init_wavefield(m_Ex_y_0)
    m_Ey_x = init_wavefield(m_Ey_x_0)

    def zero_interior_3d(
        tensor: torch.Tensor,
        fd_pad: Sequence[int],
        pml_width: Sequence[int],
        dim: int,
    ) -> None:
        shape = tensor.shape[1:]
        interior_start = fd_pad[dim * 2] + pml_width[dim * 2]
        interior_end = shape[dim] - pml_width[dim * 2 + 1] - fd_pad[dim * 2 + 1]

        if dim == 0:
            tensor[:, interior_start:interior_end, :, :].fill_(0)
        elif dim == 1:
            tensor[:, :, interior_start:interior_end, :].fill_(0)
        else:
            tensor[:, :, :, interior_start:interior_end].fill_(0)

    pml_aux_dims = [
        (m_Hz_y, 1),
        (m_Hy_z, 0),
        (m_Hx_z, 0),
        (m_Hz_x, 2),
        (m_Hy_x, 2),
        (m_Hx_y, 1),
        (m_Ey_z, 0),
        (m_Ez_y, 1),
        (m_Ez_x, 2),
        (m_Ex_z, 0),
        (m_Ex_y, 1),
        (m_Ey_x, 2),
    ]
    for wf, dim in pml_aux_dims:
        zero_interior_3d(wf, fd_pad_list, pml_width_list, dim)

    pml_profiles, kappa_profiles = staggered.set_pml_profiles_3d(
        pml_width=pml_width_list,
        accuracy=stencil,
        fd_pad=fd_pad_list,
        dt=dt,
        grid_spacing=list(grid_spacing),
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
    ) = pml_profiles
    kappa_z, kappa_z_h, kappa_y, kappa_y_h, kappa_x, kappa_x_h = kappa_profiles

    dz, dy, dx = grid_spacing
    rdz = torch.tensor(1.0 / dz, device=device, dtype=dtype)
    rdy = torch.tensor(1.0 / dy, device=device, dtype=dtype)
    rdx = torch.tensor(1.0 / dx, device=device, dtype=dtype)
    dt_tensor = torch.tensor(dt, device=device, dtype=dtype)

    flat_model_shape = padded_nz * padded_ny * padded_nx

    if source_location is not None and source_location.numel() > 0:
        source_z = source_location[..., 0] + total_pad[0]
        source_y = source_location[..., 1] + total_pad[2]
        source_x = source_location[..., 2] + total_pad[4]
        sources_i = (
            source_z * (padded_ny * padded_nx) + source_y * padded_nx + source_x
        ).long().contiguous()
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        receiver_z = receiver_location[..., 0] + total_pad[0]
        receiver_y = receiver_location[..., 1] + total_pad[2]
        receiver_x = receiver_location[..., 2] + total_pad[4]
        receivers_i = (
            receiver_z * (padded_ny * padded_nx) + receiver_y * padded_nx + receiver_x
        ).long().contiguous()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    if n_receivers > 0:
        receiver_amplitudes = torch.zeros(
            nt_steps, n_shots, n_receivers, device=device, dtype=dtype
        )
    else:
        receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

    def _normalize_component(name: Optional[str], default: str, allow_h: bool) -> str:
        if name is None:
            name = default
        if not isinstance(name, str):
            raise TypeError("component name must be a string.")
        comp = name.strip().lower()
        allowed = {"ex", "ey", "ez"}
        if allow_h:
            allowed |= {"hx", "hy", "hz"}
        if comp not in allowed:
            raise ValueError(f"Unknown component {name!r}.")
        return comp

    src_component = _normalize_component(source_component, "ez", allow_h=False)
    rec_component = _normalize_component(receiver_component, src_component, allow_h=True)

    callback_models = {
        "epsilon": epsilon_padded,
        "sigma": sigma_padded,
        "mu": mu_padded,
        "ca": ca,
        "cb": cb,
        "cq": cq,
    }
    callback_fd_pad = fd_pad_list

    source_coeff = -1.0 / (dx * dy * dz)

    for step in range(nt_steps):
        if forward_callback is not None and step % callback_frequency == 0:
            callback_wavefields = {
                "Ex": Ex,
                "Ey": Ey,
                "Ez": Ez,
                "Hx": Hx,
                "Hy": Hy,
                "Hz": Hz,
                "m_Hz_y": m_Hz_y,
                "m_Hy_z": m_Hy_z,
                "m_Hx_z": m_Hx_z,
                "m_Hz_x": m_Hz_x,
                "m_Hy_x": m_Hy_x,
                "m_Hx_y": m_Hx_y,
                "m_Ey_z": m_Ey_z,
                "m_Ez_y": m_Ez_y,
                "m_Ez_x": m_Ez_x,
                "m_Ex_z": m_Ex_z,
                "m_Ex_y": m_Ex_y,
                "m_Ey_x": m_Ey_x,
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
                grid_spacing=[dz, dy, dx],
            )
            forward_callback(callback_state)

        Hx, Hy, Hz, m_Ey_z, m_Ez_y, m_Ez_x, m_Ex_z, m_Ex_y, m_Ey_x = _update_H3d_opt(
            cq,
            Hx,
            Hy,
            Hz,
            Ex,
            Ey,
            Ez,
            m_Ey_z,
            m_Ez_y,
            m_Ez_x,
            m_Ex_z,
            m_Ex_y,
            m_Ey_x,
            kappa_z,
            kappa_z_h,
            kappa_y,
            kappa_y_h,
            kappa_x,
            kappa_x_h,
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
            rdz,
            rdy,
            rdx,
            dt_tensor,
            stencil,
        )

        Ex, Ey, Ez, m_Hz_y, m_Hy_z, m_Hx_z, m_Hz_x, m_Hy_x, m_Hx_y = _update_E3d_opt(
            ca,
            cb,
            Hx,
            Hy,
            Hz,
            Ex,
            Ey,
            Ez,
            m_Hz_y,
            m_Hy_z,
            m_Hx_z,
            m_Hz_x,
            m_Hy_x,
            m_Hx_y,
            kappa_z,
            kappa_z_h,
            kappa_y,
            kappa_y_h,
            kappa_x,
            kappa_x_h,
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
            rdz,
            rdy,
            rdx,
            dt_tensor,
            stencil,
        )

        if source_amplitude is not None and source_amplitude.numel() > 0 and n_sources > 0:
            src_amp = source_amplitude[:, :, step]
            cb_flat = cb.reshape(1, flat_model_shape).expand(n_shots, -1)
            cb_at_src = cb_flat.gather(1, sources_i)
            scaled_src = cb_at_src * src_amp * source_coeff

            if src_component == "ex":
                Ex = (
                    Ex.reshape(n_shots, flat_model_shape)
                    .scatter_add(1, sources_i, scaled_src)
                    .reshape(size_with_batch)
                )
            elif src_component == "ey":
                Ey = (
                    Ey.reshape(n_shots, flat_model_shape)
                    .scatter_add(1, sources_i, scaled_src)
                    .reshape(size_with_batch)
                )
            else:
                Ez = (
                    Ez.reshape(n_shots, flat_model_shape)
                    .scatter_add(1, sources_i, scaled_src)
                    .reshape(size_with_batch)
                )

        if n_receivers > 0:
            if rec_component == "ex":
                receiver_field = Ex
            elif rec_component == "ey":
                receiver_field = Ey
            elif rec_component == "ez":
                receiver_field = Ez
            elif rec_component == "hx":
                receiver_field = Hx
            elif rec_component == "hy":
                receiver_field = Hy
            else:
                receiver_field = Hz

            receiver_amplitudes[step] = (
                receiver_field.reshape(n_shots, flat_model_shape)
                .gather(1, receivers_i)
            )

    s = (
        slice(None),
        slice(fd_pad_list[0], padded_nz - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
        slice(fd_pad_list[2], padded_ny - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
        slice(fd_pad_list[4], padded_nx - fd_pad_list[5] if fd_pad_list[5] > 0 else None),
    )

    return (
        Ex[s],
        Ey[s],
        Ez[s],
        Hx[s],
        Hy[s],
        Hz[s],
        m_Hz_y[s],
        m_Hy_z[s],
        m_Hx_z[s],
        m_Hz_x[s],
        m_Hy_x[s],
        m_Hx_y[s],
        m_Ey_z[s],
        m_Ez_y[s],
        m_Ez_x[s],
        m_Ex_z[s],
        m_Ex_y[s],
        m_Ey_x[s],
        receiver_amplitudes,
    )


def maxwell3d_c_cuda(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: Sequence[float],
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
    m_Hz_y_0: Optional[torch.Tensor],
    m_Hy_z_0: Optional[torch.Tensor],
    m_Hx_z_0: Optional[torch.Tensor],
    m_Hz_x_0: Optional[torch.Tensor],
    m_Hy_x_0: Optional[torch.Tensor],
    m_Hx_y_0: Optional[torch.Tensor],
    m_Ey_z_0: Optional[torch.Tensor],
    m_Ez_y_0: Optional[torch.Tensor],
    m_Ez_x_0: Optional[torch.Tensor],
    m_Ex_z_0: Optional[torch.Tensor],
    m_Ex_y_0: Optional[torch.Tensor],
    m_Ey_x_0: Optional[torch.Tensor],
    nt: Optional[int],
    model_gradient_sampling_interval: int,
    freq_taper_frac: float,
    time_pad_frac: float,
    time_taper: bool,
    save_snapshots: Optional[bool],
    forward_callback: Optional[Callback],
    backward_callback: Optional[Callback],
    callback_frequency: int,
    gradient_mode: str,
    storage_mode: str,
    storage_path: str,
    storage_compression: Union[bool, str],
    storage_bytes_limit_device: Optional[int],
    storage_bytes_limit_host: Optional[int],
    storage_chunk_steps: int,
    boundary_width: int,
    source_component: str,
    receiver_component: Optional[str],
    n_threads: Optional[int] = None,
):
    """3D Maxwell propagation using the C/CUDA backend (ASM on CUDA)."""
    from .padding import create_or_pad
    from . import staggered
    from . import backend_utils

    _ = (
        callback_frequency,
        storage_chunk_steps,
        boundary_width,
        time_pad_frac,
        time_taper,
    )

    if epsilon.ndim != 3:
        raise RuntimeError("epsilon must be 3D [nz, ny, nx].")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon.")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon.")

    gradient_mode_str = gradient_mode.lower()
    if gradient_mode_str != "snapshot":
        raise NotImplementedError(
            f"gradient_mode={gradient_mode!r} is not implemented yet; "
            "only 'snapshot' is supported."
        )

    requires_grad = epsilon.requires_grad or sigma.requires_grad
    device = epsilon.device
    dtype = epsilon.dtype

    n_threads_val = 0
    if n_threads is not None:
        n_threads_val = int(n_threads)
        if n_threads_val < 0:
            raise ValueError("n_threads must be >= 0 when provided.")

    storage_mode_str = storage_mode.lower()
    if storage_mode_str not in {"device", "cpu", "disk", "none", "auto"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
            f"but got {storage_mode!r}"
        )
    if device.type == "cpu" and storage_mode_str == "cpu":
        storage_mode_str = "device"
    if forward_callback is not None or backward_callback is not None:
        raise NotImplementedError(
            "Callbacks are not supported in the 3D C/CUDA backend yet."
        )

    functorch_active = torch._C._are_functorch_transforms_active()
    if functorch_active:
        raise NotImplementedError(
            "torch.func transforms are not supported for the C/CUDA backend."
        )

    if isinstance(pml_width, int):
        pml_width_list = [pml_width] * 6
    else:
        pml_width_list = list(pml_width)
        if len(pml_width_list) == 1:
            pml_width_list = pml_width_list * 6
        elif len(pml_width_list) == 3:
            pml_width_list = [
                pml_width_list[0],
                pml_width_list[0],
                pml_width_list[1],
                pml_width_list[1],
                pml_width_list[2],
                pml_width_list[2],
            ]
        elif len(pml_width_list) != 6:
            raise ValueError(
                "pml_width must be int or sequence of length 1, 3, or 6."
            )

    if nt is None:
        if source_amplitude is None:
            raise ValueError("Either nt or source_amplitude must be provided.")
        nt = source_amplitude.shape[-1]
    nt_steps: int = int(nt)

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

    model_nz, model_ny, model_nx = epsilon.shape
    padded_nz = model_nz + total_pad[0] + total_pad[1]
    padded_ny = model_ny + total_pad[2] + total_pad[3]
    padded_nx = model_nx + total_pad[4] + total_pad[5]

    storage_kind, _, storage_bytes_per_elem = _resolve_storage_compression(
        storage_compression,
        dtype,
        device,
        context="storage_compression",
        allow_fp8=False,
    )
    storage_bf16 = storage_kind == "bf16"

    if save_snapshots is None:
        do_save_snapshots = requires_grad
    else:
        do_save_snapshots = save_snapshots

    if requires_grad and save_snapshots is False:
        import warnings

        warnings.warn(
            "save_snapshots=False but model parameters require gradients. "
            "Backward pass will fail. Consider using gradient-free methods.",
            UserWarning,
        )

    needs_storage = do_save_snapshots and requires_grad
    effective_storage_mode_str = storage_mode_str
    if not needs_storage:
        if effective_storage_mode_str == "auto":
            effective_storage_mode_str = "none"
    else:
        if effective_storage_mode_str == "none":
            raise ValueError(
                "storage_mode='none' is not compatible with gradient_mode='snapshot' "
                "when gradients are required."
            )
        if effective_storage_mode_str == "auto":
            dtype_size = storage_bytes_per_elem
            shot_numel = padded_nz * padded_ny * padded_nx
            shot_bytes_uncomp = shot_numel * dtype_size
            n_stored = (
                nt_steps + model_gradient_sampling_interval - 1
            ) // model_gradient_sampling_interval
            total_bytes = n_stored * n_shots * shot_bytes_uncomp * 6  # Ex/Ey/Ez + curl(H)
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
        if device.type != "cuda":
            raise NotImplementedError(
                "3D C/CUDA backend gradients are only supported on CUDA."
            )

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
    ca = ca[None, :, :, :].contiguous()
    cb = cb[None, :, :, :].contiguous()
    cq = cq[None, :, :, :].contiguous()
    ca_batched = False
    cb_batched = False
    cq_batched = False

    size_with_batch = (n_shots, padded_nz, padded_ny, padded_nx)

    def init_wavefield(field_0: Optional[torch.Tensor]) -> torch.Tensor:
        if field_0 is not None:
            if field_0.ndim == 3:
                field_0 = field_0[None, :, :, :].expand(n_shots, -1, -1, -1)
            return create_or_pad(
                field_0, fd_pad_list, device, dtype, size_with_batch, mode="constant"
            ).contiguous()
        return torch.zeros(size_with_batch, device=device, dtype=dtype)

    Ex = init_wavefield(Ex_0)
    Ey = init_wavefield(Ey_0)
    Ez = init_wavefield(Ez_0)
    Hx = init_wavefield(Hx_0)
    Hy = init_wavefield(Hy_0)
    Hz = init_wavefield(Hz_0)

    m_Hz_y = init_wavefield(m_Hz_y_0)
    m_Hy_z = init_wavefield(m_Hy_z_0)
    m_Hx_z = init_wavefield(m_Hx_z_0)
    m_Hz_x = init_wavefield(m_Hz_x_0)
    m_Hy_x = init_wavefield(m_Hy_x_0)
    m_Hx_y = init_wavefield(m_Hx_y_0)

    m_Ey_z = init_wavefield(m_Ey_z_0)
    m_Ez_y = init_wavefield(m_Ez_y_0)
    m_Ez_x = init_wavefield(m_Ez_x_0)
    m_Ex_z = init_wavefield(m_Ex_z_0)
    m_Ex_y = init_wavefield(m_Ex_y_0)
    m_Ey_x = init_wavefield(m_Ey_x_0)

    def zero_interior_3d(
        tensor: torch.Tensor,
        fd_pad: Sequence[int],
        pml_width: Sequence[int],
        dim: int,
    ) -> None:
        shape = tensor.shape[1:]
        interior_start = fd_pad[dim * 2] + pml_width[dim * 2]
        interior_end = shape[dim] - pml_width[dim * 2 + 1] - fd_pad[dim * 2 + 1]

        if dim == 0:
            tensor[:, interior_start:interior_end, :, :].fill_(0)
        elif dim == 1:
            tensor[:, :, interior_start:interior_end, :].fill_(0)
        else:
            tensor[:, :, :, interior_start:interior_end].fill_(0)

    pml_aux_dims = [
        (m_Hz_y, 1),
        (m_Hy_z, 0),
        (m_Hx_z, 0),
        (m_Hz_x, 2),
        (m_Hy_x, 2),
        (m_Hx_y, 1),
        (m_Ey_z, 0),
        (m_Ez_y, 1),
        (m_Ez_x, 2),
        (m_Ex_z, 0),
        (m_Ex_y, 1),
        (m_Ey_x, 2),
    ]
    for wf, dim in pml_aux_dims:
        zero_interior_3d(wf, fd_pad_list, pml_width_list, dim)

    pml_profiles, kappa_profiles = staggered.set_pml_profiles_3d(
        pml_width=pml_width_list,
        accuracy=stencil,
        fd_pad=fd_pad_list,
        dt=dt,
        grid_spacing=list(grid_spacing),
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
    ) = pml_profiles
    kappa_z, kappa_z_h, kappa_y, kappa_y_h, kappa_x, kappa_x_h = kappa_profiles

    dz, dy, dx = grid_spacing
    rdz = float(1.0 / dz)
    rdy = float(1.0 / dy)
    rdx = float(1.0 / dx)

    flat_model_shape = padded_nz * padded_ny * padded_nx

    if source_location is not None and source_location.numel() > 0:
        source_z = source_location[..., 0] + total_pad[0]
        source_y = source_location[..., 1] + total_pad[2]
        source_x = source_location[..., 2] + total_pad[4]
        sources_i = (
            source_z * (padded_ny * padded_nx) + source_y * padded_nx + source_x
        ).long()
        n_sources = source_location.shape[1]
    else:
        sources_i = torch.empty(0, device=device, dtype=torch.long)
        n_sources = 0

    if receiver_location is not None and receiver_location.numel() > 0:
        receiver_z = receiver_location[..., 0] + total_pad[0]
        receiver_y = receiver_location[..., 1] + total_pad[2]
        receiver_x = receiver_location[..., 2] + total_pad[4]
        receivers_i = (
            receiver_z * (padded_ny * padded_nx) + receiver_y * padded_nx + receiver_x
        ).long()
        n_receivers = receiver_location.shape[1]
    else:
        receivers_i = torch.empty(0, device=device, dtype=torch.long)
        n_receivers = 0

    gradient_sampling_interval = model_gradient_sampling_interval
    use_autograd_fn = needs_storage and requires_grad

    if not use_autograd_fn:
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt_steps, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

    def _component_code(name: Optional[str], default: str, allow_h: bool) -> int:
        if name is None:
            name = default
        if not isinstance(name, str):
            raise TypeError("component name must be a string.")
        comp = name.strip().lower()
        mapping = {"ex": 0, "ey": 1, "ez": 2, "hx": 3, "hy": 4, "hz": 5}
        if comp not in mapping:
            raise ValueError(f"Unknown component {name!r}.")
        if not allow_h and comp in {"hx", "hy", "hz"}:
            raise ValueError(f"Source component {name!r} must be an E field.")
        return mapping[comp]

    src_code = _component_code(source_component, "ez", allow_h=False)
    rec_code = _component_code(receiver_component, source_component, allow_h=True)

    source_coeff = -1.0 / (dx * dy * dz)

    if source_amplitude is not None and source_amplitude.numel() > 0 and n_sources > 0:
        cb_expanded = cb.expand(n_shots, -1, -1, -1)
        cb_flat = cb_expanded.reshape(n_shots, flat_model_shape)
        cb_at_src = cb_flat.gather(1, sources_i)
        f = source_amplitude.permute(2, 0, 1).contiguous()
        f = f * cb_at_src[None, :, :] * source_coeff
        f = f.reshape(nt_steps * n_shots * n_sources)
    else:
        f = torch.empty(0, device=device, dtype=dtype)

    pml_z0 = fd_pad_list[0] + pml_width_list[0]
    pml_z1 = padded_nz - fd_pad_list[1] - pml_width_list[1]
    pml_y0 = fd_pad_list[2] + pml_width_list[2]
    pml_y1 = padded_ny - fd_pad_list[3] - pml_width_list[3]
    pml_x0 = fd_pad_list[4] + pml_width_list[4]
    pml_x1 = padded_nx - fd_pad_list[5] - pml_width_list[5]

    if use_autograd_fn:
        result = Maxwell3DForwardFunc.apply(
            ca,
            cb,
            cq,
            f,
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
            kappa_z,
            kappa_z_h,
            kappa_y,
            kappa_y_h,
            kappa_x,
            kappa_x_h,
            sources_i,
            receivers_i,
            rdz,
            rdy,
            rdx,
            dt,
            nt_steps,
            n_shots,
            padded_nz,
            padded_ny,
            padded_nx,
            n_sources,
            n_receivers,
            gradient_sampling_interval,
            stencil,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_z0,
            pml_y0,
            pml_x0,
            pml_z1,
            pml_y1,
            pml_x1,
            effective_storage_mode_str,
            storage_path,
            storage_compression,
            src_code,
            rec_code,
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            m_Hz_y,
            m_Hy_z,
            m_Hx_z,
            m_Hz_x,
            m_Hy_x,
            m_Hx_y,
            m_Ey_z,
            m_Ez_y,
            m_Ez_x,
            m_Ex_z,
            m_Ex_y,
            m_Ey_x,
            n_threads_val,
        )
        (
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            m_Hz_y,
            m_Hy_z,
            m_Hx_z,
            m_Hz_x,
            m_Hy_x,
            m_Hx_y,
            m_Ey_z,
            m_Ez_y,
            m_Ez_x,
            m_Ex_z,
            m_Ex_y,
            m_Ey_x,
            receiver_amplitudes,
            _ctx_handle,
        ) = result
    else:
        forward_func = backend_utils.get_backend_function(
            "maxwell_3d", "forward", stencil, dtype, device
        )

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
            backend_utils.tensor_to_ptr(m_Hz_y),
            backend_utils.tensor_to_ptr(m_Hy_z),
            backend_utils.tensor_to_ptr(m_Hx_z),
            backend_utils.tensor_to_ptr(m_Hz_x),
            backend_utils.tensor_to_ptr(m_Hy_x),
            backend_utils.tensor_to_ptr(m_Hx_y),
            backend_utils.tensor_to_ptr(m_Ey_z),
            backend_utils.tensor_to_ptr(m_Ez_y),
            backend_utils.tensor_to_ptr(m_Ez_x),
            backend_utils.tensor_to_ptr(m_Ex_z),
            backend_utils.tensor_to_ptr(m_Ex_y),
            backend_utils.tensor_to_ptr(m_Ey_x),
            backend_utils.tensor_to_ptr(receiver_amplitudes),
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
            backend_utils.tensor_to_ptr(kappa_z),
            backend_utils.tensor_to_ptr(kappa_z_h),
            backend_utils.tensor_to_ptr(kappa_y),
            backend_utils.tensor_to_ptr(kappa_y_h),
            backend_utils.tensor_to_ptr(kappa_x),
            backend_utils.tensor_to_ptr(kappa_x_h),
            backend_utils.tensor_to_ptr(sources_i),
            backend_utils.tensor_to_ptr(receivers_i),
            rdz,
            rdy,
            rdx,
            dt,
            nt_steps,
            n_shots,
            padded_nz,
            padded_ny,
            padded_nx,
            n_sources,
            n_receivers,
            1,
            ca_batched,
            cb_batched,
            cq_batched,
            0,
            pml_z0,
            pml_y0,
            pml_x0,
            pml_z1,
            pml_y1,
            pml_x1,
            src_code,
            rec_code,
            n_threads_val,
            device.index if device.type == "cuda" else -1,
        )

    s = (
        slice(None),
        slice(fd_pad_list[0], padded_nz - fd_pad_list[1] if fd_pad_list[1] > 0 else None),
        slice(fd_pad_list[2], padded_ny - fd_pad_list[3] if fd_pad_list[3] > 0 else None),
        slice(fd_pad_list[4], padded_nx - fd_pad_list[5] if fd_pad_list[5] > 0 else None),
    )

    return (
        Ex[s],
        Ey[s],
        Ez[s],
        Hx[s],
        Hy[s],
        Hz[s],
        m_Hz_y[s],
        m_Hy_z[s],
        m_Hx_z[s],
        m_Hz_x[s],
        m_Hy_x[s],
        m_Hx_y[s],
        m_Ey_z[s],
        m_Ez_y[s],
        m_Ez_x[s],
        m_Ex_z[s],
        m_Ex_y[s],
        m_Ey_x[s],
        receiver_amplitudes,
    )


def update_E_3d(
    ca: torch.Tensor,
    cb: torch.Tensor,
    Hx: torch.Tensor,
    Hy: torch.Tensor,
    Hz: torch.Tensor,
    Ex: torch.Tensor,
    Ey: torch.Tensor,
    Ez: torch.Tensor,
    m_Hz_y: torch.Tensor,
    m_Hy_z: torch.Tensor,
    m_Hx_z: torch.Tensor,
    m_Hz_x: torch.Tensor,
    m_Hy_x: torch.Tensor,
    m_Hx_y: torch.Tensor,
    kappa_z: torch.Tensor,
    kappa_z_h: torch.Tensor,
    kappa_y: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x: torch.Tensor,
    kappa_x_h: torch.Tensor,
    az: torch.Tensor,
    az_h: torch.Tensor,
    ay: torch.Tensor,
    ay_h: torch.Tensor,
    ax: torch.Tensor,
    ax_h: torch.Tensor,
    bz: torch.Tensor,
    bz_h: torch.Tensor,
    by: torch.Tensor,
    by_h: torch.Tensor,
    bx: torch.Tensor,
    bx_h: torch.Tensor,
    rdz: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    stencil: int,
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
]:
    """Update electric fields with CPML absorbing boundary conditions."""
    _ = (kappa_z_h, kappa_y_h, kappa_x_h, az_h, ay_h, ax_h, bz_h, by_h, bx_h, dt)

    dHz_dy = staggered.diffy1(Hz, stencil, rdy)
    dHy_dz = staggered.diffz1(Hy, stencil, rdz)
    dHx_dz = staggered.diffz1(Hx, stencil, rdz)
    dHz_dx = staggered.diffx1(Hz, stencil, rdx)
    dHy_dx = staggered.diffx1(Hy, stencil, rdx)
    dHx_dy = staggered.diffy1(Hx, stencil, rdy)

    m_Hz_y = by * m_Hz_y + ay * dHz_dy
    m_Hy_z = bz * m_Hy_z + az * dHy_dz
    m_Hx_z = bz * m_Hx_z + az * dHx_dz
    m_Hz_x = bx * m_Hz_x + ax * dHz_dx
    m_Hy_x = bx * m_Hy_x + ax * dHy_dx
    m_Hx_y = by * m_Hx_y + ay * dHx_dy

    dHz_dy = dHz_dy / kappa_y + m_Hz_y
    dHy_dz = dHy_dz / kappa_z + m_Hy_z
    dHx_dz = dHx_dz / kappa_z + m_Hx_z
    dHz_dx = dHz_dx / kappa_x + m_Hz_x
    dHy_dx = dHy_dx / kappa_x + m_Hy_x
    dHx_dy = dHx_dy / kappa_y + m_Hx_y

    Ex = ca * Ex + cb * (dHz_dy - dHy_dz)
    Ey = ca * Ey + cb * (dHx_dz - dHz_dx)
    Ez = ca * Ez + cb * (dHy_dx - dHx_dy)

    return Ex, Ey, Ez, m_Hz_y, m_Hy_z, m_Hx_z, m_Hz_x, m_Hy_x, m_Hx_y


def update_H_3d(
    cq: torch.Tensor,
    Hx: torch.Tensor,
    Hy: torch.Tensor,
    Hz: torch.Tensor,
    Ex: torch.Tensor,
    Ey: torch.Tensor,
    Ez: torch.Tensor,
    m_Ey_z: torch.Tensor,
    m_Ez_y: torch.Tensor,
    m_Ez_x: torch.Tensor,
    m_Ex_z: torch.Tensor,
    m_Ex_y: torch.Tensor,
    m_Ey_x: torch.Tensor,
    kappa_z: torch.Tensor,
    kappa_z_h: torch.Tensor,
    kappa_y: torch.Tensor,
    kappa_y_h: torch.Tensor,
    kappa_x: torch.Tensor,
    kappa_x_h: torch.Tensor,
    az: torch.Tensor,
    az_h: torch.Tensor,
    ay: torch.Tensor,
    ay_h: torch.Tensor,
    ax: torch.Tensor,
    ax_h: torch.Tensor,
    bz: torch.Tensor,
    bz_h: torch.Tensor,
    by: torch.Tensor,
    by_h: torch.Tensor,
    bx: torch.Tensor,
    bx_h: torch.Tensor,
    rdz: torch.Tensor,
    rdy: torch.Tensor,
    rdx: torch.Tensor,
    dt: torch.Tensor,
    stencil: int,
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
]:
    """Update magnetic fields with CPML absorbing boundary conditions."""
    _ = (kappa_z, kappa_y, kappa_x, az, ay, ax, bz, by, bx, dt)

    dEy_dz = staggered.diffzh1(Ey, stencil, rdz)
    dEz_dy = staggered.diffyh1(Ez, stencil, rdy)
    dEz_dx = staggered.diffxh1(Ez, stencil, rdx)
    dEx_dz = staggered.diffzh1(Ex, stencil, rdz)
    dEx_dy = staggered.diffyh1(Ex, stencil, rdy)
    dEy_dx = staggered.diffxh1(Ey, stencil, rdx)

    m_Ey_z = bz_h * m_Ey_z + az_h * dEy_dz
    m_Ez_y = by_h * m_Ez_y + ay_h * dEz_dy
    m_Ez_x = bx_h * m_Ez_x + ax_h * dEz_dx
    m_Ex_z = bz_h * m_Ex_z + az_h * dEx_dz
    m_Ex_y = by_h * m_Ex_y + ay_h * dEx_dy
    m_Ey_x = bx_h * m_Ey_x + ax_h * dEy_dx

    dEy_dz = dEy_dz / kappa_z_h + m_Ey_z
    dEz_dy = dEz_dy / kappa_y_h + m_Ez_y
    dEz_dx = dEz_dx / kappa_x_h + m_Ez_x
    dEx_dz = dEx_dz / kappa_z_h + m_Ex_z
    dEx_dy = dEx_dy / kappa_y_h + m_Ex_y
    dEy_dx = dEy_dx / kappa_x_h + m_Ey_x

    Hx = Hx - cq * (dEy_dz - dEz_dy)
    Hy = Hy - cq * (dEz_dx - dEx_dz)
    Hz = Hz - cq * (dEx_dy - dEy_dx)

    return Hx, Hy, Hz, m_Ey_z, m_Ez_y, m_Ez_x, m_Ex_z, m_Ex_y, m_Ey_x


_update_E3d_opt = update_E_3d
_update_H3d_opt = update_H_3d

class MaxwellTMForwardBoundaryFunc(torch.autograd.Function):
    """Autograd function for Maxwell TM with boundary storage reconstruction.

    Stores a compact boundary ring of (Ey, Hx, Hz) for every time step and
    reconstructs forward wavefields during the backward pass by inverting the
    update equations in the non-PML region.
    """

    @staticmethod
    def forward(
        ctx: Any,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        boundary_indices: torch.Tensor,
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
        accuracy: int,
        ca_batched: bool,
        cb_batched: bool,
        cq_batched: bool,
        pml_y0: int,
        pml_x0: int,
        pml_y1: int,
        pml_x1: int,
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
    ) -> tuple[torch.Tensor, ...]:
        from . import backend_utils

        import ctypes

        device = Ey.device
        dtype = Ey.dtype

        # Initialize receiver amplitudes
        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        if device.type == "cpu" and storage_mode_str == "cpu":
            storage_mode_str = "device"
        storage_mode = storage_mode_to_int(storage_mode_str)
        if storage_mode == STORAGE_NONE:
            raise ValueError(
                "storage_mode='none' is not compatible with gradient_mode='boundary' "
                "when gradients are required."
            )

        boundary_indices = boundary_indices.to(device=device, dtype=torch.int64).contiguous()
        boundary_numel = int(boundary_indices.numel())
        if boundary_numel <= 0:
            raise ValueError("boundary_indices must be non-empty for boundary storage.")

        _, store_dtype, _ = _resolve_storage_compression(
            storage_compression,
            dtype,
            device,
            context="storage_compression",
        )

        shot_bytes_uncomp = boundary_numel * store_dtype.itemsize
        num_steps_stored = nt + 1

        char_ptr_type = ctypes.c_char_p

        ctx.boundary_storage_objects = []
        ctx.boundary_storage_filename_arrays = []

        def alloc_boundary_storage():
            store_1 = torch.empty(0)
            store_3 = torch.empty(0)
            filenames_arr = (char_ptr_type * 0)()

            if storage_mode == STORAGE_DEVICE:
                store_1 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    boundary_numel,
                    device=device,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_CPU:
                # Multi-buffer device staging to enable safe async copies.
                store_1 = torch.empty(
                    _CPU_STORAGE_BUFFERS,
                    n_shots,
                    boundary_numel,
                    device=device,
                    dtype=store_dtype,
                )
                store_3 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    boundary_numel,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_DISK:
                storage_obj = TemporaryStorage(storage_path, 1)
                ctx.boundary_storage_objects.append(storage_obj)
                filenames_list = [
                    f.encode("utf-8") for f in storage_obj.get_filenames()
                ]
                filenames_arr = (char_ptr_type * len(filenames_list))()
                for i_file, f_name in enumerate(filenames_list):
                    filenames_arr[i_file] = ctypes.cast(
                        ctypes.create_string_buffer(f_name), char_ptr_type
                    )

                store_1 = torch.empty(
                    n_shots, boundary_numel, device=device, dtype=store_dtype
                )
                store_3 = torch.empty(
                    n_shots,
                    boundary_numel,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )

            ctx.boundary_storage_filename_arrays.append(filenames_arr)

            filenames_ptr = (
                ctypes.cast(filenames_arr, ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0
            )
            return store_1, store_3, filenames_ptr

        boundary_ey_store_1, boundary_ey_store_3, boundary_ey_filenames_ptr = (
            alloc_boundary_storage()
        )
        boundary_hx_store_1, boundary_hx_store_3, boundary_hx_filenames_ptr = (
            alloc_boundary_storage()
        )
        boundary_hz_store_1, boundary_hz_store_3, boundary_hz_filenames_ptr = (
            alloc_boundary_storage()
        )

        device_idx = device.index if device.index is not None else 0

        forward_func = backend_utils.get_backend_function(
            "maxwell_tm", "forward_with_boundary_storage", accuracy, dtype, device
        )

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
            backend_utils.tensor_to_ptr(boundary_ey_store_1),
            backend_utils.tensor_to_ptr(boundary_ey_store_3),
            boundary_ey_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hx_store_1),
            backend_utils.tensor_to_ptr(boundary_hx_store_3),
            boundary_hx_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hz_store_1),
            backend_utils.tensor_to_ptr(boundary_hz_store_3),
            boundary_hz_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_indices),
            boundary_numel,
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
            storage_mode,
            shot_bytes_uncomp,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            n_threads,
            device_idx,
        )

        ctx.save_for_backward(
            ca,
            cb,
            cq,
            source_amplitudes_scaled,
            boundary_indices,
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
            boundary_ey_store_1,
            boundary_ey_store_3,
            boundary_hx_store_1,
            boundary_hx_store_3,
            boundary_hz_store_1,
            boundary_hz_store_3,
            Ey,
            Hx,
            Hz,
        )

        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.ny = ny
        ctx.nx = nx
        ctx.n_sources = n_sources
        ctx.n_receivers = n_receivers
        ctx.accuracy = accuracy
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.pml_y0 = pml_y0
        ctx.pml_x0 = pml_x0
        ctx.pml_y1 = pml_y1
        ctx.pml_x1 = pml_x1
        ctx.ca_requires_grad = ca.requires_grad
        ctx.cb_requires_grad = cb.requires_grad
        ctx.storage_mode = storage_mode
        ctx.shot_bytes_uncomp = shot_bytes_uncomp
        ctx.boundary_numel = boundary_numel
        ctx.n_threads = n_threads

        return (
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            receiver_amplitudes,
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], ...]:
        from . import backend_utils

        (
            grad_Ey,
            grad_Hx,
            grad_Hz,
            grad_m_Ey_x,
            grad_m_Ey_z,
            grad_m_Hx_z,
            grad_m_Hz_x,
            grad_r,
        ) = grad_outputs

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        source_amplitudes_scaled = saved[3]
        boundary_indices = saved[4]
        ay, by, ay_h, by_h = saved[5], saved[6], saved[7], saved[8]
        ax, bx, ax_h, bx_h = saved[9], saved[10], saved[11], saved[12]
        ky, ky_h, kx, kx_h = saved[13], saved[14], saved[15], saved[16]
        sources_i, receivers_i = saved[17], saved[18]
        boundary_ey_store_1, boundary_ey_store_3 = saved[19], saved[20]
        boundary_hx_store_1, boundary_hx_store_3 = saved[21], saved[22]
        boundary_hz_store_1, boundary_hz_store_3 = saved[23], saved[24]
        Ey_end, Hx_end, Hz_end = saved[25], saved[26], saved[27]

        device = ca.device
        dtype = ca.dtype

        rdy = ctx.rdy
        rdx = ctx.rdx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        ny = ctx.ny
        nx = ctx.nx
        n_sources = ctx.n_sources
        n_receivers = ctx.n_receivers
        accuracy = ctx.accuracy
        ca_batched = ctx.ca_batched
        cb_batched = ctx.cb_batched
        cq_batched = ctx.cq_batched
        pml_y0 = ctx.pml_y0
        pml_x0 = ctx.pml_x0
        pml_y1 = ctx.pml_y1
        pml_x1 = ctx.pml_x1
        ca_requires_grad = ctx.ca_requires_grad
        cb_requires_grad = ctx.cb_requires_grad
        storage_mode = ctx.storage_mode
        shot_bytes_uncomp = ctx.shot_bytes_uncomp
        boundary_numel = ctx.boundary_numel
        n_threads = ctx.n_threads

        import ctypes

        if storage_mode == STORAGE_DISK:
            boundary_ey_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[0], ctypes.c_void_p
            )
            boundary_hx_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[1], ctypes.c_void_p
            )
            boundary_hz_filenames_ptr = ctypes.cast(
                ctx.boundary_storage_filename_arrays[2], ctypes.c_void_p
            )
        else:
            boundary_ey_filenames_ptr = 0
            boundary_hx_filenames_ptr = 0
            boundary_hz_filenames_ptr = 0

        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            grad_r = grad_r.contiguous()

        # Reconstruction buffers (initialized from final forward fields)
        ey_recon = Ey_end.clone()
        hx_recon = Hx_end.clone()
        hz_recon = Hz_end.clone()
        curl_recon = torch.empty_like(ey_recon)

        # Adjoint fields
        lambda_ey = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hx = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        lambda_hz = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_ey_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_ey_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hx_z = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        m_lambda_hz_x = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)

        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            grad_ca = torch.zeros(ny, nx, device=device, dtype=dtype)
            grad_ca_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            grad_cb = torch.zeros(ny, nx, device=device, dtype=dtype)
            grad_cb_shot = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad or cb_requires_grad:
            if ca_batched:
                grad_eps = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(n_shots, ny, nx, device=device, dtype=dtype)
            else:
                grad_eps = torch.zeros(ny, nx, device=device, dtype=dtype)
                grad_sigma = torch.zeros(ny, nx, device=device, dtype=dtype)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        device_idx = device.index if device.type == "cuda" and device.index is not None else 0

        backward_func = backend_utils.get_backend_function(
            "maxwell_tm", "backward_with_boundary", accuracy, dtype, device
        )

        backward_func(
            backend_utils.tensor_to_ptr(ca),
            backend_utils.tensor_to_ptr(cb),
            backend_utils.tensor_to_ptr(cq),
            backend_utils.tensor_to_ptr(source_amplitudes_scaled),
            backend_utils.tensor_to_ptr(grad_r),
            backend_utils.tensor_to_ptr(ey_recon),
            backend_utils.tensor_to_ptr(hx_recon),
            backend_utils.tensor_to_ptr(hz_recon),
            backend_utils.tensor_to_ptr(curl_recon),
            backend_utils.tensor_to_ptr(lambda_ey),
            backend_utils.tensor_to_ptr(lambda_hx),
            backend_utils.tensor_to_ptr(lambda_hz),
            backend_utils.tensor_to_ptr(m_lambda_ey_x),
            backend_utils.tensor_to_ptr(m_lambda_ey_z),
            backend_utils.tensor_to_ptr(m_lambda_hx_z),
            backend_utils.tensor_to_ptr(m_lambda_hz_x),
            backend_utils.tensor_to_ptr(boundary_ey_store_1),
            backend_utils.tensor_to_ptr(boundary_ey_store_3),
            boundary_ey_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hx_store_1),
            backend_utils.tensor_to_ptr(boundary_hx_store_3),
            boundary_hx_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_hz_store_1),
            backend_utils.tensor_to_ptr(boundary_hz_store_3),
            boundary_hz_filenames_ptr,
            backend_utils.tensor_to_ptr(boundary_indices),
            boundary_numel,
            backend_utils.tensor_to_ptr(grad_f),
            backend_utils.tensor_to_ptr(grad_ca),
            backend_utils.tensor_to_ptr(grad_cb),
            backend_utils.tensor_to_ptr(grad_eps),
            backend_utils.tensor_to_ptr(grad_sigma),
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
            nt,
            n_shots,
            ny,
            nx,
            n_sources,
            n_receivers,
            storage_mode,
            shot_bytes_uncomp,
            ca_requires_grad,
            cb_requires_grad,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            n_threads,
            device_idx,
        )

        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None

        if ca_requires_grad and not ca_batched:
            grad_ca = grad_ca.unsqueeze(0)
        if cb_requires_grad and not cb_batched:
            grad_cb = grad_cb.unsqueeze(0)

        return (
            grad_ca if ca_requires_grad else None,  # ca
            grad_cb if cb_requires_grad else None,  # cb
            None,  # cq
            grad_f_flat,  # source_amplitudes_scaled
            None,  # boundary_indices
            None, None, None, None,  # ay, by, ay_h, by_h
            None, None, None, None,  # ax, bx, ax_h, bx_h
            None, None, None, None,  # ky, ky_h, kx, kx_h
            None, None,  # sources_i, receivers_i
            None, None, None,  # rdy, rdx, dt
            None, None, None, None, None, None,  # nt, n_shots, ny, nx, n_sources, n_receivers
            None,  # accuracy
            None, None, None,  # ca_batched, cb_batched, cq_batched
            None, None, None, None,  # pml_y0, pml_x0, pml_y1, pml_x1
            None, None, None,  # storage_mode_str, storage_path, storage_compression
            None, None, None,  # Ey, Hx, Hz
            None, None, None, None,  # m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
            None,  # n_threads
        )
