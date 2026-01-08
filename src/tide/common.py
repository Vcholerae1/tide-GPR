"""Common helper classes and functions for TIDE propagators.

This module provides utility classes used across various TIDE propagators,
including callback state management and input validation helpers.
"""

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from types import EllipsisType

import torch


class CallbackState:
    """State provided to user callbacks during wave propagation.

    This class encapsulates the simulation state at a given time step,
    providing convenient access to wavefields, model parameters, and
    gradients with different views (full, pml, inner).

    The three views correspond to different regions of the computational domain:
    - 'full': The entire padded domain including FD padding
    - 'pml': The model region plus PML absorbing layers
    - 'inner': Only the physical model region (excluding PML)

    Example:
        >>> def my_callback(state: CallbackState):
        ...     # Get the Ey field in the inner (physical) region
        ...     ey = state.get_wavefield("Ey", view="inner")
        ...     print(f"Step {state.step}, max |Ey| = {ey.abs().max():.6e}")
        ...
        ...     # Get the permittivity model
        ...     eps = state.get_model("epsilon", view="inner")
        ...
        ...     # During backward pass, get gradients
        ...     if state.is_backward:
        ...         grad_eps = state.get_gradient("epsilon", view="inner")
        >>>
        >>> # Use with maxwell propagator
        >>> result = maxwell.maxwelltm(..., forward_callback=my_callback)

    Attributes:
        dt: The time step size in seconds.
        step: The current time step number (0-indexed).
        nt: Total number of time steps.
        is_backward: Whether this is during backward (adjoint) propagation.
    """

    def __init__(
        self,
        dt: float,
        step: int,
        nt: int,
        wavefields: Dict[str, torch.Tensor],
        models: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        fd_pad: Optional[List[int]] = None,
        pml_width: Optional[List[int]] = None,
        is_backward: bool = False,
        grid_spacing: Optional[List[float]] = None,
    ) -> None:
        """Initialize the callback state.

        Args:
            dt: The time step size in seconds.
            step: The current time step number.
            nt: Total number of time steps.
            wavefields: A dictionary mapping wavefield names to tensors.
                For Maxwell TM: {"Ey", "Hx", "Hz", "m_Ey_x", "m_Ey_z", ...}
            models: A dictionary mapping model names to tensors.
                For Maxwell TM: {"epsilon", "sigma", "mu", "ca", "cb", "cq"}
            gradients: A dictionary mapping gradient names to tensors.
                Only available during backward pass.
            fd_pad: Padding for finite difference stencil [y0, y1, x0, x1].
                If None, assumes no padding.
            pml_width: Width of PML layers [top, bottom, left, right].
                If None, assumes no PML.
            is_backward: Whether this is during backward propagation.
            grid_spacing: Grid spacing [dy, dx] in meters.
        """
        self.dt = dt
        self.step = step
        self.nt = nt
        self.is_backward = is_backward
        self._wavefields = wavefields
        self._models = models
        self._gradients = gradients if gradients is not None else {}
        self._fd_pad = fd_pad if fd_pad is not None else [0, 0, 0, 0]
        self._pml_width = pml_width if pml_width is not None else [0, 0, 0, 0]
        self._grid_spacing = grid_spacing

        # Determine spatial ndim from padding (preferred) or model tensors.
        # Padding lists are in [d0_low, d0_high, d1_low, d1_high, ...] format.
        if fd_pad is not None and len(fd_pad) in {4, 6}:
            self._ndim = len(fd_pad) // 2
        elif pml_width is not None and len(pml_width) in {4, 6}:
            self._ndim = len(pml_width) // 2
        elif models:
            first_model = next(iter(models.values()))
            # Heuristic:
            # - 2D unbatched: [ny, nx] -> 2
            # - 2D batched:   [n_shots, ny, nx] -> 2
            # - 3D unbatched: [nz, ny, nx] -> ambiguous with 2D batched; callers
            #   should pass fd_pad/pml_width to disambiguate.
            # - 3D batched:   [n_shots, nz, ny, nx] -> 3
            if first_model.ndim == 2:
                self._ndim = 2
            elif first_model.ndim == 4:
                self._ndim = 3
            else:
                # Preserve existing behavior (Maxwell TM callbacks) as default.
                self._ndim = 2
        else:
            # Default to 2D when no other information is available.
            self._ndim = 2

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self.step * self.dt

    @property
    def progress(self) -> float:
        """Simulation progress as a fraction [0, 1]."""
        return self.step / max(self.nt - 1, 1)

    @property
    def wavefield_names(self) -> List[str]:
        """List of available wavefield names."""
        return list(self._wavefields.keys())

    @property
    def model_names(self) -> List[str]:
        """List of available model names."""
        return list(self._models.keys())

    @property
    def gradient_names(self) -> List[str]:
        """List of available gradient names."""
        return list(self._gradients.keys())

    def get_wavefield(self, name: str, view: str = "inner") -> torch.Tensor:
        """Get a wavefield tensor.

        Args:
            name: The name of the wavefield. For Maxwell TM mode:
                - "Ey": Electric field (y-component)
                - "Hx": Magnetic field (x-component)
                - "Hz": Magnetic field (z-component)
                - "m_Ey_x", "m_Ey_z", "m_Hx_z", "m_Hz_x": CPML auxiliary fields
                - During backward: "lambda_Ey", "lambda_Hx", "lambda_Hz"
            view: The part of the wavefield to return:
                - 'inner': The physical model region (default)
                - 'pml': Model region plus PML layers
                - 'full': Entire domain including FD padding

        Returns:
            The specified part of the wavefield tensor.
            Shape depends on view and whether batched: [n_shots, ny, nx] or [ny, nx]

        Raises:
            KeyError: If the wavefield name is not found.
            ValueError: If view is not valid.
        """
        if name not in self._wavefields:
            available = ", ".join(self._wavefields.keys())
            raise KeyError(f"Wavefield '{name}' not found. Available: {available}")
        return self._get_view(self._wavefields[name], view)

    def get_model(self, name: str, view: str = "inner") -> torch.Tensor:
        """Get a model parameter tensor.

        Args:
            name: The name of the model parameter. For Maxwell TM:
                - "epsilon": Relative permittivity
                - "sigma": Electrical conductivity (S/m)
                - "mu": Relative permeability
                - "ca", "cb", "cq": Update coefficients
            view: The part of the model to return:
                - 'inner': The physical model region (default)
                - 'pml': Model region plus PML layers
                - 'full': Entire domain including FD padding

        Returns:
            The specified part of the model tensor.

        Raises:
            KeyError: If the model name is not found.
            ValueError: If view is not valid.
        """
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._get_view(self._models[name], view)

    def get_gradient(self, name: str, view: str = "inner") -> torch.Tensor:
        """Get a gradient tensor (only available during backward pass).

        Args:
            name: The name of the gradient. For Maxwell TM:
                - "epsilon" or "ca": Gradient w.r.t. permittivity/Ca
                - "sigma" or "cb": Gradient w.r.t. conductivity/Cb
            view: The part of the gradient to return:
                - 'inner': The physical model region (default)
                - 'pml': Model region plus PML layers
                - 'full': Entire domain including FD padding

        Returns:
            The specified part of the gradient tensor.

        Raises:
            KeyError: If the gradient name is not found.
            ValueError: If view is not valid.
            RuntimeError: If called during forward pass (no gradients available).
        """
        if not self._gradients:
            raise RuntimeError(
                "Gradients are only available during backward propagation. "
                "Use backward_callback instead of forward_callback."
            )
        if name not in self._gradients:
            available = ", ".join(self._gradients.keys())
            raise KeyError(f"Gradient '{name}' not found. Available: {available}")
        return self._get_view(self._gradients[name], view)

    def _get_view(self, x: torch.Tensor, view: str) -> torch.Tensor:
        """Extract a view of a tensor based on the specified region.

        Args:
            x: The tensor to extract a view from.
            view: One of 'full', 'pml', or 'inner'.

        Returns:
            A view of the tensor corresponding to the specified region.
        """
        if view == "full":
            return x

        if view not in {"pml", "inner"}:
            raise ValueError(
                f"view must be 'full', 'pml', or 'inner', but got '{view}'"
            )

        spatial_ndim = self._ndim
        if spatial_ndim not in {2, 3}:
            raise ValueError(f"Unsupported spatial ndim {spatial_ndim}.")

        if view == "pml":
            starts = [self._fd_pad[2 * i] for i in range(spatial_ndim)]
            ends = [self._fd_pad[2 * i + 1] for i in range(spatial_ndim)]
        else:
            starts = [
                self._fd_pad[2 * i] + self._pml_width[2 * i]
                for i in range(spatial_ndim)
            ]
            ends = [
                self._fd_pad[2 * i + 1] + self._pml_width[2 * i + 1]
                for i in range(spatial_ndim)
            ]

        def _slice(dim_size: int, start: int, end: int) -> slice:
            stop = dim_size - end if end > 0 else None
            return slice(start, stop)

        if x.ndim == spatial_ndim:
            # Non-batched: [ny, nx] or [nz, ny, nx]
            idx = tuple(
                _slice(x.shape[i], starts[i], ends[i]) for i in range(spatial_ndim)
            )
            return x[idx]

        # Batched: [..., ny, nx] or [..., nz, ny, nx]
        idx_batched: Tuple[Union["EllipsisType", slice], ...] = (
            ...,
            *(
                _slice(
                    x.shape[-spatial_ndim + i],
                    starts[i],
                    ends[i],
                )
                for i in range(spatial_ndim)
            ),
        )
        return x[idx_batched]

    def __repr__(self) -> str:
        """Return a string representation of the callback state."""
        return (
            f"CallbackState(step={self.step}/{self.nt}, "
            f"time={self.time:.2e}s, "
            f"is_backward={self.is_backward}, "
            f"wavefields={self.wavefield_names}, "
            f"models={self.model_names})"
        )


# Type alias for callback functions
Callback = Callable[[CallbackState], None]


# Special value for ignored source/receiver locations
IGNORE_LOCATION = -1 << 31


def create_callback_state(
    dt: float,
    step: int,
    nt: int,
    wavefields: Dict[str, torch.Tensor],
    models: Dict[str, torch.Tensor],
    gradients: Optional[Dict[str, torch.Tensor]] = None,
    fd_pad: Optional[List[int]] = None,
    pml_width: Optional[List[int]] = None,
    is_backward: bool = False,
    grid_spacing: Optional[List[float]] = None,
) -> CallbackState:
    """Factory function to create a CallbackState.

    This is a convenience function that creates a CallbackState with
    the given parameters. It's equivalent to calling the CallbackState
    constructor directly.

    Args:
        dt: The time step size in seconds.
        step: The current time step number.
        nt: Total number of time steps.
        wavefields: A dictionary mapping wavefield names to tensors.
        models: A dictionary mapping model names to tensors.
        gradients: A dictionary mapping gradient names to tensors (backward only).
        fd_pad: Padding for finite difference stencil [y0, y1, x0, x1].
        pml_width: Width of PML layers [top, bottom, left, right].
        is_backward: Whether this is during backward propagation.
        grid_spacing: Grid spacing [dy, dx] in meters.

    Returns:
        A new CallbackState instance.
    """
    return CallbackState(
        dt=dt,
        step=step,
        nt=nt,
        wavefields=wavefields,
        models=models,
        gradients=gradients,
        fd_pad=fd_pad,
        pml_width=pml_width,
        is_backward=is_backward,
        grid_spacing=grid_spacing,
    )


# =============================================================================
# Signal Processing Utilities for CFL Resampling
# =============================================================================

import math


def cosine_taper_end(signal: torch.Tensor, taper_len: int) -> torch.Tensor:
    """Apply a cosine taper to the end of the signal in the last dimension.

    Args:
        signal: Input tensor to taper.
        taper_len: Number of samples to taper at the end.

    Returns:
        Tapered signal.
    """
    if taper_len <= 0 or signal.shape[-1] <= taper_len:
        return signal

    # Create taper: 1 -> 0 over taper_len samples
    taper = 0.5 * (
        1 + torch.cos(torch.linspace(0, math.pi, taper_len, device=signal.device))
    )
    # Apply taper to the last taper_len elements
    signal = signal.clone()
    signal[..., -taper_len:] = signal[..., -taper_len:] * taper
    return signal


def zero_last_element_of_final_dimension(signal: torch.Tensor) -> torch.Tensor:
    """Zero the last element of the final dimension (Nyquist frequency).

    This is used to avoid aliasing when resampling signals in the frequency domain.

    Args:
        signal: Input tensor.

    Returns:
        Signal with last element of final dimension set to zero.
    """
    signal = signal.clone()
    signal[..., -1] = 0
    return signal


def upsample(
    signal: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
) -> torch.Tensor:
    """Upsample the final dimension of a tensor by a factor.

    Low-pass upsampling is used to produce an upsampled signal without
    introducing higher frequencies than were present in the input.

    This is typically used when the CFL condition requires a smaller internal
    time step than the user-provided time step.

    Args:
        signal: Tensor to upsample (time should be the last dimension).
        step_ratio: Integer factor by which to upsample.
        freq_taper_frac: Fraction of frequency spectrum end to taper (0.0-1.0).
            Helps reduce ringing artifacts.
        time_pad_frac: Fraction of signal length for zero padding (0.0-1.0).
            Helps reduce wraparound artifacts.
        time_taper: Whether to apply a Hann window in time.
            Useful for correctness tests to ensure signals taper to zero.

    Returns:
        Upsampled signal.

    Example:
        >>> # Source with 100 time samples, need 3x internal steps for CFL
        >>> source = torch.randn(1, 1, 100)
        >>> source_upsampled = upsample(source, step_ratio=3)
        >>> print(source_upsampled.shape)  # [1, 1, 300]
    """
    if signal.numel() == 0 or step_ratio == 1:
        return signal

    # Optional zero padding to reduce wraparound artifacts
    n_time_pad = int(time_pad_frac * signal.shape[-1]) if time_pad_frac > 0.0 else 0
    if n_time_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, n_time_pad))

    nt = signal.shape[-1]
    up_nt = nt * step_ratio

    # Transform to frequency domain
    signal_f = torch.fft.rfft(signal, norm="ortho") * math.sqrt(step_ratio)

    # Apply frequency taper or zero Nyquist
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)

    # Zero-pad in frequency domain for upsampling
    pad_len = up_nt // 2 + 1 - signal_f.shape[-1]
    if pad_len > 0:
        signal_f = torch.nn.functional.pad(signal_f, (0, pad_len))

    # Back to time domain
    signal = torch.fft.irfft(signal_f, n=up_nt, norm="ortho")

    # Remove padding
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad * step_ratio]

    # Optional time taper
    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1],
            periodic=False,
            device=signal.device,
        )

    return signal


def downsample(
    signal: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> torch.Tensor:
    """Downsample the final dimension of a tensor by a factor.

    Frequencies higher than or equal to the Nyquist frequency of the
    downsampled signal will be zeroed before downsampling.

    This is typically used when the internal time step is smaller than the
    user-provided time step due to CFL requirements.

    Args:
        signal: Tensor to downsample (time should be the last dimension).
        step_ratio: Integer factor by which to downsample.
        freq_taper_frac: Fraction of frequency spectrum end to taper (0.0-1.0).
            Helps reduce ringing artifacts.
        time_pad_frac: Fraction of signal length for zero padding (0.0-1.0).
            Helps reduce wraparound artifacts.
        time_taper: Whether to apply a Hann window in time.
            Useful for correctness tests.
        shift: Amount to shift in time before downsampling (in time samples).

    Returns:
        Downsampled signal.

    Example:
        >>> # Receiver data at internal rate, downsample to user rate
        >>> data = torch.randn(300, 1, 1)  # [nt_internal, shot, receiver]
        >>> data_ds = downsample(data.movedim(0, -1), step_ratio=3).movedim(-1, 0)
        >>> print(data_ds.shape)  # [100, 1, 1]
    """
    if signal.numel() == 0 or (step_ratio == 1 and shift == 0.0):
        return signal

    # Optional time taper
    if time_taper:
        signal = signal * torch.hann_window(
            signal.shape[-1],
            periodic=False,
            device=signal.device,
        )

    # Optional zero padding
    n_time_pad = (
        int(time_pad_frac * (signal.shape[-1] // step_ratio))
        if time_pad_frac > 0.0
        else 0
    )
    if n_time_pad > 0:
        signal = torch.nn.functional.pad(signal, (0, n_time_pad * step_ratio))

    nt = signal.shape[-1]
    down_nt = nt // step_ratio

    # Transform to frequency domain, keeping only frequencies below new Nyquist
    signal_f = torch.fft.rfft(signal, norm="ortho")[..., : down_nt // 2 + 1]

    # Apply frequency taper or zero Nyquist
    if freq_taper_frac > 0.0:
        freq_taper_len = int(freq_taper_frac * signal_f.shape[-1])
        signal_f = cosine_taper_end(signal_f, freq_taper_len)
    elif signal_f.shape[-1] > 1:
        signal_f = zero_last_element_of_final_dimension(signal_f)

    # Apply time shift in frequency domain
    if shift != 0.0:
        freqs = torch.fft.rfftfreq(signal.shape[-1], device=signal.device)[
            : down_nt // 2 + 1
        ]
        signal_f = signal_f * torch.exp(-1j * 2 * math.pi * freqs * shift)

    # Back to time domain
    signal = torch.fft.irfft(signal_f, n=down_nt, norm="ortho") / math.sqrt(step_ratio)

    # Remove padding
    if n_time_pad > 0:
        signal = signal[..., : signal.shape[-1] - n_time_pad]

    return signal


def downsample_and_movedim(
    receiver_amplitudes: torch.Tensor,
    step_ratio: int,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    shift: float = 0.0,
) -> torch.Tensor:
    """Downsample receiver data and move time dimension to last axis.

    Convenience function that combines downsampling with moving the time
    dimension to the expected output format [shot, receiver, time].

    Args:
        receiver_amplitudes: Receiver data [time, shot, receiver].
        step_ratio: Integer factor by which to downsample.
        freq_taper_frac: Fraction of frequency spectrum to taper.
        time_pad_frac: Fraction for zero padding.
        time_taper: Whether to apply Hann window.
        shift: Time shift before downsampling.

    Returns:
        Processed receiver data [shot, receiver, time].
    """
    if receiver_amplitudes.numel() > 0:
        # Move time to last dimension for processing
        receiver_amplitudes = torch.movedim(receiver_amplitudes, 0, -1)
        receiver_amplitudes = downsample(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            shift=shift,
        )
    return receiver_amplitudes


def cfl_condition(
    grid_spacing: Union[float, List[float]],
    dt: float,
    max_vel: float,
    c_max: float = 1,
    eps: float = 1e-15,
) -> tuple[float, int]:
    """Calculate time step interval to satisfy CFL condition.

    The CFL (Courant-Friedrichs-Lewy) condition ensures numerical stability
    for explicit FDTD schemes. If the user-provided dt is too large, this
    function computes a smaller internal dt and the ratio between them.

    Args:
        grid_spacing: Grid spacing [dy, dx] or single value for isotropic.
        dt: User-provided time step.
        max_vel: Maximum wave velocity in the model.
        c_max: Maximum Courant number (default 0.6 for stability margin).
        eps: Small value to prevent division by zero.

    Returns:
        Tuple of (inner_dt, step_ratio) where:
        - inner_dt: Time step satisfying CFL condition
        - step_ratio: Integer ratio dt / inner_dt

    Example:
        >>> # Check if dt=1e-9 is stable for v=3e8 m/s, dx=1e-3 m
        >>> inner_dt, ratio = cfl_condition([1e-3, 1e-3], 1e-9, 3e8)
        >>> print(f"Need {ratio}x smaller time step")
    """
    # Normalize grid_spacing to list
    if isinstance(grid_spacing, (int, float)):
        grid_spacing = [float(grid_spacing), float(grid_spacing)]
    else:
        grid_spacing = list(grid_spacing)

    if max_vel <= 0:
        raise ValueError("max_vel must be positive")

    # Maximum stable dt from CFL condition
    max_dt = (
        c_max / math.sqrt(sum(1 / dx**2 for dx in grid_spacing)) / (max_vel**2 + eps)
    ) * max_vel

    step_ratio = math.ceil(abs(dt) / max_dt)
    inner_dt = dt / step_ratio

    if step_ratio >= 2:
        import warnings

        warnings.warn(
            f"CFL condition requires {step_ratio} internal time steps per "
            f"user time step (dt={dt}, inner_dt={inner_dt}). Consider using "
            "a smaller dt or coarser grid.",
            stacklevel=2,
        )

    return inner_dt, step_ratio


def validate_model_gradient_sampling_interval(
    model_gradient_sampling_interval: int,
) -> int:
    """Validate the model gradient sampling interval parameter.

    The gradient sampling interval controls memory usage during backpropagation.
    Setting it > 1 reduces memory by storing fewer snapshots.

    Args:
        model_gradient_sampling_interval: Number of time steps between
            gradient snapshots.

    Returns:
        Validated interval value.

    Raises:
        TypeError: If not an integer.
        ValueError: If negative.
    """
    if not isinstance(model_gradient_sampling_interval, int):
        raise TypeError("model_gradient_sampling_interval must be an int")
    if model_gradient_sampling_interval < 0:
        raise ValueError("model_gradient_sampling_interval must be >= 0")
    return model_gradient_sampling_interval


def validate_freq_taper_frac(freq_taper_frac: float) -> float:
    """Validate the frequency taper fraction parameter.

    Args:
        freq_taper_frac: Fraction of frequencies to taper (0.0-1.0).

    Returns:
        Validated fraction value.

    Raises:
        TypeError: If not convertible to float.
        ValueError: If not in [0, 1].
    """
    try:
        freq_taper_frac = float(freq_taper_frac)
    except (TypeError, ValueError) as e:
        raise TypeError("freq_taper_frac must be convertible to float") from e
    if not 0.0 <= freq_taper_frac <= 1.0:
        raise ValueError(f"freq_taper_frac must be in [0, 1], got {freq_taper_frac}")
    return freq_taper_frac


def validate_time_pad_frac(time_pad_frac: float) -> float:
    """Validate the time padding fraction parameter.

    Args:
        time_pad_frac: Fraction of time axis for zero padding (0.0-1.0).

    Returns:
        Validated fraction value.

    Raises:
        TypeError: If not convertible to float.
        ValueError: If not in [0, 1].
    """
    try:
        time_pad_frac = float(time_pad_frac)
    except (TypeError, ValueError) as e:
        raise TypeError("time_pad_frac must be convertible to float") from e
    if not 0.0 <= time_pad_frac <= 1.0:
        raise ValueError(f"time_pad_frac must be in [0, 1], got {time_pad_frac}")
    return time_pad_frac


# =============================================================================
# Model and Wavefield Padding Utilities
# =============================================================================


def reverse_pad(pad: List[int]) -> List[int]:
    """Reverse the padding order for use with torch.nn.functional.pad.

    PyTorch's pad function expects padding in reverse order (last dim first).
    This function converts [y0, y1, x0, x1] to [x0, x1, y0, y1].

    Args:
        pad: Padding values in [y0, y1, x0, x1] format.

    Returns:
        Padding values in PyTorch format [x0, x1, y0, y1].
    """
    # For 2D: [y0, y1, x0, x1] -> [x0, x1, y0, y1]
    result = []
    for i in range(len(pad) // 2 - 1, -1, -1):
        result.extend([pad[i * 2], pad[i * 2 + 1]])
    return result


def create_or_pad(
    tensor: torch.Tensor,
    pad: Union[int, List[int]],
    device: torch.device,
    dtype: torch.dtype,
    size: Tuple[int, ...],
    mode: str = "constant",
) -> torch.Tensor:
    """Creates a zero tensor of specified size or pads an existing tensor.

    If the input tensor is empty (numel == 0), a new zero tensor with the
    given size is created. Otherwise, the tensor is padded according to
    the specified mode.

    This is a unified padding function that supports:
    - Zero padding (mode='constant') for wavefields
    - Replicate padding (mode='replicate') for models

    Args:
        tensor: The input tensor to be created or padded.
        pad: The padding to apply. Can be an integer (for uniform padding)
            or a list of integers [y0, y1, x0, x1] for per-side padding.
        device: The PyTorch device for the tensor.
        dtype: The PyTorch data type for the tensor.
        size: The desired size of the tensor if it needs to be created.
        mode: Padding mode ('constant', 'replicate', 'reflect', 'circular').
            Default is 'constant' (zero padding)

    Returns:
        The created or padded tensor.

    Example:
        >>> # Create a zero tensor of size [2, 110, 110] (batch=2, with padding)
        >>> wf = create_or_pad(torch.empty(0), 5, device, dtype, (2, 110, 110))
        >>>
        >>> # Pad a wavefield with zeros [2, 100, 100] -> [2, 110, 110]
        >>> wf_padded = create_or_pad(wf, [5, 5, 5, 5], device, dtype, (2, 110, 110))
        >>>
        >>> # Pad a model with replicate mode [100, 100] -> [110, 110]
        >>> eps_padded = create_or_pad(eps, [5, 5, 5, 5], device, dtype, (110, 110), mode='replicate')
    """
    if isinstance(pad, int):
        # Convert single int to [pad, pad, pad, pad, ...] for each spatial dimension
        # size includes batch dimension if len > 2, so spatial ndim = len(size) - 1 or len(size)
        ndim = len(size) - 1 if len(size) > 2 else len(size)
        pad = [pad] * ndim * 2

    if tensor.numel() == 0:
        return torch.zeros(size, device=device, dtype=dtype)

    if max(pad) == 0:
        return tensor.clone()

    # Reverse padding for PyTorch's pad function
    reversed_pad = reverse_pad(pad)

    # For non-constant padding modes (replicate, reflect, circular),
    # PyTorch requires:
    # - 2D spatial padding: 3D or 4D input
    # - 3D spatial padding: 4D or 5D input
    original_ndim = tensor.ndim
    needs_unsqueeze = original_ndim in {2, 3} and mode != "constant"

    if needs_unsqueeze:
        tensor = tensor.unsqueeze(0)

    result = torch.nn.functional.pad(tensor, reversed_pad, mode=mode)

    if needs_unsqueeze:
        result = result.squeeze(0)

    # PyTorch's autograd system automatically tracks gradients through operations.
    # Explicitly calling requires_grad_() is incompatible with torch.func transforms.
    # Simply return the result; gradient tracking is handled automatically.
    return result


def zero_interior(
    tensor: torch.Tensor,
    fd_pad: Union[int, List[int]],
    pml_width: List[int],
    dim: int,
) -> torch.Tensor:
    """Zero out the interior region of a tensor (keeping only PML regions).

    This is used for CPML auxiliary variables which should only be non-zero
    in the PML regions. Setting the interior to zero allows the propagator
    to skip unnecessary PML calculations in those regions.

    Args:
        tensor: The input tensor with shape [batch, ny, nx].
        fd_pad: Finite difference padding. Can be an int or list [y0, y1, x0, x1].
        pml_width: The width of PML regions [top, bottom, left, right].
        dim: The spatial dimension to zero (0 for y, 1 for x).

    Returns:
        The tensor with interior region zeroed out.
    """
    shape = tensor.shape[1:]  # Spatial dimensions (without batch)
    ndim = len(shape)

    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * 2 * ndim

    # Calculate interior slice for the specified dimension
    interior_start = fd_pad[dim * 2] + pml_width[dim * 2]
    interior_end = shape[dim] - pml_width[dim * 2 + 1] - fd_pad[dim * 2 + 1]

    # Zero out the interior
    if dim == 0:  # y dimension
        tensor[:, interior_start:interior_end, :].fill_(0)
    else:  # x dimension
        tensor[:, :, interior_start:interior_end].fill_(0)

    return tensor
