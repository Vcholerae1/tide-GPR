"""Common electromagnetic wavelets for TIDE simulations.

This module provides various source wavelet functions commonly used in
electromagnetic wave simulations, particularly for Ground Penetrating Radar (GPR)
and other time-domain electromagnetic methods.

All wavelets return PyTorch tensors and support optional dtype specification.
"""

import math

import torch


def ricker(
    freq: float,
    length: int,
    dt: float,
    peak_time: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a Ricker wavelet (Mexican hat wavelet).

    The Ricker wavelet is the negative normalized second derivative of a
    Gaussian function. It is commonly used in seismic and GPR simulations.

    The formula used is:
        w(t) = -(2*pi^2*(f*t' - 1)^2 - 1) * exp(-pi^2*(f*t' - 1)^2)

    where t' = t - peak_time.

    Args:
        freq: The central (dominant) frequency in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The time (in seconds) of the peak amplitude. If None,
            defaults to 1/freq (one period after start).
        dtype: The PyTorch datatype to use. Optional, defaults to float32.
        device: The PyTorch device to use. Optional, defaults to CPU.

    Returns:
        A PyTorch tensor representing the Ricker wavelet.

    Example:
        >>> # Create a 100 MHz Ricker wavelet for GPR
        >>> freq = 100e6  # 100 MHz
        >>> dt = 1e-10    # 0.1 ns time step
        >>> length = 500  # 500 time samples
        >>> wavelet = ricker(freq, length, dt)

    """
    if dt == 0:
        raise ValueError("dt cannot be zero.")
    if freq <= 0:
        raise ValueError("freq must be positive.")
    if length <= 0:
        raise ValueError("length must be positive.")

    if peak_time is None:
        peak_time = 1.0 / freq  # Default: one period

    t = torch.arange(float(length), dtype=dtype, device=device) * dt
    t_shifted = t - peak_time

    # Ricker wavelet formula
    pi2_f2_t2 = (math.pi * freq * t_shifted) ** 2
    y = (1 - 2 * pi2_f2_t2) * torch.exp(-pi2_f2_t2)

    return y
