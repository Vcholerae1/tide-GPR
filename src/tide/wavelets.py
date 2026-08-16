"""Common electromagnetic wavelets for TIDE simulations.

This module provides source wavelet functions commonly used in electromagnetic
wave simulations, particularly for Ground Penetrating Radar (GPR) and other
time-domain electromagnetic methods.

All wavelets return one-dimensional PyTorch tensors and support optional dtype
and device specification.
"""

import math
import operator

import torch


def _positive_scalar(name: str, value: float) -> float:
    """Return a finite positive scalar."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive scalar.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return result


def _time_axis(
    freq: float,
    length: int,
    dt: float,
    peak_time: float | None,
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> tuple[torch.Tensor, float]:
    """Build a validated time axis centered on the wavelet peak."""
    frequency = _positive_scalar("freq", freq)
    time_step = _positive_scalar("dt", dt)
    if isinstance(length, bool):
        raise TypeError("length must be a positive integer.")
    try:
        sample_count = operator.index(length)
    except TypeError as exc:
        raise TypeError("length must be a positive integer.") from exc
    if sample_count <= 0:
        raise ValueError("length must be a positive integer.")

    center_time = 1.0 / frequency if peak_time is None else float(peak_time)
    if not math.isfinite(center_time):
        raise ValueError("peak_time must be a finite scalar.")
    if dtype is not None and not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch dtype.")

    time = torch.arange(
        sample_count,
        dtype=torch.get_default_dtype() if dtype is None else dtype,
        device=device,
    )
    return time * time_step - center_time, frequency


def ricker(
    freq: float,
    length: int,
    dt: float,
    peak_time: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a zero-phase Ricker wavelet with unit peak amplitude.

    Args:
        freq: The central (dominant) frequency in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The time of peak amplitude. Defaults to one period.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default dtype.
        device: The PyTorch device to use. Defaults to CPU.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    phase_squared = (math.pi * frequency * time).square()
    return (1.0 - 2.0 * phase_squared) * torch.exp(-phase_squared)


def gaussian(
    freq: float,
    length: int,
    dt: float,
    peak_time: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a unit-amplitude Gaussian pulse.

    Args:
        freq: The pulse frequency scale in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The time of peak amplitude. Defaults to one period.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default dtype.
        device: The PyTorch device to use. Defaults to CPU.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    return torch.exp(-(math.pi * frequency * time).square())


def gaussian_derivative(
    freq: float,
    length: int,
    dt: float,
    peak_time: float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a first-derivative Gaussian pulse normalized to unit magnitude.

    Args:
        freq: The pulse frequency scale in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The zero-crossing time. Defaults to one period.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default dtype.
        device: The PyTorch device to use. Defaults to CPU.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    phase = math.pi * frequency * time
    return -math.sqrt(2.0 * math.e) * phase * torch.exp(-phase.square())


def morlet(
    freq: float,
    length: int,
    dt: float,
    peak_time: float | None = None,
    cycles: float = 3.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a real Morlet pulse with unit center amplitude.

    Args:
        freq: The carrier frequency in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The time of peak amplitude. Defaults to one period.
        cycles: The Gaussian envelope width in carrier cycles.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default dtype.
        device: The PyTorch device to use. Defaults to CPU.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    cycle_count = _positive_scalar("cycles", cycles)
    carrier_phase = 2.0 * math.pi * frequency * time
    envelope_phase = math.pi * frequency * time / cycle_count
    return torch.cos(carrier_phase) * torch.exp(-envelope_phase.square())


def sine_burst(
    freq: float,
    length: int,
    dt: float,
    peak_time: float | None = None,
    cycles: float = 3.0,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a finite-cycle cosine burst with a Hann envelope.

    Args:
        freq: The carrier frequency in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The center time. Defaults to one period.
        cycles: The duration in carrier cycles.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default dtype.
        device: The PyTorch device to use. Defaults to CPU.
    """
    time, frequency = _time_axis(freq, length, dt, peak_time, dtype, device)
    cycle_count = _positive_scalar("cycles", cycles)
    duration = cycle_count / frequency
    inside = time.abs() <= 0.5 * duration
    window = 0.5 * (1.0 + torch.cos(2.0 * math.pi * time / duration))
    pulse = window * torch.cos(2.0 * math.pi * frequency * time)
    return torch.where(inside, pulse, torch.zeros_like(pulse))
