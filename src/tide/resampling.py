"""Signal resampling utilities for CFL handling."""

import math

import torch


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
