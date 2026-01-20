"""Validation helpers for user-facing parameters."""


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
