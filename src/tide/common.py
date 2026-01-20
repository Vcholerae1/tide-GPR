"""Compatibility re-exports for legacy tide.common usage.

Use the dedicated modules instead:
- callbacks
- resampling
- cfl
- validation
- padding
"""

from .callbacks import CallbackState, Callback, create_callback_state
from .cfl import cfl_condition
from .padding import create_or_pad, reverse_pad, zero_interior
from .resampling import (
    cosine_taper_end,
    zero_last_element_of_final_dimension,
    upsample,
    downsample,
    downsample_and_movedim,
)
from .validation import (
    validate_model_gradient_sampling_interval,
    validate_freq_taper_frac,
    validate_time_pad_frac,
)

__all__: list[str] = [
    "CallbackState",
    "Callback",
    "create_callback_state",
    "cfl_condition",
    "create_or_pad",
    "reverse_pad",
    "zero_interior",
    "cosine_taper_end",
    "zero_last_element_of_final_dimension",
    "upsample",
    "downsample",
    "downsample_and_movedim",
    "validate_model_gradient_sampling_interval",
    "validate_freq_taper_frac",
    "validate_time_pad_frac",
]
