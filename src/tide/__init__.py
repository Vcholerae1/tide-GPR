"""TIDE: Torch-based Inversion & Intelligence Engine.

A PyTorch-based library for electromagnetic wave propagation and inversion.
"""

from . import callbacks
from . import cfl
from . import maxwell
from . import padding
from . import resampling
from . import staggered
from . import utils
from . import validation
from . import wavelets

from .callbacks import CallbackState, Callback, create_callback_state
from .cfl import cfl_condition
from .padding import create_or_pad, zero_interior, reverse_pad
from .resampling import upsample, downsample, downsample_and_movedim
from .validation import (
    validate_model_gradient_sampling_interval,
    validate_freq_taper_frac,
    validate_time_pad_frac,
)
from .maxwell import MaxwellTM, maxwelltm
from .wavelets import ricker

__all__ = [
    # Modules
    "callbacks",
    "cfl",
    "maxwell",
    "padding",
    "resampling",
    "staggered",
    "validation",
    "utils",
    "wavelets",
    # Classes
    "MaxwellTM",
    "CallbackState",
    # Type aliases
    "Callback",
    # Functions
    "maxwelltm",
    "create_callback_state",
    # Signal processing
    "upsample",
    "downsample",
    "downsample_and_movedim",
    "cfl_condition",
    # Validation
    "validate_model_gradient_sampling_interval",
    "validate_freq_taper_frac",
    "validate_time_pad_frac",
    # Model padding utilities
    "create_or_pad",
    "zero_interior",
    "reverse_pad",
    # Wavelets
    "ricker",
]


__version__ = "0.0.10"
