"""TIDE: Torch-based Inversion & Intelligence Engine.

A PyTorch-based library for electromagnetic wave propagation and inversion.
"""

# Handle OpenMP runtime conflicts (common on Windows with Intel libraries)
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from . import (
    callbacks,
    cfl,
    maxwell,
    padding,
    resampling,
    staggered,
    utils,
    validation,
    wavelets,
)
from .callbacks import Callback, CallbackState, create_callback_state
from .cfl import cfl_condition
from .maxwell import MaxwellTM, maxwelltm
from .padding import create_or_pad, reverse_pad, zero_interior
from .resampling import downsample, downsample_and_movedim, upsample
from .validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)
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


__version__ = "0.0.20"
