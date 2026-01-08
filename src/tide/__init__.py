"""TIDE: Torch-based Inversion & Intelligence Engine.

A PyTorch-based library for electromagnetic wave propagation and inversion.
"""

from . import common
from . import maxwell
from . import staggered
from . import utils
from . import wavelets

from .common import (
    CallbackState,
    Callback,
    create_callback_state,
    upsample,
    downsample,
    downsample_and_movedim,
    cfl_condition,
    validate_model_gradient_sampling_interval,
    validate_freq_taper_frac,
    validate_time_pad_frac,
    create_or_pad,
    zero_interior,
    reverse_pad,
    IGNORE_LOCATION,
)
from .maxwell import MaxwellTM, Maxwell3D, maxwelltm, maxwell3d
from .wavelets import ricker

__all__ = [
    # Modules
    "common",
    "maxwell",
    "staggered",
    "utils",
    "wavelets",
    # Classes
    "MaxwellTM",
    "Maxwell3D",
    "CallbackState",
    # Type aliases
    "Callback",
    # Functions
    "maxwelltm",
    "maxwell3d",
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
    # Constants
    "IGNORE_LOCATION",
    # Wavelets
    "ricker",
]


__version__ = "0.1.0"
