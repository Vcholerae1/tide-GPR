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
    optim,
    padding,
    resampling,
    staggered,
    utils,
    validation,
    wavelets,
)
from .callbacks import Callback, CallbackState, create_callback_state
from .cfl import cfl_condition
from .dispersion import DebyeDispersion
from .maxwell import (
    Born3D,
    BornTM,
    Maxwell3D,
    MaxwellTM,
    born3d,
    borntm,
    maxwell3d,
    maxwell3d_hvp,
    maxwelltm,
    maxwelltm_hvp,
)
from .padding import create_or_pad, reverse_pad, zero_interior
from .resampling import downsample, downsample_and_movedim, upsample
from .typing import (
    BatchedModel2D,
    BatchedModel3D,
    Field2DLike,
    Field3DLike,
    Location2D,
    Location3D,
    MatrixF32,
    Model2D,
    Model2DLike,
    Model3D,
    Model3DLike,
    ReceiverData,
    ReceiverLocation2D,
    ReceiverLocation3D,
    SourceLocation2D,
    SourceLocation3D,
    VectorF32,
    WaveletBatch,
    runtime_typecheck,
)
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
    "optim",
    "padding",
    "resampling",
    "staggered",
    "validation",
    "utils",
    "wavelets",
    # Classes
    "BornTM",
    "Born3D",
    "MaxwellTM",
    "Maxwell3D",
    "CallbackState",
    "DebyeDispersion",
    # Type aliases
    "BatchedModel2D",
    "BatchedModel3D",
    "Callback",
    "Field2DLike",
    "Field3DLike",
    "Location2D",
    "Location3D",
    "MatrixF32",
    "Model2D",
    "Model2DLike",
    "Model3D",
    "Model3DLike",
    "ReceiverData",
    "ReceiverLocation2D",
    "ReceiverLocation3D",
    "SourceLocation2D",
    "SourceLocation3D",
    "VectorF32",
    "WaveletBatch",
    # Functions
    "maxwelltm",
    "maxwell3d",
    "maxwelltm_hvp",
    "maxwell3d_hvp",
    "born3d",
    "borntm",
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
    "runtime_typecheck",
]


__version__ = "0.0.28"
