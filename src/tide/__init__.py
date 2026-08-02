"""TIDE: Torch-based Inversion & Intelligence Engine.

A PyTorch-based library for electromagnetic wave propagation and inversion.
"""

from . import (
    callbacks,
    cfl,
    core,
    maxwell,
    optim,
    padding,
    resampling,
    staggered,
    utils,
    validation,
    wavelets,
    workflow,
)
from .callbacks import Callback, CallbackState, create_callback_state
from .cfl import cfl_condition
from .core import (
    BackendPreference,
    ComputeMode,
    Dimension,
    FallbackPolicy,
    RuntimeOptions,
    SimulationPlan,
    StorageMode,
    StorageOptions,
    compile_simulation_plan,
    normalize_backend_request,
    select_backend,
)
from .dispersion import DebyeDispersion
from .maxwell import (
    Born3D,
    BornTM,
    Maxwell3D,
    MaxwellTM,
    TM2DLinearizationContext,
    born3d,
    borntm,
    maxwell3d,
    maxwell3d_hvp,
    maxwelltm,
    maxwelltm_hvp,
    linearize_maxwelltm,
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
    "core",
    "maxwell",
    "optim",
    "padding",
    "resampling",
    "staggered",
    "validation",
    "utils",
    "wavelets",
    "workflow",
    # Classes
    "BornTM",
    "Born3D",
    "MaxwellTM",
    "TM2DLinearizationContext",
    "Maxwell3D",
    "CallbackState",
    "DebyeDispersion",
    "BackendPreference",
    "ComputeMode",
    "Dimension",
    "FallbackPolicy",
    "RuntimeOptions",
    "SimulationPlan",
    "StorageMode",
    "StorageOptions",
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
    "linearize_maxwelltm",
    "maxwell3d_hvp",
    "born3d",
    "borntm",
    "create_callback_state",
    "compile_simulation_plan",
    "normalize_backend_request",
    "select_backend",
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


__version__ = "0.0.32"
