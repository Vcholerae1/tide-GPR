"""Structured differentiable Maxwell operators."""

from . import _kernel_api as _kernel_api

from .contracts import (
    Acquisition,
    CPML,
    Discretization,
    EMDirection,
    EMModel,
    ExecutionOptions,
    Experiment,
    Observers,
    SourceConvention,
)
from .linearization import LinearizedMaxwell3D, LinearizedMaxwellTM
from .operators import Maxwell3D, MaxwellTM
from .results import EM3DState, EMGradient, ForwardResult, TMState, TangentResult

__all__ = [
    "Acquisition",
    "CPML",
    "Discretization",
    "EM3DState",
    "EMDirection",
    "EMGradient",
    "EMModel",
    "ExecutionOptions",
    "Experiment",
    "ForwardResult",
    "LinearizedMaxwell3D",
    "LinearizedMaxwellTM",
    "Maxwell3D",
    "MaxwellTM",
    "Observers",
    "SourceConvention",
    "TMState",
    "TangentResult",
]
