"""Torch-native optimization algorithms for TIDE inverse problems."""

from .cgnr import cgnr_solve
from .first_order import nlcg_minimize, steepest_descent_minimize
from .lbfgs import lbfgs_minimize
from .truncated_newton import truncated_newton_minimize
from .types import (
    CGNROptions,
    CGNRResult,
    CGNRTraceEntry,
    HessianVectorProduct,
    LBFGSOptions,
    LineSearchMethod,
    LineSearchOptions,
    NLCGOptions,
    Objective,
    OptimizerEvent,
    OptimizerEventType,
    OptimizerOptions,
    OptimizerResult,
    OptimizerStatus,
    OptimizerTraceEntry,
    Preconditioner,
    SteepestDescentOptions,
    StoppingCriteria,
    TraceOptions,
    TruncatedNewtonOptions,
)

__all__ = [
    "CGNROptions",
    "CGNRResult",
    "CGNRTraceEntry",
    "HessianVectorProduct",
    "LBFGSOptions",
    "LineSearchMethod",
    "LineSearchOptions",
    "NLCGOptions",
    "Objective",
    "OptimizerEvent",
    "OptimizerEventType",
    "OptimizerOptions",
    "OptimizerResult",
    "OptimizerStatus",
    "OptimizerTraceEntry",
    "Preconditioner",
    "SteepestDescentOptions",
    "StoppingCriteria",
    "TraceOptions",
    "TruncatedNewtonOptions",
    "cgnr_solve",
    "lbfgs_minimize",
    "nlcg_minimize",
    "steepest_descent_minimize",
    "truncated_newton_minimize",
]
