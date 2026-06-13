"""Shared types and constants for TIDE optimization prototypes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..typing import VectorF32


FLAG_INIT = 0
FLAG_GRAD = 1
FLAG_CONV = 2
FLAG_NSTE = 3
FLAG_FAIL = 4
FLAG_PREC = 5
FLAG_DES = 6
FLAG_HESS = 7

STATUS_CONVERGED = "converged"
STATUS_MAX_ITER = "max_iter"
STATUS_LINE_SEARCH_FAILED = "line_search_failed"
STATUS_MAX_EVALUATIONS = "max_evaluations"
STATUS_NONFINITE = "nonfinite"
STATUS_INNER_CG_FAILED = "inner_cg_failed"

TASK_NEW_STEP = 0
TASK_NEW_GRAD = 1
TASK_FAILURE = 2

Objective = Callable[[VectorF32, VectorF32], float]
LinearOperator = Callable[[VectorF32, VectorF32], None]
Preconditioner = Callable[[VectorF32, VectorF32, VectorF32], None]
HessianVectorProduct = Callable[[VectorF32, VectorF32, VectorF32], None]
Callback = Callable[["OptimizerTraceEntry"], None]
LineSearchMethod = Literal["weak_wolfe", "hager_zhang", "more_thuente"]


@dataclass(slots=True)
class OptimizerOptions:
    """Common CPU-state optimizer options."""

    max_iter: int = 10000
    initial_step: float = 1.0
    max_line_search: int = 20
    line_search: LineSearchMethod = "weak_wolfe"
    tolerance: float = 1e-8
    wolfe_c1: float = 1e-4
    wolfe_c2: float = 0.9
    growth: float = 10.0
    hager_zhang_delta: float = 0.1
    hager_zhang_epsilon: float = 1e-6
    hager_zhang_gamma: float = 2.0 / 3.0
    hager_zhang_rho: float = 5.0
    hager_zhang_finite_shrink: float = 0.1
    hager_zhang_max_finite_checks: int = 50
    more_thuente_xtol: float = 1e-8
    more_thuente_step_min: float = 1e-16
    more_thuente_step_max: float = 65536.0
    bound_margin: float = 0.0
    max_evaluations: int | None = None
    record_trace: bool = True
    dtype: np.dtype[Any] = field(default_factory=lambda: np.dtype(np.float32))

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype != np.dtype(np.float32):
            raise ValueError("Optimizers currently use float32 CPU state.")
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.initial_step <= 0.0 or not np.isfinite(self.initial_step):
            raise ValueError("initial_step must be finite and positive.")
        if self.max_line_search <= 0:
            raise ValueError("max_line_search must be positive.")
        if self.line_search not in ("weak_wolfe", "hager_zhang", "more_thuente"):
            raise ValueError(
                "line_search must be 'weak_wolfe', 'hager_zhang', or "
                "'more_thuente'."
            )
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        if not (0.0 < self.wolfe_c1 < self.wolfe_c2 < 1.0):
            raise ValueError("Wolfe parameters must satisfy 0 < c1 < c2 < 1.")
        if self.growth <= 1.0 or not np.isfinite(self.growth):
            raise ValueError("growth must be finite and greater than 1.")
        if not (
            0.0 < self.hager_zhang_delta <= self.wolfe_c2 < 1.0
            and self.hager_zhang_delta < 0.5
        ):
            raise ValueError(
                "Hager-Zhang parameters must satisfy 0 < delta <= wolfe_c2 < 1 "
                "and delta < 0.5."
            )
        if self.hager_zhang_epsilon < 0.0:
            raise ValueError("hager_zhang_epsilon must be non-negative.")
        if not (0.0 < self.hager_zhang_gamma < 1.0):
            raise ValueError("hager_zhang_gamma must be in (0, 1).")
        if self.hager_zhang_rho <= 1.0 or not np.isfinite(self.hager_zhang_rho):
            raise ValueError("hager_zhang_rho must be finite and greater than 1.")
        if not (0.0 < self.hager_zhang_finite_shrink < 1.0):
            raise ValueError("hager_zhang_finite_shrink must be in (0, 1).")
        if self.hager_zhang_max_finite_checks < 0:
            raise ValueError("hager_zhang_max_finite_checks must be non-negative.")
        if self.more_thuente_xtol < 0.0:
            raise ValueError("more_thuente_xtol must be non-negative.")
        if not (
            0.0 < self.more_thuente_step_min < self.more_thuente_step_max
            and np.isfinite(self.more_thuente_step_min)
            and np.isfinite(self.more_thuente_step_max)
        ):
            raise ValueError(
                "More-Thuente step bounds must satisfy "
                "0 < more_thuente_step_min < more_thuente_step_max."
            )
        if self.bound_margin < 0.0:
            raise ValueError("bound_margin must be non-negative.")
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive when provided.")


@dataclass(slots=True)
class SteepestDescentOptions(OptimizerOptions):
    """Options for steepest descent."""


@dataclass(slots=True)
class NLCGOptions(OptimizerOptions):
    """Options for nonlinear conjugate gradient."""

    beta_abs_max: float = 1e5

    def __post_init__(self) -> None:
        OptimizerOptions.__post_init__(self)
        if self.beta_abs_max <= 0.0 or not np.isfinite(self.beta_abs_max):
            raise ValueError("beta_abs_max must be finite and positive.")


@dataclass(slots=True)
class LBFGSOptions(OptimizerOptions):
    """Options for L-BFGS."""

    history_size: int = 10

    def __post_init__(self) -> None:
        OptimizerOptions.__post_init__(self)
        if self.history_size <= 0:
            raise ValueError("history_size must be positive.")


@dataclass(slots=True)
class TruncatedNewtonOptions(OptimizerOptions):
    """Options for truncated Newton."""

    max_cg_iter: int = 10
    eta_initial: float = 0.9

    def __post_init__(self) -> None:
        OptimizerOptions.__post_init__(self)
        if self.max_cg_iter <= 0:
            raise ValueError("max_cg_iter must be positive.")
        if self.eta_initial <= 0.0 or not np.isfinite(self.eta_initial):
            raise ValueError("eta_initial must be finite and positive.")


@dataclass(slots=True)
class CGNROptions:
    """Options for CGNR and preconditioned CGNR."""

    max_iter: int = 100
    tolerance: float = 1e-8
    max_matvec: int | None = None
    record_trace: bool = True
    dtype: np.dtype[Any] = field(default_factory=lambda: np.dtype(np.float32))

    def __post_init__(self) -> None:
        self.dtype = np.dtype(self.dtype)
        if self.dtype != np.dtype(np.float32):
            raise ValueError("CGNR currently uses float32 CPU state.")
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")
        if self.max_matvec is not None and self.max_matvec <= 0:
            raise ValueError("max_matvec must be positive when provided.")


@dataclass(slots=True)
class OptimizerTraceEntry:
    """One optimizer event."""

    flag: int
    iteration: int
    evaluations: int
    f: float
    alpha: float
    line_search_iter: int
    accepted: bool
    task: int
    x: VectorF32
    grad: VectorF32
    q0: float
    q: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizerResult:
    """Result returned by CPU-state optimizers."""

    x: VectorF32
    f: float
    grad: VectorF32
    status: str
    flag: int
    success: bool
    n_iter: int
    n_eval: int
    n_prec: int
    n_hess: int
    elapsed_s: float
    trace: list[OptimizerTraceEntry]


@dataclass(slots=True)
class CGNRTraceEntry:
    """One CGNR iteration."""

    iteration: int
    f: float
    residual_norm: float
    normal_residual_norm: float
    alpha: float
    beta: float
    x: VectorF32
    residual: VectorF32
    normal_residual: VectorF32
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CGNRResult:
    """Result returned by :func:`cgnr_solve`."""

    x: VectorF32
    f: float
    residual: VectorF32
    normal_residual: VectorF32
    status: str
    success: bool
    n_iter: int
    n_forward: int
    n_adjoint: int
    n_prec: int
    elapsed_s: float
    trace: list[CGNRTraceEntry]


def _make_trace(
    *,
    flag: int,
    iteration: int,
    evaluations: int,
    f: float,
    alpha: float,
    line_search_iter: int,
    accepted: bool,
    task: int,
    x: VectorF32,
    grad: VectorF32,
    q0: float,
    q: float,
    metadata: dict[str, Any],
) -> OptimizerTraceEntry:
    return OptimizerTraceEntry(
        flag=flag,
        iteration=iteration,
        evaluations=evaluations,
        f=float(f),
        alpha=float(alpha),
        line_search_iter=int(line_search_iter),
        accepted=accepted,
        task=int(task),
        x=x.copy(),
        grad=grad.copy(),
        q0=float(q0),
        q=float(q),
        metadata=dict(metadata),
    )


__all__ = [
    "Callback",
    "CGNROptions",
    "CGNRResult",
    "CGNRTraceEntry",
    "FLAG_CONV",
    "FLAG_DES",
    "FLAG_FAIL",
    "FLAG_GRAD",
    "FLAG_HESS",
    "FLAG_INIT",
    "FLAG_NSTE",
    "FLAG_PREC",
    "HessianVectorProduct",
    "LineSearchMethod",
    "LBFGSOptions",
    "LinearOperator",
    "NLCGOptions",
    "Objective",
    "OptimizerOptions",
    "OptimizerResult",
    "OptimizerTraceEntry",
    "Preconditioner",
    "SteepestDescentOptions",
    "STATUS_CONVERGED",
    "STATUS_INNER_CG_FAILED",
    "STATUS_LINE_SEARCH_FAILED",
    "STATUS_MAX_EVALUATIONS",
    "STATUS_MAX_ITER",
    "STATUS_NONFINITE",
    "TASK_FAILURE",
    "TASK_NEW_GRAD",
    "TASK_NEW_STEP",
    "TruncatedNewtonOptions",
]
