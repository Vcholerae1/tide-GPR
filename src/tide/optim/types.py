"""Public types for the torch-native TIDE optimizers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

from torch import Tensor


ScalarLoss: TypeAlias = float | Tensor
Objective: TypeAlias = Callable[[Tensor], tuple[ScalarLoss, Tensor]]
LinearOperator: TypeAlias = Callable[[Tensor], Tensor]
Preconditioner: TypeAlias = Callable[[Tensor, Tensor], Tensor]
HessianVectorProduct: TypeAlias = Callable[[Tensor, Tensor], Tensor]
LineSearchMethod = Literal["weak_wolfe", "strong_wolfe", "armijo"]


class OptimizerStatus(StrEnum):
    """Why an optimizer stopped."""

    CONVERGED_GRADIENT = "converged_gradient"
    CONVERGED_FUNCTION = "converged_function"
    CONVERGED_STEP = "converged_step"
    MAX_ITERATIONS = "max_iterations"
    MAX_EVALUATIONS = "max_evaluations"
    LINE_SEARCH_FAILED = "line_search_failed"
    NONFINITE = "nonfinite"
    BREAKDOWN = "breakdown"
    INVALID_PRECONDITIONER = "invalid_preconditioner"

    @property
    def success(self) -> bool:
        return self in {
            OptimizerStatus.CONVERGED_GRADIENT,
            OptimizerStatus.CONVERGED_FUNCTION,
            OptimizerStatus.CONVERGED_STEP,
        }


class OptimizerEventType(StrEnum):
    """Lifecycle event delivered to optimizer callbacks."""

    INITIAL = "initial"
    STEP = "step"
    TERMINATED = "terminated"


@dataclass(slots=True, frozen=True)
class StoppingCriteria:
    """Common nonlinear-optimizer stopping criteria."""

    max_iter: int = 100
    max_evaluations: int | None = None
    gtol: float = 1e-6
    ftol: float = 1e-9
    xtol: float = 1e-9

    def __post_init__(self) -> None:
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.max_evaluations is not None and self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive when provided.")
        for name, value in (
            ("gtol", self.gtol),
            ("ftol", self.ftol),
            ("xtol", self.xtol),
        ):
            if value < 0.0 or not isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(slots=True, frozen=True)
class LineSearchOptions:
    """Line-search settings.

    ``weak_wolfe`` follows the projected bracketing/dichotomy strategy used by
    the SEISCOPE Optimization Toolbox. ``strong_wolfe`` is used only for
    unconstrained problems; bounded strong-Wolfe requests fall back to
    projected Armijo.
    """

    method: LineSearchMethod = "weak_wolfe"
    initial_step: float = 1.0
    max_steps: int = 20
    c1: float = 1e-4
    c2: float = 0.9
    contraction: float = 0.5
    expansion: float = 2.0
    growth: float = 10.0
    step_min: float = 1e-16
    step_max: float = float("inf")
    zoom_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.method not in ("weak_wolfe", "strong_wolfe", "armijo"):
            raise ValueError(
                "method must be 'weak_wolfe', 'strong_wolfe', or 'armijo'."
            )
        if self.initial_step <= 0.0 or not isfinite(self.initial_step):
            raise ValueError("initial_step must be finite and positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if not (0.0 < self.c1 < self.c2 < 1.0):
            raise ValueError("line-search constants must satisfy 0 < c1 < c2 < 1.")
        if not (0.0 < self.contraction < 1.0):
            raise ValueError("contraction must be in (0, 1).")
        if self.expansion <= 1.0 or not isfinite(self.expansion):
            raise ValueError("expansion must be finite and greater than 1.")
        if self.growth <= 1.0 or not isfinite(self.growth):
            raise ValueError("growth must be finite and greater than 1.")
        if not (0.0 < self.step_min < self.step_max):
            raise ValueError("step bounds must satisfy 0 < step_min < step_max.")
        if self.zoom_tolerance < 0.0 or not isfinite(self.zoom_tolerance):
            raise ValueError("zoom_tolerance must be finite and non-negative.")


@dataclass(slots=True, frozen=True)
class TraceOptions:
    """Controls retained optimization history."""

    record: bool = False
    store_tensors: bool = False
    snapshot_interval: int = 1
    snapshot_device: Literal["cpu", "same"] = "cpu"

    def __post_init__(self) -> None:
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive.")
        if self.snapshot_device not in ("cpu", "same"):
            raise ValueError("snapshot_device must be 'cpu' or 'same'.")
        if self.store_tensors and not self.record:
            raise ValueError("store_tensors=True requires record=True.")


@dataclass(slots=True, frozen=True)
class OptimizerOptions:
    """Shared nonlinear-optimizer options."""

    stopping: StoppingCriteria = field(default_factory=StoppingCriteria)
    line_search: LineSearchOptions = field(default_factory=LineSearchOptions)
    trace: TraceOptions = field(default_factory=TraceOptions)


@dataclass(slots=True, frozen=True)
class SteepestDescentOptions(OptimizerOptions):
    """Options for steepest descent."""


@dataclass(slots=True, frozen=True)
class NLCGOptions(OptimizerOptions):
    """Options for nonlinear conjugate gradient."""

    beta_max: float = 1e5

    def __post_init__(self) -> None:
        if self.beta_max <= 0.0 or not isfinite(self.beta_max):
            raise ValueError("beta_max must be finite and positive.")


@dataclass(slots=True, frozen=True)
class LBFGSOptions(OptimizerOptions):
    """Options for limited-memory BFGS."""

    history_size: int = 10
    curvature_tolerance: float = 1e-10
    relative_objective_tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.history_size <= 0:
            raise ValueError("history_size must be positive.")
        if self.curvature_tolerance < 0.0 or not isfinite(self.curvature_tolerance):
            raise ValueError("curvature_tolerance must be finite and non-negative.")
        if self.relative_objective_tolerance is not None and (
            self.relative_objective_tolerance < 0.0
            or not isfinite(self.relative_objective_tolerance)
        ):
            raise ValueError(
                "relative_objective_tolerance must be finite and non-negative."
            )


@dataclass(slots=True, frozen=True)
class TruncatedNewtonOptions(OptimizerOptions):
    """Options for truncated Newton."""

    max_cg_iter: int = 10
    eta_initial: float = 0.5

    def __post_init__(self) -> None:
        if self.max_cg_iter <= 0:
            raise ValueError("max_cg_iter must be positive.")
        if not (0.0 < self.eta_initial < 1.0):
            raise ValueError("eta_initial must be in (0, 1).")


@dataclass(slots=True, frozen=True)
class CGNROptions:
    """Options for CGNR/PCGNR least-squares solves."""

    max_iter: int = 100
    max_matvec: int | None = None
    rtol: float = 1e-6
    atol: float = 0.0
    trace: TraceOptions = field(default_factory=TraceOptions)

    def __post_init__(self) -> None:
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.max_matvec is not None and self.max_matvec < 2:
            raise ValueError(
                "max_matvec must be at least 2 to evaluate the initial residual."
            )
        if any(value < 0.0 or not isfinite(value) for value in (self.rtol, self.atol)):
            raise ValueError("rtol and atol must be finite and non-negative.")


@dataclass(slots=True)
class OptimizerEvent:
    """One callback event.

    ``x`` and ``grad`` are detached views of live optimizer state. Callbacks
    must not mutate them.
    """

    event: OptimizerEventType
    iteration: int
    evaluations: int
    f: float
    grad_norm: float
    x: Tensor
    grad: Tensor
    status: OptimizerStatus | None = None
    alpha: float = 0.0
    line_search_iter: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


Callback: TypeAlias = Callable[[OptimizerEvent], None]


@dataclass(slots=True)
class OptimizerTraceEntry:
    """Retained scalar history, with optional tensor snapshots."""

    event: OptimizerEventType
    iteration: int
    evaluations: int
    f: float
    grad_norm: float
    alpha: float
    line_search_iter: int
    status: OptimizerStatus | None
    metadata: dict[str, Any] = field(default_factory=dict)
    x: Tensor | None = None
    grad: Tensor | None = None


@dataclass(slots=True)
class OptimizerResult:
    """Result returned by a nonlinear optimizer."""

    x: Tensor
    f: float
    grad: Tensor
    status: OptimizerStatus
    success: bool
    n_iter: int
    n_eval: int
    n_prec: int
    n_hess: int
    elapsed_s: float
    trace: list[OptimizerTraceEntry]


@dataclass(slots=True)
class CGNRTraceEntry:
    """One retained CGNR iteration."""

    iteration: int
    f: float
    residual_norm: float
    normal_residual_norm: float
    alpha: float
    beta: float
    metadata: dict[str, Any] = field(default_factory=dict)
    x: Tensor | None = None
    residual: Tensor | None = None
    normal_residual: Tensor | None = None


@dataclass(slots=True)
class CGNRResult:
    """Result returned by :func:`cgnr_solve`."""

    x: Tensor
    f: float
    residual: Tensor
    normal_residual: Tensor
    status: OptimizerStatus
    success: bool
    n_iter: int
    n_forward: int
    n_adjoint: int
    n_prec: int
    elapsed_s: float
    trace: list[CGNRTraceEntry]


__all__ = [
    "Callback",
    "CGNROptions",
    "CGNRResult",
    "CGNRTraceEntry",
    "HessianVectorProduct",
    "LBFGSOptions",
    "LineSearchMethod",
    "LineSearchOptions",
    "LinearOperator",
    "NLCGOptions",
    "Objective",
    "OptimizerEvent",
    "OptimizerEventType",
    "OptimizerOptions",
    "OptimizerResult",
    "OptimizerStatus",
    "OptimizerTraceEntry",
    "Preconditioner",
    "ScalarLoss",
    "SteepestDescentOptions",
    "StoppingCriteria",
    "TraceOptions",
    "TruncatedNewtonOptions",
]
