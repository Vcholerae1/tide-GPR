"""Shared execution machinery for torch-native optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from .array_utils import (
    _as_model_tensor,
    _norm_inf,
    _prepare_bounds,
    _project,
    _projected_gradient,
    _validate_operator_output,
)
from .types import (
    Callback,
    Objective,
    OptimizerEvent,
    OptimizerEventType,
    OptimizerOptions,
    OptimizerResult,
    OptimizerStatus,
    OptimizerTraceEntry,
    Preconditioner,
    StoppingCriteria,
    TraceOptions,
)


class _BudgetExhausted(RuntimeError):
    pass


@dataclass(slots=True)
class _EvaluationBudget:
    max_objective: int | None
    objective: int = 0
    preconditioner: int = 0
    hessian: int = 0

    def claim_objective(self) -> None:
        if self.max_objective is not None and self.objective >= self.max_objective:
            raise _BudgetExhausted
        self.objective += 1


class _ObjectiveEvaluator:
    def __init__(self, objective: Objective, budget: _EvaluationBudget) -> None:
        self.objective = objective
        self.budget = budget

    def __call__(self, x: Tensor) -> tuple[float, Tensor]:
        self.budget.claim_objective()
        output = self.objective(x.detach())
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("objective must return a (loss, gradient) tuple.")
        loss, grad = output
        if isinstance(loss, Tensor):
            if loss.numel() != 1:
                raise ValueError("objective loss tensor must be scalar.")
            f = float(loss.detach().item())
        else:
            f = float(loss)
        grad = _validate_operator_output("objective gradient", grad, x)
        return f, grad


def _apply_preconditioner(
    preconditioner: Preconditioner | None,
    x: Tensor,
    vector: Tensor,
    budget: _EvaluationBudget,
) -> Tensor:
    if preconditioner is None:
        return vector.clone()
    budget.preconditioner += 1
    return _validate_operator_output(
        "preconditioner", preconditioner(x.detach(), vector.detach()), vector
    )


def _finite_state(f: float, *values: Tensor) -> bool:
    return isfinite(f) and all(bool(torch.isfinite(v).all()) for v in values)


def _termination_status(
    *,
    x: Tensor,
    grad: Tensor,
    lower: Tensor | None,
    upper: Tensor | None,
    stopping: StoppingCriteria,
    previous_x: Tensor | None = None,
    previous_f: float | None = None,
    f: float | None = None,
) -> OptimizerStatus | None:
    projected_grad = _projected_gradient(x, grad, lower, upper)
    if _norm_inf(projected_grad) <= stopping.gtol:
        return OptimizerStatus.CONVERGED_GRADIENT
    if previous_x is None or previous_f is None or f is None:
        return None
    f_scale = max(1.0, abs(previous_f), abs(f))
    if abs(previous_f - f) <= stopping.ftol * f_scale:
        return OptimizerStatus.CONVERGED_FUNCTION
    x_scale = max(1.0, _norm_inf(previous_x), _norm_inf(x))
    if _norm_inf(x - previous_x) <= stopping.xtol * x_scale:
        return OptimizerStatus.CONVERGED_STEP
    return None


class _TraceRecorder:
    def __init__(
        self,
        options: TraceOptions,
        callback: Callback | None,
    ) -> None:
        self.options = options
        self.callback = callback
        self.entries: list[OptimizerTraceEntry] = []

    def emit(
        self,
        *,
        event: OptimizerEventType,
        iteration: int,
        evaluations: int,
        f: float,
        grad_norm: float,
        x: Tensor,
        grad: Tensor,
        status: OptimizerStatus | None = None,
        alpha: float = 0.0,
        line_search_iter: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        details = {} if metadata is None else dict(metadata)
        if self.callback is not None:
            self.callback(
                OptimizerEvent(
                    event=event,
                    iteration=iteration,
                    evaluations=evaluations,
                    f=f,
                    grad_norm=grad_norm,
                    x=x.detach(),
                    grad=grad.detach(),
                    status=status,
                    alpha=alpha,
                    line_search_iter=line_search_iter,
                    metadata=details,
                )
            )
        if not self.options.record:
            return
        snapshot = self.options.store_tensors and (
            event != OptimizerEventType.STEP
            or iteration % self.options.snapshot_interval == 0
        )
        snapshot_x: Tensor | None = None
        snapshot_grad: Tensor | None = None
        if snapshot:
            snapshot_x = x.detach().clone()
            snapshot_grad = grad.detach().clone()
            if self.options.snapshot_device == "cpu":
                snapshot_x = snapshot_x.cpu()
                snapshot_grad = snapshot_grad.cpu()
        self.entries.append(
            OptimizerTraceEntry(
                event=event,
                iteration=iteration,
                evaluations=evaluations,
                f=f,
                grad_norm=grad_norm,
                alpha=alpha,
                line_search_iter=line_search_iter,
                status=status,
                metadata=details,
                x=snapshot_x,
                grad=snapshot_grad,
            )
        )


@dataclass(slots=True)
class _InitialState:
    start: float
    x: Tensor
    lower: Tensor | None
    upper: Tensor | None
    evaluator: _ObjectiveEvaluator
    budget: _EvaluationBudget
    f: float
    grad: Tensor


@dataclass(slots=True)
class _OptimizerRun:
    """Shared lifecycle state for nonlinear optimizer implementations."""

    state: _InitialState
    trace: _TraceRecorder
    method: str

    def grad_norm(self) -> float:
        return _norm_inf(
            _projected_gradient(
                self.state.x,
                self.state.grad,
                self.state.lower,
                self.state.upper,
            )
        )

    def emit(
        self,
        event: OptimizerEventType,
        iteration: int,
        *,
        status: OptimizerStatus | None = None,
        alpha: float = 0.0,
        line_search_iter: int = 0,
        **metadata: Any,
    ) -> None:
        self.trace.emit(
            event=event,
            iteration=iteration,
            evaluations=self.state.budget.objective,
            f=self.state.f,
            grad_norm=self.grad_norm(),
            x=self.state.x,
            grad=self.state.grad,
            status=status,
            alpha=alpha,
            line_search_iter=line_search_iter,
            metadata={"method": self.method, **metadata},
        )

    def initial_status(self, stopping: StoppingCriteria) -> OptimizerStatus | None:
        self.emit(OptimizerEventType.INITIAL, 0)
        if not _finite_state(self.state.f, self.state.x, self.state.grad):
            return OptimizerStatus.NONFINITE
        status = _termination_status(
            x=self.state.x,
            grad=self.state.grad,
            lower=self.state.lower,
            upper=self.state.upper,
            stopping=stopping,
        )
        if status is not None:
            return status
        if stopping.max_iter == 0:
            return OptimizerStatus.MAX_ITERATIONS
        return None

    def step_status(
        self,
        stopping: StoppingCriteria,
        iteration: int,
        previous_x: Tensor,
        previous_f: float,
    ) -> OptimizerStatus | None:
        if not _finite_state(self.state.f, self.state.x, self.state.grad):
            return OptimizerStatus.NONFINITE
        status = _termination_status(
            x=self.state.x,
            grad=self.state.grad,
            lower=self.state.lower,
            upper=self.state.upper,
            stopping=stopping,
            previous_x=previous_x,
            previous_f=previous_f,
            f=self.state.f,
        )
        if status is not None:
            return status
        if iteration >= stopping.max_iter:
            return OptimizerStatus.MAX_ITERATIONS
        return None

    def finish(
        self,
        status: OptimizerStatus,
        n_iter: int,
        **metadata: Any,
    ) -> OptimizerResult:
        self.emit(
            OptimizerEventType.TERMINATED,
            n_iter,
            status=status,
            **metadata,
        )
        return _make_result(
            state=self.state,
            status=status,
            n_iter=n_iter,
            trace=self.trace,
        )


def _prepare_initial_state(
    objective: Objective,
    x0: Tensor,
    lower_bounds: Tensor | float | None,
    upper_bounds: Tensor | float | None,
    options: OptimizerOptions,
) -> _InitialState:
    start = perf_counter()
    x = _as_model_tensor("x0", x0)
    lower, upper = _prepare_bounds(x, lower_bounds, upper_bounds)
    x = _project(x, lower, upper)
    budget = _EvaluationBudget(options.stopping.max_evaluations)
    evaluator = _ObjectiveEvaluator(objective, budget)
    f, grad = evaluator(x)
    return _InitialState(start, x, lower, upper, evaluator, budget, f, grad)


def _make_result(
    *,
    state: _InitialState,
    status: OptimizerStatus,
    n_iter: int,
    trace: _TraceRecorder,
) -> OptimizerResult:
    return OptimizerResult(
        x=state.x.detach().clone(),
        f=float(state.f),
        grad=state.grad.detach().clone(),
        status=status,
        success=status.success,
        n_iter=n_iter,
        n_eval=state.budget.objective,
        n_prec=state.budget.preconditioner,
        n_hess=state.budget.hessian,
        elapsed_s=perf_counter() - state.start,
        trace=trace.entries,
    )
