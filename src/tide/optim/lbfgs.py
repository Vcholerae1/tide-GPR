"""Torch-native limited-memory BFGS."""

from __future__ import annotations

import torch
from torch import Tensor

from .array_utils import (
    _dot,
    _feasible_direction,
)
from .common import (
    _OptimizerRun,
    _TraceRecorder,
    _prepare_initial_state,
)
from .history import _LBFGSHistory
from .line_search import _line_search
from .types import (
    Callback,
    LBFGSOptions,
    Objective,
    OptimizerEventType,
    OptimizerResult,
    OptimizerStatus,
    Preconditioner,
)


def lbfgs_minimize(
    objective: Objective,
    x0: Tensor,
    *,
    preconditioner: Preconditioner | None = None,
    options: LBFGSOptions | None = None,
    lower_bounds: Tensor | float | None = None,
    upper_bounds: Tensor | float | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize an objective with torch-native L-BFGS.

    Box-constrained calls use a projected search path and projected-gradient
    convergence. They are intentionally not advertised as full L-BFGS-B.
    """

    resolved = options or LBFGSOptions()
    state = _prepare_initial_state(objective, x0, lower_bounds, upper_bounds, resolved)
    run = _OptimizerRun(state, _TraceRecorder(resolved.trace, callback), method="lbfgs")
    history = _LBFGSHistory(resolved.history_size, resolved.curvature_tolerance)
    status = run.initial_status(resolved.stopping)
    if status is not None:
        return run.finish(status, 0, history=len(history.s))

    for iteration in range(1, resolved.stopping.max_iter + 1):
        direction = history.direction(state.x, state.grad, preconditioner, state.budget)
        if not torch.isfinite(direction).all():
            return run.finish(
                OptimizerStatus.INVALID_PRECONDITIONER,
                iteration - 1,
                history=len(history.s),
            )
        direction = _feasible_direction(state.x, direction, state.lower, state.upper)
        if _dot(state.grad, direction) >= 0.0 or not torch.any(direction):
            history.clear()
            direction = history.direction(
                state.x, state.grad, preconditioner, state.budget
            )
            direction = _feasible_direction(
                state.x, direction, state.lower, state.upper
            )
        if _dot(state.grad, direction) >= 0.0 or not torch.any(direction):
            status = (
                OptimizerStatus.INVALID_PRECONDITIONER
                if preconditioner is not None
                else OptimizerStatus.BREAKDOWN
            )
            return run.finish(status, iteration - 1, history=len(history.s))

        previous_x = state.x
        previous_f = state.f
        previous_grad = state.grad
        result = _line_search(
            state.evaluator,
            state.x,
            state.f,
            state.grad,
            direction,
            state.lower,
            state.upper,
            resolved.line_search,
        )
        if not result.success:
            return run.finish(
                result.status or OptimizerStatus.LINE_SEARCH_FAILED,
                iteration - 1,
                history=len(history.s),
            )
        state.x, state.f, state.grad = result.x, result.f, result.grad
        history.update(state.x - previous_x, state.grad - previous_grad)
        run.emit(
            OptimizerEventType.STEP,
            iteration,
            alpha=result.alpha,
            line_search_iter=result.evaluations,
            history=len(history.s),
        )
        status = run.step_status(resolved.stopping, iteration, previous_x, previous_f)
        if status is not None:
            return run.finish(status, iteration, history=len(history.s))

    return run.finish(
        OptimizerStatus.MAX_ITERATIONS,
        resolved.stopping.max_iter,
        history=len(history.s),
    )


__all__ = ["lbfgs_minimize"]
