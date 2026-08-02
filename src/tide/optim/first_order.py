"""Torch-native steepest descent and nonlinear conjugate gradient."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor

from .array_utils import (
    _dot,
    _feasible_direction,
)
from .common import (
    _OptimizerRun,
    _TraceRecorder,
    _apply_preconditioner,
    _prepare_initial_state,
)
from .line_search import _line_search
from .types import (
    Callback,
    NLCGOptions,
    Objective,
    OptimizerEventType,
    OptimizerOptions,
    OptimizerResult,
    OptimizerStatus,
    Preconditioner,
    SteepestDescentOptions,
)


def _minimize_first_order(
    objective: Objective,
    x0: Tensor,
    *,
    preconditioner: Preconditioner | None,
    options: OptimizerOptions,
    lower_bounds: Tensor | float | None,
    upper_bounds: Tensor | float | None,
    callback: Callback | None,
    method: str,
    beta_max: float | None,
) -> OptimizerResult:
    state = _prepare_initial_state(objective, x0, lower_bounds, upper_bounds, options)
    run = _OptimizerRun(state, _TraceRecorder(options.trace, callback), method)
    status = run.initial_status(options.stopping)
    if status is not None:
        return run.finish(status, 0)

    z = _apply_preconditioner(preconditioner, state.x, state.grad, state.budget)
    if not torch.isfinite(z).all() or _dot(state.grad, z) <= 0.0:
        return run.finish(OptimizerStatus.INVALID_PRECONDITIONER, 0)
    direction = -z
    previous_grad = state.grad.clone()
    previous_z = z.clone()

    for iteration in range(1, options.stopping.max_iter + 1):
        direction = _feasible_direction(state.x, direction, state.lower, state.upper)
        if _dot(state.grad, direction) >= 0.0 or not torch.isfinite(direction).all():
            direction = _feasible_direction(state.x, -z, state.lower, state.upper)
        if _dot(state.grad, direction) >= 0.0 or not torch.any(direction):
            return run.finish(OptimizerStatus.BREAKDOWN, iteration - 1)

        previous_x = state.x
        previous_f = state.f
        result = _line_search(
            state.evaluator,
            state.x,
            state.f,
            state.grad,
            direction,
            state.lower,
            state.upper,
            options.line_search,
        )
        if not result.success:
            return run.finish(
                result.status or OptimizerStatus.LINE_SEARCH_FAILED, iteration - 1
            )
        state.x, state.f, state.grad = result.x, result.f, result.grad
        run.emit(
            OptimizerEventType.STEP,
            iteration,
            alpha=result.alpha,
            line_search_iter=result.evaluations,
        )
        status = run.step_status(options.stopping, iteration, previous_x, previous_f)
        if status is not None:
            return run.finish(status, iteration)

        z = _apply_preconditioner(preconditioner, state.x, state.grad, state.budget)
        if not torch.isfinite(z).all() or _dot(state.grad, z) <= 0.0:
            return run.finish(OptimizerStatus.INVALID_PRECONDITIONER, iteration)
        if beta_max is None:
            direction = -z
        else:
            denominator = _dot(previous_grad, previous_z)
            beta = (
                _dot(state.grad, z - previous_z) / denominator
                if denominator > 0.0
                else 0.0
            )
            beta = max(0.0, beta)
            if not isfinite(beta) or beta > beta_max:
                beta = 0.0
            direction = -z + beta * direction
            previous_grad = state.grad.clone()
            previous_z = z.clone()

    return run.finish(OptimizerStatus.MAX_ITERATIONS, options.stopping.max_iter)


def steepest_descent_minimize(
    objective: Objective,
    x0: Tensor,
    *,
    preconditioner: Preconditioner | None = None,
    options: SteepestDescentOptions | None = None,
    lower_bounds: Tensor | float | None = None,
    upper_bounds: Tensor | float | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize an objective with torch-native steepest descent."""

    return _minimize_first_order(
        objective,
        x0,
        preconditioner=preconditioner,
        options=options or SteepestDescentOptions(),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        callback=callback,
        method="steepest_descent",
        beta_max=None,
    )


def nlcg_minimize(
    objective: Objective,
    x0: Tensor,
    *,
    preconditioner: Preconditioner | None = None,
    options: NLCGOptions | None = None,
    lower_bounds: Tensor | float | None = None,
    upper_bounds: Tensor | float | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize an objective with preconditioned Polak-Ribiere+ NLCG."""

    resolved = options or NLCGOptions()
    return _minimize_first_order(
        objective,
        x0,
        preconditioner=preconditioner,
        options=resolved,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        callback=callback,
        method="nlcg",
        beta_max=resolved.beta_max,
    )


__all__ = ["nlcg_minimize", "steepest_descent_minimize"]
