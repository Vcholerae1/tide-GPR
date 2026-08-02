"""Torch-native truncated-Newton optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from .array_utils import (
    _dot,
    _feasible_direction,
    _validate_operator_output,
)
from .common import (
    _EvaluationBudget,
    _OptimizerRun,
    _TraceRecorder,
    _apply_preconditioner,
    _prepare_initial_state,
)
from .line_search import _line_search
from .types import (
    Callback,
    HessianVectorProduct,
    Objective,
    OptimizerEventType,
    OptimizerResult,
    OptimizerStatus,
    Preconditioner,
    TruncatedNewtonOptions,
)


@dataclass(slots=True)
class _CGDirection:
    direction: Tensor | None
    iterations: int
    status: OptimizerStatus | None


def _newton_direction(
    hessian_vector: HessianVectorProduct,
    preconditioner: Preconditioner | None,
    x: Tensor,
    grad: Tensor,
    options: TruncatedNewtonOptions,
    eta: float,
    budget: _EvaluationBudget,
) -> _CGDirection:
    residual = grad.clone()
    z = _apply_preconditioner(preconditioner, x, residual, budget)
    if not torch.isfinite(z).all():
        return _CGDirection(None, 0, OptimizerStatus.INVALID_PRECONDITIONER)
    rz = _dot(residual, z)
    if rz <= 0.0:
        return _CGDirection(None, 0, OptimizerStatus.INVALID_PRECONDITIONER)
    search = -z
    direction = torch.zeros_like(grad)
    target = eta * float(torch.linalg.vector_norm(grad).item())

    for iteration in range(1, options.max_cg_iter + 1):
        h_search = _validate_operator_output(
            "hessian_vector",
            hessian_vector(x.detach(), search.detach()),
            search,
        )
        budget.hessian += 1
        curvature = _dot(search, h_search)
        if not torch.isfinite(h_search).all() or not isfinite(curvature):
            return _CGDirection(None, iteration - 1, OptimizerStatus.NONFINITE)
        if curvature <= 0.0:
            if iteration == 1:
                direction = search
            return _CGDirection(direction, iteration - 1, None)
        alpha = rz / curvature
        direction = direction + alpha * search
        residual = residual + alpha * h_search
        if float(torch.linalg.vector_norm(residual).item()) <= target:
            return _CGDirection(direction, iteration, None)
        z_new = _apply_preconditioner(preconditioner, x, residual, budget)
        if not torch.isfinite(z_new).all():
            return _CGDirection(None, iteration, OptimizerStatus.INVALID_PRECONDITIONER)
        rz_new = _dot(residual, z_new)
        if rz_new <= 0.0:
            return _CGDirection(None, iteration, OptimizerStatus.INVALID_PRECONDITIONER)
        beta = rz_new / rz
        search = -z_new + beta * search
        z = z_new
        rz = rz_new
    return _CGDirection(direction, options.max_cg_iter, None)


def truncated_newton_minimize(
    objective: Objective,
    hessian_vector: HessianVectorProduct,
    x0: Tensor,
    *,
    preconditioner: Preconditioner | None = None,
    options: TruncatedNewtonOptions | None = None,
    lower_bounds: Tensor | float | None = None,
    upper_bounds: Tensor | float | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize an objective with a Hessian-free truncated-Newton method."""

    resolved = options or TruncatedNewtonOptions()
    state = _prepare_initial_state(objective, x0, lower_bounds, upper_bounds, resolved)
    run = _OptimizerRun(
        state, _TraceRecorder(resolved.trace, callback), method="truncated_newton"
    )
    eta = resolved.eta_initial
    status = run.initial_status(resolved.stopping)
    if status is not None:
        return run.finish(status, 0)

    for iteration in range(1, resolved.stopping.max_iter + 1):
        cg = _newton_direction(
            hessian_vector,
            preconditioner,
            state.x,
            state.grad,
            resolved,
            eta,
            state.budget,
        )
        if cg.status is not None:
            return run.finish(cg.status, iteration - 1)
        if cg.direction is None:
            return run.finish(OptimizerStatus.BREAKDOWN, iteration - 1)
        direction = _feasible_direction(state.x, cg.direction, state.lower, state.upper)
        if _dot(state.grad, direction) >= 0.0 or not torch.any(direction):
            direction = _feasible_direction(
                state.x, -state.grad, state.lower, state.upper
            )
        if _dot(state.grad, direction) >= 0.0 or not torch.any(direction):
            return run.finish(OptimizerStatus.BREAKDOWN, iteration - 1)

        previous_x = state.x
        previous_f = state.f
        previous_grad_norm = max(run.grad_norm(), torch.finfo(state.x.dtype).eps)
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
                result.status or OptimizerStatus.LINE_SEARCH_FAILED, iteration - 1
            )
        state.x, state.f, state.grad = result.x, result.f, result.grad
        run.emit(
            OptimizerEventType.STEP,
            iteration,
            alpha=result.alpha,
            line_search_iter=result.evaluations,
            cg_iter=cg.iterations,
            eta=eta,
        )
        status = run.step_status(resolved.stopping, iteration, previous_x, previous_f)
        if status is not None:
            return run.finish(status, iteration)
        ratio = run.grad_norm() / previous_grad_norm
        eta = min(0.9, max(0.05, ratio**0.5))

    return run.finish(OptimizerStatus.MAX_ITERATIONS, resolved.stopping.max_iter)


__all__ = ["truncated_newton_minimize"]
