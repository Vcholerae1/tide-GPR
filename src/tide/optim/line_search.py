"""Torch-native projected Armijo and strong-Wolfe line searches."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from .array_utils import _dot, _project
from .common import _BudgetExhausted, _ObjectiveEvaluator
from .types import LineSearchOptions, OptimizerStatus


@dataclass(slots=True)
class _LineSearchResult:
    success: bool
    x: Tensor
    f: float
    grad: Tensor
    alpha: float
    evaluations: int
    status: OptimizerStatus | None = None


def _failure(
    x: Tensor,
    f: float,
    grad: Tensor,
    evaluations: int,
    status: OptimizerStatus,
) -> _LineSearchResult:
    return _LineSearchResult(
        success=False,
        x=x,
        f=f,
        grad=grad,
        alpha=0.0,
        evaluations=evaluations,
        status=status,
    )


def _projected_armijo(
    evaluator: _ObjectiveEvaluator,
    x: Tensor,
    f: float,
    grad: Tensor,
    direction: Tensor,
    lower: Tensor,
    upper: Tensor,
    options: LineSearchOptions,
) -> _LineSearchResult:
    alpha = min(max(options.initial_step, options.step_min), options.step_max)
    evaluations = 0
    for _ in range(options.max_steps):
        trial_x = _project(x + alpha * direction, lower, upper)
        step = trial_x - x
        slope = _dot(grad, step)
        if slope >= 0.0 or not torch.any(step):
            alpha *= options.contraction
            if alpha < options.step_min:
                break
            continue
        try:
            trial_f, trial_grad = evaluator(trial_x)
        except _BudgetExhausted:
            return _failure(x, f, grad, evaluations, OptimizerStatus.MAX_EVALUATIONS)
        evaluations += 1
        if isfinite(trial_f) and torch.isfinite(trial_grad).all():
            if trial_f <= f + options.c1 * slope:
                return _LineSearchResult(
                    True, trial_x, trial_f, trial_grad, alpha, evaluations
                )
        alpha *= options.contraction
        if alpha < options.step_min:
            break
    return _failure(x, f, grad, evaluations, OptimizerStatus.LINE_SEARCH_FAILED)


def _strong_wolfe(
    evaluator: _ObjectiveEvaluator,
    x: Tensor,
    f: float,
    grad: Tensor,
    direction: Tensor,
    options: LineSearchOptions,
) -> _LineSearchResult:
    slope0 = _dot(grad, direction)
    if slope0 >= 0.0 or not isfinite(slope0):
        return _failure(x, f, grad, 0, OptimizerStatus.LINE_SEARCH_FAILED)

    evaluations = 0

    def evaluate(alpha: float) -> tuple[Tensor, float, Tensor, float] | None:
        nonlocal evaluations
        trial_x = x + alpha * direction
        try:
            trial_f, trial_grad = evaluator(trial_x)
        except _BudgetExhausted:
            return None
        evaluations += 1
        slope = _dot(trial_grad, direction)
        return trial_x, trial_f, trial_grad, slope

    def zoom(
        lo: float,
        hi: float,
        phi_lo: float,
    ) -> _LineSearchResult:
        nonlocal evaluations
        for _ in range(options.max_steps - evaluations):
            alpha = 0.5 * (lo + hi)
            if abs(hi - lo) <= options.zoom_tolerance * max(1.0, abs(alpha)):
                break
            trial = evaluate(alpha)
            if trial is None:
                return _failure(
                    x, f, grad, evaluations, OptimizerStatus.MAX_EVALUATIONS
                )
            trial_x, phi, trial_grad, slope = trial
            if not isfinite(phi) or not torch.isfinite(trial_grad).all():
                hi = alpha
                continue
            if phi > f + options.c1 * alpha * slope0 or phi >= phi_lo:
                hi = alpha
            else:
                if abs(slope) <= -options.c2 * slope0:
                    return _LineSearchResult(
                        True, trial_x, phi, trial_grad, alpha, evaluations
                    )
                if slope * (hi - lo) >= 0.0:
                    hi = lo
                lo = alpha
                phi_lo = phi
        return _failure(x, f, grad, evaluations, OptimizerStatus.LINE_SEARCH_FAILED)

    alpha_prev = 0.0
    phi_prev = f
    alpha = min(max(options.initial_step, options.step_min), options.step_max)
    while evaluations < options.max_steps:
        trial = evaluate(alpha)
        if trial is None:
            return _failure(x, f, grad, evaluations, OptimizerStatus.MAX_EVALUATIONS)
        trial_x, phi, trial_grad, slope = trial
        if not isfinite(phi) or not torch.isfinite(trial_grad).all():
            return zoom(alpha_prev, alpha, phi_prev)
        if phi > f + options.c1 * alpha * slope0 or evaluations > 1 and phi >= phi_prev:
            return zoom(alpha_prev, alpha, phi_prev)
        if abs(slope) <= -options.c2 * slope0:
            return _LineSearchResult(True, trial_x, phi, trial_grad, alpha, evaluations)
        if slope >= 0.0:
            return zoom(alpha, alpha_prev, phi)
        alpha_prev = alpha
        phi_prev = phi
        alpha = min(alpha * options.expansion, options.step_max)
        if alpha == alpha_prev:
            break
    return _failure(x, f, grad, evaluations, OptimizerStatus.LINE_SEARCH_FAILED)


def _line_search(
    evaluator: _ObjectiveEvaluator,
    x: Tensor,
    f: float,
    grad: Tensor,
    direction: Tensor,
    lower: Tensor | None,
    upper: Tensor | None,
    options: LineSearchOptions,
) -> _LineSearchResult:
    if lower is not None and upper is not None:
        return _projected_armijo(
            evaluator, x, f, grad, direction, lower, upper, options
        )
    if options.method == "armijo":
        infinite_lower = torch.full_like(x, -torch.inf)
        infinite_upper = torch.full_like(x, torch.inf)
        return _projected_armijo(
            evaluator,
            x,
            f,
            grad,
            direction,
            infinite_lower,
            infinite_upper,
            options,
        )
    return _strong_wolfe(evaluator, x, f, grad, direction, options)
