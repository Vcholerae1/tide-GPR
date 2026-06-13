"""Shared helpers for CPU-state optimizers."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_utils import _as_float32_vector, _project
from .line_search import (
    _LineSearchState,
    _hager_zhang_line_search,
    _more_thuente_line_search,
    _weak_wolfe_line_search,
)
from .types import (
    FLAG_CONV,
    FLAG_FAIL,
    Objective,
    OptimizerOptions,
    OptimizerResult,
    STATUS_NONFINITE,
)


def _converged(f: float, f0: float, iteration: int, options: OptimizerOptions) -> bool:
    if iteration >= options.max_iter:
        return True
    if f0 == 0.0:
        return abs(f) <= options.tolerance
    return f / f0 < options.tolerance


def _line_search_max_trials(
    options: OptimizerOptions,
    n_eval: int,
) -> int | None:
    if options.max_evaluations is None:
        remaining = None
    else:
        remaining = options.max_evaluations - n_eval
        if remaining <= 0:
            return 0
    max_trials = (
        options.max_line_search + 1
        if options.line_search == "weak_wolfe"
        else options.max_line_search
    )
    if remaining is not None:
        max_trials = min(max_trials, remaining)
    return max_trials


def _line_search_step(
    objective: Objective,
    options: OptimizerOptions,
    x: NDArray[np.float32],
    xk: NDArray[np.float32],
    descent: NDArray[np.float32],
    f: float,
    grad: NDArray[np.float32],
    ls: _LineSearchState,
    lower_bounds: NDArray[np.float32] | None,
    upper_bounds: NDArray[np.float32] | None,
    max_trials: int,
):
    if options.line_search == "weak_wolfe":
        return _weak_wolfe_line_search(
            objective,
            x,
            xk,
            descent,
            f,
            grad,
            ls,
            options,
            lower_bounds,
            upper_bounds,
            max_trials,
        )
    if options.line_search == "hager_zhang":
        return _hager_zhang_line_search(
            objective,
            x,
            xk,
            descent,
            f,
            grad,
            ls,
            options,
            lower_bounds,
            upper_bounds,
            max_trials,
        )
    if options.line_search == "more_thuente":
        return _more_thuente_line_search(
            objective,
            x,
            xk,
            descent,
            f,
            grad,
            ls,
            options,
            lower_bounds,
            upper_bounds,
            max_trials,
        )
    raise AssertionError(f"Unexpected line search: {options.line_search}")


def _prepare_initial_state(
    x0: ArrayLike,
    lower_bounds: ArrayLike | None,
    upper_bounds: ArrayLike | None,
    options: OptimizerOptions,
) -> tuple[float, NDArray[np.float32], NDArray[np.float32] | None, NDArray[np.float32] | None]:
    start = perf_counter()
    x = _as_float32_vector("x0", x0)
    n = int(x.size)
    lb: NDArray[np.float32] | None = None
    ub: NDArray[np.float32] | None = None
    if lower_bounds is not None or upper_bounds is not None:
        if lower_bounds is None or upper_bounds is None:
            raise ValueError("lower_bounds and upper_bounds must be provided together.")
        lb = _as_float32_vector("lower_bounds", lower_bounds, n)
        ub = _as_float32_vector("upper_bounds", upper_bounds, n)
        if np.any(lb > ub):
            raise ValueError("lower_bounds must be <= upper_bounds.")
        _project(x, lb, ub, options.bound_margin)
    return start, x, lb, ub


def _nonfinite_result(
    x: NDArray[np.float32],
    f: float,
    grad: NDArray[np.float32],
    *,
    n_eval: int,
    n_prec: int,
    n_hess: int,
    start: float,
) -> OptimizerResult:
    return OptimizerResult(
        x=x.copy(),
        f=f,
        grad=grad.copy(),
        status=STATUS_NONFINITE,
        flag=FLAG_FAIL,
        success=False,
        n_iter=0,
        n_eval=n_eval,
        n_prec=n_prec,
        n_hess=n_hess,
        elapsed_s=perf_counter() - start,
        trace=[],
    )


def _success_from_flag(flag: int) -> bool:
    return flag == FLAG_CONV

