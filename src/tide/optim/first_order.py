"""Steepest descent and nonlinear conjugate-gradient optimizers."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_utils import _apply_preconditioner, _evaluate_objective
from .common import (
    _converged,
    _line_search_max_trials,
    _line_search_step,
    _nonfinite_result,
    _prepare_initial_state,
)
from .line_search import _LineSearchState
from .types import (
    Callback,
    FLAG_CONV,
    FLAG_FAIL,
    FLAG_NSTE,
    NLCGOptions,
    Objective,
    OptimizerOptions,
    OptimizerResult,
    OptimizerTraceEntry,
    Preconditioner,
    STATUS_CONVERGED,
    STATUS_LINE_SEARCH_FAILED,
    STATUS_MAX_EVALUATIONS,
    STATUS_MAX_ITER,
    STATUS_NONFINITE,
    SteepestDescentOptions,
    TASK_NEW_STEP,
    _make_trace,
)


def _make_descent_result(
    *,
    x: NDArray[np.float32],
    f: float,
    grad: NDArray[np.float32],
    status: str,
    flag: int,
    n_iter: int,
    n_eval: int,
    n_prec: int,
    trace: list[OptimizerTraceEntry],
    start: float,
) -> OptimizerResult:
    return OptimizerResult(
        x=x.copy(),
        f=f,
        grad=grad.copy(),
        status=status,
        flag=flag,
        success=flag == FLAG_CONV,
        n_iter=n_iter,
        n_eval=n_eval,
        n_prec=n_prec,
        n_hess=0,
        elapsed_s=perf_counter() - start,
        trace=trace,
    )


def _minimize_first_order(
    objective: Objective,
    x0: ArrayLike,
    *,
    preconditioner: Preconditioner | None,
    options: OptimizerOptions,
    lower_bounds: ArrayLike | None,
    upper_bounds: ArrayLike | None,
    callback: Callback | None,
    method: str,
    beta_abs_max: float | None,
) -> OptimizerResult:
    start, x, lb, ub = _prepare_initial_state(
        x0, lower_bounds, upper_bounds, options
    )
    grad = np.empty_like(x)
    grad_preco = np.empty_like(x)
    descent = np.empty_like(x)
    descent_prev = np.empty_like(x)
    grad_prev = np.empty_like(x)
    xk = np.empty_like(x)
    ls = _LineSearchState(alpha=options.initial_step)
    trace: list[OptimizerTraceEntry] = []
    emit_trace = options.record_trace or callback is not None

    f = _evaluate_objective(objective, x, grad)
    n_eval = 1
    n_prec = 0
    if _apply_preconditioner(preconditioner, x, grad, grad_preco):
        n_prec += 1
    if not np.isfinite(f) or not np.all(np.isfinite(grad)):
        return _nonfinite_result(
            x, f, grad, n_eval=n_eval, n_prec=n_prec, n_hess=0, start=start
        )

    f0 = f
    n_iter = 0
    status = "running"
    flag = FLAG_NSTE
    descent[:] = -grad_preco
    grad_prev[:] = grad

    while flag not in (FLAG_CONV, FLAG_FAIL):
        if float(np.dot(grad, descent)) >= 0.0:
            descent[:] = -grad_preco

        max_trials = _line_search_max_trials(options, n_eval)
        if max_trials is not None and max_trials <= 0:
            flag = FLAG_FAIL
            status = STATUS_MAX_EVALUATIONS
            break

        result = _line_search_step(
            objective,
            options,
            x,
            xk,
            descent,
            f,
            grad,
            ls,
            lb,
            ub,
            max_trials,
        )
        n_eval += result.line_search_iter
        if result.task != TASK_NEW_STEP:
            flag = FLAG_FAIL
            status = STATUS_LINE_SEARCH_FAILED
            break

        f = result.f
        if not np.isfinite(f) or not np.all(np.isfinite(grad)):
            flag = FLAG_FAIL
            status = STATUS_NONFINITE
            break

        n_iter += 1
        if emit_trace:
            entry = _make_trace(
                flag=FLAG_NSTE,
                iteration=n_iter,
                evaluations=n_eval,
                f=f,
                alpha=result.alpha,
                line_search_iter=result.line_search_iter,
                accepted=True,
                task=result.task,
                x=x,
                grad=grad,
                q0=result.q0,
                q=result.q,
                metadata={"method": method, "line_search": options.line_search},
            )
            if options.record_trace:
                trace.append(entry)
            if callback is not None:
                callback(entry)

        if _converged(f, f0, n_iter, options):
            flag = FLAG_CONV
            status = STATUS_MAX_ITER if n_iter >= options.max_iter else STATUS_CONVERGED
            break

        descent_prev[:] = descent
        if _apply_preconditioner(preconditioner, x, grad, grad_preco):
            n_prec += 1
        if beta_abs_max is None:
            descent[:] = -grad_preco
        else:
            y = grad - grad_prev
            denom = float(np.dot(y, descent_prev))
            numer = float(np.dot(grad, grad_preco))
            beta = numer / denom if denom != 0.0 else 0.0
            if not np.isfinite(beta) or abs(beta) >= beta_abs_max:
                beta = 0.0
            descent[:] = -grad_preco + np.float32(beta) * descent_prev
            if float(np.dot(grad, descent)) >= 0.0:
                beta = 0.0
                descent[:] = -grad_preco
            grad_prev[:] = grad
        flag = FLAG_NSTE

    return _make_descent_result(
        x=x,
        f=f,
        grad=grad,
        status=status,
        flag=flag,
        n_iter=n_iter,
        n_eval=n_eval,
        n_prec=n_prec,
        trace=trace,
        start=start,
    )


def steepest_descent_minimize(
    objective: Objective,
    x0: ArrayLike,
    *,
    preconditioner: Preconditioner | None = None,
    options: SteepestDescentOptions | None = None,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize with steepest descent, optionally using a preconditioner."""

    if options is None:
        options = SteepestDescentOptions()
    return _minimize_first_order(
        objective,
        x0,
        preconditioner=preconditioner,
        options=options,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        callback=callback,
        method="steepest_descent",
        beta_abs_max=None,
    )


def nlcg_minimize(
    objective: Objective,
    x0: ArrayLike,
    *,
    preconditioner: Preconditioner | None = None,
    options: NLCGOptions | None = None,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize with nonlinear conjugate gradient, optionally preconditioned."""

    if options is None:
        options = NLCGOptions()
    return _minimize_first_order(
        objective,
        x0,
        preconditioner=preconditioner,
        options=options,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        callback=callback,
        method="nlcg",
        beta_abs_max=options.beta_abs_max,
    )


__all__ = ["nlcg_minimize", "steepest_descent_minimize"]
