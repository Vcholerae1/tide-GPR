"""Pure-Python LBFGS optimizer prototype."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_utils import (
    _apply_preconditioner,
    _as_float32_vector,
    _evaluate_objective,
    _project,
)
from .history import (
    _descent1_lbfgs,
    _descent2_lbfgs,
    _save_lbfgs,
    _update_lbfgs,
)
from .line_search import (
    _LineSearchState,
    _hager_zhang_line_search,
    _more_thuente_line_search,
    _weak_wolfe_line_search,
)
from .types import (
    Callback,
    FLAG_CONV,
    FLAG_FAIL,
    FLAG_NSTE,
    Objective,
    LBFGSOptions,
    OptimizerResult,
    OptimizerTraceEntry,
    Preconditioner,
    STATUS_CONVERGED,
    STATUS_LINE_SEARCH_FAILED,
    STATUS_MAX_EVALUATIONS,
    STATUS_MAX_ITER,
    STATUS_NONFINITE,
    TASK_NEW_STEP,
    _make_trace,
)


def _converged(f: float, f0: float, iteration: int, options: LBFGSOptions) -> bool:
    if iteration >= options.max_iter:
        return True
    if f0 == 0.0:
        return abs(f) <= options.tolerance
    return f / f0 < options.tolerance


def _line_search_step(
    objective: Objective,
    options: LBFGSOptions,
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
    raise AssertionError(f"Unexpected direct line search: {options.line_search}")


def _lbfgs_minimize_direct_line_search(
    objective: Objective,
    preconditioner: Preconditioner | None,
    options: LBFGSOptions,
    x: NDArray[np.float32],
    xk: NDArray[np.float32],
    descent: NDArray[np.float32],
    q_lbfgs: NDArray[np.float32],
    grad: NDArray[np.float32],
    grad_preco: NDArray[np.float32],
    q_preco: NDArray[np.float32],
    sk: NDArray[np.float32],
    yk: NDArray[np.float32],
    alpha_lbfgs: NDArray[np.float32],
    rho_lbfgs: NDArray[np.float32],
    gamma_lbfgs: NDArray[np.float32],
    f: float,
    n_eval: int,
    n_prec: int,
    lower_bounds: NDArray[np.float32] | None,
    upper_bounds: NDArray[np.float32] | None,
    callback: Callback | None,
    start: float,
) -> OptimizerResult:
    f0 = f
    cpt_lbfgs = 1
    n_iter = 0
    trace: list[OptimizerTraceEntry] = []
    emit_trace = options.record_trace or callback is not None
    ls = _LineSearchState(alpha=options.initial_step)

    descent[:] = -grad_preco
    _save_lbfgs(x, grad, sk, yk, cpt_lbfgs)

    status = "running"
    flag = FLAG_NSTE
    while flag not in (FLAG_CONV, FLAG_FAIL):
        if float(np.dot(grad, descent)) >= 0.0:
            cpt_lbfgs = 1
            sk.fill(0.0)
            yk.fill(0.0)
            _save_lbfgs(x, grad, sk, yk, cpt_lbfgs)
            if _apply_preconditioner(preconditioner, x, grad, grad_preco):
                n_prec += 1
            descent[:] = -grad_preco

        remaining_evals = (
            None
            if options.max_evaluations is None
            else options.max_evaluations - n_eval
        )
        if remaining_evals is not None and remaining_evals <= 0:
            flag = FLAG_FAIL
            status = STATUS_MAX_EVALUATIONS
            break
        max_trials = (
            options.max_line_search + 1
            if options.line_search == "weak_wolfe"
            else options.max_line_search
        )
        if remaining_evals is not None:
            max_trials = min(max_trials, remaining_evals)

        result = _line_search_step(
            objective,
            options,
            x,
            xk,
            descent,
            f,
            grad,
            ls,
            lower_bounds,
            upper_bounds,
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

        cpt_lbfgs = _update_lbfgs(x, grad, sk, yk, cpt_lbfgs)
        _descent1_lbfgs(
            grad, sk, yk, cpt_lbfgs, q_lbfgs, alpha_lbfgs, rho_lbfgs, q_preco
        )
        if _apply_preconditioner(preconditioner, x, q_lbfgs, q_preco):
            n_prec += 1
        q_lbfgs[:] = q_preco
        _descent2_lbfgs(
            sk,
            yk,
            cpt_lbfgs,
            q_lbfgs,
            descent,
            alpha_lbfgs,
            rho_lbfgs,
            gamma_lbfgs,
            q_preco,
        )
        _save_lbfgs(x, grad, sk, yk, cpt_lbfgs)
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
                metadata={"line_search": options.line_search},
            )
            if options.record_trace:
                trace.append(entry)
            if callback is not None:
                callback(entry)

        if _converged(f, f0, n_iter, options):
            flag = FLAG_CONV
            status = STATUS_MAX_ITER if n_iter >= options.max_iter else STATUS_CONVERGED
        else:
            flag = FLAG_NSTE

    success = flag == FLAG_CONV
    return OptimizerResult(
        x=x.copy(),
        f=f,
        grad=grad.copy(),
        status=status,
        flag=flag,
        success=success,
        n_iter=n_iter,
        n_eval=n_eval,
        n_prec=n_prec,
        n_hess=0,
        elapsed_s=perf_counter() - start,
        trace=trace,
    )


def lbfgs_minimize(
    objective: Objective,
    x0: ArrayLike,
    *,
    preconditioner: Preconditioner | None = None,
    options: LBFGSOptions | None = None,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize an objective with CPU-state L-BFGS.

    ``objective`` writes the gradient into ``grad`` and returns the objective
    value. If provided, ``preconditioner`` writes ``P(x) vector`` into ``out``;
    otherwise the method is standard L-BFGS. The model vector, L-BFGS history,
    line-search state, and workspaces are stored as CPU ``numpy.float32``
    arrays.
    """

    if options is None:
        options = LBFGSOptions()

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

    xk = np.empty_like(x)
    descent = np.empty_like(x)
    q_lbfgs = np.empty_like(x)
    grad = np.empty_like(x)
    grad_preco = np.empty_like(x)
    q_preco = np.empty_like(x)
    sk = np.zeros((options.history_size, n), dtype=np.float32)
    yk = np.zeros((options.history_size, n), dtype=np.float32)
    alpha_lbfgs = np.zeros(options.history_size, dtype=np.float32)
    rho_lbfgs = np.zeros(options.history_size, dtype=np.float32)
    gamma_lbfgs = np.ones(options.history_size, dtype=np.float32)

    f = _evaluate_objective(objective, x, grad)
    n_eval = 1
    n_prec = 0
    if _apply_preconditioner(preconditioner, x, grad, grad_preco):
        n_prec += 1
    if not np.isfinite(f) or not np.all(np.isfinite(grad)):
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
            n_hess=0,
            elapsed_s=perf_counter() - start,
            trace=[],
        )

    return _lbfgs_minimize_direct_line_search(
        objective,
        preconditioner,
        options,
        x,
        xk,
        descent,
        q_lbfgs,
        grad,
        grad_preco,
        q_preco,
        sk,
        yk,
        alpha_lbfgs,
        rho_lbfgs,
        gamma_lbfgs,
        f,
        n_eval,
        n_prec,
        lb,
        ub,
        callback,
        start,
    )


__all__ = ["lbfgs_minimize"]
