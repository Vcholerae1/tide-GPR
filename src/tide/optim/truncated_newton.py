"""Truncated Newton optimizer."""

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
    HessianVectorProduct,
    Objective,
    OptimizerResult,
    OptimizerTraceEntry,
    Preconditioner,
    STATUS_CONVERGED,
    STATUS_INNER_CG_FAILED,
    STATUS_LINE_SEARCH_FAILED,
    STATUS_MAX_EVALUATIONS,
    STATUS_MAX_ITER,
    STATUS_NONFINITE,
    TASK_NEW_STEP,
    TruncatedNewtonOptions,
    _make_trace,
)


def _forcing_term(
    eta: float,
    grad: NDArray[np.float32],
    residual: NDArray[np.float32],
    previous_grad_norm: float,
) -> float:
    if previous_grad_norm <= 0.0:
        return 0.9
    next_eta = float(np.linalg.norm(grad - residual)) / previous_grad_norm
    eta_power = eta ** ((1.0 + np.sqrt(5.0)) / 2.0)
    if eta_power > 0.1:
        next_eta = max(next_eta, eta_power)
    if next_eta > 1.0:
        next_eta = 0.9
    return float(next_eta)


def _newton_descent(
    hessian_vector: HessianVectorProduct,
    preconditioner: Preconditioner | None,
    x: NDArray[np.float32],
    grad: NDArray[np.float32],
    grad_preco: NDArray[np.float32],
    residual: NDArray[np.float32],
    residual_preco: NDArray[np.float32],
    descent: NDArray[np.float32],
    descent_prev: NDArray[np.float32],
    d: NDArray[np.float32],
    hd: NDArray[np.float32],
    options: TruncatedNewtonOptions,
    eta: float,
    grad_norm: float,
) -> tuple[int, int, int]:
    residual[:] = grad
    descent.fill(0.0)
    hd.fill(0.0)
    n_prec = 0
    n_hess = 0
    if _apply_preconditioner(preconditioner, x, grad, grad_preco):
        n_prec += 1
    residual_preco[:] = grad_preco
    d[:] = -residual_preco

    residual_norm = float(np.linalg.norm(residual))
    res_dot_preco = float(np.dot(residual, residual_preco))
    if not np.isfinite(residual_norm) or not np.isfinite(res_dot_preco):
        return 0, n_prec, n_hess

    cg_iter = 0
    while cg_iter < options.max_cg_iter:
        hessian_vector(x, d, hd)
        n_hess += 1
        d_hd = float(np.dot(d, hd))
        if not np.isfinite(d_hd):
            break
        if d_hd < 0.0:
            if cg_iter == 0:
                descent[:] = d
            break
        if d_hd == 0.0:
            break

        if preconditioner is None:
            alpha = (residual_norm * residual_norm) / d_hd
            descent_prev[:] = descent
            descent += np.float32(alpha) * d
            residual += np.float32(alpha) * hd
            previous_norm = residual_norm
            residual_norm = float(np.linalg.norm(residual))
            cg_iter += 1
            if (
                residual_norm <= eta * grad_norm
                or cg_iter >= options.max_cg_iter
            ):
                break
            beta = (residual_norm * residual_norm) / (previous_norm * previous_norm)
            d[:] = -residual + np.float32(beta) * d
        else:
            alpha = res_dot_preco / d_hd
            descent_prev[:] = descent
            descent += np.float32(alpha) * d
            residual += np.float32(alpha) * hd
            if _apply_preconditioner(preconditioner, x, residual, residual_preco):
                n_prec += 1
            previous_dot = res_dot_preco
            res_dot_preco = float(np.dot(residual, residual_preco))
            residual_norm = float(np.linalg.norm(residual))
            cg_iter += 1
            if (
                residual_norm <= eta * grad_norm
                or cg_iter >= options.max_cg_iter
            ):
                break
            beta = res_dot_preco / previous_dot if previous_dot != 0.0 else 0.0
            d[:] = -residual_preco + np.float32(beta) * d

    return cg_iter, n_prec, n_hess


def truncated_newton_minimize(
    objective: Objective,
    hessian_vector: HessianVectorProduct,
    x0: ArrayLike,
    *,
    preconditioner: Preconditioner | None = None,
    options: TruncatedNewtonOptions | None = None,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    callback: Callback | None = None,
) -> OptimizerResult:
    """Minimize with truncated Newton, optionally using a preconditioner."""

    if options is None:
        options = TruncatedNewtonOptions()
    start, x, lb, ub = _prepare_initial_state(
        x0, lower_bounds, upper_bounds, options
    )
    grad = np.empty_like(x)
    grad_preco = np.empty_like(x)
    residual = np.empty_like(x)
    residual_preco = np.empty_like(x)
    descent = np.empty_like(x)
    descent_prev = np.empty_like(x)
    d = np.empty_like(x)
    hd = np.empty_like(x)
    xk = np.empty_like(x)
    ls = _LineSearchState(alpha=options.initial_step)
    trace: list[OptimizerTraceEntry] = []
    emit_trace = options.record_trace or callback is not None

    f = _evaluate_objective(objective, x, grad)
    n_eval = 1
    n_prec = 0
    n_hess = 0
    if not np.isfinite(f) or not np.all(np.isfinite(grad)):
        return _nonfinite_result(
            x, f, grad, n_eval=n_eval, n_prec=n_prec, n_hess=n_hess, start=start
        )

    f0 = f
    eta = options.eta_initial
    grad_norm = float(np.linalg.norm(grad))
    previous_grad_norm = grad_norm
    n_iter = 0
    flag = FLAG_NSTE
    status = "running"

    while flag not in (FLAG_CONV, FLAG_FAIL):
        cg_iter, cg_prec, cg_hess = _newton_descent(
            hessian_vector,
            preconditioner,
            x,
            grad,
            grad_preco,
            residual,
            residual_preco,
            descent,
            descent_prev,
            d,
            hd,
            options,
            eta,
            grad_norm,
        )
        n_prec += cg_prec
        n_hess += cg_hess
        if cg_iter == 0 and float(np.dot(grad, descent)) >= 0.0:
            descent[:] = -grad_preco
        if float(np.dot(grad, descent)) >= 0.0 or not np.all(np.isfinite(descent)):
            flag = FLAG_FAIL
            status = STATUS_INNER_CG_FAILED
            break

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
        previous_grad_norm = grad_norm
        grad_norm = float(np.linalg.norm(grad))
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
                metadata={
                    "method": "truncated_newton",
                    "line_search": options.line_search,
                    "cg_iter": cg_iter,
                    "eta": eta,
                },
            )
            if options.record_trace:
                trace.append(entry)
            if callback is not None:
                callback(entry)

        if _converged(f, f0, n_iter, options):
            flag = FLAG_CONV
            status = STATUS_MAX_ITER if n_iter >= options.max_iter else STATUS_CONVERGED
            break
        eta = _forcing_term(eta, grad, residual, previous_grad_norm)
        flag = FLAG_NSTE

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
        n_hess=n_hess,
        elapsed_s=perf_counter() - start,
        trace=trace,
    )


__all__ = ["truncated_newton_minimize"]
