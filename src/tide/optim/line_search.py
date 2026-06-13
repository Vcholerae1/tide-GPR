"""Line-search implementations for LBFGS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .array_utils import _evaluate_objective, _project, _set_trial_step
from .types import (
    Objective,
    OptimizerOptions,
    TASK_FAILURE,
    TASK_NEW_GRAD,
    TASK_NEW_STEP,
)


@dataclass(slots=True)
class _LineSearchState:
    first: bool = True
    task: int = TASK_NEW_GRAD
    cpt_ls: int = 0
    fk: float = 0.0
    alpha_l: float = 0.0
    alpha_r: float = 0.0
    alpha: float = 1.0
    q0: float = 0.0
    q: float = 0.0


@dataclass(slots=True)
class _LineSearchTrial:
    alpha: float
    f: float
    q: float


@dataclass(slots=True)
class _LineSearchResult:
    task: int
    f: float
    alpha: float
    line_search_iter: int
    q0: float
    q: float


def _weak_wolfe_line_search(
    objective: Objective,
    x: NDArray[np.float32],
    xk: NDArray[np.float32],
    descent: NDArray[np.float32],
    f0: float,
    grad: NDArray[np.float32],
    ls: _LineSearchState,
    options: OptimizerOptions,
    lower_bounds: NDArray[np.float32] | None,
    upper_bounds: NDArray[np.float32] | None,
    max_trials: int,
) -> _LineSearchResult:
    xk[:] = x
    grad0 = grad.copy()
    q0 = float(np.dot(grad, descent))
    ls.fk = f0
    ls.q0 = q0
    ls.q = q0
    ls.alpha_l = 0.0
    ls.alpha_r = 0.0
    ls.cpt_ls = 0
    if q0 >= 0.0 or not np.isfinite(q0) or max_trials <= 0:
        return _LineSearchResult(
            task=TASK_FAILURE,
            f=f0,
            alpha=ls.alpha,
            line_search_iter=0,
            q0=q0,
            q=q0,
        )

    trial_count = 0

    def evaluate(alpha: float) -> tuple[float, float]:
        nonlocal trial_count
        _set_trial_step(x, xk, descent, alpha)
        _project(x, lower_bounds, upper_bounds, options.bound_margin)
        f_trial = _evaluate_objective(objective, x, grad)
        trial_count += 1
        q_trial = float(np.dot(grad, descent))
        return float(f_trial), q_trial

    def accept(f_trial: float, q_trial: float) -> _LineSearchResult:
        ls.q = q_trial
        ls.first = True
        ls.task = TASK_NEW_STEP
        return _LineSearchResult(
            task=TASK_NEW_STEP,
            f=f_trial,
            alpha=ls.alpha,
            line_search_iter=trial_count,
            q0=q0,
            q=q_trial,
        )

    def fail() -> _LineSearchResult:
        ls.task = TASK_FAILURE
        _set_trial_step(x, xk, descent, 0.0)
        _project(x, lower_bounds, upper_bounds, options.bound_margin)
        grad[:] = grad0
        return _LineSearchResult(
            task=TASK_FAILURE,
            f=f0,
            alpha=ls.alpha,
            line_search_iter=trial_count,
            q0=q0,
            q=ls.q,
        )

    while trial_count < max_trials:
        f_trial, q_trial = evaluate(ls.alpha)
        if not np.isfinite(f_trial) or not np.isfinite(q_trial):
            return fail()

        if ls.cpt_ls >= options.max_line_search:
            if f_trial < f0:
                return accept(f_trial, q_trial)
            return fail()

        armijo_rhs = f0 + options.wolfe_c1 * ls.alpha * q0
        if f_trial <= armijo_rhs and q_trial >= options.wolfe_c2 * q0:
            return accept(f_trial, q_trial)
        if f_trial > armijo_rhs:
            ls.alpha_r = ls.alpha
            ls.alpha = 0.5 * (ls.alpha_l + ls.alpha_r)
        else:
            ls.alpha_l = ls.alpha
            if ls.alpha_r == 0.0:
                ls.alpha = options.growth * ls.alpha
            else:
                ls.alpha = 0.5 * (ls.alpha_l + ls.alpha_r)
        ls.q = q_trial
        ls.cpt_ls += 1
        if not np.isfinite(ls.alpha) or ls.alpha <= 0.0:
            return fail()

    return fail()


def _hager_zhang_satisfies_wolfe(
    trial: _LineSearchTrial,
    f0: float,
    q0: float,
    options: OptimizerOptions,
) -> bool:
    if trial.alpha <= 0.0:
        return False
    delta = options.hager_zhang_delta
    sigma = options.wolfe_c2
    phi_lim = f0 + options.hager_zhang_epsilon * abs(f0)
    wolfe = delta * q0 >= (trial.f - f0) / trial.alpha and trial.q >= sigma * q0
    approximate_wolfe = (
        (2.0 * delta - 1.0) * q0 >= trial.q >= sigma * q0
        and trial.f <= phi_lim
    )
    return bool(wolfe or approximate_wolfe)


def _hager_zhang_line_search(
    objective: Objective,
    x: NDArray[np.float32],
    xk: NDArray[np.float32],
    descent: NDArray[np.float32],
    f0: float,
    grad: NDArray[np.float32],
    ls: _LineSearchState,
    options: OptimizerOptions,
    lower_bounds: NDArray[np.float32] | None,
    upper_bounds: NDArray[np.float32] | None,
    max_trials: int,
) -> _LineSearchResult:
    xk[:] = x
    grad0 = grad.copy()
    q0 = float(np.dot(grad, descent))
    ls.q0 = q0
    if q0 >= 0.0 or not np.isfinite(q0) or max_trials <= 0:
        return _LineSearchResult(
            task=TASK_FAILURE,
            f=f0,
            alpha=ls.alpha,
            line_search_iter=0,
            q0=q0,
            q=q0,
        )

    trial_count = 0
    trials: list[_LineSearchTrial] = [_LineSearchTrial(0.0, f0, q0)]
    current_alpha = 0.0

    def evaluate(alpha: float) -> _LineSearchTrial:
        nonlocal trial_count, current_alpha
        _set_trial_step(x, xk, descent, alpha)
        _project(x, lower_bounds, upper_bounds, options.bound_margin)
        f = _evaluate_objective(objective, x, grad)
        q = float(np.dot(grad, descent))
        trial = _LineSearchTrial(float(alpha), float(f), q)
        trial_count += 1
        current_alpha = float(alpha)
        trials.append(trial)
        return trial

    def finite_trial(trial: _LineSearchTrial) -> bool:
        return bool(np.isfinite(trial.f) and np.isfinite(trial.q))

    def accept(trial: _LineSearchTrial) -> _LineSearchResult:
        nonlocal current_alpha
        if trial.alpha != current_alpha:
            trial = evaluate(trial.alpha)
        ls.alpha = trial.alpha
        ls.q = trial.q
        ls.cpt_ls = trial_count
        ls.first = True
        ls.task = TASK_NEW_STEP
        return _LineSearchResult(
            task=TASK_NEW_STEP,
            f=trial.f,
            alpha=trial.alpha,
            line_search_iter=trial_count,
            q0=q0,
            q=trial.q,
        )

    def fail() -> _LineSearchResult:
        ls.task = TASK_FAILURE
        ls.cpt_ls = trial_count
        _set_trial_step(x, xk, descent, 0.0)
        _project(x, lower_bounds, upper_bounds, options.bound_margin)
        grad[:] = grad0
        return _LineSearchResult(
            task=TASK_FAILURE,
            f=f0,
            alpha=ls.alpha,
            line_search_iter=trial_count,
            q0=q0,
            q=ls.q,
        )

    def best_decrease() -> _LineSearchTrial | None:
        finite = [trial for trial in trials[1:] if finite_trial(trial)]
        if not finite:
            return None
        best = min(finite, key=lambda trial: trial.f)
        return best if best.f < f0 else None

    def secant(a: _LineSearchTrial, b: _LineSearchTrial) -> float:
        denom = b.q - a.q
        if denom == 0.0 or not np.isfinite(denom):
            return 0.5 * (a.alpha + b.alpha)
        alpha = (a.alpha * b.q - b.alpha * a.q) / denom
        if not np.isfinite(alpha):
            return 0.5 * (a.alpha + b.alpha)
        return float(alpha)

    phi_lim = f0 + options.hager_zhang_epsilon * abs(f0)

    def update(
        a: _LineSearchTrial,
        b: _LineSearchTrial,
        c: _LineSearchTrial,
    ) -> tuple[_LineSearchTrial, _LineSearchTrial, _LineSearchTrial | None]:
        if _hager_zhang_satisfies_wolfe(c, f0, q0, options):
            return c, c, c
        if c.alpha < a.alpha or c.alpha > b.alpha:
            return a, b, None
        if c.q >= 0.0:
            return a, c, None
        if c.f <= phi_lim and np.isfinite(c.q):
            return c, b, None
        return bisect(a, c)

    def bisect(
        a: _LineSearchTrial,
        b: _LineSearchTrial,
    ) -> tuple[_LineSearchTrial, _LineSearchTrial, _LineSearchTrial | None]:
        while trial_count < max_trials and b.alpha - a.alpha > np.finfo(np.float32).eps:
            c = evaluate(0.5 * (a.alpha + b.alpha))
            if _hager_zhang_satisfies_wolfe(c, f0, q0, options):
                return c, c, c
            if c.q >= 0.0:
                return a, c, None
            if c.f <= phi_lim and np.isfinite(c.q):
                a = c
            else:
                b = c
        return a, b, None

    def secant2(
        a: _LineSearchTrial,
        b: _LineSearchTrial,
    ) -> tuple[_LineSearchTrial, _LineSearchTrial, _LineSearchTrial | None]:
        alpha = secant(a, b)
        if not (a.alpha < alpha < b.alpha):
            alpha = 0.5 * (a.alpha + b.alpha)
        c = evaluate(alpha)
        old_a = a
        old_b = b
        a_new, b_new, accepted = update(a, b, c)
        if accepted is not None or trial_count >= max_trials:
            return a_new, b_new, accepted

        second_alpha: float | None = None
        if b_new.alpha == c.alpha:
            second_alpha = secant(old_b, b_new)
        elif a_new.alpha == c.alpha:
            second_alpha = secant(old_a, a_new)
        if second_alpha is None or not (a_new.alpha < second_alpha < b_new.alpha):
            return a_new, b_new, None

        c2 = evaluate(second_alpha)
        return update(a_new, b_new, c2)

    alpha0 = float(ls.alpha)
    c = evaluate(alpha0)
    finite_checks = 0
    while (
        not finite_trial(c)
        and trial_count < max_trials
        and finite_checks < options.hager_zhang_max_finite_checks
    ):
        finite_checks += 1
        alpha0 *= options.hager_zhang_finite_shrink
        c = evaluate(alpha0)
    if not finite_trial(c):
        best = best_decrease()
        return accept(best) if best is not None else fail()
    if _hager_zhang_satisfies_wolfe(c, f0, q0, options):
        return accept(c)

    a = trials[0]
    b: _LineSearchTrial | None = None
    while trial_count < max_trials:
        if c.q >= 0.0:
            b = c
            break
        if c.f > phi_lim:
            a, b_candidate, accepted = bisect(a, c)
            if accepted is not None:
                return accept(accepted)
            b = b_candidate
            break
        a = c
        c = evaluate(options.hager_zhang_rho * c.alpha)
        if not finite_trial(c):
            a, b_candidate, accepted = bisect(a, c)
            if accepted is not None:
                return accept(accepted)
            b = b_candidate
            break
        if _hager_zhang_satisfies_wolfe(c, f0, q0, options):
            return accept(c)

    if b is None:
        best = best_decrease()
        return accept(best) if best is not None else fail()

    while trial_count < max_trials and b.alpha > a.alpha:
        width = b.alpha - a.alpha
        a_new, b_new, accepted = secant2(a, b)
        if accepted is not None:
            return accept(accepted)
        if b_new.alpha - a_new.alpha > options.hager_zhang_gamma * width:
            mid = evaluate(0.5 * (a_new.alpha + b_new.alpha))
            a_new, b_new, accepted = update(a_new, b_new, mid)
            if accepted is not None:
                return accept(accepted)
        a, b = a_new, b_new

    best = best_decrease()
    return accept(best) if best is not None else fail()


def _more_thuente_dcstep(
    stx: float,
    fx: float,
    dx: float,
    sty: float,
    fy: float,
    dy: float,
    stp: float,
    fp: float,
    dp: float,
    brackt: bool,
    stpmin: float,
    stpmax: float,
) -> tuple[float, float, float, float, float, float, float, bool]:
    if stp == stx or dx == 0.0 or stpmax < stpmin:
        return stx, fx, dx, sty, fy, dy, float(np.clip(stp, stpmin, stpmax)), brackt

    sgnd = dp * (dx / abs(dx))

    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = max(abs(theta), abs(dx), abs(dp))
        gamma = scale * float(
            np.sqrt(max(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale)))
        )
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + 0.5 * (
            dx / ((fx - fp) / (stp - stx) + dx)
        ) * (stp - stx)
        if abs(stpc - stx) <= abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + 0.5 * (stpq - stpc)
        brackt = True
    elif sgnd < 0.0:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = max(abs(theta), abs(dx), abs(dp))
        gamma = scale * float(
            np.sqrt(max(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale)))
        )
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = max(abs(theta), abs(dx), abs(dp))
        gamma = scale * float(
            np.sqrt(max(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale)))
        )
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0.0 and gamma != 0.0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = float(np.clip(stpf, stpmin, stpmax))
    else:
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            scale = max(abs(theta), abs(dy), abs(dp))
            gamma = scale * float(
                np.sqrt(max(0.0, (theta / scale) ** 2 - (dy / scale) * (dp / scale)))
            )
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpf = stp + r * (sty - stp)
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp

    return stx, fx, dx, sty, fy, dy, stpf, brackt


def _more_thuente_line_search(
    objective: Objective,
    x: NDArray[np.float32],
    xk: NDArray[np.float32],
    descent: NDArray[np.float32],
    f0: float,
    grad: NDArray[np.float32],
    ls: _LineSearchState,
    options: OptimizerOptions,
    lower_bounds: NDArray[np.float32] | None,
    upper_bounds: NDArray[np.float32] | None,
    max_trials: int,
) -> _LineSearchResult:
    xk[:] = x
    grad0 = grad.copy()
    q0 = float(np.dot(grad, descent))
    ls.q0 = q0
    if q0 >= 0.0 or not np.isfinite(q0) or max_trials <= 0:
        return _LineSearchResult(
            task=TASK_FAILURE,
            f=f0,
            alpha=ls.alpha,
            line_search_iter=0,
            q0=q0,
            q=q0,
        )

    trial_count = 0
    current_alpha = 0.0
    best_trial: _LineSearchTrial | None = None
    best_x = np.empty_like(x)
    best_grad = np.empty_like(grad)

    def finite_trial(trial: _LineSearchTrial) -> bool:
        return bool(np.isfinite(trial.f) and np.isfinite(trial.q))

    def evaluate(alpha: float) -> _LineSearchTrial:
        nonlocal best_trial, current_alpha, trial_count
        _set_trial_step(x, xk, descent, alpha)
        _project(x, lower_bounds, upper_bounds, options.bound_margin)
        f = _evaluate_objective(objective, x, grad)
        q = float(np.dot(grad, descent))
        trial = _LineSearchTrial(float(alpha), float(f), q)
        trial_count += 1
        current_alpha = float(alpha)
        if finite_trial(trial) and trial.f < f0:
            if best_trial is None or trial.f < best_trial.f:
                best_trial = trial
                best_x[:] = x
                best_grad[:] = grad
        return trial

    def accept(trial: _LineSearchTrial) -> _LineSearchResult:
        nonlocal current_alpha
        if trial.alpha != current_alpha:
            trial = evaluate(trial.alpha)
        ls.alpha = trial.alpha
        ls.q = trial.q
        ls.cpt_ls = trial_count
        ls.first = True
        ls.task = TASK_NEW_STEP
        return _LineSearchResult(
            task=TASK_NEW_STEP,
            f=trial.f,
            alpha=trial.alpha,
            line_search_iter=trial_count,
            q0=q0,
            q=trial.q,
        )

    def fail() -> _LineSearchResult:
        ls.task = TASK_FAILURE
        ls.cpt_ls = trial_count
        _set_trial_step(x, xk, descent, 0.0)
        _project(x, lower_bounds, upper_bounds, options.bound_margin)
        grad[:] = grad0
        return _LineSearchResult(
            task=TASK_FAILURE,
            f=f0,
            alpha=ls.alpha,
            line_search_iter=trial_count,
            q0=q0,
            q=ls.q,
        )

    def fallback() -> _LineSearchResult:
        nonlocal current_alpha
        if best_trial is None:
            return fail()
        x[:] = best_x
        grad[:] = best_grad
        current_alpha = best_trial.alpha
        return accept(best_trial)

    stpmin = options.more_thuente_step_min
    stpmax = options.more_thuente_step_max
    alpha = float(np.clip(ls.alpha, stpmin, stpmax))
    c1 = options.wolfe_c1
    c2 = options.wolfe_c2
    xtol = options.more_thuente_xtol

    brackt = False
    stage = 1
    finit = f0
    ginit = q0
    gtest = c1 * ginit
    width = stpmax - stpmin
    width1 = 2.0 * width
    stx = 0.0
    fx = finit
    gx = ginit
    sty = 0.0
    fy = finit
    gy = ginit
    stmin = 0.0
    stmax = alpha + 4.0 * alpha

    trial = evaluate(alpha)
    finite_checks = 0
    while (
        not finite_trial(trial)
        and trial_count < max_trials
        and finite_checks < options.hager_zhang_max_finite_checks
    ):
        finite_checks += 1
        alpha = max(stpmin, 0.5 * (alpha + stx))
        trial = evaluate(alpha)
    if not finite_trial(trial):
        return fallback()

    while True:
        f = trial.f
        q = trial.q
        ftest = finit + alpha * gtest

        if stage == 1 and f <= ftest and q >= 0.0:
            stage = 2

        if f <= ftest and abs(q) <= c2 * -ginit:
            return accept(trial)
        if brackt and (alpha <= stmin or alpha >= stmax):
            return fallback()
        if brackt and stmax - stmin <= xtol * stmax:
            return fallback()
        if alpha <= stpmin and (f > ftest or q >= gtest):
            return fallback()
        if alpha >= stpmax and f <= ftest and q <= gtest:
            return fallback()
        if trial_count >= max_trials:
            return fallback()

        try:
            if stage == 1 and f <= fx and f > ftest:
                fm = f - alpha * gtest
                fxm = fx - stx * gtest
                fym = fy - sty * gtest
                gm = q - gtest
                gxm = gx - gtest
                gym = gy - gtest
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    (
                        stx,
                        fxm,
                        gxm,
                        sty,
                        fym,
                        gym,
                        alpha,
                        brackt,
                    ) = _more_thuente_dcstep(
                        stx,
                        fxm,
                        gxm,
                        sty,
                        fym,
                        gym,
                        alpha,
                        fm,
                        gm,
                        brackt,
                        stmin,
                        stmax,
                    )
                fx = fxm + stx * gtest
                fy = fym + sty * gtest
                gx = gxm + gtest
                gy = gym + gtest
            else:
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    (
                        stx,
                        fx,
                        gx,
                        sty,
                        fy,
                        gy,
                        alpha,
                        brackt,
                    ) = _more_thuente_dcstep(
                        stx,
                        fx,
                        gx,
                        sty,
                        fy,
                        gy,
                        alpha,
                        f,
                        q,
                        brackt,
                        stmin,
                        stmax,
                    )
        except (FloatingPointError, ZeroDivisionError):
            return fallback()

        if not np.all(np.isfinite([stx, fx, gx, sty, fy, gy, alpha])):
            return fallback()

        if brackt:
            if abs(sty - stx) >= 0.66 * width1:
                alpha = stx + 0.5 * (sty - stx)
            width1 = width
            width = abs(sty - stx)

        if brackt:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = alpha + 1.1 * (alpha - stx)
            stmax = alpha + 4.0 * (alpha - stx)

        alpha = float(np.clip(alpha, stpmin, stpmax))
        if (
            brackt
            and (alpha <= stmin or alpha >= stmax)
            or (brackt and stmax - stmin <= xtol * stmax)
        ):
            alpha = stx

        if trial_count >= max_trials:
            return fallback()

        trial = evaluate(alpha)
        finite_checks = 0
        while (
            not finite_trial(trial)
            and trial_count < max_trials
            and finite_checks < options.hager_zhang_max_finite_checks
        ):
            finite_checks += 1
            alpha = max(stpmin, 0.5 * (alpha + stx))
            trial = evaluate(alpha)
        if not finite_trial(trial):
            return fallback()
