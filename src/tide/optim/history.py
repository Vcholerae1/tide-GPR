"""L-BFGS history and two-loop recursion helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _save_lbfgs(
    x: NDArray[np.float32],
    grad: NDArray[np.float32],
    sk: NDArray[np.float32],
    yk: NDArray[np.float32],
    cpt_lbfgs: int,
) -> None:
    history_size = sk.shape[0]
    if cpt_lbfgs <= history_size:
        idx = cpt_lbfgs - 1
        sk[idx] = x
        yk[idx] = grad
    else:
        sk[:-1] = sk[1:]
        yk[:-1] = yk[1:]
        sk[-1] = x
        yk[-1] = grad


def _update_lbfgs(
    x: NDArray[np.float32],
    grad: NDArray[np.float32],
    sk: NDArray[np.float32],
    yk: NDArray[np.float32],
    cpt_lbfgs: int,
) -> int:
    history_size = sk.shape[0]
    idx = cpt_lbfgs - 1 if cpt_lbfgs <= history_size else history_size - 1
    sk_candidate = x - sk[idx]
    yk_candidate = grad - yk[idx]
    sy = float(np.dot(sk_candidate, yk_candidate))
    yy = float(np.dot(yk_candidate, yk_candidate))
    # The L-BFGS update assumes positive curvature; projected no-op steps can break that.
    if sy <= 0.0 or yy <= 0.0 or not np.isfinite(sy) or not np.isfinite(yy):
        return 1
    sk[idx] = sk_candidate
    yk[idx] = yk_candidate
    if cpt_lbfgs <= history_size:
        return cpt_lbfgs + 1
    return cpt_lbfgs


def _history_bounds(cpt_lbfgs: int, history_size: int) -> tuple[int, int]:
    count = min(max(cpt_lbfgs - 1, 0), history_size)
    return 0, count


def _descent1_lbfgs(
    grad: NDArray[np.float32],
    sk: NDArray[np.float32],
    yk: NDArray[np.float32],
    cpt_lbfgs: int,
    q_lbfgs: NDArray[np.float32],
    alpha_lbfgs: NDArray[np.float32],
    rho_lbfgs: NDArray[np.float32],
    scratch: NDArray[np.float32],
) -> int:
    history_size = sk.shape[0]
    _, borne = _history_bounds(cpt_lbfgs, history_size)
    q_lbfgs[:] = grad
    for offset in range(borne - 1, -1, -1):
        idx = offset
        sy = float(np.dot(yk[idx], sk[idx]))
        rho = 1.0 / sy
        rho_lbfgs[idx] = np.float32(rho)
        alpha = rho * float(np.dot(sk[idx], q_lbfgs))
        alpha_lbfgs[idx] = alpha
        np.multiply(yk[idx], np.float32(alpha), out=scratch)
        q_lbfgs -= scratch
    return borne


def _descent2_lbfgs(
    sk: NDArray[np.float32],
    yk: NDArray[np.float32],
    cpt_lbfgs: int,
    q_lbfgs: NDArray[np.float32],
    descent: NDArray[np.float32],
    alpha_lbfgs: NDArray[np.float32],
    rho_lbfgs: NDArray[np.float32],
    gamma_lbfgs: NDArray[np.float32],
    scratch: NDArray[np.float32],
) -> None:
    history_size = sk.shape[0]
    _, borne = _history_bounds(cpt_lbfgs, history_size)
    if borne == 0:
        descent[:] = -q_lbfgs
        return
    last = borne - 1
    sy = float(np.dot(sk[last], yk[last]))
    yy = float(np.dot(yk[last], yk[last]))
    gamma = sy / yy
    gamma_lbfgs[last] = np.float32(gamma)
    descent[:] = np.float32(gamma) * q_lbfgs
    for offset in range(borne):
        idx = offset
        beta = rho_lbfgs[idx] * float(np.dot(yk[idx], descent))
        np.multiply(sk[idx], np.float32(alpha_lbfgs[idx] - beta), out=scratch)
        descent += scratch
    descent[:] = -descent
