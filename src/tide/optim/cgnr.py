"""CGNR least-squares solver."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .array_utils import _apply_preconditioner, _as_float32_vector
from .types import (
    CGNROptions,
    CGNRResult,
    CGNRTraceEntry,
    LinearOperator,
    Preconditioner,
    STATUS_CONVERGED,
    STATUS_MAX_EVALUATIONS,
    STATUS_MAX_ITER,
    STATUS_NONFINITE,
)


STATUS_BREAKDOWN = "breakdown"


def _dot(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a.astype(np.float64, copy=False), b.astype(np.float64, copy=False)))


def _make_result(
    *,
    x: NDArray[np.float32],
    residual: NDArray[np.float32],
    normal_residual: NDArray[np.float32],
    status: str,
    success: bool,
    n_iter: int,
    n_forward: int,
    n_adjoint: int,
    n_prec: int,
    start: float,
    trace: list[CGNRTraceEntry],
) -> CGNRResult:
    f = 0.5 * _dot(residual, residual)
    return CGNRResult(
        x=x.copy(),
        f=f,
        residual=residual.copy(),
        normal_residual=normal_residual.copy(),
        status=status,
        success=success,
        n_iter=n_iter,
        n_forward=n_forward,
        n_adjoint=n_adjoint,
        n_prec=n_prec,
        elapsed_s=perf_counter() - start,
        trace=trace,
    )


def cgnr_solve(
    forward: LinearOperator,
    adjoint: LinearOperator,
    b: ArrayLike,
    x0: ArrayLike,
    *,
    preconditioner: Preconditioner | None = None,
    options: CGNROptions | None = None,
    callback: Callable[[CGNRTraceEntry], None] | None = None,
) -> CGNRResult:
    """Solve ``min_x 0.5 * ||A x - b||^2`` with CGNR or PCGNR.

    ``forward`` writes ``A x`` into ``out``. ``adjoint`` writes ``A^T r`` into
    ``out``. If ``preconditioner`` is provided, it is applied to the normal
    residual, giving the PCGNR variant used in the SMIwiz reference.
    """

    if options is None:
        options = CGNROptions()
    start = perf_counter()
    x = _as_float32_vector("x0", x0)
    data = _as_float32_vector("b", b)
    residual = np.empty_like(data)
    normal_residual = np.empty_like(x)
    z = np.empty_like(x)
    p = np.empty_like(x)
    ap = np.empty_like(data)
    trace: list[CGNRTraceEntry] = []
    emit_trace = options.record_trace or callback is not None

    forward(x, ap)
    n_forward = 1
    np.subtract(data, ap, out=residual)
    adjoint(residual, normal_residual)
    n_adjoint = 1
    n_prec = 0
    if _apply_preconditioner(preconditioner, x, normal_residual, z):
        n_prec += 1

    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(normal_residual))
        or not np.all(np.isfinite(z))
    ):
        return _make_result(
            x=x,
            residual=residual,
            normal_residual=normal_residual,
            status=STATUS_NONFINITE,
            success=False,
            n_iter=0,
            n_forward=n_forward,
            n_adjoint=n_adjoint,
            n_prec=n_prec,
            start=start,
            trace=[],
        )

    p[:] = z
    zrt_old = _dot(z, normal_residual)
    residual0 = _dot(residual, residual)
    if residual0 == 0.0 or zrt_old <= 0.0:
        return _make_result(
            x=x,
            residual=residual,
            normal_residual=normal_residual,
            status=STATUS_CONVERGED,
            success=True,
            n_iter=0,
            n_forward=n_forward,
            n_adjoint=n_adjoint,
            n_prec=n_prec,
            start=start,
            trace=trace,
        )

    status = STATUS_MAX_ITER
    success = False
    n_iter = 0
    for iteration in range(1, options.max_iter + 1):
        if options.max_matvec is not None and (
            n_forward + n_adjoint >= options.max_matvec
        ):
            status = STATUS_MAX_EVALUATIONS
            break

        forward(p, ap)
        n_forward += 1
        ws = _dot(ap, ap)
        if ws <= 0.0 or not np.isfinite(ws):
            status = STATUS_BREAKDOWN
            break
        alpha = zrt_old / ws
        x += np.float32(alpha) * p
        residual -= np.float32(alpha) * ap

        adjoint(residual, normal_residual)
        n_adjoint += 1
        if _apply_preconditioner(preconditioner, x, normal_residual, z):
            n_prec += 1
        if (
            not np.all(np.isfinite(x))
            or not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(normal_residual))
            or not np.all(np.isfinite(z))
        ):
            status = STATUS_NONFINITE
            break

        zrt_new = _dot(z, normal_residual)
        beta = zrt_new / zrt_old if zrt_old != 0.0 else 0.0
        if not np.isfinite(beta):
            status = STATUS_BREAKDOWN
            break
        p *= np.float32(beta)
        p += z
        zrt_old = zrt_new
        n_iter = iteration

        residual_norm_sq = _dot(residual, residual)
        if emit_trace:
            entry = CGNRTraceEntry(
                iteration=iteration,
                f=0.5 * residual_norm_sq,
                residual_norm=float(np.sqrt(residual_norm_sq)),
                normal_residual_norm=float(
                    np.linalg.norm(normal_residual.astype(np.float64, copy=False))
                ),
                alpha=float(alpha),
                beta=float(beta),
                x=x.copy(),
                residual=residual.copy(),
                normal_residual=normal_residual.copy(),
                metadata={"preconditioned": preconditioner is not None},
            )
            if options.record_trace:
                trace.append(entry)
            if callback is not None:
                callback(entry)

        if residual_norm_sq / residual0 < options.tolerance:
            status = STATUS_CONVERGED
            success = True
            break

    return _make_result(
        x=x,
        residual=residual,
        normal_residual=normal_residual,
        status=status,
        success=success,
        n_iter=n_iter,
        n_forward=n_forward,
        n_adjoint=n_adjoint,
        n_prec=n_prec,
        start=start,
        trace=trace,
    )


__all__ = ["cgnr_solve"]
