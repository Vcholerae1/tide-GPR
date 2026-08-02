"""Torch-native CGNR least-squares solver."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from time import perf_counter

import torch
from torch import Tensor

from .array_utils import _as_model_tensor, _dot, _validate_operator_output
from .types import (
    CGNROptions,
    CGNRResult,
    CGNRTraceEntry,
    LinearOperator,
    OptimizerStatus,
    Preconditioner,
)


def cgnr_solve(
    forward: LinearOperator,
    adjoint: LinearOperator,
    b: Tensor,
    x0: Tensor,
    *,
    preconditioner: Preconditioner | None = None,
    options: CGNROptions | None = None,
    callback: Callable[[CGNRTraceEntry], None] | None = None,
) -> CGNRResult:
    """Solve ``min_x 0.5 ||A x - b||²`` with CGNR or PCGNR."""

    resolved = options or CGNROptions()
    start = perf_counter()
    x = _as_model_tensor("x0", x0)
    data = _as_model_tensor("b", b)
    if x.device != data.device or x.dtype != data.dtype:
        raise ValueError("x0 and b must have the same device and dtype.")
    n_forward = 0
    n_adjoint = 0
    n_prec = 0
    trace: list[CGNRTraceEntry] = []

    def make_result(
        residual: Tensor,
        normal_residual: Tensor,
        status: OptimizerStatus,
        n_iter: int,
    ) -> CGNRResult:
        return CGNRResult(
            x=x.detach().clone(),
            f=0.5 * _dot(residual, residual),
            residual=residual.detach().clone(),
            normal_residual=normal_residual.detach().clone(),
            status=status,
            success=status.success,
            n_iter=n_iter,
            n_forward=n_forward,
            n_adjoint=n_adjoint,
            n_prec=n_prec,
            elapsed_s=perf_counter() - start,
            trace=trace,
        )

    ax = _validate_operator_output("forward", forward(x.detach()), data)
    n_forward += 1
    residual = data - ax
    normal_residual = _validate_operator_output(
        "adjoint", adjoint(residual.detach()), x
    )
    n_adjoint += 1
    if not all(torch.isfinite(value).all() for value in (x, residual, normal_residual)):
        return make_result(residual, normal_residual, OptimizerStatus.NONFINITE, 0)

    normal0 = float(torch.linalg.vector_norm(normal_residual).item())
    tolerance = resolved.atol + resolved.rtol * normal0
    if normal0 <= tolerance:
        return make_result(
            residual, normal_residual, OptimizerStatus.CONVERGED_GRADIENT, 0
        )
    if resolved.max_iter == 0:
        return make_result(residual, normal_residual, OptimizerStatus.MAX_ITERATIONS, 0)

    if preconditioner is None:
        z = normal_residual.clone()
    else:
        z = _validate_operator_output(
            "preconditioner",
            preconditioner(x.detach(), normal_residual.detach()),
            normal_residual,
        )
        n_prec += 1
    rz = _dot(normal_residual, z)
    if not torch.isfinite(z).all() or rz <= 0.0:
        return make_result(
            residual,
            normal_residual,
            OptimizerStatus.INVALID_PRECONDITIONER,
            0,
        )
    direction = z.clone()

    for iteration in range(1, resolved.max_iter + 1):
        used = n_forward + n_adjoint
        if resolved.max_matvec is not None and used + 2 > resolved.max_matvec:
            return make_result(
                residual,
                normal_residual,
                OptimizerStatus.MAX_EVALUATIONS,
                iteration - 1,
            )
        ad = _validate_operator_output("forward", forward(direction.detach()), data)
        n_forward += 1
        denominator = _dot(ad, ad)
        if denominator <= 0.0 or not isfinite(denominator):
            return make_result(
                residual, normal_residual, OptimizerStatus.BREAKDOWN, iteration - 1
            )
        alpha = rz / denominator
        x = x + alpha * direction
        residual = residual - alpha * ad
        normal_residual = _validate_operator_output(
            "adjoint", adjoint(residual.detach()), x
        )
        n_adjoint += 1
        if not all(
            torch.isfinite(value).all() for value in (x, residual, normal_residual)
        ):
            return make_result(
                residual, normal_residual, OptimizerStatus.NONFINITE, iteration
            )
        normal_norm = float(torch.linalg.vector_norm(normal_residual).item())
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        if normal_norm <= tolerance:
            beta = 0.0
            status: OptimizerStatus | None = OptimizerStatus.CONVERGED_GRADIENT
        else:
            if preconditioner is None:
                z_new = normal_residual.clone()
            else:
                z_new = _validate_operator_output(
                    "preconditioner",
                    preconditioner(x.detach(), normal_residual.detach()),
                    normal_residual,
                )
                n_prec += 1
            rz_new = _dot(normal_residual, z_new)
            if not torch.isfinite(z_new).all() or rz_new <= 0.0:
                return make_result(
                    residual,
                    normal_residual,
                    OptimizerStatus.INVALID_PRECONDITIONER,
                    iteration,
                )
            beta = rz_new / rz
            direction = z_new + beta * direction
            z = z_new
            rz = rz_new
            status = None

        snapshot = resolved.trace.store_tensors and (
            iteration % resolved.trace.snapshot_interval == 0
        )
        entry = CGNRTraceEntry(
            iteration=iteration,
            f=0.5 * residual_norm**2,
            residual_norm=residual_norm,
            normal_residual_norm=normal_norm,
            alpha=alpha,
            beta=beta,
            metadata={"preconditioned": preconditioner is not None},
            x=x.detach().clone() if snapshot else None,
            residual=residual.detach().clone() if snapshot else None,
            normal_residual=normal_residual.detach().clone() if snapshot else None,
        )
        if snapshot and resolved.trace.snapshot_device == "cpu":
            entry.x = entry.x.cpu() if entry.x is not None else None
            entry.residual = (
                entry.residual.cpu() if entry.residual is not None else None
            )
            entry.normal_residual = (
                entry.normal_residual.cpu()
                if entry.normal_residual is not None
                else None
            )
        if resolved.trace.record:
            trace.append(entry)
        if callback is not None:
            callback(entry)
        if status is not None:
            return make_result(residual, normal_residual, status, iteration)

    return make_result(
        residual, normal_residual, OptimizerStatus.MAX_ITERATIONS, resolved.max_iter
    )


__all__ = ["cgnr_solve"]
