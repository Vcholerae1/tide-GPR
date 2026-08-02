"""Limited-memory BFGS history."""

from __future__ import annotations

from collections import deque

from torch import Tensor

from .array_utils import _dot
from .common import _EvaluationBudget, _apply_preconditioner
from .types import Preconditioner


class _LBFGSHistory:
    def __init__(self, size: int, curvature_tolerance: float) -> None:
        self.s: deque[Tensor] = deque(maxlen=size)
        self.y: deque[Tensor] = deque(maxlen=size)
        self.curvature_tolerance = curvature_tolerance

    def clear(self) -> None:
        self.s.clear()
        self.y.clear()

    def update(self, step: Tensor, grad_delta: Tensor) -> bool:
        sy = _dot(step, grad_delta)
        scale = float(step.norm().item()) * float(grad_delta.norm().item())
        if sy <= self.curvature_tolerance * max(1.0, scale):
            return False
        self.s.append(step.detach().clone())
        self.y.append(grad_delta.detach().clone())
        return True

    def direction(
        self,
        x: Tensor,
        grad: Tensor,
        preconditioner: Preconditioner | None,
        budget: _EvaluationBudget,
    ) -> Tensor:
        if not self.s:
            return -_apply_preconditioner(preconditioner, x, grad, budget)
        q = grad.clone()
        alphas: list[float] = []
        rhos: list[float] = []
        for s, y in zip(reversed(self.s), reversed(self.y), strict=True):
            rho = 1.0 / _dot(y, s)
            alpha = rho * _dot(s, q)
            q = q - alpha * y
            alphas.append(alpha)
            rhos.append(rho)
        r = _apply_preconditioner(preconditioner, x, q, budget)
        if preconditioner is None:
            last_s = self.s[-1]
            last_y = self.y[-1]
            r = (_dot(last_s, last_y) / _dot(last_y, last_y)) * r
        for s, y, alpha, rho in zip(
            self.s, self.y, reversed(alphas), reversed(rhos), strict=True
        ):
            beta = rho * _dot(y, r)
            r = r + (alpha - beta) * s
        return -r
