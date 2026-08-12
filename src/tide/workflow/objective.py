"""Receiver-space objectives composed from Maxwell derivative primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

from ..maxwell import (
    EMDirection,
    EMGradient,
    LinearizedMaxwell3D,
    LinearizedMaxwellTM,
)

ReceiverLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LinearizedMaxwell = LinearizedMaxwellTM | LinearizedMaxwell3D


def _least_squares(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    return 0.5 * (predicted - observed).square().sum()


def _add_gradients(left: EMGradient, right: EMGradient) -> EMGradient:
    def add(
        first: torch.Tensor | None,
        second: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if first is None:
            return second
        if second is None:
            return first
        return first + second

    return EMGradient(
        epsilon=add(left.epsilon, right.epsilon),
        sigma=add(left.sigma, right.sigma),
        mu=add(left.mu, right.mu),
    )


@dataclass(frozen=True, slots=True)
class ReceiverObjective:
    """A scalar receiver-data objective independent of Maxwell physics."""

    observed_data: torch.Tensor
    loss: ReceiverLoss = _least_squares

    def _data_derivatives(
        self,
        predicted: torch.Tensor,
        tangent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        data = predicted.detach().requires_grad_(True)
        value = self.loss(data, self.observed_data)
        if value.ndim != 0:
            raise ValueError("Receiver loss must return a scalar tensor.")
        data_gradient = torch.autograd.grad(
            value,
            data,
            create_graph=tangent is not None,
        )[0]
        if tangent is None:
            return data_gradient, None
        data_hvp = torch.autograd.grad(
            data_gradient,
            data,
            tangent,
        )[0]
        return data_gradient.detach(), data_hvp

    def value(self, linearized: LinearizedMaxwell) -> torch.Tensor:
        return self.loss(linearized.primal.receiver_data, self.observed_data)

    def gradient(self, linearized: LinearizedMaxwell) -> EMGradient:
        data_gradient, _ = self._data_derivatives(linearized.primal.receiver_data)
        return linearized.vjp(data_gradient)

    def hvp(
        self,
        linearized: LinearizedMaxwell,
        direction: EMDirection,
        *,
        mode: Literal["full", "gauss_newton"] = "full",
    ) -> EMGradient:
        if mode not in {"full", "gauss_newton"}:
            raise ValueError("mode must be 'full' or 'gauss_newton'.")
        tangent = linearized.jvp(direction)
        data_gradient, data_hvp = self._data_derivatives(
            linearized.primal.receiver_data,
            tangent.receiver_data,
        )
        assert data_hvp is not None
        gauss_newton = linearized.vjp(data_hvp)
        if mode == "gauss_newton":
            return gauss_newton
        correction = linearized.second_vjp(direction, data_gradient)
        return _add_gradients(gauss_newton, correction)


__all__ = ["ReceiverLoss", "ReceiverObjective"]
