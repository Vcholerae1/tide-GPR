"""Receiver-data loss helpers for shot-batched workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import torch
import torch.nn.functional as F

from .shots import infer_receiver_shot_dim, index_shots

LossNormalization = Literal["batch", "all", "sum"]
_ShotLossFn = Callable[[torch.Tensor], torch.Tensor]
_ZeroGradFn = Callable[[], None]
_AfterBackwardFn = Callable[[torch.Tensor, torch.Tensor], None]


def take_receiver_batch(
    receiver: torch.Tensor,
    shot_indices: torch.Tensor,
    *,
    shot_dim: int | None = None,
) -> torch.Tensor:
    """Select receiver data for one shot mini-batch."""

    dim = infer_receiver_shot_dim(receiver) if shot_dim is None else shot_dim
    return index_shots(receiver, shot_indices, shot_dim=dim)


def receiver_mse_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    shot_indices: torch.Tensor,
    *,
    shot_dim: int | None = None,
    normalization: LossNormalization = "batch",
) -> torch.Tensor:
    """Return MSE between predicted receivers and the matching observed shots."""

    target = take_receiver_batch(observed, shot_indices, shot_dim=shot_dim)
    if normalization == "batch":
        return F.mse_loss(predicted, target)
    residual = predicted - target
    if normalization == "all":
        return residual.square().sum() / observed.numel()
    if normalization == "sum":
        return residual.square().sum()
    raise ValueError("normalization must be 'batch', 'all', or 'sum'.")


def backward_shot_batches(
    loss_fn: _ShotLossFn,
    shot_batches: Iterable[torch.Tensor],
    *,
    zero_grad: _ZeroGradFn | None = None,
    zero_each_batch: bool = False,
    after_backward: _AfterBackwardFn | None = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> float:
    """Backpropagate scalar losses over shot mini-batches.

    By default, ``zero_grad`` is called once before the loop so gradients
    accumulate across all batches. Set ``zero_each_batch=True`` when the caller
    wants to inspect per-batch gradients in ``after_backward``.
    """

    if zero_grad is not None and not zero_each_batch:
        zero_grad()

    total_loss = 0.0
    n_batches = 0
    for shot_indices in shot_batches:
        if zero_grad is not None and zero_each_batch:
            zero_grad()

        loss = loss_fn(shot_indices)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss_fn must return a torch.Tensor.")
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar loss tensor.")

        loss.backward(retain_graph=retain_graph, create_graph=create_graph)
        if after_backward is not None:
            after_backward(shot_indices, loss)

        total_loss += float(loss.detach())
        n_batches += 1

    if n_batches == 0:
        raise ValueError("shot_batches must contain at least one batch.")
    return total_loss


__all__ = [
    "LossNormalization",
    "backward_shot_batches",
    "receiver_mse_loss",
    "take_receiver_batch",
]
