"""Shot-batched modeling helpers for TIDE workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

ModelOutput = torch.Tensor | Sequence[torch.Tensor]
ReceiverSelector = Callable[[ModelOutput], torch.Tensor]


@dataclass(frozen=True, slots=True)
class ShotBatch:
    """Source and receiver tensors selected for one shot mini-batch."""

    source_amplitude: torch.Tensor | None
    source_location: torch.Tensor | None
    receiver_location: torch.Tensor | None


def split_shots(
    n_shots: int,
    batch_size: int,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """Return contiguous, non-shuffled shot-index batches."""

    n_shots = int(n_shots)
    batch_size = int(batch_size)
    if n_shots < 0:
        raise ValueError("n_shots must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indices = torch.arange(n_shots, device=device, dtype=torch.long)
    return list(indices.split(batch_size))


def _normalize_dim(ndim: int, dim: int) -> int:
    dim = int(dim)
    if not -ndim <= dim < ndim:
        raise ValueError(
            f"shot_dim={dim} is out of bounds for tensor with ndim={ndim}."
        )
    return dim % ndim


def index_shots(
    tensor: torch.Tensor | None,
    shot_indices: torch.Tensor,
    *,
    shot_dim: int = 0,
) -> torch.Tensor | None:
    """Select one shot mini-batch from ``tensor`` along ``shot_dim``."""

    if tensor is None:
        return None
    if shot_indices.ndim != 1:
        raise ValueError("shot_indices must be a 1D tensor.")
    indices = shot_indices.to(device=tensor.device, dtype=torch.long)
    return torch.index_select(tensor, _normalize_dim(tensor.ndim, shot_dim), indices)


def take_shot_batch(
    *,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    shot_indices: torch.Tensor,
    shot_dim: int = 0,
) -> ShotBatch:
    """Select source and receiver tensors for one shot mini-batch."""

    return ShotBatch(
        source_amplitude=index_shots(source_amplitude, shot_indices, shot_dim=shot_dim),
        source_location=index_shots(source_location, shot_indices, shot_dim=shot_dim),
        receiver_location=index_shots(
            receiver_location, shot_indices, shot_dim=shot_dim
        ),
    )


def _default_receiver_selector(output: ModelOutput) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if not output:
        raise ValueError("model output sequence is empty.")
    receiver = output[-1]
    if not isinstance(receiver, torch.Tensor):
        raise TypeError(
            "default output selector expected the last model output to be a tensor."
        )
    return receiver


def infer_receiver_shot_dim(receiver: torch.Tensor) -> int:
    """Infer the receiver shot axis for standard TIDE receiver tensors."""

    if receiver.ndim not in (3, 4):
        raise ValueError(
            "receiver chunks must be shaped [nt, S, R] or [nt, B, S, R] "
            f"when receiver_shot_dim is not provided, got {tuple(receiver.shape)}."
        )
    return receiver.ndim - 2


def merge_receiver_batches(
    chunks: Iterable[torch.Tensor],
    *,
    shot_dim: int | None = None,
) -> torch.Tensor:
    """Concatenate receiver chunks along their inferred shot axis."""

    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("chunks must contain at least one receiver tensor.")
    dim = infer_receiver_shot_dim(chunk_list[0]) if shot_dim is None else shot_dim
    dim = _normalize_dim(chunk_list[0].ndim, dim)
    if len(chunk_list) == 1:
        return chunk_list[0]
    return torch.cat(chunk_list, dim=dim)


def run_shot_batches(
    solver: Callable[..., ModelOutput],
    *,
    n_shots: int,
    batch_size: int,
    source_amplitude: torch.Tensor | None = None,
    source_location: torch.Tensor | None = None,
    receiver_location: torch.Tensor | None = None,
    shot_dim: int = 0,
    receiver_shot_dim: int | None = None,
    receiver_selector: ReceiverSelector | None = None,
    device: torch.device | str | None = None,
    **model_kwargs: Any,
) -> torch.Tensor:
    """Run ``solver`` over shot mini-batches and concatenate receivers."""

    selector = (
        _default_receiver_selector if receiver_selector is None else receiver_selector
    )
    tensors = (source_amplitude, source_location, receiver_location)
    batch_device = (
        device
        if device is not None
        else next((tensor.device for tensor in tensors if tensor is not None), None)
    )
    chunks: list[torch.Tensor] = []
    for shot_indices in split_shots(n_shots, batch_size, batch_device):
        batch = take_shot_batch(
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            shot_indices=shot_indices,
            shot_dim=shot_dim,
        )
        output = solver(
            source_amplitude=batch.source_amplitude,
            source_location=batch.source_location,
            receiver_location=batch.receiver_location,
            **model_kwargs,
        )
        chunks.append(selector(output))
    return merge_receiver_batches(chunks, shot_dim=receiver_shot_dim)


__all__ = [
    "ShotBatch",
    "infer_receiver_shot_dim",
    "index_shots",
    "merge_receiver_batches",
    "run_shot_batches",
    "split_shots",
    "take_shot_batch",
]
