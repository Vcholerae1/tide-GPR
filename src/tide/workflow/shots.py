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
    """Return contiguous shot-index batches as int64 tensors.

    The helper mirrors the shot-axis convention used by TIDE solvers:
    source amplitudes, source locations, and receiver locations are indexed by
    shot. It intentionally does not shuffle; callers that need randomized
    batches should pass their own index tensors to :func:`index_shots`.
    """

    n_shots = int(n_shots)
    batch_size = int(batch_size)
    if n_shots < 0:
        raise ValueError("n_shots must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if n_shots == 0:
        return []

    batch_size = min(batch_size, n_shots)
    indices = torch.arange(n_shots, device=device, dtype=torch.long)
    return [indices[start : start + batch_size] for start in range(0, n_shots, batch_size)]


def _normalize_dim(name: str, ndim: int, dim: int) -> int:
    dim = int(dim)
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"{name}={dim} is out of bounds for tensor with ndim={ndim}.")
    return dim


def _indices_for_tensor(
    shot_indices: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if shot_indices.ndim != 1:
        raise ValueError("shot_indices must be a 1D tensor.")
    return shot_indices.to(device=tensor.device, dtype=torch.long)


def index_shots(
    tensor: torch.Tensor | None,
    shot_indices: torch.Tensor,
    *,
    shot_dim: int = 0,
) -> torch.Tensor | None:
    """Select one shot mini-batch from ``tensor`` along ``shot_dim``.

    Use ``shot_dim=0`` for shared-shot tensors shaped ``[S, ...]`` and
    ``shot_dim=1`` for per-model-shot tensors shaped ``[B, S, ...]``.
    """

    if tensor is None:
        return None
    dim = _normalize_dim("shot_dim", tensor.ndim, shot_dim)
    indices = _indices_for_tensor(shot_indices, tensor)
    return torch.index_select(tensor, dim, indices)


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
        receiver_location=index_shots(receiver_location, shot_indices, shot_dim=shot_dim),
    )


def _default_receiver_selector(output: ModelOutput) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if not output:
        raise ValueError("model output sequence is empty.")
    receiver = output[-1]
    if not isinstance(receiver, torch.Tensor):
        raise TypeError("default output selector expected the last model output to be a tensor.")
    return receiver


def infer_receiver_shot_dim(receiver: torch.Tensor) -> int:
    """Infer the receiver shot axis for standard TIDE receiver tensors."""

    if receiver.ndim == 3:
        return 1
    if receiver.ndim == 4:
        return 2
    raise ValueError(
        "receiver chunks must be shaped [nt, S, R] or [nt, B, S, R] "
        f"when receiver_shot_dim is not provided, got {tuple(receiver.shape)}."
    )


def merge_receiver_batches(
    chunks: Iterable[torch.Tensor],
    *,
    shot_dim: int | None = None,
) -> torch.Tensor:
    """Concatenate receiver chunks along the shot axis.

    With TIDE receiver conventions, ``shot_dim`` is inferred as 1 for
    ``[nt, S, R]`` chunks and 2 for ``[nt, B, S, R]`` chunks.
    """

    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("chunks must contain at least one receiver tensor.")
    if shot_dim is None:
        shot_dim = infer_receiver_shot_dim(chunk_list[0])
    dim = _normalize_dim("shot_dim", chunk_list[0].ndim, shot_dim)
    if len(chunk_list) == 1:
        return chunk_list[0]
    return torch.cat(chunk_list, dim=dim)


def _infer_device(
    device: torch.device | str | None,
    tensors: Sequence[torch.Tensor | None],
) -> torch.device | str | None:
    if device is not None:
        return device
    for tensor in tensors:
        if tensor is not None:
            return tensor.device
    return None


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
    """Run ``solver`` over shot mini-batches and concatenate receivers.

    ``solver`` is called with selected ``source_amplitude``,
    ``source_location``, and ``receiver_location`` plus ``model_kwargs``. By
    default, the last item of a solver output tuple is treated as receiver data.
    """

    selector = _default_receiver_selector if receiver_selector is None else receiver_selector
    batch_device = _infer_device(
        device,
        (source_amplitude, source_location, receiver_location),
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
