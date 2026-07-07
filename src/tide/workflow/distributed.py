"""Small distributed helpers for shot-level data parallel workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .shots import infer_receiver_shot_dim


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Runtime metadata for one shot-parallel worker."""

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = torch.device("cpu")
    backend: str | None = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _resolve_device(
    requested_device: str | torch.device | None,
    *,
    local_rank: int,
) -> torch.device:
    requested = "auto" if requested_device is None else str(requested_device).lower()
    requested = requested.strip()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds available CUDA devices "
                f"({torch.cuda.device_count()})."
            )
        return torch.device("cuda", local_rank)
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false"
            )
        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested CUDA device {device} exceeds available CUDA devices "
                f"({torch.cuda.device_count()})."
            )
        return device
    raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")


def init_distributed(
    *,
    enabled: bool = False,
    requested_device: str | torch.device | None = None,
    backend: str | None = None,
) -> DistributedContext:
    """Initialize ``torch.distributed`` for one-rank-per-device shot sharding."""

    local_rank = _env_int("LOCAL_RANK", 0)
    device = _resolve_device(requested_device, local_rank=local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not enabled:
        return DistributedContext(device=device)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        initialized_backend = dist.get_backend()
        return DistributedContext(
            enabled=world_size > 1,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=initialized_backend,
        )

    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    if world_size <= 1:
        return DistributedContext(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=backend,
        )

    selected_backend = backend or ("nccl" if device.type == "cuda" else "gloo")
    dist.init_process_group(backend=selected_backend, rank=rank, world_size=world_size)
    return DistributedContext(
        enabled=True,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
        backend=selected_backend,
    )


def destroy_distributed(context: DistributedContext | None = None) -> None:
    """Tear down the default process group when this context owns one."""

    if (
        context is not None
        and context.enabled
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


def barrier(context: DistributedContext | None = None) -> None:
    """Synchronize ranks when distributed execution is active."""

    if context is not None and context.enabled:
        dist.barrier()


def rank_shot_indices(
    n_shots: int,
    *,
    context: DistributedContext | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the global shot indices owned by one rank."""

    n_shots = int(n_shots)
    if n_shots < 0:
        raise ValueError("n_shots must be non-negative.")
    if context is not None and context.enabled:
        actual_rank = context.rank
        actual_world_size = context.world_size
        actual_device = context.device if device is None else device
    else:
        actual_rank = 0 if rank is None else int(rank)
        actual_world_size = 1 if world_size is None else int(world_size)
        actual_device = (
            context.device if context is not None and device is None else device
        )
    if actual_world_size <= 0:
        raise ValueError("world_size must be positive.")
    if not 0 <= actual_rank < actual_world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size.")
    return torch.arange(n_shots, device=actual_device, dtype=torch.long)[
        actual_rank::actual_world_size
    ]


def split_rank_shots(
    n_shots: int,
    batch_size: int,
    *,
    context: DistributedContext | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """Return mini-batches for the shot shard owned by ``context.rank``."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indices = rank_shot_indices(
        n_shots,
        context=context,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    return list(indices.split(batch_size))


def local_shot_positions(
    global_shot_indices: torch.Tensor,
    local_shot_indices: torch.Tensor,
) -> torch.Tensor:
    """Map global shot ids in one local mini-batch to positions in a local shard."""

    if global_shot_indices.ndim != 1 or local_shot_indices.ndim != 1:
        raise ValueError("shot indices must be 1D tensors.")
    global_cpu = global_shot_indices.detach().cpu().to(dtype=torch.long)
    local_cpu = local_shot_indices.detach().cpu().to(dtype=torch.long)
    if local_cpu.numel() > 1 and bool((local_cpu[1:] < local_cpu[:-1]).any()):
        raise ValueError("local_shot_indices must be sorted in ascending order.")
    positions = torch.searchsorted(local_cpu, global_cpu)
    valid = positions < local_cpu.numel()
    if bool(valid.all()):
        valid = local_cpu[positions] == global_cpu
    if not bool(valid.all()):
        missing = global_cpu[~valid].tolist()
        raise ValueError(f"global shot indices are not in the local shard: {missing}")
    return positions.to(device=global_shot_indices.device, dtype=torch.long)


def all_reduce_tensor(
    tensor: torch.Tensor,
    *,
    context: DistributedContext | None = None,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """Return ``tensor`` reduced across all ranks, leaving the input unchanged."""

    if context is None or not context.enabled:
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=op)
    return reduced


def all_reduce_float(
    value: float,
    *,
    device: torch.device | str,
    context: DistributedContext | None = None,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> float:
    """Reduce a Python scalar across ranks and return it as ``float``."""

    if context is None or not context.enabled:
        return float(value)
    scalar = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(scalar, op=op)
    return float(scalar.item())


def all_reduce_gradients(
    tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    context: DistributedContext | None = None,
) -> None:
    """Sum parameter gradients across ranks, materializing zero grads as needed."""

    if context is None or not context.enabled:
        return
    for tensor in tensors:
        if not tensor.requires_grad:
            continue
        if tensor.grad is None:
            tensor.grad = torch.zeros_like(tensor)
        dist.all_reduce(tensor.grad, op=dist.ReduceOp.SUM)


def _gather_object(
    payload: Any,
    *,
    context: DistributedContext,
    dst: int,
) -> list[Any] | None:
    if hasattr(dist, "gather_object"):
        objects = (
            [None for _ in range(context.world_size)] if context.rank == dst else None
        )
        dist.gather_object(payload, object_gather_list=objects, dst=dst)
        return objects

    objects = [None for _ in range(context.world_size)]
    dist.all_gather_object(objects, payload)
    return objects if context.rank == dst else None


def gather_receiver_shards(
    local_receiver: torch.Tensor,
    local_shot_indices: torch.Tensor,
    n_shots: int,
    *,
    context: DistributedContext | None = None,
    dst: int = 0,
) -> torch.Tensor | None:
    """Gather receiver shot shards and assemble the full receiver tensor on ``dst``."""

    if local_shot_indices.ndim != 1:
        raise ValueError("local_shot_indices must be a 1D tensor.")
    if context is None or not context.enabled:
        return local_receiver

    shot_dim = infer_receiver_shot_dim(local_receiver)
    payload = (
        local_receiver.detach().cpu(),
        local_shot_indices.detach().cpu().to(dtype=torch.long),
    )
    objects = _gather_object(payload, context=context, dst=dst)
    if context.rank != dst:
        return None
    if objects is None:
        raise RuntimeError("receiver gather did not return objects on destination rank")

    shape = list(local_receiver.shape)
    shape[shot_dim] = int(n_shots)
    full = torch.empty(shape, dtype=local_receiver.dtype, device="cpu")
    filled = torch.zeros(int(n_shots), dtype=torch.bool)
    for shard, indices in objects:
        if indices.numel() == 0:
            continue
        if shard.ndim != local_receiver.ndim:
            raise ValueError("all receiver shards must have the same rank.")
        full.index_copy_(shot_dim, indices, shard)
        filled[indices] = True
    if not bool(filled.all()):
        missing = torch.nonzero(~filled, as_tuple=False).reshape(-1).tolist()
        raise ValueError(f"missing gathered receiver shots: {missing}")
    return full


__all__ = [
    "DistributedContext",
    "all_reduce_float",
    "all_reduce_gradients",
    "all_reduce_tensor",
    "barrier",
    "destroy_distributed",
    "gather_receiver_shards",
    "init_distributed",
    "local_shot_positions",
    "rank_shot_indices",
    "split_rank_shots",
]
