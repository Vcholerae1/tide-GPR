"""Common Python helpers shared by propagator frontends."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Any

import torch

from .storage import STORAGE_NONE, StorageMode

_CTX_HANDLE_COUNTER = itertools.count()
_CTX_HANDLE_REGISTRY: dict[int, dict[str, Any]] = {}


def register_ctx_handle(ctx_data: dict[str, Any]) -> torch.Tensor:
    """Register Python-side context data and return a tensor handle."""
    handle = next(_CTX_HANDLE_COUNTER)
    _CTX_HANDLE_REGISTRY[handle] = ctx_data
    return torch.tensor(handle, dtype=torch.int64)


def get_ctx_handle(handle: int) -> dict[str, Any]:
    """Resolve a previously registered context handle."""
    try:
        return _CTX_HANDLE_REGISTRY[handle]
    except KeyError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Unknown context handle: {handle}") from exc


def release_ctx_handle(handle: int | None) -> None:
    """Release a context handle if it exists."""
    if handle is None:
        return
    _CTX_HANDLE_REGISTRY.pop(handle, None)


def stream_handle(stream: Any | None) -> int:
    """Return the raw CUDA stream handle or 0 for host execution."""
    if stream is None:
        return 0
    return int(getattr(stream, "cuda_stream", 0) or 0)


def make_storage_streams(
    device: torch.device,
    storage_mode: StorageMode,
) -> tuple[int, int, tuple[Any, ...]]:
    """Create compute/storage streams for native CUDA kernels."""
    if device.type != "cuda":
        return 0, 0, ()

    compute_stream = torch.cuda.current_stream(device=device)
    storage_stream = None
    if storage_mode in {StorageMode.CPU, StorageMode.DISK}:
        storage_stream = torch.cuda.Stream(device=device)

    handles = (stream_handle(compute_stream), stream_handle(storage_stream))
    keepalive = tuple(
        stream for stream in (compute_stream, storage_stream) if stream is not None
    )
    return handles[0], handles[1], keepalive


def make_compute_stream(device: torch.device) -> tuple[int, tuple[Any, ...]]:
    """Return only the compute stream handle/keepalive tuple."""
    compute_stream_handle, _, keepalive = make_storage_streams(
        device, STORAGE_NONE
    )
    return compute_stream_handle, keepalive


def copy_if_present(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy data when the destination owns storage."""
    if dst.numel() > 0:
        dst.copy_(src)


def ensure_contiguous(*args: Any) -> Any:
    """Recursively make tensors, lists, or tuples contiguous."""

    def _make_contiguous(item: Any) -> Any:
        if item is None:
            return None
        if isinstance(item, torch.Tensor):
            return item.contiguous()
        if isinstance(item, list):
            return [_make_contiguous(i) for i in item]
        if isinstance(item, tuple):
            return tuple(_make_contiguous(i) for i in item)
        if isinstance(item, OrderedDict):
            return OrderedDict((k, _make_contiguous(v)) for k, v in item.items())
        return item

    out = [_make_contiguous(arg) for arg in args]
    return out[0] if len(out) == 1 else tuple(out)


def execute_stepping_loop(
    *,
    total_steps: int,
    chunk_size: int,
    run_chunk: Any,
    reverse: bool = False,
) -> None:
    """Run a chunked stepping loop shared by native forward/backward paths."""
    if total_steps <= 0:
        return

    if chunk_size <= 0:
        chunk_size = total_steps

    if reverse:
        step = total_steps
        while step > 0:
            step_count = min(step, chunk_size)
            run_chunk(step, step_count)
            step -= step_count
        return

    step = 0
    while step < total_steps:
        step_count = min(total_steps - step, chunk_size)
        run_chunk(step, step_count)
        step += step_count
