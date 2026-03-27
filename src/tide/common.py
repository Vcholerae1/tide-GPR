"""Common Python helpers shared by propagator frontends."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import Any

import torch

from .padding import create_or_pad, zero_interior
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


def make_receiver_amplitudes(
    *,
    nt: int,
    n_shots: int,
    n_receivers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate receiver traces for a forward propagation call."""
    if n_receivers > 0:
        return torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
    return torch.empty(0, device=device, dtype=dtype)


def make_native_runtime_context(
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    reference: torch.Tensor,
    nt: int,
    n_shots: int,
    n_receivers: int,
    receiver_dtype: torch.dtype | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build shared native runtime metadata for forward/backward helpers."""
    dtype = reference.dtype if receiver_dtype is None else receiver_dtype
    context = {
        "device": reference.device,
        "dtype": dtype,
        "nt": nt,
        "n_shots": n_shots,
        "n_receivers": n_receivers,
        "receiver_amplitudes": make_receiver_amplitudes(
            nt=nt,
            n_shots=n_shots,
            n_receivers=n_receivers,
            device=reference.device,
            dtype=dtype,
        ),
        "ca_requires_grad": bool(ca.requires_grad),
        "cb_requires_grad": bool(cb.requires_grad),
        "device_idx": (
            reference.device.index
            if reference.device.type == "cuda" and reference.device.index is not None
            else 0
        ),
    }
    if extra is not None:
        context.update(extra)
    return context


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


def prepare_initial_wavefields(
    wavefields: list[torch.Tensor | None],
    *,
    n_shots: int,
    size_with_batch: tuple[int, ...],
    fd_pad_list: list[int],
    device: torch.device,
    dtype: torch.dtype,
    mode: str = "constant",
    contiguous: bool = False,
    zero_interior_plan: list[tuple[int, int, list[int]]] | None = None,
) -> list[torch.Tensor]:
    """Pad or create wavefields, optionally zeroing selected interiors."""
    prepared: list[torch.Tensor] = []
    spatial_ndim = len(size_with_batch) - 1

    for field_0 in wavefields:
        if field_0 is not None:
            if field_0.ndim == spatial_ndim:
                field_0 = field_0[None, ...].expand(n_shots, *([-1] * spatial_ndim))
            wavefield = create_or_pad(
                field_0,
                fd_pad_list,
                device,
                dtype,
                size_with_batch,
                mode=mode,
            )
        else:
            wavefield = torch.zeros(size_with_batch, device=device, dtype=dtype)
        prepared.append(wavefield.contiguous() if contiguous else wavefield)

    if zero_interior_plan is not None:
        for field_idx, dim, pml_width in zero_interior_plan:
            zero_interior(prepared[field_idx], fd_pad_list, pml_width, dim)

    return prepared


def save_autograd_context(
    ctx: Any,
    *,
    tensors: tuple[torch.Tensor, ...],
    forward_tensors: tuple[torch.Tensor, ...] | None = None,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Store tensors and plain attributes on an autograd context."""
    ctx.save_for_backward(*tensors)
    if forward_tensors is not None:
        ctx.save_for_forward(*forward_tensors)
    if attrs is not None:
        for key, value in attrs.items():
            setattr(ctx, key, value)


def save_storage_context(
    ctx: Any,
    *,
    tensors: tuple[torch.Tensor, ...] = (),
    forward_tensors: tuple[torch.Tensor, ...] | None = None,
    attrs: dict[str, Any] | None = None,
    backward_storage_objects: Any = (),
    backward_storage_filename_arrays: Any = (),
) -> None:
    """Store autograd metadata and keep storage resources alive."""
    save_autograd_context(
        ctx,
        tensors=tensors,
        forward_tensors=forward_tensors,
        attrs=attrs,
    )
    ctx.backward_storage_objects = backward_storage_objects
    ctx.backward_storage_filename_arrays = backward_storage_filename_arrays


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
