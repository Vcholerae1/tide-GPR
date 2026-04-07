from typing import Any

import torch

from ..dispersion import DebyeDispersion
from ..padding import create_or_pad
from ..storage import STORAGE_CPU, STORAGE_DISK, STORAGE_NONE

_CTX_HANDLE_REGISTRY: dict[int, dict[str, Any]] = {}
_CTX_HANDLE_COUNTER = iter(range(1 << 62))


def _register_ctx_handle(ctx_data: dict[str, Any]) -> torch.Tensor:
    handle = next(_CTX_HANDLE_COUNTER)
    _CTX_HANDLE_REGISTRY[handle] = ctx_data
    return torch.tensor(handle, dtype=torch.int64)


def _get_ctx_handle(handle: int) -> dict[str, Any]:
    try:
        return _CTX_HANDLE_REGISTRY[handle]
    except KeyError as exc:
        raise RuntimeError(f"Unknown context handle: {handle}") from exc


def _release_ctx_handle(handle: int | None) -> None:
    if handle is None:
        return
    _CTX_HANDLE_REGISTRY.pop(handle, None)


def _stream_handle(stream: Any | None) -> int:
    if stream is None:
        return 0
    return int(getattr(stream, "cuda_stream", 0) or 0)


def _make_storage_streams(
    device: torch.device, storage_mode: int
) -> tuple[int, int, tuple[Any, ...]]:
    if device.type != "cuda":
        return 0, 0, ()

    compute_stream = torch.cuda.current_stream(device=device)
    storage_stream = None
    if storage_mode in {STORAGE_CPU, STORAGE_DISK}:
        storage_stream = torch.cuda.Stream(device=device)

    handles = (_stream_handle(compute_stream), _stream_handle(storage_stream))
    keepalive = tuple(
        stream for stream in (compute_stream, storage_stream) if stream is not None
    )
    return handles[0], handles[1], keepalive


def _make_compute_stream(
    device: torch.device,
) -> tuple[int, tuple[Any, ...]]:
    compute_stream_handle, _, keepalive = _make_storage_streams(device, STORAGE_NONE)
    return compute_stream_handle, keepalive


def _copy_if_present(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.numel() > 0:
        dst.copy_(src)


def _init_polarization_state(
    *,
    n_shots: int,
    n_poles: int,
    spatial_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros((n_shots, n_poles, *spatial_shape), device=device, dtype=dtype)


def _debye_polarization_term(cp: torch.Tensor, polarization: torch.Tensor) -> torch.Tensor:
    cp_view = cp.unsqueeze(0)
    return (cp_view * polarization).sum(dim=1)


def _pad_dispersion_for_model(
    dispersion: DebyeDispersion | None,
    *,
    model_shape: tuple[int, ...],
    total_pad: list[int],
    padded_size: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> DebyeDispersion | None:
    if dispersion is None:
        return None

    def _pad_param(value: torch.Tensor | float) -> torch.Tensor | float:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        model_ndim = len(model_shape)
        if tensor.ndim == 0:
            return tensor
        if tensor.ndim <= model_ndim:
            expanded = torch.broadcast_to(tensor, model_shape)
            return create_or_pad(
                expanded, total_pad, device, dtype, padded_size, mode="replicate"
            )
        if tensor.ndim == model_ndim + 1:
            padded = [
                create_or_pad(
                    torch.broadcast_to(tensor[pole], model_shape),
                    total_pad,
                    device,
                    dtype,
                    padded_size,
                    mode="replicate",
                )
                for pole in range(tensor.shape[0])
            ]
            return torch.stack(padded, dim=0)
        raise ValueError(
            "Debye dispersion parameters must be scalar, model-shaped, or "
            f"[n_poles, *model_shape], but got shape {tuple(tensor.shape)}."
        )

    return DebyeDispersion(
        delta_epsilon=_pad_param(dispersion.delta_epsilon),
        tau=_pad_param(dispersion.tau),
    )


__all__ = [
    "_copy_if_present",
    "_debye_polarization_term",
    "_get_ctx_handle",
    "_init_polarization_state",
    "_make_compute_stream",
    "_make_storage_streams",
    "_pad_dispersion_for_model",
    "_register_ctx_handle",
    "_release_ctx_handle",
    "_stream_handle",
]
