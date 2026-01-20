import itertools
from typing import Any, Optional

import torch

_CTX_HANDLE_COUNTER = itertools.count()
_CTX_HANDLE_REGISTRY: dict[int, dict[str, Any]] = {}


def _register_ctx_handle(ctx_data: dict[str, Any]) -> torch.Tensor:
    handle = next(_CTX_HANDLE_COUNTER)
    _CTX_HANDLE_REGISTRY[handle] = ctx_data
    return torch.tensor(handle, dtype=torch.int64)


def _get_ctx_handle(handle: int) -> dict[str, Any]:
    try:
        return _CTX_HANDLE_REGISTRY[handle]
    except KeyError as exc:
        raise RuntimeError(f"Unknown context handle: {handle}") from exc


def _release_ctx_handle(handle: Optional[int]) -> None:
    if handle is None:
        return
    _CTX_HANDLE_REGISTRY.pop(handle, None)
