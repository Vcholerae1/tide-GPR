from typing import Any

import torch

from ..dispersion import DebyeDispersion
from ..callbacks import Callback, CallbackState
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


def _resolve_model_batch_meta(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    *,
    spatial_ndim: int,
) -> dict[str, Any]:
    if epsilon.ndim == spatial_ndim:
        model_batched = False
        batch_size = 1
        spatial_shape = tuple(epsilon.shape)
    elif epsilon.ndim == spatial_ndim + 1:
        model_batched = True
        batch_size = int(epsilon.shape[0])
        spatial_shape = tuple(epsilon.shape[1:])
    else:
        raise RuntimeError(
            f"epsilon must be {spatial_ndim}D or {spatial_ndim + 1}D, "
            f"but got shape {tuple(epsilon.shape)}"
        )

    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    return {
        "model_batched": model_batched,
        "B": batch_size,
        "spatial_shape": spatial_shape,
    }


def _resolve_shot_tensor_S(
    name: str,
    tensor: torch.Tensor | None,
    *,
    B: int,
    tail_ndim: int,
) -> int | None:
    if tensor is None:
        return None
    if tensor.ndim == tail_ndim + 1:
        return int(tensor.shape[0])
    if tensor.ndim == tail_ndim + 2:
        if int(tensor.shape[0]) != B:
            raise RuntimeError(
                f"{name} leading batch dimension must match model batch size {B}, "
                f"but got shape {tuple(tensor.shape)}"
            )
        return int(tensor.shape[1])
    raise RuntimeError(
        f"{name} must have shape [S, ...] or [B, S, ...], "
        f"but got shape {tuple(tensor.shape)}"
    )


def _resolve_state_tensor_S(
    name: str,
    tensor: torch.Tensor | None,
    *,
    B: int,
    spatial_ndim: int,
) -> int | None:
    if tensor is None:
        return None
    if tensor.ndim == spatial_ndim:
        return None
    if tensor.ndim == spatial_ndim + 1:
        return int(tensor.shape[0])
    if tensor.ndim == spatial_ndim + 2:
        if int(tensor.shape[0]) != B:
            raise RuntimeError(
                f"{name} leading batch dimension must match model batch size {B}, "
                f"but got shape {tuple(tensor.shape)}"
            )
        return int(tensor.shape[1])
    raise RuntimeError(
        f"{name} must have shape [*spatial], [S, *spatial], or [B, S, *spatial], "
        f"but got shape {tuple(tensor.shape)}"
    )


def _structured_vmap_shot_in_dim(
    name: str,
    tensor: torch.Tensor | None,
    *,
    B: int,
    tail_ndim: int,
) -> int | None:
    if tensor is None:
        return None
    if tensor.ndim == tail_ndim + 1:
        return None
    if tensor.ndim == tail_ndim + 2:
        if int(tensor.shape[0]) != B:
            raise RuntimeError(
                f"{name} leading batch dimension must match model batch size {B}, "
                f"but got shape {tuple(tensor.shape)}"
            )
        return 0
    raise RuntimeError(
        f"{name} must have shape [S, ...] or [B, S, ...], "
        f"but got shape {tuple(tensor.shape)}"
    )


def _structured_vmap_state_in_dim(
    name: str,
    tensor: torch.Tensor | None,
    *,
    B: int,
    spatial_ndim: int,
) -> int | None:
    if tensor is None:
        return None
    if tensor.ndim in {spatial_ndim, spatial_ndim + 1}:
        return None
    if tensor.ndim == spatial_ndim + 2:
        if int(tensor.shape[0]) != B:
            raise RuntimeError(
                f"{name} leading batch dimension must match model batch size {B}, "
                f"but got shape {tuple(tensor.shape)}"
            )
        return 0
    raise RuntimeError(
        f"{name} must have shape [*spatial], [S, *spatial], or [B, S, *spatial], "
        f"but got shape {tuple(tensor.shape)}"
    )


def _normalize_shot_tensor(
    tensor: torch.Tensor | None,
    *,
    B: int,
    S: int,
    tail_ndim: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == tail_ndim + 1:
        if int(tensor.shape[0]) != S:
            raise RuntimeError(
                f"Expected shared shot tensor to have S={S}, but got shape {tuple(tensor.shape)}"
            )
        if B == 1:
            return tensor
        expand_shape = (B, *tensor.shape)
        return (
            tensor.unsqueeze(0)
            .expand(expand_shape)
            .reshape(B * S, *tensor.shape[1:])
            .contiguous()
        )
    if tensor.ndim == tail_ndim + 2:
        if int(tensor.shape[0]) != B or int(tensor.shape[1]) != S:
            raise RuntimeError(
                f"Expected per-model shot tensor to have shape [B={B}, S={S}, ...], "
                f"but got {tuple(tensor.shape)}"
            )
        if B == 1:
            return tensor.reshape(S, *tensor.shape[2:]).contiguous()
        return tensor.reshape(B * S, *tensor.shape[2:]).contiguous()
    raise RuntimeError(f"Unexpected shot tensor shape {tuple(tensor.shape)}")


def _normalize_state_tensor(
    tensor: torch.Tensor | None,
    *,
    B: int,
    S: int,
    spatial_ndim: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == spatial_ndim:
        if B == 1 and S == 1:
            return tensor
        return (
            tensor.reshape((1,) + tuple(tensor.shape))
            .expand(B * S, *tensor.shape)
            .contiguous()
        )
    if tensor.ndim == spatial_ndim + 1:
        if int(tensor.shape[0]) != S:
            raise RuntimeError(
                f"Expected shot-batched state tensor to have S={S}, but got {tuple(tensor.shape)}"
            )
        if B == 1:
            return tensor.contiguous()
        expand_shape = (B, *tensor.shape)
        return (
            tensor.unsqueeze(0)
            .expand(expand_shape)
            .reshape(B * S, *tensor.shape[1:])
            .contiguous()
        )
    if tensor.ndim == spatial_ndim + 2:
        if int(tensor.shape[0]) != B or int(tensor.shape[1]) != S:
            raise RuntimeError(
                f"Expected per-model state tensor to have shape [B={B}, S={S}, ...], "
                f"but got {tuple(tensor.shape)}"
            )
        if B == 1:
            return tensor.reshape(S, *tensor.shape[2:]).contiguous()
        return tensor.reshape(B * S, *tensor.shape[2:]).contiguous()
    raise RuntimeError(f"Unexpected state tensor shape {tuple(tensor.shape)}")


def _normalize_structured_batch(
    *,
    spatial_ndim: int,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    shot_tensors: dict[str, tuple[torch.Tensor | None, int]],
    state_tensors: dict[str, torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    model_meta = _resolve_model_batch_meta(
        epsilon,
        sigma,
        mu,
        spatial_ndim=spatial_ndim,
    )
    B = int(model_meta["B"])
    model_batched = bool(model_meta["model_batched"])

    S_candidates: list[int] = []
    for name, (tensor, tail_ndim) in shot_tensors.items():
        value = _resolve_shot_tensor_S(name, tensor, B=B, tail_ndim=tail_ndim)
        if value is not None:
            S_candidates.append(value)
    for name, tensor in (state_tensors or {}).items():
        value = _resolve_state_tensor_S(name, tensor, B=B, spatial_ndim=spatial_ndim)
        if value is not None:
            S_candidates.append(value)

    if S_candidates:
        S = S_candidates[0]
        for candidate in S_candidates[1:]:
            if candidate != S:
                raise RuntimeError(
                    f"All shot-batched tensors must agree on S, but got {S_candidates}"
                )
    else:
        S = 1

    if model_batched:
        epsilon_exec = epsilon.repeat_interleave(S, dim=0).contiguous()
        sigma_exec = sigma.repeat_interleave(S, dim=0).contiguous()
        mu_exec = mu.repeat_interleave(S, dim=0).contiguous()
    else:
        epsilon_exec = epsilon
        sigma_exec = sigma
        mu_exec = mu

    normalized_shots = {
        name: _normalize_shot_tensor(tensor, B=B, S=S, tail_ndim=tail_ndim)
        for name, (tensor, tail_ndim) in shot_tensors.items()
    }
    normalized_states = {
        name: _normalize_state_tensor(tensor, B=B, S=S, spatial_ndim=spatial_ndim)
        for name, tensor in (state_tensors or {}).items()
    }

    return {
        "B": B,
        "S": S,
        "N": B * S,
        "model_batched": model_batched,
        "structured_output": model_batched and B > 1,
        "spatial_ndim": spatial_ndim,
        "epsilon": epsilon_exec,
        "sigma": sigma_exec,
        "mu": mu_exec,
        "shot_tensors": normalized_shots,
        "state_tensors": normalized_states,
    }


def _reshape_structured_wavefield(
    tensor: torch.Tensor,
    *,
    batch_meta: dict[str, Any],
) -> torch.Tensor:
    if not batch_meta["structured_output"] or tensor.numel() == 0:
        return tensor
    spatial_ndim = int(batch_meta["spatial_ndim"])
    return tensor.reshape(
        int(batch_meta["B"]),
        int(batch_meta["S"]),
        *tensor.shape[-spatial_ndim:],
    )


def _reshape_structured_receiver_amplitudes(
    tensor: torch.Tensor,
    *,
    batch_meta: dict[str, Any],
) -> torch.Tensor:
    if not batch_meta["structured_output"] or tensor.numel() == 0:
        return tensor
    return tensor.reshape(
        tensor.shape[0],
        int(batch_meta["B"]),
        int(batch_meta["S"]),
        *tensor.shape[2:],
    )


def _wrap_structured_callback(
    callback: Callback | None,
    *,
    batch_meta: dict[str, Any],
) -> Callback | None:
    if callback is None or not batch_meta["structured_output"]:
        return callback

    B = int(batch_meta["B"])
    S = int(batch_meta["S"])
    spatial_ndim = int(batch_meta["spatial_ndim"])
    N = int(batch_meta["N"])

    def _reshape_wavefield_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < spatial_ndim + 1 or tensor.shape[0] != N:
            return tensor
        return tensor.reshape(B, S, *tensor.shape[-spatial_ndim:])

    def _reshape_model_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == spatial_ndim + 1 and tensor.shape[0] == N:
            return tensor.reshape(B, S, *tensor.shape[1:])[:, 0].contiguous()
        if tensor.ndim == spatial_ndim + 2 and tensor.shape[:2] == (B, S):
            return tensor[:, 0].contiguous()
        return tensor

    def _reshape_gradient_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == spatial_ndim + 1 and tensor.shape[0] == N:
            return tensor.reshape(B, S, *tensor.shape[1:]).sum(dim=1).contiguous()
        if tensor.ndim == spatial_ndim + 2 and tensor.shape[:2] == (B, S):
            return tensor.sum(dim=1).contiguous()
        return tensor

    def wrapped(state: CallbackState) -> None:
        wavefields = {
            name: _reshape_wavefield_tensor(value)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in state._wavefields.items()
        }
        models = {
            name: _reshape_model_tensor(value)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in state._models.items()
        }
        gradients = {
            name: _reshape_gradient_tensor(value)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in state._gradients.items()
        }
        callback(
            CallbackState(
                dt=state.dt,
                step=state.step,
                nt=state.nt,
                wavefields=wavefields,
                models=models,
                gradients=gradients,
                fd_pad=list(state._fd_pad),
                pml_width=list(state._pml_width),
                is_backward=state.is_backward,
                grid_spacing=state._grid_spacing,
            )
        )

    return wrapped


__all__ = [
    "_copy_if_present",
    "_debye_polarization_term",
    "_get_ctx_handle",
    "_init_polarization_state",
    "_make_compute_stream",
    "_make_storage_streams",
    "_normalize_structured_batch",
    "_pad_dispersion_for_model",
    "_reshape_structured_receiver_amplitudes",
    "_reshape_structured_wavefield",
    "_register_ctx_handle",
    "_release_ctx_handle",
    "_stream_handle",
    "_structured_vmap_shot_in_dim",
    "_structured_vmap_state_in_dim",
    "_wrap_structured_callback",
]
