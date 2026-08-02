"""Tensor validation and box-constraint helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def _as_model_tensor(name: str, value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.layout != torch.strided:
        raise TypeError(f"{name} must be a dense strided tensor.")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64.")
    if value.is_complex():
        raise TypeError(f"{name} must be real.")
    if value.numel() == 0:
        raise ValueError(f"{name} must contain at least one element.")
    return value.detach().clone(memory_format=torch.preserve_format)


def _as_bound(
    name: str,
    value: Tensor | float | None,
    x: Tensor,
    *,
    default: float,
) -> Tensor:
    if value is None:
        return torch.full_like(x, default)
    bound = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    try:
        return torch.broadcast_to(bound, x.shape).clone()
    except RuntimeError as exc:
        raise ValueError(f"{name} must be scalar or broadcastable to x.shape.") from exc


def _prepare_bounds(
    x: Tensor,
    lower_bounds: Tensor | float | None,
    upper_bounds: Tensor | float | None,
) -> tuple[Tensor | None, Tensor | None]:
    if lower_bounds is None and upper_bounds is None:
        return None, None
    lower = _as_bound("lower_bounds", lower_bounds, x, default=-torch.inf)
    upper = _as_bound("upper_bounds", upper_bounds, x, default=torch.inf)
    if torch.isnan(lower).any() or torch.isnan(upper).any():
        raise ValueError("bounds must not contain NaN.")
    if torch.any(lower > upper):
        raise ValueError("lower_bounds must be <= upper_bounds.")
    return lower, upper


def _project(x: Tensor, lower: Tensor | None, upper: Tensor | None) -> Tensor:
    if lower is None or upper is None:
        return x
    return torch.maximum(torch.minimum(x, upper), lower)


def _projected_gradient(
    x: Tensor,
    grad: Tensor,
    lower: Tensor | None,
    upper: Tensor | None,
) -> Tensor:
    if lower is None or upper is None:
        return grad
    result = grad.clone()
    at_lower = x <= lower
    at_upper = x >= upper
    result.masked_fill_(at_lower & (grad > 0), 0)
    result.masked_fill_(at_upper & (grad < 0), 0)
    result.masked_fill_(lower == upper, 0)
    return result


def _feasible_direction(
    x: Tensor,
    direction: Tensor,
    lower: Tensor | None,
    upper: Tensor | None,
) -> Tensor:
    if lower is None or upper is None:
        return direction
    result = direction.clone()
    result.masked_fill_((x <= lower) & (result < 0), 0)
    result.masked_fill_((x >= upper) & (result > 0), 0)
    result.masked_fill_(lower == upper, 0)
    return result


def _dot(a: Tensor, b: Tensor) -> float:
    return float(torch.dot(a.reshape(-1), b.reshape(-1)).item())


def _norm_inf(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value.reshape(-1), ord=torch.inf).item())


def _validate_operator_output(name: str, value: Tensor, reference: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must return a torch.Tensor.")
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} returned shape {tuple(value.shape)}, expected {tuple(reference.shape)}."
        )
    if value.device != reference.device:
        raise ValueError(f"{name} output must be on {reference.device}.")
    if value.dtype != reference.dtype:
        raise ValueError(f"{name} output must have dtype {reference.dtype}.")
    return value.detach().clone(memory_format=torch.preserve_format)
