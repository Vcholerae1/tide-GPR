"""Preconditioner helpers for workflow and optimizer integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class BlockPreconditioner:
    """Symmetric 2x2 block-diagonal preconditioner factors."""

    diag11: torch.Tensor
    offdiag12: torch.Tensor
    diag22: torch.Tensor


def _as_bool_mask(mask: torch.Tensor | np.ndarray, *, device: torch.device) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        return mask.to(device=device, dtype=torch.bool)
    return torch.as_tensor(mask, device=device, dtype=torch.bool)


def _finite_nonnegative_values(values: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        values.detach().to(dtype=torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)


def _smooth(values: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return values
    smoothed = gaussian_filter(values.cpu().numpy(), sigma=float(sigma))
    return torch.as_tensor(smoothed, device=values.device, dtype=torch.float32)


def _active_mask(
    values: torch.Tensor,
    inactive_mask: torch.Tensor | np.ndarray | None,
) -> torch.Tensor:
    if inactive_mask is None:
        return torch.ones_like(values, dtype=torch.bool)
    inactive = _as_bool_mask(inactive_mask, device=values.device)
    if inactive.shape != values.shape:
        raise ValueError(
            f"inactive_mask shape must match curvature shape, got "
            f"{tuple(inactive.shape)} and {tuple(values.shape)}."
        )
    return ~inactive


def curvature_preconditioner_diagonal(
    curvature: torch.Tensor,
    *,
    inactive_mask: torch.Tensor | np.ndarray | None = None,
    smooth_sigma: float = 0.0,
    damping: float = 5e-2,
    power: float = 0.5,
    scale_quantile: float = 0.95,
    clip_min: float | None = None,
    clip_max: float | None = None,
    blend: float = 1.0,
) -> torch.Tensor:
    """Build a diagonal scaling from a non-negative curvature proxy.

    This matches the common example pattern:
    smooth gradient-energy, normalize by a high quantile, invert with damping,
    normalize by median, clip, blend toward identity, and zero inactive cells.
    """

    if not isinstance(curvature, torch.Tensor):
        raise TypeError("curvature must be a torch.Tensor.")
    if smooth_sigma < 0.0:
        raise ValueError("smooth_sigma must be non-negative.")
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if power < 0.0:
        raise ValueError("power must be non-negative.")
    if not (0.0 <= scale_quantile <= 1.0):
        raise ValueError("scale_quantile must be in [0, 1].")
    if not (0.0 <= blend <= 1.0):
        raise ValueError("blend must be in [0, 1].")
    if clip_min is not None and clip_max is not None and clip_max < clip_min:
        raise ValueError("clip_max must be >= clip_min.")

    device = curvature.device
    dtype = curvature.dtype if curvature.is_floating_point() else torch.float32
    values = _smooth(_finite_nonnegative_values(curvature), smooth_sigma)
    active = _active_mask(values, inactive_mask)

    active_values = values[active]
    if active_values.numel() == 0:
        diagonal = torch.zeros_like(values)
        return diagonal.to(dtype=dtype)

    scale = torch.quantile(active_values, float(scale_quantile))
    if not torch.isfinite(scale).item() or float(scale) <= 0.0:
        scale = torch.ones((), device=device, dtype=values.dtype)
    normalized = values / scale

    diagonal = torch.pow(normalized + float(damping), -float(power))
    diagonal = torch.nan_to_num(diagonal, nan=0.0, posinf=0.0, neginf=0.0)

    active_diagonal = diagonal[active]
    if active_diagonal.numel() > 0:
        median = torch.median(active_diagonal)
        if torch.isfinite(median).item() and float(median) > 0.0:
            diagonal = diagonal / median
        if clip_min is not None or clip_max is not None:
            if clip_min is not None:
                diagonal = diagonal.clamp_min(float(clip_min))
            if clip_max is not None:
                diagonal = diagonal.clamp_max(float(clip_max))
        if blend < 1.0:
            diagonal = (1.0 - float(blend)) + float(blend) * diagonal

    diagonal = torch.where(active, diagonal, torch.zeros_like(diagonal))
    return diagonal.to(dtype=dtype)


def curvature_preconditioner_block(
    curvature_11: torch.Tensor,
    curvature_22: torch.Tensor,
    curvature_12: torch.Tensor | None = None,
    *,
    inactive_mask: torch.Tensor | np.ndarray | None = None,
    smooth_sigma: float = 0.0,
    damping: float = 5e-2,
    power: float = 0.5,
    scale_quantile: float = 0.95,
    determinant_quantile: float = 0.02,
    determinant_floor: float = 1e-8,
    clip_min: float | None = None,
    clip_max: float | None = None,
    offdiag_correlation_max: float = 0.8,
    blend: float = 1.0,
) -> BlockPreconditioner:
    """Build a symmetric 2x2 block-diagonal preconditioner from curvature proxies."""

    if not isinstance(curvature_11, torch.Tensor):
        raise TypeError("curvature_11 must be a torch.Tensor.")
    if not isinstance(curvature_22, torch.Tensor):
        raise TypeError("curvature_22 must be a torch.Tensor.")
    if curvature_11.shape != curvature_22.shape:
        raise ValueError("curvature_11 and curvature_22 must have matching shapes.")
    if curvature_12 is not None and curvature_12.shape != curvature_11.shape:
        raise ValueError("curvature_12 must match curvature_11 shape.")
    if smooth_sigma < 0.0:
        raise ValueError("smooth_sigma must be non-negative.")
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if power < 0.0:
        raise ValueError("power must be non-negative.")
    if not (0.0 <= scale_quantile <= 1.0):
        raise ValueError("scale_quantile must be in [0, 1].")
    if not (0.0 <= determinant_quantile <= 1.0):
        raise ValueError("determinant_quantile must be in [0, 1].")
    if determinant_floor <= 0.0:
        raise ValueError("determinant_floor must be positive.")
    if clip_min is not None and clip_max is not None and clip_max < clip_min:
        raise ValueError("clip_max must be >= clip_min.")
    if not (0.0 <= offdiag_correlation_max <= 1.0):
        raise ValueError("offdiag_correlation_max must be in [0, 1].")
    if not (0.0 <= blend <= 1.0):
        raise ValueError("blend must be in [0, 1].")

    device = curvature_11.device
    dtype = curvature_11.dtype if curvature_11.is_floating_point() else torch.float32
    h11 = _smooth(_finite_nonnegative_values(curvature_11), smooth_sigma)
    h22 = _smooth(_finite_nonnegative_values(curvature_22), smooth_sigma)
    if curvature_12 is None:
        h12 = torch.zeros_like(h11)
    else:
        h12 = torch.nan_to_num(
            curvature_12.detach().to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        h12 = _smooth(h12, smooth_sigma)

    active = _active_mask(h11, inactive_mask)
    active_values = torch.cat([h11[active], h22[active]])
    if active_values.numel() == 0:
        zeros = torch.zeros_like(h11, dtype=dtype)
        return BlockPreconditioner(zeros, zeros, zeros)

    scale = torch.quantile(active_values, float(scale_quantile))
    if not torch.isfinite(scale).item() or float(scale) <= 0.0:
        scale = torch.ones((), device=device, dtype=h11.dtype)
    h11 = h11 / scale
    h22 = h22 / scale
    h12 = h12 / scale

    h11 = torch.pow(h11 + float(damping), float(power))
    h22 = torch.pow(h22 + float(damping), float(power))
    h12 = torch.sign(h12) * torch.pow(torch.abs(h12), float(power))

    det = h11 * h22 - h12 * h12
    det_active = det[active]
    det_floor = torch.quantile(det_active, float(determinant_quantile))
    if not torch.isfinite(det_floor).item():
        det_floor = torch.zeros((), device=device, dtype=h11.dtype)
    det_floor = torch.clamp(det_floor, min=float(determinant_floor))
    det = torch.maximum(det, det_floor)

    inv11 = torch.nan_to_num(h22 / det, nan=0.0, posinf=0.0, neginf=0.0)
    inv22 = torch.nan_to_num(h11 / det, nan=0.0, posinf=0.0, neginf=0.0)
    inv12 = torch.nan_to_num(-h12 / det, nan=0.0, posinf=0.0, neginf=0.0)

    active_diagonal = torch.cat([inv11[active], inv22[active]])
    if active_diagonal.numel() > 0:
        median = torch.median(active_diagonal)
        if torch.isfinite(median).item() and float(median) > 0.0:
            inv11 = inv11 / median
            inv22 = inv22 / median
            inv12 = inv12 / median

    if clip_min is not None:
        inv11 = inv11.clamp_min(float(clip_min))
        inv22 = inv22.clamp_min(float(clip_min))
    if clip_max is not None:
        inv11 = inv11.clamp_max(float(clip_max))
        inv22 = inv22.clamp_max(float(clip_max))

    cross_limit = float(offdiag_correlation_max) * torch.sqrt(
        torch.clamp(inv11 * inv22, min=1e-12)
    )
    inv12 = torch.maximum(torch.minimum(inv12, cross_limit), -cross_limit)

    if blend < 1.0:
        inv11 = (1.0 - float(blend)) + float(blend) * inv11
        inv22 = (1.0 - float(blend)) + float(blend) * inv22
        inv12 = float(blend) * inv12

    zeros = torch.zeros_like(inv11)
    inv11 = torch.where(active, inv11, zeros)
    inv22 = torch.where(active, inv22, zeros)
    inv12 = torch.where(active, inv12, zeros)
    return BlockPreconditioner(
        inv11.to(dtype=dtype),
        inv12.to(dtype=dtype),
        inv22.to(dtype=dtype),
    )


def diagonal_preconditioner(
    diagonal: torch.Tensor | np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], None]:
    """Return a ``tide.optim``-style preconditioner callback."""

    if isinstance(diagonal, torch.Tensor):
        diagonal_np = diagonal.detach().cpu().numpy()
    else:
        diagonal_np = np.asarray(diagonal)
    flat = diagonal_np.reshape(-1).astype(np.float32, copy=True)
    flat[~np.isfinite(flat)] = 0.0

    def apply_diagonal(_: np.ndarray, vector: np.ndarray, out: np.ndarray) -> None:
        if vector.shape != flat.shape or out.shape != flat.shape:
            raise ValueError(
                f"vector and out must have shape {flat.shape}, got {vector.shape} and {out.shape}."
            )
        out[:] = flat * vector

    return apply_diagonal


def block_preconditioner(
    block: BlockPreconditioner,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], None]:
    """Return a ``tide.optim`` callback for a symmetric 2x2 block preconditioner."""

    diag11 = block.diag11.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)
    offdiag12 = (
        block.offdiag12.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)
    )
    diag22 = block.diag22.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)
    if not (diag11.shape == offdiag12.shape == diag22.shape):
        raise ValueError("block factors must have matching flattened shapes.")
    n = int(diag11.size)
    flat_shape = (2 * n,)
    tmp = np.empty(n, dtype=np.float32)

    for arr in (diag11, offdiag12, diag22):
        arr[~np.isfinite(arr)] = 0.0

    def apply_block(_: np.ndarray, vector: np.ndarray, out: np.ndarray) -> None:
        if vector.shape != flat_shape or out.shape != flat_shape:
            raise ValueError(
                f"vector and out must have shape {flat_shape}, got "
                f"{vector.shape} and {out.shape}."
            )
        vector_1 = vector[:n]
        vector_2 = vector[n:]
        out_1 = out[:n]
        out_2 = out[n:]
        np.multiply(diag11, vector_1, out=out_1)
        np.multiply(offdiag12, vector_2, out=tmp)
        out_1 += tmp
        np.multiply(offdiag12, vector_1, out=out_2)
        np.multiply(diag22, vector_2, out=tmp)
        out_2 += tmp
        out[~np.isfinite(out)] = 0.0

    return apply_block


__all__ = [
    "BlockPreconditioner",
    "block_preconditioner",
    "curvature_preconditioner_block",
    "curvature_preconditioner_diagonal",
    "diagonal_preconditioner",
]
