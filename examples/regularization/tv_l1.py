from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Literal, Sequence

import torch
import torch.nn as nn

Reduction = Literal["sum", "mean"]
TVParts = tuple[torch.Tensor, ...]


def _infer_spatial_ndim(model: torch.Tensor) -> int:
    if model.ndim < 2:
        raise ValueError(f"TV requires at least 2 spatial dimensions, got shape={tuple(model.shape)}.")
    if model.ndim == 2:
        return 2
    if model.ndim == 3:
        return 3
    return 2


def _normalize_spacing(spacing: float | Sequence[float], spatial_ndim: int) -> tuple[float, ...]:
    if isinstance(spacing, Real):
        values = (float(spacing),) * spatial_ndim
    else:
        values = tuple(float(value) for value in spacing)
    if len(values) != spatial_ndim:
        raise ValueError(f"spacing must have length {spatial_ndim}, got {len(values)}.")
    if any(value <= 0.0 for value in values):
        raise ValueError("spacing values must be positive.")
    return values


def _as_active_mask(
    active_mask: torch.Tensor | None,
    model: torch.Tensor,
    spatial_ndim: int,
) -> torch.Tensor | None:
    if active_mask is None:
        return None
    if active_mask.shape[-spatial_ndim:] == model.shape[-spatial_ndim:]:
        return active_mask.to(device=model.device, dtype=torch.bool)
    raise ValueError(
        "active_mask must match the model spatial shape. "
        f"Got mask shape={tuple(active_mask.shape)}, model shape={tuple(model.shape)}."
    )


def _difference_masks(active_mask: torch.Tensor | None, spatial_ndim: int) -> TVParts | None:
    if active_mask is None:
        return None
    if spatial_ndim == 2:
        dz_mask = active_mask[..., 1:, :] & active_mask[..., :-1, :]
        dx_mask = active_mask[..., :, 1:] & active_mask[..., :, :-1]
        return dz_mask, dx_mask
    if spatial_ndim == 3:
        dz_mask = active_mask[..., 1:, :, :] & active_mask[..., :-1, :, :]
        dy_mask = active_mask[..., :, 1:, :] & active_mask[..., :, :-1, :]
        dx_mask = active_mask[..., :, :, 1:] & active_mask[..., :, :, :-1]
        return dz_mask, dy_mask, dx_mask
    raise ValueError(f"Unsupported spatial_ndim={spatial_ndim}.")


def _apply_masks(parts: TVParts, masks: TVParts | None) -> TVParts:
    if masks is None:
        return parts
    return tuple(part * mask.to(device=part.device, dtype=part.dtype) for part, mask in zip(parts, masks, strict=True))


def tv_forward(
    model: torch.Tensor,
    *,
    spacing: float | Sequence[float] = 1.0,
    active_mask: torch.Tensor | None = None,
    spatial_ndim: int | None = None,
) -> TVParts:
    """Return anisotropic forward differences in ``(dz, dx)`` or ``(dz, dy, dx)`` order.

    The last two or three dimensions are treated as spatial dimensions. For the
    2D FWI examples in this repository, model shape is ``(..., nz, nx)`` and
    ``spacing`` is ``(dz, dx)``. Differences crossing inactive mask cells are
    zeroed, which avoids penalizing the air/subsurface interface.
    """

    spatial_ndim = _infer_spatial_ndim(model) if spatial_ndim is None else int(spatial_ndim)
    h = _normalize_spacing(spacing, spatial_ndim)
    mask = _as_active_mask(active_mask, model, spatial_ndim)

    if spatial_ndim == 2:
        dz = (model[..., 1:, :] - model[..., :-1, :]) / h[0]
        dx = (model[..., :, 1:] - model[..., :, :-1]) / h[1]
        return _apply_masks((dz, dx), _difference_masks(mask, spatial_ndim))

    if spatial_ndim == 3:
        dz = (model[..., 1:, :, :] - model[..., :-1, :, :]) / h[0]
        dy = (model[..., :, 1:, :] - model[..., :, :-1, :]) / h[1]
        dx = (model[..., :, :, 1:] - model[..., :, :, :-1]) / h[2]
        return _apply_masks((dz, dy, dx), _difference_masks(mask, spatial_ndim))

    raise ValueError(f"Unsupported spatial_ndim={spatial_ndim}.")


def tv_adjoint(
    parts: TVParts,
    *,
    input_shape: Sequence[int],
    spacing: float | Sequence[float] = 1.0,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the adjoint of :func:`tv_forward` to TV-domain tensors."""

    spatial_ndim = len(parts)
    h = _normalize_spacing(spacing, spatial_ndim)
    ref = parts[0]
    out = torch.zeros(tuple(input_shape), device=ref.device, dtype=ref.dtype)
    mask = _as_active_mask(active_mask, out, spatial_ndim)
    masked_parts = _apply_masks(parts, _difference_masks(mask, spatial_ndim))

    if spatial_ndim == 2:
        dz, dx = masked_parts
        out[..., :-1, :] -= dz / h[0]
        out[..., 1:, :] += dz / h[0]
        out[..., :, :-1] -= dx / h[1]
        out[..., :, 1:] += dx / h[1]
        return out

    if spatial_ndim == 3:
        dz, dy, dx = masked_parts
        out[..., :-1, :, :] -= dz / h[0]
        out[..., 1:, :, :] += dz / h[0]
        out[..., :, :-1, :] -= dy / h[1]
        out[..., :, 1:, :] += dy / h[1]
        out[..., :, :, :-1] -= dx / h[2]
        out[..., :, :, 1:] += dx / h[2]
        return out

    raise ValueError(f"Unsupported number of TV parts: {spatial_ndim}.")


def _smooth_abs(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon <= 0.0:
        return value.abs()
    eps = torch.as_tensor(epsilon, device=value.device, dtype=value.dtype)
    return torch.sqrt(value.square() + eps.square()) - eps


def _masked_count(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=target.device, dtype=target.dtype)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(0)
    return torch.broadcast_to(mask, target.shape).sum()


def tv_l1_value(
    model: torch.Tensor,
    *,
    spacing: float | Sequence[float] = 1.0,
    active_mask: torch.Tensor | None = None,
    reduction: Reduction = "sum",
    epsilon: float = 0.0,
    spatial_ndim: int | None = None,
) -> torch.Tensor:
    """Compute anisotropic TV-L1 value for a 2D or 3D model tensor."""

    spatial_ndim = _infer_spatial_ndim(model) if spatial_ndim is None else int(spatial_ndim)
    parts = tv_forward(
        model,
        spacing=spacing,
        active_mask=active_mask,
        spatial_ndim=spatial_ndim,
    )
    value = sum(_smooth_abs(part, epsilon).sum() for part in parts)

    if reduction == "sum":
        return value
    if reduction == "mean":
        masks = _difference_masks(_as_active_mask(active_mask, model, spatial_ndim), spatial_ndim)
        if masks is None:
            denom = sum(part.numel() for part in parts)
            return value / max(denom, 1)
        denom_tensor = sum(_masked_count(mask, part) for mask, part in zip(masks, parts, strict=True))
        return value / denom_tensor.clamp_min(1.0)
    raise ValueError(f"Unsupported reduction={reduction!r}.")


class TVL1Regularizer(nn.Module):
    """Differentiable anisotropic TV-L1 penalty for FWI losses."""

    def __init__(
        self,
        *,
        weight: float,
        spacing: float | Sequence[float] = 1.0,
        reduction: Reduction = "mean",
        epsilon: float = 0.0,
        spatial_ndim: int | None = None,
    ) -> None:
        super().__init__()
        if weight < 0.0:
            raise ValueError("weight must be non-negative.")
        if reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'.")
        self.weight = float(weight)
        self.spacing = spacing
        self.reduction = reduction
        self.epsilon = float(epsilon)
        self.spatial_ndim = spatial_ndim

    def forward(
        self,
        model: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.weight == 0.0:
            return model.new_zeros(())
        value = tv_l1_value(
            model,
            spacing=self.spacing,
            active_mask=active_mask,
            reduction=self.reduction,
            epsilon=self.epsilon,
            spatial_ndim=self.spatial_ndim,
        )
        return value * self.weight

    def extra_repr(self) -> str:
        return (
            f"weight={self.weight:g}, spacing={self.spacing}, "
            f"reduction={self.reduction!r}, epsilon={self.epsilon:g}"
        )


def _flatten_parts(parts: TVParts) -> tuple[torch.Tensor, tuple[torch.Size, ...], tuple[int, ...]]:
    shapes = tuple(part.shape for part in parts)
    sizes = tuple(part.numel() for part in parts)
    flat = torch.cat([part.reshape(-1) for part in parts])
    return flat, shapes, sizes


def _unflatten_parts(flat: torch.Tensor, shapes: tuple[torch.Size, ...], sizes: tuple[int, ...]) -> TVParts:
    out = []
    offset = 0
    for shape, size in zip(shapes, sizes, strict=True):
        out.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return tuple(out)


@torch.no_grad()
def project_l1_ball(vector: torch.Tensor, radius: float | torch.Tensor) -> torch.Tensor:
    """Euclidean projection of a tensor, flattened, onto an L1 ball."""

    radius_tensor = torch.as_tensor(radius, device=vector.device, dtype=vector.dtype)
    if radius_tensor.ndim != 0:
        raise ValueError("radius must be scalar.")
    radius_value = float(radius_tensor.item())
    if radius_value < 0.0:
        raise ValueError("radius must be non-negative.")
    if vector.numel() == 0:
        return vector.clone()
    if radius_value == 0.0:
        return torch.zeros_like(vector)

    flat = vector.reshape(-1)
    abs_flat = flat.abs()
    if float(abs_flat.sum().item()) <= radius_value:
        return vector.clone()

    sorted_abs = torch.sort(abs_flat, descending=True).values
    cumsum = torch.cumsum(sorted_abs, dim=0)
    indices = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=flat.dtype)
    keep = sorted_abs > (cumsum - radius_tensor) / indices
    rho = int(torch.nonzero(keep, as_tuple=False)[-1].item()) + 1
    theta = (cumsum[rho - 1] - radius_tensor) / float(rho)
    projected = flat.sign() * torch.clamp(abs_flat - theta, min=0.0)
    return projected.reshape_as(vector)


@torch.no_grad()
def project_parts_onto_l1_ball(parts: TVParts, radius: float | torch.Tensor) -> TVParts:
    """Project concatenated TV-domain parts onto one shared L1 ball."""

    flat, shapes, sizes = _flatten_parts(parts)
    projected = project_l1_ball(flat, radius)
    return _unflatten_parts(projected, shapes, sizes)


def _parts_l1(parts: TVParts) -> torch.Tensor:
    return sum(part.abs().sum() for part in parts)


def _parts_l2(parts: TVParts) -> torch.Tensor:
    return torch.sqrt(sum(part.square().sum() for part in parts))


def _cg_solve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    x0: torch.Tensor,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, int, float]:
    x = x0.clone()
    r = rhs - matvec(x)
    p = r.clone()
    rs_old = torch.sum(r * r)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    relres = float((torch.sqrt(rs_old) / rhs_norm).item())

    for iteration in range(1, max_iter + 1):
        ap = matvec(p)
        denom = torch.sum(p * ap)
        if float(denom.abs().item()) <= torch.finfo(rhs.dtype).eps:
            return x, iteration - 1, relres
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.sum(r * r)
        relres = float((torch.sqrt(rs_new) / rhs_norm).item())
        if relres <= tol:
            return x, iteration, relres
        beta = rs_new / rs_old.clamp_min(torch.finfo(rhs.dtype).eps)
        p = r + beta * p
        rs_old = rs_new

    return x, max_iter, relres


@dataclass(frozen=True)
class TVL1BallProjectionInfo:
    iterations: int
    converged: bool
    primal_residual: float
    relative_primal_residual: float
    tv_l1: float
    radius: float
    cg_iterations: int
    cg_relative_residual: float


@torch.no_grad()
def project_tv_l1_ball(
    model: torch.Tensor,
    *,
    radius: float,
    spacing: float | Sequence[float] = 1.0,
    active_mask: torch.Tensor | None = None,
    rho: float = 10.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    cg_max_iter: int = 80,
    cg_tol: float = 1e-5,
    spatial_ndim: int | None = None,
) -> tuple[torch.Tensor, TVL1BallProjectionInfo]:
    """Project ``model`` onto ``||TV(model)||_1 <= radius`` using matrix-free ADMM.

    This is the direct PyTorch analogue of the single ``set_type="l1",
    TD_OP="TV"`` projection in SetIntersectionProjection.jl. It is intended for
    projected-gradient style FWI when the model tensor itself is optimized. For
    implicit neural FWI, use :class:`TVL1Regularizer` in the loss instead.
    """

    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if cg_max_iter <= 0:
        raise ValueError("cg_max_iter must be positive.")

    spatial_ndim = _infer_spatial_ndim(model) if spatial_ndim is None else int(spatial_ndim)
    m = model.detach().clone()
    x = m.clone()
    dx = tv_forward(x, spacing=spacing, active_mask=active_mask, spatial_ndim=spatial_ndim)
    initial_tv = float(_parts_l1(dx).item())
    if initial_tv <= radius * (1.0 + tol):
        info = TVL1BallProjectionInfo(
            iterations=0,
            converged=True,
            primal_residual=0.0,
            relative_primal_residual=0.0,
            tv_l1=initial_tv,
            radius=float(radius),
            cg_iterations=0,
            cg_relative_residual=0.0,
        )
        return x, info

    y = project_parts_onto_l1_ball(dx, radius)
    u = tuple(torch.zeros_like(part) for part in y)
    total_cg_iterations = 0
    last_cg_relres = 0.0
    primal = float("inf")
    rel_primal = float("inf")
    tv_value = initial_tv

    def matvec(value: torch.Tensor) -> torch.Tensor:
        tv_value_parts = tv_forward(
            value,
            spacing=spacing,
            active_mask=active_mask,
            spatial_ndim=spatial_ndim,
        )
        return value + rho * tv_adjoint(
            tv_value_parts,
            input_shape=value.shape,
            spacing=spacing,
            active_mask=active_mask,
        )

    for iteration in range(1, max_iter + 1):
        rhs_parts = tuple(yi - ui for yi, ui in zip(y, u, strict=True))
        rhs = m + rho * tv_adjoint(
            rhs_parts,
            input_shape=m.shape,
            spacing=spacing,
            active_mask=active_mask,
        )
        x, cg_iterations, last_cg_relres = _cg_solve(
            matvec,
            rhs,
            x0=x,
            max_iter=cg_max_iter,
            tol=cg_tol,
        )
        total_cg_iterations += cg_iterations

        dx = tv_forward(x, spacing=spacing, active_mask=active_mask, spatial_ndim=spatial_ndim)
        y_arg = tuple(di + ui for di, ui in zip(dx, u, strict=True))
        y = project_parts_onto_l1_ball(y_arg, radius)
        u = tuple(ui + di - yi for ui, di, yi in zip(u, dx, y, strict=True))

        primal = float(_parts_l2(tuple(di - yi for di, yi in zip(dx, y, strict=True))).item())
        dx_l2 = float(_parts_l2(dx).item())
        rel_primal = primal / max(dx_l2, torch.finfo(x.dtype).eps)
        tv_value = float(_parts_l1(dx).item())
        if rel_primal <= tol and tv_value <= radius * (1.0 + 10.0 * tol):
            info = TVL1BallProjectionInfo(
                iterations=iteration,
                converged=True,
                primal_residual=primal,
                relative_primal_residual=rel_primal,
                tv_l1=tv_value,
                radius=float(radius),
                cg_iterations=total_cg_iterations,
                cg_relative_residual=last_cg_relres,
            )
            return x, info

    info = TVL1BallProjectionInfo(
        iterations=max_iter,
        converged=False,
        primal_residual=primal,
        relative_primal_residual=rel_primal,
        tv_l1=tv_value,
        radius=float(radius),
        cg_iterations=total_cg_iterations,
        cg_relative_residual=last_cg_relres,
    )
    return x, info
