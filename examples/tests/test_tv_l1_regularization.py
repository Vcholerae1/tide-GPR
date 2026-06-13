from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regularization.tv_l1 import (
    project_parts_onto_l1_ball,
    project_tv_l1_ball,
    tv_adjoint,
    tv_forward,
    tv_l1_value,
)


def test_tv_l1_value_matches_manual_2d() -> None:
    model = torch.tensor(
        [[1.0, 3.0, 2.0], [4.0, 6.0, 10.0]],
        dtype=torch.float64,
    )
    dz = (model[1:, :] - model[:-1, :]) / 2.0
    dx = (model[:, 1:] - model[:, :-1]) / 0.5
    expected = dz.abs().sum() + dx.abs().sum()

    actual = tv_l1_value(model, spacing=(2.0, 0.5), reduction="sum", spatial_ndim=2)

    assert torch.allclose(actual, expected)


def test_tv_adjoint_matches_inner_product_2d_with_mask() -> None:
    torch.manual_seed(1)
    model = torch.randn(5, 6, dtype=torch.float64)
    active_mask = torch.ones_like(model, dtype=torch.bool)
    active_mask[:1, :] = False
    parts = tv_forward(model, spacing=(0.7, 1.3), active_mask=active_mask, spatial_ndim=2)
    dual = tuple(torch.randn_like(part) for part in parts)

    lhs = sum((part * dual_part).sum() for part, dual_part in zip(parts, dual, strict=True))
    rhs = (
        model
        * tv_adjoint(
            dual,
            input_shape=model.shape,
            spacing=(0.7, 1.3),
            active_mask=active_mask,
        )
    ).sum()

    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_tv_adjoint_matches_inner_product_3d() -> None:
    torch.manual_seed(2)
    model = torch.randn(4, 5, 6, dtype=torch.float64)
    parts = tv_forward(model, spacing=(0.8, 1.1, 1.4), spatial_ndim=3)
    dual = tuple(torch.randn_like(part) for part in parts)

    lhs = sum((part * dual_part).sum() for part, dual_part in zip(parts, dual, strict=True))
    rhs = (
        model
        * tv_adjoint(
            dual,
            input_shape=model.shape,
            spacing=(0.8, 1.1, 1.4),
        )
    ).sum()

    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_project_parts_onto_l1_ball_hits_radius() -> None:
    parts = (
        torch.tensor([3.0, -1.0, 0.5], dtype=torch.float64),
        torch.tensor([-2.0, 4.0], dtype=torch.float64),
    )
    projected = project_parts_onto_l1_ball(parts, radius=3.0)
    projected_l1 = sum(part.abs().sum() for part in projected)

    assert torch.allclose(projected_l1, torch.tensor(3.0, dtype=torch.float64), atol=1e-12)


def test_project_tv_l1_ball_enforces_constraint() -> None:
    torch.manual_seed(3)
    model = torch.randn(7, 8, dtype=torch.float64)
    radius = float(0.65 * tv_l1_value(model, spatial_ndim=2).item())

    projected, info = project_tv_l1_ball(
        model,
        radius=radius,
        spacing=(1.0, 1.0),
        rho=8.0,
        max_iter=200,
        tol=2e-4,
        cg_max_iter=100,
        cg_tol=1e-9,
        spatial_ndim=2,
    )
    projected_tv = float(tv_l1_value(projected, spatial_ndim=2).item())

    assert info.converged
    assert projected_tv <= radius * (1.0 + 5e-3)


if __name__ == "__main__":
    test_tv_l1_value_matches_manual_2d()
    test_tv_adjoint_matches_inner_product_2d_with_mask()
    test_tv_adjoint_matches_inner_product_3d()
    test_project_parts_onto_l1_ball_hits_radius()
    test_project_tv_l1_ball_enforces_constraint()
