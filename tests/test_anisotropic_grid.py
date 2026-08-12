from __future__ import annotations

import math

import pytest
import torch

import tide
from tide.cfl import cfl_condition
from tide.grid_utils import _normalize_grid_spacing_2d, _normalize_grid_spacing_3d

from numerical_utils import MaxwellExample, assert_finite_nonzero, relative_l2


@pytest.mark.numerical
def test_grid_spacing_normalization_preserves_axis_order() -> None:
    assert _normalize_grid_spacing_2d(0.02) == [0.02, 0.02]
    assert _normalize_grid_spacing_2d([0.018, 0.022]) == [0.018, 0.022]
    assert _normalize_grid_spacing_3d(0.02) == [0.02, 0.02, 0.02]
    assert _normalize_grid_spacing_3d([0.016, 0.018, 0.022]) == [
        0.016,
        0.018,
        0.022,
    ]


@pytest.mark.numerical
def test_cfl_uses_every_active_axis_spacing() -> None:
    dt = 1.0e-10
    velocity = 1.5e8
    with pytest.warns(UserWarning, match="CFL condition requires"):
        inner_dt, ratio = cfl_condition([0.01, 0.02, 0.04], dt, velocity)
    expected_max_dt = (
        1.0
        / math.sqrt(sum(1.0 / spacing**2 for spacing in (0.01, 0.02, 0.04)))
        / velocity
    )
    expected_ratio = math.ceil(dt / expected_max_dt)
    assert ratio == expected_ratio
    assert inner_dt == pytest.approx(dt / expected_ratio)


def _tm_source_field(grid_spacing: list[float]) -> torch.Tensor:
    dtype = torch.float64
    epsilon = torch.full((12, 14), 4.0, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source = torch.ones((1, 1, 1), dtype=dtype)
    source_location = torch.tensor([[[6, 7]]], dtype=torch.long)
    return tide.maxwell._kernel_api.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing,
        1.0e-11,
        source,
        source_location,
        None,
        nt=1,
        pml_width=0,
        python_backend=True,
    )[0]


def _em3d_source_field(grid_spacing: list[float]) -> torch.Tensor:
    dtype = torch.float64
    epsilon = torch.full((8, 9, 10), 4.0, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source = torch.ones((1, 1, 1), dtype=dtype)
    source_location = torch.tensor([[[4, 4, 5]]], dtype=torch.long)
    return tide.maxwell._kernel_api.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing,
        1.0e-11,
        source,
        source_location,
        None,
        nt=1,
        pml_width=0,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[1]


@pytest.mark.numerical
@pytest.mark.parametrize(
    "runner,coarse,fine",
    [
        (_tm_source_field, [0.04, 0.02], [0.02, 0.02]),
        (_em3d_source_field, [0.04, 0.02, 0.02], [0.02, 0.02, 0.02]),
    ],
    ids=["tm2d-area", "maxwell3d-volume"],
)
def test_source_injection_uses_cell_measure(runner, coarse, fine) -> None:
    coarse_peak = float(runner(coarse).abs().max())
    fine_peak = float(runner(fine).abs().max())
    assert coarse_peak > 0.0
    assert fine_peak / coarse_peak == pytest.approx(2.0, rel=1.0e-12)


@pytest.mark.numerical
@pytest.mark.parametrize("dimension", [2, 3])
def test_anisotropic_forward_and_material_gradients_are_finite(dimension: int) -> None:
    dtype = torch.float64
    nt = 36
    if dimension == 2:
        shape = (16, 18)
        spacing = [0.018, 0.022]
        source_location = torch.tensor([[[8, 6]]], dtype=torch.long)
        receiver_location = torch.tensor([[[8, 12]]], dtype=torch.long)
        solver = tide.maxwell._kernel_api.maxwelltm
        component_kwargs = {}
    else:
        shape = (8, 9, 10)
        spacing = [0.016, 0.018, 0.022]
        source_location = torch.tensor([[[4, 4, 3]]], dtype=torch.long)
        receiver_location = torch.tensor([[[4, 4, 7]]], dtype=torch.long)
        solver = tide.maxwell._kernel_api.maxwell3d
        component_kwargs = {"source_component": "ey", "receiver_component": "ey"}

    epsilon = torch.full(shape, 4.0, dtype=dtype, requires_grad=True)
    sigma = torch.full(shape, 2.0e-4, dtype=dtype, requires_grad=True)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(450e6, nt, 2.0e-11, peak_time=5.0e-10, dtype=dtype).view(
        1, 1, nt
    )
    receiver = solver(
        epsilon,
        sigma,
        mu,
        spacing,
        2.0e-11,
        source,
        source_location,
        receiver_location,
        stencil=4,
        pml_width=3,
        python_backend=False,
        storage_compression=False,
        **component_kwargs,
    )[-1]
    receiver.square().sum().backward()
    assert epsilon.grad is not None and sigma.grad is not None
    assert_finite_nonzero(receiver, epsilon.grad, sigma.grad)


@pytest.mark.numerical
def test_scalar_and_equal_axis_spacing_match_exactly(
    tm2d_example: MaxwellExample,
) -> None:
    scalar = tm2d_example.run(grid_spacing=0.02)[-1]
    sequence = tm2d_example.run(grid_spacing=[0.02, 0.02])[-1]
    assert relative_l2(sequence, scalar) == 0.0
