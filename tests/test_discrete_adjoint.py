from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from tide import staggered

from numerical_utils import make_tm2d_example, relative_l2, require_native_backend


@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize(
    "forward,transpose,shape",
    [
        (staggered.diffx1, staggered.diffxh1, (2, 30, 31)),
        (staggered.diffy1, staggered.diffyh1, (2, 30, 31)),
        (staggered.diffz1, staggered.diffzh1, (2, 18, 19, 20)),
    ],
)
def test_staggered_derivatives_are_negative_adjoints(
    stencil: int,
    forward: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor],
    transpose: Callable[[torch.Tensor, int, torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
) -> None:
    generator = torch.Generator().manual_seed(4100 + stencil + len(shape))
    left = torch.randn(shape, generator=generator, dtype=torch.float64)
    right = torch.randn(shape, generator=generator, dtype=torch.float64)
    halo = stencil // 2
    interior = tuple(slice(halo, -halo) for _ in shape[1:])
    boundary_mask = torch.zeros_like(left, dtype=torch.bool)
    boundary_mask[(slice(None), *interior)] = True
    left = torch.where(boundary_mask, left, torch.zeros_like(left))
    right = torch.where(boundary_mask, right, torch.zeros_like(right))
    reciprocal_spacing = torch.tensor(2.5, dtype=torch.float64)

    lhs = torch.sum(forward(left, stencil, reciprocal_spacing) * right)
    rhs = -torch.sum(left * transpose(right, stencil, reciprocal_spacing))
    torch.testing.assert_close(lhs, rhs, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("pml_width", [0, 4])
def test_tm2d_born_and_model_vjp_are_adjoint(stencil: int, pml_width: int) -> None:
    require_native_backend()
    dtype = torch.float64
    example = make_tm2d_example(
        shape=(18, 20),
        nt=48,
        grid_spacing=[0.018, 0.022],
        dt=2.0e-11,
        frequency=400e6,
        peak_time=6.0e-10,
        dtype=dtype,
        sigma=1.0e-4,
        source_location=(9, 6),
        receiver_locations=((9, 12), (11, 12)),
        pml_width=pml_width,
        stencil=stencil,
    )
    generator = torch.Generator().manual_seed(5200 + stencil + pml_width)
    depsilon = 0.02 * torch.randn(
        example.epsilon.shape, generator=generator, dtype=dtype
    )
    dsigma = 2.0e-5 * torch.randn(example.sigma.shape, generator=generator, dtype=dtype)
    data_weight = torch.randn((48, 1, 2), generator=generator, dtype=dtype)

    born_data = example.run_born(
        depsilon=depsilon,
        dsigma=dsigma,
        linearize_source=True,
        python_backend=False,
        storage_compression=False,
    )[-1]
    lhs = torch.sum(born_data * data_weight)

    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    predicted = example.run(
        epsilon=epsilon,
        sigma=sigma,
        python_backend=False,
        storage_compression=False,
    )[-1]
    gradient_epsilon, gradient_sigma = torch.autograd.grad(
        torch.sum(predicted * data_weight),
        (epsilon, sigma),
    )
    rhs = torch.sum(depsilon * gradient_epsilon + dsigma * gradient_sigma)
    assert relative_l2(lhs.reshape(1), rhs.reshape(1)) < 5.0e-3
