from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import tide
from tide import staggered

from numerical_utils import relative_l2, require_native_backend


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
    ny, nx, nt = 18, 20, 48
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype)
    sigma = torch.full_like(epsilon, 1.0e-4)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(400e6, nt, 2.0e-11, peak_time=6.0e-10, dtype=dtype).view(
        1, 1, nt
    )
    source_location = torch.tensor([[[9, 6]]], dtype=torch.long)
    receiver_location = torch.tensor([[[9, 12], [11, 12]]], dtype=torch.long)
    generator = torch.Generator().manual_seed(5200 + stencil + pml_width)
    depsilon = 0.02 * torch.randn(epsilon.shape, generator=generator, dtype=dtype)
    dsigma = 2.0e-5 * torch.randn(sigma.shape, generator=generator, dtype=dtype)
    data_weight = torch.randn((nt, 1, 2), generator=generator, dtype=dtype)

    born_data = tide.borntm(
        epsilon,
        sigma,
        mu,
        [0.018, 0.022],
        2.0e-11,
        source,
        source_location,
        receiver_location,
        depsilon=depsilon,
        dsigma=dsigma,
        stencil=stencil,
        pml_width=pml_width,
        linearize_source=True,
        python_backend=False,
        storage_compression=False,
    )[-1]
    lhs = torch.sum(born_data * data_weight)

    epsilon_req = epsilon.clone().requires_grad_(True)
    sigma_req = sigma.clone().requires_grad_(True)
    predicted = tide.maxwelltm(
        epsilon_req,
        sigma_req,
        mu,
        [0.018, 0.022],
        2.0e-11,
        source,
        source_location,
        receiver_location,
        stencil=stencil,
        pml_width=pml_width,
        python_backend=False,
        storage_compression=False,
    )[-1]
    gradient_epsilon, gradient_sigma = torch.autograd.grad(
        torch.sum(predicted * data_weight),
        (epsilon_req, sigma_req),
    )
    rhs = torch.sum(depsilon * gradient_epsilon + dsigma * gradient_sigma)
    assert relative_l2(lhs.reshape(1), rhs.reshape(1)) < 5.0e-3
