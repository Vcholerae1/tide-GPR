import pytest
import torch

import tide
from numerical_utils import (
    deterministic_direction,
    directional_derivative_errors,
    taylor_remainders,
)


def _setup_case(device: torch.device):
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 10

    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.ones_like(epsilon) * 2e-4
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 5]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def test_maxwell3d_epsilon_gradient_finite_difference():
    device = torch.device("cpu")
    h = 1e-2
    (
        epsilon,
        sigma,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
    ) = _setup_case(device)

    eps_base = epsilon.clone().detach().requires_grad_(True)
    out_base = tide.maxwell3d(
        eps_base,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[-1]
    loss_base = out_base.pow(2).sum()
    loss_base.backward()
    assert eps_base.grad is not None

    iz, iy, ix = 3, 3, 4
    eps_pert = epsilon.clone()
    eps_pert[iz, iy, ix] += h
    out_pert = tide.maxwell3d(
        eps_pert,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[-1]
    fd_approx = (out_pert.pow(2).sum() - loss_base.detach()) / h
    grad_at_point = eps_base.grad[iz, iy, ix]

    assert torch.sign(grad_at_point) == torch.sign(fd_approx)
    rel_error = abs(grad_at_point - fd_approx) / (abs(fd_approx) + 1e-10)
    assert rel_error < 0.7


def test_maxwell3d_sigma_gradient_nonzero():
    device = torch.device("cpu")
    (
        epsilon,
        sigma,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
    ) = _setup_case(device)
    sigma = sigma.clone().detach().requires_grad_(True)

    out = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[-1]
    loss = out.pow(2).sum()
    loss.backward()
    assert sigma.grad is not None
    assert torch.isfinite(sigma.grad).all()
    assert sigma.grad.abs().sum() > 0


def _maxwell3d_directional_metrics(
    parameter: str, stencil: int, *, python_backend: bool
) -> tuple[list[float], list[float], list[float]]:
    dtype = torch.float64
    nz, ny, nx, nt = 9, 10, 11, 45
    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype)
    sigma = torch.full_like(epsilon, 2.0e-4)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(500e6, nt, 2.0e-11, peak_time=6.0e-10, dtype=dtype).view(
        1, 1, nt
    )
    source_location = torch.tensor([[[4, 5, 4]]], dtype=torch.long)
    receiver_location = torch.tensor([[[4, 5, 7], [5, 7, 7]]], dtype=torch.long)
    residual = torch.linspace(-0.6, 1.0, nt, dtype=dtype).view(nt, 1, 1)

    def objective(value: torch.Tensor) -> torch.Tensor:
        epsilon_i = value if parameter == "epsilon" else epsilon
        sigma_i = value if parameter == "sigma" else sigma
        receiver = tide.maxwell3d(
            epsilon_i,
            sigma_i,
            mu,
            [0.016, 0.018, 0.022],
            2.0e-11,
            source,
            source_location,
            receiver_location,
            source_component="ey",
            receiver_component="ey",
            stencil=stencil,
            pml_width=4,
            python_backend=python_backend,
            storage_compression=False,
        )[-1]
        return (receiver * residual).sum()

    base = (epsilon if parameter == "epsilon" else sigma).clone().requires_grad_(True)
    loss = objective(base)
    (gradient,) = torch.autograd.grad(loss, base)
    direction = deterministic_direction(
        base.shape,
        seed=9100 + stencil,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    scale = 1.0e-2 if parameter == "epsilon" else 1.0e-5
    steps = (scale, scale / 2.0, scale / 4.0)
    errors = directional_derivative_errors(
        objective,
        base.detach(),
        direction,
        gradient,
        steps,
    )
    zero_order, first_order = taylor_remainders(
        objective,
        base.detach(),
        direction,
        gradient,
        steps,
        base_value=loss,
    )
    return errors, zero_order, first_order


@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("parameter", ["epsilon", "sigma"])
def test_maxwell3d_native_directional_derivative(parameter: str, stencil: int) -> None:
    errors, zero_order, first_order = _maxwell3d_directional_metrics(
        parameter, stencil, python_backend=False
    )
    assert min(errors) < 1.0e-3, errors
    assert first_order[-1] < zero_order[-1], (zero_order, first_order)


@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("parameter", ["epsilon", "sigma"])
def test_maxwell3d_reference_gradient_has_second_order_taylor_remainder(
    parameter: str, stencil: int
) -> None:
    errors, zero_order, first_order = _maxwell3d_directional_metrics(
        parameter, stencil, python_backend=True
    )
    assert min(errors) < 1.0e-5, errors
    assert first_order[1] < 0.4 * first_order[0], (zero_order, first_order)
    assert first_order[2] < 0.4 * first_order[1], (zero_order, first_order)
    assert first_order[-1] < zero_order[-1], (zero_order, first_order)
