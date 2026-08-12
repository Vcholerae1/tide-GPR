import pytest
import torch

from numerical_utils import (
    deterministic_direction,
    directional_derivative_errors,
    make_maxwell3d_example,
    taylor_remainders,
)


def _example(device: torch.device = torch.device("cpu")):
    return make_maxwell3d_example(
        shape=(6, 7, 8),
        nt=10,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        frequency=80e6,
        device=device,
        sigma=2e-4,
        source_location=(2, 3, 2),
        receiver_locations=((2, 3, 5),),
        pml_width=2,
        python_backend=True,
    )


def test_maxwell3d_epsilon_gradient_finite_difference():
    example = _example()
    h = 1e-2
    epsilon = example.epsilon.clone().requires_grad_(True)
    receiver = example.run(epsilon=epsilon)[-1]
    loss = receiver.pow(2).sum()
    loss.backward()
    assert epsilon.grad is not None

    index = (3, 3, 4)
    perturbed = example.epsilon.clone()
    perturbed[index] += h
    perturbed_receiver = example.run(epsilon=perturbed)[-1]
    finite_difference = (perturbed_receiver.pow(2).sum() - loss.detach()) / h
    gradient = epsilon.grad[index]

    assert torch.sign(gradient) == torch.sign(finite_difference)
    relative_error = abs(gradient - finite_difference) / (
        abs(finite_difference) + 1e-10
    )
    assert relative_error < 0.7


def test_maxwell3d_sigma_gradient_nonzero():
    example = _example()
    sigma = example.sigma.clone().requires_grad_(True)
    receiver = example.run(sigma=sigma)[-1]
    receiver.pow(2).sum().backward()
    assert sigma.grad is not None
    assert torch.isfinite(sigma.grad).all()
    assert sigma.grad.abs().sum() > 0


def _maxwell3d_directional_metrics(
    parameter: str, stencil: int, *, python_backend: bool
) -> tuple[list[float], list[float], list[float]]:
    dtype = torch.float64
    example = make_maxwell3d_example(
        shape=(9, 10, 11),
        nt=45,
        grid_spacing=[0.016, 0.018, 0.022],
        dt=2.0e-11,
        frequency=500e6,
        peak_time=6.0e-10,
        dtype=dtype,
        sigma=2.0e-4,
        source_location=(4, 5, 4),
        receiver_locations=((4, 5, 7), (5, 7, 7)),
        pml_width=4,
        stencil=stencil,
        python_backend=python_backend,
    )
    residual = torch.linspace(-0.6, 1.0, 45, dtype=dtype).view(45, 1, 1)

    def objective(value: torch.Tensor) -> torch.Tensor:
        receiver = example.run(
            epsilon=value if parameter == "epsilon" else example.epsilon,
            sigma=value if parameter == "sigma" else example.sigma,
            storage_compression=False,
        )[-1]
        return (receiver * residual).sum()

    base = (
        (example.epsilon if parameter == "epsilon" else example.sigma)
        .clone()
        .requires_grad_(True)
    )
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
