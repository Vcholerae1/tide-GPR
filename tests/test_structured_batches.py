import pytest
import torch
from jaxtyping import TypeCheckError

import tide
from tide import backend_utils
from numerical_utils import MaxwellExample, make_maxwell3d_example, make_tm2d_example


def _skip_if_no_backend() -> None:
    if not backend_utils.is_backend_available():
        pytest.skip("native backend unavailable")


def _tm_example(device: torch.device) -> MaxwellExample:
    example = make_tm2d_example(
        shape=(8, 9),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=80e6,
        device=device,
        source_location=(4, 4),
        receiver_locations=((4, 6),),
        pml_width=2,
        stencil=2,
    )
    return example.updated(
        epsilon=torch.stack((example.epsilon, example.epsilon + 0.5)),
        sigma=torch.stack(
            (
                torch.full_like(example.sigma, 1e-3),
                torch.full_like(example.sigma, 2e-3),
            )
        ),
        mu=torch.stack((example.mu, example.mu)),
        source_amplitude=example.source_amplitude.repeat(2, 1, 1),
        source_location=torch.tensor(
            [[[4, 4]], [[4, 5]]],
            device=device,
        ),
        receiver_location=torch.tensor(
            [[[4, 6]], [[4, 7]]],
            device=device,
        ),
    )


def _maxwell3d_example(device: torch.device) -> MaxwellExample:
    example = make_maxwell3d_example(
        shape=(5, 6, 7),
        nt=8,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=70e6,
        device=device,
        source_location=(2, 2, 2),
        receiver_locations=((2, 2, 4),),
        pml_width=1,
    )
    return example.updated(
        epsilon=torch.stack((example.epsilon, example.epsilon + 0.5)),
        sigma=torch.stack(
            (
                torch.full_like(example.sigma, 1e-3),
                torch.full_like(example.sigma, 2e-3),
            )
        ),
        mu=torch.stack((example.mu, example.mu)),
        source_amplitude=example.source_amplitude.repeat(2, 1, 1),
        source_location=torch.tensor(
            [[[2, 2, 2]], [[2, 3, 2]]],
            device=device,
        ),
        receiver_location=torch.tensor(
            [[[2, 2, 4]], [[2, 3, 4]]],
            device=device,
        ),
    )


def _assert_batched_forward_matches_loop(
    example: MaxwellExample,
    *,
    python_backend: bool,
) -> None:
    output = example.run(python_backend=python_backend)
    expected_receiver = torch.stack(
        [
            example.run(
                epsilon=example.epsilon[index],
                sigma=example.sigma[index],
                mu=example.mu[index],
                python_backend=python_backend,
            )[-1]
            for index in range(example.epsilon.shape[0])
        ],
        dim=1,
    )
    assert output[-1].shape == (
        example.source_amplitude.shape[-1],
        example.epsilon.shape[0],
        example.source_amplitude.shape[0],
        example.receiver_location.shape[1],
    )
    assert output[0].shape[:2] == (
        example.epsilon.shape[0],
        example.source_amplitude.shape[0],
    )
    torch.testing.assert_close(output[-1], expected_receiver)


def _assert_batched_backward_matches_loop(
    example: MaxwellExample,
    *,
    python_backend: bool,
) -> None:
    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    example.run(
        epsilon=epsilon,
        sigma=sigma,
        python_backend=python_backend,
    )[-1].square().sum().backward()
    assert epsilon.grad is not None

    gradients = []
    for index in range(epsilon.shape[0]):
        epsilon_i = epsilon.detach()[index].clone().requires_grad_(True)
        sigma_i = sigma.detach()[index].clone().requires_grad_(True)
        example.run(
            epsilon=epsilon_i,
            sigma=sigma_i,
            mu=example.mu[index],
            python_backend=python_backend,
        )[-1].square().sum().backward()
        assert epsilon_i.grad is not None
        gradients.append(epsilon_i.grad)
    torch.testing.assert_close(epsilon.grad, torch.stack(gradients))


def test_maxwelltm_batched_models_shared_shots_forward_matches_loop():
    _skip_if_no_backend()
    _assert_batched_forward_matches_loop(
        _tm_example(torch.device("cpu")),
        python_backend=False,
    )


def test_maxwelltm_batched_models_per_model_shots_backward_matches_loop():
    _skip_if_no_backend()
    example = _tm_example(torch.device("cpu"))
    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    source_amplitude = (
        example.source_amplitude.unsqueeze(0).expand(2, -1, -1, -1).clone()
    )
    source_amplitude[1] *= 0.75
    source_location = example.source_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    receiver_location = (
        example.receiver_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    )
    receiver_location[1, :, 0, 1] -= 1

    receivers = example.run(
        epsilon=epsilon,
        sigma=sigma,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        python_backend=False,
    )[-1]
    receivers.square().sum().backward()
    assert epsilon.grad is not None

    gradients = []
    for index in range(epsilon.shape[0]):
        epsilon_i = epsilon.detach()[index].clone().requires_grad_(True)
        sigma_i = sigma.detach()[index].clone().requires_grad_(True)
        receiver_i = example.run(
            epsilon=epsilon_i,
            sigma=sigma_i,
            mu=example.mu[index],
            source_amplitude=source_amplitude[index],
            source_location=source_location[index],
            receiver_location=receiver_location[index],
            python_backend=False,
        )[-1]
        receiver_i.square().sum().backward()
        assert epsilon_i.grad is not None
        gradients.append(epsilon_i.grad)

    torch.testing.assert_close(
        epsilon.grad,
        torch.stack(gradients),
        atol=2e-5,
        rtol=1e-5,
        equal_nan=True,
    )


def test_maxwelltm_batched_model_callbacks_expose_structured_shapes():
    _skip_if_no_backend()
    example = _tm_example(torch.device("cpu"))
    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    seen: dict[str, tuple[int, ...]] = {}

    def forward_cb(state: tide.CallbackState) -> None:
        if "forward_wavefield" not in seen:
            seen["forward_wavefield"] = tuple(state.get_wavefield("Ey").shape)
            seen["forward_model"] = tuple(state.get_model("epsilon").shape)

    def backward_cb(state: tide.CallbackState) -> None:
        if "backward_gradient" not in seen:
            seen["backward_gradient"] = tuple(state.get_gradient("epsilon").shape)

    receivers = example.run(
        epsilon=epsilon,
        sigma=sigma,
        python_backend=False,
        forward_callback=forward_cb,
        backward_callback=backward_cb,
    )[-1]
    receivers.square().sum().backward()

    assert seen["forward_wavefield"] == (2, 2, 8, 9)
    assert seen["forward_model"] == (2, 8, 9)
    assert seen["backward_gradient"] == (2, 8, 9)


def test_maxwell3d_batched_models_shared_shots_forward_matches_loop():
    _skip_if_no_backend()
    _assert_batched_forward_matches_loop(
        _maxwell3d_example(torch.device("cpu")),
        python_backend=False,
    )


def test_maxwell3d_batched_models_shared_shots_backward_matches_loop():
    _skip_if_no_backend()
    _assert_batched_backward_matches_loop(
        _maxwell3d_example(torch.device("cpu")),
        python_backend=False,
    )


def test_maxwelltm_batched_models_python_backend_forward_matches_loop():
    _assert_batched_forward_matches_loop(
        _tm_example(torch.device("cpu")),
        python_backend=True,
    )


def test_maxwell3d_batched_models_python_backend_backward_matches_loop():
    _assert_batched_backward_matches_loop(
        _maxwell3d_example(torch.device("cpu")),
        python_backend=True,
    )


def test_batched_models_python_backend_callbacks_rejected():
    device = torch.device("cpu")
    for example in (_tm_example(device), _maxwell3d_example(device)):
        with pytest.raises(NotImplementedError):
            example.run(
                python_backend=True,
                forward_callback=lambda state: None,
            )


def test_batched_models_validate_B_and_S_mismatch():
    example = _tm_example(torch.device("cpu"))
    bad_source = example.source_amplitude.unsqueeze(0).expand(3, -1, -1, -1).clone()
    with pytest.raises((RuntimeError, TypeCheckError)):
        example.run(
            source_amplitude=bad_source,
            python_backend=False,
        )

    bad_receiver = example.receiver_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    bad_receiver = bad_receiver[:, :1]
    with pytest.raises((RuntimeError, TypeCheckError)):
        example.run(
            receiver_location=bad_receiver,
            python_backend=False,
        )
