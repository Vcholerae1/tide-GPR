import torch

from numerical_utils import MaxwellExample, make_maxwell3d_example


def _example() -> MaxwellExample:
    return make_maxwell3d_example(
        shape=(6, 6, 7),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=90e6,
        sigma=1e-4,
        source_location=(2, 2, 2),
        receiver_locations=((2, 2, 4),),
        pml_width=2,
    )


def _epsilon_gradient(
    example: MaxwellExample,
    *,
    python_backend: bool,
    **overrides: object,
) -> torch.Tensor:
    epsilon = example.epsilon.clone().requires_grad_(True)
    receiver = example.run(
        epsilon=epsilon,
        python_backend=python_backend,
        **overrides,
    )[-1]
    receiver.pow(2).sum().backward()
    assert epsilon.grad is not None
    return epsilon.grad


def test_maxwell3d_backend_gradient_matches_python():
    example = _example()
    reference = _epsilon_gradient(example, python_backend=True)
    actual = _epsilon_gradient(example, python_backend=False)
    torch.testing.assert_close(reference, actual, rtol=2e-4, atol=1e-3)


def test_maxwell3d_backend_shared_model_multishot_gradient_matches_shot_sum():
    example = _example()
    source_location = torch.tensor([[[2, 2, 2]], [[2, 3, 2]]], dtype=torch.long)
    receiver_location = torch.tensor([[[2, 2, 4]], [[2, 3, 4]]], dtype=torch.long)
    source_amplitude = example.source_amplitude.repeat(2, 1, 1)
    source_amplitude[1] *= 0.7

    shared_gradient = _epsilon_gradient(
        example,
        python_backend=False,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
    )
    gradient_sum = torch.zeros_like(example.epsilon)
    for shot_idx in range(source_amplitude.shape[0]):
        gradient_sum += _epsilon_gradient(
            example,
            python_backend=False,
            source_amplitude=source_amplitude[shot_idx : shot_idx + 1],
            source_location=source_location[shot_idx : shot_idx + 1],
            receiver_location=receiver_location[shot_idx : shot_idx + 1],
        )

    torch.testing.assert_close(
        shared_gradient,
        gradient_sum,
        rtol=2e-4,
        atol=1e-3,
    )
