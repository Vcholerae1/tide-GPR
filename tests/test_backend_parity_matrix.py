from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from numerical_utils import (
    MaxwellExample,
    cosine_similarity,
    make_maxwell3d_example,
    make_tm2d_example,
    relative_l2,
    require_cuda_backend,
)


def _tm2d_example(device: torch.device) -> MaxwellExample:
    return make_tm2d_example(
        shape=(18, 20),
        nt=52,
        grid_spacing=[0.018, 0.022],
        dt=2.0e-11,
        frequency=400e6,
        peak_time=7.0e-10,
        device=device,
        sigma=2.0e-4,
        source_location=(9, 6),
        receiver_locations=((9, 12), (11, 12)),
        pml_width=4,
    )


def _maxwell3d_example(device: torch.device) -> MaxwellExample:
    return make_maxwell3d_example(
        shape=(9, 10, 11),
        nt=40,
        grid_spacing=[0.016, 0.018, 0.022],
        dt=2.0e-11,
        frequency=500e6,
        peak_time=6.0e-10,
        device=device,
        sigma=2.0e-4,
        source_location=(4, 5, 4),
        receiver_locations=((4, 5, 7), (5, 7, 7)),
        pml_width=4,
    )


def _run(
    example: MaxwellExample,
    stencil: int,
    python_backend: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = example.epsilon.requires_grad_()
    sigma = example.sigma.requires_grad_()
    receiver = example.run(
        epsilon=epsilon,
        sigma=sigma,
        stencil=stencil,
        python_backend=python_backend,
        storage_compression=False,
    )[-1]
    residual = torch.linspace(
        -0.7 if example.epsilon.ndim == 2 else -0.6,
        1.1 if example.epsilon.ndim == 2 else 1.0,
        example.source_amplitude.shape[-1],
        device=example.epsilon.device,
        dtype=example.epsilon.dtype,
    ).view(-1, 1, 1)
    (receiver * residual).sum().backward()
    assert epsilon.grad is not None and sigma.grad is not None
    if example.epsilon.device.type == "cuda":
        torch.cuda.synchronize(example.epsilon.device)
    return (
        receiver.detach().cpu(),
        epsilon.grad.detach().cpu(),
        sigma.grad.detach().cpu(),
    )


@pytest.mark.cuda
@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize(
    "example_factory",
    [_tm2d_example, _maxwell3d_example],
    ids=["tm2d", "maxwell3d"],
)
def test_cuda_matches_cpu_reference(
    example_factory: Callable[[torch.device], MaxwellExample],
    stencil: int,
) -> None:
    cuda_device = require_cuda_backend()
    reference = _run(example_factory(torch.device("cpu")), stencil, True)
    actual = _run(example_factory(cuda_device), stencil, False)
    assert relative_l2(actual[0], reference[0]) < 2.0e-4
    for gradient, reference_gradient in zip(actual[1:], reference[1:], strict=True):
        assert relative_l2(gradient, reference_gradient) < 5.0e-3
        assert cosine_similarity(gradient, reference_gradient) > 0.999
