from __future__ import annotations

import pytest
import torch
from collections.abc import Callable
from numerical_utils import (
    MaxwellExample,
    make_maxwell3d_example,
    make_tm2d_example,
    cosine_similarity,
    relative_l2,
    require_cuda_backend,
)
from tide import backend_utils

# --- test_native_cpu_parity.py ---

"""Numerical parity between the native CPU backend and the Python reference.

These tests gate the native C/C++ CPU path (built by CMake when CUDA is
absent) against the Python reference for forward and Born operators in 2D and
3D, plus the policy routing that must happen before any adapter runs.
"""


def _requires_native_backend() -> None:
    if not backend_utils.is_backend_available():
        pytest.skip("native backend is not available")


def _native_tm2d_example(device: torch.device) -> MaxwellExample:
    example = make_tm2d_example(
        shape=(12, 12),
        nt=16,
        grid_spacing=0.02,
        dt=3.5e-11,
        frequency=120e6,
        dtype=torch.float64,
        device=device,
        source_location=(6, 3),
        receiver_locations=((6, 6),),
        pml_width=3,
    )
    epsilon = example.epsilon.clone()
    epsilon[5:7, 5:7] = 4.4
    return example.updated(epsilon=epsilon)


def _native_maxwell3d_example(device: torch.device) -> MaxwellExample:
    return make_maxwell3d_example(
        shape=(6, 6, 7),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=90e6,
        device=device,
        source_location=(2, 2, 2),
        receiver_locations=((2, 2, 4),),
        pml_width=1,
    )


def test_native_cpu_tm2d_forward_matches_python() -> None:
    _requires_native_backend()
    example = _native_tm2d_example(torch.device("cpu"))
    reference = example.run(python_backend=True)
    actual = example.run(python_backend=False)
    for actual_value, reference_value in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_value, reference_value, atol=1e-8, rtol=1e-8)


def test_native_cpu_em3d_forward_matches_python() -> None:
    _requires_native_backend()
    example = _native_maxwell3d_example(torch.device("cpu"))
    reference = example.run(python_backend=True)
    actual = example.run(python_backend=False)
    torch.testing.assert_close(actual[-1], reference[-1], atol=1e-4, rtol=1e-4)


def test_native_cpu_em3d_born_matches_python() -> None:
    _requires_native_backend()
    example = _native_maxwell3d_example(torch.device("cpu"))
    depsilon = 0.03 * torch.randn_like(example.epsilon)
    reference = example.run_born(depsilon=depsilon, python_backend=True)
    actual = example.run_born(depsilon=depsilon, python_backend=False)
    torch.testing.assert_close(actual[-1], reference[-1], atol=1e-4, rtol=1e-4)


def test_native_cpu_born_model_grads_with_none_storage_route_to_python() -> None:
    _requires_native_backend()
    example = _native_tm2d_example(torch.device("cpu"))
    epsilon = example.epsilon.clone().requires_grad_(True)
    output = example.run_born(
        epsilon=epsilon,
        storage_mode="none",
        fallback="reference",
    )
    gradient = torch.autograd.grad(output[-1].square().sum(), epsilon)[0]
    assert torch.isfinite(gradient).all()

    with pytest.raises(NotImplementedError, match="storage_mode='none'"):
        example.run_born(
            epsilon=example.epsilon.clone().requires_grad_(True),
            storage_mode="none",
            fallback="error",
        )


# --- test_maxwell3d_backend_parity.py ---


def _example(device: torch.device):
    return make_maxwell3d_example(
        shape=(6, 6, 7),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=90e6,
        device=device,
        source_location=(2, 2, 2),
        receiver_locations=((2, 2, 4),),
        pml_width=2,
    )


def test_maxwell3d_backend_parity_via_fallback():
    example = _example(torch.device("cpu"))
    out_python = example.run(python_backend=True)
    out_backend = example.run(python_backend=False)
    for actual, reference in zip(out_backend, out_python, strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("n_threads", [0, 128, 256])
def test_maxwell3d_native_cuda_matches_python_without_callback(n_threads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D CUDA parity test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D CUDA parity test.")

    example = _example(torch.device("cuda"))
    out_python = example.run(python_backend=True)
    out_backend = example.run(python_backend=False, n_threads=n_threads)
    torch.testing.assert_close(
        out_backend[-1],
        out_python[-1],
        atol=1e-4,
        rtol=1e-4,
    )


# --- test_maxwell3d_backend_gradients.py ---


def _gradient_example() -> MaxwellExample:
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
    example = _gradient_example()
    reference = _epsilon_gradient(example, python_backend=True)
    actual = _epsilon_gradient(example, python_backend=False)
    torch.testing.assert_close(reference, actual, rtol=2e-4, atol=1e-3)


def test_maxwell3d_backend_shared_model_multishot_gradient_matches_shot_sum():
    example = _gradient_example()
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


# --- test_backend_parity_matrix.py ---


def _matrix_tm2d_example(device: torch.device) -> MaxwellExample:
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


def _matrix_maxwell3d_example(device: torch.device) -> MaxwellExample:
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
    [_matrix_tm2d_example, _matrix_maxwell3d_example],
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
