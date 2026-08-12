"""Numerical parity between the native CPU backend and the Python reference.

These tests gate the native C/C++ CPU path (built by CMake when CUDA is
absent) against the Python reference for forward and Born operators in 2D and
3D, plus the policy routing that must happen before any adapter runs.
"""

import pytest
import torch

from tide import backend_utils
from numerical_utils import MaxwellExample, make_maxwell3d_example, make_tm2d_example


def _requires_native_backend() -> None:
    if not backend_utils.is_backend_available():
        pytest.skip("native backend is not available")


def _tm2d_example(device: torch.device) -> MaxwellExample:
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


def _maxwell3d_example(device: torch.device) -> MaxwellExample:
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
    example = _tm2d_example(torch.device("cpu"))
    reference = example.run(python_backend=True)
    actual = example.run(python_backend=False)
    for actual_value, reference_value in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_value, reference_value, atol=1e-8, rtol=1e-8)


def test_native_cpu_tm2d_born_matches_python() -> None:
    _requires_native_backend()
    example = _tm2d_example(torch.device("cpu"))
    depsilon = 0.03 * torch.randn_like(example.epsilon)
    reference = example.run_born(depsilon=depsilon, python_backend=True)
    actual = example.run_born(depsilon=depsilon, python_backend=False)
    for actual_value, reference_value in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_value, reference_value, atol=1e-8, rtol=1e-8)


def test_native_cpu_em3d_forward_matches_python() -> None:
    _requires_native_backend()
    example = _maxwell3d_example(torch.device("cpu"))
    reference = example.run(python_backend=True)
    actual = example.run(python_backend=False)
    torch.testing.assert_close(actual[-1], reference[-1], atol=1e-4, rtol=1e-4)


def test_native_cpu_em3d_born_matches_python() -> None:
    _requires_native_backend()
    example = _maxwell3d_example(torch.device("cpu"))
    depsilon = 0.03 * torch.randn_like(example.epsilon)
    reference = example.run_born(depsilon=depsilon, python_backend=True)
    actual = example.run_born(depsilon=depsilon, python_backend=False)
    torch.testing.assert_close(actual[-1], reference[-1], atol=1e-4, rtol=1e-4)


def test_native_cpu_born_model_grads_with_none_storage_route_to_python() -> None:
    _requires_native_backend()
    example = _tm2d_example(torch.device("cpu"))
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
