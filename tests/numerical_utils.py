from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence

import pytest
import torch

from tide import backend_utils


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    numerator = torch.linalg.vector_norm(actual64 - reference64)
    denominator = torch.linalg.vector_norm(reference64).clamp_min(1.0e-30)
    return float((numerator / denominator).item())


def cosine_similarity(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = (
        torch.linalg.vector_norm(actual64) * torch.linalg.vector_norm(reference64)
    ).clamp_min(1.0e-30)
    return float(torch.dot(actual64, reference64).div(denominator).item())


def signal_rms(value: torch.Tensor) -> float:
    value64 = value.detach().to(device="cpu", dtype=torch.float64)
    return float(torch.sqrt(torch.mean(value64.square())).item())


def assert_finite_nonzero(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        assert tensor is not None
        assert bool(torch.isfinite(tensor).all())
        assert float(tensor.detach().abs().max()) > 0.0


def deterministic_direction(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(tuple(shape), generator=generator, dtype=torch.float64)
    direction = direction.to(device=device, dtype=dtype)
    if mask is not None:
        direction = torch.where(mask, direction, torch.zeros_like(direction))
    norm = torch.linalg.vector_norm(direction.to(torch.float64)).clamp_min(1.0e-30)
    return direction / norm.to(direction.dtype)


def directional_derivative_errors(
    objective: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    direction: torch.Tensor,
    gradient: torch.Tensor,
    steps: Iterable[float],
) -> list[float]:
    adjoint = float(
        (gradient.detach().to(torch.float64) * direction.to(torch.float64)).sum()
    )
    errors: list[float] = []
    for step in steps:
        with torch.no_grad():
            finite_difference = float(
                (
                    objective(base + step * direction)
                    - objective(base - step * direction)
                )
                / (2.0 * step)
            )
        denominator = max(abs(adjoint), abs(finite_difference), 1.0e-30)
        errors.append(abs(adjoint - finite_difference) / denominator)
    return errors


def taylor_remainders(
    objective: Callable[[torch.Tensor], torch.Tensor],
    base: torch.Tensor,
    direction: torch.Tensor,
    gradient: torch.Tensor,
    steps: Iterable[float],
    *,
    base_value: torch.Tensor | None = None,
) -> tuple[list[float], list[float]]:
    if base_value is None:
        with torch.no_grad():
            base_value = objective(base)
    base_scalar = float(base_value.detach())
    directional_derivative = float(
        (gradient.detach().to(torch.float64) * direction.to(torch.float64)).sum()
    )
    zero_order: list[float] = []
    first_order: list[float] = []
    for step in steps:
        with torch.no_grad():
            perturbed = float(objective(base + step * direction))
        difference = perturbed - base_scalar
        zero_order.append(abs(difference))
        first_order.append(abs(difference - step * directional_derivative))
    return zero_order, first_order


def require_native_backend() -> None:
    if not backend_utils.is_backend_available():
        pytest.skip("native backend is unavailable")


def require_cuda_backend() -> torch.device:
    required = os.environ.get("TIDE_REQUIRE_CUDA", "0") == "1"
    available = torch.cuda.is_available() and backend_utils.is_backend_available()
    if not available and required:
        pytest.fail("TIDE_REQUIRE_CUDA=1 but a native CUDA backend is unavailable")
    if not available:
        pytest.skip("native CUDA backend is unavailable")
    device = torch.device(os.environ.get("TIDE_TEST_CUDA_DEVICE", "cuda:0"))
    torch.cuda.set_device(device)
    return device
