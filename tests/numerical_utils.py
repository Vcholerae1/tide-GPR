from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import pytest
import torch

import tide
from tide import backend_utils


@dataclass(frozen=True)
class MaxwellExample:
    """A complete, reusable Maxwell propagation example."""

    name: str
    solver: Callable[..., tuple[torch.Tensor, ...]]
    epsilon: torch.Tensor
    sigma: torch.Tensor
    mu: torch.Tensor
    grid_spacing: float | Sequence[float]
    dt: float
    source_amplitude: torch.Tensor
    source_location: torch.Tensor
    receiver_location: torch.Tensor
    pml_width: int | Sequence[int]
    solver_kwargs: Mapping[str, object] = field(default_factory=dict)
    born_solver: Callable[..., tuple[torch.Tensor, ...]] | None = None
    depsilon: torch.Tensor | None = None
    dsigma: torch.Tensor | None = None
    vepsilon: torch.Tensor | None = None
    vsigma: torch.Tensor | None = None
    observed_data: torch.Tensor | None = None

    def arguments(self, **overrides: object) -> dict[str, object]:
        """Build the complete solver argument set for this example."""
        return {
            "epsilon": self.epsilon,
            "sigma": self.sigma,
            "mu": self.mu,
            "grid_spacing": self.grid_spacing,
            "dt": self.dt,
            "source_amplitude": self.source_amplitude,
            "source_location": self.source_location,
            "receiver_location": self.receiver_location,
            "pml_width": self.pml_width,
            **self.solver_kwargs,
            **overrides,
        }

    @property
    def stencil(self) -> int:
        return int(self.solver_kwargs["stencil"])

    @property
    def source_component(self) -> str:
        return str(self.solver_kwargs.get("source_component", ""))

    @property
    def receiver_component(self) -> str:
        return str(self.solver_kwargs.get("receiver_component", ""))

    def run(self, **overrides: object) -> tuple[torch.Tensor, ...]:
        """Run this example, optionally replacing any solver argument."""
        return self.solver(**self.arguments(**overrides))

    def run_born(self, **overrides: object) -> tuple[torch.Tensor, ...]:
        """Run the Born operator using this example's propagation setup."""
        if self.born_solver is None:
            raise ValueError(f"{self.name} does not define a Born solver")
        return self.born_solver(**self.arguments(**overrides))

    def updated(self, **changes: object) -> MaxwellExample:
        """Return an example with selected fields replaced."""
        return replace(self, **changes)

    def receiver_zeros(self) -> torch.Tensor:
        """Create receiver data with the shape produced by this example."""
        return torch.zeros(
            self.source_amplitude.shape[-1],
            self.source_amplitude.shape[0],
            self.receiver_location.shape[1],
            device=self.epsilon.device,
            dtype=self.epsilon.dtype,
        )

    def __str__(self) -> str:
        return self.name


def make_tm2d_example(
    *,
    shape: tuple[int, int],
    nt: int,
    grid_spacing: float | Sequence[float],
    dt: float,
    frequency: float,
    peak_time: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    epsilon: float = 4.0,
    sigma: float = 0.0,
    mu: float = 1.0,
    source_location: Sequence[int] | None = None,
    receiver_locations: Sequence[Sequence[int]] | None = None,
    pml_width: int | Sequence[int] = 0,
    stencil: int = 2,
    python_backend: bool | None = None,
    name: str = "tm2d",
) -> MaxwellExample:
    """Create a deterministic 2D Maxwell example."""
    device = torch.device(device)
    ny, nx = shape
    epsilon_tensor = torch.full(shape, epsilon, device=device, dtype=dtype)
    if source_location is None:
        source_location = (ny // 2, nx // 4)
    if receiver_locations is None:
        receiver_locations = ((ny // 2, nx // 2),)
    solver_kwargs: dict[str, object] = {"stencil": stencil}
    if python_backend is not None:
        solver_kwargs["python_backend"] = python_backend
    return MaxwellExample(
        name=name,
        solver=tide.maxwell._kernel_api.maxwelltm,
        born_solver=tide.maxwell._kernel_api.borntm,
        epsilon=epsilon_tensor,
        sigma=torch.full_like(epsilon_tensor, sigma),
        mu=torch.full_like(epsilon_tensor, mu),
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=tide.ricker(
            frequency,
            nt,
            dt,
            peak_time=1.0 / frequency if peak_time is None else peak_time,
            dtype=dtype,
            device=device,
        ).view(1, 1, nt),
        source_location=torch.tensor(
            [[source_location]],
            dtype=torch.long,
            device=device,
        ),
        receiver_location=torch.tensor(
            [receiver_locations],
            dtype=torch.long,
            device=device,
        ),
        pml_width=pml_width,
        solver_kwargs=solver_kwargs,
    )


def make_maxwell3d_example(
    *,
    shape: tuple[int, int, int],
    nt: int,
    grid_spacing: float | Sequence[float],
    dt: float,
    frequency: float,
    peak_time: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    epsilon: float = 4.0,
    sigma: float = 0.0,
    mu: float = 1.0,
    source_location: Sequence[int] | None = None,
    receiver_locations: Sequence[Sequence[int]] | None = None,
    pml_width: int | Sequence[int] = 0,
    stencil: int = 2,
    python_backend: bool | None = None,
    source_component: str = "ey",
    receiver_component: str = "ey",
    name: str = "maxwell3d",
) -> MaxwellExample:
    """Create a deterministic 3D Maxwell example."""
    device = torch.device(device)
    nz, ny, nx = shape
    epsilon_tensor = torch.full(shape, epsilon, device=device, dtype=dtype)
    if source_location is None:
        source_location = (nz // 2, ny // 2, nx // 4)
    if receiver_locations is None:
        receiver_locations = ((nz // 2, ny // 2, nx // 2),)
    solver_kwargs: dict[str, object] = {
        "source_component": source_component,
        "receiver_component": receiver_component,
        "stencil": stencil,
    }
    if python_backend is not None:
        solver_kwargs["python_backend"] = python_backend
    return MaxwellExample(
        name=name,
        solver=tide.maxwell._kernel_api.maxwell3d,
        born_solver=tide.maxwell._kernel_api.born3d,
        epsilon=epsilon_tensor,
        sigma=torch.full_like(epsilon_tensor, sigma),
        mu=torch.full_like(epsilon_tensor, mu),
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=tide.ricker(
            frequency,
            nt,
            dt,
            peak_time=1.0 / frequency if peak_time is None else peak_time,
            dtype=dtype,
            device=device,
        ).view(1, 1, nt),
        source_location=torch.tensor(
            [[source_location]],
            dtype=torch.long,
            device=device,
        ),
        receiver_location=torch.tensor(
            [receiver_locations],
            dtype=torch.long,
            device=device,
        ),
        pml_width=pml_width,
        solver_kwargs=solver_kwargs,
    )


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64)
    assert bool(torch.isfinite(actual64).all())
    assert bool(torch.isfinite(reference64).all())
    denominator = torch.linalg.vector_norm(reference64)
    assert denominator > 0.0, "relative error requires a nonzero reference"
    return float(
        (torch.linalg.vector_norm(actual64 - reference64) / denominator).item()
    )


def cosine_similarity(actual: torch.Tensor, reference: torch.Tensor) -> float:
    actual64 = actual.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference64 = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    assert bool(torch.isfinite(actual64).all())
    assert bool(torch.isfinite(reference64).all())
    denominator = torch.linalg.vector_norm(actual64) * torch.linalg.vector_norm(
        reference64
    )
    assert denominator > 0.0, "cosine similarity requires two nonzero vectors"
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
    norm = torch.linalg.vector_norm(direction.to(torch.float64))
    assert norm > 0.0, "direction mask removed every perturbation"
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
    assert math.isfinite(adjoint) and adjoint != 0.0
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
        assert math.isfinite(finite_difference) and finite_difference != 0.0
        denominator = max(abs(adjoint), abs(finite_difference))
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
    assert math.isfinite(base_scalar)
    assert math.isfinite(directional_derivative) and directional_derivative != 0.0
    zero_order: list[float] = []
    first_order: list[float] = []
    for step in steps:
        with torch.no_grad():
            perturbed = float(objective(base + step * direction))
        difference = perturbed - base_scalar
        zero_order.append(abs(difference))
        first_order.append(abs(difference - step * directional_derivative))
    assert all(math.isfinite(error) and error > 0.0 for error in zero_order)
    assert all(math.isfinite(error) and error > 0.0 for error in first_order)
    return zero_order, first_order


def convergence_orders(errors: Sequence[float]) -> list[float]:
    assert len(errors) >= 3
    assert all(math.isfinite(error) and error > 0.0 for error in errors)
    return [math.log2(left / right) for left, right in zip(errors, errors[1:])]


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
