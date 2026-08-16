"""Public value objects for Maxwell simulations and linearizations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import torch

from ..core import BackendPreference, FallbackPolicy
from ..callbacks import Callback
from ..dispersion import DebyeDispersion


class SourceConvention(StrEnum):
    """How the supplied source amplitude enters the discrete Maxwell update."""

    PHYSICAL_CURRENT = "physical_current"
    FIELD_INCREMENT = "field_increment"


@dataclass(frozen=True, slots=True)
class CPML:
    """Convolutional PML boundary configuration."""

    width: int | tuple[int, ...] = 20

    def __post_init__(self) -> None:
        widths = (self.width,) if isinstance(self.width, int) else self.width
        if not widths or any(width < 0 for width in widths):
            raise ValueError("CPML widths must be non-negative.")


@dataclass(frozen=True, slots=True)
class Discretization:
    """Spatial and temporal discretization shared by one Maxwell operator."""

    spacing: float | tuple[float, ...]
    dt: float
    stencil: int = 2
    boundary: CPML = CPML()
    max_velocity: float | None = None

    def __post_init__(self) -> None:
        spacing = (
            (self.spacing,) if isinstance(self.spacing, (int, float)) else self.spacing
        )
        if not spacing or any(value <= 0 for value in spacing):
            raise ValueError("Grid spacing must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.stencil not in {2, 4, 6, 8}:
            raise ValueError("stencil must be one of 2, 4, 6, or 8.")
        if self.max_velocity is not None and self.max_velocity <= 0:
            raise ValueError("max_velocity must be positive when provided.")


@dataclass(frozen=True, slots=True)
class Acquisition:
    """Source and receiver geometry for a shot collection."""

    source_location: torch.Tensor | None
    receiver_location: torch.Tensor | None

    @property
    def n_shots(self) -> int:
        for tensor in (self.source_location, self.receiver_location):
            if tensor is not None:
                return int(tensor.shape[0])
        return 1

    @property
    def n_receivers(self) -> int:
        if self.receiver_location is None:
            return 0
        return int(self.receiver_location.shape[1])

    @property
    def spatial_ndim(self) -> int | None:
        for tensor in (self.source_location, self.receiver_location):
            if tensor is not None:
                return int(tensor.shape[-1])
        return None


@dataclass(frozen=True, slots=True)
class Experiment:
    """Fixed source, receiver, and time-axis definition for an operator."""

    acquisition: Acquisition
    source_amplitude: torch.Tensor | None
    nt: int | None = None
    source_component: str = "ey"
    receiver_component: str = "ey"
    source_convention: SourceConvention = SourceConvention.PHYSICAL_CURRENT
    frequency_taper_fraction: float = 0.0
    time_padding_fraction: float = 0.0
    time_taper: bool = False

    def __post_init__(self) -> None:
        if self.nt is not None and self.nt <= 0:
            raise ValueError("nt must be positive when provided.")
        if self.source_amplitude is None and self.nt is None:
            raise ValueError("nt is required when source_amplitude is absent.")
        if not self.source_component or not self.receiver_component:
            raise ValueError("Source and receiver components must be non-empty.")
        for name, value in (
            ("frequency_taper_fraction", self.frequency_taper_fraction),
            ("time_padding_fraction", self.time_padding_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")


@dataclass(frozen=True, slots=True)
class EMModel:
    """Physical electromagnetic material fields."""

    epsilon: torch.Tensor
    sigma: torch.Tensor
    mu: torch.Tensor
    dispersion: DebyeDispersion | None = None

    def __post_init__(self) -> None:
        if (
            self.sigma.shape != self.epsilon.shape
            or self.mu.shape != self.epsilon.shape
        ):
            raise ValueError("epsilon, sigma, and mu must have the same shape.")
        if (
            self.sigma.device != self.epsilon.device
            or self.mu.device != self.epsilon.device
        ):
            raise ValueError("epsilon, sigma, and mu must be on the same device.")
        if (
            self.sigma.dtype != self.epsilon.dtype
            or self.mu.dtype != self.epsilon.dtype
        ):
            raise ValueError("epsilon, sigma, and mu must have the same dtype.")


@dataclass(frozen=True, slots=True)
class EMDirection:
    """A direction in physical electromagnetic model space."""

    epsilon: torch.Tensor | None = None
    sigma: torch.Tensor | None = None
    mu: torch.Tensor | None = None

    def validate_for(self, model: EMModel) -> None:
        if self.epsilon is None and self.sigma is None and self.mu is None:
            raise ValueError("At least one model direction must be provided.")
        for name, direction, parameter in (
            ("epsilon", self.epsilon, model.epsilon),
            ("sigma", self.sigma, model.sigma),
            ("mu", self.mu, model.mu),
        ):
            if direction is not None and (
                direction.shape != parameter.shape
                or direction.device != parameter.device
                or direction.dtype != parameter.dtype
            ):
                raise ValueError(
                    f"The {name} direction must match the corresponding model tensor."
                )


@dataclass(frozen=True, slots=True)
class Observers:
    """Optional wavefield observers for forward and reverse propagation."""

    forward: Callback | None = None
    backward: Callback | None = None
    frequency: int = 1

    def __post_init__(self) -> None:
        if self.frequency <= 0:
            raise ValueError("Observer frequency must be positive.")


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Backend policy independent of a particular model tensor."""

    backend: BackendPreference = BackendPreference.AUTO
    fallback: FallbackPolicy = FallbackPolicy.ERROR
    reference_mode: Literal["eager", "jit", "compile"] = "eager"
    n_threads: int | None = None

    def __post_init__(self) -> None:
        if self.n_threads is not None and self.n_threads < 0:
            raise ValueError("n_threads must be non-negative when provided.")

    @property
    def legacy_backend_request(self) -> bool | str:
        if self.backend is BackendPreference.NATIVE:
            return "native"
        if self.backend is BackendPreference.REFERENCE:
            return self.reference_mode
        return False


__all__ = [
    "Acquisition",
    "CPML",
    "Discretization",
    "EMDirection",
    "Observers",
    "EMModel",
    "ExecutionOptions",
    "Experiment",
    "SourceConvention",
]
