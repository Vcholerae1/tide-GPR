"""Immutable contracts shared by the public API and backend dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class Dimension(StrEnum):
    TM2D = "tm2d"
    EM3D = "em3d"


class BackendPreference(StrEnum):
    AUTO = "auto"
    REFERENCE = "reference"
    NATIVE = "native"


class FallbackPolicy(StrEnum):
    REFERENCE = "reference"
    ERROR = "error"


class StorageMode(StrEnum):
    AUTO = "auto"
    DEVICE = "device"
    CPU = "cpu"
    DISK = "disk"
    NONE = "none"


class Operation(StrEnum):
    FORWARD = "forward"
    JVP = "jvp"
    VJP = "vjp"
    SECOND_VJP = "second_vjp"


class GradientTarget(StrEnum):
    """Differentiation targets a solver may be asked to back-propagate into.

    ``epsilon``/``sigma``/``mu`` are background-model tensors; ``perturbation``
    covers the Born perturbation inputs (`depsilon`, `dsigma`, `dca`, `dcb`);
    ``source`` is the source-amplitude wavelet; ``state`` is any initial
    wavefield or Born-derivative state tensor. Rows in the capability matrix
    declare the subset they support, and ``select_backend`` rejects native
    execution for plans that require targets a row does not offer.
    """

    EPSILON = "epsilon"
    SIGMA = "sigma"
    MU = "mu"
    PERTURBATION = "perturbation"
    SOURCE = "source"
    STATE = "state"


MODEL_GRADIENT_TARGETS = frozenset(
    {GradientTarget.EPSILON, GradientTarget.SIGMA, GradientTarget.MU}
)


@dataclass(frozen=True, slots=True)
class StorageOptions:
    """Snapshot storage policy after compatibility normalization."""

    mode: StorageMode = StorageMode.DEVICE
    path: str = "."
    compression: bool | str = False
    bytes_limit_device: int | None = None
    bytes_limit_host: int | None = None
    chunk_steps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("storage path must be a non-empty string.")
        for name, value in (
            ("bytes_limit_device", self.bytes_limit_device),
            ("bytes_limit_host", self.bytes_limit_host),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when provided.")
        if self.chunk_steps < 0:
            raise ValueError("storage_chunk_steps must be non-negative.")


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Execution policy independent of a particular Maxwell dimension."""

    backend: BackendPreference = BackendPreference.AUTO
    fallback: FallbackPolicy = FallbackPolicy.REFERENCE
    n_threads: int | None = None

    def __post_init__(self) -> None:
        if self.n_threads is not None and self.n_threads < 0:
            raise ValueError("n_threads must be non-negative when provided.")


@dataclass(frozen=True, slots=True)
class SimulationPlan:
    """Validated execution metadata passed from API to a backend.

    The plan deliberately contains no model or source tensors. That keeps it
    cheap to inspect, safe to cache, and independent from autograd graphs.
    """

    dimension: Dimension
    device: torch.device
    dtype: torch.dtype
    model_batched: bool
    runtime: RuntimeOptions
    storage: StorageOptions
    operation: Operation
    has_callbacks: bool
    model_gradient_sampling_interval: int = 1
    hessian_mode: str | None = None
    source_component: str = "ey"
    receiver_component: str = "ey"
    gradient_targets: frozenset[GradientTarget] = frozenset()
    has_dispersion: bool = False

    @property
    def has_model_gradients(self) -> bool:
        return bool(self.gradient_targets & MODEL_GRADIENT_TARGETS)

    @property
    def backend(self) -> BackendPreference:
        return self.runtime.backend

    def require_native(self, reason: str) -> None:
        if self.backend is BackendPreference.REFERENCE:
            raise NotImplementedError(
                f"{reason} requires the native backend, "
                "but backend='python' was requested."
            )


__all__ = [
    "BackendPreference",
    "Dimension",
    "FallbackPolicy",
    "GradientTarget",
    "Operation",
    "RuntimeOptions",
    "SimulationPlan",
    "StorageMode",
    "StorageOptions",
]
