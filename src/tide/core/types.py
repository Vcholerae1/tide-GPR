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
    PYTHON = "python"
    NATIVE = "native"


class FallbackPolicy(StrEnum):
    REFERENCE = "reference"
    ERROR = "error"


class ComputeMode(StrEnum):
    NATIVE = "native"
    FP16_IO = "fp16_io"


class StorageMode(StrEnum):
    AUTO = "auto"
    DEVICE = "device"
    CPU = "cpu"
    DISK = "disk"
    NONE = "none"


class Operation(StrEnum):
    FORWARD = "forward"
    BORN = "born"
    HVP = "hvp"
    LINEARIZATION = "linearization"


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
    compute_mode: ComputeMode = ComputeMode.NATIVE
    execution_backend: str = "standard"
    n_threads: int | None = None

    def __post_init__(self) -> None:
        if self.execution_backend != "standard":
            raise ValueError(
                f"execution_backend must be 'standard', got {self.execution_backend!r}."
            )
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
    has_model_gradients: bool
    has_callbacks: bool
    model_gradient_sampling_interval: int = 1
    hessian_mode: str | None = None
    source_component: str = "ey"
    receiver_component: str = "ey"

    @property
    def backend(self) -> BackendPreference:
        return self.runtime.backend

    @property
    def compute_mode(self) -> ComputeMode:
        return self.runtime.compute_mode

    def require_native(self, reason: str) -> None:
        if self.backend is BackendPreference.PYTHON:
            raise NotImplementedError(
                f"{reason} requires the native backend, "
                "but backend='python' was requested."
            )
        if (
            self.compute_mode is ComputeMode.FP16_IO
            and self.backend is BackendPreference.PYTHON
        ):
            raise NotImplementedError("fp16_io requires the native backend.")


__all__ = [
    "BackendPreference",
    "ComputeMode",
    "Dimension",
    "FallbackPolicy",
    "Operation",
    "RuntimeOptions",
    "SimulationPlan",
    "StorageMode",
    "StorageOptions",
]
