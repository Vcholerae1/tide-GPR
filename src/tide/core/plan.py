"""Compatibility normalization and validation for simulation execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from .types import (
    BackendPreference,
    Dimension,
    FallbackPolicy,
    GradientTarget,
    Operation,
    RuntimeOptions,
    SimulationPlan,
    StorageMode,
    StorageOptions,
)


def normalize_backend_request(
    python_backend: bool | str,
) -> BackendPreference:
    """Normalize the legacy ``python_backend`` flag."""

    if not isinstance(python_backend, (bool, str)):
        raise TypeError(
            "python_backend must be bool or str, "
            f"but got {type(python_backend).__name__}."
        )
    if python_backend is True:
        return BackendPreference.REFERENCE
    if python_backend is False:
        return BackendPreference.AUTO
    mode = python_backend.lower()
    if mode in {"eager", "jit", "compile", "python", "reference"}:
        return BackendPreference.REFERENCE
    if mode in {"auto", "native", "standard"}:
        return BackendPreference.NATIVE if mode == "native" else BackendPreference.AUTO
    raise ValueError(f"Unknown python_backend value {python_backend!r}.")


def derive_gradient_targets(
    *,
    epsilon: torch.Tensor | None,
    sigma: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    perturbation_tensors: Sequence[torch.Tensor | None] = (),
    source_amplitude: torch.Tensor | None = None,
    state_tensors: Sequence[torch.Tensor | None] = (),
) -> frozenset[GradientTarget]:
    """Record which inputs actually require gradients.

    Solvers compute this from the user-supplied tensors and pass the result to
    ``compile_simulation_plan`` so the backend decision (and only the backend
    decision) owns every fallback.
    """
    targets: set[GradientTarget] = set()
    if epsilon is not None and epsilon.requires_grad:
        targets.add(GradientTarget.EPSILON)
    if sigma is not None and sigma.requires_grad:
        targets.add(GradientTarget.SIGMA)
    if mu is not None and mu.requires_grad:
        targets.add(GradientTarget.MU)
    if any(
        tensor is not None and tensor.requires_grad for tensor in perturbation_tensors
    ):
        targets.add(GradientTarget.PERTURBATION)
    if source_amplitude is not None and source_amplitude.requires_grad:
        targets.add(GradientTarget.SOURCE)
    if any(tensor is not None and tensor.requires_grad for tensor in state_tensors):
        targets.add(GradientTarget.STATE)
    return frozenset(targets)


def _normalize_gradient_targets(
    value: GradientTarget | str | Sequence[str | GradientTarget],
) -> frozenset[GradientTarget]:
    if isinstance(value, (str, GradientTarget)):
        items = (value,)
    else:
        items = tuple(value)
    targets: set[GradientTarget] = set()
    for item in items:
        try:
            targets.add(GradientTarget(str(item).lower()))
        except ValueError as exc:
            raise ValueError(
                "gradient_targets must be a subset of "
                "{'epsilon', 'sigma', 'mu', 'source', 'state', "
                "'perturbation'}."
            ) from exc
    return frozenset(targets)


def _normalize_storage_mode(value: str) -> StorageMode:
    try:
        return StorageMode(str(value).lower())
    except ValueError as exc:
        raise ValueError(
            "storage_mode must be one of 'auto', 'device', 'cpu', 'disk', or 'none'."
        ) from exc


def _normalize_fallback(value: str) -> FallbackPolicy:
    try:
        return FallbackPolicy(str(value).lower())
    except ValueError as exc:
        raise ValueError("fallback must be 'reference' or 'error'.") from exc


def compile_simulation_plan(
    *,
    dimension: Dimension | Literal["tm2d", "em3d"] | str,
    epsilon: torch.Tensor,
    sigma: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    python_backend: bool | str = False,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    storage_chunk_steps: int = 0,
    n_threads: int | None = None,
    fallback: str = "reference",
    has_callbacks: bool = False,
    source_component: str = "ey",
    receiver_component: str = "ey",
    operation: Operation
    | Literal["forward", "jvp", "vjp", "second_vjp"]
    | str = Operation.FORWARD,
    model_gradient_sampling_interval: int = 1,
    hessian_mode: str | None = None,
    gradient_targets: GradientTarget
    | str
    | Sequence[str | GradientTarget]
    | None = None,
    has_dispersion: bool = False,
) -> SimulationPlan:
    """Compile a dimension-independent execution plan.

    This function intentionally validates only cross-cutting concerns. Shape,
    location, CFL, material and stencil validation remain with each physics
    implementation until the compatibility migration is complete.
    """

    try:
        resolved_dimension = Dimension(str(dimension).lower())
    except ValueError as exc:
        raise ValueError("dimension must be 'tm2d' or 'em3d'.") from exc
    try:
        resolved_operation = Operation(str(operation).lower())
    except ValueError as exc:
        raise ValueError(
            "operation must be 'forward', 'jvp', 'vjp', or 'second_vjp'."
        ) from exc
    if not isinstance(epsilon, torch.Tensor):
        raise TypeError("epsilon must be a torch.Tensor.")
    if epsilon.dtype not in (torch.float32, torch.float64):
        raise TypeError("epsilon must have dtype torch.float32 or torch.float64.")
    if epsilon.device.type not in {"cpu", "cuda"}:
        raise NotImplementedError(
            "TIDE simulation plans support only CPU and CUDA tensors."
        )
    for name, tensor in (("sigma", sigma), ("mu", mu)):
        if tensor is not None and (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != epsilon.device
            or tensor.dtype != epsilon.dtype
        ):
            raise ValueError(f"{name} must share device and dtype with epsilon.")

    expected_ndim = 2 if resolved_dimension is Dimension.TM2D else 3
    if epsilon.ndim not in {expected_ndim, expected_ndim + 1}:
        raise ValueError(
            f"{resolved_dimension.value} expects a {expected_ndim}-D model "
            f"or a leading model-batch dimension, got rank {epsilon.ndim}."
        )

    backend = normalize_backend_request(python_backend)
    resolved_storage_mode = _normalize_storage_mode(storage_mode)
    if not isinstance(model_gradient_sampling_interval, int):
        raise TypeError("model_gradient_sampling_interval must be an integer.")
    if model_gradient_sampling_interval < 0:
        raise ValueError("model_gradient_sampling_interval must be non-negative.")
    if hessian_mode is not None and hessian_mode not in {"full", "gauss_newton"}:
        raise ValueError("hessian_mode must be 'full' or 'gauss_newton'.")
    if resolved_operation in {Operation.SECOND_VJP, Operation.JVP}:
        hessian_mode = "full" if hessian_mode is None else hessian_mode
    if resolved_dimension is Dimension.TM2D and source_component != "ey":
        raise ValueError("TM2D source_component must be 'ey'.")
    if resolved_dimension is Dimension.TM2D and receiver_component != "ey":
        raise ValueError("TM2D receiver_component must be 'ey'.")
    if resolved_dimension is Dimension.EM3D:
        if not isinstance(source_component, str) or not isinstance(
            receiver_component, str
        ):
            raise TypeError(
                "3-D source_component and receiver_component must be strings."
            )
        valid_components = {"ex", "ey", "ez"}
        if source_component.lower() not in valid_components:
            raise ValueError(f"invalid source_component {source_component!r}.")
        if receiver_component.lower() not in valid_components:
            raise ValueError(f"invalid receiver_component {receiver_component!r}.")

    if gradient_targets is None:
        resolved_gradient_targets = derive_gradient_targets(
            epsilon=epsilon,
            sigma=sigma,
            mu=mu,
        )
    else:
        resolved_gradient_targets = _normalize_gradient_targets(gradient_targets)

    return SimulationPlan(
        dimension=resolved_dimension,
        device=epsilon.device,
        dtype=epsilon.dtype,
        model_batched=epsilon.ndim == expected_ndim + 1,
        runtime=RuntimeOptions(
            backend=backend,
            fallback=_normalize_fallback(fallback),
            n_threads=n_threads,
        ),
        storage=StorageOptions(
            mode=resolved_storage_mode,
            path=storage_path,
            compression=storage_compression,
            bytes_limit_device=storage_bytes_limit_device,
            bytes_limit_host=storage_bytes_limit_host,
            chunk_steps=storage_chunk_steps,
        ),
        operation=resolved_operation,
        has_callbacks=has_callbacks,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        hessian_mode=hessian_mode,
        source_component=source_component.lower(),
        receiver_component=receiver_component.lower(),
        gradient_targets=resolved_gradient_targets,
        has_dispersion=has_dispersion,
    )


__all__ = [
    "compile_simulation_plan",
    "derive_gradient_targets",
    "normalize_backend_request",
]
