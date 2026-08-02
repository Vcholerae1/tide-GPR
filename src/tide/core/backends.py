"""Backend capability checks and explicit fallback decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .types import (
    BackendPreference,
    Dimension,
    FallbackPolicy,
    Operation,
    SimulationPlan,
)


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Capabilities that affect dispatch, not numerical correctness."""

    name: BackendPreference
    cpu: bool
    cuda: bool
    gradients: bool
    callbacks: bool
    storage_modes: frozenset[str]
    operations: frozenset[Operation]
    reusable_background: bool = False

    def unsupported_reason(self, plan: SimulationPlan) -> str | None:
        """Return the first capability mismatch in user-facing terms."""

        backend_name = "Python" if self.name is BackendPreference.PYTHON else "Native"
        dimension_name = "TM2D" if plan.dimension is Dimension.TM2D else "3D"
        operation_name = (
            "HVP"
            if plan.operation is Operation.HVP
            else plan.operation.value
        )
        if plan.device.type == "cpu" and not self.cpu:
            return f"{backend_name} {dimension_name} {operation_name} does not support CPU."
        if plan.device.type == "cuda" and not self.cuda:
            return f"{backend_name} {dimension_name} {operation_name} does not support CUDA."
        if plan.storage.mode.value not in self.storage_modes:
            return (
                f"{backend_name} {dimension_name} {operation_name} does not support "
                f"storage_mode={plan.storage.mode.value!r}."
            )
        if plan.has_model_gradients and not self.gradients:
            return f"{backend_name} {dimension_name} does not support model gradients."
        if plan.has_callbacks and not self.callbacks:
            return f"{backend_name} backend does not support callbacks."
        if plan.operation not in self.operations:
            return f"{backend_name} backend does not support {plan.operation.value}."
        if (
            plan.operation in {Operation.HVP, Operation.LINEARIZATION}
            and self.name is BackendPreference.PYTHON
            and plan.model_gradient_sampling_interval > 1
        ):
            return (
                f"Python {dimension_name} {operation_name} currently requires "
                "model_gradient_sampling_interval in {0, 1}."
            )
        if (
            plan.operation in {Operation.HVP, Operation.LINEARIZATION}
            and self.name is BackendPreference.NATIVE
            and plan.dimension is Dimension.TM2D
            and plan.device.type == "cpu"
            and plan.model_gradient_sampling_interval > 1
        ):
            return (
                f"Native TM2D {operation_name} on CPU currently requires "
                "model_gradient_sampling_interval in {0, 1}."
            )
        if (
            plan.operation in {Operation.HVP, Operation.LINEARIZATION}
            and plan.hessian_mode == "full"
            and self.name is BackendPreference.NATIVE
            and plan.dimension is Dimension.TM2D
            and plan.storage.mode.value != "device"
        ):
            return (
                f"Native TM2D full {operation_name} currently requires "
                "storage_mode='device'."
            )
        return None

    def supports(self, plan: SimulationPlan) -> bool:
        return self.unsupported_reason(plan) is None

    def can_reuse_background(self, plan: SimulationPlan) -> bool:
        return bool(
            self.reusable_background
            and plan.operation is Operation.LINEARIZATION
            and plan.dimension is Dimension.TM2D
            and plan.device.type == "cuda"
            and plan.storage.mode.value == "device"
            and plan.model_gradient_sampling_interval in {0, 1}
        )


@dataclass(frozen=True, slots=True)
class BackendDecision:
    requested: BackendPreference
    selected: BackendPreference
    fallback: bool
    reason: str | None = None
    capabilities: BackendCapabilities | None = None

    def can_reuse_background(self, plan: SimulationPlan) -> bool:
        return bool(
            self.capabilities is not None
            and self.capabilities.can_reuse_background(plan)
        )


def _capabilities(name: BackendPreference) -> BackendCapabilities:
    if name is BackendPreference.PYTHON:
        return BackendCapabilities(
            name=name,
            cpu=True,
            cuda=True,
            gradients=True,
            callbacks=True,
            storage_modes=frozenset({"auto", "device", "none"}),
            operations=frozenset(Operation),
        )
    return BackendCapabilities(
        name=BackendPreference.NATIVE,
        cpu=True,
        cuda=True,
        gradients=True,
        callbacks=True,
        storage_modes=frozenset({"auto", "device", "cpu", "disk", "none"}),
        operations=frozenset(Operation),
        reusable_background=True,
    )


def select_backend(
    plan: SimulationPlan,
    *,
    native_available: bool,
) -> BackendDecision:
    """Resolve a plan without silently changing a requested backend."""

    python_capabilities = _capabilities(BackendPreference.PYTHON)
    native_capabilities = _capabilities(BackendPreference.NATIVE)
    if plan.backend is BackendPreference.PYTHON:
        python_reason = python_capabilities.unsupported_reason(plan)
        if python_reason is not None:
            raise NotImplementedError(python_reason)
        return BackendDecision(
            plan.backend,
            BackendPreference.PYTHON,
            False,
            capabilities=python_capabilities,
        )
    native_reason = native_capabilities.unsupported_reason(plan)
    if native_available and native_reason is None:
        return BackendDecision(
            plan.backend,
            BackendPreference.NATIVE,
            False,
            capabilities=native_capabilities,
        )
    reason = (
        "native backend library is unavailable"
        if not native_available
        else native_reason
    )
    if (
        plan.runtime.fallback is FallbackPolicy.ERROR
        or plan.backend is BackendPreference.NATIVE
    ):
        if native_available:
            raise NotImplementedError(reason)
        raise RuntimeError(reason)
    python_reason = python_capabilities.unsupported_reason(plan)
    if python_reason is not None:
        raise NotImplementedError(native_reason if native_available else python_reason)
    return BackendDecision(
        requested=plan.backend,
        selected=BackendPreference.PYTHON,
        fallback=True,
        reason=reason,
        capabilities=python_capabilities,
    )


__all__ = ["BackendCapabilities", "BackendDecision", "select_backend"]
