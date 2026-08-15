"""Backend capability checks and explicit fallback decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import (
    BackendPreference,
    Dimension,
    FallbackPolicy,
    GradientTarget,
    Operation,
    SimulationPlan,
    StorageMode,
)


#: Gradient targets whose native execution requires snapshot storage. Source
#: gradients use the autograd wrappers without stored wavefields, so they are
#: excluded from the storage-none interaction for forward operations.
SNAPSHOT_REQUIRING_TARGETS = frozenset({GradientTarget.EPSILON, GradientTarget.SIGMA})


@dataclass(frozen=True, slots=True)
class BackendCapability:
    """One supported capability row in the execution matrix.

    Rows are intentionally dimension-scoped today. Keeping the row as a
    first-class value lets a future operation or backend add a narrow cell
    without putting another branch in every public solver function.
    """

    dimension: Dimension
    operations: frozenset[Operation]
    devices: frozenset[str]
    dtypes: frozenset[torch.dtype]
    storage_modes: frozenset[str]
    callbacks: bool
    reusable_background: bool = False
    gradient_targets: frozenset[GradientTarget] = frozenset()

    def matches(self, plan: SimulationPlan) -> bool:
        return self.dimension is plan.dimension and plan.operation in self.operations


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Capabilities that affect dispatch, not numerical correctness.

    The scalar fields are retained as compatibility summaries for callers that
    inspected the old object. New dispatch decisions use ``matrix`` so the
    supported dimension/operation/device/precision cells have one owner.
    """

    name: BackendPreference
    cpu: bool
    cuda: bool
    callbacks: bool
    storage_modes: frozenset[str]
    operations: frozenset[Operation]
    reusable_background: bool = False
    matrix: tuple[BackendCapability, ...] = ()

    def capability_for(self, plan: SimulationPlan) -> BackendCapability | None:
        """Return the matrix row responsible for a simulation plan."""
        for capability in self.matrix:
            if capability.matches(plan):
                return capability
        return None

    def unsupported_reason(self, plan: SimulationPlan) -> str | None:
        """Return the first capability mismatch in user-facing terms."""

        backend_name = (
            "Python" if self.name is BackendPreference.REFERENCE else "Native"
        )
        dimension_name = "TM2D" if plan.dimension is Dimension.TM2D else "3D"
        operation_name = plan.operation.value
        capability = self.capability_for(plan)
        if capability is None:
            return f"{backend_name} {dimension_name} is not in the capability matrix."
        if plan.device.type not in capability.devices:
            return (
                f"{backend_name} {dimension_name} {operation_name} does not support "
                f"{plan.device.type.upper()}."
            )
        if plan.dtype not in capability.dtypes:
            return (
                f"{backend_name} {dimension_name} {operation_name} does not support "
                f"dtype={plan.dtype}."
            )
        if plan.operation not in capability.operations:
            return f"{backend_name} backend does not support {plan.operation.value}."
        if plan.device.type == "cpu" and not self.cpu:
            return f"{backend_name} {dimension_name} {operation_name} does not support CPU."
        if plan.device.type == "cuda" and not self.cuda:
            return f"{backend_name} {dimension_name} {operation_name} does not support CUDA."
        if plan.storage.mode.value not in capability.storage_modes:
            return (
                f"{backend_name} {dimension_name} {operation_name} does not support "
                f"storage_mode={plan.storage.mode.value!r}."
            )
        missing_targets = plan.gradient_targets - capability.gradient_targets
        if missing_targets:
            target_names = ", ".join(sorted(target.value for target in missing_targets))
            return (
                f"{backend_name} {dimension_name} {operation_name} does not "
                f"support gradients w.r.t. {target_names}."
            )
        if (
            self.name is BackendPreference.NATIVE
            and plan.storage.mode is StorageMode.NONE
        ):
            storage_requiring = plan.gradient_targets & capability.gradient_targets
            if plan.operation is Operation.FORWARD:
                # Native forward source gradients run through the autograd
                # wrappers without stored wavefields; only model gradients
                # require snapshot storage.
                storage_requiring = storage_requiring & SNAPSHOT_REQUIRING_TARGETS
            if storage_requiring:
                target_names = ", ".join(
                    sorted(target.value for target in storage_requiring)
                )
                return (
                    f"{backend_name} {dimension_name} {operation_name} does "
                    f"not support gradients w.r.t. {target_names} with "
                    "storage_mode='none'."
                )
        if (
            plan.operation is Operation.FORWARD
            and plan.has_dispersion
            and plan.gradient_targets
            and self.name is BackendPreference.NATIVE
        ):
            return (
                f"{backend_name} {dimension_name} forward does not support "
                "gradients with dispersion; use the Python reference backend."
            )
        if plan.has_callbacks and not capability.callbacks:
            return f"{backend_name} backend does not support callbacks."
        if (
            plan.operation in {Operation.SECOND_VJP, Operation.JVP}
            and self.name is BackendPreference.REFERENCE
            and plan.model_gradient_sampling_interval > 1
        ):
            return (
                f"Python {dimension_name} {operation_name} currently requires "
                "model_gradient_sampling_interval in {0, 1}."
            )
        if (
            plan.operation in {Operation.SECOND_VJP, Operation.JVP}
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
            plan.operation in {Operation.SECOND_VJP, Operation.JVP}
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
        capability = self.capability_for(plan)
        return bool(
            capability is not None
            and capability.reusable_background
            and self.reusable_background
            and plan.operation is Operation.JVP
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


def backend_capabilities(name: BackendPreference) -> BackendCapabilities:
    """Return the immutable capability matrix for a backend.

    Rows are the dispatch contract: every advertised derivative primitive,
    storage mode, and target must have an executable adapter.
    """
    all_targets = frozenset(GradientTarget)
    if name is BackendPreference.REFERENCE:
        forward_storage = frozenset({"auto", "device", "cpu", "disk", "none"})
        jvp_tm2d_storage = frozenset({"auto", "device", "cpu", "disk", "none"})
        jvp_em3d_storage = frozenset({"device", "none"})
        second_vjp_tm2d_storage = frozenset({"device", "cpu", "disk"})
        second_vjp_em3d_storage = frozenset({"device"})
        rows: list[BackendCapability] = []
        for dimension in Dimension:
            rows.append(
                BackendCapability(
                    dimension=dimension,
                    operations=frozenset({Operation.FORWARD, Operation.VJP}),
                    devices=frozenset({"cpu", "cuda"}),
                    dtypes=frozenset({torch.float32, torch.float64}),
                    storage_modes=forward_storage,
                    gradient_targets=all_targets,
                    callbacks=True,
                )
            )
            rows.append(
                BackendCapability(
                    dimension=dimension,
                    operations=frozenset({Operation.JVP}),
                    devices=frozenset({"cpu", "cuda"}),
                    dtypes=frozenset({torch.float32, torch.float64}),
                    storage_modes=(
                        jvp_tm2d_storage
                        if dimension is Dimension.TM2D
                        else jvp_em3d_storage
                    ),
                    gradient_targets=all_targets,
                    callbacks=False,
                )
            )
            rows.append(
                BackendCapability(
                    dimension=dimension,
                    operations=frozenset({Operation.SECOND_VJP}),
                    devices=frozenset({"cpu", "cuda"}),
                    dtypes=frozenset({torch.float32, torch.float64}),
                    storage_modes=(
                        second_vjp_tm2d_storage
                        if dimension is Dimension.TM2D
                        else second_vjp_em3d_storage
                    ),
                    gradient_targets=all_targets,
                    callbacks=False,
                )
            )
        matrix = tuple(rows)
    else:
        model_targets = frozenset({GradientTarget.EPSILON, GradientTarget.SIGMA})
        forward_targets = model_targets | frozenset({GradientTarget.SOURCE})
        born_targets = model_targets | frozenset({GradientTarget.PERTURBATION})
        rows: list[BackendCapability] = []
        for dimension in Dimension:
            rows.append(
                BackendCapability(
                    dimension=dimension,
                    operations=frozenset({Operation.FORWARD, Operation.VJP}),
                    devices=frozenset({"cpu", "cuda"}),
                    dtypes=frozenset({torch.float32, torch.float64}),
                    storage_modes=frozenset({"auto", "device", "cpu", "disk", "none"}),
                    gradient_targets=forward_targets,
                    callbacks=True,
                )
            )
            if dimension is Dimension.TM2D:
                rows.append(
                    BackendCapability(
                        dimension=dimension,
                        operations=frozenset({Operation.JVP}),
                        devices=frozenset({"cpu", "cuda"}),
                        dtypes=frozenset({torch.float32, torch.float64}),
                        storage_modes=frozenset(
                            {"auto", "device", "cpu", "disk", "none"}
                        ),
                        gradient_targets=born_targets,
                        callbacks=False,
                        reusable_background=True,
                    )
                )
                rows.append(
                    BackendCapability(
                        dimension=dimension,
                        operations=frozenset({Operation.SECOND_VJP}),
                        devices=frozenset({"cpu", "cuda"}),
                        dtypes=frozenset({torch.float32, torch.float64}),
                        storage_modes=frozenset({"device", "cpu", "disk"}),
                        gradient_targets=model_targets,
                        callbacks=False,
                    )
                )
            else:
                rows.append(
                    BackendCapability(
                        dimension=dimension,
                        operations=frozenset({Operation.JVP}),
                        devices=frozenset({"cpu", "cuda"}),
                        dtypes=frozenset({torch.float32, torch.float64}),
                        storage_modes=frozenset({"device", "none"}),
                        gradient_targets=born_targets,
                        callbacks=False,
                    )
                )
                rows.append(
                    BackendCapability(
                        dimension=dimension,
                        operations=frozenset({Operation.SECOND_VJP}),
                        devices=frozenset({"cpu", "cuda"}),
                        dtypes=frozenset({torch.float32, torch.float64}),
                        storage_modes=frozenset({"device"}),
                        gradient_targets=model_targets,
                        callbacks=False,
                    )
                )
        matrix = tuple(rows)

    devices = frozenset(device for row in matrix for device in row.devices)
    return BackendCapabilities(
        name=name,
        cpu="cpu" in devices,
        cuda="cuda" in devices,
        callbacks=any(row.callbacks for row in matrix),
        storage_modes=frozenset(
            storage for row in matrix for storage in row.storage_modes
        ),
        operations=frozenset(
            operation for row in matrix for operation in row.operations
        ),
        reusable_background=any(row.reusable_background for row in matrix),
        matrix=matrix,
    )


def select_backend(
    plan: SimulationPlan,
    *,
    native_available: bool,
) -> BackendDecision:
    """Resolve a plan without silently changing a requested backend."""

    python_capabilities = backend_capabilities(BackendPreference.REFERENCE)
    native_capabilities = backend_capabilities(BackendPreference.NATIVE)
    if plan.backend is BackendPreference.REFERENCE:
        python_reason = python_capabilities.unsupported_reason(plan)
        if python_reason is not None:
            raise NotImplementedError(python_reason)
        return BackendDecision(
            plan.backend,
            BackendPreference.REFERENCE,
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
        selected=BackendPreference.REFERENCE,
        fallback=True,
        reason=reason,
        capabilities=python_capabilities,
    )


__all__ = [
    "BackendCapability",
    "BackendCapabilities",
    "BackendDecision",
    "backend_capabilities",
    "select_backend",
]
