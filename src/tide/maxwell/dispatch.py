"""Shared execution-policy construction for Maxwell public facades.

The physics entry points still own shape, CFL, and material validation. This
module owns only the cross-cutting transition from a normalized
``SimulationPlan`` to one explicit backend decision, so the fallback policy is
not reimplemented in every solver family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core import (
    BackendDecision,
    BackendPreference,
    SimulationPlan,
    compile_simulation_plan,
    select_backend,
)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """Immutable plan and backend selection passed to a solver adapter."""

    plan: SimulationPlan
    decision: BackendDecision
    requested_backend: bool | str

    @property
    def use_python(self) -> bool:
        return self.decision.selected is BackendPreference.PYTHON

    @property
    def dispatch_backend(self) -> bool | str:
        """Final execution signal for the TM2D dispatcher.

        When the central decision selects Python, downstream dispatch must not
        re-probe the native backend: ``False`` (auto) is replaced with ``True``
        so ``maxwell_func`` takes the Python branch directly. Explicit python
        execution modes are preserved.
        """
        if not self.use_python:
            return False
        if (
            isinstance(self.requested_backend, str)
            and self.requested_backend.lower() in {"eager", "jit", "compile"}
        ):
            return self.requested_backend
        return True

    @property
    def compute_mode(self) -> str:
        return self.plan.compute_mode.value

    @property
    def storage_mode(self) -> str:
        return self.plan.storage.mode.value


def resolve_execution_policy(
    plan: SimulationPlan,
    *,
    requested_backend: bool | str,
) -> ExecutionPolicy:
    """Resolve one plan through the central capability matrix."""
    from .. import backend_utils

    decision = select_backend(
        plan,
        native_available=backend_utils.is_backend_available(),
    )
    return ExecutionPolicy(
        plan=plan,
        decision=decision,
        requested_backend=requested_backend,
    )


def compile_execution_policy(
    *,
    requested_backend: bool | str,
    **plan_kwargs: Any,
) -> ExecutionPolicy:
    """Compile a plan and resolve it without duplicating backend plumbing."""
    plan = compile_simulation_plan(
        python_backend=requested_backend,
        **plan_kwargs,
    )
    return resolve_execution_policy(plan, requested_backend=requested_backend)


__all__ = [
    "ExecutionPolicy",
    "compile_execution_policy",
    "resolve_execution_policy",
]
