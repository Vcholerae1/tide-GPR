"""Shared execution contracts for TIDE propagators.

The public Maxwell functions remain source-compatible, while the execution
configuration is normalized through this package before it reaches a backend.
Keeping these contracts independent from the numerical kernels prevents the
2-D and 3-D implementations from growing another copy of option parsing.
"""

from .backends import (
    BackendCapability,
    BackendCapabilities,
    BackendDecision,
    backend_capabilities,
    select_backend,
)
from .plan import (
    compile_simulation_plan,
    derive_gradient_targets,
    normalize_backend_request,
)
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

__all__ = [
    "BackendCapabilities",
    "BackendCapability",
    "BackendDecision",
    "BackendPreference",
    "Dimension",
    "FallbackPolicy",
    "GradientTarget",
    "Operation",
    "RuntimeOptions",
    "SimulationPlan",
    "StorageMode",
    "StorageOptions",
    "compile_simulation_plan",
    "derive_gradient_targets",
    "backend_capabilities",
    "normalize_backend_request",
    "select_backend",
]
