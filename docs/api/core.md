# Module: tide.core

Cross-cutting execution contracts shared by the 2D and 3D Maxwell entry points.

## Planning flow

`compile_simulation_plan` normalizes legacy `python_backend` values together with
the runtime, storage, dtype, device, batching, and component choices. The resulting
`SimulationPlan` is then passed to `select_backend`, which applies the explicit
fallback policy and reports whether the reference implementation was selected.

```python
from tide import compile_simulation_plan, select_backend

plan = compile_simulation_plan(dimension="tm2d", epsilon=epsilon)
decision = select_backend(plan, native_available=True)
```

The contract is intentionally dimension-independent; stencil, CFL, material, and
boundary validation remains in the physics-specific modules.

## Capability matrix

`backend_capabilities(BackendPreference.NATIVE)` and
`backend_capabilities(BackendPreference.PYTHON)` return immutable rows covering
dimension, operation, device, dtype, compute mode, storage, gradient targets,
and callbacks. Dispatch uses these rows as the source of truth; see the
[feature matrix](../dev/feature-matrix.md) for the current stable baseline.

## Main types

- `Dimension`: `tm2d` or `em3d`
- `BackendPreference`: `auto`, `python`, or `native`
- `FallbackPolicy`: `reference` or `error`
- `ComputeMode`: `native` only
- `StorageOptions` and `RuntimeOptions`
- `SimulationPlan` and `BackendDecision`
- `BackendCapability` and `BackendCapabilities`
