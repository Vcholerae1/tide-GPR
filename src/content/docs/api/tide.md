---
title: "Top-Level API"
description: "The stable names exported directly from the tide package."
---

Application code should prefer the names exported by `tide.__all__`. This surface contains stable domain objects, operators, result types, physical helpers, and package namespaces. Tuple-based kernel adapters remain private.

## Core construction

```python
model = tide.EMModel(epsilon, sigma, mu)

discretization = tide.Discretization(
    spacing=0.02,
    dt=4.0e-11,
    stencil=4,
    boundary=tide.CPML(12),
)

experiment = tide.Experiment(
    tide.Acquisition(source_location, receiver_location),
    source_amplitude,
)

operator = tide.MaxwellTM(discretization, experiment)
```

The same construction pattern applies to `Maxwell3D`. Only model dimensionality, location width, field components, and computational scale change.

## Material and experiment types

- `EMModel`: physical material values.
- `EMDirection`: material-space direction for JVP and Hessian products.
- `DebyeDispersion`: optional single-pole dispersive material data.
- `CPML`: per-side absorbing-boundary width.
- `Discretization`: grid and external time sampling.
- `Acquisition`: source and receiver index tensors.
- `Experiment`: fixed acquisition, source, time axis, and component choices.
- `Observers`: optional forward and backward callbacks with cadence.

These objects validate local invariants when they are constructed. Solver-specific checks, such as dimensional shape and backend capability, occur when an operator is created or called.

## Execution and storage types

- `ExecutionOptions` selects `BackendPreference`, `FallbackPolicy`, reference execution mode, and native CPU threads.
- `StorageOptions` selects `StorageMode`, path, compression, byte limits, and chunking.
- `BackendDecision` and `SimulationPlan` expose normalized dispatch decisions for advanced inspection.

Use enum members instead of free-form strings in maintained application code. They make invalid values fail earlier and keep configuration discoverable.

## Operators

- `MaxwellTM`: nonlinear 2D TM operator.
- `Maxwell3D`: nonlinear full 3D operator.
- `LinearizedMaxwellTM`: derivative session at one TM2D model.
- `LinearizedMaxwell3D`: derivative session at one EM3D model.

Create linearized objects through `operator.linearize(model)` rather than constructing them directly. The operator supplies consistent discretization, experiment, execution, and storage settings.

## Named results

- `ForwardResult`: receiver data and final nonlinear state.
- `TangentResult`: tangent receiver data and final tangent state.
- `TMState` and `EM3DState`: named final field state.
- `EMGradient`: optional `epsilon`, `sigma`, and `mu` gradient fields.

Named results replace positional tuple indexing. Access `result.receiver_data` and `result.final_state` explicitly so code remains stable when internal state representation changes.

## Utility exports

`tide.ricker` generates source wavelets. CFL, resampling, callback, padding, validation, and dispersion utilities are also available through package modules. Workflow and optimizer APIs live under `tide.workflow` and `tide.optim` to keep the modeling namespace focused.

## Compatibility boundary

Only names present in `tide.__all__` and documented subpackage exports are part of the supported surface. Files ending in `_python`, `_cuda`, or `_autograd`, native ABI declarations, and tuple-returning kernel functions are implementation details. They may change while the structured operator behavior remains stable.
