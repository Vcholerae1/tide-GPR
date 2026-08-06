# Architecture Boundaries

TIDE is maintained as one Maxwell-focused package. The package is split into
four dependency layers:

```text
public API (tide.__init__, tide.maxwell)
        |
        v
core contracts (plan, types, capability decisions)
        |
        v
solver implementations (reference Python and native adapters)
        |
        v
C/CUDA kernels and ABI declarations
```

The `workflow` and `optim` packages are consumers of the public solver API;
the solver must not import either package. `core` contains only cross-cutting
execution contracts and must not import a solver, workflow, or optimizer.

## Stable versus internal code

The supported surface is the API exported by `tide.__all__` and
`tide.maxwell.__all__`. Files under `tide.maxwell` that end in `_python`,
`_cuda`, or `_autograd` are implementation details. A change to them is safe
to review as an internal change only when the public behavior and numerical
parity tests remain unchanged.

Research prototypes, generated outputs, large datasets, and profiling scripts
stay outside the supported package surface. A prototype is promoted only when
it has a stable API, a reference path, capability coverage, tests, and user
documentation. Otherwise it belongs in the archive policy described in
`experimental-paths.md`.

## Execution contract

Public solver functions normalize legacy options into `SimulationPlan`, pass it
through `tide.maxwell.dispatch.ExecutionPolicy`, ask the central capability
matrix for a `BackendDecision`, and then call one execution adapter. All public
TM2D/EM3D forward, Born, and HVP entry points, plus the TM2D linearization
context, now use that shared transition. New options must be added to the plan
and matrix before they are threaded into a kernel. A solver-specific silent
fallback is not allowed: unsupported combinations must either select the
declared reference fallback or raise a descriptive error.

## Dependency rules

The following import directions are enforced by tests:

- `core` may depend on the standard library and PyTorch, but not on solvers,
  workflow, optimizers, or the native ABI loader;
- `maxwell` may depend on `core` and shared numerical utilities, but not on
  `workflow` or `optim`;
- `optim` is self-contained and must not depend on `maxwell`, `workflow`, or
  `core`;
- `workflow` may consume public solver behavior, but solver code cannot depend
  on workflow helpers.
