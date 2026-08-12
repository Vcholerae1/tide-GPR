---
title: "API Reference"
description: "Supported structured operators, domain objects, workflow helpers, and numerical utilities in TIDE."
---

The supported modeling surface is organized around structured Maxwell operators. Start with the domain objects below, then use module pages for signatures and lower-level utilities.

## Modeling objects

| Object | Purpose |
| --- | --- |
| `EMModel` | Material tensors: relative permittivity, conductivity, permeability, and optional dispersion |
| `EMDirection` | A perturbation direction in the same material space |
| `Discretization` | Spacing, external time step, stencil, CPML, and velocity bound |
| `Acquisition` | Source and receiver index tensors |
| `Experiment` | Acquisition, source samples, components, and signal controls |
| `ExecutionOptions` | Backend preference, fallback policy, reference mode, and threads |
| `StorageOptions` | Snapshot location, compression, limits, and chunking |

## Operators and results

`MaxwellTM` implements 2D transverse magnetic propagation. `Maxwell3D` implements full six-component 3D propagation. Both return `ForwardResult`, which contains named `receiver_data` and `final_state` fields.

```python
result = operator(model)
receiver_data = result.receiver_data
final_state = result.final_state
```

`operator.linearize(model)` creates a derivative session with `primal`, `jvp`, `vjp`, and `second_vjp`. These methods apply derivative operators without constructing a Jacobian matrix.

## Workflow layer

`tide.workflow` provides acquisition builders, shot indexing, mini-batch execution, receiver losses, distributed shot sharding, receiver objectives, and curvature preconditioners. It composes public operators and does not replace their physics or backend policy.

`tide.optim` provides torch-native first-order, LBFGS, CGNR, and truncated Newton routines. Optimizer state remains on the same device and dtype as the model tensor.

## Numerical utilities

| Module | Use |
| --- | --- |
| `tide.wavelets` | Ricker source generation |
| `tide.cfl` | Stable internal time-step planning |
| `tide.resampling` | Source upsampling and receiver downsampling |
| `tide.callbacks` | Forward and backward state inspection |
| `tide.storage` | Snapshot policy and temporary storage |
| `tide.validation` | Validation of sampling and taper controls |
| `tide.padding` | Padding and CPML-region masking |
| `tide.staggered` | Staggered derivative and CPML profile helpers |

## Recommended entry points

Most application code should import from `tide` and `tide.workflow`:

```python
import tide

model = tide.EMModel(epsilon, sigma, mu)
operator = tide.MaxwellTM(discretization, experiment)
receiver = operator(model).receiver_data
```

Low-level kernel adapters and backend function pointers are documented for maintainers, but they are not a substitute for the structured operator contract.

Continue with [Maxwell operators](/tide-GPR/api/maxwell/) for the complete forward and derivative model, or [API orientation](/tide-GPR/guides/api-orientation/) for a guided introduction.
