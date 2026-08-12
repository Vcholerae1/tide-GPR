---
title: "API Orientation"
description: "How TIDE separates physical models, fixed experiments, execution policy, and derivative sessions."
---

TIDE's public API is organized around one idea: the electromagnetic model changes during inversion, while the numerical experiment usually does not. The API therefore separates model tensors from the operator that owns geometry, discretization, execution, and storage policy.

## The six domain objects

| Object | Owns | Changes during inversion? |
| --- | --- | --- |
| `EMModel` | `epsilon`, `sigma`, `mu`, optional dispersion | Yes |
| `Discretization` | spacing, `dt`, stencil, CPML, optional maximum velocity | Usually no |
| `Acquisition` | source and receiver indices | Usually no |
| `Experiment` | acquisition, source samples, `nt`, components, signal conditioning | Usually no |
| `ExecutionOptions` | backend, fallback, reference mode, thread count | No |
| `StorageOptions` | snapshot location, compression, memory limits, chunking | No |

Keeping these roles separate prevents a common inversion bug: rebuilding a solver with slightly different numerical settings on every iteration.

## Constructing an operator

```python
import torch
import tide

ny, nx = 80, 120
dt = 4.0e-11
nt = 400

model = tide.EMModel(
    epsilon=torch.full((ny, nx), 4.0),
    sigma=torch.zeros(ny, nx),
    mu=torch.ones(ny, nx),
)

source = tide.ricker(8.0e8, nt, dt).reshape(1, 1, nt)
source_location = torch.tensor([[[10, 50]]], dtype=torch.long)
receiver_location = torch.tensor([[[10, 70], [10, 90]]], dtype=torch.long)

operator = tide.MaxwellTM(
    tide.Discretization(
        spacing=0.02,
        dt=dt,
        stencil=4,
        boundary=tide.CPML(12),
    ),
    tide.Experiment(
        tide.Acquisition(source_location, receiver_location),
        source,
    ),
    execution=tide.ExecutionOptions(
        fallback=tide.FallbackPolicy.REFERENCE,
    ),
)

result = operator(model)
print(result.receiver_data.shape)  # torch.Size([400, 1, 2])
```

`ForwardResult.receiver_data` is the quantity used by most workflows. `ForwardResult.final_state` contains the final padded wavefields and CPML state, which can be passed back to lower-level continuation workflows when needed.

## Operators are reusable

A `MaxwellTM` or `Maxwell3D` instance contains no trainable material parameters. Reuse it with different `EMModel` values:

```python
background = operator(model)

perturbed = tide.EMModel(
    epsilon=model.epsilon * 1.05,
    sigma=model.sigma,
    mu=model.mu,
)
changed = operator(perturbed)
```

Reusing the operator makes the fixed experiment visible in code and avoids coupling optimization state to propagation configuration.

## Explicit derivatives

`operator.linearize(model)` fixes the point at which derivatives are evaluated. The returned session provides:

$$
\begin{aligned}
\text{primal} &= F(m),\\
\text{jvp}(v) &= J(m)v,\\
\text{vjp}(r) &= J(m)^\top r,\\
\text{second\_vjp}(v, r) &= (DJ(m)[v])^\top r.
\end{aligned}
$$

```python
direction = tide.EMDirection(
    epsilon=torch.ones_like(model.epsilon),
)

with operator.linearize(model) as linearized:
    born_data = linearized.jvp(direction).receiver_data
    gradient = linearized.vjp(torch.ones_like(linearized.primal.receiver_data))

print(born_data.shape)
print(gradient.epsilon.shape)
```

Use a context manager. A derivative session may own forward snapshots, host buffers, or temporary files, and `with` guarantees deterministic release.

:::note[JVP and VJP are actions]
Neither method returns a Jacobian matrix. `jvp` maps one model-space direction to receiver space. `vjp` maps one receiver-space cotangent back to the selected material fields.
:::

## Standard autograd still works

For a scalar objective, ordinary PyTorch code is often shortest:

```python
epsilon = model.epsilon.clone().requires_grad_(True)
trial = tide.EMModel(epsilon, model.sigma, model.mu)
predicted = operator(trial).receiver_data
loss = predicted.square().mean()
loss.backward()
print(epsilon.grad.shape)
```

Use explicit `jvp`, `vjp`, and `second_vjp` when implementing linearized imaging, Gauss-Newton, Hessian-vector products, or derivative tests. Use standard autograd when a scalar loss and model gradient are sufficient.

## Backend policy is explicit

`BackendPreference.AUTO` selects a supported native implementation when available. `FallbackPolicy.REFERENCE` permits a compatible Python path. `FallbackPolicy.ERROR` turns unsupported combinations into an immediate error. No solver-specific silent fallback is intended.

Before a large run, inspect backend availability and verify the exact combination of dimension, dtype, device, storage mode, derivative operation, and callback use that the run requires.

## Continue reading

- [Modeling](/tide-GPR/guides/modeling/) explains units, resolution, and model construction.
- [Maxwell operators](/tide-GPR/api/maxwell/) documents the structured forward and derivative contracts.
- [Storage](/tide-GPR/guides/storage/) explains snapshot lifetime and memory trade-offs.
- [Workflow](/tide-GPR/api/workflow/) covers shot batching and receiver objectives.
