---
title: "Maxwell Operators"
description: "Structured forward, JVP, VJP, and second-VJP contracts for TM2D and EM3D."
---

`MaxwellTM` and `Maxwell3D` represent fixed electromagnetic experiments. They own discretization, acquisition, source definition, execution policy, storage policy, and gradient sampling. The material model is supplied at evaluation time.

## Constructors

```python
MaxwellTM(
    discretization,
    experiment,
    execution=ExecutionOptions(),
    storage=StorageOptions(),
    model_gradient_sampling_interval=1,
)

Maxwell3D(
    discretization,
    experiment,
    execution=ExecutionOptions(),
    storage=StorageOptions(),
    model_gradient_sampling_interval=1,
)
```

`MaxwellTM` expects 2D `[ny, nx]` materials or `[batch, ny, nx]` batched materials. `Maxwell3D` expects `[nz, ny, nx]` or `[batch, nz, ny, nx]`.

## Forward

```python
result = operator.forward(model)
# Equivalent:
result = operator(model)
```

The nonlinear map is

$$
F:m\mapsto d.
$$

`result.receiver_data` is `[nt, shots, receivers]` for a shared model and `[nt, batch, shots, receivers]` for batched models. `result.final_state` is `TMState` or `EM3DState` with named field and CPML state tensors.

```python
model = tide.EMModel(epsilon, sigma, mu)
result = operator(model)
print(result.receiver_data.shape)
print(type(result.final_state).__name__)
```

A forward call remains differentiable through standard PyTorch autograd when model tensors require gradients.

## Linearize

```python
linearized = operator.linearize(
    model,
    storage=None,
    targets=("epsilon", "sigma"),
)
```

The session fixes a background model and selected derivative targets. `storage` overrides the operator's default for the session. Use the object as a context manager because it may retain forward snapshots, host allocations, or disk files.

```python
with operator.linearize(model, targets=("epsilon",)) as linearized:
    primal = linearized.primal
```

`primal` is evaluated lazily and cached for the lifetime of the open session.

## JVP

```python
tangent = linearized.jvp(direction)
```

`direction` is an `EMDirection` with one or more material perturbations. The result applies

$$
J(m)v
$$

and returns `TangentResult.receiver_data` plus a final tangent state. Missing direction fields mean zero perturbation. Each supplied tensor must match the corresponding background material shape, dtype, and device.

```python
direction = tide.EMDirection(
    epsilon=delta_epsilon,
    sigma=delta_sigma,
)
with operator.linearize(model) as linearized:
    born_receiver = linearized.jvp(direction).receiver_data
```

This operation is useful for Born modeling, linearized imaging, and the inner action of a Gauss-Newton Hessian product.

## VJP

```python
gradient = linearized.vjp(receiver_cotangent)
```

The receiver cotangent must have exactly the same shape as `linearized.primal.receiver_data`. The method applies

$$
J(m)^\top r
$$

and returns `EMGradient`. A field is `None` when it was not selected as a target or no derivative is available for it.

```python
with operator.linearize(model, targets=("epsilon", "sigma")) as linearized:
    residual = linearized.primal.receiver_data - observed
    gradient = linearized.vjp(residual)

print(gradient.epsilon.shape)
print(gradient.sigma.shape)
```

For a least-squares loss defined as `0.5 * residual.square().sum()`, this VJP is the data-term gradient.

## Second VJP

```python
nonlinear_term = linearized.second_vjp(direction, receiver_cotangent)
```

This applies

$$
(DJ(m)[v])^\top r,
$$

which is the nonlinear-physics contribution to a full Hessian-vector product. It is not the complete Hessian of an arbitrary receiver loss.

For $\Phi(F(m))$, the full product contains

$$
J^\top \Phi'' Jv + (DJ[v])^\top \Phi'.
$$

`tide.workflow.ReceiverObjective.hvp` composes these terms for a receiver objective and supports `mode="gauss_newton"` or `mode="full"`.

## Standard autograd and explicit VJP

These two gradients are mathematically aligned for a scalar objective:

```python
predicted = operator(model).receiver_data
loss = 0.5 * (predicted - observed).square().sum()
loss.backward()
```

```python
with operator.linearize(model) as linearized:
    residual = linearized.primal.receiver_data - observed
    gradient = linearized.vjp(residual)
```

Use standard autograd for ordinary scalar-loss training loops. Use explicit derivative methods for linear operators, adjoint tests, controlled snapshot reuse, and Hessian-vector composition.

## Lifetime and reuse

A linearized session caches its primal state. Reuse it for several JVP or VJP actions at the same background model when supported. Do not mutate the background tensors in place while the session is open. Close the session before changing model values.

After `close()`, access to `primal` or derivative methods raises. Context-manager exit calls `close()` automatically.

## Errors and capability selection

Operator construction validates dimensional agreement between acquisition and solver. Calls also validate material shapes, location bounds, dtype, and supported backend combinations. A native request can fail because of derivative target, storage mode, callbacks, dispersion, device, or missing ABI symbols.

Use `FallbackPolicy.ERROR` when backend identity matters. Use the [capability matrix](/tide-GPR/dev/feature-matrix/) to inspect supported combinations and the [limitations guide](/tide-GPR/guides/limitations/) for current boundaries.
