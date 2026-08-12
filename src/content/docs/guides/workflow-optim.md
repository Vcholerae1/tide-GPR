---
title: "Workflow + Optim"
description: "Compose shot batches, receiver losses, preconditioners, and torch-native optimizers."
---

`tide.optim` keeps optimizer state as `torch.Tensor` on the same device and
with the same dtype as the initial model. `tide.workflow` owns shot indexing,
receiver losses, and mini-batch gradient accumulation; user code owns model
parameterization and experiment-specific forward modeling.

## Torch-native LBFGS pattern

```python
import torch
import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny, nx = 32, 32
n_shots = 8
batch_size = 2

shot_batches = tide.workflow.split_shots(
    n_shots,
    batch_size,
    device=device,
)

base_operator = tide.MaxwellTM(
    tide.Discretization(dx, dt, boundary=tide.CPML(8)),
    tide.Experiment(
        tide.Acquisition(source_location, receiver_location),
        source_amplitude,
    ),
)

def objective(x: torch.Tensor) -> tuple[float, torch.Tensor]:
    epsilon = x.detach().clone().requires_grad_(True)
    model = tide.EMModel(epsilon, sigma, mu)
    predicted = tide.workflow.run_shot_batches(
        base_operator,
        model,
        batch_size=batch_size,
    )
    loss = torch.mean((predicted - observed) ** 2)
    loss.backward()
    if epsilon.grad is None:
        raise RuntimeError("objective did not produce a gradient")
    return float(loss.detach()), epsilon.grad.detach()

x0 = torch.full((ny, nx), 3.5, device=device)
result = tide.optim.lbfgs_minimize(
    objective,
    x0,
    lower_bounds=1.0,
    upper_bounds=9.0,
    options=tide.optim.LBFGSOptions(
        stopping=tide.optim.StoppingCriteria(
            max_iter=10,
            max_evaluations=50,
            gtol=1e-6,
        )
    ),
)
epsilon_inverted = result.x
```

The objective returns a scalar loss and a gradient tensor with the same shape,
dtype, and device as `x`. The loss may be a Python float, which matches
`backward_shot_batches`, or a scalar tensor. Optimizer updates are detached and
do not retain an autograd graph across iterations.

Box constraints use projected-gradient convergence and a projected Armijo
search. Unconstrained problems use strong-Wolfe search by default.

## Preconditioners

Workflow preconditioners are torch-native and stay on the model device:

```python
diag = tide.workflow.curvature_preconditioner_diagonal(
    curvature,
    inactive_mask=air_mask,
    smooth_sigma=3.0,
    damping=5e-2,
    power=0.5,
    clip_min=0.3,
    clip_max=3.0,
    blend=0.7,
)
preconditioner = tide.workflow.diagonal_preconditioner(diag)

result = tide.optim.lbfgs_minimize(
    objective,
    x0,
    preconditioner=preconditioner,
)
```

For two coupled fields, concatenate them into one tensor and build a symmetric
block preconditioner:

```python
block = tide.workflow.curvature_preconditioner_block(
    h_ee,
    h_ss,
    h_es,
    inactive_mask=air_mask,
    smooth_sigma=3.0,
    damping=5e-2,
    power=0.5,
    offdiag_correlation_max=0.8,
)
preconditioner = tide.workflow.block_preconditioner(block)
```

The factors, model, and vectors must share dtype and device.

## Stopping and history

`StoppingCriteria` separates gradient, function-change, step-change, iteration,
and objective-evaluation limits. Reaching an iteration or evaluation limit is
not reported as successful convergence.

Trace retention is disabled by default. Enable scalar history explicitly:

```python
options = tide.optim.LBFGSOptions(
    trace=tide.optim.TraceOptions(record=True),
)
```

Full model and gradient snapshots require `store_tensors=True` and should be
sampled with `snapshot_interval` for large inverse problems.

## Objective contract

Every `tide.optim` method expects an objective that is deterministic for a
given input tensor and returns both value and gradient:

```python
def objective(x: torch.Tensor) -> tuple[float, torch.Tensor]:
    ...
    return value, gradient
```

The returned gradient must have the same shape, dtype, and device as `x`.
Detach it before returning so optimizer history does not retain the Maxwell
autograd graph. If the objective uses random shot subsets, either fix the
subset during a line search or use an optimizer designed for stochastic
gradients.

## Packing coupled material fields

When optimizing permittivity and conductivity together, make the packing
convention explicit:

```python
def pack(epsilon: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return torch.stack((epsilon, sigma))

def unpack(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[0], x[1]
```

Raw stacking assumes comparable optimizer scales. In practice, conductivity
and relative permittivity can differ by orders of magnitude. Transform each
field to a dimensionless optimization variable or use a tested block
preconditioner. Bounds must be expressed in the same packed coordinates as the
optimizer input.

## Line searches and repeated propagation

LBFGS and truncated Newton methods may evaluate the objective several times per
iteration. Each trial runs the Maxwell operator again. A reported iteration
count therefore does not equal a propagation count. Use
`StoppingCriteria.max_evaluations` to bound this cost and record objective
evaluations in addition to iterations.

Reset LBFGS history when changing frequency band, loss normalization,
regularization weight, shot subset, or parameterization. Those changes define
a different local objective, so old curvature pairs may no longer be useful.

## Mini-batch normalization

Choose whether the objective is a mean over the full dataset, a mean over each
batch, or a sum. For unequal final batch sizes, averaging each batch and then
summing gives the last small batch too much weight. Weight batch losses by
their number of samples when reproducing a global mean.

`receiver_mse_loss(..., normalization="all")` uses full-dataset normalization
when the required global size is available. Keep that convention consistent
between single-process and distributed runs.

## Preconditioner safeguards

Curvature proxies must be non-negative and finite. Apply masks before computing
quantiles, add damping before inversion, and clip extreme scaling when a small
curvature value would create an excessive step. For a 2 by 2 block
preconditioner, verify that the damped determinant stays positive and that both
parameter fields use the same packing order everywhere.

Always compare the preconditioned direction with the unpreconditioned gradient
on a small problem. A preconditioner should improve scaling, not change the
objective or silently update inactive cells.

## Interpreting termination

An optimizer result distinguishes convergence from exhaustion:

- A gradient, function-change, or step-change tolerance can indicate local
  convergence.
- `max_iter` and `max_evaluations` are resource limits, not success criteria.
- A failed line search can indicate poor scaling, non-finite trials, or an
  inconsistent objective.

Inspect the result status and trace instead of assuming the final tensor is a
valid solution.
