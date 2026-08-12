---
title: "Workflow"
description: "Shot batching, receiver objectives, distributed helpers, and preconditioners."
---

`tide.workflow` composes structured Maxwell operators into shot-batched,
distributed, loss, and optimization workflows. Physics and derivative
execution remain owned by `MaxwellTM`, `Maxwell3D`, and their linearized
sessions.

## Shot-Batched Modeling

Use `tide.workflow.split_shots` for custom mini-batch loops, or use
`run_shot_batches` directly:

```python
receiver = tide.workflow.run_shot_batches(
    operator,
    model,
    batch_size=8,
)
```

`run_shot_batches` rebuilds only the experiment geometry for each batch. The
model, discretization, execution policy, storage policy, and named result
contract remain unchanged.

`index_shots` and `take_shot_batch` expose the underlying indexing when a
custom objective needs direct control. Use `shot_dim=0` for shared-shot tensors
shaped `[S, ...]` and `shot_dim=1` for per-model-shot tensors shaped
`[B, S, ...]`.

Use `expand_source_amplitude` for the common wavelet-to-shot-amplitude step:

```python
wavelet = tide.ricker(freq, nt, dt, device=device)
source_amplitude = tide.workflow.expand_source_amplitude(wavelet, n_shots)
```

Use `line_acquisition_2d` when a script only needs line coordinates:

```python
acquisition = tide.workflow.line_acquisition_2d(
    source_x=torch.arange(n_shots, device=device) + 8,
    receiver_x=torch.arange(n_shots, device=device) + 12,
    source_depth=4,
    receiver_mode="paired",
)
source_location = acquisition.source_location
receiver_location = acquisition.receiver_location
```

## Receiver Concatenation

`merge_receiver_batches` concatenates receiver chunks along the TIDE shot axis.
It infers:

- `[nt, S, R]` -> shot axis 1
- `[nt, B, S, R]` -> shot axis 2

This keeps shared-model and batched-model outputs aligned with the solver API.

## Structured Runner

`run_shot_batches(operator, model, batch_size=...)` accepts `MaxwellTM` or
`Maxwell3D` and returns concatenated receiver data. It preserves autograd
connections to the model tensors.

For loss-specific first- and second-order derivatives, use
`ReceiverObjective` with a linearized operator:

```python
objective = tide.workflow.ReceiverObjective(observed)
with operator.linearize(model) as linearized:
    gradient = objective.gradient(linearized)
    hvp = objective.hvp(linearized, direction, mode="gauss_newton")
```


## With `tide.optim`

Optimizer state, gradients, bounds, and preconditioners remain as Torch tensors
on the model device. A structured operator can be evaluated in shot batches
while retaining autograd:

```python
def objective(x: torch.Tensor) -> tuple[float, torch.Tensor]:
    packed = x.detach().clone().requires_grad_(True)
    epsilon, sigma = unpack_model(packed)
    model = tide.EMModel(epsilon, sigma, mu)
    pred = tide.workflow.run_shot_batches(
        operator,
        model,
        batch_size=batch_size,
    )
    loss = torch.mean((pred - observed) ** 2)
    loss.backward()
    if packed.grad is None:
        raise RuntimeError("objective did not produce a gradient")
    return float(loss.detach()), packed.grad.detach()

result = tide.optim.lbfgs_minimize(
    objective,
    x0,
    options=tide.optim.LBFGSOptions(
        stopping=tide.optim.StoppingCriteria(max_iter=10),
    ),
)
```

## Preconditioners

Use `curvature_preconditioner_diagonal` for the common diagonal GN-style
preconditioner pattern used in examples: accumulate a non-negative curvature
proxy such as squared gradients, optionally smooth it, normalize it, invert it
with damping, clip the scaling, and zero inactive cells.

```python
curvature = torch.zeros_like(epsilon)

def record_curvature(_shot_indices: torch.Tensor, _loss: torch.Tensor) -> None:
    if epsilon.grad is not None:
        grad = torch.nan_to_num(epsilon.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        curvature.add_(grad.square())

tide.workflow.backward_shot_batches(
    objective_batch,
    shot_batches,
    zero_grad=clear_model_grads,
    zero_each_batch=True,
    after_backward=record_curvature,
)

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
    options=tide.optim.LBFGSOptions(
        stopping=tide.optim.StoppingCriteria(max_iter=10),
    ),
)
```

For two coupled parameter fields, accumulate the three symmetric block proxies
and use `curvature_preconditioner_block`:

```python
block = tide.workflow.curvature_preconditioner_block(
    curvature_ee,
    curvature_ss,
    curvature_es,
    inactive_mask=air_mask,
    smooth_sigma=3.0,
    damping=5e-2,
    power=0.5,
    clip_min=0.3,
    clip_max=3.0,
    offdiag_correlation_max=0.8,
    blend=0.7,
)
preconditioner = tide.workflow.block_preconditioner(block)
```

## Receiver losses

`receiver_mse_loss` selects the observed shots corresponding to a predicted
mini-batch and applies an explicit normalization policy. Use `"batch"` when
each mini-batch objective should be a local mean. Use `"all"` when accumulated
batches must reproduce a full-dataset mean.

```python
loss = tide.workflow.receiver_mse_loss(
    predicted,
    observed,
    shot_indices,
    normalization="all",
)
```

`receiver_sinkhorn_loss` and `receiver_gsot_loss` compare traces as point
clouds in time-amplitude space. Their `dt`, sparse sampling, transport power,
and scale parameters affect the objective itself. Validate their gradient,
runtime, and trace normalization on a controlled example before replacing
MSE.

Shard-aware variants accept the global shot identifiers owned by one rank and
the local observed-data shard. They keep data indexing and loss normalization
consistent without gathering every receiver trace for each objective call.

## Distributed shot sharding

`init_distributed` initializes one worker per selected device and returns a
`DistributedContext`. `rank_shot_indices` and `split_rank_shots` assign global
shot identifiers deterministically:

```python
context = tide.workflow.init_distributed(enabled=True)
shot_batches = tide.workflow.split_rank_shots(
    n_shots,
    batch_size=4,
    context=context,
)
```

After local backward passes, call `all_reduce_gradients` to sum model gradients
across ranks. Scale local objectives so the reduced gradient matches the
intended global sum or mean. `gather_receiver_shards` is only needed when rank
zero requires the complete receiver tensor for output or diagnostics.

```python
tide.workflow.all_reduce_gradients(
    [epsilon, sigma],
    context=context,
)
```

`destroy_distributed` tears down a process group owned by the context. Put
cleanup in `finally` when a worker can fail during propagation.

## ReceiverObjective

`ReceiverObjective` separates a scalar receiver loss from Maxwell physics:

```python
objective = tide.workflow.ReceiverObjective(
    observed_data,
    loss=custom_receiver_loss,
)
```

`gradient(linearized)` applies the loss cotangent through the operator VJP.
`hvp(linearized, direction, mode="gauss_newton")` computes the positive
semi-definite Gauss-Newton approximation when the receiver loss supports the
required second action. `mode="full"` adds the nonlinear second-VJP term.

The observed tensor must match the linearized primal receiver layout. The loss
callable owns reduction and scaling, so changing a mean to a sum scales both
gradient and Hessian products.

## Scope

The workflow module is intentionally narrow:

- no optimizer-state, model-packing, or constraint ownership
- no file I/O, plotting, logging, or device selection policy
- no replacement for the solver's native batched-model support

It is meant to remove repeated shot-batching boilerplate from examples while
keeping experiment-specific choices in user code.
