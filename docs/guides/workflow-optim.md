# Workflow + Optim

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

def objective(x: torch.Tensor) -> tuple[float, torch.Tensor]:
    epsilon = x.detach().clone().requires_grad_(True)

    def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
        batch = tide.workflow.take_shot_batch(
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            shot_indices=shot_indices,
        )
        predicted = tide.maxwelltm(
            epsilon=epsilon,
            sigma=sigma,
            mu=mu,
            grid_spacing=dx,
            dt=dt,
            source_amplitude=batch.source_amplitude,
            source_location=batch.source_location,
            receiver_location=batch.receiver_location,
            pml_width=8,
        )[-1]
        return tide.workflow.receiver_mse_loss(
            predicted,
            observed,
            shot_indices,
            normalization="all",
        )

    total_loss = tide.workflow.backward_shot_batches(
        batch_loss,
        shot_batches,
    )
    if epsilon.grad is None:
        raise RuntimeError("objective did not produce a gradient")
    return total_loss, epsilon.grad.detach()

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
