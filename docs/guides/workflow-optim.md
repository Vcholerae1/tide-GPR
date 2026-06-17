# Workflow + Optim

This page shows the compact pattern for using `tide.workflow` with
`tide.optim`. The key split is:

- `tide.workflow` owns shot indexing, receiver concatenation, receiver loss,
  and mini-batch gradient accumulation.
- `tide.optim` owns the CPU-state optimizer loop.
- user code owns model packing, constraints, and experiment-specific forward or
  filtering inside the batch loss callable.

## Compact LBFGS Pattern

```python
import numpy as np
import torch

import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ny, nx = 32, 32
n_shots = 8
nt = 200
dx = 0.02
dt = 4e-11
batch_size = 2

wavelet = tide.ricker(80e6, nt, dt, device=device)
source_amplitude = tide.workflow.expand_source_amplitude(wavelet, n_shots)
acquisition = tide.workflow.line_acquisition_2d(
    torch.arange(n_shots, device=device) + 8,
    torch.arange(n_shots, device=device) + 12,
    source_depth=4,
    receiver_mode="paired",
)
source_location = acquisition.source_location
receiver_location = acquisition.receiver_location
shot_batches = tide.workflow.split_shots(n_shots, batch_size, device)

epsilon_true = torch.full((ny, nx), 4.0, device=device)
sigma = torch.zeros_like(epsilon_true)
mu = torch.ones_like(epsilon_true)

with torch.no_grad():
    observed = tide.workflow.run_shot_batches(
        tide.maxwelltm,
        n_shots=n_shots,
        batch_size=batch_size,
        epsilon=epsilon_true,
        sigma=sigma,
        mu=mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=8,
    )

epsilon_shape = (ny, nx)

def unpack_epsilon(x: np.ndarray) -> torch.Tensor:
    epsilon_np = x.reshape(epsilon_shape).astype(np.float32, copy=False)
    return torch.from_numpy(epsilon_np).to(device=device).requires_grad_(True)

def pack_epsilon_grad(epsilon: torch.Tensor, grad_out: np.ndarray) -> None:
    grad = torch.zeros_like(epsilon) if epsilon.grad is None else epsilon.grad
    grad_out[:] = grad.detach().cpu().numpy().reshape(-1)

def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
    epsilon = unpack_epsilon(x)

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

    total_loss = tide.workflow.backward_shot_batches(batch_loss, shot_batches)

    pack_epsilon_grad(epsilon, grad_out)
    return total_loss

x0 = np.full(ny * nx, 3.5, dtype=np.float32)
result = tide.optim.lbfgs_minimize(
    objective,
    x0,
    lower_bounds=np.full_like(x0, 1.0),
    upper_bounds=np.full_like(x0, 9.0),
    options=tide.optim.LBFGSOptions(max_iter=10),
)
epsilon_inverted = torch.from_numpy(result.x.reshape(epsilon_shape)).to(device)
```

The important part is that `backward_shot_batches` runs one mini-batch backward
at a time. That avoids building a single graph for all shots while keeping
gradient accumulation out of the inversion logic.

## When To Use `run_shot_batches`

Use `run_shot_batches` when you only need forward receiver data, for example:

- generating observed or synthetic reference data;
- evaluating a fixed model without gradients;
- running a batched forward pass for plotting or diagnostics.

Inside optimizer objectives, prefer `backward_shot_batches` around a
`batch_loss(shot_indices)` callable so each mini-batch can backpropagate before
the next graph is built.

## Adding A Diagonal Preconditioner

When an example accumulates squared gradients or another diagonal curvature
proxy, use `backward_shot_batches(..., zero_each_batch=True)` to sample
per-batch gradients and hand the normalization to `tide.workflow`:

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
    options=tide.optim.LBFGSOptions(max_iter=10),
)
```

This keeps experiment-specific choices, such as which loss builds `curvature`,
outside the workflow module while removing the repeated smoothing, scaling,
clipping, and optimizer callback glue.

For coupled two-parameter inversions, build a symmetric block preconditioner
from the three curvature proxies:

```python
block = tide.workflow.curvature_preconditioner_block(
    h_ee,
    h_ss,
    h_es,
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
