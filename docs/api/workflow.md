# Workflow

`tide.workflow` contains small helpers for composing solver calls into larger
modeling and inversion scripts. These helpers do not replace `tide.maxwelltm`,
`tide.maxwell3d`, `tide.borntm`, or `tide.born3d`; they handle repeated workflow
glue around those solvers.

## Shot-Batched Modeling

Use `tide.workflow.split_shots` to split the solver shot axis into
contiguous mini-batches:

```python
shot_batches = tide.workflow.split_shots(n_shots, batch_size, device)
for shot_indices in shot_batches:
    pred = tide.maxwelltm(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=source_amplitude[shot_indices],
        source_location=source_location[shot_indices],
        receiver_location=receiver_location[shot_indices],
        pml_width=pml_width,
    )[-1]
```

`index_shots` and `take_shot_batch` provide the same indexing as reusable
helpers. Use `shot_dim=0` for shared-shot tensors shaped `[S, ...]` and
`shot_dim=1` for per-model-shot tensors shaped `[B, S, ...]`.

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

## Callable Runner

`run_shot_batches` runs a solver-like callable over mini-batches and returns
the concatenated receiver data:

```python
receiver = tide.workflow.run_shot_batches(
    tide.maxwelltm,
    n_shots=source_amplitude.shape[0],
    batch_size=8,
    epsilon=epsilon,
    sigma=sigma,
    mu=mu,
    grid_spacing=dx,
    dt=dt,
    source_amplitude=source_amplitude,
    source_location=source_location,
    receiver_location=receiver_location,
    pml_width=pml_width,
)
```

By default, the last item of a solver output tuple is treated as receiver data.
Pass `receiver_selector` when wrapping a callable with a different output shape.
The same helper can wrap `tide.maxwell3d`; 3D-specific options stay in the
solver keyword arguments:

```python
receiver = tide.workflow.run_shot_batches(
    tide.maxwell3d,
    n_shots=source_amplitude.shape[0],
    batch_size=4,
    epsilon=epsilon,
    sigma=sigma,
    mu=mu,
    grid_spacing=dx,
    dt=dt,
    source_amplitude=source_amplitude,
    source_location=source_location,
    receiver_location=receiver_location,
    pml_width=pml_width,
    source_component="ey",
    receiver_component="ey",
)
```

## With `tide.optim`

For optimizer-driven inversion, keep model packing and constraints in the
optimizer objective, and let `backward_shot_batches` own the repeated
mini-batch backward pass:

```python
shot_batches = tide.workflow.split_shots(n_shots, batch_size, device)

def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
    unpack_model(x, epsilon, sigma)

    def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
        batch = tide.workflow.take_shot_batch(
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            shot_indices=shot_indices,
        )
        pred = tide.maxwelltm(
            epsilon=epsilon,
            sigma=sigma,
            mu=mu,
            grid_spacing=dx,
            dt=dt,
            source_amplitude=batch.source_amplitude,
            source_location=batch.source_location,
            receiver_location=batch.receiver_location,
            pml_width=pml_width,
        )[-1]
        return tide.workflow.receiver_mse_loss(
            pred,
            observed,
            shot_indices,
            normalization="all",
        )

    total_loss = tide.workflow.backward_shot_batches(
        batch_loss,
        shot_batches,
        zero_grad=clear_model_grads,
    )

    grad_out[:] = pack_model_grads(epsilon, sigma)
    return total_loss

result = tide.optim.lbfgs_minimize(
    objective,
    x0,
    options=tide.optim.LBFGSOptions(max_iter=10),
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
    options=tide.optim.LBFGSOptions(max_iter=10),
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

## Scope

The workflow module is intentionally narrow:

- no optimizer-state, model-packing, or constraint ownership
- no file I/O, plotting, logging, or device selection policy
- no replacement for the solver's native batched-model support

It is meant to remove repeated shot-batching boilerplate from examples while
keeping experiment-specific choices in user code.
