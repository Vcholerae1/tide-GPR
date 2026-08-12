---
title: "Getting Started"
description: "Install TIDE, verify the backend, and run a minimal 2D Maxwell simulation."
---

## Installation

### From PyPI

```bash
uv pip install tide-GPR
```

Or with pip:

```bash
pip install tide-GPR
```

### From source

```bash
git clone https://github.com/vcholerae1/tide.git
cd tide
uv build
```

:::caution[GPU support]
If you use CUDA, install a CUDA-enabled PyTorch build before installing TIDE.
:::

### Requirements

| Dependency | Version |
|------------|---------|
| Python | ≥ 3.12 |
| PyTorch | ≥ 2.12 |
| CUDA Toolkit | optional, for GPU support |
| CMake | ≥ 3.28, optional, for building from source |

## First Success Criteria

:::tip[Goals for this page]
You are done with this page when you can:

- [x] `import tide` successfully
- [x] verify backend availability
- [x] run one small 2D forward simulation
- [x] identify where to find inversion and API docs next
:::

## Minimal 2D Forward Run

```python title="2d_forward.py"
import torch
import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

ny, nx = 96, 96
epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype)
sigma = torch.zeros_like(epsilon)
mu = torch.ones_like(epsilon)

nt = 300
dt = 4e-11
src = tide.ricker(freq=8e8, length=nt, dt=dt, device=device, dtype=dtype).view(1, 1, nt)
src_loc = torch.tensor([[[20, 48]]], device=device, dtype=torch.long)
rec_loc = torch.tensor([[[20, 60]]], device=device, dtype=torch.long)

model = tide.EMModel(epsilon, sigma, mu)
operator = tide.MaxwellTM(
    tide.Discretization(0.02, dt, boundary=tide.CPML(10)),
    tide.Experiment(tide.Acquisition(src_loc, rec_loc), src),
    execution=tide.ExecutionOptions(fallback=tide.FallbackPolicy.REFERENCE),
)
receivers = operator(model).receiver_data

print(receivers.shape)  # [nt, n_shots, n_receivers]
```

## Optional 3D Preview

```python title="3d_forward.py"
import torch
import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

nz, ny, nx = 32, 32, 32
epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
sigma = torch.zeros_like(epsilon)
mu = torch.ones_like(epsilon)

nt = 200
dt = 4e-11
src = tide.ricker(freq=1e8, length=nt, dt=dt, device=device, dtype=dtype).view(1, 1, nt)
src_loc = torch.tensor([[[16, 16, 16]]], device=device, dtype=torch.long)
rec_loc = torch.tensor([[[16, 16, 20]]], device=device, dtype=torch.long)

model = tide.EMModel(epsilon, sigma, mu)
operator = tide.Maxwell3D(
    tide.Discretization(
        [0.03, 0.03, 0.03],
        dt,
        boundary=tide.CPML(6),
    ),
    tide.Experiment(tide.Acquisition(src_loc, rec_loc), src),
    execution=tide.ExecutionOptions(
        backend=tide.BackendPreference.NATIVE,
        fallback=tide.FallbackPolicy.REFERENCE,
    ),
)
rec = operator(model).receiver_data

print(rec.shape)
```

## Verify Backend Availability

```python
from tide import backend_utils

print("backend available:", backend_utils.is_backend_available())
print("library path:     ", backend_utils.get_library_path())
```

:::note
If the backend is unavailable, TIDE can still run on reference paths for
supported configurations, but performance will be lower.
:::

## What To Read Next

- [API orientation](/tide-GPR/guides/api-orientation/): understand models,
  experiments, operators, and derivative sessions.
- [Modeling guide](/tide-GPR/guides/modeling/): configure forward simulations.
- [Inversion workflow](/tide-GPR/guides/inversion/): connect receiver objectives
  and model updates.
- [API reference](/tide-GPR/api/): inspect the supported public contracts.

## Common Startup Issues

### Shape mismatch

- `source_amplitude` must be `[n_shots, n_sources, nt]`.
- `source_location` and `receiver_location` must be
  `[n_shots, n_points, ndim]`.

### Out-of-bounds indices

Coordinates must satisfy `0 <= index < model_size` for each spatial dimension.

### Instability warning

TIDE adjusts the internal time step using CFL and resamples time signals.
Consider reducing `dt` or coarsening grid spacing.

## Read the first result

The minimal example has one shot, one source, and one receiver. Its result shape
is therefore `[300, 1, 1]`. The first axis is physical time sampled every
`4e-11` seconds. The other axes identify shot and receiver.

The returned tensor is differentiable. Confirm this before building an
inversion:

```python
epsilon = epsilon.clone().requires_grad_(True)
result = operator(tide.EMModel(epsilon, sigma, mu))
objective = result.receiver_data.square().mean()
objective.backward()

print("loss:", float(objective.detach()))
print("gradient shape:", epsilon.grad.shape)
print("finite gradient:", bool(torch.isfinite(epsilon.grad).all()))
```

This is a computational check, not a physical validation. A finite gradient can
still be wrong because of units, geometry, insufficient resolution, or a
misinterpreted component.

## Make one controlled change

Change a compact region of relative permittivity and compare traces:

```python
epsilon_perturbed = epsilon.detach().clone()
epsilon_perturbed[45:55, 50:60] = 7.0

baseline = operator(tide.EMModel(epsilon.detach(), sigma, mu)).receiver_data
perturbed = operator(
    tide.EMModel(epsilon_perturbed, sigma, mu)
).receiver_data

difference = perturbed - baseline
print("maximum trace change:", float(difference.abs().max()))
```

This exercise establishes the core forward-modeling contract: the operator and
experiment stay fixed, the material model changes, and receiver data records
the effect.

## Understand internal sub-stepping

The source and returned receiver tensors use the `dt` supplied to
`Discretization`. TIDE may choose a smaller internal time step to satisfy the
CFL condition:

```python
from tide.cfl import cfl_condition

inner_dt, step_ratio = cfl_condition(
    grid_spacing=0.02,
    dt=dt,
    max_vel=299_792_458.0,
)
print(inner_dt, step_ratio)
```

When `step_ratio > 1`, propagation performs several internal updates per user
sample. TIDE upsamples the source and downsamples receiver data automatically.
The output still contains `nt` samples, but runtime increases with the ratio.

## First physical checks

Before replacing the homogeneous model with survey data:

1. Estimate the direct-arrival time from source-receiver distance and expected
   material velocity.
2. Confirm the trace becomes active near that time.
3. Move the receiver farther away and confirm the arrival is later.
4. Increase CPML width and confirm the direct arrival stays fixed while late
   boundary energy changes.
5. Repeat with a finer grid or higher stencil order and compare phase.

These checks reveal coordinate, unit, boundary, and dispersion errors earlier
than an inversion loop will.
