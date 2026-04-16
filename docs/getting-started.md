# Getting Started

## Installation

=== "From PyPI (recommended)"

    ```bash
    uv pip install tide-GPR
    ```

    Or with pip:

    ```bash
    pip install tide-GPR
    ```

=== "From Source"

    ```bash
    git clone https://github.com/vcholerae1/tide.git
    cd tide
    uv build
    ```

!!! warning "GPU Support"
    If you use CUDA, install a CUDA-enabled PyTorch build before installing TIDE.

### Requirements

| Dependency | Version |
|------------|---------|
| Python | ≥ 3.12 |
| PyTorch | ≥ 2.9.1 |
| CUDA Toolkit | optional, for GPU support |
| CMake | ≥ 3.28, optional, for building from source |

## First Success Criteria

!!! success "Goals for this page"
    You are done with this page when you can:

    - [x] `import tide` successfully
    - [x] verify backend availability
    - [x] run one small 2D forward simulation
    - [x] identify where to find inversion and API docs next

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

*_, receivers = tide.maxwelltm(
    epsilon=epsilon,
    sigma=sigma,
    mu=mu,
    grid_spacing=0.02,
    dt=dt,
    source_amplitude=src,
    source_location=src_loc,
    receiver_location=rec_loc,
    pml_width=10,
)

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

*_, rec = tide.maxwell3d(
    epsilon=epsilon,
    sigma=sigma,
    mu=mu,
    grid_spacing=[0.03, 0.03, 0.03],
    dt=dt,
    source_amplitude=src,
    source_location=src_loc,
    receiver_location=rec_loc,
    pml_width=6,
    python_backend=False,
)

print(rec.shape)
```

## Verify Backend Availability

```python
from tide import backend_utils

print("backend available:", backend_utils.is_backend_available())
print("library path:     ", backend_utils.get_library_path())
```

!!! note
    If the backend is unavailable, TIDE can still run on Python fallback paths for supported configurations, but performance will be lower.

## What To Read Next

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } **API Orientation**

    ---
    [:octicons-arrow-right-24: guides/api-orientation.md](guides/api-orientation.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Modeling Guide**

    ---
    [:octicons-arrow-right-24: guides/modeling.md](guides/modeling.md)

-   :material-sine-wave:{ .lg .middle } **Inversion Workflow**

    ---
    [:octicons-arrow-right-24: guides/inversion.md](guides/inversion.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---
    [:octicons-arrow-right-24: api/index.md](api/index.md)

</div>

## Common Startup Issues

??? warning "Shape mismatch"
    - `source_amplitude` must be `[n_shots, n_sources, nt]`
    - `source_location` and `receiver_location` must be `[n_shots, n_points, ndim]`

??? warning "Out-of-bounds indices"
    Coordinates must satisfy `0 <= index < model_size` for each spatial dimension.

??? warning "Instability warning"
    TIDE auto-adjusts the internal time step using CFL and resamples time signals. Consider reducing `dt` or coarsening grid spacing.
