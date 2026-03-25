# Getting Started

## Installation

### From PyPI

```bash
uv pip install tide-GPR
```

or

```bash
pip install tide-GPR
```

If you use CUDA, install a CUDA-enabled PyTorch build first.

### From Source

```bash
git clone https://github.com/vcholerae1/tide.git
cd tide
uv build
```

Requirements:
- Python >= 3.12
- PyTorch >= 2.9.1
- CUDA Toolkit (optional, for GPU support)
- CMake >= 3.28 (optional, for building from source)

## Quick Start

### 2D TM forward modeling (minimal)

```python
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

### 3D forward modeling (minimal)

```python
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
print("library path:", backend_utils.get_library_path())
```

If backend is unavailable, TIDE can still run on Python fallback paths for supported configurations, but performance will be lower.

## Next Steps
- Read guides/modeling.md and guides/sources-receivers.md first.
- Then read guides/storage.md and guides/performance.md for scaling.
- Run one script from examples/ and compare outputs with docs/examples/*.md.
- Use docs/api/index.md as function-level reference.

## Common Startup Issues

1. Shape mismatch:
	- Ensure source_amplitude is [n_shots, n_sources, nt].
	- Ensure source_location and receiver_location have shape [n_shots, n_points, ndim].
2. Out-of-bounds indices:
	- Coordinates must satisfy 0 <= index < model size for each spatial dimension.
3. Instability warning:
	- TIDE auto-adjusts internal time step using CFL and resamples time signals.
	- Consider reducing dt or coarsening grid spacing.
