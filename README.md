# TIDE

**T**orch-based **I**nversion & **D**evelopment **E**ngine

TIDE is a PyTorch-based library for high-frequency electromagnetic wave propagation and inversion, built on Maxwell's equations. It provides CPU and CUDA implementations for forward modeling, gradient computation, and full-waveform inversion workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Maxwell Equation Solvers**:
  - 2D TM mode propagation
  - 3D Maxwell propagation
- **Automatic Differentiation**: Gradient support through PyTorch's autograd hooks
- **High Performance**: Optimized C/CUDA kernels for critical operations
- **Flexible Storage**: Device/CPU/disk snapshot modes for gradient computation
- **Staggered Grid**: Industry-standard FDTD staggered grid implementation
- **PML Boundaries**: Perfectly Matched Layer absorbing boundaries
- **Snapshot Compression**: Optional BF16 snapshot compression on the default path

## Feature Matrix

| Capability | Entry Point | Status | Notes |
| --- | --- | --- | --- |
| 2D TM forward modeling | `tide.maxwelltm` | Stable | Primary onboarding path |
| 2D TM inversion / autograd | `tide.maxwelltm`, `MaxwellTM` | Stable | Uses PyTorch autograd |
| 3D forward modeling | `tide.maxwell3d` | Stable | Supports component selection |
| 3D inversion / gradients | `tide.maxwell3d`, `Maxwell3D` | Stable with constraints | Check the limitations guide before scaling up |
| Snapshot storage modes | `storage_mode=*` | Stable | Device, CPU, disk, none, and auto |
| Callbacks | `forward_callback`, `backward_callback` | Stable | Keep callback work lightweight |
| Debye dispersion | `DebyeDispersion` | Advanced | Requires explicit time-step validation |

## Installation

### From PyPI

Ensure you have proper PyTorch installation with CUDA binding for your system.

For CUDA environments, you may need to install a CUDA-enabled PyTorch build first:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
``` 
The cu128 tag is for CUDA 12.8. Replace it based on your CUDA version.

Then install TIDE via uv or pip:

```bash
uv pip install tide-GPR
```

or

```bash
pip install tide-GPR
```


### From Source

We recommend using [uv](https://github.com/astral-sh/uv) for building:

```bash
git clone https://github.com/vcholerae1/tide.git
cd tide
uv build
```

To rebuild only the native backend during development:

```bash
bash scripts/build_csrc.sh
```

**Requirements:**
- Python >= 3.12
- PyTorch >= 2.9.1
- CUDA Toolkit (optional, for GPU support)
- CMake >= 3.28 (optional, for building from source)

## Quick Start

```python
import torch
import tide

# Create a simple model
nx, ny = 200, 100
epsilon = torch.ones(ny, nx) * 4.0  # Relative permittivity
sigma = torch.zeros_like(epsilon)    # Conductivity (S/m)
mu = torch.ones_like(epsilon)        # Relative permeability
epsilon[50:, :] = 9.0  # Add a layer

# Set up source
source_amplitude = tide.ricker(
    freq=4e8,           # 400 MHz
    length=1000,
    dt=1e-11,
    peak_time=5e-10
).reshape(1, 1, -1)

source_location = torch.tensor([[[10, 100]]], dtype=torch.long)
receiver_location = torch.tensor([[[10, 150]]], dtype=torch.long)

# Run forward simulation
*_, receiver_data = tide.maxwelltm(
    epsilon=epsilon,
  sigma=sigma,
  mu=mu,
  grid_spacing=0.01,
    dt=1e-11,
  source_amplitude=source_amplitude,
  source_location=source_location,
  receiver_location=receiver_location,
    pml_width=10
)

print(f"Recorded data shape: {receiver_data.shape}")
```

## Core Modules

- `tide.maxwelltm`: 2D TM solver
- `tide.maxwell3d`: 3D solver
- `tide.wavelets`: Source wavelet generation
- `tide.callbacks`: Callback state and factories
- `tide.storage`: Snapshot storage and compression controls
- `tide.resampling`: CFL resampling helpers
- `tide.cfl`: CFL condition helper
- `tide.padding`: Padding and interior masking helpers
- `tide.validation`: Input validation helpers

## Precision and Storage

Storage and precision controls:

```python
out = tide.maxwelltm(
    epsilon,
    sigma,
    mu,
    grid_spacing=0.02,
    dt=4e-11,
    source_amplitude=src,
    source_location=src_loc,
    receiver_location=rec_loc,
    storage_mode="auto",
    storage_compression="bf16",
)
```

Notes:
- `storage_mode` accepts device, cpu, disk, none, and auto.
- `storage_compression` accepts none or bf16 for float32 snapshot storage on
  Maxwell and Born native paths that save backward intermediates.

## Documentation

Recommended reading path:

1. `docs/getting-started.md` for installation and the first 2D forward run
2. `docs/guides/api-orientation.md` for choosing between `tide.maxwelltm`, `tide.maxwell3d`, `MaxwellTM`, and `Maxwell3D`
3. `docs/guides/modeling.md` and `docs/guides/inversion.md` for forward modeling and inversion workflows
4. `docs/guides/configuration.md` for storage, callbacks, backend, and CFL-related controls
5. `docs/guides/limitations.md` and `docs/guides/verification.md` before enabling advanced features broadly

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project includes code derived from [Deepwave](https://github.com/ar4/deepwave) by Alan Richardson. We gratefully acknowledge the foundational work that made TIDE possible.

## Citation

If you use TIDE in your research, please cite:

```bibtex
@software{tide2025,
  author = {Vcholerae1},
  title = {TIDE: Torch-based Inversion \& Development Engine},
  year = {2025},
  url = {https://github.com/vcholerae1/tide}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
