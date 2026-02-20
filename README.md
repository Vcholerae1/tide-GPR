# TIDE

**T**orch-based **I**nversion & **D**evelopment **E**ngine

TIDE is a PyTorch-based library for  high frequa electromagnetic wave propagation and inversion, built on Maxwell's equations. It provides efficient CPU and CUDA implementations for forward modeling, gradient computation, and full waveform inversion (FWI).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Maxwell Equation Solvers**: 
  - 2D TM mode propagation (`MaxwellTM`)
  - Other propagation is on the way 
- **Automatic Differentiation**: Gradient support through PyTorch's autograd
- **High Performance**: Optimized C/CUDA kernels for critical operations
- **Flexible Storage**: Multiple storage modes for gradient computation (memory/disk/optional BF16 compressed)
- **Staggered Grid**: Industry-standard FDTD staggered grid implementation
- **PML Boundaries**: Perfectly Matched Layer absorbing boundaries
- **Mixed Precision (CUDA)**: `compute_dtype="fp16"` with internal nondimensional scaling

## Installation

### From PyPI

Ensure you have proper PyTorch installation with CUDA binding for your system.

Maybe you need 
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
``` 
cu128 is for CUDA 12.8, change it according to your CUDA version.

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
epsilon[50:, :] = 9.0  # Add a layer

# Set up source
source_amplitudes = tide.ricker(
    freq=1e9,           # 1 GHz
    nt=1000,
    dt=1e-11,
    peak_time=5e-10
).reshape(1, 1, -1)

source_locations = torch.tensor([[[10, 100]]])
receiver_locations = torch.tensor([[[10, 150]]])

# Run forward simulation
receiver_data = tide.maxwelltm(
    epsilon=epsilon,
    dx=0.01,
    dt=1e-11,
    source_amplitudes=source_amplitudes,
    source_locations=source_locations,
    receiver_locations=receiver_locations,
    pml_width=10
)

print(f"Recorded data shape: {receiver_data.shape}")
```

## Core Modules

- **`tide.maxwelltm`**: 2D TM mode Maxwell solver
- **`tide.wavelets`**: Source wavelet generation (Ricker, etc.)
- **`tide.staggered`**: Staggered grid finite difference operators
- **`tide.callbacks`**: Callback state and factories
- **`tide.resampling`**: Upsampling/downsampling utilities
- **`tide.cfl`**: CFL condition helpers
- **`tide.padding`**: Padding and interior masking helpers
- **`tide.validation`**: Input validation helpers
- **`tide.storage`**: Gradient checkpointing and storage management

## Mixed Precision

`tide.maxwelltm` provides a mixed-precision interface:

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
    compute_dtype="fp16",      # "fp32" (default) or "fp16"
    mp_mode="throughput",      # "throughput" | "balanced" | "robust"
)
```

Notes:
- `compute_dtype="fp16"` is currently CUDA-only.
- External API remains SI-compatible (`epsilon_r`, `mu_r`, `sigma`, `dx/dy`, `dt`).
- Internal updates use nondimensional scaling for better reduced-precision stability.
- Current C/CUDA kernels execute in fp32 with fp16 mixed-precision I/O/scaling.

## Examples

See the [`examples/`](examples/) directory for complete workflows:

- [`example_multiscale_filtered.py`](examples/example_multiscale_filtered.py): Multi-scale FWI with frequency filtering
- [`example_multiscale_random_sources.py`](examples/example_multiscale_random_sources.py): FWI with random source encoding
- [`example_wavefield_animation.py`](examples/example_wavefield_animation.py): Visualize wave propagation

## Documentation

For detailed API documentation and tutorials, visit: [Documentation]() *(coming soon)*

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
