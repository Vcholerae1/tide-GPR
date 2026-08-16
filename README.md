# TIDE

**[Documentation](https://vcholerae1.github.io/tide-GPR/)**

**T**orch-based **I**nversion & **D**evelopment **E**ngine

[![Test coverage](https://codecov.io/gh/Vcholerae1/tide-GPR/branch/main/graph/badge.svg)](https://codecov.io/gh/Vcholerae1/tide-GPR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TIDE is a PyTorch-first electromagnetic FDTD library for forward modeling and full-waveform inversion. It provides differentiable 2D and 3D Maxwell solvers, native C/CUDA kernels, and configurable snapshot storage for memory-intensive gradient calculations.

## Capabilities

| Capability | API | Status |
| --- | --- | --- |
| 2D TM forward/inverse modeling | `tide.MaxwellTM` | Stable |
| 3D forward/inverse modeling | `tide.Maxwell3D` | Stable with constraints |
| JVP, VJP, and second VJP | `operator.linearize(model)` | Stable |
| Snapshot storage | `storage_mode` | Device, CPU, disk, none, or auto |
| Snapshot compression | `storage_compression` | Optional BF16 compression |
| Debye dispersion | `DebyeDispersion` | Advanced |

TIDE also includes PML boundaries, staggered-grid operators, callbacks, CFL resampling, shot batching, and inversion workflow helpers. Check the [limitations guide](src/content/docs/guides/limitations.md) before scaling up 3D or inversion workloads.

## Installation

TIDE requires Python 3.12 or newer and PyTorch 2.12 or newer.

Install the package from PyPI:

```bash
uv pip install tide-GPR
```

You can also use `pip`:

```bash
pip install tide-GPR
```

For GPU use, install the [PyTorch build](https://pytorch.org/get-started/locally/) that matches your CUDA environment before installing TIDE.

### Build from source

Building from source requires CMake 3.28 or newer. A CUDA Toolkit is optional.

```bash
git clone https://github.com/vcholerae1/tide.git
cd tide
uv build
```

See the [build guide](docs/dev/build.md) for native-backend builds and troubleshooting.

## Quick start

This example runs a small 2D TM forward simulation on CUDA when available and falls back to CPU otherwise:

```python
import torch
import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = torch.full((96, 96), 4.0, device=device)
model = tide.EMModel(
    epsilon=epsilon,
    sigma=torch.zeros_like(epsilon),
    mu=torch.ones_like(epsilon),
)

nt, dt = 300, 4e-11
experiment = tide.Experiment(
    acquisition=tide.Acquisition(
        source_location=torch.tensor([[[20, 48]]], device=device),
        receiver_location=torch.tensor([[[20, 60]]], device=device),
    ),
    source_amplitude=tide.ricker(8e8, nt, dt, device=device).view(1, 1, nt),
)
operator = tide.MaxwellTM(
    tide.Discretization(
        spacing=0.02,
        dt=dt,
        boundary=tide.CPML(width=10),
    ),
    experiment,
    execution=tide.ExecutionOptions(fallback=tide.FallbackPolicy.REFERENCE),
)

result = operator(model)
print(result.receiver_data.shape)  # [nt, n_shots, n_receivers]
```

## Documentation

Start with the path that matches your task:

- [Getting started](src/content/docs/getting-started.md): installation, backend checks, and a first 2D simulation
- [API orientation](src/content/docs/guides/api-orientation.md): models, experiments, operators, and derivative sessions
- [Modeling](src/content/docs/guides/modeling.md): sources, receivers, boundaries, and tensor layouts
- [Inversion](src/content/docs/guides/inversion.md): losses, backpropagation, and optimizer workflows
- [Configuration](src/content/docs/guides/configuration.md): storage, callbacks, backends, and CFL controls
- [API reference](src/content/docs/api/index.md): public contracts and operators

Before relying on advanced configurations, review the [known limitations](src/content/docs/guides/limitations.md) and [verification guide](src/content/docs/guides/verification.md).

## Development

Install the development dependencies and run the test suite:

```bash
uv sync --group dev
uv run pytest
```

Preview the documentation:

```bash
npm install
npm run dev
```

Issues and pull requests are welcome.

## Citation

If you use TIDE in your research, cite:

```bibtex
@software{tide2025,
  author = {Vcholerae1},
  title = {TIDE: Torch-based Inversion \& Development Engine},
  year = {2025},
  url = {https://github.com/vcholerae1/tide}
}
```

## Acknowledgments

TIDE includes code derived from [Deepwave](https://github.com/ar4/deepwave) by Alan Richardson.

## License

TIDE is available under the [MIT License](LICENSE).
