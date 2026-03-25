# Project Overview

TIDE is a PyTorch-first electromagnetic simulation and inversion toolkit centered on finite-difference time-domain Maxwell solvers.

The project targets research and engineering workflows that need all of the following in one stack:
- numerically stable wave propagation,
- autograd-compatible inversion,
- practical CPU/CUDA performance,
- flexible memory and storage strategies.

## What TIDE Provides
- A 2D TM Maxwell solver (FDTD) with CPML boundaries.
- A 3D Maxwell solver with selectable source and receiver field components.
- Automatic differentiation hooks for inversion workflows.
- CPU and CUDA execution paths.
- Storage modes for wavefield snapshots (device/CPU/disk).

## Core Concepts
- Model parameters: epsilon (relative permittivity), sigma (conductivity), mu (relative permeability).
- Grid spacing and time step with CFL constraints and internal resampling when required.
- Sources and receivers: locations, amplitudes, and batching.
- PML boundary configuration and padding.

Coordinate conventions:
- 2D TM uses [y, x].
- 3D uses [z, y, x].

## Data Flow
1. Define model parameters and grid.
2. Configure sources/receivers.
3. Run forward modeling to generate synthetic data.
4. Compute gradients and update model (inversion).

Pseudo-flow:

```text
model -> stability check (CFL) -> optional source upsample
	-> forward propagation -> receiver traces
	-> (if gradients needed) snapshot storage + backward propagation
	-> gradients wrt model parameters
```

## Repository Layout
- src/tide: Python public API and helpers.
- src/tide/csrc: C/CUDA kernels and CMake build.
- examples: runnable scripts and workflows.
- tests: test suite.
- outputs: generated outputs (not tracked).

Recommended navigation:
1. docs/getting-started.md
2. docs/guides/modeling.md
3. docs/guides/storage.md
4. docs/api/maxwell.md
