# Project Overview

TODO: Add a concise project description and scope statement.

## What TIDE Provides
- A 2D TM Maxwell solver (FDTD) with CPML boundaries.
- Automatic differentiation hooks for inversion workflows.
- CPU and CUDA execution paths.
- Storage modes for wavefield snapshots (device/CPU/disk).

## Core Concepts
- Model parameters: epsilon (relative permittivity), sigma (conductivity), mu (relative permeability).
- Grid spacing and time step (dx/dy, dt) with CFL constraints.
- Sources and receivers: locations, amplitudes, and batching.
- PML boundary configuration and padding.

## Data Flow
1. Define model parameters and grid.
2. Configure sources/receivers.
3. Run forward modeling to generate synthetic data.
4. Compute gradients and update model (inversion).
TODO: Add diagram or pseudo-code flow.

## Repository Layout
- src/tide: Python public API and helpers.
- src/tide/csrc: C/CUDA kernels and CMake build.
- examples: runnable scripts and workflows.
- tests: test suite.
- outputs: generated outputs (not tracked).
