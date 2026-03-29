# Project Overview

TIDE is a PyTorch-first electromagnetic modeling and inversion library built around finite-difference time-domain Maxwell solvers.

## What You Can Do With TIDE

- Run 2D TM forward simulations with `tide.maxwelltm`
- Run 3D forward simulations with `tide.maxwell3d`
- Compute gradients with respect to `epsilon` and `sigma`
- Build inversion loops in raw PyTorch or with `MaxwellTM` / `Maxwell3D`
- Control memory and runtime with storage, callback, and backend options

## Core Concepts

- Model tensors: `epsilon`, `sigma`, and `mu`
- Source amplitude tensors shaped `[n_shots, n_sources, nt]`
- Receiver traces returned as `[nt, n_shots, n_receivers]`
- CPML boundaries, finite-difference stencils, and CFL-driven internal resampling

Coordinate conventions:

- 2D TM uses `[y, x]`
- 3D uses `[z, y, x]`

## Typical Workflow

1. Build model tensors on the target device.
2. Define source and receiver geometry.
3. Run forward modeling to produce synthetic traces.
4. Compute a misfit against observed data.
5. Backpropagate and update the model in an inversion loop.

## Recommended Learning Order

1. Run a small 2D forward example from `getting-started.md`
2. Read `guides/api-orientation.md`
3. Read `guides/modeling.md`
4. Read `guides/inversion.md`
5. Review `guides/configuration.md`, `guides/limitations.md`, and `guides/verification.md` before scaling up
