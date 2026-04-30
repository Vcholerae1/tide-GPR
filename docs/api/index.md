# API Reference

Start here if you already know which workflow you want:

1. `tide.maxwelltm` for 2D TM forward modeling and inversion
2. `tide.maxwell3d` for 3D forward modeling and inversion
3. `tide.borntm` / `tide.born3d` for unified two-wavefield Born-Maxwell propagation
4. `tide.MaxwellTM` / `tide.Maxwell3D` / `tide.BornTM` / `tide.Born3D` for reusable module-based workflows
5. `tide.ricker` for source generation
6. `tide.CallbackState` for forward and backward callbacks
7. `tide.DebyeDispersion` and storage/backend helpers for advanced workflows

## Public Modules
- tide (top-level)
- tide.callbacks
- tide.resampling
- tide.cfl
- tide.padding
- tide.validation
- tide.maxwell
- tide.wavelets
- tide.staggered
- tide.utils
- tide.storage

## Internal Modules
- tide.backend_utils
- tide.csrc (C/CUDA)

Conventions used in this API reference:
- Tensor shape notation uses [dim0, dim1, ...].
- Time axis is generally the last axis for source wavelets and moved to [nt, ...] for returned receiver traces.
- 2D coordinates are [y, x], 3D coordinates are [z, y, x].
