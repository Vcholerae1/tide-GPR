# API Reference

This section documents the public Python API and key internal modules.

Public API priority order:
1. tide.maxwelltm and tide.maxwell3d
2. tide.ricker
3. tide.CallbackState and create_callback_state
4. tide.cfl_condition, upsample, downsample
5. padding and validation helpers

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
