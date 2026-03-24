# Module: tide.maxwell

## Public API
- class MaxwellTM
- function maxwelltm

### `maxwelltm(...)` precision arguments

- Default compute uses the input tensor dtype (`float32` or `float64`).
- `compute_precision="fp16_scaled"` enables a CUDA-only TM2D mixed-precision
  path with float32 public tensors and internal fp16 field/snapshot storage.
- Snapshot storage compression remains configurable via `storage_compression`,
  including `"bf16"` on the default path.

Behavior:
- External parameter units are unchanged (SI-compatible inputs).
- Returned fields remain in physical units.
- `compute_precision="fp16_scaled"` rejects `python_backend=True`,
  float64 public tensors, and `storage_compression="bf16"`.

## Advanced or Internal Functions
- prepare_parameters
- maxwell_func
- maxwell_python
- update_E
- update_H
- maxwell_c_cuda

## Autograd Functions
- MaxwellTMForwardFunc
- MaxwellTMForwardBoundaryFunc

## Internal Helpers
- _register_ctx_handle
- _get_ctx_handle
- _release_ctx_handle
- _compute_boundary_indices_flat
