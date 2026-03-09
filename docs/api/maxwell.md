# Module: tide.maxwell

## Public API
- class MaxwellTM
- function maxwelltm

### `maxwelltm(...)` precision arguments

- Compute uses the input tensor dtype (`float32` or `float64`).
- Snapshot storage compression remains configurable via `storage_compression`,
  including `"bf16"`.

Behavior:
- External parameter units are unchanged (SI-compatible inputs).
- Returned fields remain in physical units.

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
