# Module: tide.maxwell

## Public API
- class MaxwellTM
- function maxwelltm

### `maxwelltm(...)` precision arguments

- `compute_dtype`: `"fp32"` (default) or `"fp16"`.
- `mp_mode`: `"throughput" | "balanced" | "robust"`.

Behavior:
- `compute_dtype="fp16"` is CUDA-only.
- External parameter units are unchanged (SI-compatible inputs).
- Internally, fp16 mode uses nondimensional scaling and returns fields in
  physical units.
- `mp_mode="throughput"` uses native `half` symbols when the CUDA extension is
  built with half-kernel support.
- `mp_mode="balanced"` currently routes TM2D through the Python backend so it
  can keep fp32-sensitive workspaces in higher precision.
- `mp_mode="robust"` keeps the TM2D solver in fp32.

## Advanced or Internal Functions
- prepare_parameters
- prepare_parameters_nondim
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
- _normalize_compute_dtype
- _normalize_mp_mode
- _scale_tm_states_to_nondim
- _restore_tm_states_from_nondim
