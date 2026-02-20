# Performance Notes

## Precision Modes

`tide.maxwelltm` supports:
- `compute_dtype="fp32"`: default baseline.
- `compute_dtype="fp16"`: CUDA-only mixed precision path with internal
  nondimensional scaling.
- Current native kernels still execute fp32 math while using fp16 mixed
  precision I/O/scaling.

`mp_mode` controls numerical policy:
- `throughput`: favors reduced precision.
- `balanced`: intermediate behavior.
- `robust`: keeps more operations in fp32.

## CPU vs CUDA

- CPU path supports `float32`/`float64` propagation and gradients.
- CUDA path is recommended for large `ny x nx x nt` workloads.
- `compute_dtype="fp16"` is rejected on non-CUDA devices.

## Batching

- Throughput scales with `n_shots` until memory bandwidth saturation.
- For gradient workloads, tune `model_gradient_sampling_interval` and
  `storage_mode` to control memory use.

## Storage Impact

- `storage_mode="device"` is fastest but VRAM-heavy.
- `storage_mode="cpu"` / `"disk"` reduce VRAM pressure with transfer overhead.
- In fp16 compute mode, use raw storage (`storage_compression=False`).
