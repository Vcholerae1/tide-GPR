# Performance Notes

## Precision Modes

`tide.maxwelltm` supports:
- `compute_dtype="fp32"`: default baseline.
- `compute_dtype="fp16"`: CUDA-only mixed precision path with internal
  nondimensional scaling.
- `mp_mode="throughput"` uses the native all-half TM2D path when available.
- `mp_mode="balanced"` currently routes TM2D through the Python backend so it
  can keep the recurrent state, coefficient preparation, and receiver
  accumulation in fp32.

`mp_mode` controls numerical policy:
- `throughput`: favors reduced precision.
- `balanced`: mixed precision with fp32-sensitive TM2D workspaces.
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
