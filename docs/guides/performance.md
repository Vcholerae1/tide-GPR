# Performance Notes

## Precision Modes

Snapshot storage supports:
- `storage_compression=False` / `"none"`: raw storage.
- `storage_compression="bf16"`: bfloat16-compressed storage.

## CPU vs CUDA

- CPU path supports `float32`/`float64` propagation and gradients.
- CUDA path is recommended for large `ny x nx x nt` workloads.

## Batching

- Throughput scales with `n_shots` until memory bandwidth saturation.
- For gradient workloads, tune `model_gradient_sampling_interval` and
  `storage_mode` to control memory use.

## Storage Impact

- `storage_mode="device"` is fastest but VRAM-heavy.
- `storage_mode="cpu"` / `"disk"` reduce VRAM pressure with transfer overhead.
