 # Performance Notes

## Precision Modes

Snapshot storage supports:
- `storage_compression=False` / `"none"`: raw storage.
- `storage_compression="bf16"`: bfloat16-compressed storage.

TM2D compute also supports:
- `compute_precision="default"`: existing float32/float64 execution.
- `compute_precision="fp16_scaled"`: CUDA-only mixed precision with float32
  coefficients/CPML/accumulation and fp16 field-state + snapshot payloads.
- `compute_precision="fp16_scaled"` is primarily a memory-reduction mode. It
  does not guarantee faster kernels unless later vectorized half I/O is added.

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
- Under `compute_precision="fp16_scaled"`, snapshot payloads are implicitly
  fp16 and `storage_compression="bf16"` is not supported.
