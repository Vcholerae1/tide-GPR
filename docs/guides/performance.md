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

## Practical Optimization Checklist

1. Choose CUDA for medium/large models and batches.
2. Increase n_shots until throughput saturates, then stop.
3. Balance stencil order and grid spacing for target fidelity.
4. Use storage_mode=auto with byte limits on memory-constrained systems.
5. Profile representative workloads rather than synthetic tiny benchmarks.

## Read This Before Enabling Advanced Modes

Before enabling advanced runtime options broadly:

1. Confirm correctness on a small case.
2. Read `guides/limitations.md`.
3. Run the checks in `guides/verification.md`.
