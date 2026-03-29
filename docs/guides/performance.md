# Performance Notes

## Precision Modes

Snapshot storage supports:
- `storage_compression=False` / `"none"`: raw storage.
- `storage_compression="bf16"`: bfloat16-compressed storage.

## CPU vs CUDA

- CPU path supports `float32`/`float64` propagation and gradients.
- CUDA path is recommended for large `ny x nx x nt` workloads.
- `maxwell3d(..., cuda_graph=True)` is available as an experimental
  forward-only CUDA mode on the C/CUDA backend.

## Batching

- Throughput scales with `n_shots` until memory bandwidth saturation.
- For gradient workloads, tune `model_gradient_sampling_interval` and
  `storage_mode` to control memory use.

## Maxwell3D CUDA Graph

- The current implementation only applies to forward-only `maxwell3d` calls on
  CUDA with `python_backend=False`.
- Graph capture can use an independent iteration batch size via
  `cuda_graph_batch_size`. If omitted, the graph chunk defaults to the whole
  callback window.
- Callback boundaries are still hard boundaries for graph replay. With a
  callback present, the effective graph chunk is
  `min(callback_frequency, cuda_graph_batch_size)` when
  `cuda_graph_batch_size` is set.
- Graphs are cached at module scope and reused across compatible later calls on
  the same device. The cache key is shape/layout/config based rather than tied
  to one `maxwell3d` invocation.
- Callback execution stays outside the graph. That keeps `CallbackState`
  semantics unchanged while still allowing graph replay between callbacks.
- In-place edits to callback-visible wavefields are respected by later graph
  replays because the graph keeps the same storage. Avoid resizing, reassigning,
  or changing tensor layout/device inside callbacks.
- Speedup is workload-dependent. Small, launch-bound chunks may benefit; larger
  chunks can be neutral or slightly slower because graph capture and staging
  copies are not free. Use
  `uv run python examples/benchmark_maxwell3d_cuda_graph.py --verify` on the
  target GPU before enabling it broadly.

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
