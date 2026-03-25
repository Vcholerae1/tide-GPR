# Performance Notes

## Precision Modes

Snapshot storage supports:
- `storage_compression=False` / `"none"`: raw storage.
- `storage_compression="bf16"`: bfloat16-compressed storage.

TM2D compute also supports:
- `compute_precision="default"`: existing float32/float64 execution.

Current API note:
- The historical fp16_scaled TM2D path is no longer part of the current public API surface.

## CPU vs CUDA

- CPU path supports `float32`/`float64` propagation and gradients.
- CUDA path is recommended for large `ny x nx x nt` workloads.
- `maxwell3d(..., experimental_cuda_graph=True)` is available as an experimental
  forward-only CUDA mode on the C/CUDA backend.

## Batching

- Throughput scales with `n_shots` until memory bandwidth saturation.
- For gradient workloads, tune `model_gradient_sampling_interval` and
  `storage_mode` to control memory use.

## Maxwell3D CUDA Graph

- The current implementation only applies to forward-only `maxwell3d` calls on
  CUDA with `python_backend=False`.
- Graph capture is aligned to the existing chunk boundary:
  `effective_chunk = callback_frequency` when a `forward_callback` is present,
  otherwise the whole forward pass is one chunk.
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
