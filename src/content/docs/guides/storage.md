---
title: "Storage and Gradient Snapshots"
description: "Choose where derivative snapshots live and understand the memory, I/O, and precision trade-offs."
---

Reverse-mode differentiation needs information from the forward trajectory. TIDE stores selected forward-time states so the adjoint pass can correlate them with reverse fields and accumulate material gradients. `StorageOptions` controls where those snapshots live, how they are represented, and when capacity limits trigger another placement choice.

## Why storage matters

Snapshot demand grows with spatial volume, shot batch size, number of stored field components, and number of sampled time levels. A forward-only call may need no derivative snapshots. A VJP or full Hessian-vector product can require substantial state, even when the final receiver tensor is small.

The receiver data shape is therefore not a useful estimate of gradient memory.

## Storage modes

| Mode | Snapshot location | Strength | Cost |
| --- | --- | --- | --- |
| `device` | Compute device | Lowest access latency | Highest VRAM use |
| `cpu` | Host memory | Releases VRAM | Device-host transfer and host capacity |
| `disk` | Files below `path` | Handles larger states | Filesystem bandwidth and latency |
| `none` | No snapshot allocation | Minimal storage | Invalid for operations that require stored states |
| `auto` | Selected from byte limits | Portable policy | Requires realistic capacity limits |

Start with `device` for a small gradient run. Move to `cpu` when VRAM is the limit. Use `disk` only on fast local storage and measure end-to-end runtime. Network filesystems can turn every reverse pass into an I/O bottleneck.

## Configuring storage

```python
storage = tide.StorageOptions(
    mode=tide.StorageMode.AUTO,
    path="./tide-storage",
    compression="bf16",
    bytes_limit_device=6 * 1024**3,
    bytes_limit_host=32 * 1024**3,
    chunk_steps=0,
)

operator = tide.MaxwellTM(
    discretization,
    experiment,
    storage=storage,
)
```

Capacity limits should leave headroom for material tensors, live wavefields, receiver data, gradients, optimizer state, CUDA context, and unrelated processes. Do not set the device limit equal to total advertised VRAM.

## Compression

`compression="bf16"` stores eligible float32 snapshots in bfloat16 while propagation and model tensors remain float32. This reduces storage volume and transfer traffic, but quantizes stored states.

Validate compression on the quantity the workflow uses:

1. Run the same small objective with and without compression.
2. Compare receiver data to confirm the primal path is unchanged.
3. Compare gradient norms and a directional derivative.
4. Compare one or more optimizer steps, not only one gradient image.

A visually similar gradient can still change line-search behavior.

## Temporal sampling

`model_gradient_sampling_interval` reduces the cadence used for model-gradient accumulation. It is separate from storage mode and compression. Increasing the interval can reduce stored time levels and correlation work, but it changes the derivative approximation. Establish an acceptable interval against the default value of 1.

## Disk lifecycle

Disk-backed allocations use isolated temporary paths and are released when their owner closes. Derivative sessions should therefore use a context manager:

```python
with operator.linearize(model, storage=storage) as linearized:
    gradient = linearized.vjp(receiver_cotangent)
```

The context prevents temporary files and host buffers from surviving longer than intended. Place `storage.path` on a filesystem with enough free space and predictable cleanup semantics.

## Shot batching and storage

Reducing shot batch size lowers simultaneous wavefield and snapshot demand, often more predictably than switching storage modes. The total computation remains similar, while peak memory falls. Gradient accumulation must preserve the desired dataset normalization across batches.

A useful tuning order is:

1. Measure one shot with `device` storage.
2. Increase batch size until memory or throughput stops improving.
3. If one shot does not fit, try BF16 snapshots.
4. Then try host storage.
5. Use disk after verifying local I/O bandwidth.
6. Increase gradient sampling interval only after a derivative comparison.

## Diagnosing storage failures

### Device out of memory before propagation

Model, optimizer state, or batch size is already too large. Snapshot placement cannot solve allocations that occur before the forward trajectory begins.

### Device out of memory during forward

Reduce shot or model batch size, lower stored-state demand, or choose host or automatic placement.

### Reverse pass is much slower than forward

Host transfer, disk I/O, compression conversion, or sparse access to stored chunks may dominate. Profile the full forward-backward operation.

### Disk usage remains after an exception

Keep derivative sessions inside `with` blocks and use a dedicated storage directory that can be inspected safely after failed runs.

See [configuration](/tide-GPR/guides/configuration/) for all storage fields and [performance](/tide-GPR/guides/performance/) for measurement strategy.
