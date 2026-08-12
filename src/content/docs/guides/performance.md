---
title: "Performance"
description: "Measure and tune TIDE workloads without trading away numerical correctness."
---

TIDE runtime is governed by propagated grid cells, internal time steps, shots, field components, derivative passes, and snapshot traffic. Optimize the representative end-to-end workload, not a tiny forward kernel that omits the dominant storage or adjoint cost.

## Cost model

A rough forward cost grows with

$$
N_{shots}\,N_z\,N_y\,N_x\,N_{internal\ steps}\,C_{stencil}.
$$

For TM2D, omit $N_z$ and use fewer field components. Gradient and Hessian-vector workflows add forward snapshots, reverse propagation, and correlation work. CFL sub-stepping multiplies `N_internal steps` even though the returned receiver tensor keeps the original `nt`.

## Establish a baseline

Record the following before tuning:

- Model and padded shape.
- Number of shots, sources, receivers, and user samples.
- CFL step ratio.
- Dtype, stencil, and CPML widths.
- Backend and device.
- Forward, backward, and total objective time.
- Peak device and host memory.
- Storage mode and compression.

Run one warm-up before timing CUDA. Synchronize the device around measured regions. Repeat enough times to distinguish startup overhead from steady-state behavior.

## CPU or CUDA

CPU is useful for small deterministic tests, float64 studies, installation checks, and environments without a supported GPU. CUDA is normally preferred for larger grids and shot batches.

GPU use is not automatically faster. Tiny models, short time axes, sparse callbacks, host-backed snapshots, and repeated Python setup can dominate kernel work. Compare complete calls on the actual target device.

## Shot batch size

Shots are independent for a shared model and provide natural parallelism. Increase batch size until one of these occurs:

- Throughput stops improving.
- Device memory approaches the safe limit.
- Snapshot transfer saturates host or disk bandwidth.
- A larger batch changes optimizer semantics or latency requirements.

Use samples per second or shot-time-steps per second, not only elapsed time, when comparing batch sizes.

## Grid, stencil, and CPML

A finer grid increases every spatial dimension and often forces more internal time steps. In 3D, halving isotropic spacing can increase cell count by roughly eight before considering the smaller stable step.

Higher stencil order costs more per cell but can reduce dispersion at a given spacing. Larger CPML widths increase padded volume. These are accuracy choices with performance consequences, so tune them through convergence and reflection tests rather than runtime alone.

## Snapshot placement

For derivative workloads:

- `device` avoids transfer but consumes VRAM.
- `cpu` trades VRAM for transfer and pinned or pageable host behavior.
- `disk` adds filesystem I/O.
- BF16 compression reduces capacity and bandwidth with a precision trade-off.

Measure forward and backward separately. A storage mode can leave forward time unchanged while making the reverse pass much slower.

## Gradient sampling

Increasing `model_gradient_sampling_interval` can reduce snapshot and correlation work. It also approximates the model gradient. Benchmark only after comparing the resulting gradient and objective step against interval 1.

## Reference execution modes

The reference backend may run eagerly, through JIT, or through `torch.compile`, depending on supported configuration. Compilation has an upfront cost and can recompile when shapes or options change. Keep shapes stable across repeated calls and separate compile time from steady-state timing.

## Callbacks

A callback that copies a wavefield to CPU forces synchronization and can dominate runtime. Use sparse callback cadence, scalar summaries, spatial downsampling, and deferred plotting. Benchmark once with callbacks disabled to quantify their cost.

## Memory-first tuning order

1. Reduce shot batch size.
2. Confirm no unnecessary tensor copies or retained autograd graphs exist in user code.
3. Use BF16 snapshot compression after a gradient comparison.
4. Move snapshots from device to host.
5. Use local disk storage when host memory is insufficient.
6. Increase gradient sampling interval only after numerical validation.
7. Reduce grid size or time window only when the physical question permits it.

## Benchmark hygiene

- Warm up the exact operation being measured.
- Synchronize asynchronous devices.
- Report median and spread over repeats.
- Keep model, source, receiver, storage, and derivative settings identical.
- Record software and hardware versions.
- Exclude compilation only when reporting steady-state separately.
- Do not compare forward-only timing with a forward-backward objective.

A faster result is useful only when receiver traces and required derivatives remain within the stated tolerance. Run [verification](/tide-GPR/guides/verification/) after any configuration change that affects numerical behavior.
