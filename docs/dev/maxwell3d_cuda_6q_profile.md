# Maxwell 3D CUDA 6Q Profile

This note captures the first fixed-workload CUDA performance baseline for the
native `tide.maxwell3d` 3D forward path.

## Workload

- Operator: `tide.maxwell3d(..., python_backend=False)` native CUDA forward
- Hardware: NVIDIA GeForce RTX 4070, CC 8.9, PyTorch 2.9.1+cu128
- Shape: `64 x 64 x 64` model, estimated padded shape `81 x 81 x 81`
- Time: `nt=500`, one shot, one source per shot, 32 receivers per shot
- Numerics: `float32`, stencil `2`, PML width `8` each side
- Components: `source_component="ey"`, `receiver_component="ey"`

End-to-end baseline:

```bash
uv run python tools/profile_maxwell3d_cuda_6q.py \
  --nz 64 --ny 64 --nx 64 --nt 500 \
  --shots 1 --receivers 32 --pml 8 \
  --warmup 2 --iters 5 \
  --output artifacts/profiles/em3d_cuda_6q_baseline_64_nt500.json
```

Measured mean was `29.93 ms`, or `8.88e9` padded cell-steps/s. Peak allocated
CUDA memory was `92.6 MB`.

## NSYS Ranking

Command:

```bash
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output artifacts/profiles/em3d_cuda_6q_baseline_64_nt500_nsys \
  uv run python tools/profile_maxwell3d_cuda_6q.py \
    --nz 64 --ny 64 --nx 64 --nt 500 \
    --shots 1 --receivers 32 --pml 8 \
    --warmup 0 --iters 1 \
    --output artifacts/profiles/em3d_cuda_6q_baseline_64_nt500_nsys_run.json
```

Kernel summary:

| Kernel | Time | Instances | Avg |
|---|---:|---:|---:|
| `forward_kernel_h` | 48.1% | 500 | 22.3 us |
| `forward_kernel_e` | 43.5% | 500 | 20.2 us |
| `record_receivers_component` | 2.7% | 500 | 1.27 us |
| `add_sources_component` | 2.6% | 500 | 1.21 us |

Main evidence:

- `artifacts/profiles/em3d_cuda_6q_baseline_64_nt500_nsys.nsys-rep`
- `artifacts/profiles/em3d_cuda_6q_baseline_64_nt500_nsys_cuda_gpu_kern_sum.csv`

## NCU Kernel Evidence

Commands:

```bash
ncu --force-overwrite --target-processes all \
  --kernel-name-base demangled --kernel-name regex:forward_kernel_h \
  --launch-skip 5 --launch-count 1 \
  --section LaunchStats --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SchedulerStats \
  --csv --page raw \
  --log-file artifacts/profiles/em3d_cuda_6q_forward_kernel_h_ncu.csv \
  --export artifacts/profiles/em3d_cuda_6q_forward_kernel_h_ncu \
  uv run python tools/profile_maxwell3d_cuda_6q.py \
    --nz 64 --ny 64 --nx 64 --nt 20 \
    --shots 1 --receivers 32 --pml 8 \
    --warmup 0 --iters 1 --no-nvtx \
    --output artifacts/profiles/em3d_cuda_6q_forward_kernel_h_ncu_run.json

ncu --force-overwrite --target-processes all \
  --kernel-name-base demangled --kernel-name regex:forward_kernel_e \
  --launch-skip 5 --launch-count 1 \
  --section LaunchStats --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SchedulerStats \
  --csv --page raw \
  --log-file artifacts/profiles/em3d_cuda_6q_forward_kernel_e_ncu.csv \
  --export artifacts/profiles/em3d_cuda_6q_forward_kernel_e_ncu \
  uv run python tools/profile_maxwell3d_cuda_6q.py \
    --nz 64 --ny 64 --nx 64 --nt 20 \
    --shots 1 --receivers 32 --pml 8 \
    --warmup 0 --iters 1 --no-nvtx \
    --output artifacts/profiles/em3d_cuda_6q_forward_kernel_e_ncu_run.json
```

Important metrics:

| Kernel | Time | DRAM peak | SM peak | L1 hit | L2 hit | Regs | Spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| `forward_kernel_h` | 48.5 us | 80.4% | 41.8% | 55.7% | 46.7% | 47 | 0 |
| `forward_kernel_e` | 47.6 us | 86.5% | 22.7% | 46.7% | 42.5% | 31 | 0 |

The NCU DRAM throughput values imply an effective sustained ceiling around
`491 GB/s` on this run (`394.9 GB/s / 80.4%`, `424.3 GB/s / 86.5%`).

## Six Answers

1. Memory-bound or compute-bound?

   The dominant H/E kernels are memory-throughput-bound for this workload.
   DRAM throughput is already 80-86% of NCU sustained peak while SM throughput is
   only 23-42%.

2. Current throughput?

   End-to-end unprofiled throughput is `8.88e9` padded cell-steps/s for the
   fixed `64^3, nt=500` workload. Kernel-level NSYS time is dominated by 1000
   main H/E launches totaling about `21.25 ms` GPU time.

3. Does compiled code match assumptions?

   Launch configuration is one-dimensional `256` threads per block, `2076`
   blocks for `531456` padded cells. NCU reports 47 registers/thread for H,
   31 registers/thread for E, and zero local spilling. `cuobjdump` against
   `sm_89` shows the fatbin was built with `-dlcm=ca -maxrregcount=64` for the
   relevant CUDA image, but detailed resource extraction did not produce better
   per-kernel data than NCU.

4. Real bottleneck layer?

   DRAM traffic is the first bottleneck. L1/L2 hit rates are only moderate, and
   the kernels are already close to sustained DRAM peak. Source and receiver
   kernels are small, together about 5.3% of GPU kernel time.

5. Reachable ceilings?

   The strongest ceiling evidence in this pass is NCU's sustained peak estimate:
   roughly `491 GB/s` DRAM bandwidth. A separate memory microbenchmark was not
   added yet, so treat this as profiler-derived rather than independently
   measured.

6. Near ceiling, and what next?

   Yes, the main kernels are close enough to DRAM ceiling that pure arithmetic
   cleanup is unlikely to move the needle first. The next implementation unit
   should reduce bytes moved per timestep:

   - Split interior and PML work so interior cells do not stream CPML auxiliary
     arrays.
   - Or introduce compact PML storage for the 12 CPML memory arrays.
   - After that, revisit shared-memory/temporal tiling if the new profile still
     shows high neighbor reload traffic.

CUDA Graph replay is still useful for repeated small calls, but this workload's
dominant cost is main-kernel memory traffic, not source/receiver kernels.
