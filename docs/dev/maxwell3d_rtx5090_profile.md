# Maxwell 3D RTX 5090 Profile

This note records measurements from the native CUDA backend on an RTX 5090
(compute capability 12.0) using CUDA 13.0 and PyTorch 2.12.1+cu130.

## Workload

- Model: `70 x 70 x 70`
- CPML width: 20 cells on every side (`110 x 110 x 110` padded grid)
- Time steps: 300
- Stencil: fourth order unless noted
- Precision: float32
- Source/receiver: one each per shot

The reproducible driver is `benchmarks/maxwell3d_cuda_launch.py`.

## Findings

Nsight Systems attributes 50.0% of CUDA kernel time to the E update and 48.6%
to the H update for four shots. Source injection and receiver recording together
account for less than 1%. For one shot, H/E still account for 91.0%, while the
two small source/receiver kernels account for 4.6%.

Shot batching crosses a sharp cache-capacity boundary:

| Shots | Forward ms | Forward ms/shot | Forward+backward ms/shot |
|---:|---:|---:|---:|
| 1 | 17.72 | 17.72 | 34.27 |
| 2 | 59.59 | 29.79 | 59.16 |
| 4 | 128.93 | 32.23 | 65.10 |
| 8 | 248.98 | 31.12 | not measured |
| 12 | 369.60 | 30.80 | not measured |

For this workload, single-shot execution is about 1.8x faster per shot than a
four-shot batch and also uses substantially less snapshot memory. Do not assume
that increasing shot batch size improves RTX 5090 throughput; sweep it using the
actual padded grid and gradient storage settings.

Stencil order has a much smaller effect than expected from the additional
neighbor loads:

| Stencil | Forward ms |
|---:|---:|
| 2 | 127.92 |
| 4 | 128.90 |
| 6 | 132.06 |
| 8 | 134.03 |

The small 2-to-8 order delta indicates that the hardware caches already capture
most spatial reuse. A shared-memory tiled kernel must therefore demonstrate an
end-to-end win rather than relying on reduced nominal global loads.

## Rejected Experiments

- 32-bit linear-index specialization: no measurable change (129.048 ms versus
  129.047 ms).
- 3D spatial launch shapes exposed through `n_threads`: no meaningful win; the
  smallest shapes were 2-3% slower.
- Removing the 64-register cap: about 0.8% slower on the primary workload.
- Full-grid interior/CPML kernel split: about 5.1% slower because the PML pass
  scanned the whole grid and returned early for interior cells.
- Compact interior/CPML slab launches: numerically matched the baseline
  (maximum receiver error 1.2e-7), but took 23.23 ms versus 17.68 ms for the
  single-shot case, about 31% slower. Compact index decoding and two launches
  per field update outweighed the eliminated CPML branch work.
- A 32x4 single-z-plane shared-memory XY tile: took 21.31 ms, about 21% slower,
  and failed receiver parity (maximum error 92.5). The prototype was rejected
  and is not present in the production kernel.
- Cooperative persistent multi-step forward: numerically compatible on the
  non-dispersive forward path, but took 41.5 ms for one shot and 300 steps
  with the default occupancy configuration, versus 17.7 ms for the baseline.
  Limiting the launch to 512 threads improved it to 24.8 ms but remained
  slower. Repeated full-domain strided sweeps and grid-wide barriers dominate;
  dispersion/storage/backward paths are not covered by this prototype.
- Single-shot source/receiver fusion: below 0.1% after controlled retesting.

## Implications

Keep the one-pass H/E kernels and the 64-register default for now. Both a
compact CPML decomposition and a first 2.5D shared-memory tile lost to the
one-pass baseline on this workload, so neither experimental path is retained.
The persistent implementation remains opt-in for further research only
(TIDE_EM3D_PERSISTENT_FORWARD=OFF by default).
L2 persistence hints should be treated cautiously because single-shot execution
already gets strong natural cache residency, while multi-shot working sets
exceed cache capacity. A more promising next step is temporal blocking or a
persistent multi-step kernel that amortizes launch and synchronization costs
without duplicating the full field working set.

Nsight Compute hardware-counter collection was unavailable in the cloud
container (`ERR_NVGPUCTRPERM`). Nsight Systems reports are stored with the
remote benchmark artifacts.

## Gradient Sampling

For one shot, changing `model_gradient_sampling_interval` produced:

| Interval | Forward+backward ms | Gradient cosine vs. interval 5 | Relative L2 error |
|---:|---:|---:|---:|
| 5 | 41.89 | 1.0000 | 0.000 |
| 10 | 37.90 | 0.9999 | 0.013 |
| 20 | 36.66 | 0.9956 | 0.095 |
| 40 | 34.91 | 0.7022 | 2.062 |
| 60 | 33.05 | 0.7926 | 0.905 |

The gradient comparison used a separate heterogeneous layered model. Intervals
40 and 60 are not acceptable general speed presets despite their lower runtime.
Interval 20 remains a reasonable performance/accuracy compromise for the
measured workload; interval 10 is the safer choice when gradient fidelity is
more important. This parameter must be validated again for each acquisition and
frequency band.
