# 3D Maxwell FDTD CUDA Acceleration Research

This note summarizes CUDA-oriented 3D FDTD Maxwell implementations and
papers that are relevant to speeding up tide's current 3D backend.

## Executive Summary

tide already has a native 3D CUDA backend, but it is intentionally
conservative: a flat one-thread-per-cell launch, one H update kernel and one
E update kernel per time step, full-domain CPML auxiliary arrays, optional
Debye dispersion kernels, and snapshot storage for gradients. That makes it
easy to maintain and verify, but it leaves several performance paths open.

The strongest near-term opportunities are:

1. Establish a roofline/profile baseline before changing kernels. 3D FDTD is
   usually DRAM-bandwidth bound, so the useful metrics are achieved memory
   bandwidth, bytes per cell-step, branch efficiency, register count, and
   launch overhead.
2. Add a fast indexing path for ordinary problem sizes. The current CUDA
   kernels use 64-bit index arithmetic and division/modulo in the inner path;
   32-bit index/stride specializations should be validated first because this
   is a low-risk change compared with stencil tiling.
3. Reduce per-time-step orchestration overhead. The forward path launches H,
   E, source, receiver, and sometimes Debye polarization kernels every time
   step. CUDA Graph capture is a good fit for fixed-shape propagation windows.
4. Optimize the Debye path before attempting complex temporal blocking. The
   current Debye path copies Ex/Ey/Ez each step and launches three
   polarization kernels. Fusing old-E capture and polarization updates into
   the E-Debye pass is likely a cleaner win than full space-time tiling.
5. Treat compact PML storage primarily as a memory feature. Existing artifacts
   in this repo show memory savings but mixed speed results, which matches the
   literature: CPML optimization matters, but extra indexing and face kernels
   can erase bandwidth wins if not designed carefully.
6. Keep fdtd-z-style systolic/diamond kernels as a high-risk, specialized
   backend, not the first optimization. They get speed by constraining the
   physics and output model, while tide needs sources, receivers, CPML,
   conductivity, Debye, gradients, storage modes, and callbacks.

## Tide Current State

Relevant code:

- `src/tide/maxwell/maxwell3d_cuda.py`
- `src/tide/csrc/em3d/maxwell_3d_cuda_inst.cu`
- `src/tide/csrc/common/staggered_grid.h`
- `artifacts/profiles/em3d_*.json`

The Python wrapper pads model tensors, builds material coefficients `ca`,
`cb`, and `cq`, initializes six field arrays, and initializes twelve CPML
memory arrays as full-domain tensors. CPML interiors are zeroed, but memory
is still allocated across the full padded domain.

The CUDA implementation instantiates stencil orders 2, 4, 6, and 8. The
staggered derivative macros are compile-time specialized through
`TIDE_STENCIL`. Runtime scalar constants such as grid spacing, dimensions,
PML bounds, and batched/unbatched flags are copied to CUDA constant memory.

The main forward loop does this per time step:

1. `forward_kernel_h`
2. `forward_kernel_e` or `forward_kernel_e_debye`
3. optional source injection kernel
4. optional Debye polarization updates, currently three kernels
5. optional receiver recording kernel

The base kernels use a 1D grid of 256-thread blocks. Each thread decodes a
linear index into shot, z, y, and x coordinates, checks active-cell bounds,
checks PML regions, computes the required curls, optionally updates CPML
auxiliary fields, and writes three field components.

Important implications:

- The implementation is global-memory heavy. Each H/E pass streams several
  full-domain arrays, and CPML adds more arrays near boundaries.
- There is no shared-memory stencil tile in the current 3D Maxwell path.
- There is no temporal blocking or multi-time-step kernel in the current path.
- Source/receiver kernels can become nontrivial launch overhead for small or
  medium grids.
- `n_threads` is accepted in Python but ignored by the CUDA forward entry.
- Existing profile artifacts include experimental fp16 and compact-PML runs,
  but the current public CUDA path rejects non-`standard` execution backends.

## Primary Sources

### gprMax CUDA Engine

Source: "A CUDA-based GPU engine for gprMax: open source FDTD
electromagnetic simulation software"

- PDF: https://researchportal.northumbria.ac.uk/files/17691735/gprMax_GPU.pdf
- Repo: https://github.com/gprMax/gprMax
- Docs: https://docs.gprmax.com/en/latest/gpu.html

The useful ideas for tide are pragmatic rather than exotic:

- Use 1D indexing when it gives coalesced consecutive memory access. gprMax
  explicitly chose 1D indexing for simplicity, flexibility, and similar
  performance to higher-dimensional decompositions.
- Keep kernel source readable with index macros while preserving linear
  coalesced access.
- Put small material coefficient tables in CUDA constant memory.
- Mark read-only pointers `const` and `restrict` so the compiler can use the
  read-only/texture path.
- Combine Ex/Ey/Ez updates in one E kernel, and Hx/Hy/Hz updates in one H
  kernel.
- Measure actual read/write throughput. The paper reports E/H kernels close
  to the measured memory bandwidth available from BabelStream on their GPU.

Relevance to tide: tide already follows several of these choices. The gaps
are mainly measurement, 32-bit/index arithmetic cost, launch overhead, and the
fact that full-domain CPML and Debye support make tide heavier than the
non-dispersive gprMax kernel shown in the paper.

### NVIDIA / Micikevicius 3D Finite Difference CUDA

Source: "3D Finite Difference Computation on GPUs using CUDA"

- PDF: https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf
- CUDA sample overview: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/FDTD3d

The paper focuses on high-order finite-difference stencil computation. It uses
2D tiles such as 16x16 and 32x16 for 3D data, improves reuse through shared
memory, and reports multi-GPU scaling by exchanging ghost slices along the
slow dimension with asynchronous copy and MPI.

Relevance to tide:

- Shared-memory tile reuse is plausible for stencil orders 4, 6, and 8.
- The paper's target stencil is simpler than full Maxwell with six staggered
  fields, CPML branches, source/receiver operations, and gradient storage.
- Multi-GPU domain decomposition is only worth considering after single-GPU
  bandwidth and memory format issues are understood.

### Roofline Analysis for 3D FDTD on GPU

Source: "Performance analysis and optimization of three-dimensional FDTD on
GPU using roofline model"

- DOI page: https://www.sciencedirect.com/science/article/abs/pii/S0010465511000452

The paper's core result is that 3D FDTD performance on GPU is memory-bandwidth
dominated. It recommends removing misaligned access, exploiting temporal
locality, and using texture/read-only cache paths.

Relevance to tide:

- A roofline measurement should come before large rewrites.
- If the current kernels are already close to DRAM bandwidth, smaller code
  changes may only move performance a few percent.
- If actual bandwidth is far below the hardware limit, index arithmetic,
  branch divergence, uncoalesced access, or launch overhead are likely causes.

### B-CALM

Source: "B-CALM: An Open-Source GPU-based 3D-FDTD with Multi-Pole Dispersion
for Plasmonics"

- PDF: https://ee.stanford.edu/~dabm/407.pdf

B-CALM is useful because it discusses dispersive materials, not just a simple
vacuum/dielectric stencil. It stores constant parameters in fast read-only
memory, uses shared memory for neighboring field values, and splits the
electric update into two kernels for multi-pole Drude-Lorentz materials to
reduce thread divergence. The paper argues the extra read/write can be worth
it when the number of poles is greater than one.

Relevance to tide:

- tide's Debye path is a concrete optimization target.
- For multi-pole dispersion, avoiding divergent per-cell material paths and
  reducing redundant global-memory traffic may matter more than generic H/E
  kernel tuning.
- Unlike B-CALM, tide's Debye path currently performs full-field device copies
  and separate polarization kernels, so there is likely low-hanging cleanup.

### CPML Memory Optimization

Source: "Accelerating Electromagnetic Field Simulations Based on
Memory-Optimized CPML-FDTD with OpenACC"

- Article: https://www.mdpi.com/2076-3417/12/22/11430

The paper is 2D/OpenACC rather than 3D/CUDA, but the CPML lesson transfers:
CPML variables only need to exist in absorbing regions, not the whole domain.
The paper also validates single precision against double-precision CPU
reference cases and emphasizes persistent device data regions and asynchronous
output.

Relevance to tide:

- Full-domain CPML auxiliary arrays are a memory cost and become especially
  painful in gradient workloads.
- Compact PML layouts should be evaluated as memory pressure relief first.
- Speed gains depend on whether compact indexing and boundary-only kernels add
  more overhead than they remove.

### fdtd-z

Source: fdtd-z repository

- Repo: https://github.com/spinsphotonics/fdtdz
- README: https://raw.githubusercontent.com/spinsphotonics/fdtdz/main/README.md

fdtd-z is the most aggressive open-source GPU-oriented design in this survey.
It exposes a low-level GPU FDTD API, uses a systolic scheme with CUDA
cooperative groups, and deliberately limits the physics to get throughput.
Its own README states that dispersion and richer outputs are avoided because
they add bandwidth pressure and register pressure.

Relevance to tide:

- fdtd-z is a good proof that big speedups require architectural constraints.
- Its design conflicts with tide's broader API: arbitrary source/receiver
  patterns, conductivity, CPML on all sides, Debye, gradients, storage modes,
  and callbacks.
- A fdtd-z-style backend could be useful later for a restricted
  "fast photonics forward-only" mode, but it is not the right first step for
  the current general 3D backend.

### FDTDX

Source: FDTDX JOSS paper and repository

- JOSS paper: https://joss.theoj.org/papers/10.21105/joss.08912.pdf
- Repo: https://github.com/ymahlau/fdtdx
- arXiv: https://arxiv.org/abs/2412.12360

FDTDX is a JAX-based GPU FDTD package with automatic differentiation and
multi-GPU scaling. The key idea for gradients is not raw CUDA optimization,
but a custom gradient algorithm based on time reversibility, avoiding the need
to save every E/H field after every time step.

Relevance to tide:

- This is directly relevant to tide's memory-heavy inverse workloads.
- It is not directly applicable to lossy/conductive/CPML/Debye simulations
  without careful derivation, because dissipation and auxiliary-memory states
  complicate reversibility.
- A practical tide version would likely start as checkpoint/recompute or
  reversible subcases, not a blanket replacement for the current adjoint path.

### fdtd3d

Source: fdtd3d repository

- Repo: https://github.com/zer011b/fdtd3d

fdtd3d is a portable FDTD Maxwell solver with MPI/OpenMP/CUDA support and a
componentized build system. Its repo is useful as an engineering reference for
multi-platform FDTD infrastructure and benchmarking rather than as a direct
kernel template for tide.

Relevance to tide:

- It reinforces the value of building solver variants for specific needs.
- For tide, this argues for explicit backend variants instead of one kernel
  trying to cover every layout, material mode, and gradient mode.

### Multi-GPU Overlap

Source: "Overlapping computation and communication of three-dimensional FDTD
on a GPU cluster"

- DOI page: https://www.sciencedirect.com/science/article/abs/pii/S0010465512002044

The paper proposes kernel-split and host-buffer methods for hiding boundary
exchange overhead in GPU-cluster 3D FDTD. It reports 92 percent of ideal
six-GPU scaling in the tested setup.

Relevance to tide:

- Multi-GPU domain decomposition is a later-stage feature.
- For current inversion/GPR workloads, distributing independent shots or
  independent simulations is easier and closer to gprMax's MPI task-farm
  pattern.
- Domain decomposition becomes relevant only when a single 3D model no longer
  fits or when one shot is too slow even after single-GPU tuning.

### CUDA Graphs

Source: NVIDIA CUDA Graphs docs and technical blog

- Docs: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- Blog: https://developer.nvidia.com/blog/cuda-graphs/

CUDA Graphs reduce repeated launch overhead by capturing multiple GPU
operations and replaying them through a single launch path.

Relevance to tide:

- The forward loop has a fixed topology for a given mode and shape.
- This is especially attractive for small/medium grids, Debye mode, or
  source/receiver-heavy runs where per-step kernels are short.
- It is less likely to matter for very large grids where each H/E kernel is
  long enough to dominate launch overhead.

## Prioritized Optimization Backlog

### P0: Measure the Current Backend

Before changing kernels, add a repeatable benchmark/profile harness:

- Shapes: 64x48x48, 100x100x100, and one target production shape.
- Shots: 1, 4, and saturation point.
- Stencils: 2 and 4 first; 8 if high-order use is important.
- Modes: no-gradient forward, gradient forward-with-storage, backward, Debye.
- PML widths: 0 or minimal if supported, 8, and production width.
- Metrics: cell-steps/sec, receiver relative error, achieved DRAM bandwidth,
  branch efficiency, memory transactions, register count, occupancy, and
  launch overhead.

Use Nsight Compute for kernel-level roofline data and Nsight Systems for
timeline/launch overhead.

Acceptance criteria:

- We know whether `forward_kernel_h` and `forward_kernel_e` are near memory
  bandwidth limit.
- We know whether time is dominated by H/E kernels, source/receiver launches,
  Debye copies, or storage traffic.
- We have one "do not regress" correctness and performance script.

### P1: Low-Risk CUDA Cleanup

1. Add int32 index specializations for normal grid sizes.

   Most practical GPU-resident arrays cannot approach int64 indexing limits.
   A guarded int32 path can remove 64-bit division/modulo and reduce register
   pressure. Keep the int64 path for oversized or unusual cases.

2. Split shared-model and batched-model kernels.

   `ca_batched`, `cb_batched`, and `cq_batched` are runtime constant-memory
   flags. Separate launch symbols can eliminate uniform runtime branches and
   make generated code easier to inspect.

3. Autotune block size.

   The current 256-thread scalar launch is reasonable, but test 128, 256, and
   512 for H, E, backward, and Debye. If 256 remains best, document it and
   make `n_threads` either meaningful or remove it from the CUDA path.

4. Add CUDA Graph replay for fixed propagation windows.

   Capture H/E/source/receiver sequences for forward-only first. Extend to
   Debye and storage only after the simple path is stable.

5. Compile and inspect generated SASS/PTX.

   Verify whether read-only loads are emitted for `const __restrict__`
   pointers. If not, test explicit `__ldg` or texture objects for coefficient
   arrays on supported architectures.

### P2: Medium-Risk Data Movement Reductions

1. Fuse Debye old-field capture and polarization updates.

   The current Debye path performs device-to-device copies of Ex/Ey/Ez before
   the E update, then launches three polarization kernels. A fused E-Debye
   kernel can keep old E values in registers, compute new E, and update
   per-pole polarization state in the same pass. This should be tested for
   one-pole and multi-pole cases separately.

2. Compact CPML storage.

   Store CPML auxiliary variables only in the PML regions. Prior artifacts in
   this repo show memory savings around 15-19 percent for 100^3 sparse
   gradient cases, with speed roughly neutral to mixed. Treat this as a memory
   enabler, then optimize face indexing only if profiling shows benefit.

3. Interior/PML split kernels.

   Run a branch-free interior H/E kernel over the non-PML box, and handle PML
   faces separately. This is useful only if branch and CPML checks are a
   measurable issue; for large domains with narrow PML, extra launches may
   lose.

4. Sparse receiver/source batching.

   If Nsight Systems shows source/receiver launches dominating small-grid
   cases, either capture them in CUDA Graphs or combine sparse operations more
   carefully. Do not add per-cell source checks to the main E kernel.

### P3: High-Risk Kernel Architecture Changes

1. Shared-memory spatial tiling.

   Start with stencil 4 and forward-only, no Debye, no gradients. The
   implementation burden rises quickly for six fields, staggered access,
   arbitrary stencil order, CPML, and storage.

2. Temporal blocking or diamond/systolic kernels.

   This is where fdtd-z and DiamondTorre-style algorithms live. It can win, but
   it changes the backend contract. It should be a separate restricted backend
   with explicit constraints.

3. Multi-GPU domain decomposition.

   Start with shot-level parallelism first. Domain decomposition needs halo
   exchange for staggered fields and CPML auxiliary state, plus autograd
   storage coordination.

4. Time-reversible gradient.

   Use FDTDX as inspiration, but treat lossy media, conductivity, CPML, Debye,
   and checkpointing as a research project. A realistic first step is
   checkpoint/recompute for lossless or weakly lossy subcases.

## What I Would Build First

The first reviewable unit should be a profiling and micro-optimization PR,
not a new solver:

1. Add `examples/profile_maxwell3d_cuda.py` or a testable benchmark script
   that reports cell-steps/sec and optional Nsight-friendly kernel ranges.
2. Add an int32 index specialization for forward H/E kernels.
3. Add a small block-size switch or compile-time variants for 128/256/512,
   guarded behind an internal option.
4. Run correctness parity against existing tests and benchmark the same shapes
   before/after.

If that shows launch overhead is meaningful, the next PR should add CUDA Graph
capture for forward-only fixed-window propagation. If Debye is the bottleneck,
skip graphs temporarily and fuse the Debye path first.

## Expected Tradeoffs

- Compact PML reduces memory but may hurt speed if indexing becomes complex.
- Shared-memory tiling helps only when redundant global loads are the limiting
  factor. If the current kernels are already bandwidth-saturated, gains may be
  modest.
- CUDA Graphs improve CPU launch overhead, not device-side memory bandwidth.
- fp16/bf16 field propagation can be fast, but the repo artifacts show
  receiver relative L2 differences around a few percent in some 100^3 cases.
  That is a separate numerical-mode decision, not a transparent optimization.
- Time-reversible gradients are attractive, but tide's lossy and CPML states
  mean the math must be handled explicitly.

## Source Shortlist

Best starting points for implementation:

1. gprMax GPU paper for practical CUDA FDTD kernel design:
   https://researchportal.northumbria.ac.uk/files/17691735/gprMax_GPU.pdf
2. Roofline paper for deciding whether a change can matter:
   https://www.sciencedirect.com/science/article/abs/pii/S0010465511000452
3. NVIDIA 3D finite difference paper for shared-memory stencil tiling:
   https://developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf
4. B-CALM for dispersive-material kernel structure:
   https://ee.stanford.edu/~dabm/407.pdf
5. fdtd-z README for the high-risk/high-speed restricted-backend design:
   https://raw.githubusercontent.com/spinsphotonics/fdtdz/main/README.md
6. FDTDX JOSS paper for memory-efficient gradients:
   https://joss.theoj.org/papers/10.21105/joss.08912.pdf
