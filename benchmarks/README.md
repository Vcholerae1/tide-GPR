# Benchmarks

Benchmarks measure runtime or memory behavior; they are not reduced-size
examples. Their workloads are configured through explicit command-line
arguments. Results are written under the ignored `artifacts/benchmarks/`
directory.

`maxwell3d_cuda_launch.py` sweeps native 3D CUDA launch configurations on a
synthetic workload. Use `--backward` to include snapshot storage and the
adjoint/model-gradient pass, and compare `median_ms_per_shot` before choosing a
shot batch size.

`maxwell3d_triton.py` is a deliberately narrow prepared-to-prepared comparison
of the native CUDA forward loop with experimental flat Triton kernels. It
supports FP32, fourth-order differences, one shot/source/receiver, and standard
CPML. Triton JIT and CUDA Graph capture are warmed outside timing; pass
`--no-cuda-graph` to expose Python launch overhead.

`tm2d_hvp.py` compares native full and Gauss-Newton TM2D Hessian-vector
products with a central finite difference of two native gradients. It reports
CUDA/wall time and peak allocated memory for fixed small, medium, and large
cases. Add `--include-python` to measure the Python reference path on the small
case.

For the large float32 case on an RTX 4070 (20 repeats), the fused full-Hessian
path measured 66.59 ms versus 63.25 ms for finite differences and 39.61 ms for
Gauss-Newton. The pre-fusion full-Hessian baseline was 78.34 ms, so merging the
two reverse passes reduced its median runtime by about 15%.

`tm2d_hvp_batch.py` measures several full HVP directions independently, through
a reusable `TM2DLinearizationContext`, and as `2K` central-difference gradient
evaluations. It also reports peak allocation and relative error. Use
`--storage-compression none` to isolate reuse accuracy from BF16 background
snapshot quantization.

On the same RTX 4070, four large BF16 directions (20 repeats) measured
149.49 ms with `block_size=4`, 274.31 ms as independent HVPs, and 260.94 ms as
central differences. Caching the forward and background-adjoint histories and
using a native direction batch was therefore about 45.5% faster than
independent HVPs and 42.7% faster than finite differences. Peak allocation was
1207.0 MiB for the context versus 517.1 MiB for independent HVPs and 258.5 MiB
for finite differences.

For a lower-memory setting, `block_size=2` measured 206.03 ms and 926.7 MiB.
Reusing BF16 snapshots changed the result by relative L2 `1.28e-3`;
full-precision storage reduced the medium-case reuse error to `1.80e-7` and is
the accuracy-first option.
