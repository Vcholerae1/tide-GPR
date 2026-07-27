# Benchmarks

Benchmarks measure runtime or memory behavior; they are not reduced-size
examples. Their workload comes from explicit arguments and experiment configs.
Results are written under `artifacts/benchmarks/`.

`maxwell3d_cuda_launch.py` sweeps native 3D CUDA launch configurations on a
synthetic workload. Use `--backward` to include snapshot storage and the
adjoint/model-gradient pass, and compare `median_ms_per_shot` before choosing a
shot batch size.

`maxwell3d_fp16_io.py` compares native FP32, scalar FP16-I/O, and the
SeisCL-style two-x-cell `half2` FP16-I/O experiment. It reports timing, peak
incremental allocation, and receiver-data error against native FP32.

`tm2d_fp16_io.py` compares native FP32 propagation with the experimental TM2D
FP16-wavefield-I/O/FP32-compute forward path and reports runtime, peak allocated
memory, receiver relative L2 error, and waveform correlation.

`tm2d_fp16_overthrust.py` measures complete 100-shot `(200, 400)` Overthrust
passes, including shot batching, the scalar FP16 baseline, the default half2
packed path, aggressive half2 arithmetic, and optionally the material-gradient
backward pass.

`tm2d_hvp.py` compares native full and Gauss-Newton TM2D Hessian-vector
products with a central finite difference of two native gradients. It reports
CUDA/wall time and peak allocated memory for fixed small, medium, and large
cases. Add `--include-python` to measure the Python reference path on the small
case.

For the large float32 case on an RTX 4070 (20 repeats), the fused full-Hessian
path measured 66.59 ms versus 63.25 ms for finite differences and 39.61 ms for
Gauss-Newton. The pre-fusion full-Hessian baseline was 78.34 ms, so merging the
two reverse passes reduced its median runtime by about 15%.
