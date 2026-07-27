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
