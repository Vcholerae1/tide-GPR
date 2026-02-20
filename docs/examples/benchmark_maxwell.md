# Example: Maxwell Benchmark

Script: `examples/benchmark_maxwell.py`

## Goal
Measure end-to-end runtime for `tide.maxwelltm` forward and forward+backward under fixed workloads.

## Inputs
The script runs two scenarios:

- `small`: `ny=96, nx=96, nt=128, n_shots=2`
- `medium`: `ny=160, nx=160, nt=192, n_shots=4`

Common settings: `stencil=2`, `pml_width=8`, Ricker source.
Precision is configurable with `--compute-dtype` and `--mp-mode`.

## Steps
1. Pick runtime device in priority order: `mps` (with Metal backend), then `cuda`, then `cpu`.
2. For each scenario and mode (`forward`, `forward+backward`):
3. Run 3 warmup iterations.
4. Run 10 timed iterations and synchronize device around each measurement.
5. Print mean/P50/P90 latency in milliseconds.

## Output
CSV-like rows:

`Case,Mode,Mean(ms),P50(ms),P90(ms)`

Use these rows to compare before/after optimization on the same machine and config.
