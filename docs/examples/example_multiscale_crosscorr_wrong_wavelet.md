# Example: Multiscale Inversion With Wavelet Mismatch

Script: `examples/example_multiscale_crosscorr_wrong_wavelet.py`

## Goal

Show how inversion behavior changes when the observed and inversion wavelets do not match exactly.

## What It Demonstrates

- multiscale inversion under source mismatch
- cross-correlation-style objectives
- filtering observed data into staged low-pass bands
- tracking forward and adjoint PDE counts while batching shots

## Expected Outputs

- wavelet comparison figure
- filtered-data comparison figure
- inversion progress figures in a dedicated output directory

## Runtime Notes

- the script assumes the `examples/data/OverThrust.npy` model is present
- CUDA is strongly recommended for practical runtimes
- it is a larger workflow example, not a minimal smoke test
