# Example: Multiscale Inversion with Random Source Encoding

Script: `examples/example_multiscale_random_sources.py`

## Goal
Show how random source encoding can reduce per-iteration cost while preserving multiscale inversion behavior.

## Inputs

- Base model tensors for epsilon, sigma, and mu
- Source wavelet bank and random encoding vectors
- Source and receiver geometry per shot
- Inversion schedule settings (bands, iterations, optimizer)

## Steps

1. Build observed data or load precomputed traces.
2. Encode multiple physical shots into randomized super-shots.
3. Run staged inversion from low to higher effective frequencies.
4. Update model parameters using configured optimizer.
5. Track loss and model snapshots per stage.

## Outputs

- Loss curves over iterations/stages
- Intermediate and final epsilon reconstructions
- Optional comparisons between encoded and reference traces
