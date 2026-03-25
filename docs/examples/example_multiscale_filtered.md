# Example: Multiscale Filtered Inversion

Script: `examples/example_multiscale_filtered.py`

## Goal
Demonstrate multiscale inversion by progressively increasing data bandwidth using FIR low-pass schedules.

## Inputs
- Model file: `examples/data/OverThrust.npy`
- Key parameters: dx, dt, nt, pml_width, n_shots, storage_mode

## Steps
1. Generate base observed data at a fixed forward frequency.
2. Apply FIR low-pass filters to create multiscale datasets.
3. Run staged inversion (AdamW then LBFGS per stage).

## Outputs
- Filtered data comparison image.
- Stage snapshots of epsilon.
- Summary plot with loss curve.

## Notes

- CUDA is strongly recommended for practical runtime.
- Runtime depends on grid size, shot count, and inversion stage count.
- For debugging, reduce nt and n_shots first.
