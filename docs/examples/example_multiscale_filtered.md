# Example: Multiscale Filtered Inversion

Script: `examples/example_multiscale_filtered.py`

## Goal
TODO: Explain the multiscale workflow and FIR low-pass schedule.

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
TODO: Document expected runtime and device requirements.
