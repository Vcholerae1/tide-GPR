# Modeling

## Parameters
- epsilon: relative permittivity (dimensionless).
- sigma: electrical conductivity (S/m).
- mu: relative permeability (dimensionless).

## Units and Scaling

The public API is SI-compatible:
- `grid_spacing`: meters
- `dt`: seconds
- `sigma`: S/m
- `epsilon`, `mu`: relative values

When `compute_dtype="fp16"` is selected, the solver internally applies
nondimensional scaling:
- `L0 = min(dx, dy)`
- `T0 = L0 / cmax`
- `epsilon_hat = epsilon / min(epsilon)`
- `mu_hat = mu / min(mu)`
- `sigma_hat = sigma * T0 / eps_ref_abs`

Outputs are mapped back to physical units before returning.

## Grid and Shapes

- Model tensors: `[ny, nx]`.
- Sources: `source_amplitude [n_shots, n_sources, nt]`.
- Source/receiver indices: `[n_shots, n_{src/rec}, 2]`.
- Returned receiver data: `[nt, n_shots, n_receivers]`.

## Constraints and Masks

- `epsilon > 0`, `mu > 0` are required.
- Source/receiver coordinates must remain in model bounds.
- For fp16 mode, prefer moderate parameter contrast for best stability.
