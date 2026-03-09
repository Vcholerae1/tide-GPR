# Modeling

## Parameters
- epsilon: relative permittivity (dimensionless).
- sigma: electrical conductivity (S/m).
- mu: relative permeability (dimensionless).

## Units

The public API is SI-compatible:
- `grid_spacing`: meters
- `dt`: seconds
- `sigma`: S/m
- `epsilon`, `mu`: relative values

## Grid and Shapes

- Model tensors: `[ny, nx]`.
- Sources: `source_amplitude [n_shots, n_sources, nt]`.
- Source/receiver indices: `[n_shots, n_{src/rec}, 2]`.
- Returned receiver data: `[nt, n_shots, n_receivers]`.

## Constraints and Masks

- `epsilon > 0`, `mu > 0` are required.
- Source/receiver coordinates must remain in model bounds.
