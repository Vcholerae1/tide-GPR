# Modeling

## Parameters
- epsilon: relative permittivity (dimensionless).
- sigma: electrical conductivity (S/m).
- mu: relative permeability (dimensionless).

For physically meaningful setups:
- epsilon must be strictly positive.
- mu is typically 1.0 for non-magnetic media.
- sigma should be non-negative for passive media.

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

For 3D:
- Model tensors: [nz, ny, nx]
- Source/receiver indices: [n_shots, n_points, 3] in [z, y, x]

## Constraints and Masks

- `epsilon > 0`, `mu > 0` are required.
- Source/receiver coordinates must remain in model bounds.

## Recommended Defaults

- stencil: 2 for exploratory runs, 4 or higher for reduced dispersion.
- pml_width: 8 to 20 depending on frequency content and grid size.
- dtype: float32 for most workloads, float64 for strict numerical studies.
