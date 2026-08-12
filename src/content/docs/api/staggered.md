---
title: "Module: tide.staggered"
description: "Construct staggered-grid CPML profiles for 2D and 3D solvers."
---

Staggered-grid CPML profile builders for 2D and 3D propagation.

## Functions
- set_pml_profiles
- setup_pml_profiles_1d
- set_pml_profiles_3d

## set_pml_profiles

2D helper that returns 12 tensors:
- a/b profiles: ay, ayh, ax, axh, by, byh, bx, bxh
- kappa profiles: ky, kyh, kx, kxh

Output tensors are reshaped for broadcasting against [batch, ny, nx] fields.

## setup_pml_profiles_1d

Builds 1D CPML profiles for integer and half-grid points.
Useful for isolated profile testing.

## set_pml_profiles_3d

3D helper that returns:
- 12 a/b profile tensors for z/y/x integer and half grids
- 6 kappa profile tensors

## Derivative helpers

The module also contains staggered first-derivative operators:

- `diffx1`, `diffy1`, and `diffz1` evaluate at integer grid points.
- `diffxh1`, `diffyh1`, and `diffzh1` evaluate at half-grid points.

Their reciprocal-spacing tensors must be broadcast-compatible with the field,
and `stencil` selects the finite-difference coefficients.

## Intended use

These functions are public for numerical studies and backend parity tests, but
most applications should obtain CPML and staggered updates through
`MaxwellTM` or `Maxwell3D`. Calling profile builders directly requires the
caller to preserve axis order, integer versus half-grid placement, dtype,
device, and padded-domain sizes.

For a profile study, inspect both integer and half-grid tensors and verify that
coefficients are inactive in the physical interior and vary smoothly through
each absorbing layer.
