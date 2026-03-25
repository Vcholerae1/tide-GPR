# Module: tide.staggered

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
