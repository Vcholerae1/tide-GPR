# Module: tide.utils

Core physical constants and low-level coefficient builders.

## Constants
- EP0
- MU0

Also exposed internally:
- C0 (speed of light in vacuum)

## Functions
- setup_pml
- setup_pml_half

## prepare_parameters

Builds electromagnetic update coefficients from model parameters:
- ca and cb for electric-field update
- cq for magnetic-field update

## compile_material_coefficients

Compiles coefficient dictionaries for default and Debye-dispersive materials.

Returns keys including:
- ca, cb, cq
- has_dispersion
- debye (when dispersion is enabled)

## setup_pml / setup_pml_half

Generates CPML profile tensors used by staggered-grid kernels.

These functions are generally consumed through tide.staggered helpers rather than called directly.
