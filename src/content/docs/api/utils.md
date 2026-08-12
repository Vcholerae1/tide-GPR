---
title: "Module: tide.utils"
description: "Physical constants and finite-difference coefficient builders used by Maxwell kernels."
---

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

## Material coefficient equations

For the standard conductive electric-field update, `prepare_parameters`
constructs coefficients equivalent to

$$
c_a = \frac{1-\sigma\Delta t/(2\epsilon)}
           {1+\sigma\Delta t/(2\epsilon)},\qquad
c_b = \frac{\Delta t/\epsilon}
           {1+\sigma\Delta t/(2\epsilon)}.
$$

The implementation uses relative material tensors together with `EP0` and
`MU0` to form physical coefficients. `cq` supplies the magnetic-field update
scale.

```python
ca, cb, cq = tide.utils.prepare_parameters(
    epsilon,
    sigma,
    mu,
    dt,
)
```

## Compiled material dictionaries

`compile_material_coefficients` validates material inputs and returns the
coefficient set required by the selected constitutive path. With Debye
dispersion, the dictionary also contains auxiliary update data and marks
`has_dispersion=True`.

`linearize_material_coefficients` applies the coefficient derivative for
`depsilon_r` and `dsigma`. It is used by tangent propagation so the JVP follows
the same discrete material update as the nonlinear solver.

## Low-level boundary profiles

`setup_pml` and `setup_pml_half` construct one-dimensional CPML `a`, `b`, and
`k` profiles for integer and half-grid locations. Most callers should use
`tide.staggered`, which reshapes and orders these profiles for 2D or 3D field
updates.
