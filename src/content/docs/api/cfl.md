---
title: "Module: tide.cfl"
description: "Compute stable internal time steps for explicit FDTD propagation."
---

Numerical stability helper for explicit FDTD time stepping.

## Functions
- cfl_condition

## cfl_condition

Signature:

```python
cfl_condition(grid_spacing, dt, max_vel, c_max=1.0, eps=1e-15)
```

Returns:
- inner_dt: stable internal time step
- step_ratio: integer number of internal steps per user step

Usage notes:
- grid_spacing accepts a scalar or list of spacings.
- If step_ratio >= 2, a warning is emitted to indicate sub-stepping.
- max_vel must be positive.

## Interpretation

The returned `inner_dt` divides the requested `dt` by the integer
`step_ratio`. TIDE can therefore map source and receiver data between the user
sampling grid and a stable internal grid without changing the number of
returned user samples.

```python
inner_dt, ratio = tide.cfl_condition(
    grid_spacing=(0.02, 0.03),
    dt=4.0e-11,
    max_vel=299_792_458.0,
)
assert ratio >= 1
assert inner_dt == dt / ratio
```

`grid_spacing` may be scalar or per-axis. `max_vel` must bound the fastest
material used with the operator. Underestimating it invalidates the stability
calculation.

`c_max` is the method-specific Courant factor. Application code should normally
leave it at the solver default rather than using it as a performance control.
