# Module: tide.cfl

Numerical stability helper for explicit FDTD time stepping.

## Functions
- cfl_condition

## cfl_condition

Signature:

```python
cfl_condition(grid_spacing, dt, max_vel, c_max=0.8, eps=1e-15)
```

Returns:
- inner_dt: stable internal time step
- step_ratio: integer number of internal steps per user step

Usage notes:
- grid_spacing accepts a scalar or list of spacings.
- If step_ratio >= 2, a warning is emitted to indicate sub-stepping.
- max_vel must be positive.
