"""CFL condition helpers for stable time stepping."""

import math
from typing import Union


def cfl_condition(
    grid_spacing: Union[float, list[float]],
    dt: float,
    max_vel: float,
    c_max: float = 1,
    eps: float = 1e-15,
) -> tuple[float, int]:
    """Calculate time step interval to satisfy CFL condition.

    The CFL (Courant-Friedrichs-Lewy) condition ensures numerical stability
    for explicit FDTD schemes. If the user-provided dt is too large, this
    function computes a smaller internal dt and the ratio between them.

    Args:
        grid_spacing: Grid spacing [dy, dx] or single value for isotropic.
        dt: User-provided time step.
        max_vel: Maximum wave velocity in the model.
        c_max: Maximum Courant number (default 0.6 for stability margin).
        eps: Small value to prevent division by zero.

    Returns:
        Tuple of (inner_dt, step_ratio) where:
        - inner_dt: Time step satisfying CFL condition
        - step_ratio: Integer ratio dt / inner_dt

    Example:
        >>> # Check if dt=1e-9 is stable for v=3e8 m/s, dx=1e-3 m
        >>> inner_dt, ratio = cfl_condition([1e-3, 1e-3], 1e-9, 3e8)
        >>> print(f"Need {ratio}x smaller time step")
    """
    # Normalize grid_spacing to list
    if isinstance(grid_spacing, (int, float)):
        grid_spacing = [float(grid_spacing), float(grid_spacing)]
    else:
        grid_spacing = list(grid_spacing)

    if max_vel <= 0:
        raise ValueError("max_vel must be positive")

    # Maximum stable dt from CFL condition
    max_dt = (
        c_max / math.sqrt(sum(1 / dx**2 for dx in grid_spacing)) / (max_vel**2 + eps)
    ) * max_vel

    step_ratio = math.ceil(abs(dt) / max_dt)
    inner_dt = dt / step_ratio

    if step_ratio >= 2:
        import warnings

        warnings.warn(
            f"CFL condition requires {step_ratio} internal time steps per "
            f"user time step (dt={dt}, inner_dt={inner_dt}). Consider using "
            "a smaller dt or coarser grid.",
            stacklevel=2,
        )

    return inner_dt, step_ratio
