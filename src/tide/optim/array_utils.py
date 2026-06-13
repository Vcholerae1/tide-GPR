"""Array helpers for CPU-state optimization prototypes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .types import Objective, Preconditioner


def _as_float32_vector(
    name: str, value: ArrayLike, size: int | None = None
) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"{name} has size {array.size}, expected {size}.")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array.copy()


def _evaluate_objective(
    objective: Objective,
    x: NDArray[np.float32],
    grad: NDArray[np.float32],
) -> float:
    return float(objective(x, grad))


def _apply_preconditioner(
    preconditioner: Preconditioner | None,
    x: NDArray[np.float32],
    vector: NDArray[np.float32],
    out: NDArray[np.float32],
) -> bool:
    if preconditioner is None:
        out[:] = vector
        return False
    preconditioner(x, vector, out)
    return True


def _project(
    x: NDArray[np.float32],
    lower_bounds: NDArray[np.float32] | None,
    upper_bounds: NDArray[np.float32] | None,
    margin: float,
) -> None:
    if lower_bounds is None:
        return
    np.minimum(x, upper_bounds - margin, out=x)
    np.maximum(x, lower_bounds + margin, out=x)


def _set_trial_step(
    x: NDArray[np.float32],
    xk: NDArray[np.float32],
    descent: NDArray[np.float32],
    alpha: float,
) -> None:
    np.multiply(descent, np.float32(alpha), out=x)
    x += xk

