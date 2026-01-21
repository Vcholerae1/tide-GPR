"""Grid-related helpers for padding and boundary bookkeeping."""

from typing import Sequence, Union


def _normalize_grid_spacing_2d(
    grid_spacing: Union[float, Sequence[float]],
) -> list[float]:
    """Normalize 2D grid spacing to [dy, dx]."""
    if isinstance(grid_spacing, (int, float)):
        return [float(grid_spacing), float(grid_spacing)]
    return list(grid_spacing)


def _normalize_pml_width_2d(
    pml_width: Union[int, Sequence[int]],
) -> list[int]:
    """Normalize 2D PML width to [top, bottom, left, right]."""
    if isinstance(pml_width, int):
        return [pml_width] * 4
    pml_width_list = list(pml_width)
    if len(pml_width_list) == 1:
        return pml_width_list * 4
    if len(pml_width_list) == 2:
        return [
            pml_width_list[0],
            pml_width_list[0],
            pml_width_list[1],
            pml_width_list[1],
        ]
    return pml_width_list
