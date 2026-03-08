"""Grid-related helpers for padding and boundary bookkeeping."""

from collections.abc import Sequence


def _normalize_grid_spacing_2d(
    grid_spacing: float | Sequence[float],
) -> list[float]:
    """Normalize 2D grid spacing to [dy, dx]."""
    if isinstance(grid_spacing, (int, float)):
        return [float(grid_spacing), float(grid_spacing)]
    grid_spacing_list = list(grid_spacing)
    if len(grid_spacing_list) == 1:
        return [float(grid_spacing_list[0]), float(grid_spacing_list[0])]
    if len(grid_spacing_list) != 2:
        raise ValueError(
            f"2D grid_spacing must have length 1 or 2, got {len(grid_spacing_list)}."
        )
    return [float(grid_spacing_list[0]), float(grid_spacing_list[1])]


def _normalize_grid_spacing_3d(
    grid_spacing: float | Sequence[float],
) -> list[float]:
    """Normalize 3D grid spacing to [dz, dy, dx]."""
    if isinstance(grid_spacing, (int, float)):
        val = float(grid_spacing)
        return [val, val, val]
    grid_spacing_list = list(grid_spacing)
    if len(grid_spacing_list) == 1:
        val = float(grid_spacing_list[0])
        return [val, val, val]
    if len(grid_spacing_list) != 3:
        raise ValueError(
            f"3D grid_spacing must have length 1 or 3, got {len(grid_spacing_list)}."
        )
    return [
        float(grid_spacing_list[0]),
        float(grid_spacing_list[1]),
        float(grid_spacing_list[2]),
    ]


def _normalize_pml_width_2d(
    pml_width: int | Sequence[int],
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
    if len(pml_width_list) != 4:
        raise ValueError(
            f"2D pml_width must have length 1, 2, or 4, got {len(pml_width_list)}."
        )
    return [int(v) for v in pml_width_list]


def _normalize_pml_width_3d(
    pml_width: int | Sequence[int],
) -> list[int]:
    """Normalize 3D PML width to [z0, z1, y0, y1, x0, x1]."""
    if isinstance(pml_width, int):
        return [pml_width] * 6
    pml_width_list = list(pml_width)
    if len(pml_width_list) == 1:
        return [int(pml_width_list[0])] * 6
    if len(pml_width_list) == 3:
        return [
            int(pml_width_list[0]),
            int(pml_width_list[0]),
            int(pml_width_list[1]),
            int(pml_width_list[1]),
            int(pml_width_list[2]),
            int(pml_width_list[2]),
        ]
    if len(pml_width_list) != 6:
        raise ValueError(
            f"3D pml_width must have length 1, 3, or 6, got {len(pml_width_list)}."
        )
    return [int(v) for v in pml_width_list]
