"""Grid-related helpers for padding and boundary bookkeeping."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch


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


@dataclass(frozen=True, slots=True)
class _CompactCPMLLayout:
    """Internal shape bookkeeping for axis-compact CPML state."""

    shots: int
    spatial_shape: tuple[int, ...]
    pml_bounds: tuple[tuple[int, int], ...]

    def shape(self, axis: int) -> tuple[int, ...]:
        size = self.spatial_shape[axis]
        low, high = self.pml_bounds[axis]
        shape = [self.shots, *self.spatial_shape]
        shape[axis + 1] = low + size - max(low, high - 1)
        return tuple(shape)

    def coordinates(
        self, axis: int, *, device: torch.device | None = None
    ) -> torch.Tensor:
        size = self.spatial_shape[axis]
        low, high = self.pml_bounds[axis]
        high_start = max(low, high - 1)
        return torch.cat(
            (
                torch.arange(low, device=device, dtype=torch.int64),
                torch.arange(high_start, size, device=device, dtype=torch.int64),
            )
        )

    def pack(self, state: torch.Tensor, axis: int) -> torch.Tensor:
        expected = (self.shots, *self.spatial_shape)
        if tuple(state.shape) != expected:
            raise ValueError(f"full CPML state must have shape {expected}")
        return state.index_select(axis + 1, self.coordinates(axis, device=state.device))

    def unpack(self, state: torch.Tensor, axis: int) -> torch.Tensor:
        expected = self.shape(axis)
        if tuple(state.shape) != expected:
            raise ValueError(f"compact CPML state must have shape {expected}")
        result = torch.zeros(
            (self.shots, *self.spatial_shape),
            device=state.device,
            dtype=state.dtype,
        )
        return result.index_copy(
            axis + 1, self.coordinates(axis, device=state.device), state
        )

    def zero_interior_(self, state: torch.Tensor, axis: int) -> None:
        low, high = self.pml_bounds[axis]
        slices = [slice(None)] * state.ndim
        slices[axis + 1] = slice(low, max(low, high - 1))
        state[tuple(slices)].zero_()

    def zeros(
        self, axis: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(self.shape(axis), device=device, dtype=dtype)
