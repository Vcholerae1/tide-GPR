from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

PMLDirection = Literal["z", "y", "x"]

_AUX_DIRECTIONS_3D: dict[str, PMLDirection] = {
    "m_hy_z": "z",
    "m_hx_z": "z",
    "m_ey_z": "z",
    "m_ex_z": "z",
    "m_hz_y": "y",
    "m_hx_y": "y",
    "m_ez_y": "y",
    "m_ex_y": "y",
    "m_hz_x": "x",
    "m_hy_x": "x",
    "m_ez_x": "x",
    "m_ey_x": "x",
}


@dataclass(frozen=True)
class CompactPMLLayout3D:
    n_shots: int
    nz: int
    ny: int
    nx: int
    z_indices: tuple[int, ...]
    y_indices: tuple[int, ...]
    x_indices: tuple[int, ...]

    def direction_indices(self, direction: PMLDirection) -> tuple[int, ...]:
        if direction == "z":
            return self.z_indices
        if direction == "y":
            return self.y_indices
        return self.x_indices

    def direction_shape(self, direction: PMLDirection) -> tuple[int, int, int, int]:
        if direction == "z":
            return (self.n_shots, len(self.z_indices), self.ny, self.nx)
        if direction == "y":
            return (self.n_shots, self.nz, len(self.y_indices), self.nx)
        return (self.n_shots, self.nz, self.ny, len(self.x_indices))

    def aux_shape(self, aux_name: str) -> tuple[int, int, int, int]:
        return self.direction_shape(aux_direction_3d(aux_name))

    @property
    def full_aux_elements(self) -> int:
        return 12 * self.n_shots * self.nz * self.ny * self.nx

    @property
    def compact_aux_elements(self) -> int:
        z_count = 4 * self.n_shots * len(self.z_indices) * self.ny * self.nx
        y_count = 4 * self.n_shots * self.nz * len(self.y_indices) * self.nx
        x_count = 4 * self.n_shots * self.nz * self.ny * len(self.x_indices)
        return z_count + y_count + x_count

    @property
    def compact_to_full_ratio(self) -> float:
        if self.full_aux_elements == 0:
            return 0.0
        return self.compact_aux_elements / self.full_aux_elements

    def as_dict(self) -> dict[str, object]:
        return {
            "n_shots": self.n_shots,
            "shape": [self.nz, self.ny, self.nx],
            "z_compact_width": len(self.z_indices),
            "y_compact_width": len(self.y_indices),
            "x_compact_width": len(self.x_indices),
            "full_aux_elements": self.full_aux_elements,
            "compact_aux_elements": self.compact_aux_elements,
            "compact_to_full_ratio": self.compact_to_full_ratio,
            "aux_directions": dict(sorted(_AUX_DIRECTIONS_3D.items())),
        }


def aux_direction_3d(aux_name: str) -> PMLDirection:
    try:
        return _AUX_DIRECTIONS_3D[aux_name]
    except KeyError as exc:
        raise KeyError(f"Unknown 3D CPML auxiliary field: {aux_name}") from exc


def _axis_indices(
    *,
    size: int,
    fd_low: int,
    fd_high: int,
    pml_low: int,
    pml_high: int,
) -> tuple[int, ...]:
    low_end = max(0, min(size, fd_low + pml_low))
    high_start = max(low_end, size - fd_high - pml_high)
    return tuple(range(low_end)) + tuple(range(high_start, size))


def build_compact_pml_layout_3d(
    *,
    n_shots: int,
    nz: int,
    ny: int,
    nx: int,
    fd_pad: tuple[int, int, int, int, int, int],
    pml_width: tuple[int, int, int, int, int, int],
) -> CompactPMLLayout3D:
    if len(fd_pad) != 6:
        raise ValueError("fd_pad must have 6 entries: z0, z1, y0, y1, x0, x1")
    if len(pml_width) != 6:
        raise ValueError("pml_width must have 6 entries: z0, z1, y0, y1, x0, x1")
    return CompactPMLLayout3D(
        n_shots=n_shots,
        nz=nz,
        ny=ny,
        nx=nx,
        z_indices=_axis_indices(
            size=nz,
            fd_low=fd_pad[0],
            fd_high=fd_pad[1],
            pml_low=pml_width[0],
            pml_high=pml_width[1],
        ),
        y_indices=_axis_indices(
            size=ny,
            fd_low=fd_pad[2],
            fd_high=fd_pad[3],
            pml_low=pml_width[2],
            pml_high=pml_width[3],
        ),
        x_indices=_axis_indices(
            size=nx,
            fd_low=fd_pad[4],
            fd_high=fd_pad[5],
            pml_low=pml_width[4],
            pml_high=pml_width[5],
        ),
    )


def pack_directional_aux_3d(
    full: torch.Tensor,
    *,
    layout: CompactPMLLayout3D,
    direction: PMLDirection,
) -> torch.Tensor:
    if full.shape != (layout.n_shots, layout.nz, layout.ny, layout.nx):
        raise ValueError(
            "full aux shape must match layout "
            f"{(layout.n_shots, layout.nz, layout.ny, layout.nx)}, got {tuple(full.shape)}"
        )
    dim = {"z": 1, "y": 2, "x": 3}[direction]
    indices = torch.as_tensor(
        layout.direction_indices(direction), device=full.device, dtype=torch.long
    )
    return full.index_select(dim, indices).contiguous()


def unpack_directional_aux_3d(
    compact: torch.Tensor,
    *,
    layout: CompactPMLLayout3D,
    direction: PMLDirection,
) -> torch.Tensor:
    expected_shape = layout.direction_shape(direction)
    if compact.shape != expected_shape:
        raise ValueError(f"compact aux shape must be {expected_shape}, got {tuple(compact.shape)}")
    dim = {"z": 1, "y": 2, "x": 3}[direction]
    indices = torch.as_tensor(
        layout.direction_indices(direction), device=compact.device, dtype=torch.long
    )
    full = torch.zeros(
        layout.n_shots,
        layout.nz,
        layout.ny,
        layout.nx,
        device=compact.device,
        dtype=compact.dtype,
    )
    full.index_copy_(dim, indices, compact)
    return full


def pack_aux_field_3d(
    aux_name: str,
    full: torch.Tensor,
    *,
    layout: CompactPMLLayout3D,
) -> torch.Tensor:
    return pack_directional_aux_3d(
        full,
        layout=layout,
        direction=aux_direction_3d(aux_name),
    )


def unpack_aux_field_3d(
    aux_name: str,
    compact: torch.Tensor,
    *,
    layout: CompactPMLLayout3D,
) -> torch.Tensor:
    return unpack_directional_aux_3d(
        compact,
        layout=layout,
        direction=aux_direction_3d(aux_name),
    )


__all__ = [
    "CompactPMLLayout3D",
    "aux_direction_3d",
    "build_compact_pml_layout_3d",
    "pack_aux_field_3d",
    "pack_directional_aux_3d",
    "unpack_aux_field_3d",
    "unpack_directional_aux_3d",
]
