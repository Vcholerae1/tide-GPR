"""Grid-related helpers for padding and boundary bookkeeping."""

from typing import Sequence, Union

import torch


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


def _compute_boundary_indices_flat(
    ny: int,
    nx: int,
    pml_y0: int,
    pml_x0: int,
    pml_y1: int,
    pml_x1: int,
    boundary_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Return flattened indices (y * nx + x) for a PML-side boundary ring.

    The ring is a union of strips adjacent to the physical domain inside the
    PML region. It is intended for boundary saving/injection and should have
    width >= FD_PAD (stencil/2).
    """
    if boundary_width <= 0:
        raise ValueError("boundary_width must be positive.")

    max_width = min(pml_y0, ny - pml_y1, pml_x0, nx - pml_x1)
    if boundary_width > max_width:
        raise ValueError(
            "boundary_width must fit within the padded grid; "
            f"got boundary_width={boundary_width}, max_width={max_width}."
        )

    # Construct indices in a cache-friendly order:
    # - top strip rows (contiguous in memory per row)
    # - bottom strip rows
    # - middle rows: left strip then right strip per row
    # This ordering avoids the mask+nonzero pattern which produces a more
    # irregular access pattern on GPU gather/scatter.
    x_all = torch.arange(nx, device="cpu", dtype=torch.int64)

    top_y = torch.arange(
        pml_y0 - boundary_width, pml_y0, device="cpu", dtype=torch.int64
    )
    bottom_y = torch.arange(
        pml_y1, pml_y1 + boundary_width, device="cpu", dtype=torch.int64
    )
    mid_y = torch.arange(pml_y0, pml_y1, device="cpu", dtype=torch.int64)
    x_left = torch.arange(
        pml_x0 - boundary_width, pml_x0, device="cpu", dtype=torch.int64
    )
    x_right = torch.arange(
        pml_x1, pml_x1 + boundary_width, device="cpu", dtype=torch.int64
    )

    top = (top_y[:, None] * nx + x_all[None, :]).reshape(-1)
    bottom = (bottom_y[:, None] * nx + x_all[None, :]).reshape(-1)

    mid_base = mid_y * nx
    mid_left = mid_base[:, None] + x_left[None, :]
    mid_right = mid_base[:, None] + x_right[None, :]
    mid = torch.cat((mid_left, mid_right), dim=1).reshape(-1)

    flat = torch.cat((top, bottom, mid), dim=0).to(torch.int64)
    return flat.to(device=device)
