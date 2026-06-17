"""Acquisition geometry helpers for TIDE workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ReceiverMode = Literal["shared", "paired"]


@dataclass(frozen=True, slots=True)
class Acquisition:
    """Source and receiver locations following TIDE solver conventions."""

    source_location: torch.Tensor
    receiver_location: torch.Tensor

    @property
    def n_shots(self) -> int:
        return int(self.source_location.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.receiver_location.shape[1])

    @property
    def spatial_ndim(self) -> int:
        return int(self.source_location.shape[-1])


def _as_long_tensor(
    name: str,
    value: torch.Tensor,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=torch.long)
    if value.ndim not in (1, 2, 3):
        raise ValueError(f"{name} must be a 1D, 2D, or 3D tensor.")
    return value


def _as_location_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(1)
    if value.ndim == 3:
        return value
    raise ValueError(f"{name} must be shaped [N, D] or [N, M, D].")


def point_acquisition(
    source_points: torch.Tensor,
    receiver_points: torch.Tensor,
    *,
    receiver_mode: ReceiverMode = "shared",
    device: torch.device | str | None = None,
) -> Acquisition:
    """Build solver-ready locations from source and receiver point tensors.

    ``source_points`` may be shaped ``[S, D]`` or ``[S, Ns, D]``. If
    ``receiver_points`` is shaped ``[R, D]``, ``receiver_mode="shared"`` repeats
    those receivers for every shot and ``receiver_mode="paired"`` treats each
    receiver point as the single receiver for the corresponding shot. A
    ``[S, R, D]`` receiver tensor is accepted as already shot-indexed.
    """

    if receiver_mode not in ("shared", "paired"):
        raise ValueError("receiver_mode must be 'shared' or 'paired'.")

    source = _as_location_tensor(
        "source_points",
        _as_long_tensor("source_points", source_points, device=device),
    )
    receiver = _as_long_tensor("receiver_points", receiver_points, device=device)
    n_shots = int(source.shape[0])
    spatial_ndim = int(source.shape[-1])

    if receiver.ndim == 2:
        if int(receiver.shape[-1]) != spatial_ndim:
            raise ValueError("source_points and receiver_points must have the same spatial dimension.")
        if receiver_mode == "shared":
            receiver_location = receiver.unsqueeze(0).expand(n_shots, -1, -1).contiguous()
        else:
            if int(receiver.shape[0]) != n_shots:
                raise ValueError(
                    "paired receiver_points must have the same first dimension as source_points."
                )
            receiver_location = receiver.unsqueeze(1)
    elif receiver.ndim == 3:
        if int(receiver.shape[0]) != n_shots:
            raise ValueError("receiver_points shaped [S, R, D] must match source shot count.")
        if int(receiver.shape[-1]) != spatial_ndim:
            raise ValueError("source_points and receiver_points must have the same spatial dimension.")
        receiver_location = receiver
    else:
        raise ValueError("receiver_points must be shaped [R, D], [S, D], or [S, R, D].")

    return Acquisition(source_location=source.contiguous(), receiver_location=receiver_location.contiguous())


def line_acquisition_2d(
    source_x: torch.Tensor,
    receiver_x: torch.Tensor,
    *,
    source_depth: int,
    receiver_depth: int | None = None,
    receiver_mode: ReceiverMode = "shared",
    device: torch.device | str | None = None,
) -> Acquisition:
    """Build a 2D line acquisition from x coordinates and fixed depths."""

    source_x = _as_long_tensor("source_x", source_x, device=device)
    receiver_x = _as_long_tensor("receiver_x", receiver_x, device=device)
    if source_x.ndim != 1 or receiver_x.ndim != 1:
        raise ValueError("source_x and receiver_x must be 1D tensors.")

    receiver_depth = source_depth if receiver_depth is None else receiver_depth
    source_points = torch.stack(
        [
            torch.full_like(source_x, int(source_depth)),
            source_x,
        ],
        dim=-1,
    )
    receiver_points = torch.stack(
        [
            torch.full_like(receiver_x, int(receiver_depth)),
            receiver_x,
        ],
        dim=-1,
    )
    return point_acquisition(
        source_points,
        receiver_points,
        receiver_mode=receiver_mode,
        device=device,
    )


__all__ = ["Acquisition", "ReceiverMode", "line_acquisition_2d", "point_acquisition"]
