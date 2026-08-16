"""Shared construction helpers for Maxwell module wrappers."""

from __future__ import annotations


import torch


def _same_receiver_locations(
    requested: torch.Tensor | None,
    primary: torch.Tensor | None,
) -> bool:
    return bool(
        requested is not None
        and requested.numel() > 0
        and primary is not None
        and torch.equal(requested, primary)
    )


__all__ = [
    "_same_receiver_locations",
]
