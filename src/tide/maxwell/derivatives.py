"""Internal derivative adapters for Maxwell operator implementations."""

from __future__ import annotations

from typing import Any

from .maxwell3d_born import born3d
from .tm2d_born import borntm


def tm2d_jvp(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Run the TM2D tangent adapter."""
    return borntm(*args, **kwargs)


def em3d_jvp(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Run the EM3D tangent adapter."""
    return born3d(*args, **kwargs)


__all__: list[str] = []
