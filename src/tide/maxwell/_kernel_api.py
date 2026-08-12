"""Private tuple-based kernel contract used by backend regression tests.

The structured operator API is the only supported public surface.
"""

from .maxwell3d import maxwell3d, maxwell3d_hvp
from .maxwell3d_born import born3d
from .tm2d import maxwelltm, maxwelltm_hvp
from .tm2d_born import borntm
from .tm2d_linearization import linearize_maxwelltm

__all__ = [
    "born3d",
    "borntm",
    "linearize_maxwelltm",
    "maxwell3d",
    "maxwell3d_hvp",
    "maxwelltm",
    "maxwelltm_hvp",
]
