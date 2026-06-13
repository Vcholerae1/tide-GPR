"""Shared shape and dtype aliases for TIDE public APIs."""

from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from numpy.typing import NDArray
from torch import Tensor

runtime_typecheck = jaxtyped(typechecker=beartype)

Model2D = Float[Tensor, "y x"]
BatchedModel2D = Float[Tensor, "batch y x"]
Model2DLike = Model2D | BatchedModel2D

Model3D = Float[Tensor, "z y x"]
BatchedModel3D = Float[Tensor, "batch z y x"]
Model3DLike = Model3D | BatchedModel3D

Field2DLike = (
    Float[Tensor, "y x"]
    | Float[Tensor, "shot y x"]
    | Float[Tensor, "batch shot y x"]
)
Field3DLike = (
    Float[Tensor, "z y x"]
    | Float[Tensor, "shot z y x"]
    | Float[Tensor, "batch shot z y x"]
)

WaveletBatch = (
    Float[Tensor, "shot source time"]
    | Float[Tensor, "batch shot source time"]
)

Location2D = Int[Tensor, "shot _ 2"] | Int[Tensor, "batch shot _ 2"]
SourceLocation2D = (
    Int[Tensor, "shot source 2"] | Int[Tensor, "batch shot source 2"]
)
ReceiverLocation2D = (
    Int[Tensor, "shot receiver 2"] | Int[Tensor, "batch shot receiver 2"]
)

Location3D = Int[Tensor, "shot _ 3"] | Int[Tensor, "batch shot _ 3"]
SourceLocation3D = (
    Int[Tensor, "shot source 3"] | Int[Tensor, "batch shot source 3"]
)
ReceiverLocation3D = (
    Int[Tensor, "shot receiver 3"] | Int[Tensor, "batch shot receiver 3"]
)

ReceiverData = (
    Float[Tensor, "time shot receiver"]
    | Float[Tensor, "batch time shot receiver"]
)

VectorF32 = Float[NDArray[np.float32], "n"]
MatrixF32 = Float[NDArray[np.float32], "m n"]

__all__ = [
    "BatchedModel2D",
    "BatchedModel3D",
    "Field2DLike",
    "Field3DLike",
    "Location2D",
    "Location3D",
    "MatrixF32",
    "Model2D",
    "Model2DLike",
    "Model3D",
    "Model3DLike",
    "ReceiverData",
    "ReceiverLocation2D",
    "ReceiverLocation3D",
    "SourceLocation2D",
    "SourceLocation3D",
    "VectorF32",
    "WaveletBatch",
    "runtime_typecheck",
]
