from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DebyeDispersion:
    """Debye dispersion parameters for one or more poles.

    `delta_epsilon` and `tau` may be scalars, model-shaped tensors, or
    `[n_poles, *model_shape]` tensors. Scalars and model-shaped tensors are
    interpreted as single-pole inputs.
    """

    delta_epsilon: torch.Tensor | float
    tau: torch.Tensor | float

