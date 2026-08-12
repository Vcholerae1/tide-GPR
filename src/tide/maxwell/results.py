"""Named state and result objects returned by Maxwell operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch


@dataclass(frozen=True, slots=True)
class TMState:
    Ey: torch.Tensor
    Hx: torch.Tensor
    Hz: torch.Tensor
    m_Ey_x: torch.Tensor
    m_Ey_z: torch.Tensor
    m_Hx_z: torch.Tensor
    m_Hz_x: torch.Tensor

    @classmethod
    def from_tensors(cls, tensors: tuple[torch.Tensor, ...]) -> TMState:
        if len(tensors) != 7:
            raise ValueError(f"TM state requires 7 tensors, got {len(tensors)}.")
        return cls(*tensors)


@dataclass(frozen=True, slots=True)
class EM3DState:
    Ex: torch.Tensor
    Ey: torch.Tensor
    Ez: torch.Tensor
    Hx: torch.Tensor
    Hy: torch.Tensor
    Hz: torch.Tensor
    m_hz_y: torch.Tensor
    m_hy_z: torch.Tensor
    m_hx_z: torch.Tensor
    m_hz_x: torch.Tensor
    m_hy_x: torch.Tensor
    m_hx_y: torch.Tensor
    m_ey_z: torch.Tensor
    m_ez_y: torch.Tensor
    m_ez_x: torch.Tensor
    m_ex_z: torch.Tensor
    m_ex_y: torch.Tensor
    m_ey_x: torch.Tensor

    @classmethod
    def from_tensors(cls, tensors: tuple[torch.Tensor, ...]) -> EM3DState:
        if len(tensors) != 18:
            raise ValueError(f"3-D state requires 18 tensors, got {len(tensors)}.")
        return cls(*tensors)


StateT = TypeVar("StateT", TMState, EM3DState)


@dataclass(frozen=True, slots=True)
class ForwardResult(Generic[StateT]):
    """Receiver samples and final state from a nonlinear propagation."""

    receiver_data: torch.Tensor
    final_state: StateT


@dataclass(frozen=True, slots=True)
class TangentResult(Generic[StateT]):
    """Receiver samples and final state from one tangent propagation."""

    receiver_data: torch.Tensor
    final_state: StateT


@dataclass(frozen=True, slots=True)
class EMGradient:
    """Named cotangent in electromagnetic model space."""

    epsilon: torch.Tensor | None
    sigma: torch.Tensor | None
    mu: torch.Tensor | None = None


__all__ = [
    "EM3DState",
    "EMGradient",
    "ForwardResult",
    "TMState",
    "TangentResult",
]
