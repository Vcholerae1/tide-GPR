"""Shared construction helpers for Maxwell module wrappers."""

from __future__ import annotations

from typing import Literal

import torch

from .validation_internal import _validate_optional_bool, _validate_tensor_arg


def _register_maxwell_model(
    module: torch.nn.Module,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    *,
    epsilon_requires_grad: bool | None,
    sigma_requires_grad: bool | None,
) -> None:
    for name, value, requires_grad in (
        ("epsilon", epsilon, epsilon_requires_grad),
        ("sigma", sigma, sigma_requires_grad),
    ):
        _validate_optional_bool(f"{name}_requires_grad", requires_grad)
        _validate_tensor_arg(name, value)
        resolved_requires_grad = (
            value.requires_grad if requires_grad is None else requires_grad
        )
        module.register_parameter(
            name,
            torch.nn.Parameter(value, requires_grad=resolved_requires_grad),
        )
    _validate_tensor_arg("mu", mu)
    module.register_buffer("mu", mu)


def _register_optional_parameter(
    module: torch.nn.Module,
    name: str,
    value: torch.Tensor | None,
    requires_grad: bool | None,
) -> None:
    _validate_optional_bool(f"{name}_requires_grad", requires_grad)
    if value is None:
        module.register_parameter(name, None)
        return
    _validate_tensor_arg(name, value)
    resolved_requires_grad = value.requires_grad if requires_grad is None else requires_grad
    module.register_parameter(
        name,
        torch.nn.Parameter(value, requires_grad=resolved_requires_grad),
    )


def _validate_born_parameterization(
    parameterization: str,
) -> Literal["epsilon_sigma", "ca_cb"]:
    if parameterization not in {"epsilon_sigma", "ca_cb"}:
        raise ValueError(
            "parameterization must be 'epsilon_sigma' or 'ca_cb', "
            f"got {parameterization!r}."
        )
    return parameterization  # type: ignore[return-value]


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
    "_register_maxwell_model",
    "_register_optional_parameter",
    "_same_receiver_locations",
    "_validate_born_parameterization",
]
