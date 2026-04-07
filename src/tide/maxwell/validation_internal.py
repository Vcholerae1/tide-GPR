from typing import Any

import torch

from ..dispersion import DebyeDispersion

_COMPONENT_TO_INDEX_3D = {"ex": 0, "ey": 1, "ez": 2}


def _validate_optional_bool(name: str, value: bool | None) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(
            f"{name} must be bool or None, got {type(value).__name__}",
        )


def _validate_tensor_arg(name: str, value: Any) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(value).__name__}")


def _validate_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_optional_positive_int(name: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int or None.")
    if value <= 0:
        raise ValueError(f"{name} must be positive when provided.")


def _validate_location_bounds(
    location: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    name: str,
    check_lower_bound: bool,
) -> None:
    if location is None or location.numel() == 0:
        return
    for dim, bound in enumerate(shape):
        values = location[..., dim]
        lower_invalid = check_lower_bound and values.min() < 0
        upper_invalid = values.max() >= bound
        if lower_invalid or upper_invalid:
            if check_lower_bound:
                raise RuntimeError(f"{name} dim {dim} must be in [0, {bound - 1}]")
            raise RuntimeError(f"{name} dim {dim} must be less than {bound}")


def _validate_dispersion_time_step(
    dispersion: DebyeDispersion | None,
    *,
    dt: float,
) -> None:
    if dispersion is None:
        return
    tau = torch.as_tensor(dispersion.tau)
    min_tau = float(tau.detach().amin().item())
    if dt >= min_tau:
        raise ValueError(
            f"Debye dispersion requires dt < min(tau), but got dt={dt} and min(tau)={min_tau}."
        )


def _normalize_component_3d(component: str, *, name: str) -> str:
    if not isinstance(component, str):
        raise TypeError(f"{name} must be a string, got {type(component).__name__}.")
    value = component.strip().lower()
    if value not in _COMPONENT_TO_INDEX_3D:
        raise ValueError(
            f"{name} must be one of 'ex', 'ey', or 'ez', got {component!r}."
        )
    return value


__all__ = [
    "_COMPONENT_TO_INDEX_3D",
    "_normalize_component_3d",
    "_validate_dispersion_time_step",
    "_validate_location_bounds",
    "_validate_optional_bool",
    "_validate_optional_positive_int",
    "_validate_positive_int",
    "_validate_tensor_arg",
]
