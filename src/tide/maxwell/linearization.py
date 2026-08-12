"""Explicit first- and second-order derivatives of Maxwell operators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

import torch

from ..core import StorageOptions
from .contracts import EMDirection, EMModel, SourceConvention
from .operators import Maxwell3D, MaxwellTM
from .results import (
    EM3DState,
    EMGradient,
    ForwardResult,
    TMState,
    TangentResult,
)

StateT = TypeVar("StateT", TMState, EM3DState)


def _differentiable_model(model: EMModel, targets: frozenset[str]) -> EMModel:
    def prepare(name: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().requires_grad_(name in targets)

    return EMModel(
        epsilon=prepare("epsilon", model.epsilon),
        sigma=prepare("sigma", model.sigma),
        mu=prepare("mu", model.mu),
        dispersion=model.dispersion,
    )


class _LinearizedMaxwell(Generic[StateT]):
    _state_size: int

    def __init__(
        self,
        operator: MaxwellTM | Maxwell3D,
        model: EMModel,
        *,
        storage: StorageOptions,
        targets: Iterable[str] = ("epsilon", "sigma"),
    ) -> None:
        target_set = frozenset(targets)
        unknown = target_set - {"epsilon", "sigma", "mu"}
        if unknown:
            raise ValueError(f"Unknown linearization targets: {sorted(unknown)!r}.")
        if not target_set:
            raise ValueError("At least one linearization target is required.")
        self.operator = (
            operator
            if storage == operator.storage
            else type(operator)(
                operator.discretization,
                operator.experiment,
                execution=operator.execution,
                storage=storage,
                model_gradient_sampling_interval=(
                    operator.model_gradient_sampling_interval
                ),
            )
        )
        self.model = _differentiable_model(model, target_set)
        self.storage = storage
        self.targets = target_set
        self._primal: ForwardResult[StateT] | None = None
        self._closed = False

    def _validate_open(self) -> None:
        if self._closed:
            raise RuntimeError("The linearized Maxwell operator is closed.")

    @property
    def primal(self) -> ForwardResult[StateT]:
        self._validate_open()
        if self._primal is None:
            self._primal = self.operator.forward(self.model)  # type: ignore[assignment]
        return self._primal

    def _parameters(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            getattr(self.model, name)
            for name in ("epsilon", "sigma", "mu")
            if name in self.targets
        )

    def _gradient(self, tensors: tuple[torch.Tensor | None, ...]) -> EMGradient:
        values: dict[str, torch.Tensor | None] = {
            "epsilon": None,
            "sigma": None,
            "mu": None,
        }
        for name, tensor in zip(
            (name for name in ("epsilon", "sigma", "mu") if name in self.targets),
            tensors,
            strict=True,
        ):
            values[name] = tensor
        return EMGradient(**values)

    def vjp(self, cotangent: torch.Tensor) -> EMGradient:
        """Apply the adjoint Jacobian ``J(model).T`` to receiver cotangents."""
        receiver_data = self.primal.receiver_data
        if cotangent.shape != receiver_data.shape:
            raise ValueError("Receiver cotangent must match primal receiver data shape.")
        gradients = torch.autograd.grad(
            receiver_data,
            self._parameters(),
            cotangent,
            retain_graph=True,
            allow_unused=True,
        )
        return self._gradient(gradients)

    def close(self) -> None:
        if self._closed:
            return
        self._primal = None
        self._closed = True

    def __enter__(self):
        self._validate_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class LinearizedMaxwellTM(_LinearizedMaxwell[TMState]):
    """Derivative of a :class:`MaxwellTM` operator at one background model."""

    _state_size = 7

    def __init__(
        self,
        operator: MaxwellTM,
        model: EMModel,
        *,
        storage: StorageOptions,
        targets: Iterable[str] = ("epsilon", "sigma"),
    ) -> None:
        super().__init__(operator, model, storage=storage, targets=targets)
        self.operator = operator

    def _run_jvp(self, direction: EMDirection) -> tuple[TangentResult[TMState], torch.Tensor]:
        self._validate_open()
        direction.validate_for(self.model)
        if direction.mu is not None:
            raise NotImplementedError("TM2D tangent propagation for mu is not available.")
        from .derivatives import tm2d_jvp

        discretization = self.operator.discretization
        experiment = self.operator.experiment
        execution = self.operator.execution
        result = tm2d_jvp(
            self.model.epsilon,
            self.model.sigma,
            self.model.mu,
            grid_spacing=discretization.spacing,
            dt=discretization.dt,
            source_amplitude=experiment.source_amplitude,
            source_location=experiment.acquisition.source_location,
            receiver_location=experiment.acquisition.receiver_location,
            bg_receiver_location=experiment.acquisition.receiver_location,
            depsilon=direction.epsilon,
            dsigma=direction.sigma,
            stencil=discretization.stencil,
            pml_width=discretization.boundary.width,
            max_vel=discretization.max_velocity,
            nt=experiment.nt,
            model_gradient_sampling_interval=self.operator.model_gradient_sampling_interval,
            linearize_source=(
                experiment.source_convention is SourceConvention.PHYSICAL_CURRENT
            ),
            freq_taper_frac=experiment.frequency_taper_fraction,
            time_pad_frac=experiment.time_padding_fraction,
            time_taper=experiment.time_taper,
            python_backend=execution.legacy_backend_request,
            storage_mode=self.storage.mode.value,
            storage_path=self.storage.path,
            storage_compression=self.storage.compression,
            storage_bytes_limit_device=self.storage.bytes_limit_device,
            storage_bytes_limit_host=self.storage.bytes_limit_host,
            n_threads=execution.n_threads,
            fallback=execution.fallback.value,
        )
        background_state = TMState.from_tensors(tuple(result[: self._state_size]))
        tangent_state = TMState.from_tensors(
            tuple(result[self._state_size : 2 * self._state_size])
        )
        self._primal = ForwardResult(result[-2], background_state)
        return TangentResult(result[-1], tangent_state), result[-1]

    def jvp(self, direction: EMDirection) -> TangentResult[TMState]:
        tangent, _ = self._run_jvp(direction)
        return tangent

    def second_vjp(
        self,
        direction: EMDirection,
        cotangent: torch.Tensor,
    ) -> EMGradient:
        tangent, tangent_receiver = self._run_jvp(direction)
        if cotangent.shape != tangent.receiver_data.shape:
            raise ValueError("Receiver cotangent must match tangent receiver data shape.")
        gradients = torch.autograd.grad(
            tangent_receiver,
            self._parameters(),
            cotangent,
            retain_graph=True,
            allow_unused=True,
        )
        return self._gradient(gradients)


class LinearizedMaxwell3D(_LinearizedMaxwell[EM3DState]):
    """Derivative of a :class:`Maxwell3D` operator at one background model."""

    _state_size = 18

    def __init__(
        self,
        operator: Maxwell3D,
        model: EMModel,
        *,
        storage: StorageOptions,
        targets: Iterable[str] = ("epsilon", "sigma"),
    ) -> None:
        super().__init__(operator, model, storage=storage, targets=targets)
        self.operator = operator

    def _run_jvp(self, direction: EMDirection) -> tuple[TangentResult[EM3DState], torch.Tensor]:
        self._validate_open()
        direction.validate_for(self.model)
        if direction.mu is not None:
            raise NotImplementedError("3-D tangent propagation for mu is not available.")
        from .derivatives import em3d_jvp

        discretization = self.operator.discretization
        experiment = self.operator.experiment
        execution = self.operator.execution
        result = em3d_jvp(
            self.model.epsilon,
            self.model.sigma,
            self.model.mu,
            grid_spacing=discretization.spacing,
            dt=discretization.dt,
            source_amplitude=experiment.source_amplitude,
            source_location=experiment.acquisition.source_location,
            receiver_location=experiment.acquisition.receiver_location,
            bg_receiver_location=experiment.acquisition.receiver_location,
            depsilon=direction.epsilon,
            dsigma=direction.sigma,
            stencil=discretization.stencil,
            pml_width=discretization.boundary.width,
            max_vel=discretization.max_velocity,
            nt=experiment.nt,
            model_gradient_sampling_interval=self.operator.model_gradient_sampling_interval,
            linearize_source=(
                experiment.source_convention is SourceConvention.PHYSICAL_CURRENT
            ),
            source_component=experiment.source_component,
            receiver_component=experiment.receiver_component,
            freq_taper_frac=experiment.frequency_taper_fraction,
            time_pad_frac=experiment.time_padding_fraction,
            time_taper=experiment.time_taper,
            python_backend=execution.legacy_backend_request,
            storage_mode=self.storage.mode.value,
            storage_path=self.storage.path,
            storage_compression=self.storage.compression,
            storage_bytes_limit_device=self.storage.bytes_limit_device,
            storage_bytes_limit_host=self.storage.bytes_limit_host,
            n_threads=execution.n_threads,
            fallback=execution.fallback.value,
        )
        background_state = EM3DState.from_tensors(tuple(result[: self._state_size]))
        tangent_state = EM3DState.from_tensors(
            tuple(result[self._state_size : 2 * self._state_size])
        )
        self._primal = ForwardResult(result[-2], background_state)
        return TangentResult(result[-1], tangent_state), result[-1]

    def jvp(self, direction: EMDirection) -> TangentResult[EM3DState]:
        tangent, _ = self._run_jvp(direction)
        return tangent

    def second_vjp(
        self,
        direction: EMDirection,
        cotangent: torch.Tensor,
    ) -> EMGradient:
        tangent, tangent_receiver = self._run_jvp(direction)
        if cotangent.shape != tangent.receiver_data.shape:
            raise ValueError("Receiver cotangent must match tangent receiver data shape.")
        gradients = torch.autograd.grad(
            tangent_receiver,
            self._parameters(),
            cotangent,
            retain_graph=True,
            allow_unused=True,
        )
        return self._gradient(gradients)


__all__ = ["LinearizedMaxwell3D", "LinearizedMaxwellTM"]
