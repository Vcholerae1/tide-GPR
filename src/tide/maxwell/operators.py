"""Nonlinear Maxwell operators with structured inputs and outputs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from ..core import StorageOptions
from .contracts import (
    Discretization,
    EMModel,
    ExecutionOptions,
    Experiment,
    Observers,
)
from .results import EM3DState, ForwardResult, TMState

if TYPE_CHECKING:
    from .linearization import LinearizedMaxwell3D, LinearizedMaxwellTM


def _pml_width(discretization: Discretization) -> int | Sequence[int]:
    return discretization.boundary.width


class MaxwellTM:
    """Differentiable 2-D transverse-magnetic Maxwell operator."""

    def __init__(
        self,
        discretization: Discretization,
        experiment: Experiment,
        *,
        execution: ExecutionOptions = ExecutionOptions(),
        storage: StorageOptions = StorageOptions(),
        model_gradient_sampling_interval: int = 1,
    ) -> None:
        spatial_ndim = experiment.acquisition.spatial_ndim
        if spatial_ndim not in {None, 2}:
            raise ValueError(
                "MaxwellTM requires two-dimensional acquisition coordinates."
            )
        self.discretization = discretization
        self.experiment = experiment
        self.execution = execution
        self.storage = storage
        self.model_gradient_sampling_interval = model_gradient_sampling_interval

    def forward(
        self,
        model: EMModel,
        *,
        state: TMState | None = None,
        observers: Observers = Observers(),
    ) -> ForwardResult[TMState]:
        if model.epsilon.ndim not in {2, 3}:
            raise ValueError(
                "MaxwellTM model tensors must be 2-D or batched 3-D tensors."
            )
        from .tm2d import maxwelltm

        initial = state or (None,) * 7
        if state is not None:
            initial = (
                state.Ey,
                state.Hx,
                state.Hz,
                state.m_Ey_x,
                state.m_Ey_z,
                state.m_Hx_z,
                state.m_Hz_x,
            )
        result = maxwelltm(
            model.epsilon,
            model.sigma,
            model.mu,
            grid_spacing=self.discretization.spacing,
            dt=self.discretization.dt,
            source_amplitude=self.experiment.source_amplitude,
            source_location=self.experiment.acquisition.source_location,
            receiver_location=self.experiment.acquisition.receiver_location,
            stencil=self.discretization.stencil,
            pml_width=_pml_width(self.discretization),
            max_vel=self.discretization.max_velocity,
            Ey_0=initial[0],
            Hx_0=initial[1],
            Hz_0=initial[2],
            m_Ey_x=initial[3],
            m_Ey_z=initial[4],
            m_Hx_z=initial[5],
            m_Hz_x=initial[6],
            nt=self.experiment.nt,
            model_gradient_sampling_interval=self.model_gradient_sampling_interval,
            freq_taper_frac=self.experiment.frequency_taper_fraction,
            time_pad_frac=self.experiment.time_padding_fraction,
            time_taper=self.experiment.time_taper,
            forward_callback=observers.forward,
            backward_callback=observers.backward,
            callback_frequency=observers.frequency,
            python_backend=self.execution.legacy_backend_request,
            storage_mode=self.storage.mode.value,
            storage_path=self.storage.path,
            storage_compression=self.storage.compression,
            storage_bytes_limit_device=self.storage.bytes_limit_device,
            storage_bytes_limit_host=self.storage.bytes_limit_host,
            storage_chunk_steps=self.storage.chunk_steps,
            n_threads=self.execution.n_threads,
            dispersion=model.dispersion,
            fallback=self.execution.fallback.value,
        )
        return ForwardResult(
            receiver_data=result[-1],
            final_state=TMState.from_tensors(tuple(result[:-1])),
        )

    __call__ = forward

    def linearize(
        self,
        model: EMModel,
        *,
        storage: StorageOptions | None = None,
        targets: Iterable[str] = ("epsilon", "sigma"),
    ) -> LinearizedMaxwellTM:
        from .linearization import LinearizedMaxwellTM

        return LinearizedMaxwellTM(
            self, model, storage=storage or self.storage, targets=targets
        )


class Maxwell3D:
    """Differentiable full 3-D Maxwell operator."""

    def __init__(
        self,
        discretization: Discretization,
        experiment: Experiment,
        *,
        execution: ExecutionOptions = ExecutionOptions(),
        storage: StorageOptions = StorageOptions(),
        model_gradient_sampling_interval: int = 1,
    ) -> None:
        spatial_ndim = experiment.acquisition.spatial_ndim
        if spatial_ndim not in {None, 3}:
            raise ValueError(
                "Maxwell3D requires three-dimensional acquisition coordinates."
            )
        self.discretization = discretization
        self.experiment = experiment
        self.execution = execution
        self.storage = storage
        self.model_gradient_sampling_interval = model_gradient_sampling_interval

    def forward(
        self,
        model: EMModel,
        *,
        state: EM3DState | None = None,
        observers: Observers = Observers(),
    ) -> ForwardResult[EM3DState]:
        if model.epsilon.ndim not in {3, 4}:
            raise ValueError(
                "Maxwell3D model tensors must be 3-D or batched 4-D tensors."
            )
        from .maxwell3d import maxwell3d

        initial: tuple[object, ...] = (None,) * 18
        if state is not None:
            initial = (
                state.Ex,
                state.Ey,
                state.Ez,
                state.Hx,
                state.Hy,
                state.Hz,
                state.m_hz_y,
                state.m_hy_z,
                state.m_hx_z,
                state.m_hz_x,
                state.m_hy_x,
                state.m_hx_y,
                state.m_ey_z,
                state.m_ez_y,
                state.m_ez_x,
                state.m_ex_z,
                state.m_ex_y,
                state.m_ey_x,
            )
        result = maxwell3d(
            model.epsilon,
            model.sigma,
            model.mu,
            grid_spacing=self.discretization.spacing,
            dt=self.discretization.dt,
            source_amplitude=self.experiment.source_amplitude,
            source_location=self.experiment.acquisition.source_location,
            receiver_location=self.experiment.acquisition.receiver_location,
            stencil=self.discretization.stencil,
            pml_width=_pml_width(self.discretization),
            max_vel=self.discretization.max_velocity,
            Ex_0=initial[0],
            Ey_0=initial[1],
            Ez_0=initial[2],
            Hx_0=initial[3],
            Hy_0=initial[4],
            Hz_0=initial[5],
            m_hz_y=initial[6],
            m_hy_z=initial[7],
            m_hx_z=initial[8],
            m_hz_x=initial[9],
            m_hy_x=initial[10],
            m_hx_y=initial[11],
            m_ey_z=initial[12],
            m_ez_y=initial[13],
            m_ez_x=initial[14],
            m_ex_z=initial[15],
            m_ex_y=initial[16],
            m_ey_x=initial[17],
            nt=self.experiment.nt,
            model_gradient_sampling_interval=self.model_gradient_sampling_interval,
            freq_taper_frac=self.experiment.frequency_taper_fraction,
            time_pad_frac=self.experiment.time_padding_fraction,
            time_taper=self.experiment.time_taper,
            forward_callback=observers.forward,
            backward_callback=observers.backward,
            callback_frequency=observers.frequency,
            source_component=self.experiment.source_component,
            receiver_component=self.experiment.receiver_component,
            python_backend=self.execution.legacy_backend_request,
            storage_mode=self.storage.mode.value,
            storage_path=self.storage.path,
            storage_compression=self.storage.compression,
            storage_bytes_limit_device=self.storage.bytes_limit_device,
            storage_bytes_limit_host=self.storage.bytes_limit_host,
            storage_chunk_steps=self.storage.chunk_steps,
            n_threads=self.execution.n_threads,
            dispersion=model.dispersion,
            fallback=self.execution.fallback.value,
        )
        return ForwardResult(
            receiver_data=result[-1],
            final_state=EM3DState.from_tensors(tuple(result[:-1])),
        )

    __call__ = forward

    def linearize(
        self,
        model: EMModel,
        *,
        storage: StorageOptions | None = None,
        targets: Iterable[str] = ("epsilon", "sigma"),
    ) -> LinearizedMaxwell3D:
        from .linearization import LinearizedMaxwell3D

        return LinearizedMaxwell3D(
            self, model, storage=storage or self.storage, targets=targets
        )


__all__ = ["Maxwell3D", "MaxwellTM"]
