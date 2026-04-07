from collections.abc import Sequence
from typing import Literal

import torch

from ..callbacks import Callback
from ..cfl import cfl_condition
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_2d
from ..resampling import downsample_and_movedim, upsample
from ..utils import C0
from ..validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)
from .tm2d_python import maxwell_func
from .validation_internal import (
    _validate_dispersion_time_step,
    _validate_location_bounds,
    _validate_optional_bool,
    _validate_positive_int,
    _validate_tensor_arg,
)


class MaxwellTM(torch.nn.Module):
    """2D TM mode Maxwell equations solver using FDTD method."""

    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
        grid_spacing: float | Sequence[float],
        epsilon_requires_grad: bool | None = None,
        sigma_requires_grad: bool | None = None,
    ) -> None:
        super().__init__()
        _validate_optional_bool("epsilon_requires_grad", epsilon_requires_grad)
        _validate_optional_bool("sigma_requires_grad", sigma_requires_grad)
        _validate_tensor_arg("epsilon", epsilon)
        _validate_tensor_arg("sigma", sigma)
        _validate_tensor_arg("mu", mu)

        if epsilon_requires_grad is None:
            epsilon_requires_grad = epsilon.requires_grad
        if sigma_requires_grad is None:
            sigma_requires_grad = sigma.requires_grad

        self.epsilon = torch.nn.Parameter(epsilon, requires_grad=epsilon_requires_grad)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=sigma_requires_grad)
        self.register_buffer("mu", mu)
        self.grid_spacing = grid_spacing

    def forward(
        self,
        dt: float,
        source_amplitude: torch.Tensor | None,
        source_location: torch.Tensor | None,
        receiver_location: torch.Tensor | None,
        stencil: int = 2,
        pml_width: int | Sequence[int] = 20,
        max_vel: float | None = None,
        Ey_0: torch.Tensor | None = None,
        Hx_0: torch.Tensor | None = None,
        Hz_0: torch.Tensor | None = None,
        m_Ey_x: torch.Tensor | None = None,
        m_Ey_z: torch.Tensor | None = None,
        m_Hx_z: torch.Tensor | None = None,
        m_Hz_x: torch.Tensor | None = None,
        nt: int | None = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        save_snapshots: bool | None = None,
        forward_callback: Callback | None = None,
        backward_callback: Callback | None = None,
        callback_frequency: int = 1,
        python_backend: Literal["eager", "jit", "compile"] | bool = False,
        storage_mode: Literal["device", "cpu", "disk", "none", "auto"] = "device",
        storage_path: str = ".",
        storage_compression: bool | str = False,
        storage_bytes_limit_device: int | None = None,
        storage_bytes_limit_host: int | None = None,
        storage_chunk_steps: int = 0,
        dispersion: DebyeDispersion | None = None,
    ):
        assert isinstance(self.epsilon, torch.Tensor)
        assert isinstance(self.sigma, torch.Tensor)
        assert isinstance(self.mu, torch.Tensor)
        return maxwelltm(
            self.epsilon,
            self.sigma,
            self.mu,
            self.grid_spacing,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            stencil,
            pml_width,
            max_vel,
            Ey_0,
            Hx_0,
            Hz_0,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            nt,
            model_gradient_sampling_interval,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            save_snapshots,
            forward_callback,
            backward_callback,
            callback_frequency,
            python_backend,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            n_threads=None,
            dispersion=dispersion,
        )


def maxwelltm(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    Ey_0: torch.Tensor | None = None,
    Hx_0: torch.Tensor | None = None,
    Hz_0: torch.Tensor | None = None,
    m_Ey_x: torch.Tensor | None = None,
    m_Ey_z: torch.Tensor | None = None,
    m_Hx_z: torch.Tensor | None = None,
    m_Hz_x: torch.Tensor | None = None,
    nt: int | None = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    save_snapshots: bool | None = None,
    forward_callback: Callback | None = None,
    backward_callback: Callback | None = None,
    callback_frequency: int = 1,
    python_backend: Literal["eager", "jit", "compile"] | bool = False,
    storage_mode: Literal["device", "cpu", "disk", "none", "auto"] = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    storage_chunk_steps: int = 0,
    n_threads: int | None = None,
    dispersion: DebyeDispersion | None = None,
):
    """2D TM mode Maxwell equations solver."""
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)

    _validate_location_bounds(
        source_location,
        shape=(epsilon.shape[-2], epsilon.shape[-1]),
        name="Source location",
        check_lower_bound=False,
    )
    _validate_location_bounds(
        receiver_location,
        shape=(epsilon.shape[-2], epsilon.shape[-1]),
        name="Receiver location",
        check_lower_bound=False,
    )
    _validate_positive_int("callback_frequency", callback_frequency)
    _validate_dispersion_time_step(dispersion, dt=dt)

    grid_spacing_list = _normalize_grid_spacing_2d(grid_spacing)

    if max_vel is None:
        max_vel_computed = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    else:
        max_vel_computed = max_vel

    inner_dt, step_ratio = cfl_condition(grid_spacing_list, dt, max_vel_computed)

    source_amplitude_internal = source_amplitude
    if step_ratio > 1 and source_amplitude is not None and source_amplitude.numel() > 0:
        source_amplitude_internal = upsample(
            source_amplitude,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )

    nt_internal = None
    if nt is not None:
        nt_internal = nt * step_ratio
    elif source_amplitude_internal is not None:
        nt_internal = source_amplitude_internal.shape[-1]

    result = maxwell_func(
        python_backend,
        epsilon,
        sigma,
        mu,
        grid_spacing,
        inner_dt,
        source_amplitude_internal,
        source_location,
        receiver_location,
        stencil,
        pml_width,
        max_vel_computed,
        Ey_0,
        Hx_0,
        Hz_0,
        m_Ey_x,
        m_Ey_z,
        m_Hx_z,
        m_Hz_x,
        nt_internal,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        forward_callback,
        backward_callback,
        callback_frequency,
        storage_mode,
        storage_path,
        storage_compression,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        n_threads,
        dispersion,
    )

    (
        Ey_out,
        Hx_out,
        Hz_out,
        m_Ey_x_out,
        m_Ey_z_out,
        m_Hx_z_out,
        m_Hz_x_out,
        receiver_amplitudes,
    ) = result

    if step_ratio > 1 and receiver_amplitudes.numel() > 0:
        receiver_amplitudes = downsample_and_movedim(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
        receiver_amplitudes = torch.movedim(receiver_amplitudes, -1, 0)

    return (
        Ey_out,
        Hx_out,
        Hz_out,
        m_Ey_x_out,
        m_Ey_z_out,
        m_Hx_z_out,
        m_Hz_x_out,
        receiver_amplitudes,
    )


__all__ = ["MaxwellTM", "maxwelltm"]
