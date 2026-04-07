from collections.abc import Sequence

import torch

from ..callbacks import Callback
from ..cfl import cfl_condition
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_3d
from ..resampling import downsample_and_movedim, upsample
from ..utils import C0
from ..validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)
from .maxwell3d_cuda import maxwell3d_c_cuda
from .maxwell3d_python import maxwell3d_python
from .validation_internal import (
    _normalize_component_3d,
    _validate_dispersion_time_step,
    _validate_location_bounds,
    _validate_optional_bool,
    _validate_positive_int,
    _validate_tensor_arg,
)

class Maxwell3D(torch.nn.Module):
    """3D Maxwell equations solver using FDTD + CPML.

    This class is the 3D counterpart to `MaxwellTM`. It supports forward modeling
    and inversion through PyTorch autograd on `(epsilon, sigma)`.
    """

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
        Ex_0: torch.Tensor | None = None,
        Ey_0: torch.Tensor | None = None,
        Ez_0: torch.Tensor | None = None,
        Hx_0: torch.Tensor | None = None,
        Hy_0: torch.Tensor | None = None,
        Hz_0: torch.Tensor | None = None,
        m_hz_y: torch.Tensor | None = None,
        m_hy_z: torch.Tensor | None = None,
        m_hx_z: torch.Tensor | None = None,
        m_hz_x: torch.Tensor | None = None,
        m_hy_x: torch.Tensor | None = None,
        m_hx_y: torch.Tensor | None = None,
        m_ey_z: torch.Tensor | None = None,
        m_ez_y: torch.Tensor | None = None,
        m_ez_x: torch.Tensor | None = None,
        m_ex_z: torch.Tensor | None = None,
        m_ex_y: torch.Tensor | None = None,
        m_ey_x: torch.Tensor | None = None,
        nt: int | None = None,
        model_gradient_sampling_interval: int = 1,
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        save_snapshots: bool | None = None,
        forward_callback: Callback | None = None,
        backward_callback: Callback | None = None,
        callback_frequency: int = 1,
        source_component: str = "ey",
        receiver_component: str = "ey",
        execution_backend: str = "standard",
        python_backend: bool | str = False,
        storage_mode: str = "device",
        storage_path: str = ".",
        storage_compression: bool | str = False,
        storage_bytes_limit_device: int | None = None,
        storage_bytes_limit_host: int | None = None,
        storage_chunk_steps: int = 0,
        n_threads: int | None = None,
        dispersion: DebyeDispersion | None = None,
    ):
        assert isinstance(self.epsilon, torch.Tensor)
        assert isinstance(self.sigma, torch.Tensor)
        assert isinstance(self.mu, torch.Tensor)
        return maxwell3d(
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
            Ex_0,
            Ey_0,
            Ez_0,
            Hx_0,
            Hy_0,
            Hz_0,
            m_hz_y,
            m_hy_z,
            m_hx_z,
            m_hz_x,
            m_hy_x,
            m_hx_y,
            m_ey_z,
            m_ez_y,
            m_ez_x,
            m_ex_z,
            m_ex_y,
            m_ey_x,
            nt,
            model_gradient_sampling_interval,
            freq_taper_frac,
            time_pad_frac,
            time_taper,
            save_snapshots,
            forward_callback,
            backward_callback,
            callback_frequency,
            source_component,
            receiver_component,
            execution_backend,
            python_backend,
            storage_mode,
            storage_path,
            storage_compression,
            storage_bytes_limit_device,
            storage_bytes_limit_host,
            storage_chunk_steps,
            n_threads,
            dispersion=dispersion,
        )


def maxwell3d(
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
    Ex_0: torch.Tensor | None = None,
    Ey_0: torch.Tensor | None = None,
    Ez_0: torch.Tensor | None = None,
    Hx_0: torch.Tensor | None = None,
    Hy_0: torch.Tensor | None = None,
    Hz_0: torch.Tensor | None = None,
    m_hz_y: torch.Tensor | None = None,
    m_hy_z: torch.Tensor | None = None,
    m_hx_z: torch.Tensor | None = None,
    m_hz_x: torch.Tensor | None = None,
    m_hy_x: torch.Tensor | None = None,
    m_hx_y: torch.Tensor | None = None,
    m_ey_z: torch.Tensor | None = None,
    m_ez_y: torch.Tensor | None = None,
    m_ez_x: torch.Tensor | None = None,
    m_ex_z: torch.Tensor | None = None,
    m_ex_y: torch.Tensor | None = None,
    m_ey_x: torch.Tensor | None = None,
    nt: int | None = None,
    model_gradient_sampling_interval: int = 1,
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    save_snapshots: bool | None = None,
    forward_callback: Callback | None = None,
    backward_callback: Callback | None = None,
    callback_frequency: int = 1,
    source_component: str = "ey",
    receiver_component: str = "ey",
    execution_backend: str = "standard",
    python_backend: bool | str = False,
    storage_mode: str = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    storage_chunk_steps: int = 0,
    n_threads: int | None = None,
    dispersion: DebyeDispersion | None = None,
):
    """3D Maxwell equations solver.

    Coordinate convention is `[z, y, x]`.
    """
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)

    if epsilon.ndim != 3:
        raise RuntimeError("epsilon must be 3D")
    if sigma.shape != epsilon.shape:
        raise RuntimeError("sigma must have same shape as epsilon")
    if mu.shape != epsilon.shape:
        raise RuntimeError("mu must have same shape as epsilon")

    source_component = _normalize_component_3d(
        source_component, name="source_component"
    )
    receiver_component = _normalize_component_3d(
        receiver_component, name="receiver_component"
    )
    execution_backend = str(execution_backend).lower()
    if execution_backend != "standard":
        raise ValueError(
            "execution_backend must be 'standard', "
            f"but got {execution_backend!r}"
        )

    _validate_location_bounds(
        source_location,
        shape=(epsilon.shape[-3], epsilon.shape[-2], epsilon.shape[-1]),
        name="Source location",
        check_lower_bound=True,
    )
    _validate_location_bounds(
        receiver_location,
        shape=(epsilon.shape[-3], epsilon.shape[-2], epsilon.shape[-1]),
        name="Receiver location",
        check_lower_bound=True,
    )
    _validate_positive_int("callback_frequency", callback_frequency)

    _validate_dispersion_time_step(dispersion, dt=dt)

    grid_spacing_list = _normalize_grid_spacing_3d(grid_spacing)

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

    if isinstance(python_backend, bool):
        use_python = python_backend
    elif isinstance(python_backend, str):
        use_python = True
    else:
        raise TypeError(
            f"python_backend must be bool or str, but got {type(python_backend).__name__}"
        )

    result = (maxwell3d_python if use_python else maxwell3d_c_cuda)(
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
        Ex_0,
        Ey_0,
        Ez_0,
        Hx_0,
        Hy_0,
        Hz_0,
        m_hz_y,
        m_hy_z,
        m_hx_z,
        m_hz_x,
        m_hy_x,
        m_hx_y,
        m_ey_z,
        m_ez_y,
        m_ez_x,
        m_ex_z,
        m_ex_y,
        m_ey_x,
        nt_internal,
        model_gradient_sampling_interval,
        freq_taper_frac,
        time_pad_frac,
        time_taper,
        save_snapshots,
        forward_callback,
        backward_callback,
        callback_frequency,
        source_component,
        receiver_component,
        execution_backend,
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
        Ex_out,
        Ey_out,
        Ez_out,
        Hx_out,
        Hy_out,
        Hz_out,
        m_hz_y_out,
        m_hy_z_out,
        m_hx_z_out,
        m_hz_x_out,
        m_hy_x_out,
        m_hx_y_out,
        m_ey_z_out,
        m_ez_y_out,
        m_ez_x_out,
        m_ex_z_out,
        m_ex_y_out,
        m_ey_x_out,
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
        Ex_out,
        Ey_out,
        Ez_out,
        Hx_out,
        Hy_out,
        Hz_out,
        m_hz_y_out,
        m_hy_z_out,
        m_hx_z_out,
        m_hz_x_out,
        m_hy_x_out,
        m_hx_y_out,
        m_ey_z_out,
        m_ez_y_out,
        m_ez_x_out,
        m_ex_z_out,
        m_ex_y_out,
        m_ey_x_out,
        receiver_amplitudes,
    )

__all__ = ["Maxwell3D", "maxwell3d"]
