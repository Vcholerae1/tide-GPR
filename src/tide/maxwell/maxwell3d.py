from collections.abc import Callable, Sequence

import torch

from ..callbacks import Callback
from ..cfl import cfl_condition
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_3d
from ..resampling import downsample_and_movedim, upsample
from ..utils import C0, validate_material_inputs
from ..validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)
from .common import (
    _normalize_structured_batch,
    _reshape_structured_receiver_amplitudes,
    _reshape_structured_wavefield,
    _structured_vmap_shot_in_dim,
    _structured_vmap_state_in_dim,
    _wrap_structured_callback,
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

ReceiverMisfit = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _default_receiver_misfit(
    predicted: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    residual = predicted - observed
    return 0.5 * residual.square().sum()


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

    def hvp(
        self,
        dt: float,
        source_amplitude: torch.Tensor | None,
        source_location: torch.Tensor | None,
        receiver_location: torch.Tensor | None,
        observed_data: torch.Tensor,
        *,
        vepsilon: torch.Tensor | None = None,
        vsigma: torch.Tensor | None = None,
        misfit: ReceiverMisfit | None = None,
        stencil: int = 2,
        pml_width: int | Sequence[int] = 20,
        max_vel: float | None = None,
        nt: int | None = None,
        linearize_source: bool = True,
        source_component: str = "ey",
        receiver_component: str = "ey",
        python_backend: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_optional_bool("python_backend", python_backend)
        assert isinstance(self.epsilon, torch.Tensor)
        assert isinstance(self.sigma, torch.Tensor)
        assert isinstance(self.mu, torch.Tensor)
        return maxwell3d_hvp(
            self.epsilon,
            self.sigma,
            self.mu,
            grid_spacing=self.grid_spacing,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            observed_data=observed_data,
            vepsilon=vepsilon,
            vsigma=vsigma,
            misfit=misfit,
            stencil=stencil,
            pml_width=pml_width,
            max_vel=max_vel,
            nt=nt,
            linearize_source=linearize_source,
            source_component=source_component,
            receiver_component=receiver_component,
            python_backend=python_backend,
        )


def maxwell3d_hvp(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    observed_data: torch.Tensor,
    *,
    vepsilon: torch.Tensor | None = None,
    vsigma: torch.Tensor | None = None,
    misfit: ReceiverMisfit | None = None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    nt: int | None = None,
    linearize_source: bool = True,
    source_component: str = "ey",
    receiver_component: str = "ey",
    python_backend: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a receiver-space Hessian to a model direction.

    The current implementation is limited to the Python path
    (`python_backend=True`), which evaluates a receiver-space
    Hessian-vector product through the reference `maxwell3d` and `born3d`
    operators:

        Hv = grad_m <dPhi/dd, Jv>
    """
    _validate_optional_bool("python_backend", python_backend)
    if not python_backend:
        raise NotImplementedError("3D HVP currently requires python_backend=True.")

    _validate_tensor_arg("epsilon", epsilon)
    _validate_tensor_arg("sigma", sigma)
    _validate_tensor_arg("mu", mu)
    _validate_tensor_arg("observed_data", observed_data)
    if epsilon.ndim != 3:
        raise NotImplementedError("maxwell3d_hvp currently supports a single 3D model.")
    if sigma.shape != epsilon.shape or mu.shape != epsilon.shape:
        raise ValueError("sigma and mu must have the same shape as epsilon.")
    if vepsilon is not None:
        _validate_tensor_arg("vepsilon", vepsilon)
        if vepsilon.shape != epsilon.shape:
            raise ValueError("vepsilon must have the same shape as epsilon.")
    if vsigma is not None:
        _validate_tensor_arg("vsigma", vsigma)
        if vsigma.shape != sigma.shape:
            raise ValueError("vsigma must have the same shape as sigma.")

    misfit_fn = _default_receiver_misfit if misfit is None else misfit
    if not callable(misfit_fn):
        raise TypeError("misfit must be callable when provided.")

    from .maxwell3d_born_autograd import maxwell3d_receiver_hvp_naive

    return maxwell3d_receiver_hvp_naive(
        epsilon,
        sigma,
        mu,
        vepsilon=vepsilon,
        vsigma=vsigma,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=observed_data,
        misfit_fn=misfit_fn,
        stencil=stencil,
        pml_width=pml_width,
        max_vel=max_vel,
        nt=nt,
        linearize_source=linearize_source,
        source_component=source_component,
        receiver_component=receiver_component,
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
    epsilon_input = epsilon
    sigma_input = sigma
    mu_input = mu
    source_amplitude_input = source_amplitude
    source_location_input = source_location
    receiver_location_input = receiver_location
    Ex_0_input = Ex_0
    Ey_0_input = Ey_0
    Ez_0_input = Ez_0
    Hx_0_input = Hx_0
    Hy_0_input = Hy_0
    Hz_0_input = Hz_0
    m_hz_y_input = m_hz_y
    m_hy_z_input = m_hy_z
    m_hx_z_input = m_hx_z
    m_hz_x_input = m_hz_x
    m_hy_x_input = m_hy_x
    m_hx_y_input = m_hx_y
    m_ey_z_input = m_ey_z
    m_ez_y_input = m_ez_y
    m_ez_x_input = m_ez_x
    m_ex_z_input = m_ex_z
    m_ex_y_input = m_ex_y
    m_ey_x_input = m_ey_x

    batch_meta = _normalize_structured_batch(
        spatial_ndim=3,
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        shot_tensors={
            "source_amplitude": (source_amplitude, 2),
            "source_location": (source_location, 2),
            "receiver_location": (receiver_location, 2),
        },
        state_tensors={
            "Ex_0": Ex_0,
            "Ey_0": Ey_0,
            "Ez_0": Ez_0,
            "Hx_0": Hx_0,
            "Hy_0": Hy_0,
            "Hz_0": Hz_0,
            "m_hz_y": m_hz_y,
            "m_hy_z": m_hy_z,
            "m_hx_z": m_hx_z,
            "m_hz_x": m_hz_x,
            "m_hy_x": m_hy_x,
            "m_hx_y": m_hx_y,
            "m_ey_z": m_ey_z,
            "m_ez_y": m_ez_y,
            "m_ez_x": m_ez_x,
            "m_ex_z": m_ex_z,
            "m_ex_y": m_ex_y,
            "m_ey_x": m_ey_x,
        },
    )
    epsilon = batch_meta["epsilon"]
    sigma = batch_meta["sigma"]
    mu = batch_meta["mu"]
    source_amplitude = batch_meta["shot_tensors"]["source_amplitude"]
    source_location = batch_meta["shot_tensors"]["source_location"]
    receiver_location = batch_meta["shot_tensors"]["receiver_location"]
    Ex_0 = batch_meta["state_tensors"]["Ex_0"]
    Ey_0 = batch_meta["state_tensors"]["Ey_0"]
    Ez_0 = batch_meta["state_tensors"]["Ez_0"]
    Hx_0 = batch_meta["state_tensors"]["Hx_0"]
    Hy_0 = batch_meta["state_tensors"]["Hy_0"]
    Hz_0 = batch_meta["state_tensors"]["Hz_0"]
    m_hz_y = batch_meta["state_tensors"]["m_hz_y"]
    m_hy_z = batch_meta["state_tensors"]["m_hy_z"]
    m_hx_z = batch_meta["state_tensors"]["m_hx_z"]
    m_hz_x = batch_meta["state_tensors"]["m_hz_x"]
    m_hy_x = batch_meta["state_tensors"]["m_hy_x"]
    m_hx_y = batch_meta["state_tensors"]["m_hx_y"]
    m_ey_z = batch_meta["state_tensors"]["m_ey_z"]
    m_ez_y = batch_meta["state_tensors"]["m_ez_y"]
    m_ez_x = batch_meta["state_tensors"]["m_ez_x"]
    m_ex_z = batch_meta["state_tensors"]["m_ex_z"]
    m_ex_y = batch_meta["state_tensors"]["m_ex_y"]
    m_ey_x = batch_meta["state_tensors"]["m_ey_x"]

    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    freq_taper_frac = validate_freq_taper_frac(freq_taper_frac)
    time_pad_frac = validate_time_pad_frac(time_pad_frac)

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

    if batch_meta["model_batched"] and use_python:
        if forward_callback is not None or backward_callback is not None:
            raise NotImplementedError(
                "Batched models with python_backend do not support callbacks in v1."
            )
        validate_material_inputs(
            epsilon_input,
            dispersion=dispersion,
            dt=inner_dt,
        )

        source_amplitude_vmap = source_amplitude_input
        if (
            step_ratio > 1
            and source_amplitude_input is not None
            and source_amplitude_input.numel() > 0
        ):
            source_amplitude_vmap = upsample(
                source_amplitude_input,
                step_ratio,
                freq_taper_frac=freq_taper_frac,
                time_pad_frac=time_pad_frac,
                time_taper=time_taper,
            )

        B = int(batch_meta["B"])
        shot_in_dims = {
            "source_amplitude": _structured_vmap_shot_in_dim(
                "source_amplitude",
                source_amplitude_vmap,
                B=B,
                tail_ndim=2,
            ),
            "source_location": _structured_vmap_shot_in_dim(
                "source_location",
                source_location_input,
                B=B,
                tail_ndim=2,
            ),
            "receiver_location": _structured_vmap_shot_in_dim(
                "receiver_location",
                receiver_location_input,
                B=B,
                tail_ndim=2,
            ),
        }
        state_in_dims = {
            "Ex_0": _structured_vmap_state_in_dim(
                "Ex_0",
                Ex_0_input,
                B=B,
                spatial_ndim=3,
            ),
            "Ey_0": _structured_vmap_state_in_dim(
                "Ey_0",
                Ey_0_input,
                B=B,
                spatial_ndim=3,
            ),
            "Ez_0": _structured_vmap_state_in_dim(
                "Ez_0",
                Ez_0_input,
                B=B,
                spatial_ndim=3,
            ),
            "Hx_0": _structured_vmap_state_in_dim(
                "Hx_0",
                Hx_0_input,
                B=B,
                spatial_ndim=3,
            ),
            "Hy_0": _structured_vmap_state_in_dim(
                "Hy_0",
                Hy_0_input,
                B=B,
                spatial_ndim=3,
            ),
            "Hz_0": _structured_vmap_state_in_dim(
                "Hz_0",
                Hz_0_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_hz_y": _structured_vmap_state_in_dim(
                "m_hz_y",
                m_hz_y_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_hy_z": _structured_vmap_state_in_dim(
                "m_hy_z",
                m_hy_z_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_hx_z": _structured_vmap_state_in_dim(
                "m_hx_z",
                m_hx_z_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_hz_x": _structured_vmap_state_in_dim(
                "m_hz_x",
                m_hz_x_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_hy_x": _structured_vmap_state_in_dim(
                "m_hy_x",
                m_hy_x_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_hx_y": _structured_vmap_state_in_dim(
                "m_hx_y",
                m_hx_y_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_ey_z": _structured_vmap_state_in_dim(
                "m_ey_z",
                m_ey_z_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_ez_y": _structured_vmap_state_in_dim(
                "m_ez_y",
                m_ez_y_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_ez_x": _structured_vmap_state_in_dim(
                "m_ez_x",
                m_ez_x_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_ex_z": _structured_vmap_state_in_dim(
                "m_ex_z",
                m_ex_z_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_ex_y": _structured_vmap_state_in_dim(
                "m_ex_y",
                m_ex_y_input,
                B=B,
                spatial_ndim=3,
            ),
            "m_ey_x": _structured_vmap_state_in_dim(
                "m_ey_x",
                m_ey_x_input,
                B=B,
                spatial_ndim=3,
            ),
        }

        def _single_model_forward(
            epsilon_i: torch.Tensor,
            sigma_i: torch.Tensor,
            mu_i: torch.Tensor,
            source_amplitude_i: torch.Tensor | None,
            source_location_i: torch.Tensor | None,
            receiver_location_i: torch.Tensor | None,
            Ex_0_i: torch.Tensor | None,
            Ey_0_i: torch.Tensor | None,
            Ez_0_i: torch.Tensor | None,
            Hx_0_i: torch.Tensor | None,
            Hy_0_i: torch.Tensor | None,
            Hz_0_i: torch.Tensor | None,
            m_hz_y_i: torch.Tensor | None,
            m_hy_z_i: torch.Tensor | None,
            m_hx_z_i: torch.Tensor | None,
            m_hz_x_i: torch.Tensor | None,
            m_hy_x_i: torch.Tensor | None,
            m_hx_y_i: torch.Tensor | None,
            m_ey_z_i: torch.Tensor | None,
            m_ez_y_i: torch.Tensor | None,
            m_ez_x_i: torch.Tensor | None,
            m_ex_z_i: torch.Tensor | None,
            m_ex_y_i: torch.Tensor | None,
            m_ey_x_i: torch.Tensor | None,
        ):
            return maxwell3d_python(
                epsilon_i,
                sigma_i,
                mu_i,
                grid_spacing,
                inner_dt,
                source_amplitude_i,
                source_location_i,
                receiver_location_i,
                stencil,
                pml_width,
                max_vel_computed,
                Ex_0_i,
                Ey_0_i,
                Ez_0_i,
                Hx_0_i,
                Hy_0_i,
                Hz_0_i,
                m_hz_y_i,
                m_hy_z_i,
                m_hx_z_i,
                m_hz_x_i,
                m_hy_x_i,
                m_hx_y_i,
                m_ey_z_i,
                m_ez_y_i,
                m_ez_x_i,
                m_ex_z_i,
                m_ex_y_i,
                m_ey_x_i,
                nt_internal,
                model_gradient_sampling_interval,
                freq_taper_frac,
                time_pad_frac,
                time_taper,
                save_snapshots,
                None,
                None,
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
                validate_material_inputs=False,
            )

        result = torch.vmap(
            _single_model_forward,
            in_dims=(
                0,
                0,
                0,
                shot_in_dims["source_amplitude"],
                shot_in_dims["source_location"],
                shot_in_dims["receiver_location"],
                state_in_dims["Ex_0"],
                state_in_dims["Ey_0"],
                state_in_dims["Ez_0"],
                state_in_dims["Hx_0"],
                state_in_dims["Hy_0"],
                state_in_dims["Hz_0"],
                state_in_dims["m_hz_y"],
                state_in_dims["m_hy_z"],
                state_in_dims["m_hx_z"],
                state_in_dims["m_hz_x"],
                state_in_dims["m_hy_x"],
                state_in_dims["m_hx_y"],
                state_in_dims["m_ey_z"],
                state_in_dims["m_ez_y"],
                state_in_dims["m_ez_x"],
                state_in_dims["m_ex_z"],
                state_in_dims["m_ex_y"],
                state_in_dims["m_ey_x"],
            ),
        )(
            epsilon_input,
            sigma_input,
            mu_input,
            source_amplitude_vmap,
            source_location_input,
            receiver_location_input,
            Ex_0_input,
            Ey_0_input,
            Ez_0_input,
            Hx_0_input,
            Hy_0_input,
            Hz_0_input,
            m_hz_y_input,
            m_hy_z_input,
            m_hx_z_input,
            m_hz_x_input,
            m_hy_x_input,
            m_hx_y_input,
            m_ey_z_input,
            m_ez_y_input,
            m_ez_x_input,
            m_ex_z_input,
            m_ex_y_input,
            m_ey_x_input,
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

        if receiver_amplitudes.numel() > 0:
            receiver_amplitudes = torch.movedim(receiver_amplitudes, 1, 0)
            if step_ratio > 1:
                receiver_amplitudes = downsample_and_movedim(
                    receiver_amplitudes,
                    step_ratio,
                    freq_taper_frac=freq_taper_frac,
                    time_pad_frac=time_pad_frac,
                    time_taper=time_taper,
                )
                receiver_amplitudes = torch.movedim(receiver_amplitudes, -1, 0)

        if not batch_meta["structured_output"]:
            Ex_out = Ex_out.squeeze(0)
            Ey_out = Ey_out.squeeze(0)
            Ez_out = Ez_out.squeeze(0)
            Hx_out = Hx_out.squeeze(0)
            Hy_out = Hy_out.squeeze(0)
            Hz_out = Hz_out.squeeze(0)
            m_hz_y_out = m_hz_y_out.squeeze(0)
            m_hy_z_out = m_hy_z_out.squeeze(0)
            m_hx_z_out = m_hx_z_out.squeeze(0)
            m_hz_x_out = m_hz_x_out.squeeze(0)
            m_hy_x_out = m_hy_x_out.squeeze(0)
            m_hx_y_out = m_hx_y_out.squeeze(0)
            m_ey_z_out = m_ey_z_out.squeeze(0)
            m_ez_y_out = m_ez_y_out.squeeze(0)
            m_ez_x_out = m_ez_x_out.squeeze(0)
            m_ex_z_out = m_ex_z_out.squeeze(0)
            m_ex_y_out = m_ex_y_out.squeeze(0)
            m_ey_x_out = m_ey_x_out.squeeze(0)
            if receiver_amplitudes.ndim > 1:
                receiver_amplitudes = receiver_amplitudes.squeeze(1)

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

    forward_callback_wrapped = _wrap_structured_callback(
        forward_callback,
        batch_meta=batch_meta,
    )
    backward_callback_wrapped = _wrap_structured_callback(
        backward_callback,
        batch_meta=batch_meta,
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
        forward_callback_wrapped,
        backward_callback_wrapped,
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

    Ex_out = _reshape_structured_wavefield(Ex_out, batch_meta=batch_meta)
    Ey_out = _reshape_structured_wavefield(Ey_out, batch_meta=batch_meta)
    Ez_out = _reshape_structured_wavefield(Ez_out, batch_meta=batch_meta)
    Hx_out = _reshape_structured_wavefield(Hx_out, batch_meta=batch_meta)
    Hy_out = _reshape_structured_wavefield(Hy_out, batch_meta=batch_meta)
    Hz_out = _reshape_structured_wavefield(Hz_out, batch_meta=batch_meta)
    m_hz_y_out = _reshape_structured_wavefield(m_hz_y_out, batch_meta=batch_meta)
    m_hy_z_out = _reshape_structured_wavefield(m_hy_z_out, batch_meta=batch_meta)
    m_hx_z_out = _reshape_structured_wavefield(m_hx_z_out, batch_meta=batch_meta)
    m_hz_x_out = _reshape_structured_wavefield(m_hz_x_out, batch_meta=batch_meta)
    m_hy_x_out = _reshape_structured_wavefield(m_hy_x_out, batch_meta=batch_meta)
    m_hx_y_out = _reshape_structured_wavefield(m_hx_y_out, batch_meta=batch_meta)
    m_ey_z_out = _reshape_structured_wavefield(m_ey_z_out, batch_meta=batch_meta)
    m_ez_y_out = _reshape_structured_wavefield(m_ez_y_out, batch_meta=batch_meta)
    m_ez_x_out = _reshape_structured_wavefield(m_ez_x_out, batch_meta=batch_meta)
    m_ex_z_out = _reshape_structured_wavefield(m_ex_z_out, batch_meta=batch_meta)
    m_ex_y_out = _reshape_structured_wavefield(m_ex_y_out, batch_meta=batch_meta)
    m_ey_x_out = _reshape_structured_wavefield(m_ey_x_out, batch_meta=batch_meta)
    receiver_amplitudes = _reshape_structured_receiver_amplitudes(
        receiver_amplitudes,
        batch_meta=batch_meta,
    )

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

__all__ = ["Maxwell3D", "maxwell3d", "maxwell3d_hvp"]
