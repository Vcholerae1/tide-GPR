import warnings
from collections.abc import Sequence
from typing import Literal

import torch

from ..cfl import cfl_condition
from ..resampling import downsample_and_movedim, upsample
from ..typing import (
    Field3DLike,
    Model3D,
    ReceiverLocation3D,
    SourceLocation3D,
    WaveletBatch,
    runtime_typecheck,
)
from ..utils import C0
from ..validation import validate_model_gradient_sampling_interval
from .maxwell3d_born_cuda import born3d_c_cuda
from .maxwell3d_born_python import born3d_python
from .validation_internal import (
    _normalize_component_3d,
    _validate_optional_bool,
    _validate_tensor_arg,
)


def _register_optional_born_parameter(
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
    if requires_grad is None:
        requires_grad = value.requires_grad
    module.register_parameter(name, torch.nn.Parameter(value, requires_grad=requires_grad))


class Born3D(torch.nn.Module):
    """Module wrapper around :func:`born3d`.

    This stores the background 3D Maxwell model and an optional Born
    perturbation inside a reusable ``torch.nn.Module`` so training and
    inversion code can follow the same module-first workflow as Deepwave's
    ``ScalarBorn``.
    """

    @runtime_typecheck
    def __init__(
        self,
        epsilon: Model3D,
        sigma: Model3D,
        mu: Model3D,
        grid_spacing: float | Sequence[float],
        *,
        depsilon: Model3D | None = None,
        dsigma: Model3D | None = None,
        dca: Model3D | None = None,
        dcb: Model3D | None = None,
        epsilon_requires_grad: bool | None = None,
        sigma_requires_grad: bool | None = None,
        depsilon_requires_grad: bool | None = None,
        dsigma_requires_grad: bool | None = None,
        dca_requires_grad: bool | None = None,
        dcb_requires_grad: bool | None = None,
        parameterization: Literal["epsilon_sigma", "ca_cb"] = "epsilon_sigma",
        linearize_source: bool = True,
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
        if parameterization not in {"epsilon_sigma", "ca_cb"}:
            raise ValueError(
                "parameterization must be 'epsilon_sigma' or 'ca_cb', "
                f"got {parameterization!r}."
            )

        self.epsilon = torch.nn.Parameter(epsilon, requires_grad=epsilon_requires_grad)
        self.sigma = torch.nn.Parameter(sigma, requires_grad=sigma_requires_grad)
        self.register_buffer("mu", mu)
        _register_optional_born_parameter(
            self, "depsilon", depsilon, depsilon_requires_grad
        )
        _register_optional_born_parameter(self, "dsigma", dsigma, dsigma_requires_grad)
        _register_optional_born_parameter(self, "dca", dca, dca_requires_grad)
        _register_optional_born_parameter(self, "dcb", dcb, dcb_requires_grad)
        self.grid_spacing = grid_spacing
        self.parameterization = parameterization
        self.linearize_source = linearize_source

    @runtime_typecheck
    def forward(
        self,
        dt: float,
        source_amplitude: WaveletBatch | None = None,
        source_location: SourceLocation3D | None = None,
        receiver_location: ReceiverLocation3D | None = None,
        bg_receiver_location: ReceiverLocation3D | None = None,
        *,
        depsilon: Model3D | None = None,
        dsigma: Model3D | None = None,
        dca: Model3D | None = None,
        dcb: Model3D | None = None,
        stencil: int = 2,
        pml_width: int | Sequence[int] = 20,
        max_vel: float | None = None,
        Ex_0: Field3DLike | None = None,
        Ey_0: Field3DLike | None = None,
        Ez_0: Field3DLike | None = None,
        Hx_0: Field3DLike | None = None,
        Hy_0: Field3DLike | None = None,
        Hz_0: Field3DLike | None = None,
        m_hz_y_0: Field3DLike | None = None,
        m_hy_z_0: Field3DLike | None = None,
        m_hx_z_0: Field3DLike | None = None,
        m_hz_x_0: Field3DLike | None = None,
        m_hy_x_0: Field3DLike | None = None,
        m_hx_y_0: Field3DLike | None = None,
        m_ey_z_0: Field3DLike | None = None,
        m_ez_y_0: Field3DLike | None = None,
        m_ez_x_0: Field3DLike | None = None,
        m_ex_z_0: Field3DLike | None = None,
        m_ex_y_0: Field3DLike | None = None,
        m_ey_x_0: Field3DLike | None = None,
        dEx_0: Field3DLike | None = None,
        dEy_0: Field3DLike | None = None,
        dEz_0: Field3DLike | None = None,
        dHx_0: Field3DLike | None = None,
        dHy_0: Field3DLike | None = None,
        dHz_0: Field3DLike | None = None,
        dm_hz_y_0: Field3DLike | None = None,
        dm_hy_z_0: Field3DLike | None = None,
        dm_hx_z_0: Field3DLike | None = None,
        dm_hz_x_0: Field3DLike | None = None,
        dm_hy_x_0: Field3DLike | None = None,
        dm_hx_y_0: Field3DLike | None = None,
        dm_ey_z_0: Field3DLike | None = None,
        dm_ez_y_0: Field3DLike | None = None,
        dm_ez_x_0: Field3DLike | None = None,
        dm_ex_z_0: Field3DLike | None = None,
        dm_ex_y_0: Field3DLike | None = None,
        dm_ey_x_0: Field3DLike | None = None,
        nt: int | None = None,
        model_gradient_sampling_interval: int = 1,
        linearize_source: bool | None = None,
        source_component: str = "ey",
        receiver_component: str = "ey",
        freq_taper_frac: float = 0.0,
        time_pad_frac: float = 0.0,
        time_taper: bool = False,
        python_backend: Literal["eager", "jit", "compile"] | bool = False,
        storage_mode: Literal["device", "cpu", "disk", "none", "auto"] = "device",
        storage_path: str = ".",
        storage_compression: bool | str = False,
        storage_bytes_limit_device: int | None = None,
        storage_bytes_limit_host: int | None = None,
        n_threads: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if linearize_source is None:
            linearize_source = self.linearize_source
        assert isinstance(self.epsilon, torch.Tensor)
        assert isinstance(self.sigma, torch.Tensor)
        assert isinstance(self.mu, torch.Tensor)
        return born3d(
            self.epsilon,
            self.sigma,
            self.mu,
            grid_spacing=self.grid_spacing,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            bg_receiver_location=bg_receiver_location,
            depsilon=self.depsilon if depsilon is None else depsilon,
            dsigma=self.dsigma if dsigma is None else dsigma,
            dca=self.dca if dca is None else dca,
            dcb=self.dcb if dcb is None else dcb,
            stencil=stencil,
            pml_width=pml_width,
            max_vel=max_vel,
            Ex_0=Ex_0,
            Ey_0=Ey_0,
            Ez_0=Ez_0,
            Hx_0=Hx_0,
            Hy_0=Hy_0,
            Hz_0=Hz_0,
            m_hz_y_0=m_hz_y_0,
            m_hy_z_0=m_hy_z_0,
            m_hx_z_0=m_hx_z_0,
            m_hz_x_0=m_hz_x_0,
            m_hy_x_0=m_hy_x_0,
            m_hx_y_0=m_hx_y_0,
            m_ey_z_0=m_ey_z_0,
            m_ez_y_0=m_ez_y_0,
            m_ez_x_0=m_ez_x_0,
            m_ex_z_0=m_ex_z_0,
            m_ex_y_0=m_ex_y_0,
            m_ey_x_0=m_ey_x_0,
            dEx_0=dEx_0,
            dEy_0=dEy_0,
            dEz_0=dEz_0,
            dHx_0=dHx_0,
            dHy_0=dHy_0,
            dHz_0=dHz_0,
            dm_hz_y_0=dm_hz_y_0,
            dm_hy_z_0=dm_hy_z_0,
            dm_hx_z_0=dm_hx_z_0,
            dm_hz_x_0=dm_hz_x_0,
            dm_hy_x_0=dm_hy_x_0,
            dm_hx_y_0=dm_hx_y_0,
            dm_ey_z_0=dm_ey_z_0,
            dm_ez_y_0=dm_ez_y_0,
            dm_ez_x_0=dm_ez_x_0,
            dm_ex_z_0=dm_ex_z_0,
            dm_ex_y_0=dm_ex_y_0,
            dm_ey_x_0=dm_ey_x_0,
            nt=nt,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            parameterization=self.parameterization,
            linearize_source=linearize_source,
            source_component=source_component,
            receiver_component=receiver_component,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
            python_backend=python_backend,
            storage_mode=storage_mode,
            storage_path=storage_path,
            storage_compression=storage_compression,
            storage_bytes_limit_device=storage_bytes_limit_device,
            storage_bytes_limit_host=storage_bytes_limit_host,
            n_threads=n_threads,
        )


@runtime_typecheck
def born3d(
    epsilon: Model3D,
    sigma: Model3D,
    mu: Model3D,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: WaveletBatch | None,
    source_location: SourceLocation3D | None,
    receiver_location: ReceiverLocation3D | None,
    bg_receiver_location: ReceiverLocation3D | None = None,
    *,
    depsilon: Model3D | None = None,
    dsigma: Model3D | None = None,
    dca: Model3D | None = None,
    dcb: Model3D | None = None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    Ex_0: Field3DLike | None = None,
    Ey_0: Field3DLike | None = None,
    Ez_0: Field3DLike | None = None,
    Hx_0: Field3DLike | None = None,
    Hy_0: Field3DLike | None = None,
    Hz_0: Field3DLike | None = None,
    m_hz_y_0: Field3DLike | None = None,
    m_hy_z_0: Field3DLike | None = None,
    m_hx_z_0: Field3DLike | None = None,
    m_hz_x_0: Field3DLike | None = None,
    m_hy_x_0: Field3DLike | None = None,
    m_hx_y_0: Field3DLike | None = None,
    m_ey_z_0: Field3DLike | None = None,
    m_ez_y_0: Field3DLike | None = None,
    m_ez_x_0: Field3DLike | None = None,
    m_ex_z_0: Field3DLike | None = None,
    m_ex_y_0: Field3DLike | None = None,
    m_ey_x_0: Field3DLike | None = None,
    dEx_0: Field3DLike | None = None,
    dEy_0: Field3DLike | None = None,
    dEz_0: Field3DLike | None = None,
    dHx_0: Field3DLike | None = None,
    dHy_0: Field3DLike | None = None,
    dHz_0: Field3DLike | None = None,
    dm_hz_y_0: Field3DLike | None = None,
    dm_hy_z_0: Field3DLike | None = None,
    dm_hx_z_0: Field3DLike | None = None,
    dm_hz_x_0: Field3DLike | None = None,
    dm_hy_x_0: Field3DLike | None = None,
    dm_hx_y_0: Field3DLike | None = None,
    dm_ey_z_0: Field3DLike | None = None,
    dm_ez_y_0: Field3DLike | None = None,
    dm_ez_x_0: Field3DLike | None = None,
    dm_ex_z_0: Field3DLike | None = None,
    dm_ex_y_0: Field3DLike | None = None,
    dm_ey_x_0: Field3DLike | None = None,
    nt: int | None = None,
    model_gradient_sampling_interval: int = 1,
    parameterization: Literal["epsilon_sigma", "ca_cb"] = "epsilon_sigma",
    linearize_source: bool = True,
    source_component: str = "ey",
    receiver_component: str = "ey",
    freq_taper_frac: float = 0.0,
    time_pad_frac: float = 0.0,
    time_taper: bool = False,
    python_backend: Literal["eager", "jit", "compile"] | bool = False,
    storage_mode: Literal["device", "cpu", "disk", "none", "auto"] = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    n_threads: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """3D Maxwell Born propagator with background and scattered wavefields."""
    if epsilon.ndim != 3:
        raise NotImplementedError("born3d currently supports a single 3D model only.")
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )

    if isinstance(python_backend, bool):
        use_python = python_backend
    elif isinstance(python_backend, str):
        use_python = True
    else:
        raise TypeError(
            f"python_backend must be bool or str, but got {type(python_backend).__name__}"
        )

    source_component = _normalize_component_3d(
        source_component, name="source_component"
    )
    receiver_component = _normalize_component_3d(
        receiver_component, name="receiver_component"
    )

    if max_vel is None:
        max_vel_computed = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * C0
    else:
        max_vel_computed = max_vel

    inner_dt, step_ratio = cfl_condition(grid_spacing, dt, max_vel_computed)

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
        nt_internal = int(source_amplitude_internal.shape[-1])

    if not use_python:
        device_type = epsilon.device.type
        if device_type not in {"cpu", "cuda"}:
            use_python = True
        else:
            try:
                from .. import backend_utils

                if not backend_utils.is_backend_available():
                    warnings.warn(
                        "C/CUDA backend not available, falling back to Python born backend.",
                        RuntimeWarning,
                    )
                    use_python = True
            except ImportError:
                warnings.warn(
                    "backend_utils not available, falling back to Python born backend.",
                    RuntimeWarning,
                )
                use_python = True

    if use_python:
        result = born3d_python(
            epsilon,
            sigma,
            mu,
            depsilon,
            dsigma,
            dca,
            dcb,
            grid_spacing,
            inner_dt,
            source_amplitude_internal,
            source_location,
            receiver_location,
            bg_receiver_location,
            stencil=stencil,
            pml_width=pml_width,
            max_vel=max_vel_computed,
            Ex_0=Ex_0,
            Ey_0=Ey_0,
            Ez_0=Ez_0,
            Hx_0=Hx_0,
            Hy_0=Hy_0,
            Hz_0=Hz_0,
            m_hz_y_0=m_hz_y_0,
            m_hy_z_0=m_hy_z_0,
            m_hx_z_0=m_hx_z_0,
            m_hz_x_0=m_hz_x_0,
            m_hy_x_0=m_hy_x_0,
            m_hx_y_0=m_hx_y_0,
            m_ey_z_0=m_ey_z_0,
            m_ez_y_0=m_ez_y_0,
            m_ez_x_0=m_ez_x_0,
            m_ex_z_0=m_ex_z_0,
            m_ex_y_0=m_ex_y_0,
            m_ey_x_0=m_ey_x_0,
            dEx_0=dEx_0,
            dEy_0=dEy_0,
            dEz_0=dEz_0,
            dHx_0=dHx_0,
            dHy_0=dHy_0,
            dHz_0=dHz_0,
            dm_hz_y_0=dm_hz_y_0,
            dm_hy_z_0=dm_hy_z_0,
            dm_hx_z_0=dm_hx_z_0,
            dm_hz_x_0=dm_hz_x_0,
            dm_hy_x_0=dm_hy_x_0,
            dm_hx_y_0=dm_hx_y_0,
            dm_ey_z_0=dm_ey_z_0,
            dm_ez_y_0=dm_ez_y_0,
            dm_ez_x_0=dm_ez_x_0,
            dm_ex_z_0=dm_ex_z_0,
            dm_ex_y_0=dm_ex_y_0,
            dm_ey_x_0=dm_ey_x_0,
            nt=nt_internal,
            parameterization=parameterization,
            linearize_source=linearize_source,
            source_component=source_component,
            receiver_component=receiver_component,
        )
    else:
        result = born3d_c_cuda(
            epsilon,
            sigma,
            mu,
            depsilon,
            dsigma,
            dca,
            dcb,
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
            m_hz_y_0,
            m_hy_z_0,
            m_hx_z_0,
            m_hz_x_0,
            m_hy_x_0,
            m_hx_y_0,
            m_ey_z_0,
            m_ez_y_0,
            m_ez_x_0,
            m_ex_z_0,
            m_ex_y_0,
            m_ey_x_0,
            dEx_0,
            dEy_0,
            dEz_0,
            dHx_0,
            dHy_0,
            dHz_0,
            dm_hz_y_0,
            dm_hy_z_0,
            dm_hx_z_0,
            dm_hz_x_0,
            dm_hy_x_0,
            dm_hx_y_0,
            dm_ey_z_0,
            dm_ez_y_0,
            dm_ez_x_0,
            dm_ex_z_0,
            dm_ex_y_0,
            dm_ey_x_0,
            nt_internal,
            parameterization,
            model_gradient_sampling_interval,
            linearize_source,
            source_component,
            receiver_component,
            storage_mode=storage_mode,
            storage_path=storage_path,
            storage_compression=storage_compression,
            storage_bytes_limit_device=storage_bytes_limit_device,
            storage_bytes_limit_host=storage_bytes_limit_host,
            n_threads=n_threads,
        )
        bg_result = None
        if bg_receiver_location is not None and bg_receiver_location.numel() > 0:
            from .maxwell3d import maxwell3d

            bg_result = maxwell3d(
                epsilon,
                sigma,
                mu,
                grid_spacing=grid_spacing,
                dt=dt,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=bg_receiver_location,
                stencil=stencil,
                pml_width=pml_width,
                max_vel=max_vel,
                Ex_0=Ex_0,
                Ey_0=Ey_0,
                Ez_0=Ez_0,
                Hx_0=Hx_0,
                Hy_0=Hy_0,
                Hz_0=Hz_0,
                m_hz_y=m_hz_y_0,
                m_hy_z=m_hy_z_0,
                m_hx_z=m_hx_z_0,
                m_hz_x=m_hz_x_0,
                m_hy_x=m_hy_x_0,
                m_hx_y=m_hx_y_0,
                m_ey_z=m_ey_z_0,
                m_ez_y=m_ez_y_0,
                m_ez_x=m_ez_x_0,
                m_ex_z=m_ex_z_0,
                m_ex_y=m_ex_y_0,
                m_ey_x=m_ey_x_0,
                nt=nt,
                model_gradient_sampling_interval=model_gradient_sampling_interval,
                freq_taper_frac=freq_taper_frac,
                time_pad_frac=time_pad_frac,
                time_taper=time_taper,
                save_snapshots=False,
                source_component=source_component,
                receiver_component=receiver_component,
                python_backend=False,
                storage_mode="none",
                storage_path=storage_path,
                storage_compression=False,
                storage_bytes_limit_device=storage_bytes_limit_device,
                storage_bytes_limit_host=storage_bytes_limit_host,
                storage_chunk_steps=0,
                n_threads=n_threads,
            )

    if use_python:
        *state_outputs, bg_receiver_amplitudes, receiver_amplitudes = result
    else:
        *state_outputs, receiver_amplitudes = result
        if bg_result is None:
            bg_receiver_amplitudes = torch.empty(
                0, device=epsilon.device, dtype=epsilon.dtype
            )
        else:
            bg_receiver_amplitudes = bg_result[-1]

    if use_python and step_ratio > 1 and bg_receiver_amplitudes.numel() > 0:
        bg_receiver_amplitudes = downsample_and_movedim(
            bg_receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
        bg_receiver_amplitudes = torch.movedim(bg_receiver_amplitudes, -1, 0)

    if step_ratio > 1 and receiver_amplitudes.numel() > 0:
        receiver_amplitudes = downsample_and_movedim(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
        receiver_amplitudes = torch.movedim(receiver_amplitudes, -1, 0)

    return (*state_outputs, bg_receiver_amplitudes, receiver_amplitudes)


__all__ = ["Born3D", "born3d"]
