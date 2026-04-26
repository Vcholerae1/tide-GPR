import warnings
from collections.abc import Sequence
from typing import Literal

import torch

from ..cfl import cfl_condition
from ..resampling import downsample_and_movedim, upsample
from ..utils import C0
from .tm2d_born_cuda import borntm_c_cuda
from .tm2d_born_python import borntm_python
from .validation_internal import _validate_optional_bool, _validate_tensor_arg


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


class BornTM(torch.nn.Module):
    """Module wrapper around :func:`borntm`.

    The wrapper stores the background Maxwell model and an optional Born
    perturbation inside a reusable ``torch.nn.Module``. This mirrors the module
    oriented workflow that Deepwave exposes for ``ScalarBorn`` while preserving
    TIDE's explicit ``borntm`` functional API.
    """

    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
        grid_spacing: float | Sequence[float],
        *,
        depsilon: torch.Tensor | None = None,
        dsigma: torch.Tensor | None = None,
        dca: torch.Tensor | None = None,
        dcb: torch.Tensor | None = None,
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

    def forward(
        self,
        dt: float,
        source_amplitude: torch.Tensor | None = None,
        source_location: torch.Tensor | None = None,
        receiver_location: torch.Tensor | None = None,
        bg_receiver_location: torch.Tensor | None = None,
        *,
        depsilon: torch.Tensor | None = None,
        dsigma: torch.Tensor | None = None,
        dca: torch.Tensor | None = None,
        dcb: torch.Tensor | None = None,
        stencil: int = 2,
        pml_width: int | Sequence[int] = 20,
        max_vel: float | None = None,
        Ey_0: torch.Tensor | None = None,
        Hx_0: torch.Tensor | None = None,
        Hz_0: torch.Tensor | None = None,
        m_Ey_x_0: torch.Tensor | None = None,
        m_Ey_z_0: torch.Tensor | None = None,
        m_Hx_z_0: torch.Tensor | None = None,
        m_Hz_x_0: torch.Tensor | None = None,
        dEy_0: torch.Tensor | None = None,
        dHx_0: torch.Tensor | None = None,
        dHz_0: torch.Tensor | None = None,
        dm_Ey_x_0: torch.Tensor | None = None,
        dm_Ey_z_0: torch.Tensor | None = None,
        dm_Hx_z_0: torch.Tensor | None = None,
        dm_Hz_x_0: torch.Tensor | None = None,
        nt: int | None = None,
        linearize_source: bool | None = None,
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
        return borntm(
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
            Ey_0=Ey_0,
            Hx_0=Hx_0,
            Hz_0=Hz_0,
            m_Ey_x_0=m_Ey_x_0,
            m_Ey_z_0=m_Ey_z_0,
            m_Hx_z_0=m_Hx_z_0,
            m_Hz_x_0=m_Hz_x_0,
            dEy_0=dEy_0,
            dHx_0=dHx_0,
            dHz_0=dHz_0,
            dm_Ey_x_0=dm_Ey_x_0,
            dm_Ey_z_0=dm_Ey_z_0,
            dm_Hx_z_0=dm_Hx_z_0,
            dm_Hz_x_0=dm_Hz_x_0,
            nt=nt,
            parameterization=self.parameterization,
            linearize_source=linearize_source,
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


def borntm(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    bg_receiver_location: torch.Tensor | None = None,
    *,
    depsilon: torch.Tensor | None = None,
    dsigma: torch.Tensor | None = None,
    dca: torch.Tensor | None = None,
    dcb: torch.Tensor | None = None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    Ey_0: torch.Tensor | None = None,
    Hx_0: torch.Tensor | None = None,
    Hz_0: torch.Tensor | None = None,
    m_Ey_x_0: torch.Tensor | None = None,
    m_Ey_z_0: torch.Tensor | None = None,
    m_Hx_z_0: torch.Tensor | None = None,
    m_Hz_x_0: torch.Tensor | None = None,
    dEy_0: torch.Tensor | None = None,
    dHx_0: torch.Tensor | None = None,
    dHz_0: torch.Tensor | None = None,
    dm_Ey_x_0: torch.Tensor | None = None,
    dm_Ey_z_0: torch.Tensor | None = None,
    dm_Hx_z_0: torch.Tensor | None = None,
    dm_Hz_x_0: torch.Tensor | None = None,
    nt: int | None = None,
    parameterization: Literal["epsilon_sigma", "ca_cb"] = "epsilon_sigma",
    linearize_source: bool = True,
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
    """2D TM Born propagator with background and scattered wavefields.

    The return value follows a two-wavefield layout:

    1. background final state tensors,
    2. scattered final state tensors,
    3. background receiver amplitudes,
    4. scattered receiver amplitudes.

    ``borntm`` computes the Born scattered field ``J(m)v``. The scattered field
    remains linear in the perturbation inputs (`depsilon`, `dsigma`, `dca`,
    `dcb`), while the operator is differentiable with respect to the background
    model (`epsilon`, `sigma`) as well. Native fallback is still used for
    unsupported gradient paths such as `mu`, source amplitudes, or initial
    wavefields.
    """
    if epsilon.ndim != 2:
        raise NotImplementedError("borntm currently supports a single 2D model only.")

    if isinstance(python_backend, bool):
        use_python = python_backend
    elif isinstance(python_backend, str):
        use_python = True
    else:
        raise TypeError(
            f"python_backend must be bool or str, but got {type(python_backend).__name__}"
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
        result = borntm_python(
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
            Ey_0=Ey_0,
            Hx_0=Hx_0,
            Hz_0=Hz_0,
            m_Ey_x_0=m_Ey_x_0,
            m_Ey_z_0=m_Ey_z_0,
            m_Hx_z_0=m_Hx_z_0,
            m_Hz_x_0=m_Hz_x_0,
            dEy_0=dEy_0,
            dHx_0=dHx_0,
            dHz_0=dHz_0,
            dm_Ey_x_0=dm_Ey_x_0,
            dm_Ey_z_0=dm_Ey_z_0,
            dm_Hx_z_0=dm_Hx_z_0,
            dm_Hz_x_0=dm_Hz_x_0,
            nt=nt_internal,
            parameterization=parameterization,
            linearize_source=linearize_source,
        )
    else:
        result = borntm_c_cuda(
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
            Ey_0,
            Hx_0,
            Hz_0,
            m_Ey_x_0,
            m_Ey_z_0,
            m_Hx_z_0,
            m_Hz_x_0,
            dEy_0,
            dHx_0,
            dHz_0,
            dm_Ey_x_0,
            dm_Ey_z_0,
            dm_Hx_z_0,
            dm_Hz_x_0,
            nt_internal,
            parameterization,
            1,
            linearize_source,
            storage_mode=storage_mode,
            storage_path=storage_path,
            storage_compression=storage_compression,
            storage_bytes_limit_device=storage_bytes_limit_device,
            storage_bytes_limit_host=storage_bytes_limit_host,
            n_threads=n_threads,
        )
        bg_result = None
        if bg_receiver_location is not None and bg_receiver_location.numel() > 0:
            from .tm2d import maxwelltm

            bg_result = maxwelltm(
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
                Ey_0=Ey_0,
                Hx_0=Hx_0,
                Hz_0=Hz_0,
                m_Ey_x=m_Ey_x_0,
                m_Ey_z=m_Ey_z_0,
                m_Hx_z=m_Hx_z_0,
                m_Hz_x=m_Hz_x_0,
                nt=nt,
                freq_taper_frac=freq_taper_frac,
                time_pad_frac=time_pad_frac,
                time_taper=time_taper,
                save_snapshots=False,
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


__all__ = ["BornTM", "borntm"]
