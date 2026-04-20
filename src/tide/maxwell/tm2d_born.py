import warnings
from collections.abc import Sequence
from typing import Literal

import torch

from ..cfl import cfl_condition
from ..resampling import downsample_and_movedim, upsample
from ..utils import C0
from .tm2d_born_cuda import borntm_c_cuda
from .tm2d_born_python import borntm_python


def borntm(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    *,
    depsilon: torch.Tensor | None = None,
    dsigma: torch.Tensor | None = None,
    dca: torch.Tensor | None = None,
    dcb: torch.Tensor | None = None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """2D TM Born forward operator.

    Native backend gradients are supported with respect to the perturbation
    inputs (`depsilon`, `dsigma`, `dca`, `dcb`), while the background model is
    treated as fixed.
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
            stencil=stencil,
            pml_width=pml_width,
            max_vel=max_vel_computed,
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
            dEy_0,
            dHx_0,
            dHz_0,
            dm_Ey_x_0,
            dm_Ey_z_0,
            dm_Hx_z_0,
            dm_Hz_x_0,
            nt_internal,
            parameterization,
            linearize_source,
            storage_mode=storage_mode,
            storage_path=storage_path,
            storage_compression=storage_compression,
            storage_bytes_limit_device=storage_bytes_limit_device,
            storage_bytes_limit_host=storage_bytes_limit_host,
            n_threads=n_threads,
        )

    receiver_amplitudes = result[-1]
    if step_ratio > 1 and receiver_amplitudes.numel() > 0:
        receiver_amplitudes = downsample_and_movedim(
            receiver_amplitudes,
            step_ratio,
            freq_taper_frac=freq_taper_frac,
            time_pad_frac=time_pad_frac,
            time_taper=time_taper,
        )
        receiver_amplitudes = torch.movedim(receiver_amplitudes, -1, 0)

    return (*result[:-1], receiver_amplitudes)


def borntm_adjoint(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    residual: torch.Tensor,
    *,
    pml_width: int | Sequence[int] = 20,
    stencil: int = 2,
    max_vel: float | None = None,
    linearize_source: bool = True,
    return_sigma: bool = False,
    python_backend: Literal["eager", "jit", "compile"] | bool = False,
    storage_mode: Literal["device", "cpu", "disk", "none", "auto"] = "device",
    storage_path: str = ".",
    storage_compression: bool | str = False,
    storage_bytes_limit_device: int | None = None,
    storage_bytes_limit_host: int | None = None,
    n_threads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the 2D TM Born adjoint operator."""
    epsilon_req = torch.zeros_like(epsilon, requires_grad=True)
    sigma_req = torch.zeros_like(sigma, requires_grad=True) if return_sigma else None

    pred = borntm(
        epsilon,
        sigma,
        mu,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        depsilon=epsilon_req,
        dsigma=sigma_req,
        stencil=stencil,
        pml_width=pml_width,
        max_vel=max_vel,
        linearize_source=linearize_source,
        python_backend=python_backend,
        storage_mode=storage_mode,
        storage_path=storage_path,
        storage_compression=storage_compression,
        storage_bytes_limit_device=storage_bytes_limit_device,
        storage_bytes_limit_host=storage_bytes_limit_host,
        n_threads=n_threads,
    )[-1]

    targets = (epsilon_req, sigma_req) if return_sigma else (epsilon_req,)
    loss = torch.sum(pred * residual)
    grads = torch.autograd.grad(loss, targets)
    if return_sigma:
        return grads[0], grads[1]
    return grads[0], None


__all__ = ["borntm", "borntm_adjoint"]
