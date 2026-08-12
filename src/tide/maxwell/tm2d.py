from collections.abc import Callable, Sequence
from typing import Literal

import torch

from ..callbacks import Callback
from ..cfl import cfl_condition
from ..core import (
    BackendPreference,
)
from ..dispersion import DebyeDispersion
from ..grid_utils import _normalize_grid_spacing_2d
from ..resampling import downsample_and_movedim, upsample
from ..typing import (
    Field2DLike,
    Model2D,
    Model2DLike,
    ReceiverData,
    ReceiverLocation2D,
    SourceLocation2D,
    WaveletBatch,
    runtime_typecheck,
)
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
from ..core import GradientTarget, derive_gradient_targets
from .dispatch import ExecutionPolicy, compile_execution_policy
from .tm2d_python import maxwell_func
from .validation_internal import (
    _validate_dispersion_time_step,
    _validate_location_bounds,
    _validate_optional_bool,
    _validate_positive_int,
    _validate_tensor_arg,
)

ReceiverMisfit = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _resolve_tm2d_execution_policy(
    *,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    python_backend: Literal["eager", "jit", "compile"] | bool,
    compute_mode: str,
    storage_mode: str,
    storage_path: str,
    storage_compression: bool | str,
    storage_bytes_limit_device: int | None,
    storage_bytes_limit_host: int | None,
    storage_chunk_steps: int,
    n_threads: int | None,
    fallback: str,
    has_callbacks: bool,
    model_gradient_sampling_interval: int,
    gradient_targets: frozenset[GradientTarget] | None = None,
    has_dispersion: bool = False,
) -> ExecutionPolicy:
    """Compile the cross-cutting plan and select one TM2D execution adapter."""
    execution = compile_execution_policy(
        requested_backend=python_backend,
        operation="forward",
        dimension="tm2d",
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        compute_mode=compute_mode,
        storage_mode=storage_mode,
        storage_path=storage_path,
        storage_compression=storage_compression,
        storage_bytes_limit_device=storage_bytes_limit_device,
        storage_bytes_limit_host=storage_bytes_limit_host,
        storage_chunk_steps=storage_chunk_steps,
        n_threads=n_threads,
        fallback=fallback,
        has_callbacks=has_callbacks,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        gradient_targets=gradient_targets,
        has_dispersion=has_dispersion,
    )
    return execution


def _default_receiver_misfit(
    predicted: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    residual = predicted - observed
    return 0.5 * residual.square().sum()


@runtime_typecheck
def maxwelltm_hvp(
    epsilon: Model2D,
    sigma: Model2D,
    mu: Model2D,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: WaveletBatch | None,
    source_location: SourceLocation2D | None,
    receiver_location: ReceiverLocation2D | None,
    observed_data: ReceiverData,
    *,
    vepsilon: Model2D | None = None,
    vsigma: Model2D | None = None,
    misfit: ReceiverMisfit | None = None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    nt: int | None = None,
    model_gradient_sampling_interval: int = 1,
    linearize_source: bool = True,
    hessian_mode: Literal["full", "gauss_newton"] = "full",
    python_backend: bool = False,
    storage_mode: Literal["device", "cpu", "disk"] = "device",
    storage_compression: bool | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a receiver-space Hessian to a model direction.

    ``hessian_mode="full"`` evaluates the complete receiver-space Hessian:

        Hv = grad_m <dPhi/dd, Jv>

    ``hessian_mode="gauss_newton"`` evaluates only ``J.T @ Phi'' @ Jv``.
    The Python path (`python_backend=True`) is the reference implementation.
    The native Gauss-Newton path composes a tangent forward with the ordinary
    background adjoint. The native full path adds the nonlinear-physics
    correction through a fused incremental adjoint.

    `model_gradient_sampling_interval` follows `maxwelltm` semantics on the
    native HVP path. The Python HVP path and the native CPU HVP path currently
    support only the effective interval 1.
    """
    _validate_optional_bool("python_backend", python_backend)
    _validate_tensor_arg("epsilon", epsilon)
    _validate_tensor_arg("sigma", sigma)
    _validate_tensor_arg("mu", mu)
    _validate_tensor_arg("observed_data", observed_data)
    if hessian_mode not in {"full", "gauss_newton"}:
        raise ValueError(
            f"hessian_mode must be 'full' or 'gauss_newton', but got {hessian_mode!r}."
        )
    if storage_mode not in {"device", "cpu", "disk"}:
        raise ValueError(
            "storage_mode must be 'device', 'cpu', or 'disk', "
            f"but got {storage_mode!r}."
        )
    if epsilon.ndim != 2:
        raise NotImplementedError("maxwelltm_hvp currently supports a single 2D model.")
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
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )

    misfit_fn = _default_receiver_misfit if misfit is None else misfit
    if not callable(misfit_fn):
        raise TypeError("misfit must be callable when provided.")

    execution = compile_execution_policy(
        requested_backend=python_backend,
        operation="second_vjp",
        dimension="tm2d",
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        storage_mode=storage_mode,
        storage_compression=storage_compression or False,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        hessian_mode=hessian_mode,
        gradient_targets=derive_gradient_targets(
            epsilon=epsilon,
            sigma=sigma,
            mu=mu,
        ),
    )
    decision = execution.decision

    if decision.selected is BackendPreference.REFERENCE:
        from .tm2d_born_autograd import tm2d_receiver_hvp_naive

        return tm2d_receiver_hvp_naive(
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
            model_gradient_sampling_interval=model_gradient_sampling_interval,
            linearize_source=linearize_source,
            hessian_mode=hessian_mode,
        )

    from .tm2d_born_autograd import (
        tm2d_receiver_full_hvp_native,
        tm2d_receiver_gn_hvp_native,
    )

    native_hvp = (
        tm2d_receiver_gn_hvp_native
        if hessian_mode == "gauss_newton"
        else tm2d_receiver_full_hvp_native
    )
    return native_hvp(
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
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        linearize_source=linearize_source,
        storage_mode=storage_mode,
        storage_compression=storage_compression,
    )


@runtime_typecheck
def maxwelltm(
    epsilon: Model2DLike,
    sigma: Model2DLike,
    mu: Model2DLike,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: WaveletBatch | None,
    source_location: SourceLocation2D | None,
    receiver_location: ReceiverLocation2D | None,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    Ey_0: Field2DLike | None = None,
    Hx_0: Field2DLike | None = None,
    Hz_0: Field2DLike | None = None,
    m_Ey_x: Field2DLike | None = None,
    m_Ey_z: Field2DLike | None = None,
    m_Hx_z: Field2DLike | None = None,
    m_Hz_x: Field2DLike | None = None,
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
    compute_mode: Literal["native"] = "native",
    fallback: Literal["reference", "error"] = "reference",
):
    """2D TM mode Maxwell equations solver.

    """

    epsilon_input = epsilon
    sigma_input = sigma
    mu_input = mu
    source_amplitude_input = source_amplitude
    source_location_input = source_location
    receiver_location_input = receiver_location
    Ey_0_input = Ey_0
    Hx_0_input = Hx_0
    Hz_0_input = Hz_0
    m_Ey_x_input = m_Ey_x
    m_Ey_z_input = m_Ey_z
    m_Hx_z_input = m_Hx_z
    m_Hz_x_input = m_Hz_x

    batch_meta = _normalize_structured_batch(
        spatial_ndim=2,
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        shot_tensors={
            "source_amplitude": (source_amplitude, 2),
            "source_location": (source_location, 2),
            "receiver_location": (receiver_location, 2),
        },
        state_tensors={
            "Ey_0": Ey_0,
            "Hx_0": Hx_0,
            "Hz_0": Hz_0,
            "m_Ey_x": m_Ey_x,
            "m_Ey_z": m_Ey_z,
            "m_Hx_z": m_Hx_z,
            "m_Hz_x": m_Hz_x,
        },
    )
    epsilon = batch_meta["epsilon"]
    sigma = batch_meta["sigma"]
    mu = batch_meta["mu"]
    source_amplitude = batch_meta["shot_tensors"]["source_amplitude"]
    source_location = batch_meta["shot_tensors"]["source_location"]
    receiver_location = batch_meta["shot_tensors"]["receiver_location"]
    Ey_0 = batch_meta["state_tensors"]["Ey_0"]
    Hx_0 = batch_meta["state_tensors"]["Hx_0"]
    Hz_0 = batch_meta["state_tensors"]["Hz_0"]
    m_Ey_x = batch_meta["state_tensors"]["m_Ey_x"]
    m_Ey_z = batch_meta["state_tensors"]["m_Ey_z"]
    m_Hx_z = batch_meta["state_tensors"]["m_Hx_z"]
    m_Hz_x = batch_meta["state_tensors"]["m_Hz_x"]

    execution = _resolve_tm2d_execution_policy(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        python_backend=python_backend,
        compute_mode=compute_mode,
        storage_mode=storage_mode,
        storage_path=storage_path,
        storage_compression=storage_compression,
        storage_bytes_limit_device=storage_bytes_limit_device,
        storage_bytes_limit_host=storage_bytes_limit_host,
        storage_chunk_steps=storage_chunk_steps,
        n_threads=n_threads,
        fallback=fallback,
        has_callbacks=forward_callback is not None or backward_callback is not None,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        gradient_targets=derive_gradient_targets(
            epsilon=epsilon_input,
            sigma=sigma_input,
            mu=mu_input,
            source_amplitude=source_amplitude_input,
            state_tensors=(
                Ey_0_input,
                Hx_0_input,
                Hz_0_input,
                m_Ey_x_input,
                m_Ey_z_input,
                m_Hx_z_input,
                m_Hz_x_input,
            ),
        ),
        has_dispersion=dispersion is not None,
    )
    compute_mode = execution.compute_mode
    storage_mode = execution.storage_mode
    use_python = execution.use_python
    dispatch_backend = execution.dispatch_backend

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
            "Ey_0": _structured_vmap_state_in_dim(
                "Ey_0",
                Ey_0_input,
                B=B,
                spatial_ndim=2,
            ),
            "Hx_0": _structured_vmap_state_in_dim(
                "Hx_0",
                Hx_0_input,
                B=B,
                spatial_ndim=2,
            ),
            "Hz_0": _structured_vmap_state_in_dim(
                "Hz_0",
                Hz_0_input,
                B=B,
                spatial_ndim=2,
            ),
            "m_Ey_x": _structured_vmap_state_in_dim(
                "m_Ey_x",
                m_Ey_x_input,
                B=B,
                spatial_ndim=2,
            ),
            "m_Ey_z": _structured_vmap_state_in_dim(
                "m_Ey_z",
                m_Ey_z_input,
                B=B,
                spatial_ndim=2,
            ),
            "m_Hx_z": _structured_vmap_state_in_dim(
                "m_Hx_z",
                m_Hx_z_input,
                B=B,
                spatial_ndim=2,
            ),
            "m_Hz_x": _structured_vmap_state_in_dim(
                "m_Hz_x",
                m_Hz_x_input,
                B=B,
                spatial_ndim=2,
            ),
        }

        def _single_model_forward(
            epsilon_i: torch.Tensor,
            sigma_i: torch.Tensor,
            mu_i: torch.Tensor,
            source_amplitude_i: torch.Tensor | None,
            source_location_i: torch.Tensor | None,
            receiver_location_i: torch.Tensor | None,
            Ey_0_i: torch.Tensor | None,
            Hx_0_i: torch.Tensor | None,
            Hz_0_i: torch.Tensor | None,
            m_Ey_x_i: torch.Tensor | None,
            m_Ey_z_i: torch.Tensor | None,
            m_Hx_z_i: torch.Tensor | None,
            m_Hz_x_i: torch.Tensor | None,
        ):
            return maxwell_func(
                dispatch_backend,
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
                Ey_0_i,
                Hx_0_i,
                Hz_0_i,
                m_Ey_x_i,
                m_Ey_z_i,
                m_Hx_z_i,
                m_Hz_x_i,
                nt_internal,
                model_gradient_sampling_interval,
                freq_taper_frac,
                time_pad_frac,
                time_taper,
                save_snapshots,
                None,
                None,
                callback_frequency,
                storage_mode,
                storage_path,
                storage_compression,
                storage_bytes_limit_device,
                storage_bytes_limit_host,
                storage_chunk_steps,
                n_threads,
                dispersion,
                validate_material_inputs=False,
                fallback=fallback,
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
                state_in_dims["Ey_0"],
                state_in_dims["Hx_0"],
                state_in_dims["Hz_0"],
                state_in_dims["m_Ey_x"],
                state_in_dims["m_Ey_z"],
                state_in_dims["m_Hx_z"],
                state_in_dims["m_Hz_x"],
            ),
        )(
            epsilon_input,
            sigma_input,
            mu_input,
            source_amplitude_vmap,
            source_location_input,
            receiver_location_input,
            Ey_0_input,
            Hx_0_input,
            Hz_0_input,
            m_Ey_x_input,
            m_Ey_z_input,
            m_Hx_z_input,
            m_Hz_x_input,
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
            Ey_out = Ey_out.squeeze(0)
            Hx_out = Hx_out.squeeze(0)
            Hz_out = Hz_out.squeeze(0)
            m_Ey_x_out = m_Ey_x_out.squeeze(0)
            m_Ey_z_out = m_Ey_z_out.squeeze(0)
            m_Hx_z_out = m_Hx_z_out.squeeze(0)
            m_Hz_x_out = m_Hz_x_out.squeeze(0)
            if receiver_amplitudes.ndim > 1:
                receiver_amplitudes = receiver_amplitudes.squeeze(1)

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

    forward_callback_wrapped = _wrap_structured_callback(
        forward_callback,
        batch_meta=batch_meta,
    )
    backward_callback_wrapped = _wrap_structured_callback(
        backward_callback,
        batch_meta=batch_meta,
    )

    result = maxwell_func(
        dispatch_backend,
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
        forward_callback_wrapped,
        backward_callback_wrapped,
        callback_frequency,
        storage_mode,
        storage_path,
        storage_compression,
        storage_bytes_limit_device,
        storage_bytes_limit_host,
        storage_chunk_steps,
        n_threads,
        dispersion,
        compute_mode=compute_mode,
        fallback=fallback,
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

    Ey_out = _reshape_structured_wavefield(Ey_out, batch_meta=batch_meta)
    Hx_out = _reshape_structured_wavefield(Hx_out, batch_meta=batch_meta)
    Hz_out = _reshape_structured_wavefield(Hz_out, batch_meta=batch_meta)
    m_Ey_x_out = _reshape_structured_wavefield(m_Ey_x_out, batch_meta=batch_meta)
    m_Ey_z_out = _reshape_structured_wavefield(m_Ey_z_out, batch_meta=batch_meta)
    m_Hx_z_out = _reshape_structured_wavefield(m_Hx_z_out, batch_meta=batch_meta)
    m_Hz_x_out = _reshape_structured_wavefield(m_Hz_x_out, batch_meta=batch_meta)
    receiver_amplitudes = _reshape_structured_receiver_amplitudes(
        receiver_amplitudes,
        batch_meta=batch_meta,
    )

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


__all__: list[str] = []
