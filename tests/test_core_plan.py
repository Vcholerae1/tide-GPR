from dataclasses import replace

import pytest
import torch

import tide
from tide import backend_utils
from tide.core import (
    BackendCapabilities,
    BackendCapability,
    BackendPreference,
    ComputeMode,
    Dimension,
    FallbackPolicy,
    GradientTarget,
    Operation,
    compile_simulation_plan,
    normalize_backend_request,
    select_backend,
)
from tide.core.backends import backend_capabilities
from tide.maxwell.dispatch import compile_execution_policy


def test_compile_plan_normalizes_legacy_options() -> None:
    epsilon = torch.ones(4, 5, dtype=torch.float64)
    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        sigma=torch.zeros_like(epsilon),
        mu=torch.ones_like(epsilon),
        python_backend="eager",
        storage_mode="DEVICE",
        compute_mode="native",
        storage_chunk_steps=3,
    )

    assert plan.dimension is Dimension.TM2D
    assert plan.backend is BackendPreference.REFERENCE
    assert plan.compute_mode is ComputeMode.NATIVE
    assert plan.storage.mode.value == "device"
    assert plan.storage.chunk_steps == 3


def test_compile_plan_detects_batched_3d_models() -> None:
    epsilon = torch.ones(2, 3, 4, 5)
    plan = compile_simulation_plan(dimension="em3d", epsilon=epsilon)
    assert plan.model_batched
    assert plan.source_component == "ey"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, BackendPreference.AUTO),
        (True, BackendPreference.REFERENCE),
        ("compile", BackendPreference.REFERENCE),
        ("native", BackendPreference.NATIVE),
    ],
)
def test_normalize_backend_request(
    value: bool | str, expected: BackendPreference
) -> None:
    assert normalize_backend_request(value) is expected


def test_select_backend_has_explicit_fallback_policy() -> None:
    epsilon = torch.ones(4, 4)
    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        fallback=FallbackPolicy.REFERENCE.value,
    )
    decision = select_backend(plan, native_available=False)
    assert decision.selected is BackendPreference.REFERENCE
    assert decision.fallback

    strict_plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        fallback=FallbackPolicy.ERROR.value,
    )
    with pytest.raises(RuntimeError, match="native backend"):
        select_backend(strict_plan, native_available=False)


def test_compile_plan_derives_gradient_targets_from_tensors() -> None:
    epsilon = torch.ones(4, 4)
    sigma = torch.ones(4, 4)
    mu = torch.ones(4, 4)

    plan = compile_simulation_plan(dimension="tm2d", epsilon=epsilon)
    assert plan.gradient_targets == frozenset()

    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon.clone().requires_grad_(True),
        sigma=sigma,
        mu=mu,
    )
    assert plan.gradient_targets == frozenset({GradientTarget.EPSILON})
    assert plan.has_model_gradients

    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        sigma=sigma,
        mu=mu.clone().requires_grad_(True),
    )
    assert plan.gradient_targets == frozenset({GradientTarget.MU})
    assert plan.has_model_gradients


def test_compile_plan_accepts_explicit_gradient_targets() -> None:
    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=torch.ones(4, 4),
        gradient_targets=["epsilon", "source"],
    )
    assert plan.gradient_targets == frozenset(
        {GradientTarget.EPSILON, GradientTarget.SOURCE}
    )

    with pytest.raises(ValueError, match="gradient_targets must be a subset"):
        compile_simulation_plan(
            dimension="tm2d",
            epsilon=torch.ones(4, 4),
            gradient_targets=["epsilon", "not_a_target"],
        )


def test_select_backend_rejects_unsupported_gradient_targets() -> None:
    epsilon = torch.ones(4, 4)

    strict_plan = compile_simulation_plan(
        operation="jvp",
        dimension="tm2d",
        epsilon=epsilon,
        gradient_targets=["mu"],
        fallback=FallbackPolicy.ERROR.value,
    )
    with pytest.raises(
        NotImplementedError, match="does not support gradients w.r.t. mu"
    ):
        select_backend(strict_plan, native_available=True)

    reference_plan = compile_simulation_plan(
        operation="jvp",
        dimension="tm2d",
        epsilon=epsilon,
        gradient_targets=["mu"],
        fallback=FallbackPolicy.REFERENCE.value,
    )
    decision = select_backend(reference_plan, native_available=True)
    assert decision.selected is BackendPreference.REFERENCE
    assert decision.fallback

    state_plan = compile_simulation_plan(
        operation="jvp",
        dimension="em3d",
        epsilon=torch.ones(3, 4, 4, 4),
        gradient_targets=["state"],
        fallback=FallbackPolicy.ERROR.value,
    )
    with pytest.raises(
        NotImplementedError, match="does not support gradients w.r.t. state"
    ):
        select_backend(state_plan, native_available=True)


def test_born_adapter_rejects_unsupported_gradient_targets() -> None:
    from tide.maxwell.tm2d_born_cuda import borntm_c_cuda

    epsilon = torch.ones(4, 4)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon, requires_grad=True)

    with pytest.raises(NotImplementedError, match="Python reference backend"):
        borntm_c_cuda(
            epsilon,
            sigma,
            mu,
            None,
            None,
            None,
            None,
            1.0,
            1e-10,
            None,
            None,
            None,
            stencil=2,
            pml_width=20,
            max_vel=None,
            Ey_0=None,
            Hx_0=None,
            Hz_0=None,
            m_Ey_x_0=None,
            m_Ey_z_0=None,
            m_Hx_z_0=None,
            m_Hz_x_0=None,
            dEy_0=None,
            dHx_0=None,
            dHz_0=None,
            dm_Ey_x_0=None,
            dm_Ey_z_0=None,
            dm_Hx_z_0=None,
            dm_Hz_x_0=None,
            nt=1,
            parameterization="epsilon_sigma",
            model_gradient_sampling_interval=1,
            linearize_source=True,
        )


def test_select_backend_rejects_perturbation_gradients_without_storage() -> None:
    epsilon = torch.ones(4, 4)

    strict_plan = compile_simulation_plan(
        operation="jvp",
        dimension="tm2d",
        epsilon=epsilon,
        gradient_targets=["perturbation"],
        storage_mode="none",
        fallback=FallbackPolicy.ERROR.value,
    )
    with pytest.raises(
        NotImplementedError, match="gradients w.r.t. perturbation with storage_mode"
    ):
        select_backend(strict_plan, native_available=True)

    reference_plan = compile_simulation_plan(
        operation="jvp",
        dimension="tm2d",
        epsilon=epsilon,
        gradient_targets=["perturbation"],
        storage_mode="none",
        fallback=FallbackPolicy.REFERENCE.value,
    )
    decision = select_backend(reference_plan, native_available=True)
    assert decision.selected is BackendPreference.REFERENCE
    assert decision.fallback


def test_select_backend_rejects_dispersion_gradients_on_native() -> None:
    epsilon = torch.ones(4, 4)

    strict_plan = compile_simulation_plan(
        operation="forward",
        dimension="tm2d",
        epsilon=epsilon,
        gradient_targets=["epsilon"],
        has_dispersion=True,
        fallback=FallbackPolicy.ERROR.value,
    )
    with pytest.raises(NotImplementedError, match="does not support gradients with dispersion"):
        select_backend(strict_plan, native_available=True)

    reference_plan = compile_simulation_plan(
        operation="forward",
        dimension="tm2d",
        epsilon=epsilon,
        gradient_targets=["epsilon"],
        has_dispersion=True,
        fallback=FallbackPolicy.REFERENCE.value,
    )
    decision = select_backend(reference_plan, native_available=True)
    assert decision.selected is BackendPreference.REFERENCE
    assert decision.fallback


def test_dispatch_backend_preserves_central_python_decision() -> None:
    policy = compile_execution_policy(
        requested_backend=False,
        operation="forward",
        dimension="tm2d",
        epsilon=torch.ones(4, 4),
        gradient_targets=["mu"],
    )
    assert policy.use_python
    assert policy.dispatch_backend is True

    compiled_policy = compile_execution_policy(
        requested_backend="compile",
        operation="forward",
        dimension="tm2d",
        epsilon=torch.ones(4, 4),
        gradient_targets=["mu"],
    )
    assert compiled_policy.use_python
    assert compiled_policy.dispatch_backend == "compile"


def test_maxwelltm_gradient_fallback_runs_python_end_to_end() -> None:
    # Audit repro: the central decision selects Python for mu gradients, so the
    # downstream dispatch must not re-discover the native backend.
    epsilon = torch.full((8, 8), 4.0)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon, requires_grad=True)
    source_amplitude = torch.randn(1, 1, 16)
    source_location = torch.tensor([[[4, 4]]], dtype=torch.long)
    receiver_location = torch.tensor([[[2, 2]]], dtype=torch.long)

    out = tide.maxwell._kernel_api.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=3.5e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=3,
        fallback="reference",
    )
    receiver = out[-1]
    assert receiver.requires_grad
    torch.autograd.grad(receiver.sum(), mu)


def test_debye_gradient_fallback_honors_error_policy() -> None:
    # Audit repro: dispersion + gradients with fallback="error" must raise
    # instead of silently switching to Python.
    epsilon = torch.full((8, 8), 4.0, requires_grad=True)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_amplitude = torch.randn(1, 1, 16)
    source_location = torch.tensor([[[4, 4]]], dtype=torch.long)
    receiver_location = torch.tensor([[[2, 2]]], dtype=torch.long)

    with pytest.raises(
        (NotImplementedError, RuntimeError), match="gradients with dispersion|Debye"
    ):
        tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=1,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
            fallback="error",
        )


def test_second_vjp_capability_rejection_comes_from_backend_decision() -> None:
    epsilon = torch.ones(4, 4)
    plan = compile_simulation_plan(
        operation="second_vjp",
        dimension="tm2d",
        epsilon=epsilon,
        python_backend=True,
        model_gradient_sampling_interval=2,
    )

    assert plan.operation is Operation.SECOND_VJP
    with pytest.raises(NotImplementedError, match="Python TM2D second_vjp currently"):
        select_backend(plan, native_available=True)


def test_linearization_decision_owns_background_reuse_capability() -> None:
    cpu_plan = compile_simulation_plan(
        operation="jvp",
        dimension="tm2d",
        epsilon=torch.ones(4, 4),
        storage_mode="device",
    )
    plan = replace(cpu_plan, device=torch.device("cuda"))
    decision = select_backend(plan, native_available=True)

    assert decision.selected is BackendPreference.NATIVE
    assert decision.can_reuse_background(plan)


def test_compile_plan_rejects_removed_reduced_precision_mode() -> None:
    with pytest.raises(ValueError, match="FP16 support was removed"):
        compile_simulation_plan(
            dimension="tm2d",
            epsilon=torch.ones(4, 4),
            python_backend=True,
            compute_mode="fp16_io",
        )


def test_backend_templates_are_validated_and_named() -> None:
    backend_utils._validate_template_specs()
    signature = backend_utils.backend_signature("maxwell_tm", "forward")

    assert len(signature) == 54
    assert signature[:3] == ("ca", "cb", "cq")
    assert len(signature) == len(set(signature))


def test_backend_signature_rejects_unknown_template() -> None:
    with pytest.raises(KeyError):
        backend_utils.backend_signature("unknown", "forward")


def _capability_cells(
    capabilities: BackendCapabilities,
) -> set[
    tuple[
        Dimension,
        tuple[str, ...],
        tuple[str, ...],
        frozenset[GradientTarget],
        bool,
    ]
]:
    """Normalize a backend matrix into the documented cell set.

    Rows are compared on dimension, operation set, storage modes, gradient
    targets, and callbacks. Devices, dtypes, and compute modes are asserted
    separately below because they are shared across every row.
    """
    return {
        (
            row.dimension,
            tuple(sorted(operation.value for operation in row.operations)),
            tuple(sorted(row.storage_modes)),
            row.gradient_targets,
            row.callbacks,
        )
        for row in capabilities.matrix
    }


def test_capability_matrix_matches_documented_cells() -> None:
    reference = backend_capabilities(BackendPreference.REFERENCE)
    native = backend_capabilities(BackendPreference.NATIVE)

    all_targets = frozenset(GradientTarget)
    model_targets = frozenset({GradientTarget.EPSILON, GradientTarget.SIGMA})
    forward_targets = model_targets | frozenset({GradientTarget.SOURCE})
    jvp_targets = model_targets | frozenset({GradientTarget.PERTURBATION})
    five_modes = ("auto", "cpu", "device", "disk", "none")

    assert _capability_cells(reference) == {
        (Dimension.TM2D, ("forward", "vjp"), five_modes, all_targets, True),
        (Dimension.TM2D, ("jvp",), five_modes, all_targets, False),
        (
            Dimension.TM2D,
            ("second_vjp",),
            ("cpu", "device", "disk"),
            all_targets,
            False,
        ),
        (Dimension.EM3D, ("forward", "vjp"), five_modes, all_targets, True),
        (
            Dimension.EM3D,
            ("jvp",),
            ("device", "none"),
            all_targets,
            False,
        ),
        (
            Dimension.EM3D,
            ("second_vjp",),
            ("device",),
            all_targets,
            False,
        ),
    }
    assert _capability_cells(native) == {
        (
            Dimension.TM2D,
            ("forward", "vjp"),
            five_modes,
            forward_targets,
            True,
        ),
        (Dimension.TM2D, ("jvp",), five_modes, jvp_targets, False),
        (
            Dimension.TM2D,
            ("second_vjp",),
            ("cpu", "device", "disk"),
            model_targets,
            False,
        ),
        (
            Dimension.EM3D,
            ("forward", "vjp"),
            five_modes,
            forward_targets,
            True,
        ),
        (
            Dimension.EM3D,
            ("jvp",),
            ("device", "none"),
            jvp_targets,
            False,
        ),
        (
            Dimension.EM3D,
            ("second_vjp",),
            ("device",),
            model_targets,
            False,
        ),
    }

    for capabilities in (reference, native):
        assert {row.dimension for row in capabilities.matrix} == set(Dimension)
        assert all(isinstance(row, BackendCapability) for row in capabilities.matrix)
        assert all(
            row.devices == frozenset({"cpu", "cuda"})
            for row in capabilities.matrix
        )
        assert all(
            row.dtypes == frozenset({torch.float32, torch.float64})
            for row in capabilities.matrix
        )
        assert all(
            row.compute_modes == frozenset({ComputeMode.NATIVE})
            for row in capabilities.matrix
        )


def test_em3d_jvp_is_in_the_capability_matrix() -> None:
    plan = compile_simulation_plan(
        operation="jvp",
        dimension="em3d",
        epsilon=torch.ones(3, 4, 4, 4),
        storage_mode="device",
    )
    assert select_backend(plan, native_available=True).selected is BackendPreference.NATIVE


def test_native_em3d_born_rejects_host_backed_storage() -> None:
    for storage_mode in ("cpu", "disk"):
        plan = compile_simulation_plan(
            operation="jvp",
            dimension="em3d",
            epsilon=torch.ones(3, 4, 4, 4),
            storage_mode=storage_mode,
        )
        with pytest.raises(NotImplementedError, match="does not support storage_mode"):
            select_backend(plan, native_available=True)


def test_em3d_born_auto_storage_is_rejected_by_the_matrix() -> None:
    # The public born3d API only accepts storage_mode="device" or "none", so
    # "auto" is not reachable by any row and is rejected outright.
    plan = compile_simulation_plan(
        operation="jvp",
        dimension="em3d",
        epsilon=torch.ones(3, 4, 4, 4),
        storage_mode="auto",
    )
    with pytest.raises(NotImplementedError, match="does not support storage_mode"):
        select_backend(plan, native_available=True)
    with pytest.raises(NotImplementedError, match="does not support storage_mode"):
        select_backend(plan, native_available=False)


def test_native_tm2d_full_hvp_requires_device_storage() -> None:
    for storage_mode in ("cpu", "disk"):
        plan = compile_simulation_plan(
            operation="second_vjp",
            dimension="tm2d",
            epsilon=torch.ones(4, 4),
            hessian_mode="full",
            storage_mode=storage_mode,
            python_backend="native",
        )
        with pytest.raises(
            NotImplementedError,
            match="full second_vjp currently requires storage_mode='device'",
        ):
            select_backend(plan, native_available=True)


def test_callbacks_are_rejected_for_non_forward_operations() -> None:
    plan = compile_simulation_plan(
        operation="jvp",
        dimension="tm2d",
        epsilon=torch.ones(4, 4),
        has_callbacks=True,
    )
    with pytest.raises(NotImplementedError, match="does not support callbacks"):
        select_backend(plan, native_available=True)


def test_execution_policy_is_the_shared_solver_dispatch_boundary() -> None:
    policy = compile_execution_policy(
        requested_backend=True,
        operation="forward",
        dimension="tm2d",
        epsilon=torch.ones(4, 4),
    )

    assert policy.use_python
    assert policy.dispatch_backend is True
    assert policy.compute_mode == "native"
    assert policy.storage_mode == "device"
