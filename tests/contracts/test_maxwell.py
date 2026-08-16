from __future__ import annotations

import pytest
import tide
import torch
from dataclasses import replace
from numerical_utils import make_maxwell3d_example, make_tm2d_example, MaxwellExample
from tide import backend_utils
from tide.core import (
    BackendCapabilities,
    BackendCapability,
    BackendPreference,
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

# --- test_edge_cases.py ---

"""Tests for edge cases and error handling."""


class TestEdgeCaseGridSizes:
    """Tests for edge cases related to grid sizes."""

    def test_small_grid_cpu(self):
        """Test with very small grid size on CPU."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 4, 4
        nt = 5

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype)
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Should not raise an error for small grid
        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=1,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_single_cell_grid_cpu(self):
        """Test with single cell grid (minimum viable)."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 1, 1
        nt = 3

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype)
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[0, 0]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[0, 0]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Should handle single cell grid
        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=0,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()


class TestEdgeCaseSourceReceiver:
    """Tests for edge cases related to sources and receivers."""

    def test_source_at_boundary(self):
        """Test with source at domain boundary."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        # Source at corner
        source_locations = torch.tensor([[[0, 0]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=0,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_single_time_step(self):
        """Test with single time step."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 8, 8
        nt = 1

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=1,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()
        assert out.shape[0] == nt


class TestEdgeCasePML:
    """Tests for edge cases related to PML."""

    def test_large_pml(self):
        """Test with large PML width."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 12
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Large PML (almost half the domain)
        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=5,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()


# --- test_api_wrappers.py ---

"""Contracts for the structured Maxwell operator API."""


def _tm_case():
    return make_tm2d_example(
        shape=(12, 14),
        nt=20,
        grid_spacing=0.02,
        dt=2e-11,
        frequency=100e6,
        source_location=(4, 6),
        receiver_locations=((4, 8),),
        pml_width=2,
        python_backend=True,
    )


def _tm_operator(case) -> tide.MaxwellTM:
    return tide.MaxwellTM(
        tide.Discretization(
            case.grid_spacing,
            case.dt,
            boundary=tide.CPML(case.pml_width),
        ),
        tide.Experiment(
            tide.Acquisition(case.source_location, case.receiver_location),
            case.source_amplitude,
        ),
        execution=tide.ExecutionOptions(backend=tide.BackendPreference.REFERENCE),
    )


def _model(case) -> tide.EMModel:
    return tide.EMModel(case.epsilon, case.sigma, case.mu)


def test_maxwelltm_returns_named_result_matching_reference_kernel() -> None:
    case = _tm_case()
    operator = _tm_operator(case)

    actual = operator(_model(case))
    expected = case.run(python_backend=True)

    torch.testing.assert_close(actual.receiver_data, expected[-1])
    torch.testing.assert_close(actual.final_state.Ey, expected[0])
    assert isinstance(actual, tide.ForwardResult)
    assert isinstance(actual.final_state, tide.TMState)


def test_linearized_tm_jvp_matches_reference_tangent_kernel() -> None:
    case = _tm_case()
    operator = _tm_operator(case)
    direction = tide.EMDirection(epsilon=torch.full_like(case.epsilon, 0.05))

    with operator.linearize(_model(case)) as linearized:
        actual = linearized.jvp(direction)
    expected = case.run_born(depsilon=direction.epsilon, python_backend=True)

    torch.testing.assert_close(actual.receiver_data, expected[-1])
    torch.testing.assert_close(actual.final_state.Ey, expected[7])


def test_linearized_tm_vjp_satisfies_discrete_adjoint_identity() -> None:
    case = _tm_case()
    operator = _tm_operator(case)
    direction_tensor = torch.randn_like(case.epsilon)
    direction = tide.EMDirection(epsilon=direction_tensor)

    with operator.linearize(_model(case), targets=("epsilon",)) as linearized:
        tangent = linearized.jvp(direction).receiver_data
        cotangent = torch.randn_like(tangent)
        gradient = linearized.vjp(cotangent)

    assert gradient.epsilon is not None
    torch.testing.assert_close(
        (tangent * cotangent).sum(),
        (direction_tensor * gradient.epsilon).sum(),
        rtol=2e-5,
        atol=2e-7,
    )


def test_receiver_objective_composes_gauss_newton_hvp() -> None:
    case = _tm_case()
    operator = _tm_operator(case)
    direction = tide.EMDirection(epsilon=torch.randn_like(case.epsilon))

    with operator.linearize(_model(case), targets=("epsilon",)) as linearized:
        observed = torch.zeros_like(linearized.primal.receiver_data)
        objective = tide.workflow.ReceiverObjective(observed)
        result = objective.hvp(linearized, direction, mode="gauss_newton")
        expected = linearized.vjp(linearized.jvp(direction).receiver_data)

    torch.testing.assert_close(result.epsilon, expected.epsilon)
    assert result.sigma is expected.sigma is None


def test_maxwell3d_returns_named_result_matching_reference_kernel() -> None:
    case = make_maxwell3d_example(
        shape=(6, 7, 8),
        nt=10,
        grid_spacing=(0.03, 0.02, 0.02),
        dt=4e-11,
        frequency=80e6,
        source_location=(2, 3, 2),
        receiver_locations=((2, 3, 4),),
        pml_width=(2, 2, 2, 2, 2, 2),
        python_backend=True,
    )
    operator = tide.Maxwell3D(
        tide.Discretization(
            case.grid_spacing,
            case.dt,
            boundary=tide.CPML(case.pml_width),
        ),
        tide.Experiment(
            tide.Acquisition(case.source_location, case.receiver_location),
            case.source_amplitude,
            source_component=case.source_component,
            receiver_component=case.receiver_component,
        ),
        execution=tide.ExecutionOptions(backend=tide.BackendPreference.REFERENCE),
    )

    actual = operator(_model(case))
    expected = case.run(python_backend=True)

    torch.testing.assert_close(actual.receiver_data, expected[-1])
    torch.testing.assert_close(actual.final_state.Ey, expected[1])
    assert isinstance(actual.final_state, tide.EM3DState)


# --- test_maxwell3d_api.py ---


def _example(device: torch.device) -> MaxwellExample:
    return make_maxwell3d_example(
        shape=(6, 7, 8),
        nt=12,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        frequency=80e6,
        device=device,
        source_location=(2, 3, 2),
        receiver_locations=((2, 3, 4),),
        pml_width=[2, 2, 2, 2, 2, 2],
        python_backend=True,
    )


def test_maxwell3d_output_shape_and_order_cpu():
    example = _example(torch.device("cpu"))
    output = example.run()
    assert len(output) == 19
    assert output[-1].shape == (example.source_amplitude.shape[-1], 1, 1)
    for field in output[:-1]:
        assert field.ndim == 4
        assert field.shape[0] == 1


def test_linearized_maxwell3d_matches_reference_jvp_cpu():
    example = _example(torch.device("cpu"))
    depsilon = torch.full_like(example.epsilon, 0.05)
    operator = tide.Maxwell3D(
        tide.Discretization(
            tuple(example.grid_spacing),
            example.dt,
            boundary=tide.CPML(tuple(example.pml_width)),
        ),
        tide.Experiment(
            tide.Acquisition(example.source_location, example.receiver_location),
            example.source_amplitude,
            source_component=example.source_component,
            receiver_component=example.receiver_component,
        ),
        execution=tide.ExecutionOptions(backend=tide.BackendPreference.REFERENCE),
    )
    with operator.linearize(
        tide.EMModel(example.epsilon, example.sigma, example.mu)
    ) as linearized:
        actual = linearized.jvp(tide.EMDirection(epsilon=depsilon))
    expected = example.run_born(depsilon=depsilon, python_backend=True)
    torch.testing.assert_close(actual.receiver_data, expected[-1])


def test_maxwell3d_component_validation():
    example = _example(torch.device("cpu"))
    with pytest.raises(ValueError):
        example.run(source_component="bad")
    with pytest.raises(ValueError):
        example.run(receiver_component="bad")


def test_maxwell3d_location_bounds():
    example = _example(torch.device("cpu"))
    bad_source = example.source_location.clone()
    bad_source[0, 0, 0] = example.epsilon.shape[0]
    with pytest.raises(RuntimeError):
        example.run(source_location=bad_source)


def test_maxwell3d_requires_nt_if_no_source():
    example = _example(torch.device("cpu"))
    with pytest.raises(ValueError):
        example.run(source_amplitude=None)


# --- test_core_plan.py ---


def test_compile_plan_normalizes_legacy_options() -> None:
    epsilon = torch.ones(4, 5, dtype=torch.float64)
    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        sigma=torch.zeros_like(epsilon),
        mu=torch.ones_like(epsilon),
        python_backend="eager",
        storage_mode="DEVICE",
        storage_chunk_steps=3,
    )

    assert plan.dimension is Dimension.TM2D
    assert plan.backend is BackendPreference.REFERENCE
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
    with pytest.raises(
        NotImplementedError, match="does not support gradients with dispersion"
    ):
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
            row.devices == frozenset({"cpu", "cuda"}) for row in capabilities.matrix
        )
        assert all(
            row.dtypes == frozenset({torch.float32, torch.float64})
            for row in capabilities.matrix
        )


def test_em3d_jvp_is_in_the_capability_matrix() -> None:
    plan = compile_simulation_plan(
        operation="jvp",
        dimension="em3d",
        epsilon=torch.ones(3, 4, 4, 4),
        storage_mode="device",
    )
    assert (
        select_backend(plan, native_available=True).selected is BackendPreference.NATIVE
    )


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
    assert policy.storage_mode == "device"
