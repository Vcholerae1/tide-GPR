"""Contracts for the structured Maxwell operator API."""

import torch

import tide
from numerical_utils import make_maxwell3d_example, make_tm2d_example


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

    assert result.epsilon is not None
    assert result.epsilon.shape == case.epsilon.shape
    assert result.sigma is None


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
