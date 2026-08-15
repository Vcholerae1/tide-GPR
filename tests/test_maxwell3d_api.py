import pytest
import torch

import tide
from numerical_utils import MaxwellExample, make_maxwell3d_example


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


def test_maxwell3d_available_from_tide():
    assert hasattr(tide, "Maxwell3D")
    assert hasattr(tide, "LinearizedMaxwell3D")


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
