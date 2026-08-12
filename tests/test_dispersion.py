import warnings

import pytest
import torch

import tide
from tide import backend_utils
from numerical_utils import make_maxwell3d_example, make_tm2d_example


def _tm_example():
    return make_tm2d_example(
        shape=(8, 9),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=80e6,
        source_location=(4, 4),
        receiver_locations=((4, 5),),
        pml_width=1,
        python_backend=True,
    )


def _maxwell3d_example(device: torch.device | str = "cpu"):
    return make_maxwell3d_example(
        shape=(5, 6, 7),
        nt=8,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=70e6,
        device=device,
        source_location=(2, 3, 2),
        receiver_locations=((2, 3, 4),),
        pml_width=1,
    )


def test_debye_tm_operator_matches_reference_kernel():
    example = _tm_example()
    dispersion = tide.DebyeDispersion(delta_epsilon=2.0, tau=5e-10)
    operator = tide.MaxwellTM(
        tide.Discretization(
            example.grid_spacing,
            example.dt,
            boundary=tide.CPML(example.pml_width),
        ),
        tide.Experiment(
            tide.Acquisition(example.source_location, example.receiver_location),
            example.source_amplitude,
        ),
        execution=tide.ExecutionOptions(backend=tide.BackendPreference.REFERENCE),
    )
    actual = operator(
        tide.EMModel(
            example.epsilon,
            example.sigma,
            example.mu,
            dispersion=dispersion,
        )
    )
    expected = example.run(dispersion=dispersion)
    torch.testing.assert_close(actual.receiver_data, expected[-1])


def test_debye_zero_delta_matches_nondispersive():
    example = _tm_example()
    reference = example.run()
    actual = example.run(dispersion=tide.DebyeDispersion(delta_epsilon=0.0, tau=5e-10))
    for reference_output, actual_output in zip(reference, actual, strict=True):
        torch.testing.assert_close(reference_output, actual_output)


def test_debye_single_pole_matches_explicit_pole_axis():
    example = _tm_example()
    ny, nx = example.epsilon.shape
    scalar = example.run(dispersion=tide.DebyeDispersion(delta_epsilon=1.5, tau=5e-10))
    explicit = example.run(
        dispersion=tide.DebyeDispersion(
            delta_epsilon=torch.full(
                (1, ny, nx),
                1.5,
                dtype=example.epsilon.dtype,
            ),
            tau=torch.full(
                (1, ny, nx),
                5e-10,
                dtype=example.epsilon.dtype,
            ),
        )
    )
    for scalar_output, explicit_output in zip(scalar, explicit, strict=True):
        torch.testing.assert_close(scalar_output, explicit_output)


def test_debye_requires_dt_smaller_than_tau():
    example = _tm_example()
    with pytest.raises(ValueError, match="dt < min\\(tau\\)"):
        example.run(
            dt=5e-10,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )


def test_debye_tm_native_forward_matches_python():
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")
    example = _tm_example()
    dispersion = tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10)
    reference = example.run(python_backend=True, dispersion=dispersion)
    actual = example.run(python_backend=False, dispersion=dispersion)
    for reference_output, actual_output in zip(reference, actual, strict=True):
        torch.testing.assert_close(
            reference_output,
            actual_output,
            rtol=1e-4,
            atol=1e-5,
        )


def test_debye_em3d_native_forward_matches_python():
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for 3D native Debye parity test")
    example = _maxwell3d_example("cuda")
    dispersion = tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10)
    reference = example.run(python_backend=True, dispersion=dispersion)
    actual = example.run(python_backend=False, dispersion=dispersion)
    for reference_output, actual_output in zip(reference, actual, strict=True):
        torch.testing.assert_close(
            reference_output,
            actual_output,
            rtol=1e-4,
            atol=1e-5,
        )


def test_debye_em3d_cpu_backend_falls_back_to_python():
    example = _maxwell3d_example()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = example.run(
            python_backend=False,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )
    assert any(
        "3D Debye CPU backend is not enabled yet" in str(w.message) for w in caught
    )
    assert torch.isfinite(output[-1]).all()


def test_debye_gradient_fallback_routes_through_policy():
    example = _tm_example()
    epsilon = example.epsilon.clone().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = example.run(
            epsilon=epsilon,
            python_backend=False,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )
    assert torch.isfinite(output[-1]).all()
    assert output[-1].requires_grad
    assert not any(
        "Debye native backend currently supports forward inference only"
        in str(w.message)
        for w in caught
    )


def test_debye_callback_exposes_dispersion_and_polarization_tm():
    example = _tm_example()
    seen = {}

    def callback(state: tide.CallbackState) -> None:
        if seen:
            return
        seen["model_names"] = state.model_names
        seen["wavefield_names"] = state.wavefield_names
        seen["dispersion"] = state.get_model("dispersion")
        seen["polarization_shape"] = tuple(
            state.get_wavefield("polarization", view="inner").shape
        )

    example.run(
        forward_callback=callback,
        dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
    )
    assert "dispersion" in seen["model_names"]
    assert "polarization" in seen["wavefield_names"]
    assert isinstance(seen["dispersion"], tide.DebyeDispersion)
    assert len(seen["polarization_shape"]) == 3


def test_debye_callback_exposes_dispersion_and_polarization_3d():
    example = _maxwell3d_example()
    seen = {}

    def callback(state: tide.CallbackState) -> None:
        if seen:
            return
        seen["model_names"] = state.model_names
        seen["wavefield_names"] = state.wavefield_names
        seen["dispersion"] = state.get_model("dispersion")
        seen["polarization_shape"] = tuple(
            state.get_wavefield("polarization", view="inner").shape
        )

    output = example.run(
        python_backend=True,
        forward_callback=callback,
        dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
    )
    assert "dispersion" in seen["model_names"]
    assert "polarization" in seen["wavefield_names"]
    assert isinstance(seen["dispersion"], tide.DebyeDispersion)
    assert len(seen["polarization_shape"]) == 5
    assert torch.isfinite(output[-1]).all()
