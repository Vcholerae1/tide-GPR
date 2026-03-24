import warnings

import pytest
import torch

import tide
from tide import backend_utils


def _tm_case():
    device = torch.device("cpu")
    dtype = torch.float32
    ny, nx = 8, 9
    nt = 10
    epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[ny // 2, nx // 2]]], device=device)
    receiver_location = torch.tensor([[[ny // 2, nx // 2 + 1]]], device=device)
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def _em3d_case():
    device = torch.device("cpu")
    dtype = torch.float32
    nz, ny, nx = 5, 6, 7
    nt = 8
    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 4]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        70e6, nt, 4e-11, peak_time=1.0 / 70e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def test_debye_tm_module_matches_functional():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case()
    dispersion = tide.DebyeDispersion(delta_epsilon=2.0, tau=5e-10)

    model = tide.MaxwellTM(epsilon, sigma, mu, grid_spacing=0.02)
    out_module = model(
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        dispersion=dispersion,
    )
    out_func = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        dispersion=dispersion,
    )

    for mod_out, fn_out in zip(out_module, out_func):
        torch.testing.assert_close(mod_out, fn_out)


def test_debye_zero_delta_matches_nondispersive():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case()
    out_ref = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
    )
    out_debye = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        dispersion=tide.DebyeDispersion(delta_epsilon=0.0, tau=5e-10),
    )

    for ref, got in zip(out_ref, out_debye):
        torch.testing.assert_close(ref, got)


def test_debye_single_pole_matches_explicit_pole_axis():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case()
    ny, nx = epsilon.shape
    scalar = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        dispersion=tide.DebyeDispersion(delta_epsilon=1.5, tau=5e-10),
    )
    explicit = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        dispersion=tide.DebyeDispersion(
            delta_epsilon=torch.full((1, ny, nx), 1.5, dtype=epsilon.dtype),
            tau=torch.full((1, ny, nx), 5e-10, dtype=epsilon.dtype),
        ),
    )

    for ref, got in zip(scalar, explicit):
        torch.testing.assert_close(ref, got)


def test_debye_requires_dt_smaller_than_tau():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case()
    with pytest.raises(ValueError, match="dt < min\\(tau\\)"):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=5e-10,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=1,
            python_backend=True,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )


def test_debye_tm_native_forward_matches_python():
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case()
    dispersion = tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10)
    out_py = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        dispersion=dispersion,
    )
    out_native = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=False,
        dispersion=dispersion,
    )

    for ref, got in zip(out_py, out_native):
        torch.testing.assert_close(ref, got, rtol=1e-4, atol=1e-5)


def test_debye_em3d_native_forward_matches_python():
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for 3D native Debye parity test")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _em3d_case()
    epsilon = epsilon.cuda()
    sigma = sigma.cuda()
    mu = mu.cuda()
    source_amplitude = source_amplitude.cuda()
    source_location = source_location.cuda()
    receiver_location = receiver_location.cuda()
    dispersion = tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10)
    out_py = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        dispersion=dispersion,
    )
    out_native = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=False,
        dispersion=dispersion,
    )

    for ref, got in zip(out_py, out_native):
        torch.testing.assert_close(ref, got, rtol=1e-4, atol=1e-5)


def test_debye_em3d_cpu_backend_falls_back_to_python():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _em3d_case()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=1,
            python_backend=False,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )
    assert any("3D Debye CPU backend is not enabled yet" in str(w.message) for w in caught)
    assert torch.isfinite(out[-1]).all()


def test_debye_falls_back_to_python_backend_for_gradients():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case()
    epsilon = epsilon.clone().detach().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=1,
            python_backend=False,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )
    assert any("Debye native backend currently supports forward inference only" in str(w.message) for w in caught)
    assert torch.isfinite(out[-1]).all()


def test_debye_callback_exposes_dispersion_and_polarization_tm():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case()
    seen = {}

    def callback(state: tide.CallbackState) -> None:
        if seen:
            return
        seen["model_names"] = state.model_names
        seen["wavefield_names"] = state.wavefield_names
        seen["dispersion"] = state.get_model("dispersion")
        seen["polarization_shape"] = tuple(state.get_wavefield("polarization", view="inner").shape)

    tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        forward_callback=callback,
        dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
    )

    assert "dispersion" in seen["model_names"]
    assert "polarization" in seen["wavefield_names"]
    assert isinstance(seen["dispersion"], tide.DebyeDispersion)
    assert len(seen["polarization_shape"]) == 3


def test_debye_callback_exposes_dispersion_and_polarization_3d():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _em3d_case()
    seen = {}

    def callback(state: tide.CallbackState) -> None:
        if seen:
            return
        seen["model_names"] = state.model_names
        seen["wavefield_names"] = state.wavefield_names
        seen["dispersion"] = state.get_model("dispersion")
        seen["polarization_shape"] = tuple(state.get_wavefield("polarization", view="inner").shape)

    out = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        python_backend=True,
        forward_callback=callback,
        dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
    )

    assert "dispersion" in seen["model_names"]
    assert "polarization" in seen["wavefield_names"]
    assert isinstance(seen["dispersion"], tide.DebyeDispersion)
    assert len(seen["polarization_shape"]) == 5
    assert torch.isfinite(out[-1]).all()
