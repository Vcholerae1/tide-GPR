import pytest
import torch

import tide


def _build_small_case(device: torch.device):
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 12
    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 4]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location, nt


def test_maxwell3d_available_from_tide():
    assert hasattr(tide, "maxwell3d")
    assert hasattr(tide, "Maxwell3D")


def test_maxwell3d_output_shape_and_order_cpu():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location, nt = (
        _build_small_case(device)
    )

    out = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=[2, 2, 2, 2, 2, 2],
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )
    assert len(out) == 19
    assert out[-1].shape == (nt, 1, 1)
    for field in out[:-1]:
        assert field.ndim == 4
        assert field.shape[0] == 1


def test_maxwell3d_component_validation():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location, _ = (
        _build_small_case(device)
    )
    with pytest.raises(ValueError):
        tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            source_component="bad",
            python_backend=True,
        )
    with pytest.raises(ValueError):
        tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            receiver_component="bad",
            python_backend=True,
        )


def test_maxwell3d_location_bounds():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location, _ = (
        _build_small_case(device)
    )
    bad_source = source_location.clone()
    bad_source[0, 0, 0] = epsilon.shape[0]
    with pytest.raises(RuntimeError):
        tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=bad_source,
            receiver_location=receiver_location,
            python_backend=True,
        )


def test_maxwell3d_requires_nt_if_no_source():
    device = torch.device("cpu")
    epsilon, sigma, mu, _, source_location, receiver_location, _ = _build_small_case(
        device
    )
    with pytest.raises(ValueError):
        tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=None,
            source_location=source_location,
            receiver_location=receiver_location,
            python_backend=True,
        )
