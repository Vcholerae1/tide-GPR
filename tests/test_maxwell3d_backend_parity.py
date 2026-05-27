import torch
import pytest

import tide
from tide import backend_utils


def _case(device: torch.device):
    dtype = torch.float32
    nz, ny, nx = 6, 6, 7
    nt = 10
    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[2, 2, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 2, 4]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        90e6, nt, 4e-11, peak_time=1.0 / 90e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def test_maxwell3d_backend_parity_via_fallback():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _case(
        device
    )

    out_py = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )
    out_backend = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=False,
    )

    for a, b in zip(out_py, out_backend):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    ("n_threads", "spatial_launch"),
    [(0, False), (128, False), (256, False), (256, True)],
)
def test_maxwell3d_native_cuda_matches_python_without_callback(
    n_threads, spatial_launch, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D CUDA parity test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D CUDA parity test.")

    device = torch.device("cuda")
    if spatial_launch:
        monkeypatch.setenv("TIDE_EM3D_SPATIAL_LAUNCH", "1")
    else:
        monkeypatch.delenv("TIDE_EM3D_SPATIAL_LAUNCH", raising=False)

    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _case(
        device
    )

    out_py = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )
    out_backend = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=False,
        n_threads=n_threads,
    )

    torch.testing.assert_close(out_backend[-1], out_py[-1], atol=1e-4, rtol=1e-4)
