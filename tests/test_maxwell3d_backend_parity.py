import torch

import tide


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
