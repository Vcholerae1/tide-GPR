import pytest
import torch

import tide


@pytest.mark.parametrize("storage_mode", ["device", "cpu", "disk", "auto"])
def test_maxwell3d_storage_modes_are_accepted_in_backend_path(storage_mode: str):
    device = torch.device("cpu")
    dtype = torch.float32
    nz, ny, nx = 5, 6, 7
    nt = 8
    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 4]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        70e6, nt, 4e-11, peak_time=1.0 / 70e6, dtype=dtype, device=device
    ).view(1, 1, nt)

    out = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        python_backend=False,
        storage_mode=storage_mode,
        storage_compression=False,
    )
    assert out[-1].shape == (nt, 1, 1)
