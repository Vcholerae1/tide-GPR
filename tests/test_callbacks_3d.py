import torch

import tide
from tide.callbacks import CallbackState


def test_callback_state_views_for_3d():
    device = torch.device("cpu")
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 8
    pml_width = [2, 1, 1, 2, 1, 2]

    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 5]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)

    seen = {"calls": 0}

    def cb(state: CallbackState):
        seen["calls"] += 1
        ey_inner = state.get_wavefield("Ey", view="inner")
        ey_pml = state.get_wavefield("Ey", view="pml")
        ey_full = state.get_wavefield("Ey", view="full")

        assert ey_inner.shape == (1, nz, ny, nx)
        assert ey_pml.shape == (
            1,
            nz + pml_width[0] + pml_width[1],
            ny + pml_width[2] + pml_width[3],
            nx + pml_width[4] + pml_width[5],
        )
        assert ey_full.ndim == 4
        assert ey_full.shape[-3] >= ey_pml.shape[-3]
        assert ey_full.shape[-2] >= ey_pml.shape[-2]
        assert ey_full.shape[-1] >= ey_pml.shape[-1]

    tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=pml_width,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
        forward_callback=cb,
        callback_frequency=2,
    )
    assert seen["calls"] > 0
