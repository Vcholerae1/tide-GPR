import pytest
import torch

import tide


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


@pytest.mark.parametrize("device", _devices())
def test_maxwell3d_python_forward_smoke(device: torch.device):
    dtype = torch.float32
    nz, ny, nx = 8, 8, 10
    nt = 14
    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    epsilon[4:, :, :] = 6.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 4, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 4, 5]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        100e6, nt, 4e-11, peak_time=1.0 / 100e6, dtype=dtype, device=device
    ).view(1, 1, nt)

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
    receiver = out[-1]
    assert receiver.shape == (nt, 1, 1)
    assert torch.isfinite(receiver).all()
    assert receiver.abs().max() > 0


@pytest.mark.parametrize("device", _devices())
def test_maxwell3d_python_forward_long_nt_stability(device: torch.device):
    """Long propagation should remain finite and not show late-time blow-up."""
    dtype = torch.float32
    nz = ny = nx = 20
    nt = 1000
    dt = 1.6e-11
    dz = dy = dx = 0.01

    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[3, ny // 2, nx // 2]]],
        dtype=torch.long,
        device=device,
    )
    receiver_location = torch.tensor(
        [[[3, ny // 2, nx // 2 + 4]]],
        dtype=torch.long,
        device=device,
    )
    source_amplitude = tide.ricker(
        160e6,
        nt,
        dt,
        peak_time=1.2 / 160e6,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)

    wavefield_peaks: list[float] = []

    def cb(state):
        ey = state.get_wavefield("Ey", view="inner")
        wavefield_peaks.append(float(ey.abs().max().detach().cpu()))

    out = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=[dz, dy, dx],
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=6,
        stencil=4,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
        forward_callback=cb,
        callback_frequency=5,
    )

    receiver = out[-1]
    assert receiver.shape == (nt, 1, 1)
    assert torch.isfinite(receiver).all()
    assert receiver.abs().max() > 0

    peaks = torch.tensor(wavefield_peaks)
    assert peaks.numel() > 10
    assert torch.isfinite(peaks).all()
    assert peaks.max() > 0
    late_ratio = peaks[-20:].max() / peaks.max()
    assert late_ratio < 1e-2
