import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fp8_storage_hw_only_sm89_roundtrip_and_grad():
    # This exercises the C/CUDA "with storage" path with FP8 compression.
    # It should not fall back to any software conversion and must work on SM89+.
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        pytest.skip(f"Requires SM89+, got {major}.{minor}")

    import tide

    device = torch.device("cuda")
    ny, nx = 64, 64
    dx = 0.02
    dt = 1e-11
    nt = 64

    epsilon = (torch.ones(ny, nx, device=device, dtype=torch.float32) * 4.0).requires_grad_(
        True
    )
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[ny // 2, nx // 2]]], device=device, dtype=torch.long)
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2 + 1]]], device=device, dtype=torch.long
    )
    wavelet = tide.ricker(2e8, nt, dt, peak_time=1.0 / 2e8).to(device)
    source_amplitude = wavelet.view(1, 1, nt)

    out = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=10,
        save_snapshots=True,
        model_gradient_sampling_interval=2,
        storage_mode="device",
        storage_compression="fp8",
    )

    rec = out[-1]
    loss = (rec**2).mean()
    loss.backward()
    torch.cuda.synchronize()

    assert epsilon.grad is not None
    assert torch.isfinite(epsilon.grad).all()
