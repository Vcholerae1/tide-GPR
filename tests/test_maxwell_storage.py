import tempfile

import pytest
import torch

import tide


def _run_grad(
    storage_mode: str,
    storage_path: str,
    *,
    storage_compression: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.float32

    ny, nx = 32, 32
    eps_r = 10.0
    conductivity = 1e-3

    epsilon = torch.full(
        (ny, nx), eps_r, device=device, dtype=dtype, requires_grad=True
    )
    sigma = torch.full_like(epsilon, conductivity, requires_grad=True)
    mu = torch.ones_like(epsilon)

    freq0 = 9e8
    dt = 1e-11
    nt = 64

    wavelet = tide.ricker(freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device)
    n_shots = 2
    source_amplitude = wavelet.view(1, 1, nt).repeat(n_shots, 1, 1)

    src_y, src_x = ny // 2, nx // 2
    rec_y, rec_x = ny // 2, nx // 2 + 4
    source_location = torch.tensor(
        [[[src_y, src_x]], [[src_y, src_x]]],
        device=device,
        dtype=torch.int64,
    )
    receiver_location = torch.tensor(
        [[[rec_y, rec_x]], [[rec_y, rec_x]]],
        device=device,
        dtype=torch.int64,
    )

    *_, receivers = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.005,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=8,
        save_snapshots=None,
        model_gradient_sampling_interval=2,
        storage_mode=storage_mode,
        storage_path=storage_path,
        storage_compression=storage_compression,
    )

    loss = receivers.square().sum()
    loss.backward()
    torch.cuda.synchronize()

    assert epsilon.grad is not None
    assert sigma.grad is not None

    return (
        epsilon.grad.detach().cpu(),
        sigma.grad.detach().cpu(),
        receivers.detach().cpu(),
    )


def test_snapshot_storage_modes_match():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for snapshot storage tests.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_dev, sig_dev, rec_dev = _run_grad("device", storage_path, storage_compression=False)
        eps_cpu, sig_cpu, rec_cpu = _run_grad("cpu", storage_path, storage_compression=False)
        eps_disk, sig_disk, rec_disk = _run_grad("disk", storage_path, storage_compression=False)

    torch.testing.assert_close(rec_cpu, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rec_disk, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eps_cpu, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eps_disk, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_cpu, sig_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_disk, sig_dev, rtol=1e-4, atol=1e-5)


def test_snapshot_storage_bf16_modes_match():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_dev, sig_dev, rec_dev = _run_grad("device", storage_path, storage_compression=True)
        eps_cpu, sig_cpu, rec_cpu = _run_grad("cpu", storage_path, storage_compression=True)
        eps_disk, sig_disk, rec_disk = _run_grad("disk", storage_path, storage_compression=True)

    torch.testing.assert_close(rec_cpu, rec_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rec_disk, rec_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eps_cpu, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eps_disk, eps_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_cpu, sig_dev, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_disk, sig_dev, rtol=1e-4, atol=1e-5)


def test_storage_mode_none_rejects_gradients():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test.")

    device = torch.device("cuda")
    dtype = torch.float32

    epsilon = torch.full((16, 16), 5.0, device=device, dtype=dtype, requires_grad=True)
    sigma = torch.full_like(epsilon, 1e-3, requires_grad=True)
    mu = torch.ones_like(epsilon)

    dt = 1e-11
    nt = 16
    freq0 = 9e8
    wavelet = tide.ricker(freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device)

    source_amplitude = wavelet.view(1, 1, nt)
    source_location = torch.tensor([[[8, 8]]], device=device, dtype=torch.int64)
    receiver_location = torch.tensor([[[8, 9]]], device=device, dtype=torch.int64)

    with pytest.raises(ValueError, match="storage_mode='none'"):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.005,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            stencil=2,
            pml_width=4,
            storage_mode="none",
        )
