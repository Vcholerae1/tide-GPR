import ctypes
import tempfile
from contextlib import nullcontext

import pytest
import torch

import tide
from tide import backend_utils, maxwell as maxwell_mod
from tide.storage import STORAGE_CPU, STORAGE_DEVICE


def _run_grad(
    storage_mode: str,
    storage_path: str,
    *,
    storage_compression: bool,
    stream: torch.cuda.Stream | None = None,
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

    context = (
        torch.cuda.stream(stream) if stream is not None else nullcontext()
    )
    with context:
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


class _FakeStream:
    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle


def test_tm_storage_backend_argtypes_include_stream_handles():
    forward_argtypes = backend_utils._template_argtypes(
        "maxwell_tm_forward_with_storage", "float"
    )
    backward_argtypes = backend_utils._template_argtypes(
        "maxwell_tm_backward", "float"
    )

    assert forward_argtypes[-2:] == [ctypes.c_void_p, ctypes.c_void_p]
    assert backward_argtypes[-2:] == [ctypes.c_void_p, ctypes.c_void_p]


def test_tm_forward_backend_argtypes_include_compute_stream_handle():
    forward_argtypes = backend_utils._template_argtypes("maxwell_tm_forward", "float")

    assert forward_argtypes[-1] == ctypes.c_void_p


def test_make_tm_storage_streams_returns_zero_handles_on_cpu():
    compute_handle, storage_handle, keepalive = maxwell_mod._make_tm_storage_streams(
        torch.device("cpu"), STORAGE_CPU
    )

    assert compute_handle == 0
    assert storage_handle == 0
    assert keepalive == ()


def test_make_tm_storage_streams_uses_current_and_storage_streams(monkeypatch):
    compute_stream = _FakeStream(101)
    storage_stream = _FakeStream(202)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: compute_stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: storage_stream)

    compute_handle, storage_handle, keepalive = maxwell_mod._make_tm_storage_streams(
        torch.device("cuda"), STORAGE_CPU
    )

    assert compute_handle == 101
    assert storage_handle == 202
    assert keepalive == (compute_stream, storage_stream)


def test_make_tm_storage_streams_skips_storage_stream_for_device_mode(monkeypatch):
    compute_stream = _FakeStream(303)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: compute_stream)
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda device=None: pytest.fail("storage stream should not be created"),
    )

    compute_handle, storage_handle, keepalive = maxwell_mod._make_tm_storage_streams(
        torch.device("cuda"), STORAGE_DEVICE
    )

    assert compute_handle == 303
    assert storage_handle == 0
    assert keepalive == (compute_stream,)


def test_tm2d_rejects_removed_fp16_scaled_precision():
    epsilon = torch.ones(8, 8)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_amplitude = torch.zeros(1, 1, 4)
    source_location = torch.tensor([[[4, 4]]], dtype=torch.int64)
    receiver_location = torch.tensor([[[4, 5]]], dtype=torch.int64)

    with pytest.raises(ValueError, match="compute_precision must be 'default'"):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.01,
            dt=1e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            compute_precision="fp16_scaled",
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


def _run_tm_forward(stream: torch.cuda.Stream | None = None) -> torch.Tensor:
    device = torch.device("cuda")
    dtype = torch.float32
    ny, nx = 32, 32
    nt = 48

    epsilon = torch.full((ny, nx), 5.0, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, 1e-3)
    mu = torch.ones_like(epsilon)

    dt = 1e-11
    freq0 = 9e8
    wavelet = tide.ricker(freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device)
    source_amplitude = wavelet.view(1, 1, nt).repeat(2, 1, 1)
    source_location = torch.tensor(
        [[[ny // 2, nx // 2]], [[ny // 2, nx // 2]]],
        device=device,
        dtype=torch.int64,
    )
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2 + 3]], [[ny // 2, nx // 2 + 3]]],
        device=device,
        dtype=torch.int64,
    )

    context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with context:
        receivers = tide.maxwelltm(
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
            storage_mode="none",
        )[-1]
    torch.cuda.synchronize()
    return receivers.detach().cpu()

@pytest.mark.parametrize("storage_mode", ["cpu", "disk"])
@pytest.mark.parametrize("storage_compression", [False, True])
def test_snapshot_storage_host_backed_modes_match_on_custom_current_stream(
    storage_mode: str, storage_compression: bool
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for stream-aware snapshot storage tests.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_base, sig_base, rec_base = _run_grad(
            storage_mode, storage_path, storage_compression=storage_compression
        )
        custom_stream = torch.cuda.Stream()
        eps_stream, sig_stream, rec_stream = _run_grad(
            storage_mode,
            storage_path,
            storage_compression=storage_compression,
            stream=custom_stream,
        )

    torch.testing.assert_close(rec_stream, rec_base, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eps_stream, eps_base, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(sig_stream, sig_base, rtol=1e-4, atol=1e-5)


def test_tm_plain_forward_matches_on_custom_current_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for stream-aware forward tests.")

    rec_base = _run_tm_forward()
    rec_stream = _run_tm_forward(stream=torch.cuda.Stream())

    torch.testing.assert_close(rec_stream, rec_base, rtol=1e-5, atol=1e-6)
