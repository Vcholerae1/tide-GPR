import ctypes
import tempfile
from contextlib import nullcontext
from importlib import import_module

import pytest
import torch

import tide
from tide import backend_utils

maxwell_module = import_module("tide.maxwell")


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


def _run_3d_grad(
    storage_mode: str, storage_path: str, *, stream: torch.cuda.Stream | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 10

    epsilon = torch.full(
        (nz, ny, nx), 4.0, device=device, dtype=dtype, requires_grad=True
    )
    sigma = torch.full_like(epsilon, 2e-4, requires_grad=True)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[2, 3, 2]]], dtype=torch.long, device=device
    )
    receiver_location = torch.tensor(
        [[[2, 3, 5]]], dtype=torch.long, device=device
    )
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)

    context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with context:
        receivers = tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=[0.03, 0.02, 0.02],
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=2,
            python_backend=False,
            storage_mode=storage_mode,
            storage_path=storage_path,
            storage_compression=False,
        )[-1]
        receivers.square().sum().backward()
    torch.cuda.synchronize()

    assert epsilon.grad is not None
    assert sigma.grad is not None
    return (
        epsilon.grad.detach().cpu(),
        sigma.grad.detach().cpu(),
        receivers.detach().cpu(),
    )


def test_maxwell3d_storage_backend_argtypes_include_stream_handles():
    forward_argtypes = backend_utils._template_argtypes(
        "maxwell_3d_forward_with_storage", "float"
    )
    backward_argtypes = backend_utils._template_argtypes("maxwell_3d_backward", "float")

    assert forward_argtypes[-2:] == [ctypes.c_void_p, ctypes.c_void_p]
    assert backward_argtypes[-2:] == [ctypes.c_void_p, ctypes.c_void_p]


def test_maxwell3d_forward_backend_argtypes_include_compute_stream_handle():
    forward_argtypes = backend_utils._template_argtypes("maxwell_3d_forward", "float")

    assert forward_argtypes[-1] == ctypes.c_void_p


def test_maxwell3d_snapshot_storage_modes_match():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Maxwell3D snapshot storage tests.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_dev, sig_dev, rec_dev = _run_3d_grad("device", storage_path)
        eps_cpu, sig_cpu, rec_cpu = _run_3d_grad("cpu", storage_path)
        eps_disk, sig_disk, rec_disk = _run_3d_grad("disk", storage_path)

    torch.testing.assert_close(rec_cpu, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rec_disk, rec_dev, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eps_cpu, eps_dev, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(eps_disk, eps_dev, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(sig_cpu, sig_dev, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(sig_disk, sig_dev, rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("storage_mode", ["cpu", "disk"])
def test_maxwell3d_host_backed_storage_matches_on_custom_current_stream(
    storage_mode: str,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for stream-aware Maxwell3D storage tests.")

    with tempfile.TemporaryDirectory() as storage_path:
        eps_base, sig_base, rec_base = _run_3d_grad(storage_mode, storage_path)
        custom_stream = torch.cuda.Stream()
        eps_stream, sig_stream, rec_stream = _run_3d_grad(
            storage_mode, storage_path, stream=custom_stream
        )

    torch.testing.assert_close(rec_stream, rec_base, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eps_stream, eps_base, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(sig_stream, sig_base, rtol=2e-4, atol=1e-5)


def _run_3d_forward(
    stream: torch.cuda.Stream | None = None,
    *,
    experimental_cuda_graph: bool = False,
    callback_frequency: int = 1,
    callback_steps: list[int] | None = None,
) -> torch.Tensor:
    device = torch.device("cuda")
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 10

    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, 2e-4)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 5]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)

    kwargs = {
        "experimental_cuda_graph": experimental_cuda_graph,
    }
    if callback_steps is not None:

        def _forward_callback(state):
            callback_steps.append(state.step)
            assert state.get_wavefield("Ey", view="inner").shape == (1, nz, ny, nx)

        kwargs["forward_callback"] = _forward_callback
        kwargs["callback_frequency"] = callback_frequency

    context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with context:
        receivers = tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=[0.03, 0.02, 0.02],
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=2,
            python_backend=False,
            storage_mode="none",
            **kwargs,
        )[-1]
    torch.cuda.synchronize()
    return receivers.detach().cpu()


def test_maxwell3d_plain_forward_matches_on_custom_current_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for stream-aware Maxwell3D tests.")

    rec_base = _run_3d_forward()
    rec_stream = _run_3d_forward(stream=torch.cuda.Stream())

    torch.testing.assert_close(rec_stream, rec_base, rtol=1e-5, atol=1e-6)


def test_maxwell3d_cuda_graph_matches_plain_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Maxwell3D CUDA Graph tests.")

    rec_plain = _run_3d_forward()
    rec_graph = _run_3d_forward(experimental_cuda_graph=True)

    torch.testing.assert_close(rec_graph, rec_plain, rtol=1e-5, atol=1e-6)


def test_maxwell3d_cuda_graph_matches_with_callback_chunks():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Maxwell3D CUDA Graph tests.")

    plain_steps: list[int] = []
    graph_steps: list[int] = []

    rec_plain = _run_3d_forward(callback_frequency=3, callback_steps=plain_steps)
    rec_graph = _run_3d_forward(
        experimental_cuda_graph=True,
        callback_frequency=3,
        callback_steps=graph_steps,
    )

    assert plain_steps == [0, 3, 6, 9]
    assert graph_steps == plain_steps
    torch.testing.assert_close(rec_graph, rec_plain, rtol=1e-5, atol=1e-6)


def test_maxwell3d_cuda_graph_matches_on_custom_current_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Maxwell3D CUDA Graph tests.")

    rec_base = _run_3d_forward(experimental_cuda_graph=True)
    rec_stream = _run_3d_forward(
        stream=torch.cuda.Stream(),
        experimental_cuda_graph=True,
    )

    torch.testing.assert_close(rec_stream, rec_base, rtol=1e-5, atol=1e-6)


def test_maxwell3d_cuda_graph_cache_persists_across_calls():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Maxwell3D CUDA Graph tests.")

    maxwell_module._clear_maxwell3d_cuda_graph_cache()
    assert maxwell_module._maxwell3d_cuda_graph_cache_size() == 0

    _run_3d_forward(experimental_cuda_graph=True, callback_frequency=3, callback_steps=[])
    size_after_first = maxwell_module._maxwell3d_cuda_graph_cache_size()
    assert size_after_first == 1

    _run_3d_forward(experimental_cuda_graph=True, callback_frequency=3, callback_steps=[])
    assert maxwell_module._maxwell3d_cuda_graph_cache_size() == size_after_first
