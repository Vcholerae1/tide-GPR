import pytest
import torch

import tide
from tide import backend_utils


def _epsilon_grad(*, execution_backend: str) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.float32
    ny, nx = 36, 34
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, ny, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype),
        indexing="ij",
    )
    epsilon = (3.0 + 0.4 * torch.exp(-4.0 * (xx.square() + yy.square()))).detach()
    epsilon.requires_grad_(True)
    sigma = torch.full_like(epsilon, 1.0e-3)
    mu = torch.ones_like(epsilon)

    nt = 48
    dt = 1.0e-11
    wavelet = tide.ricker(
        8.0e8, nt, dt, peak_time=1.0 / 8.0e8, dtype=dtype, device=device
    )
    source_amplitude = wavelet.view(1, 1, nt).repeat(2, 1, 1)
    source_location = torch.tensor(
        [[[18, 12]], [[18, 21]]], device=device, dtype=torch.int64
    )
    receiver_location = torch.tensor(
        [[[18, 20], [16, 18]], [[18, 13], [20, 16]]],
        device=device,
        dtype=torch.int64,
    )

    *_, receivers = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.006,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=6,
        save_snapshots=True,
        model_gradient_sampling_interval=3,
        storage_mode="device",
        execution_backend=execution_backend,
    )
    loss = receivers.square().sum()
    loss.backward()
    assert epsilon.grad is not None
    return receivers.detach().cpu(), epsilon.grad.detach().cpu()


def _material_grads(
    *,
    execution_backend: str,
    gradient_interval: int = 1,
    storage_compression: bool | str = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.float32
    ny, nx = 34, 32
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, ny, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype),
        indexing="ij",
    )
    epsilon = (3.0 + 0.35 * torch.exp(-3.0 * (xx.square() + yy.square()))).detach()
    sigma = (8.0e-4 + 2.0e-4 * torch.exp(-5.0 * ((xx - 0.2).square() + yy.square()))).detach()
    epsilon.requires_grad_(True)
    sigma.requires_grad_(True)
    mu = torch.ones_like(epsilon)

    nt = 36
    dt = 1.0e-11
    wavelet = tide.ricker(
        8.0e8, nt, dt, peak_time=1.0 / 8.0e8, dtype=dtype, device=device
    )
    source_amplitude = wavelet.view(1, 1, nt).repeat(2, 1, 1)
    source_location = torch.tensor(
        [[[17, 11]], [[17, 20]]], device=device, dtype=torch.int64
    )
    receiver_location = torch.tensor(
        [[[17, 20], [15, 17]], [[17, 12], [20, 16]]],
        device=device,
        dtype=torch.int64,
    )

    *_, receivers = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.006,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=6,
        save_snapshots=True,
        model_gradient_sampling_interval=gradient_interval,
        storage_mode="device",
        storage_compression=storage_compression,
        execution_backend=execution_backend,
    )
    loss = receivers.square().sum()
    loss.backward()
    assert epsilon.grad is not None
    assert sigma.grad is not None
    return (
        receivers.detach().cpu(),
        epsilon.grad.detach().cpu(),
        sigma.grad.detach().cpu(),
    )


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_direct_epsilon_grad_matches_standard_cuda():
    receivers_standard, grad_standard = _epsilon_grad(execution_backend="standard")
    receivers_direct, grad_direct = _epsilon_grad(
        execution_backend="direct_epsilon_grad"
    )

    torch.testing.assert_close(receivers_direct, receivers_standard, rtol=1e-6, atol=2e-5)
    torch.testing.assert_close(grad_direct, grad_standard, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_direct_material_grad_matches_standard_cuda():
    receivers_standard, eps_standard, sigma_standard = _material_grads(
        execution_backend="standard"
    )
    receivers_direct, eps_direct, sigma_direct = _material_grads(
        execution_backend="direct_material_grad"
    )

    torch.testing.assert_close(receivers_direct, receivers_standard, rtol=1e-6, atol=2e-5)
    torch.testing.assert_close(eps_direct, eps_standard, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(sigma_direct, sigma_standard, rtol=3e-4, atol=3e-5)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_direct_material_grad_ecurl_matches_standard_bf16_cuda():
    receivers_standard, eps_standard, sigma_standard = _material_grads(
        execution_backend="standard",
        gradient_interval=3,
        storage_compression="bf16",
    )
    receivers_direct, eps_direct, sigma_direct = _material_grads(
        execution_backend="direct_material_grad_ecurl",
        gradient_interval=3,
        storage_compression="bf16",
    )

    torch.testing.assert_close(receivers_direct, receivers_standard, rtol=1e-6, atol=2e-5)
    torch.testing.assert_close(eps_direct, eps_standard, rtol=5e-4, atol=5e-5)
    torch.testing.assert_close(sigma_direct, sigma_standard, rtol=5e-4, atol=5e-5)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_direct_material_endpoint_grad_runs_cuda():
    receivers_standard, eps_standard, sigma_standard = _material_grads(
        execution_backend="standard"
    )
    receivers_endpoint, eps_endpoint, sigma_endpoint = _material_grads(
        execution_backend="direct_material_endpoint_grad",
        gradient_interval=2,
    )

    torch.testing.assert_close(
        receivers_endpoint, receivers_standard, rtol=1e-6, atol=2e-5
    )
    assert torch.isfinite(eps_endpoint).all()
    assert torch.isfinite(sigma_endpoint).all()
    assert torch.nn.functional.cosine_similarity(
        eps_endpoint.flatten(), eps_standard.flatten(), dim=0
    ) > 0.9
    assert torch.nn.functional.cosine_similarity(
        sigma_endpoint.flatten(), sigma_standard.flatten(), dim=0
    ) > 0.99
