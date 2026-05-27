import pytest
import torch

import tide


def _tm2d_case(device: torch.device) -> tuple[torch.Tensor, ...]:
    dtype = torch.float32
    ny, nx, nt = 24, 30, 64
    dx = 0.02
    dt = 4e-11
    y = torch.arange(ny, device=device, dtype=dtype)[:, None]
    x = torch.arange(nx, device=device, dtype=dtype)[None, :]
    blob = torch.exp(-(((y - 14) / 4) ** 2 + ((x - 18) / 5) ** 2) * 0.5)
    epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.ones_like(epsilon) * 1e-4
    epsilon_true = epsilon + 0.35 * blob
    sigma_true = sigma + 2e-4 * blob
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[ny // 4, nx // 3]]], device=device)
    receiver_location = torch.tensor(
        [[[ny // 4, nx // 2], [ny // 4, 2 * nx // 3]]], device=device
    )
    wavelet = tide.ricker(
        180e6,
        nt,
        dt,
        peak_time=1.0 / 180e6,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, nt)
    return (
        epsilon,
        sigma,
        epsilon_true,
        sigma_true,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
        torch.tensor(dx, device=device),
        torch.tensor(dt, device=device),
    )


def _gradient(
    *,
    python_backend: bool,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    epsilon_true: torch.Tensor,
    sigma_true: torch.Tensor,
    mu: torch.Tensor,
    source_amplitude: torch.Tensor,
    source_location: torch.Tensor,
    receiver_location: torch.Tensor,
    dx: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        observed = tide.maxwelltm(
            epsilon_true,
            sigma_true,
            mu,
            dx,
            dt,
            source_amplitude,
            source_location,
            receiver_location,
            pml_width=4,
            stencil=2,
            python_backend=python_backend,
            storage_mode="none",
        )[-1]

    eps = epsilon.detach().clone().requires_grad_(True)
    sig = sigma.detach().clone().requires_grad_(True)
    predicted = tide.maxwelltm(
        eps,
        sig,
        mu,
        dx,
        dt,
        source_amplitude,
        source_location,
        receiver_location,
        pml_width=4,
        stencil=2,
        model_gradient_sampling_interval=1,
        save_snapshots=True,
        python_backend=python_backend,
        storage_mode="device",
    )[-1]
    (0.5 * (predicted - observed).square().sum()).backward()
    assert eps.grad is not None
    assert sig.grad is not None
    return eps.grad.detach(), sig.grad.detach()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av = a.reshape(-1).double()
    bv = b.reshape(-1).double()
    return (av @ bv) / (av.norm() * bv.norm())


def _relative_error(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    floor = torch.tensor(1e-30, device=lhs.device, dtype=lhs.dtype)
    denom = torch.maximum(torch.maximum(lhs.abs(), rhs.abs()), floor)
    return (lhs - rhs).abs() / denom


def test_tm2d_cuda_coeff_backward_default_matches_python_direction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TM2D native coeff-gradient test.")

    device = torch.device("cuda")
    (
        epsilon,
        sigma,
        epsilon_true,
        sigma_true,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
        dx_tensor,
        dt_tensor,
    ) = _tm2d_case(device)
    dx = float(dx_tensor.item())
    dt = float(dt_tensor.item())

    reference_eps, reference_sig = _gradient(
        python_backend=True,
        epsilon=epsilon,
        sigma=sigma,
        epsilon_true=epsilon_true,
        sigma_true=sigma_true,
        mu=mu,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        dx=dx,
        dt=dt,
    )
    native_eps, native_sig = _gradient(
        python_backend=False,
        epsilon=epsilon,
        sigma=sigma,
        epsilon_true=epsilon_true,
        sigma_true=sigma_true,
        mu=mu,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        dx=dx,
        dt=dt,
    )

    assert _cosine(reference_eps, native_eps) > 0.98
    assert _cosine(reference_sig, native_sig) > 0.98


def test_tm2d_cuda_coeff_backward_default_dot_product_is_close_without_pml():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TM2D native coeff-gradient test.")

    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.float64
    ny, nx, nt = 24, 30, 64
    dx = 0.02
    dt = 4e-11
    y = torch.arange(ny, device=device, dtype=dtype)[:, None]
    x = torch.arange(nx, device=device, dtype=dtype)[None, :]
    blob = torch.exp(-(((y - 14) / 4) ** 2 + ((x - 18) / 5) ** 2) * 0.5)
    epsilon = (torch.ones(ny, nx, device=device, dtype=dtype) * 4.0 + 0.1 * blob)
    sigma = torch.ones_like(epsilon) * 1e-4
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[ny // 4, nx // 3]]], device=device)
    receiver_location = torch.tensor(
        [[[ny // 4, nx // 2], [ny // 4, 2 * nx // 3]]], device=device
    )
    source_amplitude = tide.ricker(
        180e6,
        nt,
        dt,
        peak_time=1.0 / 180e6,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    depsilon = 0.05 * torch.randn_like(epsilon)
    dsigma = 1e-4 * torch.randn_like(sigma)
    data_weight = torch.randn(nt, 1, 2, device=device, dtype=dtype)

    born_data = tide.borntm(
        epsilon,
        sigma,
        mu,
        dx,
        dt,
        source_amplitude,
        source_location,
        receiver_location,
        depsilon=depsilon,
        dsigma=dsigma,
        pml_width=0,
        stencil=2,
        linearize_source=True,
        python_backend=False,
        storage_mode="device",
    )[-1]
    lhs = torch.sum(born_data * data_weight)

    eps_req = epsilon.detach().clone().requires_grad_(True)
    sig_req = sigma.detach().clone().requires_grad_(True)
    predicted = tide.maxwelltm(
        eps_req,
        sig_req,
        mu,
        dx,
        dt,
        source_amplitude,
        source_location,
        receiver_location,
        pml_width=0,
        stencil=2,
        model_gradient_sampling_interval=1,
        save_snapshots=True,
        python_backend=False,
        storage_mode="device",
    )[-1]
    grad_eps, grad_sig = torch.autograd.grad(
        torch.sum(predicted * data_weight),
        (eps_req, sig_req),
    )
    rhs = torch.sum(depsilon * grad_eps + dsigma * grad_sig)

    assert _relative_error(lhs, rhs) < 1e-2
