import pytest
import torch

import tide
from numerical_utils import MaxwellExample, make_tm2d_example

pytestmark = [pytest.mark.cuda, pytest.mark.numerical]


def _tm2d_examples(
    device: torch.device,
) -> tuple[MaxwellExample, MaxwellExample]:
    example = make_tm2d_example(
        shape=(24, 30),
        nt=64,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=180e6,
        device=device,
        sigma=1e-4,
        source_location=(6, 10),
        receiver_locations=((6, 15), (6, 20)),
        pml_width=4,
        stencil=2,
    )
    y = torch.arange(24, device=device, dtype=example.epsilon.dtype)[:, None]
    x = torch.arange(30, device=device, dtype=example.epsilon.dtype)[None, :]
    blob = torch.exp(-(((y - 14) / 4) ** 2 + ((x - 18) / 5) ** 2) * 0.5)
    truth = example.updated(
        epsilon=example.epsilon + 0.35 * blob,
        sigma=example.sigma + 2e-4 * blob,
    )
    return example, truth


def _gradient(
    example: MaxwellExample,
    truth: MaxwellExample,
    *,
    python_backend: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        observed = truth.run(
            python_backend=python_backend,
            storage_mode="none",
        )[-1]

    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    predicted = example.run(
        epsilon=epsilon,
        sigma=sigma,
        model_gradient_sampling_interval=1,
        save_snapshots=True,
        python_backend=python_backend,
        storage_mode="device",
    )[-1]
    (0.5 * (predicted - observed).square().sum()).backward()
    assert epsilon.grad is not None
    assert sigma.grad is not None
    return epsilon.grad.detach(), sigma.grad.detach()


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

    example, truth = _tm2d_examples(torch.device("cuda"))
    reference_eps, reference_sig = _gradient(
        example,
        truth,
        python_backend=True,
    )
    native_eps, native_sig = _gradient(
        example,
        truth,
        python_backend=False,
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
    epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0 + 0.1 * blob
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

    born_data = tide.maxwell._kernel_api.borntm(
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
    predicted = tide.maxwell._kernel_api.maxwelltm(
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


def test_tm2d_cuda_coeff_backward_is_exact_at_high_side_pml():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TM2D native coeff-gradient test.")

    torch.manual_seed(41)
    device = torch.device("cuda")
    dtype = torch.float64
    ny, nx, nt = 18, 20, 180
    dx = 0.02
    dt = 2e-11
    epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[ny - 2, nx - 2]]], device=device)
    receiver_location = torch.tensor(
        [[[ny - 2, nx - 1], [ny - 1, nx - 2], [ny - 3, nx - 3]]],
        device=device,
    )
    source_amplitude = tide.ricker(
        1.2e9,
        nt,
        dt,
        peak_time=7e-10,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    depsilon = 0.08 * torch.randn_like(epsilon)
    dsigma = 5e-5 * torch.randn_like(sigma)
    data_weight = torch.randn(nt, 1, 3, device=device, dtype=dtype)
    common = {
        "grid_spacing": dx,
        "dt": dt,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "pml_width": [0, 6, 0, 6],
        "stencil": 2,
        "model_gradient_sampling_interval": 1,
        "save_snapshots": True,
    }

    def reference_forward(
        epsilon_arg: torch.Tensor, sigma_arg: torch.Tensor
    ) -> torch.Tensor:
        return tide.maxwell._kernel_api.maxwelltm(
            epsilon_arg,
            sigma_arg,
            mu,
            python_backend=True,
            storage_mode="none",
            **common,
        )[-1]

    _, jv = torch.autograd.functional.jvp(
        reference_forward,
        (epsilon, sigma),
        (depsilon, dsigma),
        strict=True,
    )
    lhs = torch.sum(jv * data_weight)

    epsilon_req = epsilon.clone().requires_grad_(True)
    sigma_req = sigma.clone().requires_grad_(True)
    predicted = tide.maxwell._kernel_api.maxwelltm(
        epsilon_req,
        sigma_req,
        mu,
        python_backend=False,
        storage_mode="device",
        **common,
    )[-1]
    grad_epsilon, grad_sigma = torch.autograd.grad(
        torch.sum(predicted * data_weight),
        (epsilon_req, sigma_req),
    )
    rhs = torch.sum(depsilon * grad_epsilon + dsigma * grad_sigma)

    assert _relative_error(lhs, rhs) < 1e-12
