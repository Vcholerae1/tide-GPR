import torch

import tide


def _setup_case(device: torch.device):
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 10

    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.ones_like(epsilon) * 2e-4
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 5]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def test_maxwell3d_epsilon_gradient_finite_difference():
    device = torch.device("cpu")
    h = 1e-2
    (
        epsilon,
        sigma,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
    ) = _setup_case(device)

    eps_base = epsilon.clone().detach().requires_grad_(True)
    out_base = tide.maxwell3d(
        eps_base,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[-1]
    loss_base = out_base.pow(2).sum()
    loss_base.backward()
    assert eps_base.grad is not None

    iz, iy, ix = 3, 3, 4
    eps_pert = epsilon.clone()
    eps_pert[iz, iy, ix] += h
    out_pert = tide.maxwell3d(
        eps_pert,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[-1]
    fd_approx = (out_pert.pow(2).sum() - loss_base.detach()) / h
    grad_at_point = eps_base.grad[iz, iy, ix]

    assert torch.sign(grad_at_point) == torch.sign(fd_approx)
    rel_error = abs(grad_at_point - fd_approx) / (abs(fd_approx) + 1e-10)
    assert rel_error < 0.7


def test_maxwell3d_sigma_gradient_nonzero():
    device = torch.device("cpu")
    (
        epsilon,
        sigma,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
    ) = _setup_case(device)
    sigma = sigma.clone().detach().requires_grad_(True)

    out = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[-1]
    loss = out.pow(2).sum()
    loss.backward()
    assert sigma.grad is not None
    assert torch.isfinite(sigma.grad).all()
    assert sigma.grad.abs().sum() > 0
