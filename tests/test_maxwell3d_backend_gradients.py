import torch

import tide


def _case():
    device = torch.device("cpu")
    dtype = torch.float32
    nz, ny, nx = 6, 6, 7
    nt = 10
    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.ones_like(epsilon) * 1e-4
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[2, 2, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 2, 4]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        90e6, nt, 4e-11, peak_time=1.0 / 90e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def test_maxwell3d_backend_gradient_matches_python():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _case()

    eps_py = epsilon.clone().detach().requires_grad_(True)
    out_py = tide.maxwell3d(
        eps_py,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        python_backend=True,
    )[-1]
    out_py.pow(2).sum().backward()
    assert eps_py.grad is not None
    grad_py = eps_py.grad.detach().clone()

    eps_backend = epsilon.clone().detach().requires_grad_(True)
    out_backend = tide.maxwell3d(
        eps_backend,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        python_backend=False,
    )[-1]
    out_backend.pow(2).sum().backward()
    assert eps_backend.grad is not None
    torch.testing.assert_close(
        grad_py,
        eps_backend.grad,
        rtol=2e-4,
        atol=1e-3,
    )


def test_maxwell3d_backend_shared_model_multishot_gradient_matches_shot_sum():
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _case()

    source_location = torch.tensor(
        [[[2, 2, 2]], [[2, 3, 2]]], dtype=torch.long, device=epsilon.device
    )
    receiver_location = torch.tensor(
        [[[2, 2, 4]], [[2, 3, 4]]], dtype=torch.long, device=epsilon.device
    )
    source_amplitude = source_amplitude.repeat(2, 1, 1)
    source_amplitude[1] *= 0.7

    eps_shared = epsilon.clone().detach().requires_grad_(True)
    out_shared = tide.maxwell3d(
        eps_shared,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        python_backend=False,
    )[-1]
    out_shared.pow(2).sum().backward()
    assert eps_shared.grad is not None

    grad_sum = torch.zeros_like(epsilon)
    for shot_idx in range(source_amplitude.shape[0]):
        eps_single = epsilon.clone().detach().requires_grad_(True)
        out_single = tide.maxwell3d(
            eps_single,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude[shot_idx : shot_idx + 1],
            source_location=source_location[shot_idx : shot_idx + 1],
            receiver_location=receiver_location[shot_idx : shot_idx + 1],
            pml_width=2,
            python_backend=False,
        )[-1]
        out_single.pow(2).sum().backward()
        assert eps_single.grad is not None
        grad_sum += eps_single.grad

    torch.testing.assert_close(
        eps_shared.grad,
        grad_sum,
        rtol=2e-4,
        atol=1e-3,
    )
