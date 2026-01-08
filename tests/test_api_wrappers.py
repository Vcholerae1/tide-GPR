import torch

import tide


def test_maxwelltm_module_matches_functional_cpu():
    device = torch.device("cpu")
    dtype = torch.float32

    ny, nx = 6, 6
    nt = 8
    dx = 0.02
    dt = 4e-11

    epsilon = torch.ones((ny, nx), device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[ny // 2, nx // 2]]], device=device)
    receiver_location = torch.tensor([[[ny // 2, nx // 2]]], device=device)
    torch.manual_seed(0)
    source_amplitude = torch.randn((1, 1, nt), device=device, dtype=dtype) * 1e-3

    model = tide.MaxwellTM(epsilon, sigma, mu, grid_spacing=dx)

    out_module = model(
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=1,
        python_backend=True,
    )

    out_func = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=1,
        python_backend=True,
    )

    for mod_out, fn_out in zip(out_module, out_func):
        torch.testing.assert_close(mod_out, fn_out)

