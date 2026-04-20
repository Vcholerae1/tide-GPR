import torch

import tide


def _build_tm_case(device: torch.device):
    dtype = torch.float32
    ny, nx = 6, 6
    nt = 8
    dx = 0.02
    dt = 4e-11

    epsilon = torch.ones((ny, nx), device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    depsilon = torch.full_like(epsilon, 0.05)

    source_location = torch.tensor([[[ny // 2, nx // 2]]], device=device)
    receiver_location = torch.tensor([[[ny // 2, nx // 2]]], device=device)
    torch.manual_seed(0)
    source_amplitude = torch.randn((1, 1, nt), device=device, dtype=dtype) * 1e-3
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "depsilon": depsilon,
        "dx": dx,
        "dt": dt,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "source_amplitude": source_amplitude,
    }


def test_maxwelltm_module_matches_functional_cpu():
    device = torch.device("cpu")
    case = _build_tm_case(device)

    model = tide.MaxwellTM(
        case["epsilon"], case["sigma"], case["mu"], grid_spacing=case["dx"]
    )

    out_module = model(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        stencil=2,
        pml_width=1,
        python_backend=True,
    )

    out_func = tide.maxwelltm(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        stencil=2,
        pml_width=1,
        python_backend=True,
    )

    for mod_out, fn_out in zip(out_module, out_func):
        torch.testing.assert_close(mod_out, fn_out)


def test_borntm_module_matches_functional_cpu():
    device = torch.device("cpu")
    case = _build_tm_case(device)

    model = tide.BornTM(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        depsilon=case["depsilon"],
    )

    out_module = model(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        stencil=2,
        pml_width=1,
        python_backend=True,
    )

    out_func = tide.borntm(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        depsilon=case["depsilon"],
        stencil=2,
        pml_width=1,
        python_backend=True,
    )

    for mod_out, fn_out in zip(out_module, out_func):
        torch.testing.assert_close(mod_out, fn_out)


def test_borntm_module_supports_background_and_scatter_gradients():
    device = torch.device("cpu")
    case = _build_tm_case(device)

    model = tide.BornTM(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        depsilon=case["depsilon"],
        epsilon_requires_grad=True,
        depsilon_requires_grad=True,
    )

    receiver = model(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        stencil=2,
        pml_width=1,
        python_backend=True,
    )[-1]

    loss = receiver.square().sum()
    loss.backward()

    assert model.epsilon.grad is not None
    assert model.depsilon is not None
    assert model.depsilon.grad is not None
