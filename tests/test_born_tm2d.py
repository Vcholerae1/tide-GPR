import pytest
import torch

import tide
from tide import backend_utils


@pytest.fixture
def born_tm_setup() -> dict[str, object]:
    device = torch.device("cpu")
    dtype = torch.float64

    ny, nx = 14, 16
    nt = 24
    dx = 0.02
    dt = 3.5e-11
    pml_width = 3
    stencil = 2

    epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype)
    epsilon[ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1] = 4.4
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[ny // 2, nx // 4]]], dtype=torch.long, device=device
    )
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2], [ny // 2, nx // 2 + 2]]],
        dtype=torch.long,
        device=device,
    )
    wavelet = tide.ricker(
        120e6,
        nt,
        dt,
        peak_time=1.0 / 120e6,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, nt)

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "grid_spacing": dx,
        "dt": dt,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "pml_width": pml_width,
        "stencil": stencil,
    }


def _born_receivers(
    setup: dict[str, object],
    *,
    depsilon: torch.Tensor,
    linearize_source: bool,
) -> torch.Tensor:
    return tide.borntm(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        depsilon=depsilon,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=linearize_source,
    )[-1]


def _maxwell_receivers(
    setup: dict[str, object],
    *,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    return tide.maxwelltm(
        epsilon,
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        model_gradient_sampling_interval=1,
        python_backend=True,
    )[-1]


def test_borntm_is_linear_in_depsilon(born_tm_setup):
    torch.manual_seed(0)
    setup = born_tm_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    m1 = torch.randn_like(epsilon) * 0.05
    m2 = torch.randn_like(epsilon) * 0.05
    a = 0.7
    b = -0.35

    lhs = _born_receivers(
        setup,
        depsilon=a * m1 + b * m2,
        linearize_source=False,
    )
    rhs = a * _born_receivers(
        setup, depsilon=m1, linearize_source=False
    ) + b * _born_receivers(
        setup,
        depsilon=m2,
        linearize_source=False,
    )

    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-8)


def test_borntm_matches_maxwelltm_taylor_expansion(born_tm_setup):
    torch.manual_seed(1)
    setup = born_tm_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = torch.randn_like(epsilon)
    dm = 0.1 * dm / dm.abs().amax()

    base = _maxwell_receivers(setup, epsilon=epsilon)
    born = _born_receivers(setup, depsilon=dm, linearize_source=True)

    errors = []
    for h in (1e-1, 5e-2, 2.5e-2):
        perturbed = _maxwell_receivers(setup, epsilon=epsilon + h * dm)
        errors.append(torch.linalg.norm(perturbed - base - h * born).item())

    assert errors[1] < 0.35 * errors[0]
    assert errors[2] < 0.35 * errors[1]


def test_native_borntm_matches_python_reference(born_tm_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(11)
    setup = born_tm_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)

    native = tide.borntm(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        depsilon=dm,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=True,
        python_backend=False,
    )[-1]
    reference = tide.borntm(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        depsilon=dm,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=True,
        python_backend=True,
    )[-1]

    assert torch.allclose(native, reference, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("linearize_source", [True, False])
def test_borntm_adjoint_passes_dot_product_test(born_tm_setup, linearize_source: bool):
    torch.manual_seed(2)
    setup = born_tm_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    residual = torch.randn_like(
        _born_receivers(setup, depsilon=dm, linearize_source=linearize_source)
    )

    lhs = torch.sum(
        _born_receivers(setup, depsilon=dm, linearize_source=linearize_source)
        * residual
    )
    grad_eps, _ = tide.borntm_adjoint(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        residual=residual,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=linearize_source,
        python_backend=True,
    )
    rhs = torch.sum(dm * grad_eps)
    rel_error = torch.abs(lhs - rhs) / torch.maximum(
        torch.maximum(torch.abs(lhs), torch.abs(rhs)),
        torch.tensor(1e-16, device=lhs.device, dtype=lhs.dtype),
    )

    assert rel_error.item() < 1e-6


@pytest.mark.parametrize("linearize_source", [True, False])
def test_native_borntm_adjoint_passes_dot_product_test(
    born_tm_setup, linearize_source: bool
):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(5)
    setup = born_tm_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    residual = torch.randn_like(
        tide.borntm(
            setup["epsilon"],
            setup["sigma"],
            setup["mu"],
            grid_spacing=setup["grid_spacing"],
            dt=setup["dt"],
            source_amplitude=setup["source_amplitude"],
            source_location=setup["source_location"],
            receiver_location=setup["receiver_location"],
            depsilon=dm,
            pml_width=setup["pml_width"],
            stencil=setup["stencil"],
            linearize_source=linearize_source,
            python_backend=False,
        )[-1]
    )

    lhs = torch.sum(
        tide.borntm(
            setup["epsilon"],
            setup["sigma"],
            setup["mu"],
            grid_spacing=setup["grid_spacing"],
            dt=setup["dt"],
            source_amplitude=setup["source_amplitude"],
            source_location=setup["source_location"],
            receiver_location=setup["receiver_location"],
            depsilon=dm,
            pml_width=setup["pml_width"],
            stencil=setup["stencil"],
            linearize_source=linearize_source,
            python_backend=False,
        )[-1]
        * residual
    )
    grad_eps, _ = tide.borntm_adjoint(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        residual=residual,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=linearize_source,
        python_backend=False,
    )
    rhs = torch.sum(dm * grad_eps)
    rel_error = torch.abs(lhs - rhs) / torch.maximum(
        torch.maximum(torch.abs(lhs), torch.abs(rhs)),
        torch.tensor(1e-16, device=lhs.device, dtype=lhs.dtype),
    )

    assert rel_error.item() < 1e-6


def test_borntm_adjoint_matches_maxwelltm_autograd_gradient(born_tm_setup):
    torch.manual_seed(3)
    setup = born_tm_setup
    residual = torch.randn(
        24,
        1,
        2,
        device=setup["epsilon"].device,
        dtype=setup["epsilon"].dtype,
    )

    epsilon = setup["epsilon"].clone().detach().requires_grad_(True)
    pred = _maxwell_receivers(setup, epsilon=epsilon)
    grad_ref = torch.autograd.grad(torch.sum(pred * residual), epsilon)[0]

    grad_eps, _ = tide.borntm_adjoint(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        residual=residual,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=True,
        python_backend=True,
    )

    assert torch.allclose(grad_eps, grad_ref, atol=1e-9, rtol=1e-8)


def test_native_borntm_adjoint_matches_python_reference(born_tm_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(7)
    setup = born_tm_setup
    residual = torch.randn(
        24,
        1,
        2,
        device=setup["epsilon"].device,
        dtype=setup["epsilon"].dtype,
    )

    grad_native, _ = tide.borntm_adjoint(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        residual=residual,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=True,
        python_backend=False,
    )
    grad_reference, _ = tide.borntm_adjoint(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        residual=residual,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=True,
        python_backend=True,
    )

    assert torch.allclose(grad_native, grad_reference, atol=1e-9, rtol=1e-8)
