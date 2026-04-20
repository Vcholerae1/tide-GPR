import pytest
import torch

import tide
from tide import backend_utils


def _make_born_3d_setup(device: torch.device, dtype: torch.dtype) -> dict[str, object]:
    nz, ny, nx = 8, 9, 10
    nt = 14
    dz = dy = dx = 0.04
    dt = 6.0e-11
    pml_width = 2
    stencil = 2

    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    epsilon[nz // 2 - 1 : nz // 2 + 1, ny // 2 - 1 : ny // 2 + 1, nx // 2] = 4.3
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[nz // 2, ny // 3, nx // 4]]], dtype=torch.long, device=device
    )
    receiver_location = torch.tensor(
        [[[nz // 2, ny // 2, nx // 2], [nz // 2, ny // 2, nx // 2 + 1]]],
        dtype=torch.long,
        device=device,
    )
    wavelet = tide.ricker(
        80e6,
        nt,
        dt,
        peak_time=1.0 / 80e6,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, nt)

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "grid_spacing": (dz, dy, dx),
        "dt": dt,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "pml_width": pml_width,
        "stencil": stencil,
        "source_component": "ey",
        "receiver_component": "ey",
    }


@pytest.fixture
def born_3d_setup() -> dict[str, object]:
    return _make_born_3d_setup(torch.device("cpu"), torch.float64)


def _born_outputs(
    setup: dict[str, object],
    *,
    depsilon: torch.Tensor,
    linearize_source: bool,
    python_backend: bool,
    bg_receiver_location: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    return tide.born3d(
        setup["epsilon"],
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        bg_receiver_location=bg_receiver_location,
        depsilon=depsilon,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=linearize_source,
        source_component=setup["source_component"],
        receiver_component=setup["receiver_component"],
        python_backend=python_backend,
    )


def _born_receivers(
    setup: dict[str, object],
    *,
    depsilon: torch.Tensor,
    linearize_source: bool,
    python_backend: bool,
) -> torch.Tensor:
    return _born_outputs(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=python_backend,
    )[-1]


def _maxwell_outputs(
    setup: dict[str, object],
    *,
    epsilon: torch.Tensor,
    receiver_location: torch.Tensor,
    python_backend: bool,
) -> tuple[torch.Tensor, ...]:
    return tide.maxwell3d(
        epsilon,
        setup["sigma"],
        setup["mu"],
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=receiver_location,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        model_gradient_sampling_interval=1,
        source_component=setup["source_component"],
        receiver_component=setup["receiver_component"],
        python_backend=python_backend,
    )


def _maxwell_receivers(
    setup: dict[str, object],
    *,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    return _maxwell_outputs(
        setup,
        epsilon=epsilon,
        receiver_location=setup["receiver_location"],
        python_backend=True,
    )[-1]


def test_born3d_is_linear_in_depsilon(born_3d_setup):
    torch.manual_seed(0)
    setup = born_3d_setup
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
        python_backend=True,
    )
    rhs = a * _born_receivers(
        setup,
        depsilon=m1,
        linearize_source=False,
        python_backend=True,
    ) + b * _born_receivers(
        setup,
        depsilon=m2,
        linearize_source=False,
        python_backend=True,
    )

    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-8)


def test_born3d_matches_maxwell3d_taylor_expansion(born_3d_setup):
    torch.manual_seed(1)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = torch.randn_like(epsilon)
    dm = 0.08 * dm / dm.abs().amax()

    base = _maxwell_receivers(setup, epsilon=epsilon)
    born = _born_receivers(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
    )

    errors = []
    for h in (1e-1, 5e-2, 2.5e-2):
        perturbed = _maxwell_receivers(setup, epsilon=epsilon + h * dm)
        errors.append(torch.linalg.norm(perturbed - base - h * born).item())

    assert errors[1] < 0.4 * errors[0]
    assert errors[2] < 0.4 * errors[1]


def test_born3d_returns_background_wavefields_and_receivers(born_3d_setup):
    torch.manual_seed(3)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    receiver_location = setup["receiver_location"]
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(receiver_location, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    born = _born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
        bg_receiver_location=receiver_location,
    )
    maxwell = _maxwell_outputs(
        setup,
        epsilon=epsilon,
        receiver_location=receiver_location,
        python_backend=True,
    )

    for born_out, maxwell_out in zip(born[:18], maxwell[:-1]):
        torch.testing.assert_close(born_out, maxwell_out)
    torch.testing.assert_close(born[-2], maxwell[-1])


def test_native_born3d_matches_python_reference(born_3d_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(11)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    receiver_location = setup["receiver_location"]
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(receiver_location, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)

    native = _born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=False,
        bg_receiver_location=receiver_location,
    )
    reference = _born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
        bg_receiver_location=receiver_location,
    )

    for native_out, reference_out in zip(native, reference):
        torch.testing.assert_close(native_out, reference_out, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("linearize_source", [True, False])
def test_born3d_autograd_passes_dot_product_test(
    born_3d_setup, linearize_source: bool
):
    torch.manual_seed(2)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=True,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _born_receivers(
            setup,
            depsilon=dm,
            linearize_source=linearize_source,
            python_backend=True,
        )
        * residual
    )
    grad_eps = torch.autograd.grad(torch.sum(pred * residual), depsilon)[0]
    rhs = torch.sum(dm * grad_eps)
    rel_error = torch.abs(lhs - rhs) / torch.maximum(
        torch.maximum(torch.abs(lhs), torch.abs(rhs)),
        torch.tensor(1e-16, device=lhs.device, dtype=lhs.dtype),
    )

    assert rel_error.item() < 1e-6


@pytest.mark.parametrize("linearize_source", [True, False])
def test_native_born3d_autograd_passes_dot_product_test(
    born_3d_setup, linearize_source: bool
):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(5)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=False,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _born_receivers(
            setup,
            depsilon=dm,
            linearize_source=linearize_source,
            python_backend=False,
        )
        * residual
    )
    grad_eps = torch.autograd.grad(torch.sum(pred * residual), depsilon)[0]
    rhs = torch.sum(dm * grad_eps)
    rel_error = torch.abs(lhs - rhs) / torch.maximum(
        torch.maximum(torch.abs(lhs), torch.abs(rhs)),
        torch.tensor(1e-16, device=lhs.device, dtype=lhs.dtype),
    )

    assert rel_error.item() < 1e-6


def test_born3d_autograd_matches_maxwell3d_autograd_gradient(born_3d_setup):
    torch.manual_seed(7)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    residual = torch.randn(
        14,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )

    epsilon_ref = epsilon.clone().detach().requires_grad_(True)
    pred_ref = _maxwell_receivers(setup, epsilon=epsilon_ref)
    grad_ref = torch.autograd.grad(torch.sum(pred_ref * residual), epsilon_ref)[0]

    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=True,
        python_backend=True,
    )
    grad_eps = torch.autograd.grad(torch.sum(pred * residual), depsilon)[0]

    torch.testing.assert_close(grad_eps, grad_ref, atol=5e-8, rtol=1e-7)


def test_native_born3d_autograd_matches_python_reference(born_3d_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(13)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    residual = torch.randn(
        14,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )

    depsilon_native = torch.zeros_like(epsilon, requires_grad=True)
    pred_native = _born_receivers(
        setup,
        depsilon=depsilon_native,
        linearize_source=True,
        python_backend=False,
    )
    grad_native = torch.autograd.grad(
        torch.sum(pred_native * residual), depsilon_native
    )[0]

    depsilon_reference = torch.zeros_like(epsilon, requires_grad=True)
    pred_reference = _born_receivers(
        setup,
        depsilon=depsilon_reference,
        linearize_source=True,
        python_backend=True,
    )
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual), depsilon_reference
    )[0]

    torch.testing.assert_close(grad_native, grad_reference, atol=1e-9, rtol=1e-8)


def test_native_born3d_cuda_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D Born parity test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D Born parity test.")

    torch.manual_seed(17)
    setup = _make_born_3d_setup(torch.device("cuda"), torch.float32)
    receiver_location = setup["receiver_location"]
    epsilon = setup["epsilon"]
    assert isinstance(receiver_location, torch.Tensor)
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)

    native = _born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=False,
        bg_receiver_location=receiver_location,
    )
    reference = _born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
        bg_receiver_location=receiver_location,
    )

    for native_out, reference_out in zip(native, reference):
        torch.testing.assert_close(native_out, reference_out, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("linearize_source", [True, False])
def test_native_born3d_cuda_autograd_passes_dot_product_test(
    linearize_source: bool,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D Born autograd test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D Born autograd test.")

    torch.manual_seed(19)
    setup = _make_born_3d_setup(torch.device("cuda"), torch.float32)
    epsilon = setup["epsilon"]
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=False,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _born_receivers(
            setup,
            depsilon=dm,
            linearize_source=linearize_source,
            python_backend=False,
        )
        * residual
    )
    grad_eps = torch.autograd.grad(torch.sum(pred * residual), depsilon)[0]
    rhs = torch.sum(dm * grad_eps)
    rel_error = torch.abs(lhs - rhs) / torch.maximum(
        torch.maximum(torch.abs(lhs), torch.abs(rhs)),
        torch.tensor(1e-16, device=lhs.device, dtype=lhs.dtype),
    )

    assert rel_error.item() < 1e-4
