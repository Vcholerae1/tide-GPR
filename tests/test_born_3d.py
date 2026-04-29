import warnings

import pytest
import torch

import tide
from tide import backend_utils
from tide.storage import STORAGE_FORMAT_BF16


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
def test_born3d_autograd_passes_dot_product_test(born_3d_setup, linearize_source: bool):
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


def test_born3d_autograd_samples_saved_gradient_intermediates(monkeypatch):
    from tide.maxwell.maxwell3d_born_autograd import Born3DForwardFunc

    class _Ctx:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    def fake_backend(*_args):
        return None

    monkeypatch.setattr(
        backend_utils, "get_backend_function", lambda *_args: fake_backend
    )

    device = torch.device("cpu")
    dtype = torch.float64
    nt, n_shots, nz, ny, nx = 7, 1, 3, 4, 5
    field_shape = (n_shots, nz, ny, nx)
    dca = torch.zeros(1, nz, ny, nx, device=device, dtype=dtype, requires_grad=True)
    dcb = torch.zeros_like(dca)
    ca = torch.ones_like(dca)
    cb = torch.ones_like(dca)
    cq = torch.ones_like(dca)
    f0 = torch.empty(0, device=device, dtype=dtype)
    df = torch.empty(0, device=device, dtype=dtype)
    profiles = tuple(torch.zeros(1, device=device, dtype=dtype) for _ in range(18))
    indices = (
        torch.empty(0, device=device, dtype=torch.long),
        torch.empty(0, device=device, dtype=torch.long),
    )
    background_wavefields = tuple(
        torch.zeros(field_shape, device=device, dtype=dtype) for _ in range(18)
    )
    scattered_wavefields = tuple(
        torch.zeros(field_shape, device=device, dtype=dtype) for _ in range(18)
    )
    ctx = _Ctx()

    Born3DForwardFunc.forward(
        ctx,
        dca,
        dcb,
        ca,
        cb,
        cq,
        f0,
        df,
        profiles,
        indices,
        background_wavefields,
        scattered_wavefields,
        {
            "dt": 4.0e-11,
            "nt": nt,
            "n_shots": n_shots,
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "n_sources": 0,
            "n_receivers": 0,
            "step_ratio": 3,
            "accuracy": 2,
            "pml_z0": 0,
            "pml_y0": 0,
            "pml_x0": 0,
            "pml_z1": nz,
            "pml_y1": ny,
            "pml_x1": nx,
            "source_component_idx": 1,
            "receiver_component_idx": 1,
            "n_threads": 0,
            "rdz": 1.0,
            "rdy": 1.0,
            "rdx": 1.0,
            "backend_device": device,
        },
    )

    store_ex = ctx.saved_tensors[-12]
    assert store_ex.shape == (3, n_shots, nz, ny, nx)
    assert ctx.meta["step_ratio"] == 3


def test_born3d_autograd_uses_bf16_for_saved_snapshots(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for 3D Born BF16 snapshot storage test.")

    from tide.maxwell.maxwell3d_born_autograd import Born3DForwardFunc

    class _Ctx:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    def fake_backend(*_args):
        return None

    monkeypatch.setattr(
        backend_utils, "get_backend_function", lambda *_args: fake_backend
    )

    device = torch.device("cuda")
    dtype = torch.float32
    nt, n_shots, nz, ny, nx = 5, 1, 3, 4, 5
    field_shape = (n_shots, nz, ny, nx)
    dca = torch.zeros(1, nz, ny, nx, device=device, dtype=dtype, requires_grad=True)
    dcb = torch.zeros_like(dca, requires_grad=True)
    ca = torch.ones_like(dca, requires_grad=True)
    cb = torch.ones_like(dca, requires_grad=True)
    cq = torch.ones_like(dca)
    f0 = torch.empty(0, device=device, dtype=dtype)
    df = torch.empty(0, device=device, dtype=dtype)
    profiles = tuple(torch.zeros(1, device=device, dtype=dtype) for _ in range(18))
    indices = (
        torch.empty(0, device=device, dtype=torch.long),
        torch.empty(0, device=device, dtype=torch.long),
    )
    background_wavefields = tuple(
        torch.zeros(field_shape, device=device, dtype=dtype) for _ in range(18)
    )
    scattered_wavefields = tuple(
        torch.zeros(field_shape, device=device, dtype=dtype) for _ in range(18)
    )
    ctx = _Ctx()

    Born3DForwardFunc.forward(
        ctx,
        dca,
        dcb,
        ca,
        cb,
        cq,
        f0,
        df,
        profiles,
        indices,
        background_wavefields,
        scattered_wavefields,
        {
            "dt": 4.0e-11,
            "nt": nt,
            "n_shots": n_shots,
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "n_sources": 0,
            "n_receivers": 0,
            "step_ratio": 1,
            "accuracy": 2,
            "pml_z0": 0,
            "pml_y0": 0,
            "pml_x0": 0,
            "pml_z1": nz,
            "pml_y1": ny,
            "pml_x1": nx,
            "source_component_idx": 1,
            "receiver_component_idx": 1,
            "n_threads": 0,
            "rdz": 1.0,
            "rdy": 1.0,
            "rdx": 1.0,
            "backend_device": device,
            "storage_compression": "bf16",
        },
    )

    saved = ctx.saved_tensors
    for tensor in (*saved[-12:-6], *saved[-6:]):
        assert tensor.dtype == torch.bfloat16
    assert ctx.meta["storage_format"] == STORAGE_FORMAT_BF16
    assert ctx.meta["shot_bytes_uncomp"] == nz * ny * nx * 2


def test_native_born3d_supports_background_gradients_by_default(
    born_3d_setup, monkeypatch
):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(17)
    setup = born_3d_setup
    epsilon = setup["epsilon"]
    sigma = setup["sigma"]
    mu = setup["mu"]
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)
    assert isinstance(mu, torch.Tensor)

    residual = torch.randn(
        14,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )
    depsilon_seed = 0.05 * torch.randn_like(epsilon)

    epsilon_native = epsilon.clone().detach().requires_grad_(True)
    sigma_native = sigma.clone().detach().requires_grad_(True)
    depsilon_native = depsilon_seed.clone().detach().requires_grad_(True)

    def fail_python_fallback(*_args, **_kwargs):
        raise AssertionError("native 3D Born background gradients used Python fallback")

    with monkeypatch.context() as m:
        m.setattr(
            "tide.maxwell.maxwell3d_born_cuda.born3d_python",
            fail_python_fallback,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred_native = tide.born3d(
                epsilon_native,
                sigma_native,
                mu,
                grid_spacing=setup["grid_spacing"],
                dt=setup["dt"],
                source_amplitude=setup["source_amplitude"],
                source_location=setup["source_location"],
                receiver_location=setup["receiver_location"],
                depsilon=depsilon_native,
                pml_width=setup["pml_width"],
                stencil=setup["stencil"],
                linearize_source=True,
                source_component=setup["source_component"],
                receiver_component=setup["receiver_component"],
                python_backend=False,
            )[-1]
    assert not any(
        "background model requires gradients" in str(w.message) for w in caught
    )
    grad_native = torch.autograd.grad(
        torch.sum(pred_native * residual),
        (epsilon_native, sigma_native, depsilon_native),
    )

    epsilon_reference = epsilon.clone().detach().requires_grad_(True)
    sigma_reference = sigma.clone().detach().requires_grad_(True)
    depsilon_reference = depsilon_seed.clone().detach().requires_grad_(True)
    pred_reference = tide.born3d(
        epsilon_reference,
        sigma_reference,
        mu,
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        depsilon=depsilon_reference,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=True,
        source_component=setup["source_component"],
        receiver_component=setup["receiver_component"],
        python_backend=True,
    )[-1]
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual),
        (epsilon_reference, sigma_reference, depsilon_reference),
    )

    for native, reference in zip(grad_native, grad_reference):
        torch.testing.assert_close(native, reference, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize("storage_compression", [False, "bf16"])
def test_native_born3d_cuda_supports_background_gradients_without_fallback(
    monkeypatch, storage_compression: bool | str
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D Born background gradient test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D Born background test.")

    torch.manual_seed(23)
    setup = _make_born_3d_setup(torch.device("cuda"), torch.float32)
    epsilon = setup["epsilon"]
    sigma = setup["sigma"]
    mu = setup["mu"]
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)
    assert isinstance(mu, torch.Tensor)

    residual = torch.randn(14, 1, 2, device=epsilon.device, dtype=epsilon.dtype)
    depsilon_seed = 0.05 * torch.randn_like(epsilon)

    epsilon_native = epsilon.clone().detach().requires_grad_(True)
    sigma_native = sigma.clone().detach().requires_grad_(True)
    depsilon_native = depsilon_seed.clone().detach().requires_grad_(True)

    def fail_python_fallback(*_args, **_kwargs):
        raise AssertionError("native CUDA 3D Born bggrad used Python fallback")

    with monkeypatch.context() as m:
        m.setattr(
            "tide.maxwell.maxwell3d_born_cuda.born3d_python",
            fail_python_fallback,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pred_native = tide.born3d(
                epsilon_native,
                sigma_native,
                mu,
                grid_spacing=setup["grid_spacing"],
                dt=setup["dt"],
                source_amplitude=setup["source_amplitude"],
                source_location=setup["source_location"],
                receiver_location=setup["receiver_location"],
                depsilon=depsilon_native,
                pml_width=setup["pml_width"],
                stencil=setup["stencil"],
                linearize_source=True,
                source_component=setup["source_component"],
                receiver_component=setup["receiver_component"],
                python_backend=False,
                storage_compression=storage_compression,
            )[-1]
    assert not any("falling back to Python" in str(w.message) for w in caught)

    grad_native = torch.autograd.grad(
        torch.sum(pred_native * residual),
        (epsilon_native, sigma_native, depsilon_native),
    )

    epsilon_reference = epsilon.clone().detach().requires_grad_(True)
    sigma_reference = sigma.clone().detach().requires_grad_(True)
    depsilon_reference = depsilon_seed.clone().detach().requires_grad_(True)
    pred_reference = tide.born3d(
        epsilon_reference,
        sigma_reference,
        mu,
        grid_spacing=setup["grid_spacing"],
        dt=setup["dt"],
        source_amplitude=setup["source_amplitude"],
        source_location=setup["source_location"],
        receiver_location=setup["receiver_location"],
        depsilon=depsilon_reference,
        pml_width=setup["pml_width"],
        stencil=setup["stencil"],
        linearize_source=True,
        source_component=setup["source_component"],
        receiver_component=setup["receiver_component"],
        python_backend=True,
    )[-1]
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual),
        (epsilon_reference, sigma_reference, depsilon_reference),
    )

    atol = 2e-3 if storage_compression else 5e-4
    rtol = 2e-2 if storage_compression else 5e-3
    for native, reference in zip(grad_native, grad_reference):
        torch.testing.assert_close(native, reference, atol=atol, rtol=rtol)


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
