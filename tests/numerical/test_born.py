from __future__ import annotations

import pytest
import torch
import warnings
from numerical_utils import MaxwellExample, make_tm2d_example, make_maxwell3d_example
from tide import backend_utils, staggered
from tide.maxwell.common import _get_ctx_handle, _release_ctx_handle
from tide.maxwell.tm2d_born_autograd import BornTMForwardFunc
from tide.storage import STORAGE_FORMAT_BF16, STORAGE_FORMAT_FULL

# --- test_born_tm2d.py ---


def _tm2d_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.reshape(-1).double()
    bv = b.reshape(-1).double()
    return float((av @ bv) / (av.norm() * bv.norm() + 1e-30))


@pytest.fixture
def born_tm_setup() -> MaxwellExample:
    example = make_tm2d_example(
        shape=(14, 16),
        nt=24,
        grid_spacing=0.02,
        dt=3.5e-11,
        frequency=120e6,
        dtype=torch.float64,
        source_location=(7, 4),
        receiver_locations=((7, 8), (7, 10)),
        pml_width=3,
        stencil=2,
    )
    epsilon = example.epsilon.clone()
    epsilon[6:8, 7:9] = 4.4
    return example.updated(epsilon=epsilon)


def _tm2d_born_outputs(
    setup: MaxwellExample,
    *,
    depsilon: torch.Tensor,
    linearize_source: bool,
    python_backend: bool,
    bg_receiver_location: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    return setup.run_born(
        bg_receiver_location=bg_receiver_location,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=python_backend,
    )


def _tm2d_born_receivers(
    setup: MaxwellExample,
    *,
    depsilon: torch.Tensor,
    linearize_source: bool,
    python_backend: bool,
) -> torch.Tensor:
    return _tm2d_born_outputs(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=python_backend,
    )[-1]


def _tm2d_maxwell_outputs(
    setup: MaxwellExample,
    *,
    epsilon: torch.Tensor,
    receiver_location: torch.Tensor,
    python_backend: bool,
) -> tuple[torch.Tensor, ...]:
    return setup.run(
        epsilon=epsilon,
        receiver_location=receiver_location,
        model_gradient_sampling_interval=1,
        python_backend=python_backend,
    )


def _tm2d_maxwell_receivers(
    setup: MaxwellExample,
    *,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    return _tm2d_maxwell_outputs(
        setup,
        epsilon=epsilon,
        receiver_location=setup.receiver_location,
        python_backend=True,
    )[-1]


def test_borntm_is_linear_in_depsilon(born_tm_setup):
    torch.manual_seed(0)
    setup = born_tm_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    m1 = torch.randn_like(epsilon) * 0.05
    m2 = torch.randn_like(epsilon) * 0.05
    a = 0.7
    b = -0.35

    lhs = _tm2d_born_receivers(
        setup,
        depsilon=a * m1 + b * m2,
        linearize_source=False,
        python_backend=True,
    )
    rhs = a * _tm2d_born_receivers(
        setup,
        depsilon=m1,
        linearize_source=False,
        python_backend=True,
    ) + b * _tm2d_born_receivers(
        setup,
        depsilon=m2,
        linearize_source=False,
        python_backend=True,
    )

    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-8)


def test_borntm_matches_maxwelltm_taylor_expansion(born_tm_setup):
    torch.manual_seed(1)
    setup = born_tm_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    dm = torch.randn_like(epsilon)
    dm = 0.1 * dm / dm.abs().amax()

    base = _tm2d_maxwell_receivers(setup, epsilon=epsilon)
    born = _tm2d_born_receivers(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
    )

    errors = []
    for h in (1e-1, 5e-2, 2.5e-2):
        perturbed = _tm2d_maxwell_receivers(setup, epsilon=epsilon + h * dm)
        errors.append(torch.linalg.norm(perturbed - base - h * born).item())

    assert errors[1] < 0.35 * errors[0]
    assert errors[2] < 0.35 * errors[1]


def test_borntm_returns_background_wavefields_and_receivers(born_tm_setup):
    torch.manual_seed(3)
    setup = born_tm_setup
    epsilon = setup.epsilon
    receiver_location = setup.receiver_location
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(receiver_location, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    born = _tm2d_born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
        bg_receiver_location=receiver_location,
    )
    maxwell = _tm2d_maxwell_outputs(
        setup,
        epsilon=epsilon,
        receiver_location=receiver_location,
        python_backend=True,
    )

    for born_out, maxwell_out in zip(born[:7], maxwell[:-1]):
        torch.testing.assert_close(born_out, maxwell_out)
    torch.testing.assert_close(born[-2], maxwell[-1])


def test_native_borntm_matches_python_reference(born_tm_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(11)
    setup = born_tm_setup
    epsilon = setup.epsilon
    receiver_location = setup.receiver_location
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(receiver_location, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)

    native = _tm2d_born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=False,
        bg_receiver_location=receiver_location,
    )
    reference = _tm2d_born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
        bg_receiver_location=receiver_location,
    )

    for native_out, reference_out in zip(native, reference):
        torch.testing.assert_close(native_out, reference_out, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("linearize_source", [True, False])
def test_borntm_autograd_passes_dot_product_test(born_tm_setup, linearize_source: bool):
    torch.manual_seed(2)
    setup = born_tm_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _tm2d_born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=True,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _tm2d_born_receivers(
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
def test_native_borntm_autograd_uses_coeff_gradient_direction(
    born_tm_setup, linearize_source: bool
):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(5)
    setup = born_tm_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _tm2d_born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=False,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _tm2d_born_receivers(
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

    limit = 8e-2 if linearize_source else 1e-2
    assert rel_error.item() < limit


def test_borntm_autograd_matches_maxwelltm_autograd_gradient(born_tm_setup):
    torch.manual_seed(7)
    setup = born_tm_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    residual = torch.randn(
        24,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )

    epsilon_ref = epsilon.clone().detach().requires_grad_(True)
    pred_ref = _tm2d_maxwell_receivers(setup, epsilon=epsilon_ref)
    grad_ref = torch.autograd.grad(torch.sum(pred_ref * residual), epsilon_ref)[0]

    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _tm2d_born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=True,
        python_backend=True,
    )
    grad_eps = torch.autograd.grad(torch.sum(pred * residual), depsilon)[0]

    torch.testing.assert_close(grad_eps, grad_ref, atol=1e-9, rtol=1e-8)


def test_native_borntm_autograd_matches_python_reference_direction(born_tm_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(13)
    setup = born_tm_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    residual = torch.randn(
        24,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )

    depsilon_native = torch.zeros_like(epsilon, requires_grad=True)
    pred_native = _tm2d_born_receivers(
        setup,
        depsilon=depsilon_native,
        linearize_source=True,
        python_backend=False,
    )
    grad_native = torch.autograd.grad(
        torch.sum(pred_native * residual), depsilon_native
    )[0]

    depsilon_reference = torch.zeros_like(epsilon, requires_grad=True)
    pred_reference = _tm2d_born_receivers(
        setup,
        depsilon=depsilon_reference,
        linearize_source=True,
        python_backend=True,
    )
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual), depsilon_reference
    )[0]

    assert _tm2d_cosine(grad_native, grad_reference) > 0.99


def test_native_borntm_supports_background_gradients_by_default(born_tm_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(17)
    setup = born_tm_setup
    epsilon = setup.epsilon
    sigma = setup.sigma
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)

    residual = torch.randn(
        24,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )
    depsilon_seed = 0.05 * torch.randn_like(epsilon)

    epsilon_native = epsilon.clone().detach().requires_grad_(True)
    sigma_native = sigma.clone().detach().requires_grad_(True)
    depsilon_native = depsilon_seed.clone().detach().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred_native = setup.run_born(
            epsilon=epsilon_native,
            sigma=sigma_native,
            depsilon=depsilon_native,
            linearize_source=True,
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
    pred_reference = setup.run_born(
        epsilon=epsilon_reference,
        sigma=sigma_reference,
        depsilon=depsilon_reference,
        linearize_source=True,
        python_backend=True,
    )[-1]
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual),
        (epsilon_reference, sigma_reference, depsilon_reference),
    )

    for grad_n in grad_native:
        assert torch.isfinite(grad_n).all()
        assert grad_n.norm() > 0
    assert _tm2d_cosine(grad_native[2], grad_reference[2]) > 0.99


# --- test_born_3d.py ---


def _em3d_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.reshape(-1).double()
    bv = b.reshape(-1).double()
    return float((av @ bv) / (av.norm() * bv.norm() + 1e-30))


def _make_born_3d_setup(
    device: torch.device,
    dtype: torch.dtype,
) -> MaxwellExample:
    example = make_maxwell3d_example(
        shape=(8, 9, 10),
        nt=14,
        grid_spacing=(0.04, 0.04, 0.04),
        dt=6.0e-11,
        frequency=80e6,
        dtype=dtype,
        device=device,
        source_location=(4, 3, 2),
        receiver_locations=((4, 4, 5), (4, 4, 6)),
        pml_width=2,
        stencil=2,
    )
    epsilon = example.epsilon.clone()
    epsilon[3:5, 3:5, 5] = 4.3
    return example.updated(epsilon=epsilon)


@pytest.fixture
def born_3d_setup() -> MaxwellExample:
    return _make_born_3d_setup(torch.device("cpu"), torch.float64)


def _em3d_born_outputs(
    setup: MaxwellExample,
    *,
    depsilon: torch.Tensor,
    linearize_source: bool,
    python_backend: bool,
    bg_receiver_location: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    return setup.run_born(
        bg_receiver_location=bg_receiver_location,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=python_backend,
    )


def _em3d_born_receivers(
    setup: MaxwellExample,
    *,
    depsilon: torch.Tensor,
    linearize_source: bool,
    python_backend: bool,
) -> torch.Tensor:
    return _em3d_born_outputs(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=python_backend,
    )[-1]


def _em3d_maxwell_outputs(
    setup: MaxwellExample,
    *,
    epsilon: torch.Tensor,
    receiver_location: torch.Tensor,
    python_backend: bool,
) -> tuple[torch.Tensor, ...]:
    return setup.run(
        epsilon=epsilon,
        receiver_location=receiver_location,
        model_gradient_sampling_interval=1,
        python_backend=python_backend,
    )


def _em3d_maxwell_receivers(
    setup: MaxwellExample,
    *,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    return _em3d_maxwell_outputs(
        setup,
        epsilon=epsilon,
        receiver_location=setup.receiver_location,
        python_backend=True,
    )[-1]


def test_born3d_is_linear_in_depsilon(born_3d_setup):
    torch.manual_seed(0)
    setup = born_3d_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    m1 = torch.randn_like(epsilon) * 0.05
    m2 = torch.randn_like(epsilon) * 0.05
    a = 0.7
    b = -0.35

    lhs = _em3d_born_receivers(
        setup,
        depsilon=a * m1 + b * m2,
        linearize_source=False,
        python_backend=True,
    )
    rhs = a * _em3d_born_receivers(
        setup,
        depsilon=m1,
        linearize_source=False,
        python_backend=True,
    ) + b * _em3d_born_receivers(
        setup,
        depsilon=m2,
        linearize_source=False,
        python_backend=True,
    )

    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-8)


def test_born3d_matches_maxwell3d_taylor_expansion(born_3d_setup):
    torch.manual_seed(1)
    setup = born_3d_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    dm = torch.randn_like(epsilon)
    dm = 0.08 * dm / dm.abs().amax()

    base = _em3d_maxwell_receivers(setup, epsilon=epsilon)
    born = _em3d_born_receivers(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
    )

    errors = []
    for h in (1e-1, 5e-2, 2.5e-2):
        perturbed = _em3d_maxwell_receivers(setup, epsilon=epsilon + h * dm)
        errors.append(torch.linalg.norm(perturbed - base - h * born).item())

    assert errors[1] < 0.4 * errors[0]
    assert errors[2] < 0.4 * errors[1]


def test_born3d_returns_background_wavefields_and_receivers(born_3d_setup):
    torch.manual_seed(3)
    setup = born_3d_setup
    epsilon = setup.epsilon
    receiver_location = setup.receiver_location
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(receiver_location, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    born = _em3d_born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
        bg_receiver_location=receiver_location,
    )
    maxwell = _em3d_maxwell_outputs(
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
    epsilon = setup.epsilon
    receiver_location = setup.receiver_location
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(receiver_location, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)

    native = _em3d_born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=False,
        bg_receiver_location=receiver_location,
    )
    reference = _em3d_born_outputs(
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
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _em3d_born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=True,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _em3d_born_receivers(
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
def test_native_born3d_autograd_uses_coeff_gradient_direction(
    born_3d_setup, linearize_source: bool
):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(5)
    setup = born_3d_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _em3d_born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=False,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _em3d_born_receivers(
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

    limit = 8e-2 if linearize_source else 2e-1
    assert rel_error.item() < limit


def test_born3d_autograd_matches_maxwell3d_autograd_gradient(born_3d_setup):
    torch.manual_seed(7)
    setup = born_3d_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    residual = torch.randn(
        14,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )

    epsilon_ref = epsilon.clone().detach().requires_grad_(True)
    pred_ref = _em3d_maxwell_receivers(setup, epsilon=epsilon_ref)
    grad_ref = torch.autograd.grad(torch.sum(pred_ref * residual), epsilon_ref)[0]

    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _em3d_born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=True,
        python_backend=True,
    )
    grad_eps = torch.autograd.grad(torch.sum(pred * residual), depsilon)[0]

    torch.testing.assert_close(grad_eps, grad_ref, atol=5e-8, rtol=1e-7)


def test_native_born3d_autograd_matches_python_reference_direction(born_3d_setup):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(13)
    setup = born_3d_setup
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    residual = torch.randn(
        14,
        1,
        2,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )

    depsilon_native = torch.zeros_like(epsilon, requires_grad=True)
    pred_native = _em3d_born_receivers(
        setup,
        depsilon=depsilon_native,
        linearize_source=True,
        python_backend=False,
    )
    grad_native = torch.autograd.grad(
        torch.sum(pred_native * residual), depsilon_native
    )[0]

    depsilon_reference = torch.zeros_like(epsilon, requires_grad=True)
    pred_reference = _em3d_born_receivers(
        setup,
        depsilon=depsilon_reference,
        linearize_source=True,
        python_backend=True,
    )
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual), depsilon_reference
    )[0]

    assert _em3d_cosine(grad_native, grad_reference) > 0.99


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
    born_3d_setup,
):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")

    torch.manual_seed(17)
    setup = born_3d_setup
    epsilon = setup.epsilon
    sigma = setup.sigma
    mu = setup.mu
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

    # The native adapter no longer imports or calls born3d_python at all
    # (fallbacks are owned by select_backend), so the assertion below is that
    # the native path emits no fallback-related warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred_native = setup.run_born(
            epsilon=epsilon_native,
            sigma=sigma_native,
            depsilon=depsilon_native,
            linearize_source=True,
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
    pred_reference = setup.run_born(
        epsilon=epsilon_reference,
        sigma=sigma_reference,
        depsilon=depsilon_reference,
        linearize_source=True,
        python_backend=True,
    )[-1]
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual),
        (epsilon_reference, sigma_reference, depsilon_reference),
    )

    for grad_n in grad_native:
        assert torch.isfinite(grad_n).all()
        assert grad_n.norm() > 0
    assert _em3d_cosine(grad_native[2], grad_reference[2]) > 0.99


@pytest.mark.parametrize("storage_compression", [False, "bf16"])
def test_native_born3d_cuda_supports_background_gradients_without_fallback(
    storage_compression: bool | str,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D Born background gradient test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D Born background test.")

    torch.manual_seed(23)
    setup = _make_born_3d_setup(torch.device("cuda"), torch.float32)
    epsilon = setup.epsilon
    sigma = setup.sigma
    mu = setup.mu
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)
    assert isinstance(mu, torch.Tensor)

    residual = torch.randn(14, 1, 2, device=epsilon.device, dtype=epsilon.dtype)
    depsilon_seed = 0.05 * torch.randn_like(epsilon)

    epsilon_native = epsilon.clone().detach().requires_grad_(True)
    sigma_native = sigma.clone().detach().requires_grad_(True)
    depsilon_native = depsilon_seed.clone().detach().requires_grad_(True)

    # The native adapter no longer imports or calls born3d_python at all
    # (fallbacks are owned by select_backend), so the assertion below is that
    # the native path emits no fallback-related warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred_native = setup.run_born(
            epsilon=epsilon_native,
            sigma=sigma_native,
            depsilon=depsilon_native,
            linearize_source=True,
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
    pred_reference = setup.run_born(
        epsilon=epsilon_reference,
        sigma=sigma_reference,
        depsilon=depsilon_reference,
        linearize_source=True,
        python_backend=True,
    )[-1]
    grad_reference = torch.autograd.grad(
        torch.sum(pred_reference * residual),
        (epsilon_reference, sigma_reference, depsilon_reference),
    )

    for grad_n in grad_native:
        assert torch.isfinite(grad_n).all()
        assert grad_n.norm() > 0
    assert _em3d_cosine(grad_native[2], grad_reference[2]) > 0.98


def test_native_born3d_cuda_matches_python_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D Born parity test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D Born parity test.")

    torch.manual_seed(17)
    setup = _make_born_3d_setup(torch.device("cuda"), torch.float32)
    receiver_location = setup.receiver_location
    epsilon = setup.epsilon
    assert isinstance(receiver_location, torch.Tensor)
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)

    native = _em3d_born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=False,
        bg_receiver_location=receiver_location,
    )
    reference = _em3d_born_outputs(
        setup,
        depsilon=dm,
        linearize_source=True,
        python_backend=True,
        bg_receiver_location=receiver_location,
    )

    for native_out, reference_out in zip(native, reference):
        torch.testing.assert_close(native_out, reference_out, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("linearize_source", [True, False])
def test_native_born3d_cuda_autograd_uses_coeff_gradient_direction(
    linearize_source: bool,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D Born autograd test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D Born autograd test.")

    torch.manual_seed(19)
    setup = _make_born_3d_setup(torch.device("cuda"), torch.float32)
    epsilon = setup.epsilon
    assert isinstance(epsilon, torch.Tensor)

    dm = 0.05 * torch.randn_like(epsilon)
    depsilon = torch.zeros_like(epsilon, requires_grad=True)
    pred = _em3d_born_receivers(
        setup,
        depsilon=depsilon,
        linearize_source=linearize_source,
        python_backend=False,
    )
    residual = torch.randn_like(pred.detach())

    lhs = torch.sum(
        _em3d_born_receivers(
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

    assert rel_error.item() < 6e-1


# --- test_tm2d_born_bggrad.py ---


def _assert_native_grads_are_finite(grads: tuple[torch.Tensor, ...]) -> None:
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert grad.norm() > 0


def _native_tm2d_born_receivers(
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cq: torch.Tensor,
    dca: torch.Tensor,
    dcb: torch.Tensor,
    f0: torch.Tensor,
    df: torch.Tensor,
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
    nt: int,
    n_shots: int,
    ny: int,
    nx: int,
    n_sources: int,
    n_receivers: int,
    stencil: int,
    dEy_0: torch.Tensor | None = None,
    storage_compression: bool | str = False,
) -> torch.Tensor:
    device = ca.device
    dtype = ca.dtype
    storage_format = STORAGE_FORMAT_BF16 if storage_compression else STORAGE_FORMAT_FULL
    fd_pad = stencil // 2
    pml_y0 = pml_x0 = fd_pad
    pml_y1 = ny - fd_pad + 1
    pml_x1 = nx - fd_pad + 1

    zeros = torch.zeros(n_shots, ny, nx, dtype=dtype, device=device)
    zeros_m = torch.zeros_like(zeros)
    line_zero_y = torch.zeros(ny, dtype=dtype, device=device)
    line_zero_x = torch.zeros(nx, dtype=dtype, device=device)
    line_one_y = torch.ones(ny, dtype=dtype, device=device)
    line_one_x = torch.ones(nx, dtype=dtype, device=device)

    outputs = BornTMForwardFunc.apply(
        dca,
        dcb,
        ca,
        cb,
        cq,
        f0,
        df,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_one_y,
        line_one_y,
        line_one_x,
        line_one_x,
        sources_i,
        receivers_i,
        1.0,
        1.0,
        1.0,
        nt,
        n_shots,
        ny,
        nx,
        n_sources,
        n_receivers,
        1,
        stencil,
        False,
        False,
        False,
        pml_y0,
        pml_x0,
        pml_y1,
        pml_x1,
        "device",
        storage_format,
        "",
        storage_compression,
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros.clone() if dEy_0 is None else dEy_0.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        0,
        device,
    )
    return outputs[7]


def test_tm2d_born_autograd_uses_bf16_for_saved_snapshots(monkeypatch):
    def fake_backend(*_args):
        return None

    monkeypatch.setattr(
        backend_utils, "get_backend_function", lambda *_args: fake_backend
    )

    device = torch.device("cpu")
    dtype = torch.float32
    nt, n_shots, ny, nx = 4, 1, 5, 6
    n_sources = n_receivers = 0
    zeros = torch.zeros(n_shots, ny, nx, dtype=dtype, device=device)
    line_zero_y = torch.zeros(ny, dtype=dtype, device=device)
    line_zero_x = torch.zeros(nx, dtype=dtype, device=device)
    line_one_y = torch.ones(ny, dtype=dtype, device=device)
    line_one_x = torch.ones(nx, dtype=dtype, device=device)
    empty_i = torch.empty(0, dtype=torch.long, device=device)
    empty_f = torch.empty(0, dtype=dtype, device=device)

    outputs = BornTMForwardFunc.forward(
        torch.zeros(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.zeros(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.ones(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.ones(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.ones(1, ny, nx, dtype=dtype, device=device),
        empty_f,
        empty_f,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_one_y,
        line_one_y,
        line_one_x,
        line_one_x,
        empty_i,
        empty_i,
        1.0,
        1.0,
        1.0,
        nt,
        n_shots,
        ny,
        nx,
        n_sources,
        n_receivers,
        1,
        2,
        False,
        False,
        False,
        1,
        1,
        ny,
        nx,
        "device",
        STORAGE_FORMAT_BF16,
        "",
        "bf16",
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        0,
        device,
    )
    ctx_handle = outputs[-1]
    ctx_data = _get_ctx_handle(int(ctx_handle.item()))
    try:
        for tensor in (
            *ctx_data["backward_storage_tensors"],
            *ctx_data["direct_snapshot_tensors"],
        ):
            assert tensor.dtype == torch.bfloat16
        assert ctx_data["storage_format"] == STORAGE_FORMAT_BF16
        assert ctx_data["shot_bytes_uncomp"] == ny * nx * 2
    finally:
        _release_ctx_handle(int(ctx_handle.item()))


def _reference_tm2d_born_receivers(
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cq: torch.Tensor,
    dca: torch.Tensor,
    dcb: torch.Tensor,
    f0: torch.Tensor,
    df: torch.Tensor,
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
    nt: int,
    n_shots: int,
    ny: int,
    nx: int,
    n_sources: int,
    dEy_0: torch.Tensor | None = None,
    stencil: int = 2,
) -> torch.Tensor:
    device = ca.device
    dtype = ca.dtype
    rdy = torch.tensor(1.0, dtype=dtype, device=device)
    rdx = torch.tensor(1.0, dtype=dtype, device=device)

    Ey = torch.zeros(n_shots, ny, nx, dtype=dtype, device=device)
    Hx = torch.zeros_like(Ey)
    Hz = torch.zeros_like(Ey)
    dEy = torch.zeros_like(Ey) if dEy_0 is None else dEy_0.clone()
    dHx = torch.zeros_like(Ey)
    dHz = torch.zeros_like(Ey)
    dca_eff = dca.unsqueeze(0) if dca.ndim == 2 else dca
    dcb_eff = dcb.unsqueeze(0) if dcb.ndim == 2 else dcb

    if n_sources > 0:
        source_ids = sources_i.reshape(n_shots, n_sources)
        f0_view = f0.reshape(nt, n_shots, n_sources)
        df_view = df.reshape(nt, n_shots, n_sources)

    receivers = []
    for t in range(nt):
        Hx = Hx - cq * staggered.diffyh1(Ey, stencil, rdy)
        Hz = Hz + cq * staggered.diffxh1(Ey, stencil, rdx)
        dHx = dHx - cq * staggered.diffyh1(dEy, stencil, rdy)
        dHz = dHz + cq * staggered.diffxh1(dEy, stencil, rdx)

        curl_h = staggered.diffx1(Hz, stencil, rdx) - staggered.diffy1(Hx, stencil, rdy)
        dcurl_h = staggered.diffx1(dHz, stencil, rdx) - staggered.diffy1(
            dHx, stencil, rdy
        )

        Ey_old = Ey
        Ey = ca * Ey + cb * curl_h
        dEy = ca * dEy + cb * dcurl_h + dca_eff * Ey_old + dcb_eff * curl_h

        if n_sources > 0:
            Ey.view(n_shots, -1).scatter_add_(1, source_ids, f0_view[t])
            dEy.view(n_shots, -1).scatter_add_(1, source_ids, df_view[t])

        receivers.append(
            torch.stack(
                [
                    dEy.view(n_shots, -1)[:, int(flat_idx.item())]
                    for flat_idx in receivers_i.reshape(-1)
                ],
                dim=-1,
            )
        )

    return torch.stack(receivers, dim=0)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_born_bggrad_matches_reference_with_sources():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 1
    n_receivers = 2
    stencil = 2

    source_yx = torch.tensor([[[3, 3]]], dtype=torch.long, device=device)
    receiver_yx = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    sources_i = (source_yx[..., 0] * nx + source_yx[..., 1]).long().contiguous()
    receivers_i = (receiver_yx[..., 0] * nx + receiver_yx[..., 1]).long().contiguous()

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    dcb = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    source = torch.randn(nt, n_shots, n_sources, dtype=dtype, device=device)
    f0 = source.reshape(-1).clone().detach().requires_grad_(True)
    df = (0.15 * source).reshape(-1).clone().detach().requires_grad_(True)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=f0,
        df=df,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
    )
    native_grads = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb, f0, df],
    )

    ca_ref = ca.detach().clone().requires_grad_(True)
    cb_ref = cb.detach().clone().requires_grad_(True)
    dca_ref = dca.detach().clone().requires_grad_(True)
    dcb_ref = dcb.detach().clone().requires_grad_(True)
    f0_ref = f0.detach().clone().requires_grad_(True)
    df_ref = df.detach().clone().requires_grad_(True)
    reference_receivers = _reference_tm2d_born_receivers(
        ca=ca_ref,
        cb=cb_ref,
        cq=cq,
        dca=dca_ref,
        dcb=dcb_ref,
        f0=f0_ref,
        df=df_ref,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        stencil=stencil,
    )
    reference_grads = torch.autograd.grad(
        torch.sum(reference_receivers * residual),
        [ca_ref, cb_ref, dca_ref, dcb_ref, f0_ref, df_ref],
    )

    errors = {
        name: float((native_grad - reference_grad).norm())
        / (float(reference_grad.norm()) + 1e-30)
        for name, native_grad, reference_grad in zip(
            ("ca", "cb", "dca", "dcb", "f0", "df"), native_grads, reference_grads
        )
    }
    assert max(errors.values()) < 1e-9, errors


@pytest.mark.parametrize(
    "device_type", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_born_bggrad_matches_reference_without_sources(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for TM2D Born background-gradient parity.")
    torch.manual_seed(2)
    device = torch.device(device_type)
    dtype = torch.float64
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 0
    n_receivers = 2
    stencil = 2

    receivers_i = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    receivers_flat = (
        (receivers_i[..., 0] * nx + receivers_i[..., 1]).long().contiguous()
    )

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = torch.zeros(ny, nx, dtype=dtype, device=device).requires_grad_()
    dcb = torch.zeros(ny, nx, dtype=dtype, device=device).requires_grad_()
    dEy_0 = torch.randn(n_shots, ny, nx, dtype=dtype, device=device)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        df=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        sources_i=torch.empty(0, dtype=torch.long, device=device),
        receivers_i=receivers_flat,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
        dEy_0=dEy_0,
    )
    native_grad_ca, native_grad_cb, _, _ = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb],
    )

    ca_ref = ca.detach().clone().requires_grad_(True)
    cb_ref = cb.detach().clone().requires_grad_(True)
    dca_ref = dca.detach().clone().requires_grad_(True)
    dcb_ref = dcb.detach().clone().requires_grad_(True)
    reference_receivers = _reference_tm2d_born_receivers(
        ca=ca_ref,
        cb=cb_ref,
        cq=cq,
        dca=dca_ref,
        dcb=dcb_ref,
        f0=torch.empty(0, dtype=dtype, device=device),
        df=torch.empty(0, dtype=dtype, device=device),
        sources_i=torch.empty(0, dtype=torch.long, device=device),
        receivers_i=receivers_flat,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        dEy_0=dEy_0,
        stencil=stencil,
    )
    reference_grad_ca, reference_grad_cb, _, _ = torch.autograd.grad(
        torch.sum(reference_receivers * residual),
        [ca_ref, cb_ref, dca_ref, dcb_ref],
    )

    tolerance = 1e-6 if device_type == "cuda" else 1e-9
    torch.testing.assert_close(
        native_grad_ca, reference_grad_ca, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        native_grad_cb, reference_grad_cb, atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_born_bggrad_matches_reference_with_bf16_storage(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for TM2D Born BF16 storage test.")

    torch.manual_seed(4)
    device = torch.device(device_type)
    dtype = torch.float32
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 1
    n_receivers = 2
    stencil = 2

    source_yx = torch.tensor([[[3, 3]]], dtype=torch.long, device=device)
    receiver_yx = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    sources_i = (source_yx[..., 0] * nx + source_yx[..., 1]).long().contiguous()
    receivers_i = (receiver_yx[..., 0] * nx + receiver_yx[..., 1]).long().contiguous()

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    dcb = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    source = torch.randn(nt, n_shots, n_sources, dtype=dtype, device=device)
    f0 = source.reshape(-1).clone().detach().requires_grad_(True)
    df = (0.15 * source).reshape(-1).clone().detach().requires_grad_(True)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=f0,
        df=df,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
        storage_compression="bf16",
    )
    native_grads = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb, f0, df],
    )

    ca_full = ca.detach().clone().requires_grad_(True)
    cb_full = cb.detach().clone().requires_grad_(True)
    dca_full = dca.detach().clone().requires_grad_(True)
    dcb_full = dcb.detach().clone().requires_grad_(True)
    f0_full = f0.detach().clone().requires_grad_(True)
    df_full = df.detach().clone().requires_grad_(True)
    full_receivers = _native_tm2d_born_receivers(
        ca=ca_full,
        cb=cb_full,
        cq=cq,
        dca=dca_full,
        dcb=dcb_full,
        f0=f0_full,
        df=df_full,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
    )
    full_grads = torch.autograd.grad(
        torch.sum(full_receivers * residual),
        [ca_full, cb_full, dca_full, dcb_full, f0_full, df_full],
    )

    for native_grad, full_grad in zip(native_grads, full_grads):
        torch.testing.assert_close(native_grad, full_grad, atol=8e-3, rtol=8e-2)
