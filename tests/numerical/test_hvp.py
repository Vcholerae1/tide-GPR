from __future__ import annotations

import importlib
import pytest
import tide
import torch
from numerical_utils import MaxwellExample, make_maxwell3d_example, make_tm2d_example
from tide import backend_utils
from tide.maxwell.maxwell3d_born_autograd import (
    maxwell3d_receiver_hvp_naive,
    maxwell3d_receiver_hvp_native,
)
from tide.maxwell.tm2d_born_autograd import (
    tm2d_receiver_hvp_naive,
    tm2d_receiver_hvp_native,
)

# --- test_hvp_naive.py ---


def _nonlinear_receiver_misfit(
    predicted: torch.Tensor, observed: torch.Tensor
) -> torch.Tensor:
    residual = predicted - observed
    return 0.5 * residual.square().sum() + 0.01 * predicted.sin().sum()


def _assert_relative_norm_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
) -> None:
    diff = torch.linalg.norm(actual - expected)
    ref = torch.linalg.norm(expected)
    rel = float(diff / (ref + 1e-30))
    assert rel < rtol, f"relative HVP error {rel} exceeds tolerance {rtol}"


def _assert_finite_nonzero_hvp(*parts: torch.Tensor) -> None:
    flat = torch.cat([part.reshape(-1).double().cpu() for part in parts])
    assert torch.isfinite(flat).all()
    assert flat.norm() > 0


def _fail_if_separate_forward_is_called(*args, **kwargs):
    del args, kwargs
    raise AssertionError("HVP must obtain the background trace from its Born pass")


def _exact_hvp(
    example: MaxwellExample,
    *,
    vepsilon: torch.Tensor,
    vsigma: torch.Tensor,
    observed_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    predicted = example.run(
        epsilon=epsilon,
        sigma=sigma,
        python_backend=True,
    )[-1]
    loss = _nonlinear_receiver_misfit(predicted, observed_data)
    grad_epsilon, grad_sigma = torch.autograd.grad(
        loss,
        (epsilon, sigma),
        create_graph=True,
    )
    directional = (grad_epsilon * vepsilon).sum() + (grad_sigma * vsigma).sum()
    return torch.autograd.grad(directional, (epsilon, sigma))


def test_tm2d_receiver_hvp_naive_matches_exact_nested_autodiff():
    example = make_tm2d_example(
        shape=(8, 9),
        nt=12,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=90e6,
        dtype=torch.float64,
        sigma=5e-4,
        source_location=(4, 3),
        receiver_locations=((4, 4), (4, 5)),
        pml_width=2,
        stencil=2,
    )
    epsilon = example.epsilon.clone()
    epsilon[3:5, 3:5] = 4.3
    example = example.updated(epsilon=epsilon)
    observed_data = example.receiver_zeros()

    torch.manual_seed(0)
    vepsilon = 0.03 * torch.randn_like(example.epsilon)
    vepsilon = vepsilon / vepsilon.abs().amax()
    vsigma = 0.02 * torch.randn_like(example.sigma)
    vsigma = vsigma / vsigma.abs().amax()

    exact = _exact_hvp(
        example,
        vepsilon=vepsilon,
        vsigma=vsigma,
        observed_data=observed_data,
    )
    actual = tm2d_receiver_hvp_naive(
        example.epsilon,
        example.sigma,
        example.mu,
        vepsilon=vepsilon,
        vsigma=vsigma,
        grid_spacing=example.grid_spacing,
        dt=example.dt,
        source_amplitude=example.source_amplitude,
        source_location=example.source_location,
        receiver_location=example.receiver_location,
        observed_data=observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=example.pml_width,
        stencil=example.stencil,
    )

    for actual_part, exact_part in zip(actual, exact, strict=True):
        _assert_relative_norm_close(actual_part, exact_part, rtol=1e-6)


def test_tm2d_receiver_hvp_naive_does_not_run_a_separate_forward(monkeypatch):
    case = _build_tm2d_native_example()
    forward_module = importlib.import_module("tide.maxwell.tm2d")
    monkeypatch.setattr(
        forward_module,
        "maxwelltm",
        _fail_if_separate_forward_is_called,
    )

    result = tm2d_receiver_hvp_naive(
        case.epsilon,
        case.sigma,
        case.mu,
        vepsilon=case.vepsilon,
        vsigma=case.vsigma,
        grid_spacing=case.grid_spacing,
        dt=case.dt,
        source_amplitude=case.source_amplitude,
        source_location=case.source_location,
        receiver_location=case.receiver_location,
        observed_data=case.observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
    )

    _assert_finite_nonzero_hvp(*result)


def test_tm2d_full_and_gauss_newton_hvp_match_at_zero_least_squares_residual():
    case = _build_tm2d_native_example()
    predicted = tide.maxwell._kernel_api.maxwelltm(
        case.epsilon,
        case.sigma,
        case.mu,
        grid_spacing=case.grid_spacing,
        dt=case.dt,
        source_amplitude=case.source_amplitude,
        source_location=case.source_location,
        receiver_location=case.receiver_location,
        pml_width=2,
        stencil=2,
        python_backend=True,
    )[-1].detach()

    common = {
        "epsilon": case.epsilon,
        "sigma": case.sigma,
        "mu": case.mu,
        "vepsilon": case.vepsilon,
        "vsigma": case.vsigma,
        "grid_spacing": case.grid_spacing,
        "dt": case.dt,
        "source_amplitude": case.source_amplitude,
        "source_location": case.source_location,
        "receiver_location": case.receiver_location,
        "observed_data": predicted,
        "misfit_fn": lambda actual, observed: 0.5 * (actual - observed).square().sum(),
        "pml_width": 2,
        "stencil": 2,
    }
    full = tm2d_receiver_hvp_naive(**common, hessian_mode="full")
    gauss_newton = tm2d_receiver_hvp_naive(
        **common,
        hessian_mode="gauss_newton",
    )

    for full_part, gn_part in zip(full, gauss_newton):
        torch.testing.assert_close(full_part, gn_part, rtol=1e-8, atol=1e-12)


def test_tm2d_gauss_newton_hvp_is_symmetric_positive_semidefinite():
    case = _build_tm2d_native_example()
    generator = torch.Generator().manual_seed(17)
    u = (case.vepsilon, case.vsigma)
    v = (
        torch.randn(case.epsilon.shape, generator=generator, dtype=torch.float64),
        torch.randn(case.sigma.shape, generator=generator, dtype=torch.float64),
    )

    def apply(direction: tuple[torch.Tensor, torch.Tensor]):
        return tm2d_receiver_hvp_naive(
            case.epsilon,
            case.sigma,
            case.mu,
            vepsilon=direction[0],
            vsigma=direction[1],
            grid_spacing=case.grid_spacing,
            dt=case.dt,
            source_amplitude=case.source_amplitude,
            source_location=case.source_location,
            receiver_location=case.receiver_location,
            observed_data=case.observed_data,
            misfit_fn=lambda actual, observed: 0.5 * (actual - observed).square().sum(),
            pml_width=2,
            stencil=2,
            hessian_mode="gauss_newton",
        )

    hu = apply(u)
    hv = apply(v)
    lhs = sum(torch.sum(left * right) for left, right in zip(u, hv, strict=True))
    rhs = sum(torch.sum(left * right) for left, right in zip(hu, v, strict=True))
    scale = torch.maximum(lhs.abs(), rhs.abs())
    assert scale > 0.0
    assert float((lhs - rhs).abs() / scale) <= 1.0e-6
    quadratic = sum(
        torch.sum(direction * product) for direction, product in zip(u, hu, strict=True)
    )
    direction_norm_sq = sum(torch.sum(direction.square()) for direction in u)
    assert float(quadratic) >= -1.0e-10 * float(direction_norm_sq)


@torch.no_grad()
def _build_tm2d_native_example(
    device: torch.device = torch.device("cpu"),
) -> MaxwellExample:
    example = make_tm2d_example(
        shape=(8, 9),
        nt=12,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=90e6,
        dtype=torch.float64,
        device=device,
        sigma=5e-4,
        source_location=(4, 3),
        receiver_locations=((4, 4), (4, 5)),
        pml_width=2,
        stencil=2,
    )
    epsilon = example.epsilon.clone()
    epsilon[3:5, 3:5] = 4.3
    torch.manual_seed(4)
    vepsilon = 0.03 * torch.randn_like(epsilon)
    vsigma = 0.02 * torch.randn_like(example.sigma)
    return example.updated(
        epsilon=epsilon,
        vepsilon=vepsilon / vepsilon.abs().amax(),
        vsigma=vsigma / vsigma.abs().amax(),
        observed_data=example.receiver_zeros(),
    )


def _tm2d_native_example_on(device: torch.device) -> MaxwellExample:
    return _build_tm2d_native_example(device)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_receiver_hvp_native_does_not_run_a_separate_forward(monkeypatch):
    case = _tm2d_native_example_on(torch.device("cpu"))
    forward_module = importlib.import_module("tide.maxwell.tm2d")
    monkeypatch.setattr(
        forward_module,
        "maxwelltm",
        _fail_if_separate_forward_is_called,
    )

    result = tm2d_receiver_hvp_native(
        case.epsilon,
        case.sigma,
        case.mu,
        vepsilon=case.vepsilon,
        vsigma=case.vsigma,
        grid_spacing=case.grid_spacing,
        dt=case.dt,
        source_amplitude=case.source_amplitude,
        source_location=case.source_location,
        receiver_location=case.receiver_location,
        observed_data=case.observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=0,
        stencil=2,
    )

    _assert_finite_nonzero_hvp(*result)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_receiver_hvp_native_cuda_returns_coeff_hvp_with_pml():
    case = _tm2d_native_example_on(torch.device("cuda"))

    hvp_epsilon_native, hvp_sigma_native = tm2d_receiver_hvp_native(
        case.epsilon,
        case.sigma,
        case.mu,
        vepsilon=case.vepsilon,
        vsigma=case.vsigma,
        grid_spacing=case.grid_spacing,
        dt=case.dt,
        source_amplitude=case.source_amplitude,
        source_location=case.source_location,
        receiver_location=case.receiver_location,
        observed_data=case.observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
    )

    _assert_finite_nonzero_hvp(hvp_epsilon_native, hvp_sigma_native)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_receiver_hvp_native_returns_coeff_hvp_with_pml_cpu():
    case = _tm2d_native_example_on(torch.device("cpu"))

    hvp_epsilon_native, hvp_sigma_native = tm2d_receiver_hvp_native(
        case.epsilon,
        case.sigma,
        case.mu,
        vepsilon=case.vepsilon,
        vsigma=case.vsigma,
        grid_spacing=case.grid_spacing,
        dt=case.dt,
        source_amplitude=case.source_amplitude,
        source_location=case.source_location,
        receiver_location=case.receiver_location,
        observed_data=case.observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
    )

    _assert_finite_nonzero_hvp(hvp_epsilon_native, hvp_sigma_native)


def test_maxwell3d_receiver_hvp_naive_matches_exact_nested_autodiff():
    example = make_maxwell3d_example(
        shape=(5, 6, 7),
        nt=8,
        grid_spacing=(0.03, 0.02, 0.02),
        dt=4e-11,
        frequency=80e6,
        dtype=torch.float64,
        sigma=3e-4,
        source_location=(2, 2, 1),
        receiver_locations=((2, 2, 4), (2, 2, 5)),
        pml_width=2,
        stencil=2,
    )
    epsilon = example.epsilon.clone()
    epsilon[1:3, 3, 3] = 4.25
    example = example.updated(epsilon=epsilon)
    observed_data = example.receiver_zeros()

    torch.manual_seed(1)
    vepsilon = 0.03 * torch.randn_like(example.epsilon)
    vepsilon = vepsilon / vepsilon.abs().amax()
    vsigma = 0.02 * torch.randn_like(example.sigma)
    vsigma = vsigma / vsigma.abs().amax()

    exact = _exact_hvp(
        example,
        vepsilon=vepsilon,
        vsigma=vsigma,
        observed_data=observed_data,
    )
    actual = maxwell3d_receiver_hvp_naive(
        example.epsilon,
        example.sigma,
        example.mu,
        vepsilon=vepsilon,
        vsigma=vsigma,
        grid_spacing=example.grid_spacing,
        dt=example.dt,
        source_amplitude=example.source_amplitude,
        source_location=example.source_location,
        receiver_location=example.receiver_location,
        observed_data=observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=example.pml_width,
        stencil=example.stencil,
        source_component=example.source_component,
        receiver_component=example.receiver_component,
    )

    for actual_part, exact_part in zip(actual, exact, strict=True):
        _assert_relative_norm_close(actual_part, exact_part, rtol=1e-6)


def test_maxwell3d_receiver_hvp_naive_does_not_run_a_separate_forward(monkeypatch):
    dtype = torch.float64
    nz, ny, nx = 5, 6, 7
    nt = 8
    dt = 4e-11
    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype)
    sigma = torch.full_like(epsilon, 3e-4)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[2, 2, 1]]], dtype=torch.long)
    receiver_location = torch.tensor([[[2, 2, 4]]], dtype=torch.long)
    source_amplitude = tide.ricker(
        80e6, nt, dt, peak_time=1.0 / 80e6, dtype=dtype
    ).view(1, 1, nt)
    forward_module = importlib.import_module("tide.maxwell.maxwell3d")
    monkeypatch.setattr(
        forward_module,
        "maxwell3d",
        _fail_if_separate_forward_is_called,
    )

    result = maxwell3d_receiver_hvp_naive(
        epsilon,
        sigma,
        mu,
        vepsilon=torch.full_like(epsilon, 0.01),
        vsigma=torch.full_like(sigma, 0.01),
        grid_spacing=(0.03, 0.02, 0.02),
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=torch.zeros(nt, 1, 1, dtype=dtype),
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
        source_component="ey",
        receiver_component="ey",
    )

    _assert_finite_nonzero_hvp(*result)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_maxwell3d_receiver_hvp_native_does_not_run_a_separate_forward(monkeypatch):
    dtype = torch.float64
    nz, ny, nx = 5, 6, 7
    nt = 8
    dt = 4e-11
    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype)
    sigma = torch.full_like(epsilon, 3e-4)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[2, 2, 1]]], dtype=torch.long)
    receiver_location = torch.tensor([[[2, 2, 4]]], dtype=torch.long)
    source_amplitude = tide.ricker(
        80e6, nt, dt, peak_time=1.0 / 80e6, dtype=dtype
    ).view(1, 1, nt)
    forward_module = importlib.import_module("tide.maxwell.maxwell3d")
    monkeypatch.setattr(
        forward_module,
        "maxwell3d",
        _fail_if_separate_forward_is_called,
    )

    result = maxwell3d_receiver_hvp_native(
        epsilon,
        sigma,
        mu,
        vepsilon=torch.full_like(epsilon, 0.01),
        vsigma=torch.full_like(sigma, 0.01),
        grid_spacing=(0.03, 0.02, 0.02),
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=torch.zeros(nt, 1, 1, dtype=dtype),
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
        source_component="ey",
        receiver_component="ey",
    )

    _assert_finite_nonzero_hvp(*result)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_maxwell3d_receiver_hvp_native_cuda_returns_coeff_hvp():
    device = torch.device("cuda")
    dtype = torch.float32
    nz, ny, nx = 5, 6, 7
    nt = 8
    dt = 4e-11

    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    epsilon[nz // 2 - 1 : nz // 2 + 1, ny // 2, nx // 2] = 4.25
    sigma = torch.full((nz, ny, nx), 3e-4, device=device, dtype=dtype)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 2, 1]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor(
        [[[2, 2, 4], [2, 2, 5]]],
        dtype=torch.long,
        device=device,
    )
    source_amplitude = tide.ricker(
        80e6,
        nt,
        dt,
        peak_time=1.0 / 80e6,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    observed_data = torch.zeros(
        nt, 1, receiver_location.shape[1], device=device, dtype=dtype
    )

    torch.manual_seed(11)
    vepsilon = 0.03 * torch.randn_like(epsilon)
    vepsilon = vepsilon / vepsilon.abs().amax()
    vsigma = 0.02 * torch.randn_like(sigma)
    vsigma = vsigma / vsigma.abs().amax()

    hvp_epsilon_native, hvp_sigma_native = maxwell3d_receiver_hvp_native(
        epsilon,
        sigma,
        mu,
        vepsilon=vepsilon,
        vsigma=vsigma,
        grid_spacing=(0.03, 0.02, 0.02),
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
        source_component="ey",
        receiver_component="ey",
    )

    _assert_finite_nonzero_hvp(hvp_epsilon_native, hvp_sigma_native)
