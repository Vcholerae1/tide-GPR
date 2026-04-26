import pytest
import torch

import tide
from tide import backend_utils
from tide.maxwell.maxwell3d_born_autograd import maxwell3d_receiver_hvp_naive
from tide.maxwell.tm2d_born_autograd import (
    tm2d_receiver_hvp_naive,
    tm2d_receiver_hvp_native,
)


def _nonlinear_receiver_misfit(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
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


def _tm2d_exact_hvp(
    *,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    vepsilon: torch.Tensor,
    vsigma: torch.Tensor,
    source_amplitude: torch.Tensor,
    source_location: torch.Tensor,
    receiver_location: torch.Tensor,
    observed_data: torch.Tensor,
    grid_spacing: float,
    dt: float,
    pml_width: int,
    stencil: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon_req = epsilon.detach().clone().requires_grad_(True)
    sigma_req = sigma.detach().clone().requires_grad_(True)
    predicted = tide.maxwelltm(
        epsilon_req,
        sigma_req,
        mu,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=pml_width,
        stencil=stencil,
        python_backend=True,
    )[-1]
    loss = _nonlinear_receiver_misfit(predicted, observed_data)
    grad_epsilon, grad_sigma = torch.autograd.grad(
        loss,
        (epsilon_req, sigma_req),
        create_graph=True,
    )
    directional = (grad_epsilon * vepsilon).sum() + (grad_sigma * vsigma).sum()
    return torch.autograd.grad(directional, (epsilon_req, sigma_req))


def _maxwell3d_exact_hvp(
    *,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    vepsilon: torch.Tensor,
    vsigma: torch.Tensor,
    source_amplitude: torch.Tensor,
    source_location: torch.Tensor,
    receiver_location: torch.Tensor,
    observed_data: torch.Tensor,
    grid_spacing: tuple[float, float, float],
    dt: float,
    pml_width: int,
    stencil: int,
    source_component: str,
    receiver_component: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon_req = epsilon.detach().clone().requires_grad_(True)
    sigma_req = sigma.detach().clone().requires_grad_(True)
    predicted = tide.maxwell3d(
        epsilon_req,
        sigma_req,
        mu,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=pml_width,
        stencil=stencil,
        source_component=source_component,
        receiver_component=receiver_component,
        python_backend=True,
    )[-1]
    loss = _nonlinear_receiver_misfit(predicted, observed_data)
    grad_epsilon, grad_sigma = torch.autograd.grad(
        loss,
        (epsilon_req, sigma_req),
        create_graph=True,
    )
    directional = (grad_epsilon * vepsilon).sum() + (grad_sigma * vsigma).sum()
    return torch.autograd.grad(directional, (epsilon_req, sigma_req))


def test_tm2d_receiver_hvp_naive_matches_exact_nested_autodiff():
    dtype = torch.float64
    ny, nx = 8, 9
    nt = 12
    dt = 4e-11

    epsilon = torch.full((ny, nx), 4.0, dtype=dtype)
    epsilon[ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1] = 4.3
    sigma = torch.full((ny, nx), 5e-4, dtype=dtype)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[ny // 2, nx // 3]]], dtype=torch.long)
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2], [ny // 2, nx // 2 + 1]]],
        dtype=torch.long,
    )
    source_amplitude = tide.ricker(
        90e6,
        nt,
        dt,
        peak_time=1.0 / 90e6,
        dtype=dtype,
    ).view(1, 1, nt)
    observed_data = torch.zeros(nt, 1, receiver_location.shape[1], dtype=dtype)

    torch.manual_seed(0)
    vepsilon = 0.03 * torch.randn_like(epsilon)
    vepsilon = vepsilon / vepsilon.abs().amax()
    vsigma = 0.02 * torch.randn_like(sigma)
    vsigma = vsigma / vsigma.abs().amax()

    hvp_epsilon_exact, hvp_sigma_exact = _tm2d_exact_hvp(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        vepsilon=vepsilon,
        vsigma=vsigma,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=observed_data,
        grid_spacing=0.02,
        dt=dt,
        pml_width=2,
        stencil=2,
    )
    hvp_epsilon_proto, hvp_sigma_proto = tm2d_receiver_hvp_naive(
        epsilon,
        sigma,
        mu,
        vepsilon=vepsilon,
        vsigma=vsigma,
        grid_spacing=0.02,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=observed_data,
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
    )

    _assert_relative_norm_close(hvp_epsilon_proto, hvp_epsilon_exact, rtol=1e-6)
    _assert_relative_norm_close(hvp_sigma_proto, hvp_sigma_exact, rtol=1e-6)


@torch.no_grad()
def _build_tm2d_native_case():
    dtype = torch.float64
    ny, nx = 8, 9
    nt = 12
    dt = 4e-11

    epsilon = torch.full((ny, nx), 4.0, dtype=dtype)
    epsilon[ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1] = 4.3
    sigma = torch.full((ny, nx), 5e-4, dtype=dtype)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[ny // 2, nx // 3]]], dtype=torch.long)
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2], [ny // 2, nx // 2 + 1]]],
        dtype=torch.long,
    )
    source_amplitude = tide.ricker(
        90e6,
        nt,
        dt,
        peak_time=1.0 / 90e6,
        dtype=dtype,
    ).view(1, 1, nt)
    observed_data = torch.zeros(nt, 1, receiver_location.shape[1], dtype=dtype)

    torch.manual_seed(4)
    vepsilon = 0.03 * torch.randn_like(epsilon)
    vepsilon = vepsilon / vepsilon.abs().amax()
    vsigma = 0.02 * torch.randn_like(sigma)
    vsigma = vsigma / vsigma.abs().amax()
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "vepsilon": vepsilon,
        "vsigma": vsigma,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "observed_data": observed_data,
        "grid_spacing": 0.02,
        "dt": dt,
    }


def _tm2d_native_case_on(device: torch.device) -> dict[str, torch.Tensor | float]:
    case = _build_tm2d_native_case()
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in case.items()
    }


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_receiver_hvp_native_matches_exact_nested_autodiff():
    case = _tm2d_native_case_on(torch.device("cpu"))

    hvp_epsilon_exact, hvp_sigma_exact = _tm2d_exact_hvp(
        epsilon=case["epsilon"],
        sigma=case["sigma"],
        mu=case["mu"],
        vepsilon=case["vepsilon"],
        vsigma=case["vsigma"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        pml_width=0,
        stencil=2,
    )
    hvp_epsilon_native, hvp_sigma_native = tm2d_receiver_hvp_native(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        vepsilon=case["vepsilon"],
        vsigma=case["vsigma"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=0,
        stencil=2,
    )

    _assert_relative_norm_close(hvp_epsilon_native, hvp_epsilon_exact, rtol=1e-10)
    _assert_relative_norm_close(hvp_sigma_native, hvp_sigma_exact, rtol=1e-10)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_receiver_hvp_native_cuda_matches_exact_nested_autodiff_with_pml():
    case = _tm2d_native_case_on(torch.device("cuda"))

    hvp_epsilon_exact, hvp_sigma_exact = _tm2d_exact_hvp(
        epsilon=case["epsilon"],
        sigma=case["sigma"],
        mu=case["mu"],
        vepsilon=case["vepsilon"],
        vsigma=case["vsigma"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        pml_width=2,
        stencil=2,
    )
    hvp_epsilon_native, hvp_sigma_native = tm2d_receiver_hvp_native(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        vepsilon=case["vepsilon"],
        vsigma=case["vsigma"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
    )

    _assert_relative_norm_close(hvp_epsilon_native, hvp_epsilon_exact, rtol=1e-5)
    _assert_relative_norm_close(hvp_sigma_native, hvp_sigma_exact, rtol=1e-5)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_receiver_hvp_native_matches_exact_nested_autodiff_with_pml_cpu():
    case = _tm2d_native_case_on(torch.device("cpu"))

    hvp_epsilon_exact, hvp_sigma_exact = _tm2d_exact_hvp(
        epsilon=case["epsilon"],
        sigma=case["sigma"],
        mu=case["mu"],
        vepsilon=case["vepsilon"],
        vsigma=case["vsigma"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        pml_width=2,
        stencil=2,
    )
    hvp_epsilon_native, hvp_sigma_native = tm2d_receiver_hvp_native(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        vepsilon=case["vepsilon"],
        vsigma=case["vsigma"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        misfit_fn=_nonlinear_receiver_misfit,
        pml_width=2,
        stencil=2,
    )

    _assert_relative_norm_close(hvp_epsilon_native, hvp_epsilon_exact, rtol=1e-6)
    _assert_relative_norm_close(hvp_sigma_native, hvp_sigma_exact, rtol=1e-6)


def test_maxwell3d_receiver_hvp_naive_matches_exact_nested_autodiff():
    dtype = torch.float64
    nz, ny, nx = 5, 6, 7
    nt = 8
    dt = 4e-11

    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype)
    epsilon[nz // 2 - 1 : nz // 2 + 1, ny // 2, nx // 2] = 4.25
    sigma = torch.full((nz, ny, nx), 3e-4, dtype=dtype)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 2, 1]]], dtype=torch.long)
    receiver_location = torch.tensor(
        [[[2, 2, 4], [2, 2, 5]]],
        dtype=torch.long,
    )
    source_amplitude = tide.ricker(
        80e6,
        nt,
        dt,
        peak_time=1.0 / 80e6,
        dtype=dtype,
    ).view(1, 1, nt)
    observed_data = torch.zeros(nt, 1, receiver_location.shape[1], dtype=dtype)

    torch.manual_seed(1)
    vepsilon = 0.03 * torch.randn_like(epsilon)
    vepsilon = vepsilon / vepsilon.abs().amax()
    vsigma = 0.02 * torch.randn_like(sigma)
    vsigma = vsigma / vsigma.abs().amax()

    hvp_epsilon_exact, hvp_sigma_exact = _maxwell3d_exact_hvp(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        vepsilon=vepsilon,
        vsigma=vsigma,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=observed_data,
        grid_spacing=(0.03, 0.02, 0.02),
        dt=dt,
        pml_width=2,
        stencil=2,
        source_component="ey",
        receiver_component="ey",
    )
    hvp_epsilon_proto, hvp_sigma_proto = maxwell3d_receiver_hvp_naive(
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

    _assert_relative_norm_close(hvp_epsilon_proto, hvp_epsilon_exact, rtol=1e-6)
    _assert_relative_norm_close(hvp_sigma_proto, hvp_sigma_exact, rtol=1e-6)
