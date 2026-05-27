import pytest
import torch

import tide
from tide import backend_utils


def _receiver_misfit(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    residual = predicted - observed
    return 0.5 * residual.square().sum() + 0.01 * predicted.sin().sum()


def _tm_observed_data(
    case: dict[str, torch.Tensor | float], device: torch.device
) -> torch.Tensor:
    source_amplitude = case["source_amplitude"]
    receiver_location = case["receiver_location"]
    epsilon = case["epsilon"]
    assert isinstance(source_amplitude, torch.Tensor)
    assert isinstance(receiver_location, torch.Tensor)
    assert isinstance(epsilon, torch.Tensor)
    return torch.zeros(
        source_amplitude.shape[-1],
        1,
        receiver_location.shape[1],
        device=device,
        dtype=epsilon.dtype,
    )


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


def _build_3d_case(device: torch.device):
    dtype = torch.float64
    nz, ny, nx = 5, 6, 7
    nt = 8
    dt = 4e-11

    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.full((nz, ny, nx), 3e-4, device=device, dtype=dtype)
    mu = torch.ones_like(epsilon)
    depsilon = torch.full_like(epsilon, 0.02)

    source_location = torch.tensor([[[2, 2, 1]]], device=device)
    receiver_location = torch.tensor([[[2, 2, 4], [2, 2, 5]]], device=device)
    torch.manual_seed(1)
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
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "depsilon": depsilon,
        "dsigma": torch.full_like(sigma, 0.01),
        "grid_spacing": (0.03, 0.02, 0.02),
        "dt": dt,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "source_amplitude": source_amplitude,
        "observed_data": observed_data,
        "source_component": "ey",
        "receiver_component": "ey",
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


def test_maxwelltm_hvp_module_matches_functional_cpu():
    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)
    vsigma = torch.full_like(case["sigma"], 0.01)

    model = tide.MaxwellTM(
        case["epsilon"], case["sigma"], case["mu"], grid_spacing=case["dx"]
    )

    module_hvp = model.hvp(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        vsigma=vsigma,
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=1,
    )
    func_hvp = tide.maxwelltm_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        vsigma=vsigma,
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=1,
    )

    for module_out, func_out in zip(module_hvp, func_hvp):
        torch.testing.assert_close(module_out, func_out)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_maxwelltm_hvp_native_module_matches_functional_cpu():
    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)
    vsigma = torch.full_like(case["sigma"], 0.01)

    model = tide.MaxwellTM(
        case["epsilon"], case["sigma"], case["mu"], grid_spacing=case["dx"]
    )

    module_hvp = model.hvp(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        vsigma=vsigma,
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=0,
        python_backend=False,
    )
    func_hvp = tide.maxwelltm_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        vsigma=vsigma,
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=0,
        python_backend=False,
    )

    for module_out, func_out in zip(module_hvp, func_hvp):
        torch.testing.assert_close(module_out, func_out)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_maxwelltm_hvp_native_supports_nonzero_pml_cpu():
    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)
    module_hvp = tide.MaxwellTM(
        case["epsilon"], case["sigma"], case["mu"], grid_spacing=case["dx"]
    ).hvp(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=1,
        python_backend=False,
    )
    func_hvp = tide.maxwelltm_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=1,
        python_backend=False,
    )

    for module_out, func_out in zip(module_hvp, func_hvp):
        torch.testing.assert_close(module_out, func_out)


def test_maxwelltm_hvp_python_backend_rejects_gradient_sampling_interval_gt1():
    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)

    with pytest.raises(
        NotImplementedError,
        match="Python TM2D HVP currently requires model_gradient_sampling_interval in \\{0, 1\\}.",
    ):
        tide.maxwelltm_hvp(
            case["epsilon"],
            case["sigma"],
            case["mu"],
            grid_spacing=case["dx"],
            dt=case["dt"],
            source_amplitude=case["source_amplitude"],
            source_location=case["source_location"],
            receiver_location=case["receiver_location"],
            observed_data=observed_data,
            vepsilon=case["depsilon"],
            stencil=2,
            pml_width=1,
            model_gradient_sampling_interval=2,
            python_backend=True,
        )


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_maxwelltm_hvp_native_cpu_rejects_gradient_sampling_interval_gt1():
    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)

    with pytest.raises(
        NotImplementedError,
        match="Native TM2D HVP on CPU currently requires model_gradient_sampling_interval in \\{0, 1\\}.",
    ):
        tide.maxwelltm_hvp(
            case["epsilon"],
            case["sigma"],
            case["mu"],
            grid_spacing=case["dx"],
            dt=case["dt"],
            source_amplitude=case["source_amplitude"],
            source_location=case["source_location"],
            receiver_location=case["receiver_location"],
            observed_data=observed_data,
            vepsilon=case["depsilon"],
            stencil=2,
            pml_width=1,
            model_gradient_sampling_interval=2,
            python_backend=False,
        )


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_maxwelltm_hvp_native_cuda_supports_gradient_sampling_interval():
    device = torch.device("cuda")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)
    model = tide.MaxwellTM(
        case["epsilon"], case["sigma"], case["mu"], grid_spacing=case["dx"]
    )

    module_hvp = model.hvp(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=1,
        model_gradient_sampling_interval=2,
        python_backend=False,
    )
    func_hvp = tide.maxwelltm_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=1,
        model_gradient_sampling_interval=2,
        python_backend=False,
    )
    baseline_hvp = tide.maxwelltm_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=observed_data,
        vepsilon=case["depsilon"],
        misfit=_receiver_misfit,
        stencil=2,
        pml_width=1,
        model_gradient_sampling_interval=1,
        python_backend=False,
    )

    for module_out, func_out in zip(module_hvp, func_hvp):
        torch.testing.assert_close(module_out, func_out)
        assert torch.isfinite(module_out).all()

    assert any(
        not torch.allclose(sampled_out, baseline_out)
        for sampled_out, baseline_out in zip(func_hvp, baseline_hvp)
    )


def test_maxwell3d_hvp_module_matches_functional_cpu():
    device = torch.device("cpu")
    case = _build_3d_case(device)

    model = tide.Maxwell3D(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["grid_spacing"],
    )

    module_hvp = model.hvp(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        vepsilon=case["depsilon"],
        vsigma=case["dsigma"],
        stencil=2,
        pml_width=1,
        source_component=case["source_component"],
        receiver_component=case["receiver_component"],
    )
    func_hvp = tide.maxwell3d_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        vepsilon=case["depsilon"],
        vsigma=case["dsigma"],
        stencil=2,
        pml_width=1,
        source_component=case["source_component"],
        receiver_component=case["receiver_component"],
    )

    for module_out, func_out in zip(module_hvp, func_hvp):
        torch.testing.assert_close(module_out, func_out)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_maxwell3d_hvp_native_module_matches_functional_cpu():
    device = torch.device("cpu")
    case = _build_3d_case(device)

    model = tide.Maxwell3D(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["grid_spacing"],
    )

    module_hvp = model.hvp(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        vepsilon=case["depsilon"],
        vsigma=case["dsigma"],
        stencil=2,
        pml_width=1,
        source_component=case["source_component"],
        receiver_component=case["receiver_component"],
        python_backend=False,
    )
    func_hvp = tide.maxwell3d_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        vepsilon=case["depsilon"],
        vsigma=case["dsigma"],
        stencil=2,
        pml_width=1,
        source_component=case["source_component"],
        receiver_component=case["receiver_component"],
        python_backend=False,
    )

    for module_out, func_out in zip(module_hvp, func_hvp):
        torch.testing.assert_close(module_out, func_out)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_maxwell3d_hvp_native_cuda_supports_gradient_sampling_interval():
    device = torch.device("cuda")
    case = _build_3d_case(device)

    model = tide.Maxwell3D(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["grid_spacing"],
    )

    module_hvp = model.hvp(
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        vepsilon=case["depsilon"],
        vsigma=case["dsigma"],
        stencil=2,
        pml_width=1,
        model_gradient_sampling_interval=2,
        source_component=case["source_component"],
        receiver_component=case["receiver_component"],
        python_backend=False,
    )
    func_hvp = tide.maxwell3d_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        vepsilon=case["depsilon"],
        vsigma=case["dsigma"],
        stencil=2,
        pml_width=1,
        model_gradient_sampling_interval=2,
        source_component=case["source_component"],
        receiver_component=case["receiver_component"],
        python_backend=False,
    )
    baseline_hvp = tide.maxwell3d_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["grid_spacing"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=case["observed_data"],
        vepsilon=case["depsilon"],
        vsigma=case["dsigma"],
        stencil=2,
        pml_width=1,
        model_gradient_sampling_interval=1,
        source_component=case["source_component"],
        receiver_component=case["receiver_component"],
        python_backend=False,
    )

    for module_out, func_out in zip(module_hvp, func_hvp):
        torch.testing.assert_close(module_out, func_out)
        assert torch.isfinite(module_out).all()

    assert any(
        not torch.allclose(sampled_out, baseline_out)
        for sampled_out, baseline_out in zip(func_hvp, baseline_hvp)
    )
