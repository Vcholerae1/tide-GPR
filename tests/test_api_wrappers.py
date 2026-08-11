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


def test_maxwelltm_hvp_defaults_to_native_backend(monkeypatch):
    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)
    marker = (
        torch.full_like(case["epsilon"], 3.0),
        torch.full_like(case["sigma"], 4.0),
    )

    def fake_native(*_args, **_kwargs):
        return marker

    def fail_python(*_args, **_kwargs):
        raise AssertionError("default TM2D HVP used the Python reference backend")

    monkeypatch.setattr(
        "tide.maxwell.tm2d_born_autograd.tm2d_receiver_hvp_native",
        fake_native,
    )
    monkeypatch.setattr(
        "tide.maxwell.tm2d_born_autograd.tm2d_receiver_hvp_naive",
        fail_python,
    )

    result = tide.maxwelltm_hvp(
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
        pml_width=1,
    )

    assert result is marker


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_maxwelltm_gauss_newton_hvp_matches_full_at_zero_residual_cpu():
    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = tide.maxwelltm(
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
        python_backend=False,
        storage_compression=False,
    )[-1].detach()
    kwargs = {
        "epsilon": case["epsilon"],
        "sigma": case["sigma"],
        "mu": case["mu"],
        "grid_spacing": case["dx"],
        "dt": case["dt"],
        "source_amplitude": case["source_amplitude"],
        "source_location": case["source_location"],
        "receiver_location": case["receiver_location"],
        "observed_data": observed_data,
        "vepsilon": case["depsilon"],
        "stencil": 2,
        "pml_width": 1,
        "python_backend": False,
        "storage_compression": False,
    }

    full = tide.maxwelltm_hvp(**kwargs, hessian_mode="full")
    gauss_newton = tide.maxwelltm_hvp(
        **kwargs,
        hessian_mode="gauss_newton",
    )

    for full_part, gn_part in zip(full, gauss_newton):
        torch.testing.assert_close(full_part, gn_part, rtol=2e-5, atol=1e-10)


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


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_full_hvp_equals_gn_plus_second_order_vjp_cpu():
    from tide.maxwell.tm2d_born_autograd import (
        tm2d_receiver_second_order_vjp_native,
    )

    device = torch.device("cpu")
    case = _build_tm_case(device)
    observed_data = _tm_observed_data(case, device)
    forward_outputs = tide.maxwelltm(
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
    )
    predicted_data = forward_outputs[-1]
    data_gradient = predicted_data - observed_data + 0.01 * predicted_data.cos()
    common = {
        "grid_spacing": case["dx"],
        "dt": case["dt"],
        "source_amplitude": case["source_amplitude"],
        "source_location": case["source_location"],
        "receiver_location": case["receiver_location"],
        "stencil": 2,
        "pml_width": 1,
    }
    hvp_common = {
        **common,
        "observed_data": observed_data,
        "vepsilon": case["depsilon"],
        "misfit": _receiver_misfit,
    }
    full = tide.maxwelltm_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        hessian_mode="full",
        **hvp_common,
    )
    gauss_newton = tide.maxwelltm_hvp(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        hessian_mode="gauss_newton",
        **hvp_common,
    )
    correction = tm2d_receiver_second_order_vjp_native(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        vepsilon=case["depsilon"],
        data_gradient=data_gradient,
        **common,
    )
    for full_part, gn_part, correction_part in zip(full, gauss_newton, correction):
        torch.testing.assert_close(
            full_part,
            gn_part + correction_part,
            rtol=2e-4,
            atol=2e-5,
        )


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


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native CUDA backend not available",
)
def test_tm2d_linearization_context_reuses_background_for_direction_batch():
    device = torch.device("cuda")
    case = _build_tm_case(device)
    source_amplitude = torch.cat(
        [case["source_amplitude"], 0.7 * case["source_amplitude"]], dim=0
    )
    source_location = case["source_location"].repeat(2, 1, 1)
    receiver_location = case["receiver_location"].repeat(2, 1, 1)
    observed_data = torch.zeros(
        source_amplitude.shape[-1],
        2,
        receiver_location.shape[1],
        device=device,
        dtype=case["epsilon"].dtype,
    )
    vepsilon = torch.stack(
        [case["depsilon"], 2.0 * case["depsilon"], -case["depsilon"]]
    )
    kwargs = {
        "grid_spacing": case["dx"],
        "dt": case["dt"],
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "observed_data": observed_data,
        "stencil": 2,
        "pml_width": 1,
        "storage_compression": False,
    }

    with tide.linearize_maxwelltm(
        case["epsilon"], case["sigma"], case["mu"], **kwargs
    ) as context:
        actual = context.hvp_batch(vepsilon=vepsilon, block_size=2)
        assert context.background_builds == 1
        assert context.reused_directions == 2
        assert context.batched_blocks == 1
        assert context.predicted_data is not None

    expected_parts = [
        tide.maxwelltm_hvp(
            case["epsilon"],
            case["sigma"],
            case["mu"],
            vepsilon=direction,
            **kwargs,
        )
        for direction in vepsilon
    ]
    expected = tuple(torch.stack(parts) for parts in zip(*expected_parts))
    for actual_part, expected_part in zip(actual, expected):
        relative_l2 = (actual_part - expected_part).norm() / expected_part.norm()
        assert relative_l2 < 2e-5
        assert torch.isfinite(actual_part).all()


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native CUDA backend not available",
)
@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("storage_compression", "relative_tolerance"),
    [(False, 5e-5), ("bf16", 1e-2)],
)
def test_tm2d_linearization_context_fuses_gauss_newton_directions(
    block_size, storage_compression, relative_tolerance, monkeypatch
):
    def reject_full_hvp_backend(*_args, **_kwargs):
        raise AssertionError(
            "Gauss-Newton HVP must not resolve the full-HVP "
            "incremental-adjoint backend."
        )

    monkeypatch.setattr(
        backend_utils,
        "get_tm2d_full_hvp_incremental_adjoint_function",
        reject_full_hvp_backend,
    )
    device = torch.device("cuda")
    case = _build_tm_case(device)
    source_amplitude = torch.cat(
        [case["source_amplitude"], 0.7 * case["source_amplitude"]], dim=0
    )
    source_location = case["source_location"].repeat(2, 1, 1)
    receiver_location = case["receiver_location"].repeat(2, 1, 1)
    observed_data = torch.zeros(
        source_amplitude.shape[-1],
        2,
        receiver_location.shape[1],
        device=device,
        dtype=case["epsilon"].dtype,
    )
    vepsilon = torch.stack(
        [case["depsilon"], 2.0 * case["depsilon"], -case["depsilon"]]
    )
    sigma_direction = torch.full_like(case["sigma"], 1e-4)
    vsigma = torch.stack(
        [sigma_direction, -2.0 * sigma_direction, torch.zeros_like(sigma_direction)]
    )

    def coupled_misfit(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        residual = predicted - observed
        shot_sum = predicted.sum(dim=1)
        return (
            0.5 * residual.square().sum()
            + 0.01 * predicted.sin().sum()
            + 0.02 * shot_sum.square().sum()
        )

    kwargs = {
        "grid_spacing": case["dx"],
        "dt": case["dt"],
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "observed_data": observed_data,
        "misfit": coupled_misfit,
        "stencil": 2,
        "pml_width": 1,
        "storage_compression": storage_compression,
        "hessian_mode": "gauss_newton",
    }

    with tide.linearize_maxwelltm(
        case["epsilon"], case["sigma"], case["mu"], **kwargs
    ) as context:
        actual = context.hvp_batch(
            vepsilon=vepsilon,
            vsigma=vsigma,
            block_size=block_size,
        )
        assert context.can_batch_directions
        assert context.background_builds == 1
        assert context.reused_directions == 2
        assert context.batched_blocks == (2 + block_size - 1) // block_size
        assert context.scattered_history_bytes == 0

    expected_parts = [
        tide.maxwelltm_hvp(
            case["epsilon"],
            case["sigma"],
            case["mu"],
            vepsilon=epsilon_direction,
            vsigma=sigma_direction,
            **kwargs,
        )
        for epsilon_direction, sigma_direction in zip(vepsilon, vsigma)
    ]
    expected = tuple(torch.stack(parts) for parts in zip(*expected_parts))
    for actual_part, expected_part in zip(actual, expected):
        relative_l2 = (actual_part - expected_part).norm() / expected_part.norm()
        assert relative_l2 < relative_tolerance
        assert torch.isfinite(actual_part).all()


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native CUDA backend not available",
)
def test_tm2d_gauss_newton_direction_block_is_symmetric_psd():
    device = torch.device("cuda")
    case = _build_tm_case(device)
    source_amplitude = torch.cat(
        [case["source_amplitude"], 0.7 * case["source_amplitude"]], dim=0
    )
    source_location = case["source_location"].repeat(2, 1, 1)
    # The source and receiver must be distinct: with a coincident source and
    # receiver on a tiny 6x6 grid the receiver gradient is dominated by the
    # injected wavefield, and even the Python reference implementation shows
    # ~1e-4 relative symmetry breaking in float32 (far above the 2e-5
    # tolerance), so the check would be meaningless for any backend.
    receiver_location = case["receiver_location"].repeat(2, 1, 1)
    receiver_location[..., 0, 0] = 2
    receiver_location[..., 0, 1] = 2
    observed_data = torch.zeros(
        source_amplitude.shape[-1],
        2,
        receiver_location.shape[1],
        device=device,
        dtype=case["epsilon"].dtype,
    )
    generator = torch.Generator(device=device).manual_seed(4)
    vepsilon = (
        torch.randn(
            2,
            *case["epsilon"].shape,
            device=device,
            dtype=case["epsilon"].dtype,
            generator=generator,
        )
        * 1e-2
    )
    vsigma = (
        torch.randn(
            2,
            *case["sigma"].shape,
            device=device,
            dtype=case["sigma"].dtype,
            generator=generator,
        )
        * 1e-5
    )

    def squared_error(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        return 0.5 * (predicted - observed).square().sum()

    with tide.linearize_maxwelltm(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        observed_data=observed_data,
        misfit=squared_error,
        stencil=2,
        pml_width=1,
        storage_compression=False,
        hessian_mode="gauss_newton",
    ) as context:
        hvp_epsilon, hvp_sigma = context.hvp_batch(
            vepsilon=vepsilon,
            vsigma=vsigma,
            block_size=2,
        )

    lhs = (vepsilon[0] * hvp_epsilon[1]).sum() + (vsigma[0] * hvp_sigma[1]).sum()
    rhs = (hvp_epsilon[0] * vepsilon[1]).sum() + (hvp_sigma[0] * vsigma[1]).sum()
    relative_symmetry_error = (lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs())
    assert relative_symmetry_error < 2e-5

    quadratic_forms = (vepsilon * hvp_epsilon).flatten(1).sum(1) + (
        vsigma * hvp_sigma
    ).flatten(1).sum(1)
    assert torch.all(quadratic_forms >= -1e-5)


def test_tm2d_linearization_context_rejects_mutated_model():
    device = torch.device("cpu")
    case = _build_tm_case(device)
    context = tide.linearize_maxwelltm(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=case["dx"],
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        observed_data=_tm_observed_data(case, device),
        stencil=2,
        pml_width=1,
    )
    case["epsilon"].add_(0.1)
    with pytest.raises(RuntimeError, match="epsilon changed"):
        context.hvp(vepsilon=case["depsilon"])


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


@pytest.mark.parametrize("hessian_mode", ["gauss_newton", "full"])
@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_maxwell3d_hvp_native_cuda_matches_python_reference(hessian_mode):
    case = _build_3d_case(torch.device("cuda"))
    outputs = {}
    for python_backend in (True, False):
        outputs[python_backend] = tide.maxwell3d_hvp(
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
            hessian_mode=hessian_mode,
            python_backend=python_backend,
        )

    for reference, native in zip(outputs[True], outputs[False]):
        relative_l2 = (native - reference).double().norm() / reference.double().norm()
        assert relative_l2 < 5e-6


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
