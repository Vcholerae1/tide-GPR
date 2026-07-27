import pytest
import torch

import tide


def _case(*, nt: int = 160) -> dict:
    device = torch.device("cuda")
    ny, nx = 72, 80
    epsilon = torch.full((ny, nx), 4.0, device=device)
    epsilon[ny // 2 :, :] = 9.0
    sigma = torch.full_like(epsilon, 1e-3)
    mu = torch.ones_like(epsilon)
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "grid_spacing": 0.02,
        "dt": 4e-11,
        "source_amplitude": tide.ricker(
            200e6, nt, 4e-11, dtype=torch.float32, device=device
        ).reshape(1, 1, -1),
        "source_location": torch.tensor([[[18, 24]]], device=device),
        "receiver_location": torch.tensor([[[18, 42]]], device=device),
        "pml_width": 10,
        "stencil": 4,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
def test_tm2d_fp16_io_forward_matches_native(stencil: int) -> None:
    case = _case()
    case["stencil"] = stencil
    reference = tide.maxwelltm(**case)
    reduced = tide.maxwelltm(**case, compute_mode="fp16_io")

    for actual in reduced:
        assert actual.dtype == torch.float32
        assert torch.isfinite(actual).all()

    ref_record = reference[-1].reshape(-1)
    reduced_record = reduced[-1].reshape(-1)
    relative_l2 = torch.linalg.vector_norm(
        reduced_record - ref_record
    ) / torch.linalg.vector_norm(ref_record).clamp_min(1e-30)
    correlation = torch.nn.functional.cosine_similarity(
        ref_record, reduced_record, dim=0
    )
    assert float(relative_l2) < 5e-3
    assert float(correlation) > 0.99999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tm2d_fp16_io_state_continuation() -> None:
    case = _case(nt=180)
    # Public state tensors include the PML padding, while the runtime shape
    # annotation ties state dimensions to the unpadded model dimensions.
    # Use a no-PML case here to exercise continuation independently of that
    # existing API constraint.
    case["pml_width"] = 0
    source = case.pop("source_amplitude")
    full = tide.maxwelltm(**case, source_amplitude=source, compute_mode="fp16_io")
    split = source.shape[-1] // 2
    first = tide.maxwelltm(
        **case, source_amplitude=source[..., :split], compute_mode="fp16_io"
    )
    second = tide.maxwelltm(
        **case,
        source_amplitude=source[..., split:],
        Ey_0=first[0],
        Hx_0=first[1],
        Hz_0=first[2],
        m_Ey_x=first[3],
        m_Ey_z=first[4],
        m_Hx_z=first[5],
        m_Hz_x=first[6],
        compute_mode="fp16_io",
    )
    for continued, expected in zip(second[:-1], full[:-1]):
        relative_l2 = torch.linalg.vector_norm(
            continued - expected
        ) / torch.linalg.vector_norm(expected).clamp_min(1e-30)
        assert float(relative_l2) < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tm2d_fp16_half2_matches_scalar_io(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case(nt=120)
    monkeypatch.setenv("TIDE_TM_FP16_HALF2", "0")
    scalar = tide.maxwelltm(**case, compute_mode="fp16_io")
    monkeypatch.setenv("TIDE_TM_FP16_HALF2", "1")
    packed = tide.maxwelltm(**case, compute_mode="fp16_io")
    for actual, expected in zip(packed, scalar):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tm2d_aggressive_half2_arithmetic_is_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(nt=120)
    reference = tide.maxwelltm(**case)[-1].reshape(-1)
    monkeypatch.setenv("TIDE_TM_FP16_HALF2_ARITH", "1")
    aggressive = tide.maxwelltm(**case, compute_mode="fp16_io")[-1].reshape(-1)
    assert torch.isfinite(aggressive).all()
    relative_l2 = torch.linalg.vector_norm(
        aggressive - reference
    ) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    correlation = torch.nn.functional.cosine_similarity(
        aggressive, reference, dim=0
    )
    assert float(relative_l2) < 1e-2
    assert float(correlation) > 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tm2d_full_fp16_adjoint_scales_small_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gradients = {}
    upstream = None
    for mode in ("native", "fp16_io"):
        if mode == "fp16_io":
            monkeypatch.setenv("TIDE_TM_FP16_HALF2_ARITH", "1")
            monkeypatch.setenv("TIDE_TM_FP16_ADJOINT", "1")
        case = _case(nt=180)
        case["epsilon"] = case["epsilon"].clone().requires_grad_(True)
        case["storage_mode"] = "device"
        case["storage_compression"] = "bf16"
        case["model_gradient_sampling_interval"] = 2
        record = tide.maxwelltm(**case, compute_mode=mode)[-1]
        if upstream is None:
            upstream = (
                torch.sin(torch.arange(record.numel(), device=record.device) * 0.173)
                .reshape_as(record)
                .mul(1e-8)
            )
        record.backward(upstream)
        gradients[mode] = case["epsilon"].grad.detach()

    reference = gradients["native"].reshape(-1)
    actual = gradients["fp16_io"].reshape(-1)
    relative_l2 = torch.linalg.vector_norm(
        actual - reference
    ) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    correlation = torch.nn.functional.cosine_similarity(actual, reference, dim=0)
    assert torch.isfinite(actual).all()
    assert float(relative_l2) < 5e-2
    assert float(correlation) > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tm2d_fp16_io_material_gradients_match_native() -> None:
    gradients = {}
    upstream = None
    for mode in ("native", "fp16_io"):
        case = _case(nt=180)
        case["epsilon"] = case["epsilon"].clone().requires_grad_(True)
        case["sigma"] = case["sigma"].clone().requires_grad_(True)
        case["storage_mode"] = "device"
        case["storage_compression"] = "bf16"
        case["model_gradient_sampling_interval"] = 2
        record = tide.maxwelltm(**case, compute_mode=mode)[-1]
        if upstream is None:
            # Compare the two Jacobian-transpose operators with the same
            # adjoint source.  Letting each approximate forward solve create
            # its own residual would mix forward quantisation error into this
            # gradient-specific test.
            upstream = torch.sin(
                torch.arange(record.numel(), device=record.device) * 0.173
            ).reshape_as(record)
        record.backward(upstream)
        gradients[mode] = (
            record.detach(),
            case["epsilon"].grad.detach(),
            case["sigma"].grad.detach(),
        )

    tolerances = ((2e-3, 0.99999), (3e-2, 0.999), (5e-3, 0.99999))
    for actual, expected, (relative_limit, correlation_limit) in zip(
        gradients["fp16_io"], gradients["native"], tolerances
    ):
        relative_l2 = torch.linalg.vector_norm(
            actual - expected
        ) / torch.linalg.vector_norm(expected).clamp_min(1e-30)
        correlation = torch.nn.functional.cosine_similarity(
            actual.reshape(-1), expected.reshape(-1), dim=0
        )
        assert float(relative_l2) < relative_limit
        assert float(correlation) > correlation_limit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tm2d_fp16_io_alternating_modes_refresh_cuda_constants() -> None:
    case_a = _case(nt=80)
    case_b = _case(nt=80)
    case_b["grid_spacing"] = 0.031

    native_a_first = tide.maxwelltm(**case_a)[-1]
    tide.maxwelltm(**case_b, compute_mode="fp16_io")
    native_a_second = tide.maxwelltm(**case_a)[-1]
    torch.testing.assert_close(native_a_second, native_a_first, rtol=0, atol=0)

    fp16_a_first = tide.maxwelltm(**case_a, compute_mode="fp16_io")[-1]
    tide.maxwelltm(**case_b)
    fp16_a_second = tide.maxwelltm(**case_a, compute_mode="fp16_io")[-1]
    torch.testing.assert_close(fp16_a_second, fp16_a_first, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tm2d_fp16_io_gradients_require_device_storage() -> None:
    case = _case()
    case["epsilon"] = case["epsilon"].requires_grad_(True)
    with pytest.raises(NotImplementedError, match="device"):
        tide.maxwelltm(**case, compute_mode="fp16_io", storage_mode="cpu")
