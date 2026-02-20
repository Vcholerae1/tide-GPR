"""Tests for fp16 mixed-precision and internal nondimensional scaling."""

import pytest
import torch

import tide


def _base_case(device: torch.device) -> dict[str, torch.Tensor | float | int]:
    ny, nx = 24, 24
    nt = 40
    dt = 4e-11
    dtype = torch.float32

    epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.ones(ny, nx, device=device, dtype=dtype) * 1e-3
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[ny // 2, nx // 3]]], device=device, dtype=torch.long)
    receiver_location = torch.tensor([[[ny // 2, nx // 2]]], device=device, dtype=torch.long)
    wavelet = tide.ricker(
        200e6, nt, dt, peak_time=1.0 / 200e6, dtype=dtype, device=device
    ).view(1, 1, nt)

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "source_amplitude": wavelet,
        "dt": dt,
        "nt": nt,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_forward_relative_l2():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    case = _base_case(torch.device("cuda"))

    ref = tide.maxwelltm(
        case["epsilon"],  # type: ignore[arg-type]
        case["sigma"],  # type: ignore[arg-type]
        case["mu"],  # type: ignore[arg-type]
        grid_spacing=0.02,
        dt=case["dt"],  # type: ignore[arg-type]
        source_amplitude=case["source_amplitude"],  # type: ignore[arg-type]
        source_location=case["source_location"],  # type: ignore[arg-type]
        receiver_location=case["receiver_location"],  # type: ignore[arg-type]
        pml_width=4,
        stencil=2,
        compute_dtype="fp32",
    )[-1]
    out = tide.maxwelltm(
        case["epsilon"],  # type: ignore[arg-type]
        case["sigma"],  # type: ignore[arg-type]
        case["mu"],  # type: ignore[arg-type]
        grid_spacing=0.02,
        dt=case["dt"],  # type: ignore[arg-type]
        source_amplitude=case["source_amplitude"],  # type: ignore[arg-type]
        source_location=case["source_location"],  # type: ignore[arg-type]
        receiver_location=case["receiver_location"],  # type: ignore[arg-type]
        pml_width=4,
        stencil=2,
        compute_dtype="fp16",
        mp_mode="throughput",
    )[-1]

    rel_l2 = torch.linalg.norm(out - ref) / (torch.linalg.norm(ref) + 1e-12)
    assert float(rel_l2.item()) <= 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_backward_gradient_cosine():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    case = _base_case(torch.device("cuda"))

    eps_ref = case["epsilon"].clone().detach().requires_grad_(True)  # type: ignore[assignment]
    sig_ref = case["sigma"].clone().detach().requires_grad_(True)  # type: ignore[assignment]
    rec_ref = tide.maxwelltm(
        eps_ref,
        sig_ref,
        case["mu"],  # type: ignore[arg-type]
        grid_spacing=0.02,
        dt=case["dt"],  # type: ignore[arg-type]
        source_amplitude=case["source_amplitude"],  # type: ignore[arg-type]
        source_location=case["source_location"],  # type: ignore[arg-type]
        receiver_location=case["receiver_location"],  # type: ignore[arg-type]
        pml_width=4,
        stencil=2,
        compute_dtype="fp32",
    )[-1]
    rec_ref.square().sum().backward()
    assert eps_ref.grad is not None and sig_ref.grad is not None
    g_eps_ref = eps_ref.grad.detach().reshape(-1)
    g_sig_ref = sig_ref.grad.detach().reshape(-1)

    eps_mp = case["epsilon"].clone().detach().requires_grad_(True)  # type: ignore[assignment]
    sig_mp = case["sigma"].clone().detach().requires_grad_(True)  # type: ignore[assignment]
    rec_mp = tide.maxwelltm(
        eps_mp,
        sig_mp,
        case["mu"],  # type: ignore[arg-type]
        grid_spacing=0.02,
        dt=case["dt"],  # type: ignore[arg-type]
        source_amplitude=case["source_amplitude"],  # type: ignore[arg-type]
        source_location=case["source_location"],  # type: ignore[arg-type]
        receiver_location=case["receiver_location"],  # type: ignore[arg-type]
        pml_width=4,
        stencil=2,
        compute_dtype="fp16",
        mp_mode="throughput",
    )[-1]
    rec_mp.square().sum().backward()
    assert eps_mp.grad is not None and sig_mp.grad is not None
    g_eps_mp = eps_mp.grad.detach().reshape(-1)
    g_sig_mp = sig_mp.grad.detach().reshape(-1)

    eps_cos = torch.nn.functional.cosine_similarity(g_eps_ref, g_eps_mp, dim=0)
    sig_cos = torch.nn.functional.cosine_similarity(g_sig_ref, g_sig_mp, dim=0)
    assert float(eps_cos.item()) >= 0.999
    assert float(sig_cos.item()) >= 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_state_continuation_consistency():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    case = _base_case(torch.device("cuda"))
    nt = int(case["nt"])  # type: ignore[arg-type]
    nt1 = nt // 2
    src_full = case["source_amplitude"]  # type: ignore[assignment]
    src_1 = src_full[..., :nt1]
    src_2 = src_full[..., nt1:]

    part1 = tide.maxwelltm(
        case["epsilon"],  # type: ignore[arg-type]
        case["sigma"],  # type: ignore[arg-type]
        case["mu"],  # type: ignore[arg-type]
        grid_spacing=0.02,
        dt=case["dt"],  # type: ignore[arg-type]
        source_amplitude=src_1,
        source_location=case["source_location"],  # type: ignore[arg-type]
        receiver_location=case["receiver_location"],  # type: ignore[arg-type]
        pml_width=4,
        stencil=2,
        compute_dtype="fp16",
    )
    part2 = tide.maxwelltm(
        case["epsilon"],  # type: ignore[arg-type]
        case["sigma"],  # type: ignore[arg-type]
        case["mu"],  # type: ignore[arg-type]
        grid_spacing=0.02,
        dt=case["dt"],  # type: ignore[arg-type]
        source_amplitude=src_2,
        source_location=case["source_location"],  # type: ignore[arg-type]
        receiver_location=case["receiver_location"],  # type: ignore[arg-type]
        pml_width=4,
        stencil=2,
        Ey_0=part1[0],
        Hx_0=part1[1],
        Hz_0=part1[2],
        m_Ey_x=part1[3],
        m_Ey_z=part1[4],
        m_Hx_z=part1[5],
        m_Hz_x=part1[6],
        compute_dtype="fp16",
    )
    single = tide.maxwelltm(
        case["epsilon"],  # type: ignore[arg-type]
        case["sigma"],  # type: ignore[arg-type]
        case["mu"],  # type: ignore[arg-type]
        grid_spacing=0.02,
        dt=case["dt"],  # type: ignore[arg-type]
        source_amplitude=src_full,
        source_location=case["source_location"],  # type: ignore[arg-type]
        receiver_location=case["receiver_location"],  # type: ignore[arg-type]
        pml_width=4,
        stencil=2,
        compute_dtype="fp16",
    )

    for lhs, rhs in zip(part2[:-1], single[:-1]):
        rel = torch.linalg.norm(lhs - rhs) / (torch.linalg.norm(rhs) + 1e-12)
        assert float(rel.item()) <= 5e-3


def test_fp16_requires_cuda():
    device = torch.device("cpu")
    case = _base_case(device)
    with pytest.raises(TypeError, match="only on CUDA"):
        tide.maxwelltm(
            case["epsilon"],  # type: ignore[arg-type]
            case["sigma"],  # type: ignore[arg-type]
            case["mu"],  # type: ignore[arg-type]
            grid_spacing=0.02,
            dt=case["dt"],  # type: ignore[arg-type]
            source_amplitude=case["source_amplitude"],  # type: ignore[arg-type]
            source_location=case["source_location"],  # type: ignore[arg-type]
            receiver_location=case["receiver_location"],  # type: ignore[arg-type]
            pml_width=4,
            stencil=2,
            compute_dtype="fp16",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_rejects_non_raw_storage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    case = _base_case(torch.device("cuda"))
    with pytest.raises(ValueError, match="raw snapshot storage"):
        tide.maxwelltm(
            case["epsilon"],  # type: ignore[arg-type]
            case["sigma"],  # type: ignore[arg-type]
            case["mu"],  # type: ignore[arg-type]
            grid_spacing=0.02,
            dt=case["dt"],  # type: ignore[arg-type]
            source_amplitude=case["source_amplitude"],  # type: ignore[arg-type]
            source_location=case["source_location"],  # type: ignore[arg-type]
            receiver_location=case["receiver_location"],  # type: ignore[arg-type]
            pml_width=4,
            stencil=2,
            compute_dtype="fp16",
            storage_compression="bf16",
        )
