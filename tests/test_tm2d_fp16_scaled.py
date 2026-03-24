import tempfile

import pytest
import torch

import tide


def _base_case(
    device: torch.device,
    *,
    n_shots: int = 1,
    dtype: torch.dtype = torch.float32,
    nt: int = 32,
) -> dict[str, torch.Tensor | float | int]:
    ny, nx = 24, 28
    dt = 4e-11

    epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
    epsilon[ny // 3 : 2 * ny // 3, nx // 3 : 2 * nx // 3] = 18.0
    sigma = torch.full_like(epsilon, 5e-4)
    sigma[ny // 4 : ny // 2, nx // 4 : nx // 2] = 2e-2
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[ny // 2, nx // 4]]] * n_shots,
        device=device,
        dtype=torch.long,
    )
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2]], [[ny // 2, 3 * nx // 4]]][:n_shots],
        device=device,
        dtype=torch.long,
    )
    if receiver_location.shape[0] != n_shots:
        receiver_location = receiver_location.expand(n_shots, -1, -1).contiguous()
    wavelet = tide.ricker(
        250e6,
        nt,
        dt,
        peak_time=1.0 / 250e6,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    source_amplitude = wavelet.repeat(n_shots, 1, 1)

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "source_amplitude": source_amplitude,
        "dt": dt,
        "nt": nt,
    }


def _rel_l2(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    denom = torch.linalg.norm(rhs.float()) + 1e-12
    return float((torch.linalg.norm((lhs - rhs).float()) / denom).item())


def _corr(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    a = lhs.float().reshape(-1)
    b = rhs.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def _run_tm2d(
    case: dict[str, torch.Tensor | float | int],
    *,
    stencil: int,
    compute_precision: str = "default",
    storage_mode: str = "device",
    storage_compression: bool | str = False,
    forward_callback=None,
    Ey_0=None,
    Hx_0=None,
    Hz_0=None,
    m_Ey_x=None,
    m_Ey_z=None,
    m_Hx_z=None,
    m_Hz_x=None,
    source_amplitude=None,
):
    return tide.maxwelltm(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=0.02,
        dt=case["dt"],
        source_amplitude=source_amplitude
        if source_amplitude is not None
        else case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        pml_width=4,
        stencil=stencil,
        compute_precision=compute_precision,
        storage_mode=storage_mode,
        storage_compression=storage_compression,
        forward_callback=forward_callback,
        Ey_0=Ey_0,
        Hx_0=Hx_0,
        Hz_0=Hz_0,
        m_Ey_x=m_Ey_x,
        m_Ey_z=m_Ey_z,
        m_Hx_z=m_Hx_z,
        m_Hz_x=m_Hz_x,
    )


def test_fp16_scaled_requires_cuda():
    case = _base_case(torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="requires a CUDA device"):
        _run_tm2d(case, stencil=2, compute_precision="fp16_scaled")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_rejects_python_backend():
    case = _base_case(torch.device("cuda"))
    with pytest.raises(NotImplementedError, match="python_backend only supports"):
        tide.maxwelltm(
            case["epsilon"],
            case["sigma"],
            case["mu"],
            grid_spacing=0.02,
            dt=case["dt"],
            source_amplitude=case["source_amplitude"],
            source_location=case["source_location"],
            receiver_location=case["receiver_location"],
            pml_width=4,
            stencil=2,
            compute_precision="fp16_scaled",
            python_backend=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_rejects_float64_inputs():
    case = _base_case(torch.device("cuda"), dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="requires float32 public tensors"):
        _run_tm2d(case, stencil=2, compute_precision="fp16_scaled")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_rejects_bf16_snapshot_compression():
    case = _base_case(torch.device("cuda"))
    with pytest.raises(ValueError, match="does not support storage_compression"):
        _run_tm2d(
            case,
            stencil=2,
            compute_precision="fp16_scaled",
            storage_compression="bf16",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("n_shots", [1, 2])
def test_fp16_scaled_forward_matches_float32(stencil: int, n_shots: int):
    case = _base_case(torch.device("cuda"), n_shots=n_shots)
    ref = _run_tm2d(case, stencil=stencil, compute_precision="default")
    out = _run_tm2d(case, stencil=stencil, compute_precision="fp16_scaled")

    assert _corr(out[-1], ref[-1]) > 0.999
    assert _rel_l2(out[-1], ref[-1]) <= 1e-2

    state_thresholds = {
        "Ey": 1e-2,
        "Hx": 1e-2,
        "Hz": 1e-2,
        "m_Ey_x": 1e-2,
        # CPML memory variables carry the smallest continuation magnitudes and
        # are the first tensors to show fp16 quantization noise even when the
        # physical fields/receivers remain within the target tolerance.
        "m_Ey_z": 2e-2,
        "m_Hx_z": 2e-2,
        "m_Hz_x": 1e-2,
    }
    for name, lhs, rhs in zip(state_thresholds, out[:-1], ref[:-1]):
        assert _rel_l2(lhs, rhs) <= state_thresholds[name]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
def test_fp16_scaled_backward_matches_float32(stencil: int):
    case = _base_case(torch.device("cuda"), n_shots=2)

    eps_ref = case["epsilon"].clone().detach().requires_grad_(True)
    sig_ref = case["sigma"].clone().detach().requires_grad_(True)
    ref = tide.maxwelltm(
        eps_ref,
        sig_ref,
        case["mu"],
        grid_spacing=0.02,
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        pml_width=4,
        stencil=stencil,
        compute_precision="default",
    )[-1]
    ref.float().square().sum().backward()
    assert eps_ref.grad is not None and sig_ref.grad is not None

    eps_mp = case["epsilon"].clone().detach().requires_grad_(True)
    sig_mp = case["sigma"].clone().detach().requires_grad_(True)
    out = tide.maxwelltm(
        eps_mp,
        sig_mp,
        case["mu"],
        grid_spacing=0.02,
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        pml_width=4,
        stencil=stencil,
        compute_precision="fp16_scaled",
    )[-1]
    out.float().square().sum().backward()
    assert eps_mp.grad is not None and sig_mp.grad is not None

    eps_cos = _corr(eps_mp.grad, eps_ref.grad)
    sig_cos = _corr(sig_mp.grad, sig_ref.grad)
    assert eps_cos >= 0.99
    assert sig_cos >= 0.99
    assert _rel_l2(eps_mp.grad, eps_ref.grad) <= 3e-2
    assert _rel_l2(sig_mp.grad, sig_ref.grad) <= 3e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_finite_difference_sanity():
    case = _base_case(torch.device("cuda"), n_shots=1, nt=24)
    eps = case["epsilon"].clone().detach().requires_grad_(True)
    out = tide.maxwelltm(
        eps,
        case["sigma"],
        case["mu"],
        grid_spacing=0.02,
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        pml_width=4,
        stencil=2,
        compute_precision="fp16_scaled",
    )[-1]
    loss = out.float().square().sum()
    loss.backward()
    assert eps.grad is not None

    # Smaller perturbations are quantized away by the fp16-scaled forward
    # response on this tiny sanity case; use a larger but still local step so
    # the finite-difference sign remains observable.
    h = 2e-1
    eps_pert = case["epsilon"].clone()
    idx = (eps_pert.shape[0] // 2, eps_pert.shape[1] // 2)
    eps_pert[idx] += h
    out_pert = tide.maxwelltm(
        eps_pert,
        case["sigma"],
        case["mu"],
        grid_spacing=0.02,
        dt=case["dt"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        pml_width=4,
        stencil=2,
        compute_precision="fp16_scaled",
    )[-1]
    fd = (out_pert.float().square().sum() - loss.detach()) / h
    grad_at_point = eps.grad[idx]
    assert torch.sign(grad_at_point) == torch.sign(fd)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_pml_region_matches_float32():
    case = _base_case(torch.device("cuda"), n_shots=1)
    ref = _run_tm2d(case, stencil=4, compute_precision="default")[0]
    out = _run_tm2d(case, stencil=4, compute_precision="fp16_scaled")[0]
    pml = 4
    mask = torch.zeros_like(ref, dtype=torch.bool)
    mask[:, :pml, :] = True
    mask[:, -pml:, :] = True
    mask[:, :, :pml] = True
    mask[:, :, -pml:] = True
    assert _rel_l2(out[mask], ref[mask]) <= 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_state_continuation_consistency():
    case = _base_case(torch.device("cuda"), n_shots=1, nt=36)
    nt = int(case["nt"])
    nt1 = nt // 2
    src_full = case["source_amplitude"]
    src_1 = src_full[..., :nt1]
    src_2 = src_full[..., nt1:]

    part1 = _run_tm2d(
        case,
        stencil=2,
        compute_precision="fp16_scaled",
        source_amplitude=src_1,
    )
    part2 = _run_tm2d(
        case,
        stencil=2,
        compute_precision="fp16_scaled",
        source_amplitude=src_2,
        Ey_0=part1[0],
        Hx_0=part1[1],
        Hz_0=part1[2],
        m_Ey_x=part1[3],
        m_Ey_z=part1[4],
        m_Hx_z=part1[5],
        m_Hz_x=part1[6],
    )
    single = _run_tm2d(case, stencil=2, compute_precision="fp16_scaled")

    for lhs, rhs in zip(part2[:-1], single[:-1]):
        assert _rel_l2(lhs, rhs) <= 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_callback_returns_physical_float32_wavefields():
    case = _base_case(torch.device("cuda"), n_shots=1, nt=1)
    seen: list[torch.Tensor] = []

    def cb(state):
        seen.append(state.get_wavefield("Ey", view="pml").detach().clone())

    out = _run_tm2d(
        case,
        stencil=2,
        compute_precision="fp16_scaled",
        forward_callback=cb,
    )

    assert seen
    assert seen[-1].dtype == torch.float32
    assert _rel_l2(seen[-1], out[0]) <= 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_storage_modes_match():
    case = _base_case(torch.device("cuda"), n_shots=2, nt=24)

    with tempfile.TemporaryDirectory() as storage_path:
        ref = _run_tm2d(
            case,
            stencil=2,
            compute_precision="fp16_scaled",
            storage_mode="device",
        )[-1]
        cpu = tide.maxwelltm(
            case["epsilon"],
            case["sigma"],
            case["mu"],
            grid_spacing=0.02,
            dt=case["dt"],
            source_amplitude=case["source_amplitude"],
            source_location=case["source_location"],
            receiver_location=case["receiver_location"],
            pml_width=4,
            stencil=2,
            compute_precision="fp16_scaled",
            storage_mode="cpu",
            storage_path=storage_path,
        )[-1]
        disk = tide.maxwelltm(
            case["epsilon"],
            case["sigma"],
            case["mu"],
            grid_spacing=0.02,
            dt=case["dt"],
            source_amplitude=case["source_amplitude"],
            source_location=case["source_location"],
            receiver_location=case["receiver_location"],
            pml_width=4,
            stencil=2,
            compute_precision="fp16_scaled",
            storage_mode="disk",
            storage_path=storage_path,
        )[-1]

    assert _rel_l2(cpu, ref) <= 1e-2
    assert _rel_l2(disk, ref) <= 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_scaled_stability_stress_case():
    case = _base_case(torch.device("cuda"), n_shots=1, nt=192)
    case["epsilon"][6:18, 8:20] = 80.0
    case["sigma"][:] = 1e-6
    out = _run_tm2d(case, stencil=2, compute_precision="fp16_scaled")
    ref = _run_tm2d(case, stencil=2, compute_precision="default")
    for tensor in out:
        assert torch.isfinite(tensor).all()
    assert _rel_l2(out[-1], ref[-1]) <= 2e-2
