import os

import pytest
import torch

import tide


def test_tm2d_fused_forward_matches_baseline(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fused TM2D forward test.")

    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(0)

    ny, nx = 64, 80
    nt = 8
    dt = 5e-12
    dy = dx = 0.01

    epsilon = torch.full((ny, nx), 9.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    ey0 = torch.randn((1, ny, nx), device=device, dtype=dtype) * 1e-3
    hx0 = torch.randn((1, ny, nx), device=device, dtype=dtype) * 1e-3
    hz0 = torch.randn((1, ny, nx), device=device, dtype=dtype) * 1e-3

    def run_with_env(steps: int) -> tuple[torch.Tensor, ...]:
        monkeypatch.setenv("TIDE_TM_FUSED_STEPS", "0")
        monkeypatch.setenv("TIDE_TM_EBISU_STEPS", str(steps))
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_X", "16")
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_Y", "16")
        monkeypatch.setenv("TIDE_TM_EBISU_ILP", "4")
        with torch.no_grad():
            return tide.maxwelltm(
                epsilon,
                sigma,
                mu,
                grid_spacing=dy,
                dt=dt,
                source_amplitude=None,
                source_location=None,
                receiver_location=None,
                stencil=2,
                pml_width=0,
                Ey_0=ey0.clone(),
                Hx_0=hx0.clone(),
                Hz_0=hz0.clone(),
                nt=nt,
                save_snapshots=False,
            )

    baseline = run_with_env(0)
    steps = 3
    fused = run_with_env(steps)

    for ref, got in zip(baseline[:-1], fused[:-1], strict=False):
        if ref.ndim == 3:
            ref = ref[:, steps:-steps, steps:-steps]
            got = got[:, steps:-steps, steps:-steps]
        assert torch.allclose(got, ref, atol=2e-6, rtol=2e-5)

    assert baseline[-1].numel() == 0
    assert fused[-1].numel() == 0


def test_tm2d_ebisu_forward_matches_baseline_with_pml_and_io(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TM2D EBISU forward test.")

    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(0)

    ny, nx = 64, 96
    nt = 12
    dt = 5e-12
    dy = dx = 0.01

    epsilon = torch.full((ny, nx), 9.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[ny // 2, nx // 3]]], device=device)
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 4]], [[ny // 2, nx // 2]], [[ny // 2, 3 * nx // 4]]],
        device=device,
    ).permute(1, 0, 2).contiguous()
    source_amplitude = tide.ricker(25e6, nt, dt, peak_time=1.0 / 25e6).to(
        device=device, dtype=dtype
    )
    source_amplitude = source_amplitude.view(1, 1, nt).contiguous()

    def run_with_env(steps: int) -> tuple[torch.Tensor, ...]:
        monkeypatch.setenv("TIDE_TM_FUSED_STEPS", "0")
        monkeypatch.setenv("TIDE_TM_EBISU_STEPS", str(steps))
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_X", "64")
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_Y", "16")
        monkeypatch.setenv("TIDE_TM_EBISU_ILP", "1")
        with torch.no_grad():
            return tide.maxwelltm(
                epsilon,
                sigma,
                mu,
                grid_spacing=dy,
                dt=dt,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                stencil=2,
                pml_width=8,
                nt=nt,
                save_snapshots=False,
                compute_precision="default",
            )

    baseline = run_with_env(0)
    ebisu = run_with_env(2)

    for ref, got in zip(baseline, ebisu, strict=False):
        assert torch.allclose(got, ref, atol=3e-7, rtol=1e-6)


def test_tm2d_ebisu_forward_matches_baseline_with_top_strip_pml_io(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TM2D EBISU forward test.")

    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(0)

    ny, nx = 80, 128
    nt = 8
    dt = 4e-11
    dy = dx = 0.02
    pml_width = 10
    source_depth = 2

    epsilon = torch.full((ny, nx), 6.0, device=device, dtype=dtype)
    epsilon[source_depth + 8 :, :] = 9.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[source_depth, nx // 2]]], device=device, dtype=torch.long
    )
    receiver_location = torch.tensor(
        [[[source_depth, nx // 2 - 1]], [[source_depth, nx // 2 + 1]]],
        device=device,
        dtype=torch.long,
    ).permute(1, 0, 2).contiguous()
    source_amplitude = tide.ricker(600e6, nt, dt, peak_time=1.0 / 600e6).to(
        device=device, dtype=dtype
    )
    source_amplitude = source_amplitude.view(1, 1, nt).contiguous()

    def run_with_env(steps: int) -> tuple[torch.Tensor, ...]:
        monkeypatch.setenv("TIDE_TM_FUSED_STEPS", "0")
        monkeypatch.setenv("TIDE_TM_EBISU_STEPS", str(steps))
        monkeypatch.setenv("TIDE_TM_EBISU_FACE_PML", "1")
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_X", "40")
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_Y", "16")
        monkeypatch.setenv("TIDE_TM_EBISU_ILP", "1")
        with torch.no_grad():
            return tide.maxwelltm(
                epsilon,
                sigma,
                mu,
                grid_spacing=dy,
                dt=dt,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                stencil=2,
                pml_width=pml_width,
                nt=nt,
                save_snapshots=False,
                compute_precision="default",
            )

    baseline = run_with_env(0)
    ebisu = run_with_env(3)

    for ref, got in zip(baseline, ebisu, strict=False):
        assert torch.allclose(got, ref, atol=4e-7, rtol=1e-6)


def test_tm2d_ebisu_forward_matches_baseline_with_single_io_fast_path(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TM2D EBISU forward test.")

    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(0)

    ny, nx = 72, 112
    nt = 10
    dt = 5e-12
    dy = dx = 0.01

    epsilon = torch.full((ny, nx), 7.0, device=device, dtype=dtype)
    epsilon[ny // 2 :, :] = 9.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[ny // 2, nx // 2]]], device=device, dtype=torch.long
    )
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2 + 1]]], device=device, dtype=torch.long
    )
    source_amplitude = tide.ricker(40e6, nt, dt, peak_time=1.0 / 40e6).to(
        device=device, dtype=dtype
    )
    source_amplitude = source_amplitude.view(1, 1, nt).contiguous()

    def run_with_env(steps: int) -> tuple[torch.Tensor, ...]:
        monkeypatch.setenv("TIDE_TM_FUSED_STEPS", "0")
        monkeypatch.setenv("TIDE_TM_EBISU_STEPS", str(steps))
        monkeypatch.setenv("TIDE_TM_EBISU_FACE_PML", "1")
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_X", "40")
        monkeypatch.setenv("TIDE_TM_EBISU_TILE_Y", "16")
        monkeypatch.setenv("TIDE_TM_EBISU_ILP", "1")
        with torch.no_grad():
            return tide.maxwelltm(
                epsilon,
                sigma,
                mu,
                grid_spacing=dy,
                dt=dt,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                stencil=2,
                pml_width=8,
                nt=nt,
                save_snapshots=False,
                compute_precision="default",
            )

    baseline = run_with_env(0)
    ebisu = run_with_env(3)

    for ref, got in zip(baseline, ebisu, strict=False):
        assert torch.allclose(got, ref, atol=4e-7, rtol=1e-6)
