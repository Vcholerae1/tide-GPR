import os
import time

import torch

import tide


def run_case(*, ebisu_steps: int, nt: int, ny: int, nx: int, device: torch.device) -> float:
    os.environ["TIDE_TM_FUSED_STEPS"] = "0"
    os.environ["TIDE_TM_EBISU_STEPS"] = str(ebisu_steps)
    os.environ["TIDE_TM_EBISU_TILE_X"] = "64"
    os.environ["TIDE_TM_EBISU_TILE_Y"] = "16"
    os.environ["TIDE_TM_EBISU_ILP"] = "1"

    dtype = torch.float32
    epsilon = torch.full((ny, nx), 9.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    ey0 = torch.randn((1, ny, nx), device=device, dtype=dtype) * 1e-3
    hx0 = torch.randn((1, ny, nx), device=device, dtype=dtype) * 1e-3
    hz0 = torch.randn((1, ny, nx), device=device, dtype=dtype) * 1e-3

    for _ in range(5):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.01,
            dt=5e-12,
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
    torch.cuda.synchronize(device)

    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.01,
            dt=5e-12,
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
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    nt = 256
    ny, nx = 512, 512

    baseline_ms = run_case(ebisu_steps=0, nt=nt, ny=ny, nx=nx, device=device)
    ebisu_ms = run_case(ebisu_steps=4, nt=nt, ny=ny, nx=nx, device=device)
    speedup = baseline_ms / ebisu_ms if ebisu_ms > 0 else float("inf")

    print(f"workload: nt={nt}, ny={ny}, nx={nx}, dtype=float32, stencil=2")
    print("constraints: no source, no receiver, no storage, no dispersion, pml_width=0")
    print("ebisu config: steps=4, tile=64x16, threads=256, ilp=1")
    print(f"baseline_ms: {baseline_ms:.3f}")
    print(f"ebisu_ms: {ebisu_ms:.3f}")
    print(f"speedup: {speedup:.3f}x")


if __name__ == "__main__":
    main()
