#!/usr/bin/env python3
import argparse
import time

import torch

import tide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Maxwell TM forward propagation.")
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1e-11)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float32

    ny = args.ny
    nx = args.nx
    n_shots = args.shots
    n_sources = 1
    n_receivers = 1

    epsilon = torch.ones((ny, nx), device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    wavelet = tide.ricker(
        freq=1e9,
        length=args.nt,
        dt=args.dt,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.reshape(1, 1, -1).repeat(n_shots, n_sources, 1)

    src_y = ny // 2
    src_x = nx // 2
    rec_y = ny // 2
    rec_x = min(nx - 1, nx // 2 + nx // 4)

    source_location = torch.tensor([[[src_y, src_x]]], device=device).repeat(n_shots, 1, 1)
    receiver_location = torch.tensor([[[rec_y, rec_x]]], device=device).repeat(n_shots, 1, 1)

    pml_width = args.pml_width
    grid_spacing = [args.dx, args.dx]

    print("Benchmark settings:")
    print(f"  device={device} dtype={dtype}")
    print(f"  ny={ny} nx={nx} nt={args.nt} shots={n_shots}")
    print(f"  pml_width={pml_width} dx={args.dx} dt={args.dt}")
    print("  pml_split=always_on")

    def run_once() -> None:
        tide.maxwelltm(
            epsilon=epsilon,
            sigma=sigma,
            mu=mu,
            grid_spacing=grid_spacing,
            dt=args.dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=pml_width,
            gradient_mode="snapshot",
        )

    for _ in range(args.warmup):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    steps = args.iters * args.nt
    time_per_step = elapsed / steps
    cells = ny * nx * n_shots
    cell_updates_per_s = (cells * args.nt * args.iters) / elapsed

    print("Results:")
    print(f"  total_time_s={elapsed:.4f}")
    print(f"  time_per_step_ms={time_per_step * 1e3:.4f}")
    print(f"  cell_updates_per_s={cell_updates_per_s:.2e}")


if __name__ == "__main__":
    main()
