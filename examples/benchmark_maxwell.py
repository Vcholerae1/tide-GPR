"""Benchmark MaxwellTM forward and forward+backward runtime.

Runs two fixed scenarios and prints mean/p50/p90 in milliseconds.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch

import tide
from tide import backend_utils


@dataclass(frozen=True)
class Case:
    name: str
    ny: int
    nx: int
    nt: int
    n_shots: int


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        if not backend_utils.metal_available():
            raise RuntimeError("MPS requested but Metal backend not available.")
        return torch.device("mps")

    # auto
    if torch.backends.mps.is_available() and backend_utils.metal_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_case(
    case: Case,
    device: torch.device,
    backward: bool,
    warmup: int,
    timed: int,
    n_threads: int,
    compute_dtype: str,
    mp_mode: str,
) -> list[float]:
    dtype = torch.float32
    epsilon = torch.ones(case.ny, case.nx, device=device, dtype=dtype) * 4.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    if backward:
        epsilon.requires_grad_(True)
        sigma.requires_grad_(True)

    source_location = torch.zeros(case.n_shots, 1, 2, device=device, dtype=torch.long)
    source_location[:, 0, 0] = case.ny // 2
    source_location[:, 0, 1] = case.nx // 3
    receiver_location = torch.zeros(case.n_shots, 1, 2, device=device, dtype=torch.long)
    receiver_location[:, 0, 0] = case.ny // 2
    receiver_location[:, 0, 1] = case.nx // 2

    dt = 4e-11
    wavelet = tide.ricker(200e6, case.nt, dt, peak_time=1.0 / 200e6, dtype=dtype, device=device)
    source_amplitude = wavelet.view(1, 1, case.nt).repeat(case.n_shots, 1, 1)

    times: list[float] = []

    for i in range(warmup + timed):
        if backward:
            if epsilon.grad is not None:
                epsilon.grad.zero_()
            if sigma.grad is not None:
                sigma.grad.zero_()

        sync(device)
        t0 = time.perf_counter()
        rec = tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            stencil=2,
            pml_width=8,
            n_threads=n_threads if n_threads > 0 else None,
            compute_dtype=compute_dtype,
            mp_mode=mp_mode,
        )[-1]
        if backward:
            rec.square().sum().backward()
        sync(device)
        t1 = time.perf_counter()

        if i >= warmup:
            times.append((t1 - t0) * 1000.0)

    return times


def summarize(times_ms: list[float]) -> tuple[float, float, float]:
    ordered = sorted(times_ms)
    mean = statistics.fmean(times_ms)
    p50 = ordered[len(ordered) // 2]
    p90 = ordered[int(0.9 * (len(ordered) - 1))]
    return mean, p50, p90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MaxwellTM runtime.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Runtime device selection.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iteration count per case/mode.",
    )
    parser.add_argument(
        "--timed",
        type=int,
        default=10,
        help="Timed iteration count per case/mode.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="OpenMP thread count for C backend (0 keeps default).",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Compute precision mode exposed by tide.maxwelltm.",
    )
    parser.add_argument(
        "--mp-mode",
        type=str,
        default="throughput",
        choices=["throughput", "balanced", "robust"],
        help="Mixed precision policy for compute_dtype='fp16'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.threads > 0:
        # Keep runtime and subprocess behavior consistent when benchmarking from shell.
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    print(f"Device: {device}")
    print(
        f"Warmup: {args.warmup}  Timed: {args.timed}  Threads: {args.threads}  "
        f"compute_dtype: {args.compute_dtype}  mp_mode: {args.mp_mode}"
    )
    cases = [
        Case("small", ny=96, nx=96, nt=128, n_shots=2),
        Case("medium", ny=160, nx=160, nt=192, n_shots=4),
    ]

    print("\nCase,Mode,Mean(ms),P50(ms),P90(ms)")
    for case in cases:
        fwd = run_case(
            case,
            device,
            backward=False,
            warmup=args.warmup,
            timed=args.timed,
            n_threads=args.threads,
            compute_dtype=args.compute_dtype,
            mp_mode=args.mp_mode,
        )
        fwd_bwd = run_case(
            case,
            device,
            backward=True,
            warmup=args.warmup,
            timed=args.timed,
            n_threads=args.threads,
            compute_dtype=args.compute_dtype,
            mp_mode=args.mp_mode,
        )
        fwd_stats = summarize(fwd)
        bwd_stats = summarize(fwd_bwd)
        print(f"{case.name},forward,{fwd_stats[0]:.3f},{fwd_stats[1]:.3f},{fwd_stats[2]:.3f}")
        print(f"{case.name},forward+backward,{bwd_stats[0]:.3f},{bwd_stats[1]:.3f},{bwd_stats[2]:.3f}")


if __name__ == "__main__":
    main()
