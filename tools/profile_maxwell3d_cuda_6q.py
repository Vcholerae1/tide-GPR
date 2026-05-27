"""Benchmark harness for 3D Maxwell CUDA 6Q profiling.

This script measures the end-to-end native CUDA forward operator and emits a
JSON workload record that can be reused with Nsight Systems or Nsight Compute.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

import tide
from tide import backend_utils


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def spatial_int(value: str) -> int:
    parsed = positive_int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("spatial dimensions must be at least 3")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_locations(
    *,
    shots: int,
    sources: int,
    receivers: int,
    nz: int,
    ny: int,
    nx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_locations = torch.empty(shots, sources, 3, dtype=torch.long, device=device)
    receiver_locations = torch.empty(
        shots, receivers, 3, dtype=torch.long, device=device
    )

    src_x = max(1, min(nx - 2, nx // 3))
    rec_x = max(1, min(nx - 2, (2 * nx) // 3))
    z_center = max(1, min(nz - 2, nz // 2))

    source_y = torch.linspace(
        1,
        max(1, ny - 2),
        steps=sources,
        device=device,
        dtype=torch.float32,
    ).round().to(torch.long)
    receiver_y = torch.linspace(
        1,
        max(1, ny - 2),
        steps=receivers,
        device=device,
        dtype=torch.float32,
    ).round().to(torch.long)

    for shot in range(shots):
        offset = shot % max(1, ny - 2)
        source_locations[shot, :, 0] = z_center
        source_locations[shot, :, 1] = ((source_y - 1 + offset) % max(1, ny - 2)) + 1
        source_locations[shot, :, 2] = src_x

        receiver_locations[shot, :, 0] = z_center
        receiver_locations[shot, :, 1] = (
            (receiver_y - 1 + offset) % max(1, ny - 2)
        ) + 1
        receiver_locations[shot, :, 2] = rec_x

    return source_locations, receiver_locations


def padded_shape(
    *,
    nz: int,
    ny: int,
    nx: int,
    pml: int,
    stencil: int,
) -> tuple[int, int, int]:
    fd_pad = stencil // 2
    low_high_fd = fd_pad + max(0, fd_pad - 1)
    return (
        nz + (2 * pml) + low_high_fd,
        ny + (2 * pml) + low_high_fd,
        nx + (2 * pml) + low_high_fd,
    )


def build_case(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dtype = getattr(torch, args.dtype)

    epsilon = torch.full(
        (args.nz, args.ny, args.nx),
        args.epsilon,
        device=device,
        dtype=dtype,
    )
    sigma = torch.full_like(epsilon, args.sigma)
    mu = torch.ones_like(epsilon) * args.mu
    if args.model_batched:
        epsilon = epsilon.unsqueeze(0).expand(args.shots, -1, -1, -1).contiguous()
        sigma = sigma.unsqueeze(0).expand(args.shots, -1, -1, -1).contiguous()
        mu = mu.unsqueeze(0).expand(args.shots, -1, -1, -1).contiguous()

    source_location, receiver_location = build_locations(
        shots=args.shots,
        sources=args.sources,
        receivers=args.receivers,
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        device=device,
    )

    wavelet = tide.ricker(
        args.freq,
        args.nt,
        args.dt,
        peak_time=args.peak_time if args.peak_time is not None else 1.0 / args.freq,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, args.nt).expand(
        args.shots, args.sources, -1
    ).contiguous()

    max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * tide.utils.C0
    inner_dt, step_ratio = tide.cfl_condition(args.grid_spacing, args.dt, max_vel)

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "max_vel": max_vel,
        "inner_dt": inner_dt,
        "step_ratio": int(step_ratio),
    }


def run_forward(args: argparse.Namespace, case: dict[str, Any]):
    return tide.maxwell3d(
        case["epsilon"],
        case["sigma"],
        case["mu"],
        grid_spacing=args.grid_spacing,
        dt=args.dt,
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        stencil=args.stencil,
        pml_width=args.pml,
        max_vel=case["max_vel"],
        source_component=args.source_component,
        receiver_component=args.receiver_component,
        python_backend=False,
        storage_mode="device",
        n_threads=args.n_threads,
    )


def quantiles(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"min": values[0], "median": values[0], "max": values[0]}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not backend_utils.is_backend_available():
        raise SystemExit("tide native backend is not available")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    with nvtx_range("tide_em3d_setup", not args.no_nvtx):
        case = build_case(args, device)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for i in range(args.warmup):
            with nvtx_range(f"tide_em3d_warmup_{i}", not args.no_nvtx):
                run_forward(args, case)
        torch.cuda.synchronize(device)

        times_s: list[float] = []
        last_output = None
        for i in range(args.iters):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with nvtx_range(f"tide_em3d_measure_{i}", not args.no_nvtx):
                last_output = run_forward(args, case)
            torch.cuda.synchronize(device)
            times_s.append(time.perf_counter() - started)

    assert last_output is not None
    receiver = last_output[-1]
    receiver_norm = float(receiver.float().norm().item()) if receiver.numel() else 0.0
    field_norm = float(last_output[1].float().norm().item())

    padded_nz, padded_ny, padded_nx = padded_shape(
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        pml=args.pml,
        stencil=args.stencil,
    )
    nt_internal = args.nt * case["step_ratio"]
    padded_cells = padded_nz * padded_ny * padded_nx
    cell_steps = args.shots * padded_cells * nt_internal
    mean_s = statistics.fmean(times_s)

    result = {
        "operator": "tide.maxwell3d native CUDA forward",
        "device": {
            "index": args.device,
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "torch_cuda": torch.version.cuda,
            "torch_version": torch.__version__,
        },
        "workload": {
            "dtype": args.dtype,
            "model_shape": [args.nz, args.ny, args.nx],
            "model_batched": args.model_batched,
            "padded_shape_estimate": [padded_nz, padded_ny, padded_nx],
            "padded_cells_per_shot": padded_cells,
            "shots": args.shots,
            "sources_per_shot": args.sources,
            "receivers_per_shot": args.receivers,
            "nt": args.nt,
            "nt_internal": nt_internal,
            "stencil": args.stencil,
            "pml_width_each_side": args.pml,
            "grid_spacing": args.grid_spacing,
            "dt_requested": args.dt,
            "dt_internal": case["inner_dt"],
            "step_ratio": case["step_ratio"],
            "epsilon": args.epsilon,
            "sigma": args.sigma,
            "mu": args.mu,
            "source_component": args.source_component,
            "receiver_component": args.receiver_component,
            "n_threads_arg": args.n_threads,
            "expected_forward_kernels_per_step": 4,
        },
        "measurement": {
            "warmup": args.warmup,
            "iters": args.iters,
            "times_s": times_s,
            "mean_s": mean_s,
            **quantiles(times_s),
            "cell_steps": cell_steps,
            "cell_steps_per_s": cell_steps / mean_s,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "receiver_norm": receiver_norm,
            "ey_norm": field_norm,
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile tide.maxwell3d native CUDA forward as a fixed 6Q workload."
    )
    parser.add_argument("--nz", type=spatial_int, default=100)
    parser.add_argument("--ny", type=spatial_int, default=100)
    parser.add_argument("--nx", type=spatial_int, default=100)
    parser.add_argument("--nt", type=positive_int, default=1000)
    parser.add_argument("--shots", type=positive_int, default=1)
    parser.add_argument("--sources", type=positive_int, default=1)
    parser.add_argument("--receivers", type=positive_int, default=32)
    parser.add_argument("--stencil", type=int, choices=(2, 4, 6, 8), default=2)
    parser.add_argument("--pml", type=nonnegative_int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--grid-spacing", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=4.0e-11)
    parser.add_argument("--freq", type=float, default=90.0e6)
    parser.add_argument("--peak-time", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--source-component", choices=("ex", "ey", "ez"), default="ey")
    parser.add_argument(
        "--receiver-component", choices=("ex", "ey", "ez"), default="ey"
    )
    parser.add_argument("--model-batched", action="store_true")
    parser.add_argument("--n-threads", type=nonnegative_int, default=0)
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--iters", type=positive_int, default=5)
    parser.add_argument("--device", type=nonnegative_int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-nvtx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
