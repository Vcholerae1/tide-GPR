"""Benchmark harness for 3D Maxwell CUDA 6Q profiling.

This script measures the end-to-end native CUDA forward operator and emits a
JSON workload record that can be reused with Nsight Systems or Nsight Compute.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


def normalize_cell_threads(n_threads: int) -> int:
    if n_threads <= 0:
        return 256
    if n_threads <= 32:
        return 32
    if n_threads <= 64:
        return 64
    if n_threads <= 128:
        return 128
    if n_threads <= 256:
        return 256
    return 512


def spatial_block_shape(threads: int) -> list[int]:
    if threads <= 32:
        return [32, 1, 1]
    if threads <= 64:
        return [32, 2, 1]
    if threads <= 128:
        return [32, 4, 1]
    if threads <= 256:
        return [32, 8, 1]
    return [32, 8, 2]


def ceil_div(num: int, den: int) -> int:
    return (num + den - 1) // den


def env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value != "" and value[0] != "0"


def launch_config_estimate(
    *,
    shots: int,
    padded_nz: int,
    padded_ny: int,
    padded_nx: int,
    n_threads: int,
    enable_spatial_3d: bool,
    spatial_disable_reason: str | None,
) -> dict[str, Any]:
    threads = normalize_cell_threads(n_threads)
    padded_cells = padded_nz * padded_ny * padded_nx
    total_cells = shots * padded_cells
    if n_threads <= 0 or not enable_spatial_3d:
        return {
            "cell_launch_mode": "legacy_1d",
            "requested_n_threads": n_threads,
            "normalized_threads_per_block": threads,
            "spatial_3d_enabled": False,
            "spatial_3d_disable_reason": (
                "n_threads<=0" if n_threads <= 0 else spatial_disable_reason
            ),
            "cell_block_shape_xyz": [threads, 1, 1],
            "cell_grid_shape_xyz": [ceil_div(total_cells, threads), 1, 1],
            "x_contiguous_threads": min(threads, 32),
            "padded_cells_per_shot": padded_cells,
            "total_launched_cell_threads": ceil_div(total_cells, threads) * threads,
        }

    bx, by, bz = spatial_block_shape(threads)
    gx = ceil_div(padded_nx, bx)
    gy = ceil_div(padded_ny, by)
    gz_per_shot = ceil_div(padded_nz, bz)
    gz = shots * gz_per_shot
    return {
        "cell_launch_mode": "spatial_3d",
        "requested_n_threads": n_threads,
        "normalized_threads_per_block": threads,
        "spatial_3d_enabled": True,
        "spatial_3d_disable_reason": None,
        "cell_block_shape_xyz": [bx, by, bz],
        "cell_grid_shape_xyz": [gx, gy, gz],
        "spatial_blocks_z_per_shot": gz_per_shot,
        "x_contiguous_threads": bx,
        "padded_cells_per_shot": padded_cells,
        "total_launched_cell_threads": gx * gy * gz * bx * by * bz,
    }


_NCU_METRIC_KEYS = {
    "launch__registers_per_thread": "registers_per_thread",
    "launch__occupancy_limit_registers": "occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem": "occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks": "occupancy_limit_blocks",
    "launch__occupancy_limit_warps": "occupancy_limit_warps",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
    "dram__bytes.sum": "dram_bytes_sum",
    "dram__bytes.sum.per_second": "dram_bytes_per_second",
    "lts__t_bytes.sum": "l2_bytes_sum",
    "lts__t_sectors.sum": "l2_t_sectors_sum",
}


def parse_ncu_value(raw_value: str) -> float | str:
    value_text = raw_value.replace(",", "").strip()
    try:
        return float(value_text)
    except ValueError:
        return raw_value.strip()


def parse_ncu_csv(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    metrics: dict[str, Any] = {value: None for value in _NCU_METRIC_KEYS.values()}
    if not path.exists():
        return {"path": str(path), "error": "missing", **metrics}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("=="))
        for row in reader:
            metric_name = (
                row.get("Metric Name")
                or row.get("Metric Name ")
                or row.get("Name")
                or row.get("Metric")
            )
            if metric_name in _NCU_METRIC_KEYS:
                raw_value = (
                    row.get("Metric Value")
                    or row.get("Metric Value ")
                    or row.get("Value")
                    or row.get("Avg")
                )
                if raw_value is not None:
                    metrics[_NCU_METRIC_KEYS[metric_name]] = parse_ncu_value(
                        raw_value
                    )
                continue
            for metric_name, output_name in _NCU_METRIC_KEYS.items():
                raw_value = row.get(metric_name)
                if raw_value is not None and raw_value.strip() != "":
                    metrics[output_name] = parse_ncu_value(raw_value)
    return {"path": str(path), **metrics}


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
    if args.heterogeneous_model:
        z = torch.linspace(-1.0, 1.0, args.nz, device=device, dtype=dtype).view(
            args.nz, 1, 1
        )
        y = torch.linspace(-1.0, 1.0, args.ny, device=device, dtype=dtype).view(
            1, args.ny, 1
        )
        x = torch.linspace(-1.0, 1.0, args.nx, device=device, dtype=dtype).view(
            1, 1, args.nx
        )
        epsilon = epsilon * (1.0 + 0.12 * z + 0.08 * y - 0.05 * x)
        mu = mu * (1.0 + 0.03 * y)
        if args.sigma != 0.0:
            sigma = sigma * (1.0 + 0.10 * x)
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
        dispersion=(
            tide.DebyeDispersion(
                delta_epsilon=args.debye_delta_epsilon,
                tau=args.debye_tau,
            )
            if args.debye
            else None
        ),
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
    if args.spatial_launch:
        os.environ["TIDE_EM3D_SPATIAL_LAUNCH"] = "1"
    if args.uniform_coeffs:
        os.environ["TIDE_EM3D_UNIFORM_COEFFS"] = "1"
    spatial_launch_requested = env_flag_enabled("TIDE_EM3D_SPATIAL_LAUNCH")
    uniform_coeffs_requested = env_flag_enabled("TIDE_EM3D_UNIFORM_COEFFS")

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
    spatial_disable_reason = None
    if not spatial_launch_requested:
        spatial_disable_reason = "env_disabled"
    elif args.debye:
        spatial_disable_reason = "debye"
    elif args.shots != 1:
        spatial_disable_reason = "multi_shot"
    elif args.stencil != 2:
        spatial_disable_reason = "stencil"

    launch_config = launch_config_estimate(
        shots=args.shots,
        padded_nz=padded_nz,
        padded_ny=padded_ny,
        padded_nx=padded_nx,
        n_threads=args.n_threads,
        enable_spatial_3d=spatial_disable_reason is None,
        spatial_disable_reason=spatial_disable_reason,
    )
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
            "heterogeneous_model": args.heterogeneous_model,
            "debye": {
                "enabled": args.debye,
                "delta_epsilon": args.debye_delta_epsilon if args.debye else None,
                "tau": args.debye_tau if args.debye else None,
            },
            "source_component": args.source_component,
            "receiver_component": args.receiver_component,
            "n_threads_arg": args.n_threads,
            "spatial_launch_requested": spatial_launch_requested,
            "uniform_coeffs_requested": uniform_coeffs_requested,
            "launch_config_estimate": launch_config,
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
        "profiler_metrics": {
            "forward_kernel_h": parse_ncu_csv(args.ncu_forward_kernel_h_csv),
            "forward_kernel_e": parse_ncu_csv(args.ncu_forward_kernel_e_csv),
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
    parser.add_argument("--debye", action="store_true")
    parser.add_argument("--debye-delta-epsilon", type=float, default=1.0)
    parser.add_argument("--debye-tau", type=float, default=5.0e-10)
    parser.add_argument("--source-component", choices=("ex", "ey", "ez"), default="ey")
    parser.add_argument(
        "--receiver-component", choices=("ex", "ey", "ez"), default="ey"
    )
    parser.add_argument("--model-batched", action="store_true")
    parser.add_argument("--n-threads", type=nonnegative_int, default=0)
    parser.add_argument(
        "--spatial-launch",
        action="store_true",
        help="Set TIDE_EM3D_SPATIAL_LAUNCH=1 for experimental 3D cell launch.",
    )
    parser.add_argument(
        "--uniform-coeffs",
        action="store_true",
        help="Set TIDE_EM3D_UNIFORM_COEFFS=1 for experimental scalar material coefficient loads.",
    )
    parser.add_argument(
        "--heterogeneous-model",
        action="store_true",
        help="Use a deterministic non-uniform epsilon/mu model for performance runs.",
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--iters", type=positive_int, default=5)
    parser.add_argument("--device", type=nonnegative_int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ncu-forward-kernel-h-csv", type=Path, default=None)
    parser.add_argument("--ncu-forward-kernel-e-csv", type=Path, default=None)
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
