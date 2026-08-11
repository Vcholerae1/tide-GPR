from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
import sys
from pathlib import Path
from typing import Any

import torch

import tide
from tide import backend_utils


def _shape(value: str, dimension: int) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(","))
    if len(parsed) != dimension or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            f"shape must contain {dimension} comma-separated positive integers"
        )
    return parsed


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _locations(
    shape: tuple[int, ...], shots: int, receivers: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    source_rows: list[list[list[int]]] = []
    receiver_rows: list[list[list[int]]] = []
    for shot in range(shots):
        source = [max(2, size // 3) for size in shape]
        source[-2] = min(shape[-2] - 3, source[-2] + shot % max(1, shape[-2] // 4))
        source_rows.append([source])
        receiver_row: list[list[int]] = []
        for receiver in range(receivers):
            location = source.copy()
            location[-1] = min(
                shape[-1] - 3,
                max(2, shape[-1] // 2 + receiver % max(1, shape[-1] // 3)),
            )
            receiver_row.append(location)
        receiver_rows.append(receiver_row)
    return (
        torch.tensor(source_rows, device=device, dtype=torch.long),
        torch.tensor(receiver_rows, device=device, dtype=torch.long),
    )


def _case(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dtype = torch.float32
    shape = _shape(args.shape, args.dimension)
    epsilon = torch.full(shape, 4.0, device=device, dtype=dtype)
    sigma = torch.full(shape, 2.0e-4, device=device, dtype=dtype)
    mu = torch.ones_like(epsilon)
    source_location, receiver_location = _locations(
        shape, args.shots, args.receivers, device
    )
    source = tide.ricker(
        args.frequency,
        args.nt,
        args.dt,
        peak_time=1.0 / args.frequency,
        device=device,
        dtype=dtype,
    ).view(1, 1, args.nt)
    source = source.expand(args.shots, -1, -1).contiguous()
    spacing = [args.spacing + 0.002 * axis for axis in range(args.dimension)]
    return {
        "shape": shape,
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source": source,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "spacing": spacing,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_once(
    args: argparse.Namespace, case: dict[str, Any], device: torch.device
) -> tuple[float, tuple[int, ...]]:
    epsilon = case["epsilon"].detach().clone().requires_grad_(args.backward)
    sigma = case["sigma"].detach().clone().requires_grad_(args.backward)
    compression: bool | str = (
        False if args.storage_compression == "false" else args.storage_compression
    )
    common = {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": case["mu"],
        "grid_spacing": case["spacing"],
        "dt": args.dt,
        "source_amplitude": case["source"],
        "source_location": case["source_location"],
        "receiver_location": case["receiver_location"],
        "stencil": args.stencil,
        "pml_width": args.pml_width,
        "python_backend": False,
        "storage_mode": args.storage_mode,
        "storage_compression": compression,
        "model_gradient_sampling_interval": args.gradient_sampling_interval,
    }
    _synchronize(device)
    started = time.perf_counter()
    if args.dimension == 2:
        receiver = tide.maxwelltm(**common)[-1]
    else:
        receiver = tide.maxwell3d(
            **common,
            source_component="ey",
            receiver_component="ey",
        )[-1]
    if args.backward:
        receiver.square().mean().backward()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, tuple(receiver.shape)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _load_reference(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    reference = json.loads(path.read_text())
    for key in (
        "dimension",
        "device",
        "shape",
        "nt",
        "shots",
        "receivers",
        "stencil",
        "backward",
        "storage_mode",
        "storage_compression",
    ):
        if reference["config"].get(key) != config.get(key):
            raise ValueError(f"reference config mismatch for {key}")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified native Maxwell runtime benchmark"
    )
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shape", default=None)
    parser.add_argument("--nt", type=int, default=200)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--receivers", type=int, default=4)
    parser.add_argument("--stencil", type=int, choices=(2, 4, 6, 8), default=4)
    parser.add_argument("--pml-width", type=int, default=4)
    parser.add_argument("--spacing", type=float, default=0.018)
    parser.add_argument("--dt", type=float, default=2.0e-11)
    parser.add_argument("--frequency", type=float, default=300e6)
    parser.add_argument(
        "--storage-mode", choices=("device", "cpu", "disk", "auto"), default="device"
    )
    parser.add_argument(
        "--storage-compression", choices=("false", "bf16"), default="false"
    )
    parser.add_argument("--gradient-sampling-interval", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--max-regression", type=float)
    args = parser.parse_args()
    if args.nt <= 0 or args.shots <= 0 or args.receivers <= 0:
        parser.error("nt, shots, and receivers must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative and repeats must be positive")
    if args.shape is None:
        args.shape = "48,64" if args.dimension == 2 else "24,28,32"

    device = _device(args.device)
    if not backend_utils.is_backend_available():
        raise RuntimeError("native TIDE backend is unavailable")
    case = _case(args, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):
        _run_once(args, case, device)
    timings: list[float] = []
    receiver_shape: tuple[int, ...] = ()
    for _ in range(args.repeats):
        elapsed, receiver_shape = _run_once(args, case, device)
        timings.append(elapsed)

    config = {
        "dimension": args.dimension,
        "device": str(device),
        "shape": list(case["shape"]),
        "nt": args.nt,
        "shots": args.shots,
        "receivers": args.receivers,
        "stencil": args.stencil,
        "pml_width": args.pml_width,
        "backward": args.backward,
        "storage_mode": args.storage_mode,
        "storage_compression": args.storage_compression,
        "gradient_sampling_interval": args.gradient_sampling_interval,
    }
    result: dict[str, Any] = {
        "config": config,
        "runtime": {
            "samples_seconds": timings,
            "median_seconds": statistics.median(timings),
            "min_seconds": min(timings),
            "max_seconds": max(timings),
        },
        "memory": {"peak_rss_bytes": _peak_rss_bytes()},
        "receiver_shape": list(receiver_shape),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "tide": tide.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
        },
    }
    if device.type == "cuda":
        result["memory"].update(
            {
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    if args.reference is not None:
        reference = _load_reference(args.reference, config)
        ratio = (
            result["runtime"]["median_seconds"] / reference["runtime"]["median_seconds"]
        )
        result["comparison"] = {
            "runtime_ratio": ratio,
            "regression_fraction": ratio - 1.0,
        }
        if args.max_regression is not None and ratio > 1.0 + args.max_regression:
            raise SystemExit(
                f"runtime regression {ratio - 1.0:.1%} exceeds {args.max_regression:.1%}"
            )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
