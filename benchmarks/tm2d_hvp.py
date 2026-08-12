"""Benchmark TM2D full/Gauss-Newton HVPs against gradient finite differences."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

import tide


@dataclass(frozen=True)
class Case:
    name: str
    ny: int
    nx: int
    nt: int
    n_shots: int
    n_receivers: int


CASES = {
    case.name: case
    for case in (
        Case("small", 32, 48, 60, 1, 12),
        Case("medium", 96, 128, 300, 2, 48),
        Case("large", 128, 192, 500, 4, 64),
    )
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--cases", default="small,medium,large")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--include-python", action="store_true")
    parser.add_argument("--gradient-sampling-interval", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/tm2d_hvp.json"),
    )
    return parser.parse_args()


def _build_case(
    spec: Case,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | float | int]:
    torch.manual_seed(7)
    pml_width = 10 if min(spec.ny, spec.nx) >= 64 else 4
    epsilon = torch.full((spec.ny, spec.nx), 4.0, device=device, dtype=dtype)
    epsilon[
        spec.ny // 3 : 2 * spec.ny // 3,
        spec.nx // 3 : 2 * spec.nx // 3,
    ] = 5.5
    sigma = torch.full_like(epsilon, 1e-3)
    mu = torch.ones_like(epsilon)

    source_location = torch.empty(spec.n_shots, 1, 2, device=device, dtype=torch.long)
    source_location[:, :, 0] = pml_width + 3
    source_location[:, 0, 1] = (
        torch.linspace(
            pml_width + 3,
            spec.nx - pml_width - 4,
            spec.n_shots,
            device=device,
        )
        .round()
        .long()
    )

    receiver_location = torch.empty(
        spec.n_shots,
        spec.n_receivers,
        2,
        device=device,
        dtype=torch.long,
    )
    receiver_location[:, :, 0] = pml_width + 4
    receiver_location[:, :, 1] = (
        torch.linspace(
            pml_width + 2,
            spec.nx - pml_width - 3,
            spec.n_receivers,
            device=device,
        )
        .round()
        .long()
    )

    dt = 4e-11
    wavelet = tide.ricker(
        300e6,
        spec.nt,
        dt,
        peak_time=1.0 / 300e6,
        device=device,
        dtype=dtype,
    )
    source_amplitude = wavelet.view(1, 1, spec.nt).repeat(spec.n_shots, 1, 1)
    max_vel = 1.5e8

    with torch.no_grad():
        observed_data = tide.maxwell._kernel_api.maxwelltm(
            epsilon * 1.01,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=pml_width,
            max_vel=max_vel,
            python_backend=False,
        )[-1].detach()

    vepsilon = torch.randn_like(epsilon)
    vepsilon /= vepsilon.norm()
    vsigma = torch.randn_like(sigma)
    vsigma *= 1e-3 / vsigma.norm()
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "vepsilon": vepsilon,
        "vsigma": vsigma,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "observed_data": observed_data,
        "grid_spacing": 0.02,
        "dt": dt,
        "pml_width": pml_width,
        "max_vel": max_vel,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | list[float]]:
    for _ in range(warmup):
        fn()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples_ms: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples_ms.append(float(start.elapsed_time(end)))
        else:
            start_time = time.perf_counter()
            fn()
            samples_ms.append((time.perf_counter() - start_time) * 1e3)

    peak_mib = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else 0.0
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "peak_allocated_mib": peak_mib,
        "samples_ms": samples_ms,
    }


def _make_operations(
    data: dict[str, torch.Tensor | float | int],
    *,
    gradient_sampling_interval: int,
) -> dict[str, Callable[[], object]]:
    epsilon = data["epsilon"]
    sigma = data["sigma"]
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)

    forward_kwargs = {
        "mu": data["mu"],
        "grid_spacing": data["grid_spacing"],
        "dt": data["dt"],
        "source_amplitude": data["source_amplitude"],
        "source_location": data["source_location"],
        "receiver_location": data["receiver_location"],
        "pml_width": data["pml_width"],
        "max_vel": data["max_vel"],
        "model_gradient_sampling_interval": gradient_sampling_interval,
    }

    def gradient(epsilon_value: torch.Tensor, sigma_value: torch.Tensor):
        epsilon_req = epsilon_value.detach().clone().requires_grad_(True)
        sigma_req = sigma_value.detach().clone().requires_grad_(True)
        predicted = tide.maxwell._kernel_api.maxwelltm(
            epsilon_req,
            sigma_req,
            python_backend=False,
            storage_compression="bf16",
            **forward_kwargs,
        )[-1]
        loss = 0.5 * (predicted - data["observed_data"]).square().sum()
        return torch.autograd.grad(loss, (epsilon_req, sigma_req))

    def finite_difference():
        step = 1e-2
        plus = gradient(
            epsilon + step * data["vepsilon"],
            sigma + step * data["vsigma"],
        )
        minus = gradient(
            epsilon - step * data["vepsilon"],
            sigma - step * data["vsigma"],
        )
        return tuple((p - m) / (2.0 * step) for p, m in zip(plus, minus))

    hvp_kwargs = {
        **forward_kwargs,
        "epsilon": epsilon,
        "sigma": sigma,
        "observed_data": data["observed_data"],
        "vepsilon": data["vepsilon"],
        "vsigma": data["vsigma"],
        "storage_compression": "bf16",
    }

    return {
        "finite_difference": finite_difference,
        "full_native": lambda: tide.maxwell._kernel_api.maxwelltm_hvp(
            **hvp_kwargs,
            hessian_mode="full",
            python_backend=False,
        ),
        "gauss_newton_native": lambda: tide.maxwell._kernel_api.maxwelltm_hvp(
            **hvp_kwargs,
            hessian_mode="gauss_newton",
            python_backend=False,
        ),
        "full_python": lambda: tide.maxwell._kernel_api.maxwelltm_hvp(
            **{
                key: value
                for key, value in hvp_kwargs.items()
                if key != "storage_compression"
            },
            hessian_mode="full",
            python_backend=True,
        ),
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = sorted(set(selected) - CASES.keys())
    if unknown:
        raise ValueError(f"Unknown benchmark cases: {unknown}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is not available.")

    rows: list[dict[str, object]] = []
    for case_name in selected:
        spec = CASES[case_name]
        data = _build_case(spec, device=device, dtype=dtype)
        operations = _make_operations(
            data,
            gradient_sampling_interval=args.gradient_sampling_interval,
        )
        names = ["finite_difference", "full_native", "gauss_newton_native"]
        if args.include_python and case_name == "small":
            names.append("full_python")
        for operation_name in names:
            metrics = _measure(
                operations[operation_name],
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            row = {
                "case": asdict(spec),
                "operation": operation_name,
                "device": str(device),
                "dtype": str(dtype),
                "gradient_sampling_interval": args.gradient_sampling_interval,
                **metrics,
            }
            rows.append(row)
            print(
                f"{case_name:6s} {operation_name:19s} "
                f"median={metrics['median_ms']:.3f} ms "
                f"peak={metrics['peak_allocated_mib']:.1f} MiB"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
