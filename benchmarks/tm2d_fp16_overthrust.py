"""End-to-end 100-shot Overthrust benchmark for TM2D reduced storage modes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import tide


def make_problem(device: torch.device, nt: int) -> dict:
    epsilon = torch.from_numpy(
        np.load(Path("data/examples/OverThrust.npy")).astype(np.float32)
    ).to(device)
    n_shots = 100
    source_x = torch.arange(n_shots, device=device) * 4
    receiver_x = source_x + 1
    acquisition = tide.workflow.line_acquisition_2d(
        source_x, receiver_x, source_depth=2, receiver_mode="paired"
    )
    wavelet = tide.ricker(
        6.0e8, nt, 4.0e-11, peak_time=1.0 / 6.0e8, device=device
    )
    return {
        "epsilon": epsilon,
        "sigma": torch.full_like(epsilon, 1.0e-3),
        "mu": torch.ones_like(epsilon),
        "source_amplitude": tide.workflow.expand_source_amplitude(
            wavelet, n_shots
        ),
        "source_location": acquisition.source_location,
        "receiver_location": acquisition.receiver_location,
    }


def run_once(
    problem: dict,
    *,
    mode: str,
    batch_size: int,
    gradient: bool,
    half2: bool,
    half2_arithmetic: bool = False,
) -> tuple[float, float, torch.Tensor, torch.Tensor | None]:
    if half2:
        os.environ["TIDE_TM_FP16_HALF2"] = "1"
    else:
        os.environ["TIDE_TM_FP16_HALF2"] = "0"
    if half2_arithmetic:
        os.environ["TIDE_TM_FP16_HALF2_ARITH"] = "1"
    else:
        os.environ.pop("TIDE_TM_FP16_HALF2_ARITH", None)
    epsilon = problem["epsilon"].detach().clone().requires_grad_(gradient)
    records = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for first in range(0, 100, batch_size):
        last = min(first + batch_size, 100)
        record = tide.maxwelltm(
            epsilon,
            problem["sigma"],
            problem["mu"],
            0.02,
            4.0e-11,
            source_amplitude=problem["source_amplitude"][first:last],
            source_location=problem["source_location"][first:last],
            receiver_location=problem["receiver_location"][first:last],
            pml_width=20,
            stencil=4,
            model_gradient_sampling_interval=10 if gradient else 1,
            storage_mode="device" if gradient else "auto",
            storage_compression="bf16",
            compute_mode=mode,
        )[-1]
        records.append(record.detach())
        if gradient:
            time_index = torch.arange(record.shape[0], device=record.device)[
                :, None, None
            ]
            shot_index = torch.arange(first, last, device=record.device)[
                None, :, None
            ]
            receiver_index = torch.arange(
                record.shape[2], device=record.device
            )[None, None, :]
            upstream = torch.sin(
                (
                    (time_index * 100 + shot_index) * record.shape[2]
                    + receiver_index
                )
                * 0.173
            )
            record.backward(upstream)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    return elapsed, peak_mib, torch.cat(records, dim=1), epsilon.grad


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    x = actual.float().reshape(-1)
    y = reference.float().reshape(-1)
    relative_l2 = torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(
        y
    ).clamp_min(1e-30)
    correlation = torch.nn.functional.cosine_similarity(x, y, dim=0)
    return {"relative_l2": float(relative_l2), "correlation": float(correlation)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nt", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--gradient", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    problem = make_problem(torch.device("cuda"), args.nt)
    variants = (
        ("native", False, False),
        ("fp16_io", False, False),
        ("fp16_io", True, False),
        ("fp16_io", True, True),
    )
    results = {}
    outputs = {}
    gradients = {}
    for mode, half2, half2_arithmetic in variants:
        name = (
            "fp16_half2_arithmetic"
            if half2_arithmetic
            else "fp16_half2"
            if half2
            else mode
        )
        samples = []
        for _ in range(args.repeat):
            elapsed, peak, output, gradient = run_once(
                problem,
                mode=mode,
                batch_size=args.batch_size,
                gradient=args.gradient,
                half2=half2,
                half2_arithmetic=half2_arithmetic,
            )
            samples.append(elapsed)
        results[name] = {
            "total_seconds_median": statistics.median(samples),
            "total_seconds_samples": samples,
            "peak_mib": peak,
        }
        outputs[name] = output
        gradients[name] = gradient
    for name in ("fp16_io", "fp16_half2", "fp16_half2_arithmetic"):
        results[name]["record_vs_native"] = metrics(outputs[name], outputs["native"])
        results[name]["speedup_vs_native"] = (
            results["native"]["total_seconds_median"]
            / results[name]["total_seconds_median"]
        )
        if args.gradient:
            results[name]["gradient_vs_native"] = metrics(
                gradients[name], gradients["native"]
            )
    results["fp16_half2"]["speedup_vs_fp16_scalar"] = (
        results["fp16_io"]["total_seconds_median"]
        / results["fp16_half2"]["total_seconds_median"]
    )
    results["fp16_half2_arithmetic"]["speedup_vs_fp16_half2"] = (
        results["fp16_half2"]["total_seconds_median"]
        / results["fp16_half2_arithmetic"]["total_seconds_median"]
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "shape": [200, 400],
                "shots": 100,
                "nt": args.nt,
                "batch_size": args.batch_size,
                "gradient": args.gradient,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
