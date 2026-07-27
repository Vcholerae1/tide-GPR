"""Compare native, scalar-FP16-I/O, and SeisCL-style half2 Maxwell 3D paths."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

import tide


MODES = ("native", "fp16_io", "fp16_io_half2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="70,70,70")
    parser.add_argument("--nt", type=int, default=300)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--stencil", type=int, default=4, choices=(2, 4, 6, 8))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 3 or min(shape) < 4:
        raise ValueError("shape must contain three integers >= 4")
    device = torch.device("cuda")
    dt = 1.6e-11
    z = torch.arange(shape[0], device=device).view(-1, 1, 1)
    epsilon = (3.0 + 2.0 * (z >= shape[0] // 2)).expand(shape).contiguous()
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source = torch.tensor(
        [[[5, shape[1] // 2, shape[2] // 2]]] * args.shots,
        device=device,
        dtype=torch.long,
    )
    receiver = source.clone()
    receiver[..., 2] += min(10, max(0, shape[2] // 2 - 6))
    amplitude = tide.ricker(
        160e6, args.nt, dt, peak_time=1.2 / 160e6, device=device
    ).view(1, 1, args.nt).expand(args.shots, -1, -1).contiguous()

    kwargs = dict(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=0.02,
        dt=dt,
        source_amplitude=amplitude,
        source_location=source,
        receiver_location=receiver,
        pml_width=args.pml_width,
        stencil=args.stencil,
        python_backend=False,
    )

    def run(mode: str) -> tuple[float, torch.Tensor, float]:
        if mode == "fp16_io_half2":
            os.environ["TIDE_EM3D_FP16_HALF2"] = "1"
        else:
            os.environ.pop("TIDE_EM3D_FP16_HALF2", None)
        compute_mode = "native" if mode == "native" else "fp16_io"
        torch.cuda.synchronize()
        baseline_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        outputs = tide.maxwell3d(**kwargs, compute_mode=compute_mode)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        receiver_data = outputs[-1].detach().clone()
        peak_increment = torch.cuda.max_memory_allocated() - baseline_bytes
        del outputs
        return elapsed, receiver_data, peak_increment / 2**20

    results: dict[str, dict[str, float | list[float]]] = {}
    reference = None
    for mode in MODES:
        for _ in range(args.warmup):
            run(mode)
        samples = []
        receiver_data = None
        peak_mib = 0.0
        for _ in range(args.repeat):
            elapsed, receiver_data, current_peak_mib = run(mode)
            samples.append(elapsed)
            peak_mib = max(peak_mib, current_peak_mib)
        assert receiver_data is not None
        if reference is None:
            reference = receiver_data
        delta = receiver_data - reference
        reference_norm = reference.norm()
        rel_l2 = float(delta.norm() / reference_norm) if reference_norm > 0 else 0.0
        correlation = float(torch.nn.functional.cosine_similarity(
            reference.flatten(), receiver_data.flatten(), dim=0
        ))
        results[mode] = {
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "peak_increment_mib": peak_mib,
            "relative_l2_vs_native": rel_l2,
            "correlation_vs_native": correlation,
            "samples_ms": samples,
        }

    native_ms = float(results["native"]["median_ms"])
    scalar_ms = float(results["fp16_io"]["median_ms"])
    for mode in MODES:
        mode_ms = float(results[mode]["median_ms"])
        results[mode]["speedup_vs_native"] = native_ms / mode_ms
        results[mode]["speedup_vs_scalar_fp16"] = scalar_ms / mode_ms

    print(json.dumps({
        "gpu": torch.cuda.get_device_name(device),
        "shape": shape,
        "nt": args.nt,
        "shots": args.shots,
        "pml_width": args.pml_width,
        "stencil": args.stencil,
        "results": results,
    }))


if __name__ == "__main__":
    main()
