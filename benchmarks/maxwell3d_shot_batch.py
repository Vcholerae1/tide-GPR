"""Benchmark native Maxwell 3D CUDA shot batch sizes on synthetic data."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import tide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="70,70,70")
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--shots", default="1,2,4,8,16,32,64")
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--stencil", type=int, default=4, choices=(2, 4, 6, 8))
    parser.add_argument("--n-threads", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--backward", default=True, action="store_true")
    parser.add_argument("--gradient-sampling-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("shape must contain three positive integers")
    shot_counts = [int(value) for value in args.shots.split(",")]
    if min(shot_counts) < 1:
        raise ValueError("shots must be positive integers")
    device = torch.device("cuda")
    dt = 1.6e-11
    epsilon = torch.full(shape, 4.0, device=device, requires_grad=args.backward)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    def run(shots: int) -> float:
        epsilon.grad = None
        receiver_offset = min(10, max(0, shape[2] // 2 - 6))
        z_min = 5
        z_span = shape[2] - receiver_offset - z_min
        if z_span < 1:
            raise ValueError("shape[2] is too small for the receiver offset")
        z_stride = max(1, z_span // shots)
        source = torch.tensor(
            [
                [[5, shape[1] // 2, z_min + (index * z_stride) % z_span]]
                for index in range(shots)
            ],
            device=device,
            dtype=torch.long,
        )
        receiver = source.clone()
        receiver[..., 2] += receiver_offset
        amplitude = (
            tide.ricker(
                160e6,
                args.nt,
                dt,
                peak_time=1.2 / 160e6,
                device=device,
            )
            .view(1, 1, args.nt)
            .expand(shots, -1, -1)
            .contiguous()
        )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = tide.maxwell._kernel_api.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=dt,
            source_amplitude=amplitude,
            source_location=source,
            receiver_location=receiver,
            pml_width=args.pml_width,
            stencil=args.stencil,
            source_component="ey",
            receiver_component="ey",
            n_threads=args.n_threads,
            python_backend=False,
            save_snapshots=args.backward,
            model_gradient_sampling_interval=args.gradient_sampling_interval,
        )
        if args.backward:
            output[-1].square().mean().backward()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    metadata = {
        "gpu": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "shape": shape,
        "nt": args.nt,
        "pml_width": args.pml_width,
        "stencil": args.stencil,
        "n_threads": args.n_threads,
        "backward": args.backward,
        "gradient_sampling_interval": args.gradient_sampling_interval,
    }
    print(json.dumps(metadata), flush=True)
    best: dict[str, float | int] | None = None
    for shots in shot_counts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for _ in range(args.warmup):
                run(shots)
            elapsed = [run(shots) for _ in range(args.repeat)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(json.dumps({"shots": shots, "error": "out_of_memory"}), flush=True)
            break
        median = statistics.median(elapsed)
        record = {
            "shots": shots,
            "median_ms": median,
            "median_ms_per_shot": median / shots,
            "min_ms": min(elapsed),
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "samples_ms": elapsed,
        }
        print(json.dumps(record), flush=True)
        if best is None or record["median_ms_per_shot"] < best["median_ms_per_shot"]:
            best = record
    if best is not None:
        print(
            json.dumps(
                {
                    "best_shots": best["shots"],
                    "median_ms_per_shot": best["median_ms_per_shot"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
