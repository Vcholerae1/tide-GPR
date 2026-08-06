"""Benchmark native Maxwell 3D CUDA launch configurations on synthetic data."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import tide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="70,70,70")
    parser.add_argument("--nt", type=int, default=300)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--stencil", type=int, default=4, choices=(2, 4, 6, 8))
    parser.add_argument("--n-threads", default="0,32,64,128,256,512")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--gradient-sampling-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("shape must contain three positive integers")
    thread_counts = [int(value) for value in args.n_threads.split(",")]
    device = torch.device("cuda")
    dt = 1.6e-11
    epsilon = torch.full(shape, 4.0, device=device, requires_grad=args.backward)
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
        160e6,
        args.nt,
        dt,
        peak_time=1.2 / 160e6,
        device=device,
    ).view(1, 1, args.nt).expand(args.shots, -1, -1).contiguous()

    def run(n_threads: int) -> float:
        epsilon.grad = None
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = tide.maxwell3d(
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
            n_threads=n_threads,
            python_backend=False,
            save_snapshots=args.backward,
            compute_mode="native",
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
        "shots": args.shots,
        "pml_width": args.pml_width,
        "stencil": args.stencil,
        "backward": args.backward,
        "gradient_sampling_interval": args.gradient_sampling_interval,
        "compute_mode": "native",
    }
    print(json.dumps(metadata), flush=True)
    for n_threads in thread_counts:
        for _ in range(args.warmup):
            run(n_threads)
        elapsed = [run(n_threads) for _ in range(args.repeat)]
        print(
            json.dumps(
                {
                    "n_threads": n_threads,
                    "median_ms": statistics.median(elapsed),
                    "median_ms_per_shot": statistics.median(elapsed) / args.shots,
                    "min_ms": min(elapsed),
                    "samples_ms": elapsed,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
