import argparse
import json
import os
import time

import torch

import tide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TM2D EBISU-style forward profiler harness.")
    parser.add_argument("--ny", type=int, default=512)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--nt", type=int, default=256)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--mode", choices=("baseline", "ebisu"), default="ebisu")
    parser.add_argument("--ebisu-steps", type=int, default=4)
    parser.add_argument("--tile-x", type=int, default=64)
    parser.add_argument("--tile-y", type=int, default=16)
    parser.add_argument("--ilp", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_case(args: argparse.Namespace) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float32

    os.environ["TIDE_TM_FUSED_STEPS"] = "0"
    if args.mode == "ebisu":
        os.environ["TIDE_TM_EBISU_STEPS"] = str(args.ebisu_steps)
        os.environ["TIDE_TM_EBISU_TILE_X"] = str(args.tile_x)
        os.environ["TIDE_TM_EBISU_TILE_Y"] = str(args.tile_y)
        os.environ["TIDE_TM_EBISU_ILP"] = str(args.ilp)
    else:
        os.environ["TIDE_TM_EBISU_STEPS"] = "0"

    epsilon = torch.full((args.ny, args.nx), 9.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    ey0 = torch.randn((args.shots, args.ny, args.nx), device=device, dtype=dtype) * 1e-3
    hx0 = torch.randn((args.shots, args.ny, args.nx), device=device, dtype=dtype) * 1e-3
    hz0 = torch.randn((args.shots, args.ny, args.nx), device=device, dtype=dtype) * 1e-3

    kwargs = dict(
        grid_spacing=0.01,
        dt=5e-12,
        source_amplitude=None,
        source_location=None,
        receiver_location=None,
        stencil=2,
        pml_width=0,
        nt=args.nt,
        save_snapshots=False,
    )

    def invoke() -> None:
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            Ey_0=ey0.clone(),
            Hx_0=hx0.clone(),
            Hz_0=hz0.clone(),
            **kwargs,
        )

    for _ in range(args.warmup):
        invoke()
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        invoke()
    torch.cuda.synchronize(device)
    avg_ms = (time.perf_counter() - t0) / args.iters * 1e3

    updates = args.shots * args.ny * args.nx * args.nt
    return {
        "mode": args.mode,
        "ny": args.ny,
        "nx": args.nx,
        "nt": args.nt,
        "shots": args.shots,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "ebisu_steps": args.ebisu_steps if args.mode == "ebisu" else 0,
        "tile_x": args.tile_x if args.mode == "ebisu" else 0,
        "tile_y": args.tile_y if args.mode == "ebisu" else 0,
        "ilp": args.ilp if args.mode == "ebisu" else 0,
        "avg_ms": avg_ms,
        "updates_per_s": updates / (avg_ms * 1e-3),
        "device": torch.cuda.get_device_name(device),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    result = run_case(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
