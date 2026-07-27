from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import tide


def _timed(
    case: dict, mode: str, warmup: int, repeat: int
) -> tuple[float, float, torch.Tensor]:
    for _ in range(warmup):
        tide.maxwelltm(**case, compute_mode=mode)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    output = None
    for _ in range(repeat):
        start = time.perf_counter()
        output = tide.maxwelltm(**case, compute_mode=mode)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    assert output is not None
    return (
        statistics.median(samples),
        torch.cuda.max_memory_allocated() / 2**20,
        output[-1].detach(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TM2D FP16 I/O mode")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--nt", type=int, default=400)
    parser.add_argument("--stencil", type=int, default=4, choices=(2, 4, 6, 8))
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--receivers", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    n = args.size
    epsilon = torch.full((n, n), 4.0, device=device)
    epsilon[n // 2 :, :] = 9.0
    sigma = torch.full_like(epsilon, 1e-3)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(200e6, args.nt, 4e-11, device=device).reshape(1, 1, -1)
    source = source.expand(args.shots, -1, -1).contiguous()
    source_location = torch.tensor([[[n // 4, n // 3]]], device=device)
    source_location = source_location.expand(args.shots, -1, -1).contiguous()
    receiver_x = torch.linspace(
        max(args.pml_width, 1),
        n - max(args.pml_width, 1) - 1,
        args.receivers,
        device=device,
    ).round()
    receiver_location = torch.stack(
        (torch.full_like(receiver_x, n // 4), receiver_x), dim=-1
    ).long()
    receiver_location = receiver_location[None].expand(args.shots, -1, -1).contiguous()
    case = {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "grid_spacing": 0.02,
        "dt": 4e-11,
        "source_amplitude": source,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "pml_width": args.pml_width,
        "stencil": args.stencil,
    }

    native_ms, native_mib, native_record = _timed(
        case, "native", args.warmup, args.repeat
    )
    fp16_ms, fp16_mib, fp16_record = _timed(case, "fp16_io", args.warmup, args.repeat)
    reference = native_record.float().reshape(-1)
    actual = fp16_record.float().reshape(-1)
    relative_l2 = torch.linalg.vector_norm(
        actual - reference
    ) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    correlation = torch.nn.functional.cosine_similarity(reference, actual, dim=0)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "size": n,
                "shots": args.shots,
                "receivers": args.receivers,
                "nt": args.nt,
                "stencil": args.stencil,
                "native_ms": native_ms,
                "fp16_io_ms": fp16_ms,
                "speedup": native_ms / fp16_ms,
                "native_peak_mib": native_mib,
                "fp16_io_peak_mib": fp16_mib,
                "relative_l2": float(relative_l2),
                "correlation": float(correlation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
