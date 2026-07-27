"""Benchmark reusable TM2D HVP contexts against 2K central differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

import tide
from tm2d_hvp import CASES, _build_case, _measure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case", choices=tuple(CASES), default="large")
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fd-step", type=float, default=1e-2)
    parser.add_argument(
        "--storage-compression",
        choices=("none", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/tm2d_hvp_batch.json"),
    )
    return parser.parse_args()


def _relative_error(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> float:
    actual_flat = torch.cat([part.reshape(-1) for part in actual])
    expected_flat = torch.cat([part.reshape(-1) for part in expected])
    return float(
        torch.linalg.vector_norm(actual_flat - expected_flat)
        / (torch.linalg.vector_norm(expected_flat) + 1e-30)
    )


def main() -> None:
    args = _parse_args()
    if args.directions < 1 or args.block_size < 1:
        raise ValueError("directions and block-size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    data = _build_case(CASES[args.case], device=device, dtype=torch.float32)
    epsilon = data["epsilon"]
    sigma = data["sigma"]
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)

    generator = torch.Generator(device=device).manual_seed(19)
    vepsilon = torch.randn(
        args.directions,
        *epsilon.shape,
        device=device,
        dtype=epsilon.dtype,
        generator=generator,
    )
    vepsilon /= torch.linalg.vector_norm(
        vepsilon.reshape(args.directions, -1), dim=1
    ).view(-1, 1, 1)
    vsigma = torch.randn(
        args.directions,
        *sigma.shape,
        device=device,
        dtype=sigma.dtype,
        generator=generator,
    )
    vsigma *= (
        1e-3 / torch.linalg.vector_norm(vsigma.reshape(args.directions, -1), dim=1)
    ).view(-1, 1, 1)
    storage_compression: bool | str = (
        False if args.storage_compression == "none" else args.storage_compression
    )

    common = {
        "grid_spacing": data["grid_spacing"],
        "dt": data["dt"],
        "source_amplitude": data["source_amplitude"],
        "source_location": data["source_location"],
        "receiver_location": data["receiver_location"],
        "observed_data": data["observed_data"],
        "pml_width": data["pml_width"],
        "max_vel": data["max_vel"],
        "storage_compression": storage_compression,
    }

    def independent_hvps() -> tuple[torch.Tensor, torch.Tensor]:
        results = [
            tide.maxwelltm_hvp(
                epsilon,
                sigma,
                data["mu"],
                vepsilon=vepsilon[index],
                vsigma=vsigma[index],
                hessian_mode="full",
                **common,
            )
            for index in range(args.directions)
        ]
        return tuple(torch.stack(parts) for parts in zip(*results))

    def context_hvps() -> tuple[torch.Tensor, torch.Tensor]:
        with tide.linearize_maxwelltm(
            epsilon,
            sigma,
            data["mu"],
            hessian_mode="full",
            **common,
        ) as context:
            return context.hvp_batch(
                vepsilon=vepsilon,
                vsigma=vsigma,
                block_size=args.block_size,
            )

    def gradient(
        epsilon_value: torch.Tensor, sigma_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon_req = epsilon_value.detach().clone().requires_grad_(True)
        sigma_req = sigma_value.detach().clone().requires_grad_(True)
        predicted = tide.maxwelltm(
            epsilon_req,
            sigma_req,
            data["mu"],
            grid_spacing=data["grid_spacing"],
            dt=data["dt"],
            source_amplitude=data["source_amplitude"],
            source_location=data["source_location"],
            receiver_location=data["receiver_location"],
            pml_width=data["pml_width"],
            max_vel=data["max_vel"],
            storage_compression=storage_compression,
            python_backend=False,
        )[-1]
        loss = 0.5 * (predicted - data["observed_data"]).square().sum()
        return torch.autograd.grad(loss, (epsilon_req, sigma_req))

    def central_differences() -> tuple[torch.Tensor, torch.Tensor]:
        epsilon_parts: list[torch.Tensor] = []
        sigma_parts: list[torch.Tensor] = []
        for index in range(args.directions):
            plus = gradient(
                epsilon + args.fd_step * vepsilon[index],
                sigma + args.fd_step * vsigma[index],
            )
            minus = gradient(
                epsilon - args.fd_step * vepsilon[index],
                sigma - args.fd_step * vsigma[index],
            )
            epsilon_parts.append((plus[0] - minus[0]) / (2.0 * args.fd_step))
            sigma_parts.append((plus[1] - minus[1]) / (2.0 * args.fd_step))
        return torch.stack(epsilon_parts), torch.stack(sigma_parts)

    operations: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {
        "independent_full_hvp": independent_hvps,
        "context_full_hvp": context_hvps,
        "central_difference_2k": central_differences,
    }
    reference = independent_hvps()
    accuracy = {
        "context_vs_independent": _relative_error(context_hvps(), reference),
        "central_difference_vs_independent": _relative_error(
            central_differences(), reference
        ),
    }
    rows: list[dict[str, object]] = []
    for name, operation in operations.items():
        metrics = _measure(
            operation,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        row = {
            "operation": name,
            "case": args.case,
            "directions": args.directions,
            "block_size": args.block_size,
            "storage_compression": args.storage_compression,
            "device": str(device),
            **metrics,
        }
        rows.append(row)
        print(
            f"{name:24s} median={metrics['median_ms']:.3f} ms "
            f"peak={metrics['peak_allocated_mib']:.1f} MiB"
        )
    print(json.dumps(accuracy, indent=2))
    payload = {"measurements": rows, "accuracy": accuracy}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
