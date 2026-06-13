"""Benchmark 3D Maxwell CUDA gradient snapshot storage modes.

The benchmark measures an end-to-end training step:

    forward_with_storage -> receiver loss -> backward

It is intentionally separate from the forward-only 6Q harness because gradient
performance is dominated by snapshot write/read traffic and adjoint replay
rather than the plain H/E forward kernels alone.
"""

from __future__ import annotations

import argparse
import json
import math
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
def env_override(overrides: dict[str, str | None]):
    old_values = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def quantiles(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"min": values[0], "median": values[0], "max": values[0]}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def rel_l2(actual: torch.Tensor | None, expected: torch.Tensor | None) -> float | None:
    if actual is None or expected is None:
        return None
    diff = (actual.float() - expected.float()).reshape(-1)
    denom = (
        expected.float().reshape(-1).norm().clamp_min(torch.finfo(torch.float32).tiny)
    )
    return float(diff.norm().item() / denom.item())


def max_abs(actual: torch.Tensor | None, expected: torch.Tensor | None) -> float | None:
    if actual is None or expected is None:
        return None
    if actual.numel() == 0:
        return 0.0
    return float((actual.float() - expected.float()).abs().max().item())


def cosine_similarity(
    actual: torch.Tensor | None, expected: torch.Tensor | None
) -> float | None:
    if actual is None or expected is None:
        return None
    actual_flat = actual.float().reshape(-1)
    expected_flat = expected.float().reshape(-1)
    actual_norm = actual_flat.norm()
    expected_norm = expected_flat.norm()
    if actual_norm == 0 and expected_norm == 0:
        return 1.0
    if actual_norm == 0 or expected_norm == 0:
        return 0.0
    return float(
        torch.dot(actual_flat, expected_flat).item()
        / (actual_norm * expected_norm).item()
    )


def crop_model_margin(tensor: torch.Tensor | None, margin: int) -> torch.Tensor | None:
    if tensor is None or margin <= 0:
        return tensor
    if (
        tensor.shape[-3] <= 2 * margin
        or tensor.shape[-2] <= 2 * margin
        or tensor.shape[-1] <= 2 * margin
    ):
        return tensor
    return tensor[..., margin:-margin, margin:-margin, margin:-margin]


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
    source_y = (
        torch.linspace(
            1,
            max(1, ny - 2),
            steps=sources,
            device=device,
            dtype=torch.float32,
        )
        .round()
        .to(torch.long)
    )
    receiver_y = (
        torch.linspace(
            1,
            max(1, ny - 2),
            steps=receivers,
            device=device,
            dtype=torch.float32,
        )
        .round()
        .to(torch.long)
    )

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


def build_case(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)

    if args.heterogeneous:
        z = torch.linspace(-1.0, 1.0, args.nz, device=device, dtype=dtype).view(
            args.nz, 1, 1
        )
        y = torch.linspace(-1.0, 1.0, args.ny, device=device, dtype=dtype).view(
            1, args.ny, 1
        )
        x = torch.linspace(-1.0, 1.0, args.nx, device=device, dtype=dtype).view(
            1, 1, args.nx
        )
        epsilon = args.epsilon * (1.0 + 0.08 * torch.sin(2.3 * x + 1.7 * y))
        epsilon = epsilon + 0.04 * torch.cos(1.1 * z - 0.9 * x)
        sigma = args.sigma * (1.0 + 0.15 * torch.cos(1.9 * y - 1.3 * z))
        sigma = sigma.clamp_min(0.0).expand(args.nz, args.ny, args.nx)
    else:
        epsilon = torch.full(
            (args.nz, args.ny, args.nx),
            args.epsilon,
            device=device,
            dtype=dtype,
        )
        sigma = torch.full_like(epsilon, args.sigma)
    mu = torch.ones_like(epsilon) * args.mu

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
    source_amplitude = (
        wavelet.view(1, 1, args.nt).expand(args.shots, args.sources, -1).contiguous()
    )
    max_vel = float((1.0 / torch.sqrt(epsilon * mu)).max().item()) * tide.utils.C0
    inner_dt, step_ratio = tide.cfl_condition(args.grid_spacing, args.dt, max_vel)
    return {
        "epsilon": epsilon.contiguous(),
        "sigma": sigma.contiguous(),
        "mu": mu.contiguous(),
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "max_vel": max_vel,
        "inner_dt": inner_dt,
        "step_ratio": int(step_ratio),
    }


def mode_settings(mode: str) -> tuple[str, str | bool]:
    if mode == "full":
        return "standard", False
    if mode == "bf16":
        return "standard", "bf16"
    if mode == "physical":
        return "standard", False
    if mode == "physical_bf16":
        return "standard", "bf16"
    if mode == "eonly":
        return "eonly_snapshot", False
    if mode == "eonly_bf16":
        return "eonly_snapshot", "bf16"
    if mode == "direct":
        return "direct_material_grad", False
    if mode == "direct_bf16":
        return "direct_material_grad", "bf16"
    if mode == "checkpoint":
        return "checkpoint_recompute", False
    if mode == "checkpoint_bf16":
        return "checkpoint_recompute", "bf16"
    if mode == "revolve":
        return "checkpoint_revolve", False
    if mode == "revolve_bf16":
        return "checkpoint_revolve", "bf16"
    raise ValueError(f"unsupported mode {mode!r}")


def run_training_step(
    args: argparse.Namespace,
    case: dict[str, Any],
    *,
    mode: str,
) -> dict[str, torch.Tensor | None]:
    execution_backend, storage_compression = mode_settings(mode)
    grad_epsilon = args.grad in {"epsilon", "both"}
    grad_sigma = args.grad in {"sigma", "both"}
    epsilon = case["epsilon"].detach().clone().requires_grad_(grad_epsilon)
    sigma = case["sigma"].detach().clone().requires_grad_(grad_sigma)
    mu = case["mu"].detach()

    env = {
        "TIDE_EM3D_EONLY_SNAPSHOT": None,
        "TIDE_EM3D_UNIFORM_COEFFS": None,
        "TIDE_EM3D_PHYSICAL_SNAPSHOT_STORAGE": (
            "1" if mode in {"physical", "physical_bf16"} else None
        ),
    }
    with env_override(env):
        receiver = tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=args.grid_spacing,
            dt=args.dt,
            source_amplitude=case["source_amplitude"],
            source_location=case["source_location"],
            receiver_location=case["receiver_location"],
            stencil=args.stencil,
            pml_width=args.pml,
            max_vel=case["max_vel"],
            model_gradient_sampling_interval=args.gradient_sampling_interval,
            source_component=args.source_component,
            receiver_component=args.receiver_component,
            execution_backend=execution_backend,
            python_backend=False,
            storage_mode="device",
            storage_compression=storage_compression,
            storage_chunk_steps=args.storage_chunk_steps,
            n_threads=args.n_threads,
        )[-1]
        loss = receiver.square().sum()
        loss.backward()

    return {
        "receiver": receiver.detach(),
        "epsilon_grad": None if epsilon.grad is None else epsilon.grad.detach(),
        "sigma_grad": None if sigma.grad is None else sigma.grad.detach(),
    }


def benchmark_mode(
    args: argparse.Namespace,
    case: dict[str, Any],
    *,
    mode: str,
    reference: dict[str, torch.Tensor | None] | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor | None]]:
    for _ in range(args.warmup):
        run_training_step(args, case, mode=mode)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    times_s: list[float] = []
    last_result: dict[str, torch.Tensor | None] | None = None
    for _ in range(args.iters):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        last_result = run_training_step(args, case, mode=mode)
        torch.cuda.synchronize(device)
        times_s.append(time.perf_counter() - started)

    assert last_result is not None
    mean_s = statistics.fmean(times_s)
    padded_nz, padded_ny, padded_nx = padded_shape(
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        pml=args.pml,
        stencil=args.stencil,
    )
    nt_internal = args.nt * case["step_ratio"]
    cell_steps = args.shots * padded_nz * padded_ny * padded_nx * nt_internal
    execution_backend, storage_compression = mode_settings(mode)
    eonly = execution_backend in {"eonly_snapshot", "direct_material_grad"}
    checkpoint = execution_backend in {"checkpoint_recompute", "checkpoint_revolve"}
    revolve = execution_backend == "checkpoint_revolve"
    physical_snapshots = mode in {"physical", "physical_bf16"}
    n_stored = (nt_internal + args.gradient_sampling_interval - 1) // max(
        1, args.gradient_sampling_interval
    )
    bytes_per_elem = 2 if storage_compression == "bf16" else 4
    stored_nz = args.nz if physical_snapshots else padded_nz
    stored_ny = args.ny if physical_snapshots else padded_ny
    stored_nx = args.nx if physical_snapshots else padded_nx
    if checkpoint:
        segment_steps = args.storage_chunk_steps
        if segment_steps <= 0:
            segment_steps = max(1, round((6 * nt_internal) ** 0.5))
        segment_steps = min(segment_steps, nt_internal)
        n_segments = (nt_internal + segment_steps - 1) // segment_steps
        checkpoint_slots = (
            1 if n_segments <= 1 else int(math.ceil(math.log2(n_segments))) + 1
        )
        checkpoint_components = 18 * (checkpoint_slots if revolve else n_segments)
        segment_components = 3 * segment_steps + 3
        checkpoint_bytes_est = (
            args.shots * stored_nz * stored_ny * stored_nx * checkpoint_components * 4
        )
        segment_bytes_est = (
            args.shots
            * stored_nz
            * stored_ny
            * stored_nx
            * segment_components
            * bytes_per_elem
        )
        stored_components = checkpoint_components + segment_components
        snapshot_bytes_est = checkpoint_bytes_est + segment_bytes_est
    else:
        stored_components = 3 * n_stored + 3 if eonly else 6 * n_stored
        checkpoint_components = 0
        segment_components = 0
        checkpoint_bytes_est = 0
        segment_bytes_est = snapshot_bytes_est = (
            args.shots
            * stored_nz
            * stored_ny
            * stored_nx
            * stored_components
            * bytes_per_elem
        )

    result = {
        "mode": mode,
        "execution_backend": execution_backend,
        "eonly_requested": eonly,
        "physical_snapshot_storage_requested": physical_snapshots,
        "storage_compression": storage_compression,
        "measurement": {
            "warmup": args.warmup,
            "iters": args.iters,
            "times_s": times_s,
            "mean_s": mean_s,
            **quantiles(times_s),
            "cell_steps": cell_steps,
            "cell_steps_per_s": cell_steps / mean_s,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "receiver_norm": float(last_result["receiver"].float().norm().item())
            if last_result["receiver"] is not None
            else 0.0,
        },
        "snapshot_storage_estimate": {
            "padded_shape": [padded_nz, padded_ny, padded_nx],
            "stored_shape_per_shot": [stored_nz, stored_ny, stored_nx],
            "num_steps_stored": n_stored,
            "stored_components": stored_components,
            "checkpoint_components": checkpoint_components,
            "segment_components": segment_components,
            "bytes_per_element": bytes_per_elem,
            "bytes": snapshot_bytes_est,
            "checkpoint_bytes": checkpoint_bytes_est,
            "segment_bytes": segment_bytes_est,
        },
    }
    if reference is not None:
        epsilon_grad_inner = crop_model_margin(last_result["epsilon_grad"], args.pml)
        epsilon_ref_inner = crop_model_margin(reference["epsilon_grad"], args.pml)
        sigma_grad_inner = crop_model_margin(last_result["sigma_grad"], args.pml)
        sigma_ref_inner = crop_model_margin(reference["sigma_grad"], args.pml)
        result["error_vs_full"] = {
            "receiver_rel_l2": rel_l2(last_result["receiver"], reference["receiver"]),
            "receiver_max_abs": max_abs(last_result["receiver"], reference["receiver"]),
            "receiver_cosine_similarity": cosine_similarity(
                last_result["receiver"], reference["receiver"]
            ),
            "epsilon_grad_rel_l2": rel_l2(
                last_result["epsilon_grad"], reference["epsilon_grad"]
            ),
            "epsilon_grad_max_abs": max_abs(
                last_result["epsilon_grad"], reference["epsilon_grad"]
            ),
            "epsilon_grad_cosine_similarity": cosine_similarity(
                last_result["epsilon_grad"], reference["epsilon_grad"]
            ),
            "sigma_grad_rel_l2": rel_l2(
                last_result["sigma_grad"], reference["sigma_grad"]
            ),
            "sigma_grad_max_abs": max_abs(
                last_result["sigma_grad"], reference["sigma_grad"]
            ),
            "sigma_grad_cosine_similarity": cosine_similarity(
                last_result["sigma_grad"], reference["sigma_grad"]
            ),
            "epsilon_grad_inner_rel_l2": rel_l2(epsilon_grad_inner, epsilon_ref_inner),
            "epsilon_grad_inner_max_abs": max_abs(
                epsilon_grad_inner, epsilon_ref_inner
            ),
            "epsilon_grad_inner_cosine_similarity": cosine_similarity(
                epsilon_grad_inner, epsilon_ref_inner
            ),
            "sigma_grad_inner_rel_l2": rel_l2(sigma_grad_inner, sigma_ref_inner),
            "sigma_grad_inner_max_abs": max_abs(sigma_grad_inner, sigma_ref_inner),
            "sigma_grad_inner_cosine_similarity": cosine_similarity(
                sigma_grad_inner, sigma_ref_inner
            ),
        }
    return result, last_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile tide.maxwell3d native CUDA gradient storage modes."
    )
    parser.add_argument("--nz", type=spatial_int, default=48)
    parser.add_argument("--ny", type=spatial_int, default=48)
    parser.add_argument("--nx", type=spatial_int, default=48)
    parser.add_argument("--nt", type=positive_int, default=120)
    parser.add_argument("--shots", type=positive_int, default=1)
    parser.add_argument("--sources", type=positive_int, default=1)
    parser.add_argument("--receivers", type=positive_int, default=16)
    parser.add_argument("--stencil", type=int, choices=(2, 4, 6, 8), default=2)
    parser.add_argument("--pml", type=nonnegative_int, default=8)
    parser.add_argument("--dtype", choices=("float32",), default="float32")
    parser.add_argument("--grid-spacing", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=4e-11)
    parser.add_argument("--freq", type=float, default=80e6)
    parser.add_argument("--peak-time", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=2e-4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--homogeneous", dest="heterogeneous", action="store_false")
    parser.set_defaults(heterogeneous=True)
    parser.add_argument("--model-batched", action="store_true")
    parser.add_argument("--grad", choices=("epsilon", "sigma", "both"), default="both")
    parser.add_argument("--gradient-sampling-interval", type=positive_int, default=1)
    parser.add_argument("--storage-chunk-steps", type=nonnegative_int, default=0)
    parser.add_argument("--n-threads", type=int, default=256)
    parser.add_argument("--source-component", choices=("ex", "ey", "ez"), default="ey")
    parser.add_argument(
        "--receiver-component", choices=("ex", "ey", "ez"), default="ey"
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=2)
    parser.add_argument("--iters", type=positive_int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "full",
            "bf16",
            "physical",
            "physical_bf16",
            "eonly",
            "eonly_bf16",
            "direct",
            "direct_bf16",
            "checkpoint",
            "checkpoint_bf16",
            "revolve",
            "revolve_bf16",
        ),
        default=(
            "full",
            "bf16",
            "eonly",
            "eonly_bf16",
            "checkpoint",
            "checkpoint_bf16",
            "revolve",
            "revolve_bf16",
            "direct",
            "direct_bf16",
        ),
    )
    parser.add_argument("--device", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if not backend_utils.is_backend_available():
        raise SystemExit("tide native backend is not available")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    case = build_case(args, device)

    reference: dict[str, torch.Tensor | None] | None = None
    results = []
    for mode in args.modes:
        mode_result, last_result = benchmark_mode(
            args,
            case,
            mode=mode,
            reference=reference,
            device=device,
        )
        if mode == "full":
            reference = {
                key: None if value is None else value.detach().clone()
                for key, value in last_result.items()
            }
            mode_result["error_vs_full"] = {
                "receiver_rel_l2": 0.0,
                "receiver_max_abs": 0.0,
                "epsilon_grad_rel_l2": 0.0,
                "epsilon_grad_max_abs": 0.0,
                "sigma_grad_rel_l2": 0.0,
                "sigma_grad_max_abs": 0.0,
            }
        results.append(mode_result)

    padded_nz, padded_ny, padded_nx = padded_shape(
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        pml=args.pml,
        stencil=args.stencil,
    )
    return {
        "operator": "tide.maxwell3d native CUDA gradient training step",
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
            "padded_shape_estimate": [padded_nz, padded_ny, padded_nx],
            "model_batched": args.model_batched,
            "heterogeneous": args.heterogeneous,
            "shots": args.shots,
            "sources_per_shot": args.sources,
            "receivers_per_shot": args.receivers,
            "nt": args.nt,
            "nt_internal": args.nt * case["step_ratio"],
            "step_ratio": case["step_ratio"],
            "gradient_sampling_interval": args.gradient_sampling_interval,
            "storage_chunk_steps": args.storage_chunk_steps,
            "stencil": args.stencil,
            "pml_width_each_side": args.pml,
            "grid_spacing": args.grid_spacing,
            "dt_requested": args.dt,
            "dt_internal": case["inner_dt"],
            "grad": args.grad,
            "n_threads_arg": args.n_threads,
        },
        "modes": results,
    }


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
