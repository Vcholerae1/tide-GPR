#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import tide


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


def _plus_mask(
    ny: int, nx: int, cy: int, cx: int, arm_len: int, half_width: int
) -> np.ndarray:
    yy, xx = np.ogrid[:ny, :nx]
    vertical = (np.abs(xx - cx) <= half_width) & (np.abs(yy - cy) <= arm_len)
    horizontal = (np.abs(yy - cy) <= half_width) & (np.abs(xx - cx) <= arm_len)
    return vertical | horizontal


def build_model(ny: int = 200, nx: int = 200) -> tuple[np.ndarray, np.ndarray]:
    eps = np.full((ny, nx), 3.0, dtype=np.float32)
    sigma = np.full((ny, nx), 1.0e-3, dtype=np.float32)
    mask_low = _plus_mask(ny, nx, cy=ny // 3, cx=nx // 3, arm_len=22, half_width=6)
    mask_high = _plus_mask(
        ny, nx, cy=(2 * ny) // 3, cx=(2 * nx) // 3, arm_len=22, half_width=6
    )
    eps[mask_low] = 1.0
    eps[mask_high] = 9.0
    return eps, sigma


def boundary_points(
    ny: int, nx: int, margin: int = 20, n_side: int = 100
) -> np.ndarray:
    xs = np.linspace(margin, nx - 1 - margin, n_side, dtype=np.int64)
    ys = np.linspace(margin, ny - 1 - margin, n_side, dtype=np.int64)
    top = np.stack([np.full_like(xs, margin), xs], axis=1)
    bottom = np.stack([np.full_like(xs, ny - 1 - margin), xs], axis=1)
    left = np.stack([ys[1:-1], np.full_like(ys[1:-1], margin)], axis=1)
    right = np.stack([ys[1:-1], np.full_like(ys[1:-1], nx - 1 - margin)], axis=1)
    return np.concatenate([top, right, bottom[::-1], left[::-1]], axis=0)


def make_ricker(freq: float, nt: int, dt: float, device: torch.device) -> torch.Tensor:
    t = torch.arange(nt, device=device) * dt
    t0 = 1.2 / freq
    w = np.pi * freq * (t - t0)
    return (1.0 - 2.0 * w**2) * torch.exp(-(w**2))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, fn):
    _sync(device)
    start = time.perf_counter()
    result = fn()
    _sync(device)
    return time.perf_counter() - start, result


def _grad_metrics(
    grad: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    g_tensor = grad.detach().float()
    r_tensor = ref.detach().float()
    if mask is not None:
        g_tensor = g_tensor[mask]
        r_tensor = r_tensor[mask]
    g = g_tensor.flatten()
    r = r_tensor.flatten()
    diff = g - r
    ref_norm = r.norm().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "cosine": float(F.cosine_similarity(g, r, dim=0).item()),
        "rel_l2": float((diff.norm() / ref_norm).item()),
        "max_abs_diff": float(diff.abs().max().item()),
        "norm": float(g.norm().item()),
    }


def _storage_proxy_bytes(
    *,
    mode: str,
    interval: int,
    nt: int,
    batch_size: int,
    ny: int,
    nx: int,
    pml_width: int,
    stencil: int,
    elem_bytes: int,
    sigma_grad: bool,
) -> int:
    fd_pad = stencil // 2
    padded_ny = ny + 2 * pml_width + fd_pad + (fd_pad - 1)
    padded_nx = nx + 2 * pml_width + fd_pad + (fd_pad - 1)
    stored_steps = (nt + interval - 1) // interval
    if mode in {"endpoint", "direct_material"}:
        stored_components = 1
        stored_steps += 1
    elif mode == "ecurl":
        stored_components = 2
    else:
        stored_components = 2 if sigma_grad else 1
    stored_ny = ny if mode == "physical" else padded_ny
    stored_nx = nx if mode == "physical" else padded_nx
    return stored_steps * batch_size * stored_ny * stored_nx * elem_bytes * stored_components


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TM2D gradient accuracy under standard Ey+curlH storage and "
            "E-only endpoint storage at different sampling intervals."
        )
    )
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--nt", type=int, default=320)
    parser.add_argument("--pml-width", type=int, default=12)
    parser.add_argument("--stencil", type=int, default=2)
    parser.add_argument("--n-side", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--freq", type=float, default=900e6)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.6e-11)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        default=[1, 2, 4, 5, 8, 10, 16, 20],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["standard", "endpoint"],
        choices=["standard", "physical", "endpoint", "direct_material", "ecurl"],
    )
    parser.add_argument(
        "--storage-compression",
        default="none",
        choices=["none", "bf16"],
    )
    parser.add_argument(
        "--sigma-grad",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include sigma gradients. Enabled by default.",
    )
    parser.add_argument(
        "--metric-crop-margin",
        type=int,
        default=0,
        help="Ignore this many model cells from every edge when comparing gradients.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    bad_intervals = [interval for interval in args.intervals if args.nt % interval != 0]
    if bad_intervals:
        raise ValueError(
            "All intervals must divide nt because the current native autograd "
            f"loop advances in whole sampling chunks. Bad intervals: {bad_intervals}."
        )

    if args.stencil != 2:
        raise NotImplementedError("This comparison script currently assumes stencil=2.")
    if args.metric_crop_margin < 0:
        raise ValueError("--metric-crop-margin must be >= 0.")
    if 2 * args.metric_crop_margin >= min(args.ny, args.nx):
        raise ValueError("--metric-crop-margin leaves no interior cells.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for endpoint/direct material backends.")
    device = torch.device(args.device)

    epsilon_true_np, sigma_true_np = build_model(ny=args.ny, nx=args.nx)
    epsilon_true = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
    sigma_true = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)
    mu_true = torch.ones_like(epsilon_true)

    ring = boundary_points(
        args.ny, args.nx, margin=args.pml_width, n_side=args.n_side
    )
    n_shots = int(ring.shape[0])
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = torch.from_numpy(ring[:, 0]).to(device)
    source_locations[:, 0, 1] = torch.from_numpy(ring[:, 1]).to(device)
    receiver_locations = (
        torch.from_numpy(ring)
        .to(device=device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(n_shots, 1, 1)
    )
    shot_batches = [
        torch.arange(n_shots, device=device)[i : i + args.batch_size]
        for i in range(0, n_shots, args.batch_size)
    ]

    wavelet = make_ricker(args.freq, args.nt, args.dt, device)
    source_amplitude = wavelet.view(1, 1, args.nt).repeat(n_shots, 1, 1)
    storage_compression = (
        False if args.storage_compression == "none" else args.storage_compression
    )

    def forward_shots(epsilon, sigma, mu, shot_indices, *, requires_grad: bool, mode: str, interval: int):
        kwargs = {}
        if mode == "endpoint":
            kwargs["execution_backend"] = "direct_material_endpoint_grad"
        elif mode == "direct_material":
            kwargs["execution_backend"] = "direct_material_grad"
        elif mode == "ecurl":
            kwargs["execution_backend"] = "direct_material_grad_ecurl"
        elif mode not in {"standard", "physical"}:
            raise ValueError(f"Unknown mode {mode!r}.")

        with env_override(
            {
                "TIDE_TM2D_PHYSICAL_SNAPSHOT_STORAGE": (
                    "1" if mode == "physical" and requires_grad else None
                )
            }
        ):
            *_, receivers = tide.maxwelltm(
                epsilon,
                sigma,
                mu,
                grid_spacing=args.dx,
                dt=args.dt,
                source_amplitude=source_amplitude[shot_indices],
                source_location=source_locations[shot_indices],
                receiver_location=receiver_locations[shot_indices],
                stencil=args.stencil,
                pml_width=args.pml_width,
                save_snapshots=requires_grad,
                model_gradient_sampling_interval=interval if requires_grad else 1,
                storage_mode="device",
                storage_compression=storage_compression,
                **kwargs,
            )
        return receivers

    def generate_observed() -> torch.Tensor:
        with torch.no_grad():
            return torch.cat(
                [
                    forward_shots(
                        epsilon_true,
                        sigma_true,
                        mu_true,
                        shot_indices,
                        requires_grad=False,
                        mode="standard",
                        interval=1,
                    )
                    for shot_indices in shot_batches
                ],
                dim=1,
            )

    observed_seconds, observed_data = _timed(device, generate_observed)

    epsilon_init_np = gaussian_filter(epsilon_true_np, sigma=12).copy()
    sigma_init_np = np.full_like(sigma_true_np, 1.0e-3, dtype=np.float32)
    epsilon_base = torch.tensor(epsilon_init_np, dtype=torch.float32, device=device)
    sigma_base = torch.tensor(sigma_init_np, dtype=torch.float32, device=device)
    mu_base = torch.ones_like(epsilon_base)
    global_loss_denom = max(1, int(observed_data.numel()))

    def gradient_eval(mode: str, interval: int):
        if mode == "direct_material" and interval != 1:
            raise ValueError("direct_material mode only supports interval=1.")
        epsilon_inv = epsilon_base.detach().clone().requires_grad_(True)
        sigma_inv = sigma_base.detach().clone().requires_grad_(args.sigma_grad)
        start_allocated = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_allocated = int(torch.cuda.memory_allocated(device))
        loss_accum = torch.zeros((), device=device)
        for shot_indices in shot_batches:
            syn = forward_shots(
                epsilon_inv,
                sigma_inv,
                mu_base,
                shot_indices,
                requires_grad=True,
                mode=mode,
                interval=interval,
            )
            obs = observed_data[:, shot_indices, :]
            loss = F.mse_loss(syn, obs, reduction="sum") / global_loss_denom
            loss.backward()
            loss_accum = loss_accum.detach() + loss.detach()

        peak_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        sigma_grad = (
            sigma_inv.grad.detach().cpu()
            if sigma_inv.grad is not None
            else torch.empty(0)
        )
        return {
            "loss": float(loss_accum.item()),
            "epsilon_grad": epsilon_inv.grad.detach().cpu(),
            "sigma_grad": sigma_grad,
            "peak_memory_bytes": peak_allocated,
            "peak_extra_bytes": max(0, peak_allocated - start_allocated),
        }

    for _ in range(args.warmup):
        gradient_eval("standard", 1)

    rows = []
    eval_seconds = {}
    ref_seconds, ref = _timed(device, lambda: gradient_eval("standard", 1))
    ref_eps = ref["epsilon_grad"]
    ref_sigma = ref["sigma_grad"]
    metric_mask = None
    if args.metric_crop_margin > 0:
        metric_mask = torch.zeros_like(ref_eps, dtype=torch.bool)
        m = args.metric_crop_margin
        metric_mask[m:-m, m:-m] = True
    eval_seconds[("standard", 1)] = [ref_seconds]

    requested = []
    for mode in args.modes:
        intervals = [1] if mode == "direct_material" else args.intervals
        for interval in intervals:
            item = (mode, interval)
            if item not in requested:
                requested.append(item)
    if ("standard", 1) not in requested:
        requested.insert(0, ("standard", 1))

    for mode, interval in requested:
        timings = []
        result = ref if (mode, interval) == ("standard", 1) else None
        if (mode, interval) == ("standard", 1):
            timings.extend(eval_seconds[(mode, interval)])
        for _ in range(args.iters if result is None else max(0, args.iters - 1)):
            elapsed, result = _timed(device, lambda m=mode, k=interval: gradient_eval(m, k))
            timings.append(elapsed)
        assert result is not None
        storage_bytes = _storage_proxy_bytes(
            mode=mode,
            interval=interval,
            nt=args.nt,
            batch_size=min(args.batch_size, n_shots),
            ny=args.ny,
            nx=args.nx,
            pml_width=args.pml_width,
            stencil=args.stencil,
            elem_bytes=2 if args.storage_compression == "bf16" else 4,
            sigma_grad=args.sigma_grad,
        )
        row = {
            "mode": mode,
            "interval": interval,
            "loss": result["loss"],
            "seconds": timings,
            "mean_seconds": float(np.mean(timings)),
            "min_seconds": float(np.min(timings)),
            "storage_proxy_bytes": int(storage_bytes),
            "storage_proxy_mib": storage_bytes / 2**20,
            "peak_memory_bytes": result["peak_memory_bytes"],
            "peak_extra_bytes": result["peak_extra_bytes"],
            "epsilon": _grad_metrics(result["epsilon_grad"], ref_eps, metric_mask),
        }
        if args.sigma_grad:
            row["sigma"] = _grad_metrics(result["sigma_grad"], ref_sigma, metric_mask)
        rows.append(row)

    payload = {
        "device": str(device),
        "ny": args.ny,
        "nx": args.nx,
        "nt": args.nt,
        "pml_width": args.pml_width,
        "stencil": args.stencil,
        "n_side": args.n_side,
        "n_shots": n_shots,
        "batch_size": args.batch_size,
        "num_batches": len(shot_batches),
        "storage_compression": args.storage_compression,
        "sigma_grad": args.sigma_grad,
        "metric_crop_margin": args.metric_crop_margin,
        "observed_seconds": observed_seconds,
        "reference": {"mode": "standard", "interval": 1},
        "rows": rows,
    }

    print("mode interval storage_MiB eps_cos eps_rel sigma_cos sigma_rel min_s")
    for row in rows:
        sigma = row.get("sigma")
        print(
            f"{row['mode']:15s} {row['interval']:8d} "
            f"{row['storage_proxy_mib']:11.3f} "
            f"{row['epsilon']['cosine']:7.5f} {row['epsilon']['rel_l2']:7.4f} "
            f"{sigma['cosine']:9.5f} {sigma['rel_l2']:8.4f} "
            f"{row['min_seconds']:7.3f}"
            if sigma is not None
            else (
                f"{row['mode']:15s} {row['interval']:8d} "
                f"{row['storage_proxy_mib']:11.3f} "
                f"{row['epsilon']['cosine']:7.5f} {row['epsilon']['rel_l2']:7.4f} "
                f"{row['min_seconds']:7.3f}"
            )
        )

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
