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


def plus_mask(ny: int, nx: int, cy: int, cx: int, arm_len: int, half_width: int):
    yy, xx = np.ogrid[:ny, :nx]
    vertical = (np.abs(xx - cx) <= half_width) & (np.abs(yy - cy) <= arm_len)
    horizontal = (np.abs(yy - cy) <= half_width) & (np.abs(xx - cx) <= arm_len)
    return vertical | horizontal


def build_model(ny: int, nx: int):
    eps = np.full((ny, nx), 3.0, dtype=np.float32)
    sigma = np.full((ny, nx), 1.0e-3, dtype=np.float32)
    eps[plus_mask(ny, nx, cy=ny // 3, cx=nx // 3, arm_len=22, half_width=6)] = 1.0
    eps[
        plus_mask(
            ny,
            nx,
            cy=(2 * ny) // 3,
            cx=(2 * nx) // 3,
            arm_len=22,
            half_width=6,
        )
    ] = 9.0
    return eps, sigma


def boundary_points(ny: int, nx: int, margin: int, n_side: int):
    xs = np.linspace(margin, nx - 1 - margin, n_side, dtype=np.int64)
    ys = np.linspace(margin, ny - 1 - margin, n_side, dtype=np.int64)
    top = np.stack([np.full_like(xs, margin), xs], axis=1)
    bottom = np.stack([np.full_like(xs, ny - 1 - margin), xs], axis=1)
    left = np.stack([ys[1:-1], np.full_like(ys[1:-1], margin)], axis=1)
    right = np.stack([ys[1:-1], np.full_like(ys[1:-1], nx - 1 - margin)], axis=1)
    return np.concatenate([top, right, bottom[::-1], left[::-1]], axis=0)


def ricker(freq: float, nt: int, dt: float, device: torch.device):
    t = torch.arange(nt, device=device) * dt
    t0 = 1.2 / freq
    w = np.pi * freq * (t - t0)
    return (1.0 - 2.0 * w**2) * torch.exp(-(w**2))


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metric_view(tensor: torch.Tensor, crop_margin: int) -> torch.Tensor:
    if crop_margin <= 0:
        return tensor
    return tensor[..., crop_margin:-crop_margin, crop_margin:-crop_margin]


def rms(a: torch.Tensor, b: torch.Tensor, crop_margin: int = 0) -> float:
    aa = _metric_view(a.detach().float(), crop_margin)
    bb = _metric_view(b.detach().float(), crop_margin)
    return float(torch.sqrt(torch.mean((aa - bb) ** 2)).item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Short TM2D cross inversion comparison for physical snapshot storage."
    )
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--n-side", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-interval", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--freq", type=float, default=900e6)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.6e-11)
    parser.add_argument("--storage-compression", choices=("none", "bf16"), default="none")
    parser.add_argument(
        "--metric-crop-margin",
        type=int,
        default=0,
        help="Ignore this many cells from each model edge for the cropped RMS metric.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.metric_crop_margin < 0:
        raise ValueError("--metric-crop-margin must be >= 0.")
    if 2 * args.metric_crop_margin >= min(args.ny, args.nx):
        raise ValueError("--metric-crop-margin leaves no interior cells.")
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

    eps_true_np, sigma_true_np = build_model(args.ny, args.nx)
    eps_true = torch.tensor(eps_true_np, dtype=torch.float32, device=device)
    sigma_fixed = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)
    mu = torch.ones_like(eps_true)
    eps_init = torch.tensor(
        gaussian_filter(eps_true_np, sigma=12).copy(),
        dtype=torch.float32,
        device=device,
    )

    ring = boundary_points(args.ny, args.nx, args.pml_width, args.n_side)
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
    batches = [
        torch.arange(n_shots, device=device)[i : i + args.batch_size]
        for i in range(0, n_shots, args.batch_size)
    ]
    wavelet = ricker(args.freq, args.nt, args.dt, device)
    source_amplitude = wavelet.view(1, 1, args.nt).repeat(n_shots, 1, 1)
    storage_compression = (
        False if args.storage_compression == "none" else args.storage_compression
    )

    def forward_shots(epsilon, shot_indices, requires_grad: bool, physical: bool):
        with env_override(
            {
                "TIDE_TM2D_PHYSICAL_SNAPSHOT_STORAGE": (
                    "1" if physical and requires_grad else None
                )
            }
        ):
            *_, rec = tide.maxwelltm(
                epsilon,
                sigma_fixed,
                mu,
                grid_spacing=args.dx,
                dt=args.dt,
                source_amplitude=source_amplitude[shot_indices],
                source_location=source_locations[shot_indices],
                receiver_location=receiver_locations[shot_indices],
                pml_width=args.pml_width,
                save_snapshots=requires_grad,
                model_gradient_sampling_interval=(
                    args.gradient_interval if requires_grad else 1
                ),
                storage_mode="device",
                storage_compression=storage_compression,
            )
        return rec

    with torch.no_grad():
        observed = torch.cat(
            [forward_shots(eps_true, shot_indices, False, False) for shot_indices in batches],
            dim=1,
        )

    def run_mode(physical: bool):
        eps = eps_init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([eps], lr=args.lr)
        history = []
        sync(device)
        started = time.perf_counter()
        for iteration in range(args.iterations):
            opt.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            for shot_indices in batches:
                pred = forward_shots(eps, shot_indices, True, physical)
                loss = F.mse_loss(pred, observed[:, shot_indices, :])
                loss.backward()
                loss_total = loss_total.detach() + loss.detach()
            opt.step()
            with torch.no_grad():
                eps.clamp_(1.0, 9.0)
            sync(device)
            history.append(
                {
                    "iteration": iteration + 1,
                    "loss_sum": float(loss_total.item()),
                    "rms_epsilon": rms(eps, eps_true),
                    "rms_epsilon_crop": rms(
                        eps, eps_true, crop_margin=args.metric_crop_margin
                    ),
                }
            )
        elapsed = time.perf_counter() - started
        return {
            "physical_snapshot_storage": physical,
            "seconds": elapsed,
            "initial_rms_epsilon": rms(eps_init, eps_true),
            "initial_rms_epsilon_crop": rms(
                eps_init, eps_true, crop_margin=args.metric_crop_margin
            ),
            "final_rms_epsilon": rms(eps, eps_true),
            "final_rms_epsilon_crop": rms(
                eps, eps_true, crop_margin=args.metric_crop_margin
            ),
            "history": history,
        }

    standard = run_mode(False)
    physical = run_mode(True)
    config = vars(args).copy()
    if config.get("json_out") is not None:
        config["json_out"] = str(config["json_out"])
    payload = {
        "config": config,
        "n_shots": n_shots,
        "num_batches": len(batches),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "standard": standard,
        "physical": physical,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
