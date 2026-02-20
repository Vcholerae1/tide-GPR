"""Locate gradient anomalies between native backend and eager reference.

This script compares gradients w.r.t. epsilon and sigma at the same model state:
    - reference backend (default: eager)
    - test backend (default: c/native)

It reports:
1) Global mismatch statistics (relative error, cosine similarity, sign flips).
2) Top-K anomalous grid points for epsilon and sigma.
3) Optional finite-difference checks at top points to identify which backend is closer.

Run from repo root:
    PYTHONPATH=src .venv/bin/python examples/example_gradient_anomaly_locator.py --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

import tide

Tensor = torch.Tensor


@dataclass(frozen=True)
class ProblemConfig:
    ny: int = 24
    nx: int = 28
    nt: int = 180
    air_layer: int = 2
    n_shots: int = 3
    n_receivers: int = 8
    dx: float = 0.02
    dt: float = 2e-11
    pml_width: int = 6
    stencil: int = 2
    freq: float = 220e6
    max_vel: float = 3e8
    model_gradient_sampling_interval: int = 1
    seed: int = 1234


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_backend(name: str) -> bool | str:
    token = name.strip().lower()
    if token in ("c", "native", "false"):
        return False
    if token in ("eager", "jit"):
        return token
    raise ValueError(f"Unsupported backend '{name}'. Use one of: c, eager, jit.")


def backend_label(backend: bool | str) -> str:
    return "c" if backend is False else str(backend)


def build_problem(
    cfg: ProblemConfig, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(cfg.seed)

    y = torch.linspace(0.0, 1.0, cfg.ny, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, cfg.nx, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    epsilon_true = 4.0 + 1.0 * torch.exp(-((xx - 0.42) ** 2 + (yy - 0.58) ** 2) / 0.03)
    sigma_true = 4e-4 + 1.5e-3 * torch.exp(-((xx - 0.68) ** 2 + (yy - 0.40) ** 2) / 0.04)

    epsilon_init = torch.full_like(epsilon_true, 4.0)
    sigma_init = torch.full_like(sigma_true, 7e-4)

    epsilon_true[: cfg.air_layer, :] = 1.0
    sigma_true[: cfg.air_layer, :] = 0.0
    epsilon_init[: cfg.air_layer, :] = 1.0
    sigma_init[: cfg.air_layer, :] = 0.0

    mu = torch.ones_like(epsilon_true)

    source_location = torch.zeros(cfg.n_shots, 1, 2, dtype=torch.long, device=device)
    receiver_location = torch.zeros(
        cfg.n_shots, cfg.n_receivers, 2, dtype=torch.long, device=device
    )
    source_x = torch.linspace(4, cfg.nx - 8, cfg.n_shots, device=device).round().long()
    receiver_x = torch.linspace(
        2, cfg.nx - 3, cfg.n_receivers, device=device
    ).round().long()
    for i in range(cfg.n_shots):
        source_location[i, 0, 0] = cfg.air_layer
        source_location[i, 0, 1] = source_x[i]
        receiver_location[i, :, 0] = cfg.air_layer
        receiver_location[i, :, 1] = receiver_x

    wavelet = tide.ricker(
        cfg.freq, cfg.nt, cfg.dt, peak_time=1.0 / cfg.freq, dtype=dtype, device=device
    )
    source_amplitude = wavelet.view(1, 1, cfg.nt).repeat(cfg.n_shots, 1, 1)

    return (
        epsilon_true,
        sigma_true,
        epsilon_init,
        sigma_init,
        mu,
        source_location,
        receiver_location,
        source_amplitude,
    )


def make_forward(
    cfg: ProblemConfig,
    mu: Tensor,
    source_location: Tensor,
    receiver_location: Tensor,
    source_amplitude: Tensor,
    backend: bool | str,
) -> Callable[[Tensor, Tensor], Tensor]:
    def forward(epsilon: Tensor, sigma: Tensor) -> Tensor:
        return tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=cfg.dx,
            dt=cfg.dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=cfg.pml_width,
            stencil=cfg.stencil,
            max_vel=cfg.max_vel,
            model_gradient_sampling_interval=cfg.model_gradient_sampling_interval,
            python_backend=backend,
        )[-1]

    return forward


def make_objective(
    forward: Callable[[Tensor, Tensor], Tensor],
    epsilon_true: Tensor,
    sigma_true: Tensor,
) -> Callable[[Tensor, Tensor], Tensor]:
    with torch.no_grad():
        observed = forward(epsilon_true, sigma_true)

    def objective(epsilon: Tensor, sigma: Tensor) -> Tensor:
        residual = forward(epsilon, sigma) - observed
        return 0.5 * residual.square().mean()

    return objective


def compute_gradient(
    objective: Callable[[Tensor, Tensor], Tensor], epsilon0: Tensor, sigma0: Tensor
) -> tuple[Tensor, Tensor, float]:
    epsilon = epsilon0.clone().detach().requires_grad_(True)
    sigma = sigma0.clone().detach().requires_grad_(True)
    loss = objective(epsilon, sigma)
    loss.backward()
    assert epsilon.grad is not None
    assert sigma.grad is not None
    return epsilon.grad.detach().clone(), sigma.grad.detach().clone(), float(loss.item())


def masked_relative_error(
    g_ref: Tensor, g_test: Tensor, valid_mask: Tensor
) -> tuple[Tensor, float]:
    ref_abs = g_ref.abs()
    ref_valid = ref_abs[valid_mask]
    scale = torch.quantile(ref_valid, 0.95).item() if ref_valid.numel() > 0 else 1.0
    denom = ref_abs + 1e-3 * scale + 1e-12
    rel = (g_test - g_ref).abs() / denom
    rel = torch.where(valid_mask, rel, torch.zeros_like(rel))
    return rel, scale


def global_stats(name: str, g_ref: Tensor, g_test: Tensor, valid_mask: Tensor) -> None:
    r = g_ref[valid_mask].reshape(-1)
    t = g_test[valid_mask].reshape(-1)
    diff = t - r
    rel = diff.abs() / (r.abs() + 1e-12)
    cos = float((r @ t) / (torch.norm(r) * torch.norm(t) + 1e-12))
    sign_flip = ((r * t) < 0).float()
    flip_ratio = float(sign_flip.mean().item())
    print(
        f"{name}: cos={cos:.6f}  "
        f"mean_rel={float(rel.mean()):.4e}  "
        f"p95_rel={float(torch.quantile(rel, 0.95)):.4e}  "
        f"max_rel={float(rel.max()):.4e}  "
        f"sign_flip={flip_ratio:.2%}"
    )


def topk_indices(score: Tensor, k: int, valid_mask: Tensor) -> list[tuple[int, int]]:
    score_use = torch.where(valid_mask, score, torch.zeros_like(score))
    k = min(k, int(valid_mask.sum().item()))
    _, idx = torch.topk(score_use.reshape(-1), k=k)
    nx = score.shape[1]
    out: list[tuple[int, int]] = []
    for flat in idx.tolist():
        out.append((flat // nx, flat % nx))
    return out


def print_topk_table(
    name: str,
    points: list[tuple[int, int]],
    g_ref: Tensor,
    g_test: Tensor,
    rel_map: Tensor,
) -> None:
    print(f"\nTop anomalies for {name}:")
    print("  rank  (y,x)        g_ref            g_test           abs_diff         rel_score")
    for rank, (iy, ix) in enumerate(points, start=1):
        ref = float(g_ref[iy, ix].item())
        test = float(g_test[iy, ix].item())
        diff = abs(test - ref)
        rel = float(rel_map[iy, ix].item())
        print(
            f"  {rank:4d}  ({iy:2d},{ix:2d})   {ref:13.6e}   {test:13.6e}   "
            f"{diff:13.6e}   {rel:10.4e}"
        )


def fd_point_value(
    objective: Callable[[Tensor, Tensor], Tensor],
    epsilon0: Tensor,
    sigma0: Tensor,
    iy: int,
    ix: int,
    h: float,
    for_sigma: bool,
) -> float:
    if for_sigma:
        sp = sigma0.clone()
        sm = sigma0.clone()
        sp[iy, ix] += h
        sm[iy, ix] -= h
        return float(((objective(epsilon0, sp) - objective(epsilon0, sm)) / (2.0 * h)).item())
    ep = epsilon0.clone()
    em = epsilon0.clone()
    ep[iy, ix] += h
    em[iy, ix] -= h
    return float(((objective(ep, sigma0) - objective(em, sigma0)) / (2.0 * h)).item())


def run_fd_adjudication(
    objective: Callable[[Tensor, Tensor], Tensor],
    epsilon0: Tensor,
    sigma0: Tensor,
    points: list[tuple[int, int]],
    g_ref: Tensor,
    g_test: Tensor,
    h: float,
    name: str,
    for_sigma: bool,
) -> None:
    print(f"\nFD adjudication ({name}, h={h:.1e}):")
    print("  (y,x)        FD              err_ref          err_test         closer")
    for iy, ix in points:
        fd = fd_point_value(
            objective=objective,
            epsilon0=epsilon0,
            sigma0=sigma0,
            iy=iy,
            ix=ix,
            h=h,
            for_sigma=for_sigma,
        )
        ref = float(g_ref[iy, ix].item())
        test = float(g_test[iy, ix].item())
        err_ref = abs(fd - ref)
        err_test = abs(fd - test)
        closer = "ref" if err_ref <= err_test else "test"
        print(
            f"  ({iy:2d},{ix:2d})   {fd:13.6e}   {err_ref:13.6e}   "
            f"{err_test:13.6e}   {closer}"
        )


def save_maps(
    output_dir: Path,
    name: str,
    rel_map: Tensor,
    sign_flip_map: Tensor,
) -> None:
    rel_np = rel_map.detach().cpu().numpy()
    sign_np = sign_flip_map.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = np.nanpercentile(rel_np, 99)
    im = ax.imshow(rel_np, aspect="auto", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6))
    ax.set_title(f"{name} relative anomaly score")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, label="|g_test-g_ref| / (|g_ref|+eps)")
    plt.tight_layout()
    fig.savefig(output_dir / f"{name}_rel_score.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(sign_np, aspect="auto", cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_title(f"{name} sign flip mask (1=flip)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, label="flip")
    plt.tight_layout()
    fig.savefig(output_dir / f"{name}_sign_flip.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate gradient anomaly points.")
    parser.add_argument("--device", type=str, default="cuda", help="auto | cpu | cuda | mps")
    parser.add_argument("--dtype", type=str, default="float32", help="float32 | float64")
    parser.add_argument("--ref-backend", type=str, default="eager", help="eager | jit | c")
    parser.add_argument("--test-backend", type=str, default="c", help="eager | jit | c")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fd-top-k", type=int, default=8)
    parser.add_argument("--fd-h-eps", type=float, default=5e-3)
    parser.add_argument("--fd-h-sigma", type=float, default=1e-4)
    parser.add_argument("--outdir", type=str, default="outputs/gradient_anomaly_locator")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    ref_backend = parse_backend(args.ref_backend)
    test_backend = parse_backend(args.test_backend)

    cfg = ProblemConfig(seed=args.seed)
    (
        epsilon_true,
        sigma_true,
        epsilon0,
        sigma0,
        mu,
        source_location,
        receiver_location,
        source_amplitude,
    ) = build_problem(cfg, device=device, dtype=dtype)

    f_ref = make_forward(
        cfg=cfg,
        mu=mu,
        source_location=source_location,
        receiver_location=receiver_location,
        source_amplitude=source_amplitude,
        backend=ref_backend,
    )
    f_test = make_forward(
        cfg=cfg,
        mu=mu,
        source_location=source_location,
        receiver_location=receiver_location,
        source_amplitude=source_amplitude,
        backend=test_backend,
    )
    obj_ref = make_objective(f_ref, epsilon_true=epsilon_true, sigma_true=sigma_true)
    obj_test = make_objective(f_test, epsilon_true=epsilon_true, sigma_true=sigma_true)

    g_eps_ref, g_sig_ref, loss_ref = compute_gradient(obj_ref, epsilon0, sigma0)
    g_eps_test, g_sig_test, loss_test = compute_gradient(obj_test, epsilon0, sigma0)

    valid_mask = torch.ones_like(g_eps_ref, dtype=torch.bool)
    valid_mask[: cfg.air_layer, :] = False

    print("=" * 88)
    print(
        f"Device={device.type} dtype={dtype} "
        f"ref={backend_label(ref_backend)} test={backend_label(test_backend)}"
    )
    print(f"Loss(ref)={loss_ref:.6e}  Loss(test)={loss_test:.6e}")
    global_stats("epsilon", g_eps_ref, g_eps_test, valid_mask)
    global_stats("sigma", g_sig_ref, g_sig_test, valid_mask)

    rel_eps, _ = masked_relative_error(g_eps_ref, g_eps_test, valid_mask)
    rel_sig, _ = masked_relative_error(g_sig_ref, g_sig_test, valid_mask)

    flip_eps = ((g_eps_ref * g_eps_test) < 0) & valid_mask
    flip_sig = ((g_sig_ref * g_sig_test) < 0) & valid_mask

    top_eps = topk_indices(rel_eps, k=args.top_k, valid_mask=valid_mask)
    top_sig = topk_indices(rel_sig, k=args.top_k, valid_mask=valid_mask)
    print_topk_table("epsilon", top_eps, g_eps_ref, g_eps_test, rel_eps)
    print_topk_table("sigma", top_sig, g_sig_ref, g_sig_test, rel_sig)

    fd_k_eps = top_eps[: min(args.fd_top_k, len(top_eps))]
    fd_k_sig = top_sig[: min(args.fd_top_k, len(top_sig))]
    run_fd_adjudication(
        objective=obj_ref,
        epsilon0=epsilon0,
        sigma0=sigma0,
        points=fd_k_eps,
        g_ref=g_eps_ref,
        g_test=g_eps_test,
        h=args.fd_h_eps,
        name="epsilon",
        for_sigma=False,
    )
    run_fd_adjudication(
        objective=obj_ref,
        epsilon0=epsilon0,
        sigma0=sigma0,
        points=fd_k_sig,
        g_ref=g_sig_ref,
        g_test=g_sig_test,
        h=args.fd_h_sigma,
        name="sigma",
        for_sigma=True,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_maps(outdir, "epsilon", rel_eps, flip_eps.float())
    save_maps(outdir, "sigma", rel_sig, flip_sig.float())
    print(f"\nSaved anomaly maps to: {outdir}")


if __name__ == "__main__":
    main()
