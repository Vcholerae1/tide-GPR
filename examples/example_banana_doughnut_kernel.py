"""Minimal banana-doughnut style gradient visualization (tide version).

Run from repo root:
    PYTHONPATH=src .venv/bin/python examples/example_banana_doughnut_kernel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import tide


def run_backend(
    epsilon_base: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    source_amplitude: torch.Tensor,
    source_location: torch.Tensor,
    receiver_location: torch.Tensor,
    dx: float,
    dt: float,
    pml_width: int,
    i0: int,
    i1: int,
    python_backend: bool | str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    epsilon = epsilon_base.detach().clone().requires_grad_(True)
    out = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=pml_width,
        stencil=2,
        max_vel=3e8,
        model_gradient_sampling_interval=1,
        python_backend=python_backend,
    )[-1]
    trace = out[:, 0, 0]
    loss = (trace[i0:i1] ** 2).sum()
    loss.backward()
    assert epsilon.grad is not None
    return trace.detach(), epsilon.grad.detach(), float(loss.item())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ny, nx = 100, 100
    nt = 900
    dx = 0.02
    dt = 3e-11
    pml_width = 12

    epsilon = torch.full((ny, nx), 6.0, dtype=dtype, device=device)
    epsilon[50:, :] = 10.0
    sigma = torch.full((ny, nx), 2e-4, dtype=dtype, device=device)
    mu = torch.ones((ny, nx), dtype=dtype, device=device)

    source_location = torch.tensor([[[5, 10]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[5, 90]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        180e6, nt, dt, peak_time=1.0 / 180e6, dtype=dtype, device=device
    ).view(1, 1, nt)

    i0, i1 = 520, 720

    trace_c, grad_c, loss_c = run_backend(
        epsilon,
        sigma,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
        dx,
        dt,
        pml_width,
        i0,
        i1,
        python_backend=False,
    )
    trace_eager, grad_eager, loss_eager = run_backend(
        epsilon,
        sigma,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
        dx,
        dt,
        pml_width,
        i0,
        i1,
        python_backend="eager",
    )

    grad_diff = grad_c - grad_eager

    model_np = epsilon.detach().cpu().numpy()
    trace_c_np = trace_c.cpu().numpy()
    trace_eager_np = trace_eager.cpu().numpy()
    grad_c_np = grad_c.cpu().numpy()
    grad_eager_np = grad_eager.cpu().numpy()
    grad_diff_np = grad_diff.cpu().numpy()

    vlim_grad = float(
        np.percentile(np.abs(np.concatenate([grad_c_np.ravel(), grad_eager_np.ravel()])), 99)
    )
    vlim_grad = max(vlim_grad, 1e-12)
    vlim_diff = float(np.percentile(np.abs(grad_diff_np), 99))
    vlim_diff = max(vlim_diff, 1e-12)

    fig, ax = plt.subplots(1, 5, figsize=(18, 3.5))

    im0 = ax[0].imshow(model_np, aspect="auto", cmap="viridis")
    ax[0].set_title("Epsilon model (two-layer)")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].plot([10], [5], marker="*", color="yellow", markersize=10)
    ax[0].plot([90], [5], marker="v", color="lime", markersize=8)
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02)

    t_ns = np.arange(nt) * dt * 1e9
    ax[1].plot(t_ns, trace_c_np, linewidth=1.2, label="c")
    ax[1].plot(t_ns, trace_eager_np, linewidth=1.0, linestyle="--", label="eager")
    ax[1].axvspan(i0 * dt * 1e9, i1 * dt * 1e9, color="gray", alpha=0.2)
    ax[1].set_title("Receiver trace")
    ax[1].set_xlabel("Time (ns)")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid(alpha=0.25)
    ax[1].legend(loc="upper right", fontsize=8)

    im2 = ax[2].imshow(
        grad_c_np, aspect="auto", cmap="seismic", vmin=-vlim_grad, vmax=vlim_grad
    )
    ax[2].set_title("Gradient dJ/d epsilon (c)")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Y")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02)

    im3 = ax[3].imshow(
        grad_eager_np, aspect="auto", cmap="seismic", vmin=-vlim_grad, vmax=vlim_grad
    )
    ax[3].set_title("Gradient dJ/d epsilon (eager)")
    ax[3].set_xlabel("X")
    ax[3].set_ylabel("Y")
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.02)

    im4 = ax[4].imshow(
        grad_diff_np, aspect="auto", cmap="seismic", vmin=-vlim_diff, vmax=vlim_diff
    )
    ax[4].set_title("Gradient diff (c-eager)")
    ax[4].set_xlabel("X")
    ax[4].set_ylabel("Y")
    plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.02)

    plt.tight_layout()

    outdir = Path("outputs/banana_doughnut")
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "banana_doughnut_kernel_simple.png"
    fig.savefig(out_png, dpi=180)
    plt.show()

    rel_diff = float(torch.norm(grad_diff) / (torch.norm(grad_eager) + 1e-12))
    print(f"device={device.type} loss_c={loss_c:.6e} loss_eager={loss_eager:.6e}")
    print(f"grad_rel_diff_l2(c,eager)={rel_diff:.6e}")
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
