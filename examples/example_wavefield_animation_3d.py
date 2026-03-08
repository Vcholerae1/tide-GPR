"""3D Maxwell forward modeling and slice animation example.

This script runs a single-shot 3D simulation with ``tide.maxwell3d`` and saves:
1) an animated GIF of three orthogonal Ey slices (xy/xz/yz), and
2) a receiver gather image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import tide
from tide import CallbackState


def _resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {name!r}")


def _build_model(
    nz: int,
    ny: int,
    nx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype, device=device)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    zz = torch.arange(nz, dtype=dtype, device=device).view(nz, 1, 1)
    yy = torch.arange(ny, dtype=dtype, device=device).view(1, ny, 1)
    xx = torch.arange(nx, dtype=dtype, device=device).view(1, 1, nx)

    center_z = int(nz * 0.65)
    center_y = ny // 2
    center_x = nx // 2
    radius = max(3, min(nz, ny, nx) // 6)
    sphere_mask = (
        (zz - center_z) ** 2 + (yy - center_y) ** 2 + (xx - center_x) ** 2
    ) <= radius**2
    epsilon = torch.where(
        sphere_mask,
        torch.tensor(9.0, dtype=dtype, device=device),
        epsilon,
    )

    bottom = int(nz * 0.7)
    sigma[bottom:, :, :] = 1e-3
    return epsilon, sigma, mu


def _make_receivers(
    nz: int,
    ny: int,
    nx: int,
    n_receivers: int,
    device: torch.device,
) -> torch.Tensor:
    rz = max(2, int(nz * 0.15))
    ry = ny // 2
    rx = torch.linspace(
        4,
        nx - 5,
        n_receivers,
        device=device,
        dtype=torch.long,
    )
    receivers = torch.zeros(1, n_receivers, 3, dtype=torch.long, device=device)
    receivers[0, :, 0] = rz
    receivers[0, :, 1] = ry
    receivers[0, :, 2] = rx
    return receivers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D wavefield visualization example.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--nz", type=int, default=36)
    parser.add_argument("--ny", type=int, default=36)
    parser.add_argument("--nx", type=int, default=36)
    parser.add_argument("--nt", type=int, default=1800)
    parser.add_argument("--dz", type=float, default=0.01)
    parser.add_argument("--dy", type=float, default=0.01)
    parser.add_argument("--dx", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=1.5e-11)
    parser.add_argument("--freq", type=float, default=250e6)
    parser.add_argument("--pml-width", type=int, default=8)
    parser.add_argument("--stencil", type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument("--snapshot-interval", type=int, default=2)
    parser.add_argument("--n-receivers", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        default="examples/outputs/maxwell3d_viz",
        help="Directory for GIF/PNG outputs.",
    )
    parser.add_argument("--gif-name", default="maxwell3d_ey_slices.gif")
    parser.add_argument("--gather-name", default="maxwell3d_receivers.png")
    parser.add_argument(
        "--source-component",
        default="ey",
        choices=["ex", "ey", "ez"],
    )
    parser.add_argument(
        "--receiver-component",
        default="ey",
        choices=["ex", "ey", "ez"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.snapshot_interval <= 0:
        raise ValueError("snapshot-interval must be positive.")

    device = _resolve_device(args.device)
    dtype = torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(
        f"Grid: nz={args.nz}, ny={args.ny}, nx={args.nx}, "
        f"nt={args.nt}, dt={args.dt:.3e}s"
    )

    epsilon, sigma, mu = _build_model(args.nz, args.ny, args.nx, device, dtype)

    src_z = max(2, int(args.nz * 0.15))
    src_y = args.ny // 2
    src_x = args.nx // 2
    source_location = torch.tensor(
        [[[src_z, src_y, src_x]]],
        dtype=torch.long,
        device=device,
    )
    receiver_location = _make_receivers(
        args.nz, args.ny, args.nx, args.n_receivers, device
    )

    wavelet = tide.ricker(
        args.freq,
        args.nt,
        args.dt,
        peak_time=1.2 / args.freq,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, args.nt)

    z_mid = args.nz // 2
    y_mid = args.ny // 2
    x_mid = args.nx // 2

    xy_snapshots: list[np.ndarray] = []
    xz_snapshots: list[np.ndarray] = []
    yz_snapshots: list[np.ndarray] = []
    times_ns: list[float] = []

    def save_snapshot(state: CallbackState) -> None:
        ey = state.get_wavefield("Ey", view="inner")[0]
        xy_snapshots.append(ey[z_mid, :, :].detach().cpu().numpy())
        xz_snapshots.append(ey[:, y_mid, :].detach().cpu().numpy())
        yz_snapshots.append(ey[:, :, x_mid].detach().cpu().numpy())
        times_ns.append(state.time * 1e9)

    t0 = time.perf_counter()
    out = tide.maxwell3d(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=[args.dz, args.dy, args.dx],
        dt=args.dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=args.pml_width,
        stencil=args.stencil,
        source_component=args.source_component,
        receiver_component=args.receiver_component,
        python_backend=True,
        forward_callback=save_snapshot,
        callback_frequency=args.snapshot_interval,
    )
    sim_time = time.perf_counter() - t0
    receiver_amplitudes = out[-1].detach().cpu().numpy()[:, 0, :]

    if not xy_snapshots:
        raise RuntimeError("No snapshots were captured. Check snapshot_interval/nt.")

    print(
        f"Simulation complete in {sim_time:.2f}s. "
        f"Saved {len(xy_snapshots)} snapshots."
    )

    xy = np.stack(xy_snapshots, axis=0)
    xz = np.stack(xz_snapshots, axis=0)
    yz = np.stack(yz_snapshots, axis=0)
    vmax = max(np.abs(xy).max(), np.abs(xz).max(), np.abs(yz).max())
    vmax = max(vmax, 1e-12)
    vlim = 0.3 * vmax

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_xy = axes[0, 0]
    ax_xz = axes[0, 1]
    ax_yz = axes[1, 0]
    ax_txt = axes[1, 1]
    ax_txt.axis("off")

    im_xy = ax_xy.imshow(
        xy[0],
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        extent=[0, args.nx * args.dx, args.ny * args.dy, 0],
        animated=True,
    )
    im_xz = ax_xz.imshow(
        xz[0],
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        extent=[0, args.nx * args.dx, args.nz * args.dz, 0],
        animated=True,
    )
    im_yz = ax_yz.imshow(
        yz[0],
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        extent=[0, args.ny * args.dy, args.nz * args.dz, 0],
        animated=True,
    )

    ax_xy.set_title("Ey slice (xy, z = mid)")
    ax_xz.set_title("Ey slice (xz, y = mid)")
    ax_yz.set_title("Ey slice (yz, x = mid)")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xz.set_xlabel("x (m)")
    ax_xz.set_ylabel("z (m)")
    ax_yz.set_xlabel("y (m)")
    ax_yz.set_ylabel("z (m)")

    src_x_m = src_x * args.dx
    src_y_m = src_y * args.dy
    src_z_m = src_z * args.dz
    ax_xy.plot(src_x_m, src_y_m, "k*", ms=11, label="source")
    ax_xz.plot(src_x_m, src_z_m, "k*", ms=11)
    ax_yz.plot(src_y_m, src_z_m, "k*", ms=11)

    rx = receiver_location[0, :, 2].detach().cpu().numpy() * args.dx
    ry = receiver_location[0, :, 1].detach().cpu().numpy() * args.dy
    rz = receiver_location[0, :, 0].detach().cpu().numpy() * args.dz
    ax_xy.plot(rx, ry, "gv", ms=6, label="receivers")
    ax_xz.plot(rx, rz, "gv", ms=6)
    ax_yz.plot(ry, rz, "gv", ms=6)
    ax_xy.legend(loc="upper right")

    info = ax_txt.text(
        0.02,
        0.95,
        "",
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.colorbar(im_xy, ax=[ax_xy, ax_xz, ax_yz], fraction=0.03, pad=0.03, label="Ey")
    fig.subplots_adjust(wspace=0.28, hspace=0.26, right=0.90)

    def update(frame: int):
        im_xy.set_array(xy[frame])
        im_xz.set_array(xz[frame])
        im_yz.set_array(yz[frame])
        info.set_text(
            f"frame: {frame + 1}/{xy.shape[0]}\n"
            f"time: {times_ns[frame]:7.3f} ns\n"
            "wavefield: Ey\n"
            f"src/rcv: {args.source_component}/{args.receiver_component}\n"
            f"max |Ey|: {np.abs(xy[frame]).max():.3e}"
        )
        return [im_xy, im_xz, im_yz, info]

    gif_path = output_dir / args.gif_name
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=xy.shape[0],
        interval=45,
        blit=False,
    )
    ani.save(gif_path, writer="pillow", fps=18)
    plt.close(fig)
    print(f"Saved animation: {gif_path}")

    gather_fig, gather_ax = plt.subplots(figsize=(10, 5))
    gather_lim = max(np.abs(receiver_amplitudes).max(), 1e-12) * 0.6
    im = gather_ax.imshow(
        receiver_amplitudes,
        aspect="auto",
        cmap="seismic",
        vmin=-gather_lim,
        vmax=gather_lim,
        origin="upper",
    )
    gather_ax.set_title("Receiver gather")
    gather_ax.set_xlabel("Receiver index")
    gather_ax.set_ylabel("Time step")
    gather_fig.colorbar(im, ax=gather_ax, label=args.receiver_component)
    gather_fig.tight_layout()

    gather_path = output_dir / args.gather_name
    gather_fig.savefig(gather_path, dpi=150)
    plt.close(gather_fig)
    print(f"Saved receiver gather: {gather_path}")


if __name__ == "__main__":
    main()
