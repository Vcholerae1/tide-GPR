"""Visualize numerical vs analytical traces in 3D Debye dispersive medium.

Analytical model follows the MATLAB reference scripts:
- ref/get_experiment_geometry.m
- ref/analytical_y.m (y-directed source via y<->z swap)

This script supports y- or z-directed dipole source comparison.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import tide
from tide import backend_utils


def _analytic_dispersive_dipole_z(
    wavelet: torch.Tensor,
    dt: float,
    x: float,
    y: float,
    z: float,
    epsr: float,
    delta: float,
    tau: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form E-field for z-directed dipole in Debye medium."""
    nt = int(wavelet.numel())
    wavelet = wavelet.to(torch.float64)
    n_fft = 1 << ((max(4 * nt, 512) - 1).bit_length())
    wavelet_padded = torch.zeros(n_fft, dtype=torch.float64)
    wavelet_padded[:nt] = wavelet

    eps0 = 8.854e-12
    mu0 = 4.0 * math.pi * 1e-7

    r = math.sqrt(x * x + y * y + z * z) + 1e-12
    theta = math.atan2(math.sqrt(x * x + y * y), z)
    phi = math.atan2(y, x)

    spectrum = torch.fft.rfft(wavelet_padded)
    freqs = torch.fft.rfftfreq(n_fft, d=dt)
    omega = 2.0 * math.pi * freqs

    efx = torch.zeros_like(spectrum, dtype=torch.complex128)
    efy = torch.zeros_like(spectrum, dtype=torch.complex128)
    efz = torch.zeros_like(spectrum, dtype=torch.complex128)

    idx = torch.nonzero(omega > 0.0, as_tuple=False).flatten()
    if idx.numel() == 0:
        zt = torch.zeros(nt, dtype=torch.float64)
        return zt.clone(), zt.clone(), zt.clone()

    om = omega[idx].to(torch.complex128)
    ep = epsr + delta / (1.0 + 1j * om * tau)
    k = torch.sqrt(om * om * eps0 * mu0 * (ep - 1j * sigma / (om * eps0)))
    eta = torch.sqrt(mu0 / (eps0 * (ep - 1j * sigma / (om * eps0))))

    er = (
        (eta / (2.0 * math.pi * r * r))
        * (1.0 + 1.0 / (1j * k * r))
        * math.cos(theta)
        * torch.exp(-1j * k * r)
    )
    etheta = (
        (1j * eta * k / (4.0 * math.pi * r))
        * (1.0 + 1.0 / (1j * k * r) - 1.0 / (k * r) ** 2)
        * math.sin(theta)
        * torch.exp(-1j * k * r)
    )

    ex = er * math.sin(theta) * math.cos(phi) + etheta * math.cos(theta) * math.cos(phi)
    ey = er * math.sin(theta) * math.sin(phi) + etheta * math.cos(theta) * math.sin(phi)
    ez = er * math.cos(theta) - etheta * math.sin(theta)

    efx[idx] = ex * spectrum[idx]
    efy[idx] = ey * spectrum[idx]
    efz[idx] = ez * spectrum[idx]

    tx = torch.fft.irfft(efx, n=n_fft).real[:nt]
    ty = torch.fft.irfft(efy, n=n_fft).real[:nt]
    tz = torch.fft.irfft(efz, n=n_fft).real[:nt]
    return tx, ty, tz


def _analytic_dispersive_dipole_y(
    wavelet: torch.Tensor,
    dt: float,
    x: float,
    y: float,
    z: float,
    epsr: float,
    delta: float,
    tau: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse z-directed closed form with y<->z axis swap."""
    x_swap = x
    y_swap = z
    z_swap = y

    ex_swap, ey_swap, ez_swap = _analytic_dispersive_dipole_z(
        wavelet=wavelet,
        dt=dt,
        x=x_swap,
        y=y_swap,
        z=z_swap,
        epsr=epsr,
        delta=delta,
        tau=tau,
        sigma=sigma,
    )

    ex = ex_swap
    ey = ez_swap
    ez = ey_swap
    return ex, ey, ez


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_backend(device: torch.device) -> bool:
    """Return python_backend flag.

    False means native C/CUDA backend, True means Python backend.
    """
    if device.type == "cuda" and backend_utils.is_backend_available():
        return False
    return True


def _component_for_mode(mode: str) -> tuple[str, str]:
    if mode == "y":
        return "ey", "ey"
    if mode == "z":
        return "ez", "ez"
    raise ValueError(f"Unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D dispersive numerical/analytical comparison with visualization."
    )
    parser.add_argument("--mode", default="y", choices=["y", "z"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "python", "native"]
    )

    parser.add_argument("--epsr", type=float, default=4.0)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=2e-10)
    parser.add_argument("--sigma", type=float, default=0.005)

    parser.add_argument("--freq", type=float, default=9e8)
    parser.add_argument("--dt", type=float, default=1e-11)
    parser.add_argument("--nt", type=int, default=360)
    parser.add_argument("--ds", type=float, default=0.005)

    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--pml", type=int, default=12)
    parser.add_argument("--stencil", type=int, default=4, choices=[2, 4, 6, 8])

    parser.add_argument(
        "--output",
        default="examples/outputs/maxwell3d_dispersion_analytic_compare.png",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    dtype = torch.float32

    if args.backend == "auto":
        python_backend = _default_backend(device)
    elif args.backend == "python":
        python_backend = True
    else:
        python_backend = False

    source_component, receiver_component = _component_for_mode(args.mode)

    src = (32, 32, 32)  # z, y, x
    rec = (63, 63, 63)  # z, y, x
    if not (0 <= src[0] < args.nz and 0 <= src[1] < args.ny and 0 <= src[2] < args.nx):
        raise ValueError("Source index is out of bounds for the provided grid size.")
    if not (0 <= rec[0] < args.nz and 0 <= rec[1] < args.ny and 0 <= rec[2] < args.nx):
        raise ValueError("Receiver index is out of bounds for the provided grid size.")

    epsilon = torch.full(
        (args.nz, args.ny, args.nx), args.epsr, device=device, dtype=dtype
    )
    conductivity = torch.full_like(epsilon, args.sigma)
    mu = torch.ones_like(epsilon)

    wavelet = tide.ricker(
        args.freq,
        args.nt,
        args.dt,
        peak_time=1.0 / args.freq,
        dtype=dtype,
        device=device,
    )

    out = tide.maxwell3d(
        epsilon=epsilon,
        sigma=conductivity,
        mu=mu,
        grid_spacing=[args.ds, args.ds, args.ds],
        dt=args.dt,
        source_amplitude=wavelet.view(1, 1, args.nt),
        source_location=torch.tensor([[list(src)]], dtype=torch.long, device=device),
        receiver_location=torch.tensor([[list(rec)]], dtype=torch.long, device=device),
        pml_width=args.pml,
        stencil=args.stencil,
        source_component=source_component,
        receiver_component=receiver_component,
        python_backend=python_backend,
        dispersion=tide.DebyeDispersion(delta_epsilon=args.delta, tau=args.tau),
    )
    simulated = out[-1][:, 0, 0].detach().cpu().to(torch.float64)

    dx = (rec[2] - src[2]) * args.ds
    dy = (rec[1] - src[1]) * args.ds
    dz = (rec[0] - src[0]) * args.ds

    wavelet_cpu = wavelet.detach().cpu().to(torch.float64)
    if args.mode == "y":
        _, analytic_main, _ = _analytic_dispersive_dipole_y(
            wavelet=wavelet_cpu,
            dt=args.dt,
            x=dx,
            y=dy,
            z=dz,
            epsr=args.epsr,
            delta=args.delta,
            tau=args.tau,
            sigma=args.sigma,
        )
    else:
        _, _, analytic_main = _analytic_dispersive_dipole_z(
            wavelet=wavelet_cpu,
            dt=args.dt,
            x=dx,
            y=dy,
            z=dz,
            epsr=args.epsr,
            delta=args.delta,
            tau=args.tau,
            sigma=args.sigma,
        )

    scale = torch.dot(simulated, analytic_main) / (
        torch.dot(analytic_main, analytic_main) + 1e-24
    )
    analytic_scaled = scale * analytic_main
    residual = simulated - analytic_scaled

    misfit = torch.linalg.norm(residual) / (torch.linalg.norm(analytic_main) + 1e-24)
    peak_shift = abs(
        int(simulated.abs().argmax().item()) - int(analytic_main.abs().argmax().item())
    )

    t_ns = torch.arange(args.nt, dtype=torch.float64) * args.dt * 1e9
    peak_idx = int(simulated.abs().argmax().item())
    i0 = max(0, peak_idx - 120)
    i1 = min(args.nt, peak_idx + 220)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)

    axes[0].plot(t_ns.numpy(), simulated.numpy(), lw=1.2, label="Numerical")
    axes[0].plot(
        t_ns.numpy(), analytic_scaled.numpy(), lw=1.2, label="Analytical (scaled)"
    )
    axes[0].set_title(
        "3D Debye Dispersive Trace Comparison "
        f"({source_component}->{receiver_component})"
    )
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        t_ns[i0:i1].numpy(), simulated[i0:i1].numpy(), lw=1.2, label="Numerical"
    )
    axes[1].plot(
        t_ns[i0:i1].numpy(),
        analytic_scaled[i0:i1].numpy(),
        lw=1.2,
        label="Analytical (scaled)",
    )
    axes[1].set_title("Zoom Around Main Arrival")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t_ns[i0:i1].numpy(), residual[i0:i1].numpy(), lw=1.1, color="tab:red")
    axes[2].set_title(
        f"Residual (misfit={float(misfit):.4f}, peak_shift={peak_shift}, scale={float(scale):.4f})"
    )
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_ylabel("Residual")
    axes[2].grid(True, alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)

    if args.show:
        plt.show()
    plt.close(fig)

    print(f"device={device}, python_backend={python_backend}")
    print(f"output={output_path}")
    print(
        f"misfit={float(misfit):.6f}, peak_shift={peak_shift}, scale={float(scale):.6f}"
    )


if __name__ == "__main__":
    main()
