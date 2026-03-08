"""Compare 3D numerical trace with analytic Green-function trace.

Default parameters are aligned with:
`tests/test_maxwell3d_analytic.py::test_maxwell3d_matches_uniform_medium_point_source_green_long_nt`.
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


def _analytic_trace_const_medium_point_source_3d(
    wavelet: torch.Tensor,
    dt: float,
    src_pos_m: tuple[float, float, float],
    rec_pos_m: tuple[float, float, float],
    eps_r: float,
    sigma: float,
    mu_r: float = 1.0,
    source_component: str = "ey",
    receiver_component: str = "ey",
) -> torch.Tensor:
    component_to_idx = {"ex": 0, "ey": 1, "ez": 2}
    src_i = component_to_idx[source_component]
    rec_i = component_to_idx[receiver_component]

    device = wavelet.device
    dtype = torch.float64
    nt = wavelet.numel()

    eps0 = 1.0 / (36.0 * math.pi) * 1e-9
    mu0 = 4.0 * math.pi * 1e-7

    r_zyx = (
        torch.tensor(rec_pos_m, device=device, dtype=dtype)
        - torch.tensor(src_pos_m, device=device, dtype=dtype)
    )
    r_xyz = torch.stack((r_zyx[2], r_zyx[1], r_zyx[0]))
    r_norm = torch.linalg.norm(r_xyz) + 1e-12
    r_hat = r_xyz / r_norm

    spectrum = torch.fft.rfft(wavelet.to(dtype))
    freqs = torch.fft.rfftfreq(nt, d=dt).to(device)
    omega = 2.0 * math.pi * freqs
    omega_c = omega.to(torch.complex128)
    omega_safe = omega_c.clone()
    if omega_safe.numel() > 1:
        omega_safe[0] = omega_safe[1]
    else:
        omega_safe[0] = 1.0 + 0.0j

    eps_complex = (
        eps0 * torch.tensor(eps_r, device=device, dtype=torch.complex128)
        - 1j * torch.tensor(sigma, device=device, dtype=torch.float64) / omega_safe
    )
    k = omega_safe * torch.sqrt(
        mu0 * torch.tensor(mu_r, device=device, dtype=torch.complex128) * eps_complex
    )
    green_scalar = torch.exp(-1j * k * r_norm) / (4.0 * math.pi * r_norm)

    kr = k * r_norm
    a_term = 1.0 - 1j / kr - 1.0 / (kr * kr)
    b_term = -1.0 + 3j / kr + 3.0 / (kr * kr)
    delta = 1.0 if src_i == rec_i else 0.0
    dyadic_component = a_term * delta + b_term * (r_hat[rec_i] * r_hat[src_i])
    transfer = 1j * omega_safe * mu0 * green_scalar * dyadic_component
    transfer[0] = 0.0 + 0.0j

    return torch.fft.irfft(spectrum * transfer, n=nt).real


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot numerical vs analytic trace for 3D Maxwell point source."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=8e-11)
    parser.add_argument("--freq", type=float, default=120e6)
    parser.add_argument("--nz", type=int, default=22)
    parser.add_argument("--ny", type=int, default=22)
    parser.add_argument("--nx", type=int, default=22)
    parser.add_argument("--spacing", type=float, default=0.02)
    parser.add_argument("--eps-r", type=float, default=9.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--mu-r", type=float, default=1.0)
    parser.add_argument("--pml-width", type=int, default=7)
    parser.add_argument("--stencil", type=int, default=4, choices=[2, 4, 6, 8])
    parser.add_argument("--source-component", default="ey", choices=["ex", "ey", "ez"])
    parser.add_argument("--receiver-component", default="ey", choices=["ex", "ey", "ez"])
    parser.add_argument(
        "--output",
        default="examples/outputs/maxwell3d_analytic_compare.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    dtype = torch.float32

    epsilon = torch.full((args.nz, args.ny, args.nx), args.eps_r, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, args.sigma)
    mu = torch.full_like(epsilon, args.mu_r)

    src_idx = (args.nz // 2, args.ny // 2, args.nx // 2)
    rec_idx = (src_idx[0] + 3, src_idx[1] + 2, src_idx[2] + 5)
    source_location = torch.tensor([[list(src_idx)]], device=device, dtype=torch.long)
    receiver_location = torch.tensor([[list(rec_idx)]], device=device, dtype=torch.long)

    wavelet = tide.ricker(
        args.freq,
        args.nt,
        args.dt,
        peak_time=1.2 / args.freq,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, args.nt)

    out = tide.maxwell3d(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=[args.spacing, args.spacing, args.spacing],
        dt=args.dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=args.pml_width,
        stencil=args.stencil,
        source_component=args.source_component,
        receiver_component=args.receiver_component,
        # python_backend=True,
    )
    simulated = out[-1][:, 0, 0].detach().cpu().to(torch.float64)

    analytic = _analytic_trace_const_medium_point_source_3d(
        wavelet=wavelet.detach().cpu(),
        dt=args.dt,
        src_pos_m=tuple(v * args.spacing for v in src_idx),
        rec_pos_m=tuple(v * args.spacing for v in rec_idx),
        eps_r=args.eps_r,
        sigma=args.sigma,
        mu_r=args.mu_r,
        source_component=args.source_component,
        receiver_component=args.receiver_component,
    )

    scale = torch.dot(simulated, analytic) / torch.dot(analytic, analytic)
    analytic_scaled = scale * analytic
    residual = simulated - analytic_scaled

    misfit = torch.linalg.norm(residual) / torch.linalg.norm(analytic)
    peak_shift = abs(
        int(simulated.abs().argmax().item()) - int(analytic.abs().argmax().item())
    )

    t_ns = torch.arange(args.nt, dtype=torch.float64) * args.dt * 1e9
    peak_idx = int(simulated.abs().argmax().item())
    i0 = max(0, peak_idx - 120)
    i1 = min(args.nt, peak_idx + 220)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(t_ns.numpy(), simulated.numpy(), label="Numerical", lw=1.2)
    axes[0].plot(t_ns.numpy(), analytic_scaled.numpy(), label="Analytic (scaled)", lw=1.2)
    axes[0].set_title(
        "3D Trace Comparison "
        f"({args.source_component}->{args.receiver_component})\n"
        f"misfit={float(misfit):.4f}, peak_shift={peak_shift}, scale={float(scale):.4f}"
    )
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_ns[i0:i1].numpy(), simulated[i0:i1].numpy(), label="Numerical", lw=1.2)
    axes[1].plot(
        t_ns[i0:i1].numpy(),
        analytic_scaled[i0:i1].numpy(),
        label="Analytic (scaled)",
        lw=1.2,
    )
    axes[1].plot(t_ns[i0:i1].numpy(), residual[i0:i1].numpy(), label="Residual", lw=1.0)
    axes[1].set_title("Zoomed Around Main Arrival")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Device: {device}")
    print(f"Saved figure: {output_path}")
    print(f"misfit={float(misfit):.6f}, peak_shift={peak_shift}, scale={float(scale):.6f}")


if __name__ == "__main__":
    main()
