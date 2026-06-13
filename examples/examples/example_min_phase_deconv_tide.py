"""Minimum-phase spiking deconvolution on Tide-modeled GPR data.

This example follows the standard convolutional-model assumption used by
Wiener/spiking deconvolution: a trace is approximated by reflectivity convolved
with a source wavelet, and a short inverse filter is estimated from trace
autocorrelation to compress a minimum-phase wavelet toward a spike.

References:
  - Yao, Margrave, and Gallant, CREWES Research Report 11, 1999.
    https://crewes.org/Documents/ResearchReports/1999/1999-14.pdf
  - Pranowo, J. Petrol Explor. Prod. Technol. 9, 2583-2590, 2019.
    https://doi.org/10.1007/s13202-019-00748-9
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tide  # noqa: E402
from mixed_phase_deconv import (  # noqa: E402
    convolution,
    deconvolveSpikingInvFilter,
    mean_normalized_wavelet,
    traceScaling_rms,
)


@dataclass(frozen=True)
class DeconvolutionResult:
    scaled_input: np.ndarray
    deconvolved: np.ndarray
    inverse_filters: np.ndarray
    estimated_wavelets: np.ndarray
    window_index: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small Tide forward model and minimum-phase spiking deconvolution."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/min_phase_deconv_tide"))
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda.")
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    parser.add_argument("--nt", type=int, default=700)
    parser.add_argument("--freq", type=float, default=160e6)
    parser.add_argument("--dt", type=float, default=4.0e-11)
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--nf", type=int, default=80, help="Inverse filter length in samples.")
    parser.add_argument("--wtr", type=int, default=2, help="Neighbouring traces per side for supertraces.")
    parser.add_argument("--mu", type=float, default=0.1, help="Pre-whitening fraction.")
    parser.add_argument(
        "--window-ns",
        type=float,
        nargs=2,
        default=(4.0, 22.0),
        metavar=("START", "END"),
        help="Time window used for wavelet estimation, in ns.",
    )
    return parser.parse_args()


def berlage_wavelet(
    *,
    freq: float,
    nt: int,
    dt: float,
    alpha: float = 5.0e8,
    power: int = 2,
    phase: float = -0.5 * np.pi,
) -> np.ndarray:
    """Causal Berlage-style minimum-phase source wavelet."""
    t = np.arange(nt, dtype=float) * dt
    wavelet = (t**power) * np.exp(-alpha * t) * np.cos(2.0 * np.pi * freq * t + phase)
    wavelet[0] = 0.0
    return (wavelet / np.max(np.abs(wavelet))).astype(np.float32)


def make_sparse_epsilon(nz: int, nx: int) -> np.ndarray:
    epsilon = np.full((nz, nx), 4.0, dtype=np.float32)
    scatterers = [
        (22, 18, 1.2, 2, 3),
        (27, 34, -0.7, 3, 4),
        (32, 51, 1.2, 4, 2),
        (37, 70, -0.7, 2, 3),
        (43, 84, 1.2, 3, 4),
        (49, 25, -0.7, 4, 2),
        (54, 45, 1.2, 2, 3),
        (59, 63, -0.7, 3, 4),
        (64, 79, 1.2, 4, 2),
        (68, 12, -0.7, 2, 3),
        (72, 37, 1.2, 3, 4),
        (76, 58, -0.7, 4, 2),
        (56, 88, 1.2, 2, 3),
    ]
    for z0, x0, amp, radius_z, radius_x in scatterers:
        z_slice = slice(max(0, z0 - radius_z), min(nz, z0 + radius_z + 1))
        x_slice = slice(max(0, x0 - radius_x), min(nx, x0 + radius_x + 1))
        epsilon[z_slice, x_slice] = 4.0 + amp

    x = np.arange(nx, dtype=float)
    interface = np.rint(46.0 + 4.0 * np.sin(2.0 * np.pi * x / nx * 1.5)).astype(int)
    for ix, iz in enumerate(interface):
        epsilon[iz : min(nz, iz + 2), ix] = 5.0
    return epsilon


def build_common_offset_geometry(
    *,
    nx: int,
    n_shots: int,
    source_depth: int,
    receiver_offset: int,
    pml_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = pml_width + 4
    right = nx - pml_width - receiver_offset - 5
    source_x = torch.linspace(left, right, n_shots, device=device).round().long()

    source_locations = torch.zeros((n_shots, 1, 2), dtype=torch.long, device=device)
    source_locations[:, 0, 0] = source_depth
    source_locations[:, 0, 1] = source_x

    receiver_locations = torch.zeros((n_shots, 1, 2), dtype=torch.long, device=device)
    receiver_locations[:, 0, 0] = source_depth
    receiver_locations[:, 0, 1] = source_x + receiver_offset
    return source_locations, receiver_locations


def model_gather(
    *,
    epsilon_np: np.ndarray,
    wavelet_np: np.ndarray,
    dx: float,
    dt: float,
    pml_width: int,
    device: torch.device,
    python_backend: bool,
) -> np.ndarray:
    epsilon = torch.tensor(epsilon_np, device=device, dtype=torch.float32)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    nt = int(wavelet_np.size)

    source_locations, receiver_locations = build_common_offset_geometry(
        nx=epsilon_np.shape[1],
        n_shots=25,
        source_depth=pml_width + 1,
        receiver_offset=4,
        pml_width=pml_width,
        device=device,
    )
    source_amplitude = (
        torch.tensor(wavelet_np, device=device, dtype=torch.float32)
        .view(1, 1, nt)
        .expand(source_locations.shape[0], 1, nt)
        .contiguous()
    )

    with torch.no_grad():
        receiver_data = tide.maxwelltm(
            epsilon=epsilon,
            sigma=sigma,
            mu=mu,
            grid_spacing=dx,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=pml_width,
            save_snapshots=False,
            python_backend=python_backend,
            storage_mode="none",
        )[-1]
    return receiver_data.detach().cpu().numpy()[:, :, 0]


def minimum_phase_deconvolution(
    data: np.ndarray,
    time_ns: np.ndarray,
    window_ns: tuple[float, float],
    *,
    wtr: int,
    nf: int,
    mu: float,
) -> DeconvolutionResult:
    X = traceScaling_rms(data)
    win = np.sort(np.asarray(window_ns, dtype=float))
    window_index = np.where((time_ns > win[0]) & (time_ns < win[1]))[0]
    if window_index.size == 0:
        raise ValueError("deconvolution window selects no samples")
    nf = min(int(nf), int(window_index.size))

    ns, ntr = X.shape
    deconvolved = np.empty((ns, ntr), dtype=float)
    inverse_filters = np.empty((nf, ntr), dtype=float)

    for j in range(ntr):
        neighbours = np.arange(j - wtr, j + wtr + 1)
        neighbours = neighbours[(neighbours >= 0) & (neighbours < ntr)]
        supertrace = X[np.ix_(window_index, neighbours)].reshape(-1, order="F")
        filt = deconvolveSpikingInvFilter(
            supertrace,
            n=nf,
            i=1,
            mu=mu,
            taperType="hamming",
        )["fmin"]
        inverse_filters[:, j] = filt
        deconvolved[:, j] = convolution(X[:, j], filt)[:, 0]

    estimated_wavelets = np.fft.ifft(
        1.0 / np.fft.fft(inverse_filters, axis=0),
        axis=0,
    ).real
    return DeconvolutionResult(
        scaled_input=X,
        deconvolved=deconvolved,
        inverse_filters=inverse_filters,
        estimated_wavelets=estimated_wavelets,
        window_index=window_index,
    )


def excess_kurtosis(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=float).reshape(-1)
    std = float(np.std(v))
    if std == 0.0:
        return float("nan")
    z = (v - np.mean(v)) / std
    return float(np.mean(z**4) - 3.0)


def spectral_centroid_mhz(x: np.ndarray, dt: float) -> float:
    traces = np.asarray(x, dtype=float)
    spectrum = np.fft.rfft(traces, axis=0)
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    denom = float(np.sum(power))
    if denom == 0.0:
        return float("nan")
    freq = np.fft.rfftfreq(traces.shape[0], dt)
    return float(np.sum(freq * power) / denom / 1.0e6)


def save_model_plot(epsilon: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(epsilon, cmap="viridis", aspect="auto")
    ax.set_title("Sparse synthetic relative permittivity")
    ax.set_xlabel("x grid index")
    ax.set_ylabel("z grid index")
    fig.colorbar(im, ax=ax, label="epsilon_r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_wavelet_plot(
    *,
    time_ns: np.ndarray,
    ricker_reference: np.ndarray,
    source_wavelet: np.ndarray,
    output_path: Path,
) -> None:
    cumulative = np.cumsum(source_wavelet**2) / np.sum(source_wavelet**2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(
        time_ns,
        ricker_reference / np.max(np.abs(ricker_reference)),
        label="zero-phase Ricker reference",
        alpha=0.75,
    )
    axes[0].plot(time_ns, source_wavelet, label="causal minimum-phase source")
    axes[0].set_xlim(0, 18)
    axes[0].set_xlabel("time (ns)")
    axes[0].set_ylabel("normalized amplitude")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_ns, cumulative, color="tab:green")
    axes[1].set_xlim(0, 18)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("time (ns)")
    axes[1].set_ylabel("cumulative source energy")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Source wavelet used by Tide")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_gather_plot(
    *,
    before: np.ndarray,
    after: np.ndarray,
    time_ns: np.ndarray,
    window_index: np.ndarray,
    output_path: Path,
) -> None:
    def display_normalize(image: np.ndarray) -> np.ndarray:
        scale = max(float(np.percentile(np.abs(image), 99.0)), 1.0e-12)
        return image / scale

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True, sharey=True)
    extent = [0, before.shape[1] - 1, time_ns[-1], time_ns[0]]
    for ax, image, title in zip(axes, (before, after), ("Before deconvolution", "After deconvolution"), strict=True):
        im = ax.imshow(
            display_normalize(image),
            cmap="seismic",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            vmin=-1.0,
            vmax=1.0,
        )
        ax.axhspan(time_ns[window_index[-1]], time_ns[window_index[0]], color="black", alpha=0.08)
        ax.set_title(title)
        ax.set_xlabel("trace")
        ax.set_ylabel("time (ns)")
    y0 = max(0.0, float(time_ns[window_index[0]]) - 1.0)
    y1 = min(float(time_ns[-1]), float(time_ns[window_index[-1]]) + 1.0)
    axes[0].set_ylim(y1, y0)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="panel-normalized amplitude")
    fig.suptitle("Common-offset reflection gather")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_estimated_wavelet_plot(
    *,
    estimated_wavelets: np.ndarray,
    dt: float,
    output_path: Path,
) -> None:
    mean_wavelet = mean_normalized_wavelet(estimated_wavelets)
    t_ns = np.arange(mean_wavelet.size, dtype=float) * dt * 1e9
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t_ns, mean_wavelet, color="tab:purple", lw=1.8)
    ax.set_title("Mean estimated minimum-phase wavelet")
    ax.set_xlabel("lag (ns)")
    ax.set_ylabel("normalized amplitude")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    python_backend = args.backend == "python"

    nz, nx = 80, 96
    pml_width = 8
    epsilon_true = make_sparse_epsilon(nz, nx)
    epsilon_background = np.full_like(epsilon_true, 4.0)

    time_axis_ns = np.arange(args.nt, dtype=float) * args.dt * 1e9
    ricker_reference = tide.ricker(
        args.freq,
        args.nt,
        args.dt,
        peak_time=1.5 / args.freq,
        dtype=torch.float64,
    ).numpy()
    source_wavelet = berlage_wavelet(freq=args.freq, nt=args.nt, dt=args.dt)

    t0 = time.time()
    observed = model_gather(
        epsilon_np=epsilon_true,
        wavelet_np=source_wavelet,
        dx=args.dx,
        dt=args.dt,
        pml_width=pml_width,
        device=device,
        python_backend=python_backend,
    )
    background = model_gather(
        epsilon_np=epsilon_background,
        wavelet_np=source_wavelet,
        dx=args.dx,
        dt=args.dt,
        pml_width=pml_width,
        device=device,
        python_backend=python_backend,
    )
    scattered = observed - background

    result = minimum_phase_deconvolution(
        scattered,
        time_axis_ns,
        tuple(args.window_ns),
        wtr=args.wtr,
        nf=args.nf,
        mu=args.mu,
    )

    save_model_plot(epsilon_true, args.output_dir / "model.png")
    save_wavelet_plot(
        time_ns=time_axis_ns,
        ricker_reference=ricker_reference,
        source_wavelet=source_wavelet,
        output_path=args.output_dir / "source_wavelet.png",
    )
    save_gather_plot(
        before=result.scaled_input,
        after=result.deconvolved,
        time_ns=time_axis_ns,
        window_index=result.window_index,
        output_path=args.output_dir / "deconvolution_gather.png",
    )
    save_estimated_wavelet_plot(
        estimated_wavelets=result.estimated_wavelets,
        dt=args.dt,
        output_path=args.output_dir / "estimated_wavelet.png",
    )

    before_window = result.scaled_input[result.window_index, :]
    after_window = result.deconvolved[result.window_index, :]
    print(f"Tide forward + deconvolution finished in {time.time() - t0:.2f}s")
    print(f"Output directory: {args.output_dir}")
    print(f"Data shape: nt={scattered.shape[0]}, traces={scattered.shape[1]}")
    print(f"Deconvolution window: {args.window_ns[0]:.2f}-{args.window_ns[1]:.2f} ns")
    print(f"Inverse filter length: {result.inverse_filters.shape[0]} samples")
    print(
        "Window excess kurtosis: "
        f"before={excess_kurtosis(before_window):.3f}, "
        f"after={excess_kurtosis(after_window):.3f}"
    )
    print(
        "Window spectral centroid: "
        f"before={spectral_centroid_mhz(before_window, args.dt):.1f} MHz, "
        f"after={spectral_centroid_mhz(after_window, args.dt):.1f} MHz"
    )


if __name__ == "__main__":
    main()
