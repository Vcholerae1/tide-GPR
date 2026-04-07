from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment


def phi_q_from_sqdist(r2: np.ndarray, q: float) -> np.ndarray:
    if not (1.0 < q < 3.0):
        raise ValueError("q must satisfy 1 < q < 3")
    return np.log1p(((q - 1.0) / (3.0 - q)) * r2) / (q - 1.0)


def q_gsot_cost_matrix(
    t: np.ndarray,
    d_mod: np.ndarray,
    d_obs: np.ndarray,
    q: float,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    d_mod = np.asarray(d_mod, dtype=float)
    d_obs = np.asarray(d_obs, dtype=float)

    if not (len(t) == len(d_mod) == len(d_obs)):
        raise ValueError("t, d_mod, d_obs must have the same length")

    dt2 = (t[:, None] - t[None, :]) ** 2
    da2 = (d_mod[:, None] - d_obs[None, :]) ** 2
    return phi_q_from_sqdist(dt2, q) + phi_q_from_sqdist(da2, q)


def q_gsot_trace_pot(
    t: np.ndarray,
    d_mod: np.ndarray,
    d_obs: np.ndarray,
    q: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if ot is None:
        raise RuntimeError(
            "POT is not installed. Install package 'POT' to enable ot.emd."
        )

    C = q_gsot_cost_matrix(t, d_mod, d_obs, q)
    n_samples = len(t)
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    G = ot.emd(a, b, C)
    sigma = G.argmax(axis=1)
    loss = n_samples * np.sum(G * C)
    return loss, sigma, G, C


def q_gsot_trace_hungarian(
    t: np.ndarray,
    d_mod: np.ndarray,
    d_obs: np.ndarray,
    q: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    C = q_gsot_cost_matrix(t, d_mod, d_obs, q)
    row_ind, col_ind = linear_sum_assignment(C)
    loss = C[row_ind, col_ind].sum()
    return loss, col_ind, C


def mse_loss(d_mod: np.ndarray, d_obs: np.ndarray) -> float:
    return float(np.mean((d_mod - d_obs) ** 2))


def l1_loss(d_mod: np.ndarray, d_obs: np.ndarray) -> float:
    return float(np.mean(np.abs(d_mod - d_obs)))


def ncc_loss(d_mod: np.ndarray, d_obs: np.ndarray, eps: float = 1e-12) -> float:
    numerator = float(np.dot(d_mod, d_obs))
    denominator = float(np.linalg.norm(d_mod) * np.linalg.norm(d_obs) + eps)
    return 1.0 - numerator / denominator


def ricker(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    x = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * x**2) * np.exp(-(x**2))


def shift_trace(t: np.ndarray, trace: np.ndarray, shift_seconds: float) -> np.ndarray:
    return np.interp(t - shift_seconds, t, trace, left=0.0, right=0.0)


def normalize_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def summarize_curve(name: str, shifts: np.ndarray, values: np.ndarray) -> str:
    index = int(np.argmin(values))
    best_shift_ns = shifts[index] * 1e9
    best_value = values[index]
    return f"{name:<18} min={best_value:>12.6g} at shift={best_shift_ns:>8.3f} ns"


def build_objective_curves(
    t: np.ndarray,
    d_obs: np.ndarray,
    shifts: np.ndarray,
    q_values: list[float],
) -> tuple[dict[str, np.ndarray], dict[float, np.ndarray]]:
    objective_curves: dict[str, np.ndarray] = {
        "MSE": np.zeros_like(shifts),
        "L1": np.zeros_like(shifts),
        "1-NCC": np.zeros_like(shifts),
    }
    modeled_by_q: dict[float, np.ndarray] = {}
    for q in q_values:
        objective_curves[f"q-GSOT (q={q:g})"] = np.zeros_like(shifts)
        if ot is not None:
            objective_curves[f"q-GSOT POT (q={q:g})"] = np.zeros_like(shifts)

    for idx, shift in enumerate(shifts):
        d_mod = shift_trace(t, d_obs, shift)
        amplitude_scale = 1.0 + 0.10 * np.sin(shift / (t[1] - t[0]) * 0.25)
        d_mod = amplitude_scale * d_mod
        modeled_by_q[idx] = d_mod

        objective_curves["MSE"][idx] = mse_loss(d_mod, d_obs)
        objective_curves["L1"][idx] = l1_loss(d_mod, d_obs)
        objective_curves["1-NCC"][idx] = ncc_loss(d_mod, d_obs)

        for q in q_values:
            loss_h, _, _ = q_gsot_trace_hungarian(t, d_mod, d_obs, q)
            objective_curves[f"q-GSOT (q={q:g})"][idx] = loss_h
            if ot is not None:
                loss_pot, _, _, _ = q_gsot_trace_pot(t, d_mod, d_obs, q)
                objective_curves[f"q-GSOT POT (q={q:g})"][idx] = loss_pot

    return objective_curves, modeled_by_q


def plot_trace_examples(
    ax: plt.Axes,
    t: np.ndarray,
    d_obs: np.ndarray,
    sample_traces: list[tuple[str, np.ndarray]],
) -> None:
    ax.plot(t * 1e9, d_obs, linewidth=2.5, label="Observed", color="black")
    for label, trace in sample_traces:
        ax.plot(t * 1e9, trace, linewidth=1.8, label=label)
    ax.set_title("Observed vs shifted modeled traces")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_objective_curves(
    ax: plt.Axes,
    shifts: np.ndarray,
    curves: dict[str, np.ndarray],
    normalize: bool,
) -> None:
    for name, values in curves.items():
        y = normalize_curve(values) if normalize else values
        ax.plot(shifts * 1e9, y, linewidth=2, label=name)
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Normalized objective curves" if normalize else "Raw objective curves")
    ax.set_xlabel("Time shift applied to modeled trace (ns)")
    ax.set_ylabel("Normalized value" if normalize else "Objective value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_cost_matrix(
    output_dir: Path,
    t: np.ndarray,
    d_obs: np.ndarray,
    d_mod: np.ndarray,
    q: float,
) -> None:
    loss, sigma, C = q_gsot_trace_hungarian(t, d_mod, d_obs, q)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    im = axes[0].imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[t[0] * 1e9, t[-1] * 1e9, t[0] * 1e9, t[-1] * 1e9],
        cmap="magma",
    )
    axes[0].plot(t[sigma] * 1e9, t * 1e9, color="cyan", linewidth=1.5)
    axes[0].set_title(f"q-GSOT cost matrix and assignment (q={q:g})")
    axes[0].set_xlabel("Observed sample time (ns)")
    axes[0].set_ylabel("Modeled sample time (ns)")
    fig.colorbar(im, ax=axes[0], label="Cost")

    axes[1].plot(t * 1e9, d_obs, label="Observed", color="black", linewidth=2.2)
    axes[1].plot(t * 1e9, d_mod, label="Modeled", linewidth=1.8)
    axes[1].set_title(f"Trace pair used for assignment, loss={loss:.4f}")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure_path = output_dir / f"qgsot_cost_matrix_q{str(q).replace('.', 'p')}.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"Saved q-GSOT cost matrix figure to {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MSE/L1/NCC/q-GSOT objectives on a simple shifted trace example."
    )
    parser.add_argument("--nt", type=int, default=161, help="Number of time samples.")
    parser.add_argument(
        "--dt", type=float, default=2.5e-10, help="Time interval in seconds."
    )
    parser.add_argument(
        "--freq", type=float, default=80e6, help="Ricker central frequency in Hz."
    )
    parser.add_argument(
        "--q-values",
        type=float,
        nargs="+",
        default=[1.2, 1.5, 2.0],
        help="List of q values for q-GSOT.",
    )
    parser.add_argument(
        "--max-shift-samples",
        type=int,
        default=18,
        help="Maximum left/right shift in samples for the objective scan.",
    )
    parser.add_argument(
        "--num-shifts",
        type=int,
        default=37,
        help="Number of shift values in the scan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/q_gsot_objectives"),
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(args.nt, dtype=float) * args.dt
    t0 = t[len(t) // 3]
    d_obs = ricker(t, f0=args.freq, t0=t0)
    shifts = np.linspace(
        -args.max_shift_samples * args.dt,
        args.max_shift_samples * args.dt,
        args.num_shifts,
    )

    curves, modeled_traces = build_objective_curves(t, d_obs, shifts, args.q_values)

    summary_lines = ["Objective minima over shift scan:"]
    for name, values in curves.items():
        summary_lines.append(summarize_curve(name, shifts, values))
    print("\n".join(summary_lines))
    if ot is None:
        print("POT package not found, skipped exact ot.emd comparison.")

    sample_indices = [0, len(shifts) // 2, len(shifts) - 1]
    sample_traces = [
        (f"Shift {shifts[index] * 1e9:+.2f} ns", modeled_traces[index])
        for index in sample_indices
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)
    plot_trace_examples(axes[0], t, d_obs, sample_traces)
    plot_objective_curves(axes[1], shifts, curves, normalize=False)
    plot_objective_curves(axes[2], shifts, curves, normalize=True)

    figure_path = args.output_dir / "objective_comparison.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"Saved objective comparison figure to {figure_path}")

    representative_index = min(len(shifts) - 1, len(shifts) // 2 + 7)
    representative_trace = modeled_traces[representative_index]
    plot_cost_matrix(
        args.output_dir, t, d_obs, representative_trace, q=args.q_values[0]
    )


if __name__ == "__main__":
    main()
