"""Native versus aggressive-half2 inversion of the GPRImagingPy lab data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

import tide


DATA_DIR = Path("references/GPRImagingPy/01Lab/02FullWaveformInversion")


def gaussian_wavelet(nt: int, dt: float, frequency: float) -> np.ndarray:
    t = np.arange(nt) * dt
    coefficients = np.array([0.3532222, -0.488, 0.145, -0.010222222])
    duration = 1.14 / frequency
    wavelet = sum(
        coefficient * np.cos(2 * order * np.pi * t / duration)
        for order, coefficient in enumerate(coefficients)
    )
    wavelet[t >= duration] = 0.0
    return wavelet.astype(np.float32)


def load_problem(device: torch.device) -> dict[str, torch.Tensor]:
    nt, n_shots = 700, 200
    observed_source = torch.from_numpy(
        np.load(DATA_DIR / "LabData.npy").astype(np.float32)
    )[None, None]
    observed = functional.interpolate(
        observed_source,
        size=(nt, n_shots),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0, 0].numpy()
    initial = np.load(DATA_DIR / "lab_init.npy").astype(np.float32)
    initial[:2] = 1.0
    source_x = torch.arange(n_shots, device=device) * 2
    locations = torch.stack((torch.zeros_like(source_x), source_x), dim=-1)
    locations = locations[:, None, :].contiguous()
    wavelet = torch.from_numpy(gaussian_wavelet(nt, 2.0e-11, 2.0e8)).to(device)
    return {
        "observed": torch.from_numpy(observed).to(device)[:, :, None],
        "initial": torch.from_numpy(initial).to(device),
        "sigma": torch.zeros((100, 400), device=device),
        "mu": torch.ones((100, 400), device=device),
        "source_amplitude": wavelet[None, None, :]
        .expand(n_shots, 1, nt)
        .contiguous(),
        "source_location": locations,
        "receiver_location": locations.clone(),
    }


def trace_normalize(data: torch.Tensor) -> torch.Tensor:
    centered = data - data.mean(dim=0, keepdim=True)
    rms = centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-8)
    return centered / rms


def epsilon_from_parameter(parameter: torch.Tensor) -> torch.Tensor:
    epsilon = 1.0 + 8.0 * torch.sigmoid(parameter)
    fixed_air = torch.ones_like(epsilon)
    mask = torch.ones_like(epsilon)
    mask[:2] = 0.0
    return epsilon * mask + fixed_air * (1.0 - mask)


def invert(
    problem: dict[str, torch.Tensor],
    *,
    mode: str,
    aggressive: bool,
    iterations: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    os.environ["TIDE_TM_FP16_HALF2"] = "1" if aggressive else "0"
    if aggressive:
        os.environ["TIDE_TM_FP16_HALF2_ARITH"] = "1"
    else:
        os.environ.pop("TIDE_TM_FP16_HALF2_ARITH", None)
    initial = problem["initial"]
    fraction = ((initial - 1.0) / 8.0).clamp(1e-5, 1.0 - 1e-5)
    parameter = torch.nn.Parameter(torch.logit(fraction))
    optimizer = torch.optim.Adam([parameter], lr=learning_rate)
    observed = trace_normalize(problem["observed"])
    losses: list[float] = []
    iteration_seconds: list[float] = []
    peak_mib = 0.0

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        data_loss_value = 0.0
        for first in range(0, 200, batch_size):
            last = min(first + batch_size, 200)
            epsilon = epsilon_from_parameter(parameter)
            predicted = tide.maxwelltm(
                epsilon,
                problem["sigma"],
                problem["mu"],
                0.01,
                2.0e-11,
                source_amplitude=problem["source_amplitude"][first:last],
                source_location=problem["source_location"][first:last],
                receiver_location=problem["receiver_location"][first:last],
                stencil=4,
                pml_width=10,
                model_gradient_sampling_interval=5,
                storage_mode="device",
                storage_compression="bf16",
                compute_mode=mode,
            )[-1]
            residual = trace_normalize(predicted) - observed[:, first:last]
            batch_loss = 0.5 * residual.square().sum() / observed.numel()
            batch_loss.backward()
            data_loss_value += float(batch_loss.detach())
        epsilon = epsilon_from_parameter(parameter)
        smoothness = (
            (epsilon[1:] - epsilon[:-1]).square().mean()
            + (epsilon[:, 1:] - epsilon[:, :-1]).square().mean()
        )
        regularization = 2.0e-4 * smoothness
        regularization.backward()
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        loss_value = data_loss_value + float(regularization.detach())
        losses.append(loss_value)
        iteration_seconds.append(elapsed)
        peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 2**20)
        print(
            f"{mode}{'+half2arith' if aggressive else ''} "
            f"iteration={iteration + 1:03d} loss={loss_value:.7e} "
            f"seconds={elapsed:.4f}",
            flush=True,
        )

    return {
        "epsilon": epsilon_from_parameter(parameter).detach().cpu().numpy(),
        "losses": losses,
        "iteration_seconds": iteration_seconds,
        "total_seconds": float(sum(iteration_seconds)),
        "median_iteration_seconds": float(np.median(iteration_seconds[1:])),
        "peak_mib": peak_mib,
    }


def make_figure(
    initial: np.ndarray,
    native: dict,
    half2: dict,
    output: Path,
    *,
    half2_title: str = "Aggressive half2",
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
        }
    )
    fig = plt.figure(figsize=(10.5, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.72))
    models = (initial, native["epsilon"], half2["epsilon"])
    titles = ("Initial model", "Native FP32", half2_title)
    vmin = min(float(model.min()) for model in models)
    vmax = max(float(model.max()) for model in models)
    image = None
    for column, (model, title) in enumerate(zip(models, titles)):
        axis = fig.add_subplot(grid[0, column])
        image = axis.imshow(
            model,
            cmap="cividis",
            vmin=vmin,
            vmax=vmax,
            extent=(0, 4.0, 1.0, 0),
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("Horizontal position (m)")
        if column == 0:
            axis.set_ylabel("Depth (m)")
    assert image is not None
    fig.colorbar(image, ax=fig.axes[:3], label="Relative permittivity", shrink=0.82)

    loss_axis = fig.add_subplot(grid[1, :2])
    loss_axis.plot(native["losses"], color="#0072B2", label="Native FP32")
    loss_axis.plot(
        half2["losses"], color="#D55E00", linestyle="--", label=half2_title
    )
    loss_axis.set_yscale("log")
    loss_axis.set_xlabel("Iteration")
    loss_axis.set_ylabel("Normalized full-waveform loss")
    loss_axis.legend(frameon=False)
    loss_axis.spines[["top", "right"]].set_visible(False)

    difference_axis = fig.add_subplot(grid[1, 2])
    difference = half2["epsilon"] - native["epsilon"]
    limit = max(float(np.abs(difference).max()), 1e-6)
    diff_image = difference_axis.imshow(
        difference,
        cmap="PuOr",
        vmin=-limit,
        vmax=limit,
        extent=(0, 4.0, 1.0, 0),
        aspect="auto",
    )
    difference_axis.set_title("half2 − native")
    difference_axis.set_xlabel("Horizontal position (m)")
    difference_axis.set_ylabel("Depth (m)")
    fig.colorbar(diff_image, ax=difference_axis, label="Permittivity difference")
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--full-fp16-adjoint", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/benchmarks/tm2d_lab_glass_half2"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(2026)
    if args.full_fp16_adjoint:
        os.environ["TIDE_TM_FP16_ADJOINT"] = "1"
    else:
        os.environ.pop("TIDE_TM_FP16_ADJOINT", None)
    problem = load_problem(torch.device("cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    native = invert(
        problem,
        mode="native",
        aggressive=False,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    half2 = invert(
        problem,
        mode="fp16_io",
        aggressive=True,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    initial = problem["initial"].cpu().numpy()
    difference = half2["epsilon"] - native["epsilon"]
    summary = {
        "device": torch.cuda.get_device_name(),
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "full_fp16_adjoint": args.full_fp16_adjoint,
        "native": {key: value for key, value in native.items() if key != "epsilon"},
        "aggressive_half2": {
            key: value for key, value in half2.items() if key != "epsilon"
        },
        "speedup_total": native["total_seconds"] / half2["total_seconds"],
        "speedup_median_iteration": native["median_iteration_seconds"]
        / half2["median_iteration_seconds"],
        "final_model_relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(native["epsilon"]), 1e-30)
        ),
        "final_model_correlation": float(
            np.corrcoef(native["epsilon"].ravel(), half2["epsilon"].ravel())[0, 1]
        ),
        "final_model_max_abs_difference": float(np.abs(difference).max()),
    }
    np.savez_compressed(
        args.output_dir / "results.npz",
        initial=initial,
        native=native["epsilon"],
        aggressive_half2=half2["epsilon"],
        native_loss=np.asarray(native["losses"]),
        aggressive_half2_loss=np.asarray(half2["losses"]),
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    make_figure(
        initial,
        native,
        half2,
        args.output_dir / "comparison",
        half2_title=("Full FP16" if args.full_fp16_adjoint else "Aggressive half2"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
