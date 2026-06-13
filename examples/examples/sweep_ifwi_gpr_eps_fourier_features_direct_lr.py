from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import example_ifwi_gpr_eps_fourier_features_direct as direct
except ImportError:
    from examples import example_ifwi_gpr_eps_fourier_features_direct as direct


DEFAULT_LRS = (3e-5, 1e-4, 2e-4, 5e-4, 1e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short learning-rate sweep for direct Fourier-feature GPR IFWI."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lrs", type=float, nargs="+", default=list(DEFAULT_LRS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--compile", action="store_true", dest="compile_network")
    return parser.parse_args()


def make_model(
    *,
    ny: int,
    nx: int,
    mu: torch.Tensor,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    compile_network: bool,
) -> direct.MaxwellTMFourierFeatureDirectFWI:
    model = direct.MaxwellTMFourierFeatureDirectFWI(
        ny=ny,
        nx=nx,
        dx=direct.DX,
        dt=direct.DT,
        nt=direct.NT,
        pml_width=direct.PML_WIDTH,
        eps_min=direct.EPS_MIN,
        eps_max=direct.EPS_MAX,
        sigma_min=direct.SIGMA_MIN,
        sigma_max=direct.SIGMA_MAX,
        mu=mu,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        air_layers=direct.AIR_LAYERS,
        fourier_mapping_size=direct.FOURIER_MAPPING_SIZE,
        fourier_scale=direct.FOURIER_SCALE,
        fourier_include_input=direct.FOURIER_INCLUDE_INPUT,
        hidden_features=direct.HIDDEN_FEATURES,
        hidden_layers=direct.HIDDEN_LAYERS,
        activation=direct.ACTIVATION,
        final_layer_scale=direct.FINAL_LAYER_SCALE,
        model_gradient_sampling_interval=direct.MODEL_GRADIENT_SAMPLING_INTERVAL,
        output_smoothing_kernel=direct.OUTPUT_SMOOTHING_KERNEL,
        output_smoothing_passes=direct.OUTPUT_SMOOTHING_PASSES,
        coord_chunk_size=direct.COORD_CHUNK_SIZE,
        compile_network=compile_network,
    ).to(mu.device)
    model.maybe_compile_network()
    return model


def save_sweep_curve(
    *,
    histories: dict[float, list[float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr, losses in histories.items():
        ax.plot(range(1, len(losses) + 1), losses, label=f"lr={lr:g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if not args.lrs:
        raise ValueError("--lrs must contain at least one learning rate.")

    device = direct.base.resolve_device(direct.DEVICE_NAME)
    output_dir = args.output_dir
    if output_dir is None:
        lrs_label = "-".join(f"{lr:g}" for lr in args.lrs)
        output_dir = (
            Path("outputs")
            / "lr_sweep_direct_fourier"
            / f"epochs{args.epochs}_fls{direct.FINAL_LAYER_SCALE:g}_lr{lrs_label}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Saving outputs to: {output_dir}")
    print(
        "Sweep config: "
        f"epochs={args.epochs}, lrs={args.lrs}, "
        f"final_layer_scale={direct.FINAL_LAYER_SCALE:g}, "
        f"torch_compile={'on' if args.compile_network else 'off'}"
    )

    torch.manual_seed(direct.SEED)
    np.random.seed(direct.SEED)

    epsilon_true_np = direct.base.make_true_epsilon_model(
        model_path=direct.MODEL_PATH,
        air_layers=direct.AIR_LAYERS,
        ny=direct.NY,
        nx=direct.NX,
    )
    ny, nx = epsilon_true_np.shape
    sigma_true_np = direct.base.sigma_from_epsilon_np(epsilon_true_np)
    sigma_true_np[: direct.AIR_LAYERS, :] = 0.0

    epsilon_true = torch.tensor(epsilon_true_np, device=device, dtype=torch.float32)
    sigma_true = torch.tensor(sigma_true_np, device=device, dtype=torch.float32)
    mu = torch.ones_like(epsilon_true)

    source_locations, receiver_locations = direct.base.build_geometry(
        nx=nx,
        air_layers=direct.AIR_LAYERS,
        n_shots=direct.N_SHOTS,
        source_x_min=direct.SOURCE_X_MIN,
        source_x_max=direct.SOURCE_X_MAX,
        source_depth=direct.SOURCE_DEPTH,
        receiver_depth=direct.RECEIVER_DEPTH,
        device=device,
    )

    wavelet = direct.base.tide.ricker(
        direct.BASE_FREQ,
        direct.NT,
        direct.DT,
        peak_time=1.5 / direct.BASE_FREQ,
        device=device,
        dtype=torch.float32,
    )

    time_start = time.time()
    observed = direct.base.generate_observed_data(
        epsilon_true=epsilon_true,
        sigma_fixed=sigma_true,
        mu=mu,
        wavelet=wavelet,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        dx=direct.DX,
        dt=direct.DT,
        pml_width=direct.PML_WIDTH,
        batch_size=direct.BATCH_SIZE,
        model_gradient_sampling_interval=direct.MODEL_GRADIENT_SAMPLING_INTERVAL,
    )
    print(f"Observed data generated in {time.time() - time_start:.2f}s")

    histories: dict[float, list[float]] = {}
    rows: list[dict[str, float | int | str]] = []
    valid_mask = np.ones_like(epsilon_true_np, dtype=bool)
    valid_mask[: direct.AIR_LAYERS, :] = False

    for lr in args.lrs:
        print(f"\n=== LR {lr:g} ===")
        torch.manual_seed(direct.SEED)
        np.random.seed(direct.SEED)
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model = make_model(
            ny=ny,
            nx=nx,
            mu=mu,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            compile_network=args.compile_network,
        )

        with torch.no_grad():
            epsilon_start, sigma_start = model.predict_models()
            epsilon_start_np = epsilon_start.detach().cpu().numpy()
            sigma_start_np = sigma_start.detach().cpu().numpy()

        lr_start = time.time()
        losses = direct.run_waveform_inversion(
            model=model,
            wavelet=wavelet,
            observed=observed,
            epochs=args.epochs,
            lr=lr,
            weight_decay=direct.WEIGHT_DECAY,
            batch_size=direct.BATCH_SIZE,
            grad_clip=direct.GRAD_CLIP,
            log_interval=args.log_interval,
        )
        histories[lr] = losses

        with torch.no_grad():
            epsilon_pred, sigma_pred = model.predict_models()
        epsilon_pred_np = epsilon_pred.detach().cpu().numpy()
        sigma_pred_np = sigma_pred.detach().cpu().numpy()

        final_loss = losses[-1]
        best_loss = min(losses)
        rel_drop = (losses[0] - final_loss) / losses[0] if losses[0] > 0.0 else float("nan")
        row = {
            "lr": lr,
            "epochs": args.epochs,
            "initial_loss": losses[0],
            "final_loss": final_loss,
            "best_loss": best_loss,
            "relative_drop": rel_drop,
            "epsilon_start_std": float(epsilon_start_np[direct.AIR_LAYERS :, :].std()),
            "sigma_start_std": float(sigma_start_np[direct.AIR_LAYERS :, :].std()),
            "epsilon_rel_l2": direct.base.relative_l2(epsilon_pred_np, epsilon_true_np, valid_mask),
            "sigma_rel_l2": direct.base.relative_l2(sigma_pred_np, sigma_true_np, valid_mask),
            "elapsed_s": time.time() - lr_start,
        }
        rows.append(row)
        print(
            "LR result: "
            f"initial={losses[0]:.6e}, final={final_loss:.6e}, "
            f"best={best_loss:.6e}, drop={rel_drop:.2%}, "
            f"elapsed={row['elapsed_s']:.2f}s"
        )

    csv_path = output_dir / "lr_sweep_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        output_dir / "lr_sweep_losses.npz",
        **{f"lr_{lr:g}".replace(".", "p").replace("-", "m"): np.asarray(losses) for lr, losses in histories.items()},
    )
    save_sweep_curve(histories=histories, output_path=output_dir / "lr_sweep_loss.jpg")

    rows_by_loss = sorted(rows, key=lambda row: float(row["best_loss"]))
    print("\nSummary by best loss:")
    for row in rows_by_loss:
        print(
            f"lr={row['lr']:g} best={row['best_loss']:.6e} "
            f"final={row['final_loss']:.6e} drop={row['relative_drop']:.2%} "
            f"eps_rel_l2={row['epsilon_rel_l2']:.4f}"
        )
    print(f"\nWrote sweep summary to: {csv_path}")


if __name__ == "__main__":
    main()
