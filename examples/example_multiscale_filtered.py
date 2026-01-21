import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dx = 0.02
dt = 4e-11
nt = 1500
pml_width = 10
air_layer = 3

n_shots = 100
d_source = 4
first_source = 0
# Shots per batch (batch size).
batch_size = 16
model_gradient_sampling_interval = 5
storage_mode = "device"
storage_compression = "fp8"

model_path = "examples/data/OverThrust.npy"
epsilon_true_raw = np.load(model_path)
print(f"Loaded model shape: {epsilon_true_raw.shape}")
print(
    f"Permittivity range: {epsilon_true_raw.min():.2f} - {epsilon_true_raw.max():.2f}"
)

ny, nx = epsilon_true_raw.shape
epsilon_true_np = epsilon_true_raw.copy()
epsilon_true_np[:air_layer, :] = 1.0

sigma_true_np = np.ones_like(epsilon_true_np) * 1e-3
sigma_true_np[:air_layer, :] = 0.0

epsilon_true = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
sigma_true = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)
mu_true = torch.ones_like(epsilon_true)

source_depth = air_layer + 1
source_x = torch.arange(n_shots, device=device) * d_source + first_source

source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
source_locations[:, 0, 0] = source_depth
source_locations[:, 0, 1] = source_x

receiver_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
receiver_locations[:, 0, 0] = source_depth
receiver_locations[:, 0, 1] = source_x + 1

n_shots_per_batch = batch_size

base_forward_freq = 600e6
filter_specs = {
    "lp250": {"lowpass_mhz": 200, "desc": "600 MHz forward result low-pass to 200 MHz"},
    "lp500": {"lowpass_mhz": 400, "desc": "600 MHz forward result low-pass to 400 MHz"},
    "lp700": {"lowpass_mhz": 600, "desc": "600 MHz forward result low-pass to 600 MHz"},
}
inversion_schedule = [
    {"data_key": "lp250", "adamw_epochs": 40, "lbfgs_epochs": 6},
    {"data_key": "lp500", "adamw_epochs": 30, "lbfgs_epochs": 6},
    {"data_key": "lp700", "adamw_epochs": 10, "lbfgs_epochs": 6},
]

print(f"Base forward frequency: {base_forward_freq / 1e6:.0f} MHz")
print("FIR low-pass schedule on observed data:")
for key, spec in filter_specs.items():
    print(f"  {key}: {spec['desc']} (cutoff {spec['lowpass_mhz']} MHz)")
print("Inversion schedule:")
for item in inversion_schedule:
    print(
        f"  {item['data_key']}: AdamW {item['adamw_epochs']}e  "
        f"LBFGS {item['lbfgs_epochs']}e"
    )

lowpass_tag = "-".join(str(spec["lowpass_mhz"]) for spec in filter_specs.values())
output_dir = Path("outputs") / (
    f"multiscale_fir_base{int(base_forward_freq / 1e6)}MHz_lp{lowpass_tag}_shots{n_shots}_bs{batch_size}_nt{nt}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving figures to: {output_dir}")


pde_counts = {"forward": 0.0, "adjoint": 0.0}


def add_pde_counts(
    batch_size: int, forward: bool = False, adjoint: bool = False
) -> None:
    if batch_size <= 0:
        return
    frac = batch_size / n_shots
    if forward:
        pde_counts["forward"] += frac
    if adjoint:
        pde_counts["adjoint"] += frac


def format_pde_counts(forward: float, adjoint: float) -> str:
    total = forward + adjoint
    return f"forward {forward:.2f}, adjoint {adjoint:.2f}, total {total:.2f}"


def report_pde_totals(prefix: str) -> None:
    print(
        f"{prefix}PDE solves (100 shots = 1): {format_pde_counts(pde_counts['forward'], pde_counts['adjoint'])}"
    )


def report_pde_delta(prefix: str, forward_start: float, adjoint_start: float) -> None:
    forward = pde_counts["forward"] - forward_start
    adjoint = pde_counts["adjoint"] - adjoint_start
    print(f"{prefix}PDE solves: {format_pde_counts(forward, adjoint)}")


def make_shot_batches() -> list[torch.Tensor]:
    perm = torch.arange(n_shots, device=device)
    return [
        perm[i : i + n_shots_per_batch] for i in range(0, n_shots, n_shots_per_batch)
    ]


def design_fir_filter(cutoff_hz: float, fs: float, numtaps: int) -> torch.Tensor:
    """Design a Hamming-windowed low-pass FIR filter."""
    n = torch.arange(numtaps, dtype=torch.float32)
    window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (numtaps - 1))
    sinc = torch.sin(2 * torch.pi * (cutoff_hz / fs) * (n - (numtaps - 1) / 2)) / (
        torch.pi * (n - (numtaps - 1) / 2)
    )
    center = (numtaps - 1) // 2
    sinc[center] = 2 * cutoff_hz / fs
    h = window * sinc
    return h / h.sum()


def apply_fir_lowpass(data: torch.Tensor, dt: float, cutoff_hz: float) -> torch.Tensor:
    """Apply FIR low-pass filter along the time axis to observed/synthetic data."""
    if cutoff_hz <= 0:
        return data

    fs = 1.0 / dt
    numtaps = max(3, int(fs / cutoff_hz))
    if numtaps % 2 == 0:
        numtaps += 1
    fir_coeff = design_fir_filter(cutoff_hz, fs, numtaps).to(
        device=data.device, dtype=data.dtype
    )

    if data.ndim == 1:
        data_2d = data.view(1, 1, -1)
        padded = F.pad(data_2d, (numtaps - 1, 0), mode="reflect")
        filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
        return filtered.view(-1)

    if data.ndim == 3:
        nt_local, n_shots_local, n_rx_local = data.shape
        reshaped = data.permute(1, 2, 0).reshape(-1, 1, nt_local)
        padded = F.pad(reshaped, (numtaps - 1, 0), mode="reflect")
        filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
        return filtered.view(n_shots_local, n_rx_local, nt_local).permute(2, 0, 1)

    raise ValueError(
        f"Unsupported data dimension: {data.ndim}. Expected 1D or 3D tensor."
    )


def save_filter_comparison(
    observed_base: torch.Tensor, observed_sets: dict, output_dir: Path
) -> None:
    """Save base vs filtered data comparison figure."""
    base_np = observed_base.detach().cpu().numpy()[:, :, 0]
    filtered_arrays = []
    for key in filter_specs:
        data_np = observed_sets[key]["data"].detach().cpu().numpy()[:, :, 0]
        filtered_arrays.append((key, data_np, observed_sets[key]["desc"]))

    absmax = max(
        np.abs(base_np).max(), *(np.abs(arr).max() for _, arr, _ in filtered_arrays)
    )
    vlim = (-absmax, absmax)

    n_cols = 1 + len(filtered_arrays)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5 * n_cols, 5), sharex=True, sharey=True
    )
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(base_np, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
    axes[0].set_title(f"{base_forward_freq / 1e6:.0f} MHz base")
    axes[0].set_xlabel("Shots")
    axes[0].set_ylabel("Time samples")

    for idx, (_, arr, desc) in enumerate(filtered_arrays, start=1):
        axes[idx].imshow(arr, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
        axes[idx].set_title(desc)
        axes[idx].set_xlabel("Shots")

    plt.tight_layout()
    filename = (
        output_dir
        / f"data_filter_comparison_base{int(base_forward_freq / 1e6)}_lp{lowpass_tag}.jpg"
    )
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved data filter comparison to '{filename}'")


def save_model_snapshot(
    eps_array: np.ndarray, title: str, filename: Path, vmin: float, vmax: float
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(eps_array, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="εr")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved model snapshot to '{filename}'")


def forward_shots(
    epsilon, sigma, mu, shot_indices, source_amplitude_full, requires_grad=True
):
    src_amp = source_amplitude_full[shot_indices]
    src_loc = source_locations[shot_indices]
    rec_loc = receiver_locations[shot_indices]

    out = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=src_amp,
        source_location=src_loc,
        receiver_location=rec_loc,
        pml_width=pml_width,
        save_snapshots=requires_grad,
        model_gradient_sampling_interval=model_gradient_sampling_interval
        if requires_grad
        else 1,
        storage_mode=storage_mode,
        storage_compression=storage_compression,
    )
    return out[-1]  # [nt, shots_in_batch, 1]


def generate_base_and_filtered_observed():
    with torch.no_grad():
        wavelet = tide.ricker(
            base_forward_freq, nt, dt, peak_time=1.0 / base_forward_freq
        ).to(device)
        src_amp_full = wavelet.view(1, 1, nt).repeat(n_shots, 1, 1)

        obs_list = []
        for shot_indices in make_shot_batches():
            obs_list.append(
                forward_shots(
                    epsilon_true,
                    sigma_true,
                    mu_true,
                    shot_indices,
                    src_amp_full,
                    requires_grad=False,
                )
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
        observed_base = torch.cat(obs_list, dim=1)

        observed_sets = {}
        for key, spec in filter_specs.items():
            lowpass_hz = float(spec["lowpass_mhz"]) * 1e6
            data_filtered = (
                apply_fir_lowpass(observed_base, dt=dt, cutoff_hz=lowpass_hz)
                if lowpass_hz > 0
                else observed_base
            )
            observed_sets[key] = {
                "data": data_filtered,
                "lowpass_hz": lowpass_hz,
                "desc": spec["desc"],
            }

    return observed_base, observed_sets, src_amp_full


sigma_smooth = 8
epsilon_init_raw = gaussian_filter(epsilon_true_raw, sigma=sigma_smooth)
epsilon_init_np = epsilon_init_raw.copy()
epsilon_init_np[:air_layer, :] = 1.0

sigma_init_np = np.ones_like(epsilon_init_np) * 0
sigma_init_np[:air_layer, :] = 0.0

epsilon_init = torch.tensor(epsilon_init_np, dtype=torch.float32, device=device)
sigma_init = torch.tensor(sigma_init_np, dtype=torch.float32, device=device)

epsilon_inv = epsilon_init.clone().detach()
epsilon_inv.requires_grad_(True)

sigma_fixed = sigma_init.clone().detach()
mu_fixed = torch.ones_like(epsilon_inv)

air_mask = torch.zeros_like(epsilon_inv, dtype=torch.bool)
air_mask[:air_layer, :] = True

loss_fn = torch.nn.MSELoss()
all_losses = []
stage_breaks = []

print("Starting multiscale filtered inversion")
time_start_all = time.time()

print("Generating base observed data once, then FIR filtering...")
observed_raw, observed_sets, src_amp_full = generate_base_and_filtered_observed()
print(f"Base forward modeled at {base_forward_freq / 1e6:.0f} MHz.")
report_pde_totals("After observed generation: ")
save_filter_comparison(observed_raw, observed_sets, output_dir)

vmin_stage = epsilon_true_np.min()
vmax_stage = epsilon_true_np.max()

for stage_idx, cfg in enumerate(inversion_schedule, 1):
    data_key = cfg["data_key"]
    obs_cfg = observed_sets[data_key]
    n_epochs_adamw = int(cfg["adamw_epochs"])
    n_epochs_lbfgs = int(cfg["lbfgs_epochs"])
    lowpass_hz = obs_cfg["lowpass_hz"]

    print(f"\n==== Stage {stage_idx}: {obs_cfg['desc']} ====")
    observed_filtered = obs_cfg["data"]
    stage_forward_start = pde_counts["forward"]
    stage_adjoint_start = pde_counts["adjoint"]

    # Stage 1: AdamW
    optimizer_adamw = torch.optim.AdamW(
        [epsilon_inv], lr=0.01, betas=(0.9, 0.99), weight_decay=1e-3
    )
    for epoch in range(n_epochs_adamw):
        optimizer_adamw.zero_grad()
        epoch_loss = 0.0

        for shot_indices in make_shot_batches():
            syn = forward_shots(
                epsilon_inv,
                sigma_fixed,
                mu_fixed,
                shot_indices,
                src_amp_full,
                requires_grad=True,
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
            syn_filtered = apply_fir_lowpass(syn, dt=dt, cutoff_hz=lowpass_hz)
            obs_batch = observed_filtered[:, shot_indices, :]

            loss = loss_fn(syn_filtered, obs_batch)
            loss.backward()
            add_pde_counts(int(shot_indices.numel()), adjoint=True)
            epoch_loss += loss.item()

        if epsilon_inv.grad is not None:
            epsilon_inv.grad[air_mask] = 0.0
            valid_grads = epsilon_inv.grad[~air_mask].abs()
            if valid_grads.numel() > 0:
                clip_val = torch.quantile(valid_grads, 0.98)
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val.item())

        optimizer_adamw.step()

        with torch.no_grad():
            epsilon_inv.clamp_(1.0, 9.0)
            epsilon_inv[air_mask] = 1.0

        all_losses.append(epoch_loss)
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"  AdamW epoch {epoch + 1}/{n_epochs_adamw}  Loss={epoch_loss:.6e}")

    # Stage 2: L-BFGS
    optimizer_lbfgs = torch.optim.LBFGS(
        [epsilon_inv],
        lr=1.0,
        history_size=10,
        max_iter=5,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss = torch.zeros((), device=device)
        for shot_indices in make_shot_batches():
            syn = forward_shots(
                epsilon_inv,
                sigma_fixed,
                mu_fixed,
                shot_indices,
                src_amp_full,
                requires_grad=True,
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
            syn_filtered = apply_fir_lowpass(syn, dt=dt, cutoff_hz=lowpass_hz)
            obs_batch = observed_filtered[:, shot_indices, :]

            loss = loss_fn(syn_filtered, obs_batch)
            loss.backward()
            add_pde_counts(int(shot_indices.numel()), adjoint=True)
            total_loss = total_loss + loss

        if epsilon_inv.grad is not None:
            epsilon_inv.grad[air_mask] = 0.0
            valid_grads = epsilon_inv.grad[~air_mask].abs()
            if valid_grads.numel() > 0:
                clip_val = torch.quantile(valid_grads, 0.98)
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val.item())
        return total_loss

    for epoch in range(n_epochs_lbfgs):
        loss = optimizer_lbfgs.step(closure)
        with torch.no_grad():
            epsilon_inv.clamp_(1.0, 9.0)
            epsilon_inv[air_mask] = 1.0
        loss_value = loss.item()
        all_losses.append(loss_value)
        print(f"  LBFGS epoch {epoch + 1}/{n_epochs_lbfgs}  Loss={loss_value:.6e}")

    stage_breaks.append(len(all_losses) - 1)
    report_pde_delta(f"Stage {stage_idx} ", stage_forward_start, stage_adjoint_start)
    eps_stage = epsilon_inv.detach().cpu().numpy()
    stage_title = f"{obs_cfg['desc']} inversion result"
    stage_fname = output_dir / f"epsilon_stage_{data_key}.jpg"
    save_model_snapshot(eps_stage, stage_title, stage_fname, vmin_stage, vmax_stage)

time_all = time.time() - time_start_all
print(f"\nTotal inversion time: {time_all:.2f}s")
report_pde_totals("Total ")

eps_true = epsilon_true.cpu().numpy()
eps_init = epsilon_init.cpu().numpy()
eps_result = epsilon_inv.detach().cpu().numpy()

vmin = eps_true.min()
vmax = eps_true.max()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
im = ax.imshow(eps_true, aspect="auto", vmin=vmin, vmax=vmax)
ax.set_title("True Model")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[0, 1]
im = ax.imshow(eps_init, aspect="auto", vmin=vmin, vmax=vmax)
ax.set_title("Initial Model (Smoothed)")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[1, 0]
im = ax.imshow(eps_result, aspect="auto", vmin=vmin, vmax=vmax)
ax.set_title("Multiscale Filtered Result")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[1, 1]
ax.semilogy(all_losses, label="Loss")
for idx in stage_breaks:
    ax.axvline(idx, color="r", linestyle="--", alpha=0.5)
ax.set_title("Loss Curve (AdamW -> LBFGS stages)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.grid(True)
ax.legend()

plt.tight_layout()
final_plot = output_dir / "multiscale_filtered_summary.jpg"
plt.savefig(final_plot, dpi=150)
print(f"\nResults saved to '{final_plot}'")

# Save inverted model for metrics computation
np.save(output_dir / "epsilon_inverted.npy", eps_result)
print(f"Saved inverted model to '{output_dir / 'epsilon_inverted.npy'}'")

mask = ~(air_mask.cpu().numpy())
rms_init = np.sqrt(np.mean((eps_init[mask] - eps_true[mask]) ** 2))
rms_result = np.sqrt(np.mean((eps_result[mask] - eps_true[mask]) ** 2))

print(f"RMS Error (Initial):  {rms_init:.4f}")
print(f"RMS Error (Inverted): {rms_result:.4f}")
print(f"Improvement: {(1 - rms_result / rms_init) * 100:.1f}%")

print("\n=== Timing Summary ===")
print(f"Total inversion time: {time_all:.2f}s")
