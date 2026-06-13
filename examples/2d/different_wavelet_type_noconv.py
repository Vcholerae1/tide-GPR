import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sotb_wrapper import interface as sotb_interface

import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dx = 0.02
dt = 4e-11
nt = 1800
pml_width = 20
air_layer = 3

n_shots = 100
d_source = 4
first_source = 0
# Shots per batch (batch size).
batch_size = 16
model_gradient_sampling_interval = 10
storage_mode = "device"
storage_compression = "bf16"


model_path = "multi_freq_inv/data/OverThrust.npy"
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

source_depth = air_layer - 1
source_x = torch.arange(n_shots, device=device) * d_source + first_source

source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
source_locations[:, 0, 0] = source_depth
source_locations[:, 0, 1] = source_x

receiver_offset = 1
receiver_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
receiver_locations[:, 0, 0] = source_depth
receiver_locations[:, 0, 1] = source_x + receiver_offset

n_shots_per_batch = batch_size

# 观测使用 Ricker；反演使用 Gaussian derivative（guassdot）
base_freq = 600e6
obs_freq = base_freq
inv_freq = base_freq
obs_peak = 1.5 / obs_freq 
inv_peak = 1.5 / inv_freq 
base_forward_freq = obs_freq
filter_specs = {
    # "lp100": {"lowpass_mhz": 100, "desc": "600 MHz forward result low-pass to 200 MHz"},    
    # "lp250": {"lowpass_mhz": 150, "desc": "600 MHz forward result low-pass to 200 MHz"},
    # "lp300": {"lowpass_mhz": 200, "desc": "600 MHz forward result low-pass to 400 MHz"},
    "lp400": {"lowpass_mhz": 400, "desc": "600 MHz forward result low-pass to 400 MHz"},    
    "lp600": {"lowpass_mhz": 600, "desc": "600 MHz forward result low-pass to 600 MHz"},
}
inversion_schedule = [
    # {"data_key": "lp100", "adamw_epochs": 40, "lbfgs_epochs": 10},    
    # {"data_key": "lp250", "adamw_epochs": 40, "lbfgs_epochs": 15},
    # {"data_key": "lp300", "adamw_epochs": 40, "lbfgs_epochs": 15},
    {"data_key": "lp400", "adamw_epochs": 40, "lbfgs_epochs": 15},
    {"data_key": "lp600", "adamw_epochs": 30, "lbfgs_epochs": 15},
]
# SOTB PLBFGS settings (paper-inspired diagonal preconditioning).
plbfgs_conv = 1e-8
plbfgs_nls_max = 20
plbfgs_l = 5
plbfgs_precond_smooth_sigma = 3.0
plbfgs_precond_damping = 5e-2
plbfgs_precond_power = 0.5
plbfgs_precond_clip_lo = 0.3
plbfgs_precond_clip_hi = 3.0
plbfgs_precond_blend = 0.7


print(
    "Observed wavelet: "
    f"Ricker {obs_freq / 1e6:.0f} MHz, peak {obs_peak * 1e9:.3f} ns"
)
print(
    "Inversion wavelet: "
    f"Gaussian derivative {inv_freq / 1e6:.0f} MHz, center {inv_peak * 1e9:.3f} ns"
)
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
    "crosscorr_wavelettype_noconv_"
    f"obsRicker{int(obs_freq / 1e6)}MHz_invGaussD{int(inv_freq / 1e6)}MHz_"
    f"lp{lowpass_tag}_shots{n_shots}_bs{batch_size}_nt{nt}"
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


def make_ricker(
    freq: float, nt_: int, dt_: float, peak_time: float, dev: torch.device
) -> torch.Tensor:
    t = torch.arange(nt_, device=dev, dtype=torch.float32) * dt_
    w = np.pi * freq * (t - peak_time)
    return (1.0 - 2.0 * w**2) * torch.exp(-(w**2))


def make_gaussian_derivative(
    freq: float,
    nt_: int,
    dt_: float,
    center_time: float,
    dev: torch.device,
) -> torch.Tensor:
    t = torch.arange(nt_, device=dev, dtype=torch.float32) * dt_
    sigma = 1.0 / (2.5 * freq)
    x = (t - center_time) / sigma
    w = -x * torch.exp(-0.5 * x**2)
    max_abs = torch.max(torch.abs(w))
    if float(max_abs) > 0.0:
        w = w / max_abs
    return w


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


def apply_fir_lowpass(data: torch.Tensor, dt_: float, cutoff_hz: float) -> torch.Tensor:
    """Apply FIR low-pass filter along the time axis to observed/synthetic data."""
    if cutoff_hz <= 0:
        return data

    fs = 1.0 / dt_
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


def save_wavelet_comparison(
    wavelet_obs: torch.Tensor, wavelet_inv: torch.Tensor, output_dir_: Path
) -> None:
    t_ns = torch.arange(nt, device=wavelet_obs.device, dtype=wavelet_obs.dtype) * dt * 1e9
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_ns.cpu().numpy(), wavelet_obs.detach().cpu().numpy(), label="Observed (Ricker)")
    ax.plot(t_ns.cpu().numpy(), wavelet_inv.detach().cpu().numpy(), label="Inversion (Gaussian derivative)")
    ax.set_title("Original Wavelets Comparison")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    filename = output_dir_ / "wavelet_original_comparison.jpg"
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved wavelet comparison to '{filename}'")


def data_mse_loss_no_conv(
    d_obs: torch.Tensor,
    d_pred: torch.Tensor,
) -> torch.Tensor:
    """No-convolution objective, aligned with example_multiscale_filtered_common.py."""
    if d_obs.dim() == 3 and d_obs.shape[1] == 1:
        d_obs = d_obs.squeeze(1)
        d_pred = d_pred.squeeze(1)

    if d_obs.dim() == 3 and d_pred.dim() == 3:
        d_obs = d_obs.reshape(d_obs.shape[0] * d_obs.shape[1], d_obs.shape[2])
        d_pred = d_pred.reshape(d_pred.shape[0] * d_pred.shape[1], d_pred.shape[2])

    if d_obs.dim() != 2 or d_pred.dim() != 2:
        raise ValueError("Expected d_obs and d_pred with shape [n_traces, nt].")

    return F.mse_loss(d_pred, d_obs)


def save_filter_comparison(
    observed_base: torch.Tensor,
    observed_sets: dict,
    output_dir_: Path,
) -> None:
    """Save base vs filtered common-offset comparison figure."""
    base_np = observed_base.detach().cpu().numpy()[:, :, 0]
    filtered_arrays = []
    for key in filter_specs:
        data_np = observed_sets[key]["data"].detach().cpu().numpy()[:, :, 0]
        filtered_arrays.append((key, data_np, observed_sets[key]["desc"]))

    vlim = (-50, 50)

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
        output_dir_
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
    return out[-1]  # [nt, shots_in_batch, n_receivers]


def generate_base_and_filtered_observed():
    with torch.no_grad():
        wavelet_obs = make_ricker(obs_freq, nt, dt, peak_time=obs_peak, dev=device)
        wavelet_inv = make_gaussian_derivative(
            inv_freq, nt, dt, center_time=inv_peak, dev=device
        )
        src_amp_obs = wavelet_obs.view(1, 1, nt).repeat(n_shots, 1, 1)
        src_amp_inv = wavelet_inv.view(1, 1, nt).repeat(n_shots, 1, 1)

        obs_list = []
        for shot_indices in make_shot_batches():
            obs_list.append(
                forward_shots(
                    epsilon_true,
                    sigma_true,
                    mu_true,
                    shot_indices,
                    src_amp_obs,
                    requires_grad=False,
                )
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
        observed_base = torch.cat(obs_list, dim=1)

        observed_sets = {}
        for key, spec in filter_specs.items():
            lowpass_hz = float(spec["lowpass_mhz"]) * 1e6
            data_filtered = (
                apply_fir_lowpass(observed_base, dt_=dt, cutoff_hz=lowpass_hz)
                if lowpass_hz > 0
                else observed_base
            )
            observed_sets[key] = {
                "data": data_filtered,
                "lowpass_hz": lowpass_hz,
                "desc": spec["desc"],
            }

    return observed_base, observed_sets, wavelet_obs, wavelet_inv, src_amp_inv


sigma_smooth = 20
epsilon_init_raw = gaussian_filter(epsilon_true_raw, sigma=sigma_smooth)
epsilon_init_np = epsilon_init_raw.copy()
epsilon_init_np[:air_layer, :] = 1.0

sigma_init_np = np.ones_like(epsilon_init_np) * 1e-3
sigma_init_np[:air_layer, :] = 0.0

epsilon_init = torch.tensor(epsilon_init_np, dtype=torch.float32, device=device)
sigma_init = torch.tensor(sigma_init_np, dtype=torch.float32, device=device)

epsilon_inv = epsilon_init.clone().detach()
epsilon_inv.requires_grad_(True)

sigma_fixed = sigma_init.clone().detach()
mu_fixed = torch.ones_like(epsilon_inv)

air_mask = torch.zeros_like(epsilon_inv, dtype=torch.bool)
air_mask[:air_layer, :] = True

all_losses = []
stage_breaks = []

print("Starting multiscale cross-correlation inversion")
time_start_all = time.time()

print("Generating base observed data once, then FIR filtering...")
(
    observed_raw,
    observed_sets,
    wavelet_obs,
    wavelet_inv,
    src_amp_inv,
) = generate_base_and_filtered_observed()
print(f"Observed data modeled at {obs_freq / 1e6:.0f} MHz.")
print(f"Inversion modeled at {inv_freq / 1e6:.0f} MHz.")
if sotb_interface is None:
    raise RuntimeError(
        "sotb-wrapper is not importable. Install via "
        "`uv pip install --python .venv/bin/python sotb-wrapper`."
    )
print("Loss mode: mse")
print("LBFGS backend: SOTB PLBFGS (diagonal GN-style preconditioner)")
print(
    "PLBFGS preconditioner: "
    f"smooth_sigma={plbfgs_precond_smooth_sigma}, "
    f"damping={plbfgs_precond_damping:.1e}, "
    f"power={plbfgs_precond_power:.2f}, "
    f"clip=[{plbfgs_precond_clip_lo:.2f}, {plbfgs_precond_clip_hi:.2f}], "
    f"blend={plbfgs_precond_blend:.2f}"
)
report_pde_totals("After observed generation: ")
save_wavelet_comparison(wavelet_obs, wavelet_inv, output_dir)
save_filter_comparison(observed_raw, observed_sets, output_dir)

vmin_stage = epsilon_true_np.min()
vmax_stage = epsilon_true_np.max()
n_param_eps = int(epsilon_inv.numel())
air_mask_np = air_mask.detach().cpu().numpy().reshape(-1)


def pack_eps_param(epsilon_param: torch.Tensor) -> np.ndarray:
    return (
        epsilon_param.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .astype(np.float32, copy=False)
    )


def unpack_eps_param(x: np.ndarray, epsilon_param: torch.Tensor) -> None:
    eps_vec = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        epsilon_param.copy_(eps_vec.view_as(epsilon_param))


def pack_eps_grad(epsilon_param: torch.Tensor) -> np.ndarray:
    if epsilon_param.grad is None:
        grad = torch.zeros_like(epsilon_param)
    else:
        grad = epsilon_param.grad
    grad_np = (
        grad.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .astype(np.float32, copy=False)
    )
    np.nan_to_num(grad_np, copy=False)
    return grad_np


def build_eps_bounds() -> tuple[np.ndarray, np.ndarray]:
    lb = np.full(n_param_eps, 1.0, dtype=np.float32)
    ub = np.full(n_param_eps, 9.0, dtype=np.float32)
    lb[air_mask_np] = 1.0
    ub[air_mask_np] = 1.0
    return lb, ub


lb_bounds, ub_bounds = build_eps_bounds()

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
                src_amp_inv,
                requires_grad=True,
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
            syn_filtered = apply_fir_lowpass(syn, dt_=dt, cutoff_hz=lowpass_hz)
            obs_batch = observed_filtered[:, shot_indices, :]
            obs_batch = obs_batch.permute(1, 2, 0)
            syn_batch = syn_filtered.permute(1, 2, 0)
            loss = data_mse_loss_no_conv(obs_batch, syn_batch)
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

    # Stage 2: SOTB P-LBFGS
    sotb = sotb_interface.sotb_wrapper()
    sotb.udf = sotb_interface.UserDefined()
    x = pack_eps_param(epsilon_inv)
    n = int(x.size)

    def build_plbfgs_preconditioner_diag() -> np.ndarray:
        if epsilon_inv.grad is not None:
            epsilon_inv.grad.zero_()

        p_diag = torch.zeros_like(epsilon_inv)
        for shot_indices in make_shot_batches():
            if epsilon_inv.grad is not None:
                epsilon_inv.grad.zero_()

            syn = forward_shots(
                epsilon_inv,
                sigma_fixed,
                mu_fixed,
                shot_indices,
                src_amp_inv,
                requires_grad=True,
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
            syn_filtered = apply_fir_lowpass(syn, dt_=dt, cutoff_hz=lowpass_hz)
            obs_batch = observed_filtered[:, shot_indices, :]
            obs_batch = obs_batch.permute(1, 2, 0)
            syn_batch = syn_filtered.permute(1, 2, 0)
            loss = data_mse_loss_no_conv(obs_batch, syn_batch)
            loss.backward()
            add_pde_counts(int(shot_indices.numel()), adjoint=True)

            if epsilon_inv.grad is not None:
                g = torch.nan_to_num(
                    epsilon_inv.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0
                )
                g = g.clone()
                g[air_mask] = 0.0
                p_diag += g * g

        p_np = p_diag.detach().cpu().numpy().astype(np.float32, copy=False)
        if plbfgs_precond_smooth_sigma > 0:
            p_np = gaussian_filter(p_np, sigma=plbfgs_precond_smooth_sigma)
        p_flat = p_np.reshape(-1)
        p_flat[~np.isfinite(p_flat)] = 0.0

        valid = ~air_mask_np
        p_valid = p_flat[valid]
        if p_valid.size == 0:
            b0 = np.ones(n_param_eps, dtype=np.float32)
            b0[air_mask_np] = 0.0
            return b0

        scale = float(np.quantile(p_valid, 0.95))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        p_flat = p_flat / scale

        b0 = np.power(p_flat + plbfgs_precond_damping, -plbfgs_precond_power)
        b0[~np.isfinite(b0)] = 0.0
        b_valid = b0[valid]
        if b_valid.size > 0:
            med = float(np.median(b_valid))
            if np.isfinite(med) and med > 0.0:
                b0 = b0 / med

            np.clip(b0, plbfgs_precond_clip_lo, plbfgs_precond_clip_hi, out=b0)

            # Blend towards identity to avoid over-aggressive scaling.
            b0 = (1.0 - plbfgs_precond_blend) + plbfgs_precond_blend * b0

        b0[air_mask_np] = 0.0
        return b0.astype(np.float32, copy=False)

    print("  Building PLBFGS preconditioner (diag GN proxy + smoothing)...")
    b0_diag = build_plbfgs_preconditioner_diag()
    b0_valid = b0_diag[~air_mask_np]
    if b0_valid.size > 0:
        print(
            f"  B0 stats (valid): min={b0_valid.min():.3e} "
            f"median={np.median(b0_valid):.3e} max={b0_valid.max():.3e}"
        )

    def evaluate_from_x() -> tuple[float, np.ndarray, np.ndarray]:
        unpack_eps_param(x, epsilon_inv)
        with torch.no_grad():
            epsilon_inv.clamp_(1.0, 9.0)
            epsilon_inv[air_mask] = 1.0
        x[:] = pack_eps_param(epsilon_inv)

        if epsilon_inv.grad is not None:
            epsilon_inv.grad.zero_()

        total_loss = 0.0
        for shot_indices in make_shot_batches():
            syn = forward_shots(
                epsilon_inv,
                sigma_fixed,
                mu_fixed,
                shot_indices,
                src_amp_inv,
                requires_grad=True,
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
            syn_filtered = apply_fir_lowpass(syn, dt_=dt, cutoff_hz=lowpass_hz)
            obs_batch = observed_filtered[:, shot_indices, :]
            obs_batch = obs_batch.permute(1, 2, 0)
            syn_batch = syn_filtered.permute(1, 2, 0)
            loss = data_mse_loss_no_conv(obs_batch, syn_batch)
            loss.backward()
            add_pde_counts(int(shot_indices.numel()), adjoint=True)
            total_loss += float(loss.item())

        if epsilon_inv.grad is not None:
            epsilon_inv.grad[air_mask] = 0.0
            epsilon_inv.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            valid_grads = epsilon_inv.grad[~air_mask].abs()
            if valid_grads.numel() > 0:
                clip_val = torch.quantile(valid_grads, 0.98)
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val.item())

        grad = pack_eps_grad(epsilon_inv)
        grad_preco = (b0_diag * grad).astype(np.float32, copy=False)
        np.nan_to_num(grad_preco, copy=False)
        x[:] = pack_eps_param(epsilon_inv)
        return total_loss, grad, grad_preco

    fcost, grad, grad_preco = evaluate_from_x()
    sotb.set_inputs(
        fcost,
        niter_max=n_epochs_lbfgs,
        conv=plbfgs_conv,
        print_flag=0,
        nls_max=plbfgs_nls_max,
        l=plbfgs_l,
    )
    q_plb = np.zeros(n, dtype=np.float32)

    flag = 0
    eval_count = 0
    safety_max_evals = max(20, n_epochs_lbfgs * 80)
    last_eval_loss = float(fcost)
    last_logged_iter = int(sotb.udf.cpt_iter)

    while flag not in (2, 4):
        flag = sotb.PLBFGS(
            n, x, fcost, grad, grad_preco, q_plb, flag, lb=lb_bounds, ub=ub_bounds
        )
        curr_iter_after = int(sotb.udf.cpt_iter)

        # Log only accepted iterations (cpt_iter increases when a step is accepted).
        if curr_iter_after > last_logged_iter:
            all_losses.append(last_eval_loss)
            print(
                f"  SOTB PLBFGS iter {curr_iter_after}/{n_epochs_lbfgs}  "
                f"Loss={last_eval_loss:.6e}"
            )
            last_logged_iter = curr_iter_after

        if flag == 1:
            fcost, grad, grad_preco = evaluate_from_x()
            eval_count += 1
            last_eval_loss = float(fcost)
        elif flag == 5:
            # SOTB PLBFGS reverse-communication:
            # flag=5 asks user to apply preconditioner to q_plb.
            q_plb[:] = (b0_diag * q_plb).astype(np.float32, copy=False)
            np.nan_to_num(q_plb, copy=False)
            q_plb[air_mask_np] = 0.0
        elif flag not in (2, 3, 4):
            print(f"  SOTB PLBFGS returned unexpected flag={flag}")

        if eval_count >= safety_max_evals:
            print(
                f"  SOTB PLBFGS safety stop after {eval_count} evaluations "
                f"(last flag={flag})"
            )
            break

    unpack_eps_param(x, epsilon_inv)
    with torch.no_grad():
        epsilon_inv.clamp_(1.0, 9.0)
        epsilon_inv[air_mask] = 1.0
    print(f"  SOTB PLBFGS finished with flag={flag}")

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
ax.set_title("Multiscale Cross-correlation Result")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[1, 1]
ax.plot(all_losses, label="MSE loss")
for idx in stage_breaks:
    ax.axvline(idx, color="r", linestyle="--", alpha=0.5)
ax.set_title("Loss Curve (AdamW -> LBFGS stages)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss")
ax.grid(True)
ax.legend()

plt.tight_layout()
final_plot = output_dir / "multiscale_crosscorr_summary.jpg"
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
