import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
from sotb_wrapper import interface as sotb_interface

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
batch_size = 8
model_gradient_sampling_interval = 10
compute_precision = os.getenv("TIDE_EXAMPLE_COMPUTE_PRECISION", "default")


# Empirical conductivity model (sigma) from permittivity (epsilon)
sigma_min = 0.0
sigma_k = 1e-4
sigma_p = 2.0
sigma_max = 0.005
sigma_init_scale = 1.0

# SOTB PLBFGS settings (block GN-style preconditioning).
plbfgs_conv = 1e-8
plbfgs_nls_max = 20
plbfgs_l = 5
plbfgs_precond_smooth_sigma = 3.0
plbfgs_precond_damping = 5e-2
plbfgs_precond_power = 0.5
plbfgs_precond_clip_lo = 0.3
plbfgs_precond_clip_hi = 3.0
plbfgs_precond_blend = 0.7
plbfgs_precond_rho_max = 0.8

# Sigma parameter scaling (x stores sigma_hat = sigma / (sigma0 * beta_scale)).
sigma0 = 1.0 / 377.0
beta_scale = 2.0
gamma_sigma = 0.25
sigma_scale = sigma0 * beta_scale

model_path = "examples/data/OverThrust.npy"
epsilon_true_raw = np.load(model_path)
print(f"Loaded model shape: {epsilon_true_raw.shape}")
print(
    f"Permittivity range: {epsilon_true_raw.min():.2f} - {epsilon_true_raw.max():.2f}"
)

ny, nx = epsilon_true_raw.shape
epsilon_true_np = epsilon_true_raw.copy()
epsilon_true_np[:air_layer, :] = 1.0

def sigma_from_epsilon_np(eps: np.ndarray) -> np.ndarray:
    eps_eff = np.maximum(eps - 1.0, 0.0)
    sigma = sigma_min + sigma_k * np.power(eps_eff, sigma_p)
    return np.clip(sigma, 0.0, sigma_max)


sigma_true_np = sigma_from_epsilon_np(epsilon_true_np)
sigma_true_np[:air_layer, :] = 0.0
print(
    f"Sigma range (empirical): {sigma_true_np.min():.2e} - {sigma_true_np.max():.2e}"
)
print("Sigma true model: empirical model")

epsilon_true = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
sigma_true = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)
mu_true = torch.ones_like(epsilon_true)

source_depth = air_layer - 1
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
    {"data_key": "lp250", "lbfgs_epochs": 10},
    {"data_key": "lp500", "lbfgs_epochs": 10},
    {"data_key": "lp700", "lbfgs_epochs": 20},
]

print(f"Base forward frequency: {base_forward_freq / 1e6:.0f} MHz")
print(f"Compute precision: {compute_precision}")
print("FIR low-pass schedule on observed data:")
for key, spec in filter_specs.items():
    print(f"  {key}: {spec['desc']} (cutoff {spec['lowpass_mhz']} MHz)")
print("Inversion schedule:")
for item in inversion_schedule:
    print(f"  {item['data_key']}: PLBFGS {item['lbfgs_epochs']}e")
print("Stage strategy: all stages = joint epsilon + sigma")
print(
    "Sigma empirical model: "
    f"sigma = clamp({sigma_min} + {sigma_k} * max(eps-1,0)^{sigma_p}, 0, {sigma_max})"
)
print(
    f"Sigma parameterization: sigma_hat = sigma / (sigma0*beta), "
    f"sigma0={sigma0:.6e}, beta={beta_scale:.3f}, gamma={gamma_sigma:.3f}"
)

lowpass_tag = "-".join(str(spec["lowpass_mhz"]) for spec in filter_specs.values())
output_dir = Path("outputs") / (
    f"multiscale_fir_joint_eps_sigma_base{int(base_forward_freq / 1e6)}MHz_lp{lowpass_tag}_shots{n_shots}_bs{batch_size}_nt{nt}"
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


def sigma_scaled_to_physical(sigma_scaled: torch.Tensor) -> torch.Tensor:
    return sigma_scaled * sigma_scale


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

    percentile = 90
    abs_percentile = max(
        np.nanpercentile(np.abs(base_np), percentile),
        *(np.nanpercentile(np.abs(arr), percentile) for _, arr, _ in filtered_arrays),
    )
    vlim = (-abs_percentile, abs_percentile)

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
    array: np.ndarray,
    title: str,
    filename: Path,
    vmin: float,
    vmax: float,
    cbar_label: str,
) -> None:

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(array, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label=cbar_label)
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
        compute_precision=compute_precision,
        # storage_mode='cpu'
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

# Build sigma initial model in physical domain first.
sigma_init_base_np = sigma_from_epsilon_np(epsilon_init_np) * sigma_init_scale
sigma_init_base_np = np.clip(sigma_init_base_np, 0.0, sigma_max)
sigma_init_base_np[:air_layer, :] = 0.0

# Apply smoothing after parameter scaling (sigma_hat domain).
sigma_init_scaled_np = sigma_init_base_np / sigma_scale
sigma_init_scaled_np[air_layer:, :] = gaussian_filter(
    sigma_init_scaled_np[air_layer:, :], sigma=sigma_smooth
)
sigma_init_scaled_np = np.maximum(sigma_init_scaled_np, 0.0)
sigma_init_scaled_np[:air_layer, :] = 0.0
sigma_init_np = sigma_init_scaled_np * sigma_scale

epsilon_init = torch.tensor(epsilon_init_np, dtype=torch.float32, device=device)
sigma_init = torch.tensor(sigma_init_np, dtype=torch.float32, device=device)
sigma_init_scaled = torch.tensor(
    sigma_init_scaled_np, dtype=torch.float32, device=device
)

epsilon_inv = epsilon_init.clone().detach()
epsilon_inv.requires_grad_(True)

sigma_fixed = sigma_init_scaled.clone().detach()
sigma_inv = sigma_init_scaled.clone().detach()
sigma_inv.requires_grad_(False)
mu_fixed = torch.ones_like(epsilon_inv)

air_mask = torch.zeros_like(epsilon_inv, dtype=torch.bool)
air_mask[:air_layer, :] = True

eps_min = 1.0
eps_max = 81.0
sigma_floor = 1e-8
sigma_floor_scaled = sigma_floor / sigma_scale
n_param_eps = int(epsilon_inv.numel())
n_param_sigma = int(sigma_inv.numel())
n_param_total = n_param_eps + n_param_sigma
air_mask_np = air_mask.detach().cpu().numpy().reshape(-1)


def project_model_parameters(optimize_sigma: bool) -> None:
    with torch.no_grad():
        epsilon_inv.nan_to_num_(nan=eps_min, posinf=eps_max, neginf=eps_min)
        epsilon_inv.clamp_(eps_min, eps_max)
        epsilon_inv[air_mask] = 1.0
        if optimize_sigma:
            sigma_inv[~torch.isfinite(sigma_inv)] = sigma_floor_scaled
            sigma_inv.clamp_min_(sigma_floor_scaled)
            sigma_inv[air_mask] = 0.0


def pack_params(epsilon_param: torch.Tensor, sigma_param: torch.Tensor) -> np.ndarray:
    eps_np = (
        epsilon_param.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    sigma_np = (
        sigma_param.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    return np.concatenate([eps_np, sigma_np]).astype(np.float32, copy=False)


def unpack_params(
    x: np.ndarray, epsilon_param: torch.Tensor, sigma_param: torch.Tensor
) -> None:
    eps_vec = torch.from_numpy(x[:n_param_eps]).to(device=device, dtype=torch.float32)
    sigma_vec = torch.from_numpy(x[n_param_eps:]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        epsilon_param.copy_(eps_vec.view_as(epsilon_param))
        sigma_param.copy_(sigma_vec.view_as(sigma_param))


def pack_grads(
    epsilon_param: torch.Tensor, sigma_param: torch.Tensor, scale_sigma_grad: float
) -> np.ndarray:
    if epsilon_param.grad is None:
        eps_grad = torch.zeros_like(epsilon_param)
    else:
        eps_grad = epsilon_param.grad

    if sigma_param.grad is None:
        sigma_grad = torch.zeros_like(sigma_param)
    else:
        sigma_grad = sigma_param.grad * scale_sigma_grad

    eps_np = (
        eps_grad.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    sigma_np = (
        sigma_grad.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
    )
    out = np.concatenate([eps_np, sigma_np]).astype(np.float32, copy=False)
    np.nan_to_num(out, copy=False)
    return out


def build_bounds() -> tuple[np.ndarray, np.ndarray]:
    lb = np.empty(n_param_total, dtype=np.float32)
    ub = np.empty(n_param_total, dtype=np.float32)

    lb[:n_param_eps] = eps_min
    ub[:n_param_eps] = eps_max
    lb[:n_param_eps][air_mask_np] = 1.0
    ub[:n_param_eps][air_mask_np] = 1.0

    lb[n_param_eps:] = sigma_floor_scaled
    ub[n_param_eps:] = 1.0e6
    lb[n_param_eps:][air_mask_np] = 0.0
    ub[n_param_eps:][air_mask_np] = 0.0
    return lb, ub


lb_bounds, ub_bounds = build_bounds()


def clear_grads() -> None:
    if epsilon_inv.grad is not None:
        epsilon_inv.grad.zero_()
    if sigma_inv.grad is not None:
        sigma_inv.grad.zero_()


def sanitize_grads() -> None:
    if epsilon_inv.grad is not None:
        epsilon_inv.grad[air_mask] = 0.0
        epsilon_inv.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if sigma_inv.grad is not None:
        sigma_inv.grad[air_mask] = 0.0
        sigma_inv.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)


loss_fn = torch.nn.MSELoss()
all_losses = []
stage_breaks = []

print("Starting multiscale joint epsilon-sigma inversion")
time_start_all = time.time()

print("Generating base observed data once, then FIR filtering...")
observed_raw, observed_sets, src_amp_full = generate_base_and_filtered_observed()
print(f"Base forward modeled at {base_forward_freq / 1e6:.0f} MHz.")
if sotb_interface is None:
    raise RuntimeError(
        "sotb-wrapper is not importable. Install via "
        "`uv pip install --python .venv/bin/python sotb-wrapper`."
    )
print("LBFGS backend: SOTB PLBFGS (block GN-style preconditioner)")
print(
    "PLBFGS preconditioner: "
    f"smooth_sigma={plbfgs_precond_smooth_sigma}, "
    f"damping={plbfgs_precond_damping:.1e}, "
    f"power={plbfgs_precond_power:.2f}, "
    f"clip=[{plbfgs_precond_clip_lo:.2f}, {plbfgs_precond_clip_hi:.2f}], "
    f"blend={plbfgs_precond_blend:.2f}, "
    f"rho_max={plbfgs_precond_rho_max:.2f}"
)
report_pde_totals("After observed generation: ")
save_filter_comparison(observed_raw, observed_sets, output_dir)

vmin_eps = epsilon_true_np.min()
vmax_eps = epsilon_true_np.max()
vmin_sigma = sigma_true_np.min()
vmax_sigma = sigma_true_np.max()

for stage_idx, cfg in enumerate(inversion_schedule, 1):
    data_key = cfg["data_key"]
    obs_cfg = observed_sets[data_key]
    n_epochs_lbfgs = int(cfg["lbfgs_epochs"])
    lowpass_hz = obs_cfg["lowpass_hz"]

    optimize_sigma = True
    sigma_inv.requires_grad_(True)
    sigma_current = sigma_inv
    stage_mode = "joint eps+sigma"
    project_model_parameters(optimize_sigma)

    print(f"\n==== Stage {stage_idx}: {obs_cfg['desc']} ({stage_mode}) ====")
    observed_filtered = obs_cfg["data"]
    stage_forward_start = pde_counts["forward"]
    stage_adjoint_start = pde_counts["adjoint"]

    # SOTB PLBFGS
    sotb = sotb_interface.sotb_wrapper()
    sotb.udf = sotb_interface.UserDefined()
    x = pack_params(epsilon_inv, sigma_inv)
    n = int(x.size)

    valid_spatial_mask = ~air_mask_np

    def build_plbfgs_preconditioner_block() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clear_grads()
        h_ee = torch.zeros_like(epsilon_inv)
        h_ss = torch.zeros_like(sigma_inv)
        h_es = torch.zeros_like(epsilon_inv)

        for shot_indices in make_shot_batches():
            clear_grads()
            syn = forward_shots(
                epsilon_inv,
                sigma_scaled_to_physical(sigma_current),
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

            sanitize_grads()
            if epsilon_inv.grad is not None:
                g_eps = epsilon_inv.grad.detach().clone()
            else:
                g_eps = torch.zeros_like(epsilon_inv)
            if optimize_sigma and sigma_inv.grad is not None:
                g_sigma = (sigma_inv.grad.detach() * gamma_sigma).clone()
            else:
                g_sigma = torch.zeros_like(sigma_inv)

            g_eps[air_mask] = 0.0
            g_sigma[air_mask] = 0.0
            h_ee += g_eps * g_eps
            h_ss += g_sigma * g_sigma
            h_es += g_eps * g_sigma

        h_ee_np = h_ee.detach().cpu().numpy().astype(np.float32, copy=False)
        h_ss_np = h_ss.detach().cpu().numpy().astype(np.float32, copy=False)
        h_es_np = h_es.detach().cpu().numpy().astype(np.float32, copy=False)
        if plbfgs_precond_smooth_sigma > 0:
            h_ee_np = gaussian_filter(h_ee_np, sigma=plbfgs_precond_smooth_sigma)
            h_ss_np = gaussian_filter(h_ss_np, sigma=plbfgs_precond_smooth_sigma)
            h_es_np = gaussian_filter(h_es_np, sigma=plbfgs_precond_smooth_sigma)

        h_ee_flat = h_ee_np.reshape(-1)
        h_ss_flat = h_ss_np.reshape(-1)
        h_es_flat = h_es_np.reshape(-1)
        h_ee_flat[~np.isfinite(h_ee_flat)] = 0.0
        h_ss_flat[~np.isfinite(h_ss_flat)] = 0.0
        h_es_flat[~np.isfinite(h_es_flat)] = 0.0

        h_valid = np.concatenate([h_ee_flat[valid_spatial_mask], h_ss_flat[valid_spatial_mask]])
        if h_valid.size == 0:
            inv11 = np.zeros(n_param_eps, dtype=np.float32)
            inv12 = np.zeros(n_param_eps, dtype=np.float32)
            inv22 = np.zeros(n_param_eps, dtype=np.float32)
            return inv11, inv12, inv22

        scale = float(np.quantile(h_valid, 0.95))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        h_ee_flat = h_ee_flat / scale
        h_ss_flat = h_ss_flat / scale
        h_es_flat = h_es_flat / scale

        h_ee_flat = np.power(h_ee_flat + plbfgs_precond_damping, plbfgs_precond_power)
        h_ss_flat = np.power(h_ss_flat + plbfgs_precond_damping, plbfgs_precond_power)
        h_es_flat = np.sign(h_es_flat) * np.power(
            np.abs(h_es_flat), plbfgs_precond_power
        )

        det = h_ee_flat * h_ss_flat - h_es_flat * h_es_flat
        det_floor = float(np.quantile(det[valid_spatial_mask], 0.02))
        det_floor = max(det_floor, 1e-8)
        det = np.maximum(det, det_floor)

        inv11 = h_ss_flat / det
        inv22 = h_ee_flat / det
        inv12 = -h_es_flat / det
        inv11[~np.isfinite(inv11)] = 0.0
        inv22[~np.isfinite(inv22)] = 0.0
        inv12[~np.isfinite(inv12)] = 0.0

        inv_diag_valid = np.concatenate(
            [inv11[valid_spatial_mask], inv22[valid_spatial_mask]]
        )
        if inv_diag_valid.size > 0:
            med = float(np.median(inv_diag_valid))
            if np.isfinite(med) and med > 0.0:
                inv11 = inv11 / med
                inv22 = inv22 / med
                inv12 = inv12 / med

        np.clip(inv11, plbfgs_precond_clip_lo, plbfgs_precond_clip_hi, out=inv11)
        np.clip(inv22, plbfgs_precond_clip_lo, plbfgs_precond_clip_hi, out=inv22)
        cross_limit = plbfgs_precond_rho_max * np.sqrt(
            np.maximum(inv11 * inv22, 1e-12)
        )
        np.clip(inv12, -cross_limit, cross_limit, out=inv12)

        inv11 = (1.0 - plbfgs_precond_blend) + plbfgs_precond_blend * inv11
        inv22 = (1.0 - plbfgs_precond_blend) + plbfgs_precond_blend * inv22
        inv12 = plbfgs_precond_blend * inv12

        inv11[air_mask_np] = 0.0
        inv22[air_mask_np] = 0.0
        inv12[air_mask_np] = 0.0
        return (
            inv11.astype(np.float32, copy=False),
            inv12.astype(np.float32, copy=False),
            inv22.astype(np.float32, copy=False),
        )

    def apply_block_preconditioner(vec: np.ndarray) -> np.ndarray:
        eps_vec = vec[:n_param_eps]
        sigma_vec = vec[n_param_eps:]
        out_eps = inv11_flat * eps_vec + inv12_flat * sigma_vec
        out_sigma = inv12_flat * eps_vec + inv22_flat * sigma_vec
        out = np.concatenate([out_eps, out_sigma]).astype(np.float32, copy=False)
        out[~np.isfinite(out)] = 0.0
        out[:n_param_eps][air_mask_np] = 0.0
        out[n_param_eps:][air_mask_np] = 0.0
        return out

    print("  Building PLBFGS preconditioner (joint eps+sigma block GN)...")
    inv11_flat, inv12_flat, inv22_flat = build_plbfgs_preconditioner_block()
    inv_diag_valid = np.concatenate(
        [inv11_flat[valid_spatial_mask], inv22_flat[valid_spatial_mask]]
    )
    if inv_diag_valid.size > 0:
        coupling_ratio = np.abs(inv12_flat[valid_spatial_mask]) / np.sqrt(
            np.maximum(inv11_flat[valid_spatial_mask] * inv22_flat[valid_spatial_mask], 1e-12)
        )
        print(
            f"  B0 diag stats (valid): min={inv_diag_valid.min():.3e} "
            f"median={np.median(inv_diag_valid):.3e} max={inv_diag_valid.max():.3e}"
        )
        print(
            f"  B0 coupling |inv12|/sqrt(inv11*inv22): "
            f"median={np.median(coupling_ratio):.3f} p95={np.quantile(coupling_ratio, 0.95):.3f}"
        )

    def evaluate_from_x() -> tuple[float, np.ndarray, np.ndarray, float]:
        unpack_params(x, epsilon_inv, sigma_inv)
        project_model_parameters(optimize_sigma)
        x[:] = pack_params(epsilon_inv, sigma_inv)

        clear_grads()
        total_data_loss = 0.0
        for shot_indices in make_shot_batches():
            syn = forward_shots(
                epsilon_inv,
                sigma_scaled_to_physical(sigma_current),
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
            total_data_loss += float(loss.item())

        sanitize_grads()
        if epsilon_inv.grad is not None:
            valid_grads = epsilon_inv.grad[~air_mask].abs()
            if valid_grads.numel() > 0:
                clip_val = torch.quantile(valid_grads, 0.98)
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val.item())
        if optimize_sigma and sigma_inv.grad is not None:
            valid_sigma = sigma_inv.grad[~air_mask].abs()
            if valid_sigma.numel() > 0:
                clip_val = torch.quantile(valid_sigma, 0.98)
                torch.nn.utils.clip_grad_value_([sigma_inv], clip_val.item())

        grad_vec = pack_grads(epsilon_inv, sigma_inv, scale_sigma_grad=gamma_sigma)
        grad_preco = apply_block_preconditioner(grad_vec)
        x[:] = pack_params(epsilon_inv, sigma_inv)
        return total_data_loss, grad_vec, grad_preco, total_data_loss

    fcost, grad, grad_preco, data_now = evaluate_from_x()
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
    last_eval_data = float(data_now)
    last_logged_iter = int(sotb.udf.cpt_iter)

    while flag not in (2, 4):
        flag = sotb.PLBFGS(
            n, x, fcost, grad, grad_preco, q_plb, flag, lb=lb_bounds, ub=ub_bounds
        )
        curr_iter_after = int(sotb.udf.cpt_iter)

        if curr_iter_after > last_logged_iter:
            all_losses.append(last_eval_loss)
            print(
                f"  SOTB PLBFGS iter {curr_iter_after}/{n_epochs_lbfgs}  "
                f"Loss={last_eval_loss:.6e}  "
                f"Data={last_eval_data:.6e}"
            )
            last_logged_iter = curr_iter_after

        if flag == 1:
            fcost, grad, grad_preco, data_now = evaluate_from_x()
            eval_count += 1
            last_eval_loss = float(fcost)
            last_eval_data = float(data_now)
        elif flag == 5:
            q_plb[:] = apply_block_preconditioner(q_plb)
        elif flag not in (2, 3, 4):
            print(f"  SOTB PLBFGS returned unexpected flag={flag}")

        if eval_count >= safety_max_evals:
            print(
                f"  SOTB PLBFGS safety stop after {eval_count} evaluations "
                f"(last flag={flag})"
            )
            break

    unpack_params(x, epsilon_inv, sigma_inv)
    project_model_parameters(optimize_sigma)
    print(f"  SOTB PLBFGS finished with flag={flag}")

    stage_breaks.append(len(all_losses) - 1)
    report_pde_delta(f"Stage {stage_idx} ", stage_forward_start, stage_adjoint_start)
    eps_stage = epsilon_inv.detach().cpu().numpy()
    sigma_stage = sigma_scaled_to_physical(sigma_inv).detach().cpu().numpy()
    stage_title_eps = f"{obs_cfg['desc']} epsilon inversion result"
    stage_fname_eps = output_dir / f"epsilon_stage_{data_key}.jpg"
    save_model_snapshot(
        eps_stage, stage_title_eps, stage_fname_eps, vmin_eps, vmax_eps, "εr"
    )
    sigma_suffix = "inversion result"
    stage_title_sigma = f"{obs_cfg['desc']} sigma {sigma_suffix}"
    stage_fname_sigma = output_dir / f"sigma_stage_{data_key}.jpg"
    save_model_snapshot(
        sigma_stage,
        stage_title_sigma,
        stage_fname_sigma,
        vmin_sigma,
        vmax_sigma,
        "σ (S/m)",
    )

time_all = time.time() - time_start_all
print(f"\nTotal inversion time: {time_all:.2f}s")
report_pde_totals("Total ")

eps_true = epsilon_true.cpu().numpy()
eps_init = epsilon_init.cpu().numpy()
eps_result = epsilon_inv.detach().cpu().numpy()

sigma_true_arr = sigma_true.cpu().numpy()
sigma_init_arr = sigma_init.cpu().numpy()
sigma_result = sigma_scaled_to_physical(sigma_inv).detach().cpu().numpy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
im = ax.imshow(eps_true, aspect="auto", vmin=vmin_eps, vmax=vmax_eps)
ax.set_title("Epsilon True")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[0, 1]
im = ax.imshow(eps_init, aspect="auto", vmin=vmin_eps, vmax=vmax_eps)
ax.set_title("Epsilon Init (Smoothed)")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[0, 2]
im = ax.imshow(eps_result, aspect="auto", vmin=vmin_eps, vmax=vmax_eps)
ax.set_title("Epsilon Result")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[1, 0]
im = ax.imshow(sigma_true_arr, aspect="auto", vmin=vmin_sigma, vmax=vmax_sigma)
ax.set_title("Sigma True")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="σ (S/m)")

ax = axes[1, 1]
im = ax.imshow(sigma_init_arr, aspect="auto", vmin=vmin_sigma, vmax=vmax_sigma)
ax.set_title("Sigma Init")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="σ (S/m)")

ax = axes[1, 2]
im = ax.imshow(sigma_result, aspect="auto", vmin=vmin_sigma, vmax=vmax_sigma)
ax.set_title("Sigma Result")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="σ (S/m)")

plt.tight_layout()
final_plot = output_dir / "multiscale_joint_eps_sigma_summary.jpg"
plt.savefig(final_plot, dpi=150)
print(f"\nResults saved to '{final_plot}'")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(all_losses, label="Loss")
for idx in stage_breaks:
    ax.axvline(idx, color="r", linestyle="--", alpha=0.5)
ax.set_title("Loss Curve (SOTB PLBFGS stages)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.grid(True)
ax.legend()
plt.tight_layout()
loss_plot = output_dir / "multiscale_joint_eps_sigma_loss.jpg"
plt.savefig(loss_plot, dpi=150)
print(f"Saved loss curve to '{loss_plot}'")

# Save inverted models for metrics computation
np.save(output_dir / "epsilon_inverted.npy", eps_result)
np.save(output_dir / "sigma_inverted.npy", sigma_result)
print(f"Saved inverted model to '{output_dir / 'epsilon_inverted.npy'}'")
print(f"Saved inverted model to '{output_dir / 'sigma_inverted.npy'}'")

mask = ~(air_mask.cpu().numpy())
rms_init_eps = np.sqrt(np.mean((eps_init[mask] - eps_true[mask]) ** 2))
rms_result_eps = np.sqrt(np.mean((eps_result[mask] - eps_true[mask]) ** 2))
rms_init_sigma = np.sqrt(np.mean((sigma_init_arr[mask] - sigma_true_arr[mask]) ** 2))
rms_result_sigma = np.sqrt(
    np.mean((sigma_result[mask] - sigma_true_arr[mask]) ** 2)
)

print(f"RMS Error ε (Initial):  {rms_init_eps:.4f}")
print(f"RMS Error ε (Inverted): {rms_result_eps:.4f}")
print(f"ε Improvement: {(1 - rms_result_eps / rms_init_eps) * 100:.1f}%")
print(f"RMS Error σ (Initial):  {rms_init_sigma:.4e}")
print(f"RMS Error σ (Inverted): {rms_result_sigma:.4e}")
print(f"σ Improvement: {(1 - rms_result_sigma / rms_init_sigma) * 100:.1f}%")


def global_ssim_1d(x: np.ndarray, y: np.ndarray, data_range: float) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.mean((x - mean_x) ** 2))
    var_y = float(np.mean((y - mean_y) ** 2))
    cov_xy = float(np.mean((x - mean_x) * (y - mean_y)))
    denom = (mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2)
    if denom <= 0.0:
        return float("nan")
    return float(((2.0 * mean_x * mean_y + c1) * (2.0 * cov_xy + c2)) / denom)


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    std_x = float(np.std(x))
    std_y = float(np.std(y))
    if std_x == 0.0 or std_y == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_masked_metrics(
    true_arr: np.ndarray, pred_arr: np.ndarray, mask_arr: np.ndarray
) -> dict[str, float]:
    true_vec = true_arr[mask_arr].astype(np.float64)
    pred_vec = pred_arr[mask_arr].astype(np.float64)
    diff = pred_vec - true_vec
    data_range = max(float(np.max(true_vec) - np.min(true_vec)), 1e-12)
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(true_vec) + 1e-12))
    bias = float(np.mean(diff))
    pearson_r = safe_corrcoef(true_vec, pred_vec)
    spearman_res = spearmanr(true_vec, pred_vec)
    spearman_rho = float(spearman_res.statistic)
    ssim = global_ssim_1d(true_vec, pred_vec, data_range)
    return {
        "rmse": rmse,
        "mae": mae,
        "rel_l2": rel_l2,
        "bias": bias,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "ssim": ssim,
    }


eps_metrics_init = compute_masked_metrics(eps_true, eps_init, mask)
eps_metrics_inv = compute_masked_metrics(eps_true, eps_result, mask)
sigma_metrics_init = compute_masked_metrics(sigma_true_arr, sigma_init_arr, mask)
sigma_metrics_inv = compute_masked_metrics(sigma_true_arr, sigma_result, mask)

print("\n=== Additional Metrics (Subsurface / Air Excluded) ===")
print(
    "ε MAE: "
    f"{eps_metrics_init['mae']:.4e} -> {eps_metrics_inv['mae']:.4e} | "
    "RelL2: "
    f"{eps_metrics_init['rel_l2']:.4e} -> {eps_metrics_inv['rel_l2']:.4e}"
)
print(
    "ε Pearson/Spearman: "
    f"{eps_metrics_init['pearson_r']:.4f}/{eps_metrics_init['spearman_rho']:.4f} -> "
    f"{eps_metrics_inv['pearson_r']:.4f}/{eps_metrics_inv['spearman_rho']:.4f}"
)
print(
    f"ε SSIM: {eps_metrics_init['ssim']:.4f} -> {eps_metrics_inv['ssim']:.4f}"
)
print(
    "σ MAE: "
    f"{sigma_metrics_init['mae']:.4e} -> {sigma_metrics_inv['mae']:.4e} | "
    "RelL2: "
    f"{sigma_metrics_init['rel_l2']:.4e} -> {sigma_metrics_inv['rel_l2']:.4e}"
)
print(
    "σ Pearson/Spearman: "
    f"{sigma_metrics_init['pearson_r']:.4f}/{sigma_metrics_init['spearman_rho']:.4f} -> "
    f"{sigma_metrics_inv['pearson_r']:.4f}/{sigma_metrics_inv['spearman_rho']:.4f}"
)
print(
    f"σ SSIM: {sigma_metrics_init['ssim']:.4f} -> {sigma_metrics_inv['ssim']:.4f}"
)
print(
    "σ Bias (pred-true): "
    f"{sigma_metrics_init['bias']:.4e} -> {sigma_metrics_inv['bias']:.4e}"
)

print("\n=== Timing Summary ===")
print(f"Total inversion time: {time_all:.2f}s")
