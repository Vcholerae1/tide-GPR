import os
import sys
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tide


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


dx = _env_float("TIDE_DX", 0.02)
dt = _env_float("TIDE_DT", 4e-11)
nt = _env_int("TIDE_NT", 1500)
pml_width = _env_int("TIDE_PML_WIDTH", 10)
air_layer = _env_int("TIDE_AIR_LAYER", 3)

n_shots = _env_int("TIDE_N_SHOTS", 100)
d_source = _env_int("TIDE_D_SOURCE", 4)
first_source = _env_int("TIDE_FIRST_SOURCE", 0)
n_batch = _env_int("TIDE_N_BATCH", 4)
model_gradient_sampling_interval = _env_int("TIDE_GRAD_INTERVAL", 10)
storage_mode = os.getenv("TIDE_STORAGE_MODE", "device")
storage_compression = os.getenv("TIDE_STORAGE_COMPRESSION", "fp8")
profile_enabled = _env_int("TIDE_PROFILE", 0) > 0

n_batch = max(1, min(n_batch, n_shots))

model_path = "examples/data/OverThrust.npy"
epsilon_true_raw = np.load(model_path)
print(f"Loaded model shape: {epsilon_true_raw.shape}")
print(f"Permittivity range: {epsilon_true_raw.min():.2f} - {epsilon_true_raw.max():.2f}")

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

n_shots_per_batch = (n_shots + n_batch - 1) // n_batch

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

stages_limit = _env_int("TIDE_STAGES", 0)
if stages_limit > 0:
    inversion_schedule = inversion_schedule[:stages_limit]

adamw_override = os.getenv("TIDE_ADAMW_EPOCHS")
lbfgs_override = os.getenv("TIDE_LBFGS_EPOCHS")
if adamw_override is not None or lbfgs_override is not None:
    for entry in inversion_schedule:
        if adamw_override is not None and adamw_override != "":
            entry["adamw_epochs"] = int(adamw_override)
        if lbfgs_override is not None and lbfgs_override != "":
            entry["lbfgs_epochs"] = int(lbfgs_override)

print(f"Base forward frequency: {base_forward_freq/1e6:.0f} MHz")
print(f"Shots: {n_shots}, batches: {n_batch}, shots per batch: {n_shots_per_batch}")
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
    f"multiscale_fir_base{int(base_forward_freq/1e6)}MHz_lp{lowpass_tag}_shots{n_shots}_nb{n_batch}_nt{nt}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving figures to: {output_dir}")


pde_counts = {"forward": 0.0, "adjoint": 0.0}
timers = {"forward": 0.0, "adjoint": 0.0, "filter": 0.0}


def _sync_if_cuda() -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def add_pde_counts(batch_size: int, forward: bool = False, adjoint: bool = False) -> None:
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
    print(f"{prefix}PDE solves (100 shots = 1): {format_pde_counts(pde_counts['forward'], pde_counts['adjoint'])}")


def report_pde_delta(prefix: str, forward_start: float, adjoint_start: float) -> None:
    forward = pde_counts["forward"] - forward_start
    adjoint = pde_counts["adjoint"] - adjoint_start
    print(f"{prefix}PDE solves: {format_pde_counts(forward, adjoint)}")


def _time_block(name: str, fn):
    if not profile_enabled:
        return fn()
    _sync_if_cuda()
    start = time.perf_counter()
    result = fn()
    _sync_if_cuda()
    timers[name] += time.perf_counter() - start
    return result


def report_profile_times() -> None:
    if not profile_enabled:
        return
    total = sum(timers.values())
    if total <= 0:
        return
    print("\n=== Profiling Summary ===")
    for key in ("forward", "adjoint", "filter"):
        value = timers[key]
        pct = value / total * 100.0
        print(f"{key.capitalize():>8}: {value:.2f}s ({pct:4.1f}%)")


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


# Cache for FIR filter coefficients to avoid recomputation
_fir_cache: dict[tuple[float, float, str, str], torch.Tensor] = {}


def fast_percentile_clip(grad: torch.Tensor, mask: torch.Tensor, percentile: float = 0.98) -> float:
    """Approximate percentile-based gradient clipping using sampling.

    Much faster than torch.quantile for large tensors by using random sampling.
    """
    valid_grads = grad[~mask].abs()
    n = valid_grads.numel()
    if n == 0:
        return float("inf")

    # For small tensors, use exact computation
    if n < 10000:
        return torch.quantile(valid_grads, percentile).item()

    # For large tensors, sample and estimate percentile
    sample_size = min(5000, n)
    indices = torch.randperm(n, device=valid_grads.device)[:sample_size]
    sampled = valid_grads[indices]
    return torch.quantile(sampled, percentile).item()


def get_cached_fir_filter(
    cutoff_hz: float, dt: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, int]:
    """Get cached FIR filter coefficients or compute and cache them."""
    cache_key = (cutoff_hz, dt, str(device), str(dtype))
    if cache_key not in _fir_cache:
        fs = 1.0 / dt
        numtaps = max(3, int(fs / cutoff_hz))
        if numtaps % 2 == 0:
            numtaps += 1
        fir_coeff = design_fir_filter(cutoff_hz, fs, numtaps).to(device=device, dtype=dtype)
        _fir_cache[cache_key] = fir_coeff
    fir_coeff = _fir_cache[cache_key]
    return fir_coeff, len(fir_coeff)


def _apply_fir_3d(data: torch.Tensor, kernel: torch.Tensor, numtaps: int) -> torch.Tensor:
    """Core FIR filtering for 3D tensors [nt, n_shots, n_rx]."""
    nt_local, n_shots_local, n_rx_local = data.shape
    reshaped = data.permute(1, 2, 0).reshape(-1, 1, nt_local)
    padded = F.pad(reshaped, (numtaps - 1, 0), mode="reflect")
    filtered = F.conv1d(padded, kernel, padding=0)
    return filtered.view(n_shots_local, n_rx_local, nt_local).permute(2, 0, 1)


# Note: torch.compile is not used for FIR filtering because F.pad with
# mode="reflect" has issues when padding > input size (common for FIR filters)
_apply_fir_3d_compiled = _apply_fir_3d


def apply_fir_lowpass(data: torch.Tensor, dt: float, cutoff_hz: float) -> torch.Tensor:
    """Apply FIR low-pass filter along the time axis to observed/synthetic data."""
    if cutoff_hz <= 0:
        return data

    fir_coeff, numtaps = get_cached_fir_filter(cutoff_hz, dt, data.device, data.dtype)
    kernel = fir_coeff.view(1, 1, -1)

    if data.ndim == 1:
        data_2d = data.view(1, 1, -1)
        padded = F.pad(data_2d, (numtaps - 1, 0), mode="reflect")
        filtered = F.conv1d(padded, kernel, padding=0)
        return filtered.view(-1)

    if data.ndim == 3:
        return _apply_fir_3d_compiled(data, kernel, numtaps)

    raise ValueError(f"Unsupported data dimension: {data.ndim}. Expected 1D or 3D tensor.")


def apply_fir_lowpass_kernel(
    data: torch.Tensor, kernel: torch.Tensor | None, numtaps: int
) -> torch.Tensor:
    """Apply a precomputed FIR kernel; skips filtering if kernel is None."""
    if kernel is None:
        return data
    if data.ndim == 1:
        data_2d = data.view(1, 1, -1)
        padded = F.pad(data_2d, (numtaps - 1, 0), mode="reflect")
        filtered = F.conv1d(padded, kernel, padding=0)
        return filtered.view(-1)
    if data.ndim == 3:
        return _apply_fir_3d_compiled(data, kernel, numtaps)
    raise ValueError(f"Unsupported data dimension: {data.ndim}. Expected 1D or 3D tensor.")


def save_filter_comparison(observed_base: torch.Tensor, observed_sets: dict, output_dir: Path) -> None:
    """Save base vs filtered data comparison figure."""
    base_np = observed_base.detach().cpu().numpy()[:, :, 0]
    filtered_arrays = []
    for key in filter_specs:
        data_np = observed_sets[key]["data"].detach().cpu().numpy()[:, :, 0]
        filtered_arrays.append((key, data_np, observed_sets[key]["desc"]))

    absmax = max(np.abs(base_np).max(), *(np.abs(arr).max() for _, arr, _ in filtered_arrays))
    vlim = (-absmax, absmax)

    n_cols = 1 + len(filtered_arrays)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(base_np, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
    axes[0].set_title(f"{base_forward_freq/1e6:.0f} MHz base")
    axes[0].set_xlabel("Shots")
    axes[0].set_ylabel("Time samples")

    for idx, (_, arr, desc) in enumerate(filtered_arrays, start=1):
        axes[idx].imshow(arr, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
        axes[idx].set_title(desc)
        axes[idx].set_xlabel("Shots")

    plt.tight_layout()
    filename = output_dir / f"data_filter_comparison_base{int(base_forward_freq/1e6)}_lp{lowpass_tag}.jpg"
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved data filter comparison to '{filename}'")


def save_model_snapshot(eps_array: np.ndarray, title: str, filename: Path, vmin: float, vmax: float) -> None:
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


def forward_batch(
    epsilon,
    sigma,
    mu,
    source_amplitude,
    source_location,
    receiver_location,
    requires_grad=True,
):
    out = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=pml_width,
        save_snapshots=requires_grad,
        model_gradient_sampling_interval=model_gradient_sampling_interval if requires_grad else 1,
        storage_mode=storage_mode,
        storage_compression=storage_compression,
    )
    return out[-1]  # [nt, n_batch_shots, 1]


def build_batch_meta(source_amplitude_full: torch.Tensor) -> list[dict]:
    batches = []
    for b in range(n_batch):
        start = b * n_shots_per_batch
        end = min(start + n_shots_per_batch, n_shots)
        batches.append(
            {
                "start": start,
                "end": end,
                "size": end - start,
                "src_amp": source_amplitude_full[start:end],
                "src_loc": source_locations[start:end],
                "rec_loc": receiver_locations[start:end],
            }
        )
    return batches


def generate_base_and_filtered_observed(batch_meta: list[dict]):
    with torch.no_grad():
        obs_list = []
        for batch in batch_meta:
            obs_list.append(
                _time_block(
                    "forward",
                    lambda: forward_batch(
                        epsilon_true,
                        sigma_true,
                        mu_true,
                        batch["src_amp"],
                        batch["src_loc"],
                        batch["rec_loc"],
                        requires_grad=False,
                    ),
                )
            )
            add_pde_counts(batch["size"], forward=True)
        observed_base = torch.cat(obs_list, dim=1)

        observed_sets = {}
        for key, spec in filter_specs.items():
            lowpass_hz = spec["lowpass_mhz"] * 1e6
            data_filtered = (
                _time_block(
                    "filter",
                    lambda: apply_fir_lowpass(observed_base, dt=dt, cutoff_hz=lowpass_hz),
                )
                if lowpass_hz > 0
                else observed_base
            )
            observed_sets[key] = {
                "data": data_filtered,
                "lowpass_hz": lowpass_hz,
                "desc": spec["desc"],
            }

    return observed_base, observed_sets


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
wavelet = tide.ricker(base_forward_freq, nt, dt, peak_time=1.0 / base_forward_freq).to(device)
src_amp_full = wavelet.view(1, 1, nt).repeat(n_shots, 1, 1)
batch_meta = build_batch_meta(src_amp_full)
observed_raw, observed_sets = generate_base_and_filtered_observed(batch_meta)
print(f"Base forward modeled at {base_forward_freq/1e6:.0f} MHz.")
report_pde_totals("After observed generation: ")
save_filter_comparison(observed_raw, observed_sets, output_dir)

vmin_stage = epsilon_true_np.min()
vmax_stage = epsilon_true_np.max()

for stage_idx, cfg in enumerate(inversion_schedule, 1):
    data_key = cfg["data_key"]
    obs_cfg = observed_sets[data_key]
    n_epochs_adamw = cfg["adamw_epochs"]
    n_epochs_lbfgs = cfg["lbfgs_epochs"]
    lowpass_hz = obs_cfg["lowpass_hz"]

    print(f"\n==== Stage {stage_idx}: {obs_cfg['desc']} ====")
    observed_filtered = obs_cfg["data"]

    # Pre-slice observed data for each batch to avoid repeated slicing in loops
    obs_batches = [
        observed_filtered[:, batch["start"]:batch["end"], :]
        for batch in batch_meta
    ]
    if lowpass_hz > 0:
        fir_coeff, numtaps = get_cached_fir_filter(
            lowpass_hz, dt, device, epsilon_inv.dtype
        )
        fir_kernel = fir_coeff.view(1, 1, -1)
    else:
        fir_kernel = None
        numtaps = 0
    stage_forward_start = pde_counts["forward"]
    stage_adjoint_start = pde_counts["adjoint"]

    # Stage 1: AdamW
    optimizer_adamw = torch.optim.AdamW(
        [epsilon_inv], lr=0.01, betas=(0.9, 0.99), weight_decay=1e-3
    )
    for epoch in range(n_epochs_adamw):
        optimizer_adamw.zero_grad()
        epoch_loss = torch.zeros((), device=device)

        for batch, obs in zip(batch_meta, obs_batches):
            syn = _time_block(
                "forward",
                lambda: forward_batch(
                    epsilon_inv,
                    sigma_fixed,
                    mu_fixed,
                    batch["src_amp"],
                    batch["src_loc"],
                    batch["rec_loc"],
                    requires_grad=True,
                ),
            )
            add_pde_counts(batch["size"], forward=True)
            if fir_kernel is None:
                syn_filtered = syn
            else:
                syn_filtered = _time_block(
                    "filter",
                    lambda: apply_fir_lowpass_kernel(syn, fir_kernel, numtaps),
                )

            loss = loss_fn(syn_filtered, obs)
            if profile_enabled:
                _sync_if_cuda()
                start = time.perf_counter()
                loss.backward()
                _sync_if_cuda()
                timers["adjoint"] += time.perf_counter() - start
            else:
                loss.backward()
            add_pde_counts(batch["size"], adjoint=True)
            epoch_loss = epoch_loss + loss.detach()

        if epsilon_inv.grad is not None:
            epsilon_inv.grad[air_mask] = 0.0
            clip_val = fast_percentile_clip(epsilon_inv.grad, air_mask, 0.98)
            if clip_val < float("inf"):
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val)

        optimizer_adamw.step()

        with torch.no_grad():
            epsilon_inv.clamp_(1.0, 9.0)
            epsilon_inv[air_mask] = 1.0

        epoch_loss_value = epoch_loss.item()
        all_losses.append(epoch_loss_value)
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"  AdamW epoch {epoch + 1}/{n_epochs_adamw}  "
                  f"Loss={epoch_loss_value:.6e}")

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
        for batch, obs in zip(batch_meta, obs_batches):
            syn = _time_block(
                "forward",
                lambda: forward_batch(
                    epsilon_inv,
                    sigma_fixed,
                    mu_fixed,
                    batch["src_amp"],
                    batch["src_loc"],
                    batch["rec_loc"],
                    requires_grad=True,
                ),
            )
            add_pde_counts(batch["size"], forward=True)
            if fir_kernel is None:
                syn_filtered = syn
            else:
                syn_filtered = _time_block(
                    "filter",
                    lambda: apply_fir_lowpass_kernel(syn, fir_kernel, numtaps),
                )
            loss = loss_fn(syn_filtered, obs)
            if profile_enabled:
                _sync_if_cuda()
                start = time.perf_counter()
                loss.backward()
                _sync_if_cuda()
                timers["adjoint"] += time.perf_counter() - start
            else:
                loss.backward()
            add_pde_counts(batch["size"], adjoint=True)
            total_loss = total_loss + loss.detach()

        if epsilon_inv.grad is not None:
            epsilon_inv.grad[air_mask] = 0.0
            clip_val = fast_percentile_clip(epsilon_inv.grad, air_mask, 0.98)
            if clip_val < float("inf"):
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val)
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

mask = ~(air_mask.cpu().numpy())
rms_init = np.sqrt(np.mean((eps_init[mask] - eps_true[mask]) ** 2))
rms_result = np.sqrt(np.mean((eps_result[mask] - eps_true[mask]) ** 2))

print(f"RMS Error (Initial):  {rms_init:.4f}")
print(f"RMS Error (Inverted): {rms_result:.4f}")
print(f"Improvement: {(1 - rms_result / rms_init) * 100:.1f}%")

print("\n=== Timing Summary ===")
print(f"Total inversion time: {time_all:.2f}s")
report_profile_times()
