import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sotb_wrapper import interface as sotb_interface

import tide


def run_inversion(loss_variant: str = "norm_only") -> None:
    if loss_variant not in {"norm_only", "conv_only"}:
        raise ValueError(f"Unknown loss_variant={loss_variant}")

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

    base_freq = 600e6
    obs_freq = base_freq
    inv_freq = base_freq
    obs_peak = 1.5 / obs_freq
    inv_peak = 1.5 / inv_freq
    base_forward_freq = obs_freq

    filter_specs = {
        "lp400": {"lowpass_mhz": 400, "desc": "600 MHz forward result low-pass to 400 MHz"},
        "lp600": {"lowpass_mhz": 600, "desc": "600 MHz forward result low-pass to 600 MHz"},
    }
    inversion_schedule = [
        {"data_key": "lp400", "adamw_epochs": 40, "lbfgs_epochs": 15},
        {"data_key": "lp600", "adamw_epochs": 30, "lbfgs_epochs": 15},
    ]

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

    lowpass_tag = "-".join(str(spec["lowpass_mhz"]) for spec in filter_specs.values())
    output_dir = Path("outputs") / (
        f"crosscorr_wavelettype_{loss_variant}_"
        f"obsRicker{int(obs_freq / 1e6)}MHz_invGaussD{int(inv_freq / 1e6)}MHz_"
        f"lp{lowpass_tag}_shots{n_shots}_bs{batch_size}_nt{nt}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {output_dir}")

    pde_counts = {"forward": 0.0, "adjoint": 0.0}

    def add_pde_counts(bs: int, forward: bool = False, adjoint: bool = False) -> None:
        if bs <= 0:
            return
        frac = bs / n_shots
        if forward:
            pde_counts["forward"] += frac
        if adjoint:
            pde_counts["adjoint"] += frac

    def report_pde_totals(prefix: str) -> None:
        fwd = pde_counts["forward"]
        adj = pde_counts["adjoint"]
        print(f"{prefix}PDE solves (100 shots = 1): forward {fwd:.2f}, adjoint {adj:.2f}, total {fwd+adj:.2f}")

    def report_pde_delta(prefix: str, f0: float, a0: float) -> None:
        fwd = pde_counts["forward"] - f0
        adj = pde_counts["adjoint"] - a0
        print(f"{prefix}PDE solves: forward {fwd:.2f}, adjoint {adj:.2f}, total {fwd+adj:.2f}")

    def make_shot_batches() -> list[torch.Tensor]:
        perm = torch.arange(n_shots, device=device)
        return [perm[i : i + n_shots_per_batch] for i in range(0, n_shots, n_shots_per_batch)]

    def make_ricker(freq: float, nt_: int, dt_: float, peak_time: float) -> torch.Tensor:
        t = torch.arange(nt_, device=device, dtype=torch.float32) * dt_
        w = np.pi * freq * (t - peak_time)
        return (1.0 - 2.0 * w**2) * torch.exp(-(w**2))

    def make_gaussian_derivative(freq: float, nt_: int, dt_: float, center_time: float) -> torch.Tensor:
        t = torch.arange(nt_, device=device, dtype=torch.float32) * dt_
        sigma = 1.0 / (2.5 * freq)
        x = (t - center_time) / sigma
        w = -x * torch.exp(-0.5 * x**2)
        m = torch.max(torch.abs(w))
        if float(m) > 0.0:
            w = w / m
        return w

    def design_fir_filter(cutoff_hz: float, fs: float, numtaps: int) -> torch.Tensor:
        n = torch.arange(numtaps, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (numtaps - 1))
        sinc = torch.sin(2 * torch.pi * (cutoff_hz / fs) * (n - (numtaps - 1) / 2)) / (
            torch.pi * (n - (numtaps - 1) / 2)
        )
        center = (numtaps - 1) // 2
        sinc[center] = 2 * cutoff_hz / fs
        h = window * sinc
        return h / h.sum()

    def apply_fir_lowpass(data: torch.Tensor, cutoff_hz: float) -> torch.Tensor:
        if cutoff_hz <= 0:
            return data
        fs = 1.0 / dt
        numtaps = max(3, int(fs / cutoff_hz))
        if numtaps % 2 == 0:
            numtaps += 1
        fir_coeff = design_fir_filter(cutoff_hz, fs, numtaps).to(device=data.device, dtype=data.dtype)

        if data.ndim == 1:
            data_2d = data.view(1, 1, -1)
            padded = F.pad(data_2d, (numtaps - 1, 0), mode="reflect")
            filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
            return filtered.view(-1)

        nt_local, n_shots_local, n_rx_local = data.shape
        reshaped = data.permute(1, 2, 0).reshape(-1, 1, nt_local)
        padded = F.pad(reshaped, (numtaps - 1, 0), mode="reflect")
        filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
        return filtered.view(n_shots_local, n_rx_local, nt_local).permute(2, 0, 1)

    def save_wavelet_comparison(wavelet_obs: torch.Tensor, wavelet_inv: torch.Tensor) -> None:
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
        filename = output_dir / "wavelet_original_comparison.jpg"
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved wavelet comparison to '{filename}'")

    def loss_fn(d_obs: torch.Tensor, d_pred: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        if d_obs.dim() == 3 and d_obs.shape[1] == 1:
            d_obs = d_obs.squeeze(1)
            d_pred = d_pred.squeeze(1)
        if d_obs.dim() == 3 and d_pred.dim() == 3:
            d_obs = d_obs.reshape(d_obs.shape[0] * d_obs.shape[1], d_obs.shape[2])
            d_pred = d_pred.reshape(d_pred.shape[0] * d_pred.shape[1], d_pred.shape[2])
        if d_obs.dim() != 2 or d_pred.dim() != 2:
            raise ValueError("Expected d_obs and d_pred with shape [n_traces, nt].")

        if loss_variant == "norm_only":
            d_obs_norm = d_obs / (torch.norm(d_obs) + eps)
            d_pred_norm = d_pred / (torch.norm(d_pred) + eps)
            return -torch.sum(d_obs_norm * d_pred_norm)

        nt_local = d_obs.shape[1]
        stack_obs = d_obs.sum(dim=0)
        stack_pred = d_pred.sum(dim=0)
        d_obs_conv = F.conv1d(
            d_obs.unsqueeze(1),
            stack_pred.view(1, 1, -1).flip([2]),
            padding=nt_local - 1,
        ).squeeze(1)
        d_pred_conv = F.conv1d(
            d_pred.unsqueeze(1),
            stack_obs.view(1, 1, -1).flip([2]),
            padding=nt_local - 1,
        ).squeeze(1)
        return -torch.sum(d_obs_conv * d_pred_conv)

    def save_filter_comparison(observed_base: torch.Tensor, observed_sets: dict) -> None:
        base_np = observed_base.detach().cpu().numpy()[:, :, 0]
        filtered_arrays = []
        for key in filter_specs:
            arr = observed_sets[key]["data"].detach().cpu().numpy()[:, :, 0]
            filtered_arrays.append((arr, observed_sets[key]["desc"]))

        vlim = (-50, 50)
        n_cols = 1 + len(filtered_arrays)
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharex=True, sharey=True)
        if n_cols == 1:
            axes = [axes]
        axes[0].imshow(base_np, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
        axes[0].set_title(f"{base_forward_freq / 1e6:.0f} MHz base")
        axes[0].set_xlabel("Shots")
        axes[0].set_ylabel("Time samples")
        for i, (arr, desc) in enumerate(filtered_arrays, start=1):
            axes[i].imshow(arr, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
            axes[i].set_title(desc)
            axes[i].set_xlabel("Shots")
        plt.tight_layout()
        filename = output_dir / f"data_filter_comparison_base{int(base_forward_freq / 1e6)}_lp{lowpass_tag}.jpg"
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

    def forward_shots(epsilon, sigma, mu, shot_indices, source_amplitude_full, requires_grad=True):
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
            model_gradient_sampling_interval=model_gradient_sampling_interval if requires_grad else 1,
            storage_mode=storage_mode,
            storage_compression=storage_compression,
        )
        return out[-1]

    def generate_base_and_filtered_observed():
        with torch.no_grad():
            wavelet_obs = make_ricker(obs_freq, nt, dt, peak_time=obs_peak)
            wavelet_inv = make_gaussian_derivative(inv_freq, nt, dt, center_time=inv_peak)
            src_amp_obs = wavelet_obs.view(1, 1, nt).repeat(n_shots, 1, 1)
            src_amp_inv = wavelet_inv.view(1, 1, nt).repeat(n_shots, 1, 1)

            obs_list = []
            for shot_indices in make_shot_batches():
                obs_list.append(
                    forward_shots(epsilon_true, sigma_true, mu_true, shot_indices, src_amp_obs, requires_grad=False)
                )
                add_pde_counts(int(shot_indices.numel()), forward=True)
            observed_base = torch.cat(obs_list, dim=1)

            observed_sets = {}
            for key, spec in filter_specs.items():
                lowpass_hz = float(spec["lowpass_mhz"]) * 1e6
                data_filtered = apply_fir_lowpass(observed_base, lowpass_hz) if lowpass_hz > 0 else observed_base
                observed_sets[key] = {"data": data_filtered, "lowpass_hz": lowpass_hz, "desc": spec["desc"]}

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

    observed_raw, observed_sets, wavelet_obs, wavelet_inv, src_amp_inv = generate_base_and_filtered_observed()
    print(f"Observed data modeled at {obs_freq / 1e6:.0f} MHz.")
    print(f"Inversion modeled at {inv_freq / 1e6:.0f} MHz.")
    if sotb_interface is None:
        raise RuntimeError("sotb-wrapper is not importable.")
    print(f"Loss mode: {loss_variant}")
    report_pde_totals("After observed generation: ")
    save_wavelet_comparison(wavelet_obs, wavelet_inv)
    save_filter_comparison(observed_raw, observed_sets)

    vmin_stage = epsilon_true_np.min()
    vmax_stage = epsilon_true_np.max()
    n_param_eps = int(epsilon_inv.numel())
    air_mask_np = air_mask.detach().cpu().numpy().reshape(-1)

    def pack_eps_param(epsilon_param: torch.Tensor) -> np.ndarray:
        return (
            epsilon_param.detach().contiguous().view(-1).to(device="cpu", dtype=torch.float32).numpy().astype(np.float32, copy=False)
        )

    def unpack_eps_param(x: np.ndarray, epsilon_param: torch.Tensor) -> None:
        eps_vec = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            epsilon_param.copy_(eps_vec.view_as(epsilon_param))

    def pack_eps_grad(epsilon_param: torch.Tensor) -> np.ndarray:
        grad = torch.zeros_like(epsilon_param) if epsilon_param.grad is None else epsilon_param.grad
        grad_np = grad.detach().contiguous().view(-1).to(device="cpu", dtype=torch.float32).numpy().astype(np.float32, copy=False)
        np.nan_to_num(grad_np, copy=False)
        return grad_np

    lb_bounds = np.full(n_param_eps, 1.0, dtype=np.float32)
    ub_bounds = np.full(n_param_eps, 9.0, dtype=np.float32)
    lb_bounds[air_mask_np] = 1.0
    ub_bounds[air_mask_np] = 1.0

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

        optimizer_adamw = torch.optim.AdamW([epsilon_inv], lr=0.01, betas=(0.9, 0.99), weight_decay=1e-3)
        for epoch in range(n_epochs_adamw):
            optimizer_adamw.zero_grad()
            epoch_loss = 0.0
            for shot_indices in make_shot_batches():
                syn = forward_shots(epsilon_inv, sigma_fixed, mu_fixed, shot_indices, src_amp_inv, requires_grad=True)
                add_pde_counts(int(shot_indices.numel()), forward=True)
                syn_filtered = apply_fir_lowpass(syn, lowpass_hz)
                obs_batch = observed_filtered[:, shot_indices, :].permute(1, 2, 0)
                syn_batch = syn_filtered.permute(1, 2, 0)
                loss = loss_fn(obs_batch, syn_batch)
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
            print(f"  AdamW epoch {epoch + 1}/{n_epochs_adamw}  Loss={epoch_loss:.6e}")

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
                syn = forward_shots(epsilon_inv, sigma_fixed, mu_fixed, shot_indices, src_amp_inv, requires_grad=True)
                add_pde_counts(int(shot_indices.numel()), forward=True)
                syn_filtered = apply_fir_lowpass(syn, lowpass_hz)
                obs_batch = observed_filtered[:, shot_indices, :].permute(1, 2, 0)
                syn_batch = syn_filtered.permute(1, 2, 0)
                loss = loss_fn(obs_batch, syn_batch)
                loss.backward()
                add_pde_counts(int(shot_indices.numel()), adjoint=True)
                if epsilon_inv.grad is not None:
                    g = torch.nan_to_num(epsilon_inv.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
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
                b0 = (1.0 - plbfgs_precond_blend) + plbfgs_precond_blend * b0
            b0[air_mask_np] = 0.0
            return b0.astype(np.float32, copy=False)

        print("  Building PLBFGS preconditioner (diag GN proxy + smoothing)...")
        b0_diag = build_plbfgs_preconditioner_diag()

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
                syn = forward_shots(epsilon_inv, sigma_fixed, mu_fixed, shot_indices, src_amp_inv, requires_grad=True)
                add_pde_counts(int(shot_indices.numel()), forward=True)
                syn_filtered = apply_fir_lowpass(syn, lowpass_hz)
                obs_batch = observed_filtered[:, shot_indices, :].permute(1, 2, 0)
                syn_batch = syn_filtered.permute(1, 2, 0)
                loss = loss_fn(obs_batch, syn_batch)
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
        sotb.set_inputs(fcost, niter_max=n_epochs_lbfgs, conv=plbfgs_conv, print_flag=0, nls_max=plbfgs_nls_max, l=plbfgs_l)
        q_plb = np.zeros(n, dtype=np.float32)

        flag = 0
        eval_count = 0
        safety_max_evals = max(20, n_epochs_lbfgs * 80)
        last_eval_loss = float(fcost)
        last_logged_iter = int(sotb.udf.cpt_iter)

        while flag not in (2, 4):
            flag = sotb.PLBFGS(n, x, fcost, grad, grad_preco, q_plb, flag, lb=lb_bounds, ub=ub_bounds)
            curr_iter_after = int(sotb.udf.cpt_iter)
            if curr_iter_after > last_logged_iter:
                all_losses.append(last_eval_loss)
                print(f"  SOTB PLBFGS iter {curr_iter_after}/{n_epochs_lbfgs}  Loss={last_eval_loss:.6e}")
                last_logged_iter = curr_iter_after

            if flag == 1:
                fcost, grad, grad_preco = evaluate_from_x()
                eval_count += 1
                last_eval_loss = float(fcost)
            elif flag == 5:
                q_plb[:] = (b0_diag * q_plb).astype(np.float32, copy=False)
                np.nan_to_num(q_plb, copy=False)
                q_plb[air_mask_np] = 0.0
            elif flag not in (2, 3, 4):
                print(f"  SOTB PLBFGS returned unexpected flag={flag}")

            if eval_count >= safety_max_evals:
                print(f"  SOTB PLBFGS safety stop after {eval_count} evaluations (last flag={flag})")
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
    im = axes[0, 0].imshow(eps_true, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("True Model")
    axes[0, 0].set_xlabel("X (grid points)")
    axes[0, 0].set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=axes[0, 0], label="εr")

    im = axes[0, 1].imshow(eps_init, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Initial Model (Smoothed)")
    axes[0, 1].set_xlabel("X (grid points)")
    axes[0, 1].set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=axes[0, 1], label="εr")

    im = axes[1, 0].imshow(eps_result, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Multiscale Result")
    axes[1, 0].set_xlabel("X (grid points)")
    axes[1, 0].set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=axes[1, 0], label="εr")

    axes[1, 1].plot(all_losses, label=f"Loss ({loss_variant})")
    for idx in stage_breaks:
        axes[1, 1].axvline(idx, color="r", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Loss Curve (AdamW -> LBFGS stages)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    final_plot = output_dir / "multiscale_crosscorr_summary.jpg"
    plt.savefig(final_plot, dpi=150)
    print(f"\nResults saved to '{final_plot}'")

    np.save(output_dir / "epsilon_inverted.npy", eps_result)
    print(f"Saved inverted model to '{output_dir / 'epsilon_inverted.npy'}'")

    mask = ~(air_mask.cpu().numpy())
    rms_init = np.sqrt(np.mean((eps_init[mask] - eps_true[mask]) ** 2))
    rms_result = np.sqrt(np.mean((eps_result[mask] - eps_true[mask]) ** 2))
    print(f"RMS Error (Initial):  {rms_init:.4f}")
    print(f"RMS Error (Inverted): {rms_result:.4f}")
    print(f"Improvement: {(1 - rms_result / rms_init) * 100:.1f}%")


if __name__ == "__main__":
    run_inversion("norm_only")
