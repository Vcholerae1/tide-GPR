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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dx = 0.01
dt = 1.6e-11
nt = 2000

pml_width = 20

# 模型尺寸
ny, nx = 150, 300
air_layer = 10
# Shots per batch (batch size).
batch_size = 16
model_gradient_sampling_interval = 5

def _plus_mask(
    ny: int, nx: int, cy: int, cx: int, arm_len: int, half_width: int
) -> np.ndarray:
    yy, xx = np.ogrid[:ny, :nx]
    vertical = (np.abs(xx - cx) <= half_width) & (np.abs(yy - cy) <= arm_len)
    horizontal = (np.abs(yy - cy) <= half_width) & (np.abs(xx - cx) <= arm_len)
    return vertical | horizontal


def build_model(ny: int = 200, nx: int = 200) -> tuple[np.ndarray, np.ndarray]:
    # 背景初始化
    eps = np.full((ny, nx), 1.0, dtype=np.float32)
    sigma = np.full((ny, nx), 1e-3, dtype=np.float32)

    # 顶部10层空气
    eps[:10, :] = 1.0
    sigma[:10, :] = 0.0

    # 三条随机不相交曲面（可复现），且越深越平缓
    rng = np.random.default_rng(2026)
    x = np.arange(nx, dtype=np.float32)

    # 第一条曲面（较浅，起伏相对明显）
    s1 = 30.0 + 8.0 * np.sin(2.0 * np.pi * x / nx * 1.2 + 0.7)
    s1 += gaussian_filter(rng.normal(0.0, 1.5, size=nx), sigma=8)
    s1 = np.clip(s1, 18.0, 42.0)

    # 第二条曲面（中间，更平缓）
    s2 = 60.0 + 8.0 * np.sin(2.0 * np.pi * x / nx * 1.2 + 0.7)
    s2 += gaussian_filter(rng.normal(0.0, 1.2, size=nx), sigma=12)
    s2 = np.maximum(s2, s1 + 12.0)
    s2 = np.clip(s2, 48.0, 120.0)

    # 第三条曲面（更深，最平缓）
    s3 = 90.0 + 5 * np.sin(2.0 * np.pi * x / nx * 1.2 + 0.7)
    s3 += gaussian_filter(rng.normal(0.0, 1.0, size=nx), sigma=13)
    s3 = np.maximum(s3, s2 + 14.0)
    s3 = np.clip(s3, 80.0, 110)

    s4 = 120.0 + 5 * np.sin(2.0 * np.pi * x / nx * 1.2 + 0.7)
    s4 += gaussian_filter(rng.normal(0.0, 1.0, size=nx), sigma=16)
    s4 = np.maximum(s4, s3 + 14.0)
    s4 = np.clip(s4, 105.0, ny - 8.0)

    # 分层赋值：air(0-10) | layer1(10-s1) | layer2(s1-s2) | layer3(s2-s3) | layer4(s3-bottom)
    yy = np.arange(ny, dtype=np.float32)[:, None]
    s1_2d = s1[None, :]
    s2_2d = s2[None, :]
    s3_2d = s3[None, :]
    s4_2d = s4[None, :]
    layer1 = (yy >= 10.0) & (yy < s1_2d)
    layer2 = (yy >= s1_2d) & (yy < s2_2d)
    layer3 = (yy >= s2_2d) & (yy < s3_2d)
    layer4 = (yy >= s3_2d) & (yy < s4_2d)
    layer5 = yy >= s4_2d

    eps[layer1] = 4.0
    sigma[layer1] = 1e-3
    eps[layer2] = 7.0
    sigma[layer2] = 1e-3
    eps[layer3] = 10.0
    sigma[layer3] = 1e-3
    eps[layer4] = 13.0
    sigma[layer4] = 1e-3
    eps[layer5] = 17.0
    sigma[layer5] = 1e-3

    return eps, sigma





def make_ricker(freq: float, nt: int, dt: float, device: torch.device) -> torch.Tensor:
    t = torch.arange(nt, device=device) * dt
    t0 = 1.2 / freq
    w = np.pi * freq * (t - t0)
    return (1.0 - 2.0 * w**2) * torch.exp(-(w**2))


epsilon_true_np, sigma_true_np = build_model(ny=ny, nx=nx)
print(f"Loaded cross-anomaly model shape: {epsilon_true_np.shape}")
print(
    f"Permittivity range: {epsilon_true_np.min():.2f} - {epsilon_true_np.max():.2f}"
)

epsilon_true = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
sigma_true = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)
mu_true = torch.ones_like(epsilon_true)

base_forward_freq = 500e6
# 地面观测：源和接收点均放在地表（空气层下边界）
n_shots = 100
n_receivers = 100
eps_smooth = 20
# 源沿测线分布（地表：空气层底部）
src_y = air_layer - 1
source_x_positions = torch.linspace(2, nx-5, n_shots, dtype=torch.long, device=device)

source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
source_locations[:, 0, 0] = src_y
source_locations[:, 0, 1] = source_x_positions  # 每炮源位置不同

# 接收线：所有炮共享一条接收线
receiver_x_positions = torch.linspace(4, nx-3, n_receivers, dtype=torch.long, device=device)

receiver_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
receiver_locations[:, 0, 0] = src_y
receiver_locations[:, 0, 1] = receiver_x_positions

print(f"source_locations shape (surface): {tuple(source_locations.shape)}")
print(f"receiver_locations shape (common): {tuple(receiver_locations.shape)}")

n_shots_per_batch = batch_size

# SOTB PLBFGS settings (与 example_multiscale_filtered.py 一致风格)
plbfgs_conv = 1e-8
plbfgs_nls_max = 20
plbfgs_l = 5
plbfgs_precond_smooth_sigma = 3.0
plbfgs_precond_damping = 5e-2
plbfgs_precond_power = 0.5
plbfgs_precond_clip_lo = 0.3
plbfgs_precond_clip_hi = 3.0
plbfgs_precond_blend = 0.7

output_dir = Path("outputs") / (
    f"multi_wave_common_{int(base_forward_freq / 1e6)}MHz_shots{n_shots}_bs{batch_size}_nt{nt}"
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
        f"{prefix}PDE solves ({n_shots} shots = 1): {format_pde_counts(pde_counts['forward'], pde_counts['adjoint'])}"
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
    )
    return out[-1]  # [nt, shots_in_batch, n_receivers]


def generate_observed_data():
    with torch.no_grad():
        wavelet = make_ricker(base_forward_freq, nt, dt, device)
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
        observed_data = torch.cat(obs_list, dim=1)

    return observed_data, src_amp_full



epsilon_init_raw = gaussian_filter(epsilon_true_np, sigma=eps_smooth)
epsilon_init_np = epsilon_init_raw.copy()
sigma_init_np = sigma_true_np.copy()

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


def save_six_models_plot(
    eps_true_arr: np.ndarray,
    sig_true_arr: np.ndarray,
    eps_init_arr: np.ndarray,
    sig_init_arr: np.ndarray,
    eps_inv_arr: np.ndarray,
    sig_inv_arr: np.ndarray,
    out_path: Path,
) -> None:
    eps_vmin = float(eps_true_arr.min())
    eps_vmax = float(eps_true_arr.max())
    sig_vmin = 0.0
    sig_vmax = 10.0e-3

    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)

    ax = axes[0, 0]
    im = ax.imshow(
        eps_true_arr, aspect="auto", vmin=eps_vmin, vmax=eps_vmax, cmap="jet"
    )
    ax.set_title("True εr")
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="εr")

    ax = axes[0, 1]
    im = ax.imshow(
        sig_true_arr, aspect="auto", vmin=sig_vmin, vmax=sig_vmax, cmap="jet"
    )
    ax.set_title("True σ")
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="σ (S/m)")

    ax = axes[1, 0]
    im = ax.imshow(
        eps_init_arr, aspect="auto", vmin=eps_vmin, vmax=eps_vmax, cmap="jet"
    )
    ax.set_title("Initial εr")
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="εr")

    ax = axes[1, 1]
    im = ax.imshow(
        sig_init_arr, aspect="auto", vmin=sig_vmin, vmax=sig_vmax, cmap="jet"
    )
    ax.set_title("Initial σ")
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="σ (S/m)")

    ax = axes[2, 0]
    im = ax.imshow(
        eps_inv_arr, aspect="auto", vmin=eps_vmin, vmax=eps_vmax, cmap="jet"
    )
    ax.set_title("Inverted εr")
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="εr")

    ax = axes[2, 1]
    im = ax.imshow(
        sig_inv_arr, aspect="auto", vmin=sig_vmin, vmax=sig_vmax, cmap="jet"
    )
    ax.set_title("Inverted σ")
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="σ (S/m)")

    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Starting direct-data inversion (epsilon only)")
time_start_all = time.time()

print("Generating observed data by forward modeling...")
observed_raw, src_amp_full = generate_observed_data()
print(f"Observed data generated at {base_forward_freq / 1e6:.0f} MHz.")
print("Loss mode: mse")
report_pde_totals("After observed generation: ")
print("Inversion config: direct observed data, no filtering")
print("LBFGS backend: SOTB PLBFGS (diagonal GN-style preconditioner)")
print(
    "PLBFGS preconditioner: "
    f"smooth_sigma={plbfgs_precond_smooth_sigma}, "
    f"damping={plbfgs_precond_damping:.1e}, "
    f"power={plbfgs_precond_power:.2f}, "
    f"clip=[{plbfgs_precond_clip_lo:.2f}, {plbfgs_precond_clip_hi:.2f}], "
    f"blend={plbfgs_precond_blend:.2f}"
)



n_epochs = 40
n_param_eps = int(epsilon_inv.numel())
n_param_total = n_param_eps
air_mask_np = air_mask.detach().cpu().numpy().reshape(-1)


def pack_param(eps_param: torch.Tensor) -> np.ndarray:
    eps_np = (
        eps_param.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .astype(np.float32, copy=False)
    )
    return eps_np.astype(np.float32, copy=False)


def unpack_param(x: np.ndarray, eps_param: torch.Tensor) -> None:
    eps_vec = torch.from_numpy(x).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        eps_param.copy_(eps_vec.view_as(eps_param))


def pack_grad(eps_param: torch.Tensor) -> np.ndarray:
    if eps_param.grad is None:
        grad_eps = torch.zeros_like(eps_param)
    else:
        grad_eps = eps_param.grad

    g_eps_np = (
        grad_eps.detach()
        .contiguous()
        .view(-1)
        .to(device="cpu", dtype=torch.float32)
        .numpy()
        .astype(np.float32, copy=False)
    )
    grad = g_eps_np.astype(np.float32, copy=False)
    np.nan_to_num(grad, copy=False)
    return grad


def build_bounds() -> tuple[np.ndarray, np.ndarray]:
    lb_eps = np.full(n_param_eps, 1.0, dtype=np.float32)
    ub_eps = np.full(n_param_eps, 20.0, dtype=np.float32)
    lb_eps[air_mask_np] = 1.0
    ub_eps[air_mask_np] = 1.0
    lb = lb_eps.astype(np.float32, copy=False)
    ub = ub_eps.astype(np.float32, copy=False)
    return lb, ub


lb_bounds, ub_bounds = build_bounds()

stage_forward_start = pde_counts["forward"]
stage_adjoint_start = pde_counts["adjoint"]

if sotb_interface is None:
    raise RuntimeError(
        "sotb-wrapper is not importable. Install via "
        "`uv pip install --python .venv/bin/python sotb-wrapper`."
    )

sotb = sotb_interface.sotb_wrapper()
sotb.udf = sotb_interface.UserDefined()
x = pack_param(epsilon_inv)


def build_plbfgs_preconditioner_diag() -> np.ndarray:
    if epsilon_inv.grad is not None:
        epsilon_inv.grad.zero_()

    p_diag_eps = torch.zeros_like(epsilon_inv)

    for shot_indices in make_shot_batches():
        if epsilon_inv.grad is not None:
            epsilon_inv.grad.zero_()

        syn = forward_shots(
            epsilon_inv,
            sigma_fixed,
            mu_fixed,
            shot_indices,
            src_amp_full,
            requires_grad=True,
        )
        add_pde_counts(int(shot_indices.numel()), forward=True)
        obs_batch = observed_raw[:, shot_indices, :]
        loss = F.mse_loss(syn, obs_batch)
        loss.backward()
        add_pde_counts(int(shot_indices.numel()), adjoint=True)

        if epsilon_inv.grad is not None:
            g_eps = torch.nan_to_num(
                epsilon_inv.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            g_eps = g_eps.clone()
            g_eps[air_mask] = 0.0
            p_diag_eps += g_eps * g_eps

    p_eps_np = p_diag_eps.detach().cpu().numpy().astype(np.float32, copy=False)
    if plbfgs_precond_smooth_sigma > 0:
        p_eps_np = gaussian_filter(p_eps_np, sigma=plbfgs_precond_smooth_sigma)

    def _build_block_b0(p_block: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        p_flat = p_block.reshape(-1).astype(np.float32, copy=False)
        p_flat[~np.isfinite(p_flat)] = 0.0
        valid = ~mask_np
        p_valid = p_flat[valid]
        if p_valid.size == 0:
            b0 = np.ones_like(p_flat, dtype=np.float32)
            b0[~valid] = 0.0
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

        b0[~valid] = 0.0
        return b0.astype(np.float32, copy=False)

    b0_eps = _build_block_b0(p_eps_np, air_mask_np)
    return b0_eps.astype(np.float32, copy=False)


print("  Building PLBFGS preconditioner (diag GN proxy + smoothing)...")
b0_diag = build_plbfgs_preconditioner_diag()
b0_valid = b0_diag[np.isfinite(b0_diag) & (b0_diag > 0)]
if b0_valid.size > 0:
    print(
        f"  B0 stats: min={b0_valid.min():.3e} "
        f"median={np.median(b0_valid):.3e} max={b0_valid.max():.3e}"
    )


def evaluate_from_x() -> tuple[float, np.ndarray, np.ndarray]:
    unpack_param(x, epsilon_inv)
    with torch.no_grad():
        epsilon_inv.clamp_(1.0, 20.0)
        epsilon_inv[air_mask] = 1.0
    x[:] = pack_param(epsilon_inv)

    if epsilon_inv.grad is not None:
        epsilon_inv.grad.zero_()

    total_loss = 0.0
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
        obs_batch = observed_raw[:, shot_indices, :]
        loss = F.mse_loss(syn, obs_batch)
        loss.backward()
        add_pde_counts(int(shot_indices.numel()), adjoint=True)
        total_loss += float(loss.item())

    if epsilon_inv.grad is not None:
        epsilon_inv.grad[air_mask] = 0.0
        epsilon_inv.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    grad = pack_grad(epsilon_inv)
    grad_preco = (b0_diag * grad).astype(np.float32, copy=False)
    np.nan_to_num(grad_preco, copy=False)
    x[:] = pack_param(epsilon_inv)
    return total_loss, grad, grad_preco


fcost, grad, grad_preco = evaluate_from_x()
sotb.set_inputs(
    fcost,
    niter_max=n_epochs,
    conv=plbfgs_conv,
    print_flag=0,
    nls_max=plbfgs_nls_max,
    l=plbfgs_l,
)
q_plb = np.zeros(n_param_total, dtype=np.float32)

flag = 0
eval_count = 0
safety_max_evals = max(20, n_epochs * 80)
last_eval_loss = float(fcost)
last_logged_iter = int(sotb.udf.cpt_iter)

while flag not in (2, 4):
    flag = sotb.PLBFGS(
        n_param_total,
        x,
        fcost,
        grad,
        grad_preco,
        q_plb,
        flag,
        lb=lb_bounds,
        ub=ub_bounds,
    )
    curr_iter_after = int(sotb.udf.cpt_iter)

    if curr_iter_after > last_logged_iter:
        all_losses.append(last_eval_loss)
        print(
            f"  SOTB PLBFGS iter {curr_iter_after}/{n_epochs}  "
            f"Loss={last_eval_loss:.6e}"
        )
        if curr_iter_after % 10 == 0:
            save_six_models_plot(
                eps_true_arr=epsilon_true_np,
                sig_true_arr=sigma_true_np,
                eps_init_arr=epsilon_init_np,
                sig_init_arr=sigma_init_np,
                eps_inv_arr=epsilon_inv.detach().cpu().numpy(),
                sig_inv_arr=sigma_fixed.detach().cpu().numpy(),
                out_path=output_dir / f"models_eps_sigma_epoch_{curr_iter_after:04d}.jpg",
            )
        last_logged_iter = curr_iter_after

    if flag == 1:
        fcost, grad, grad_preco = evaluate_from_x()
        eval_count += 1
        last_eval_loss = float(fcost)
    elif flag == 5:
        q_plb[:] = (b0_diag * q_plb).astype(np.float32, copy=False)
        np.nan_to_num(q_plb, copy=False)
    elif flag not in (2, 3, 4):
        print(f"  SOTB PLBFGS returned unexpected flag={flag}")

    if eval_count >= safety_max_evals:
        print(
            f"  SOTB PLBFGS safety stop after {eval_count} evaluations "
            f"(last flag={flag})"
        )
        break

unpack_param(x, epsilon_inv)
with torch.no_grad():
    epsilon_inv.clamp_(1.0, 20.0)
    epsilon_inv[air_mask] = 1.0
print(f"  SOTB PLBFGS finished with flag={flag}")

stage_breaks.append(len(all_losses) - 1)
report_pde_delta("Inversion ", stage_forward_start, stage_adjoint_start)

time_all = time.time() - time_start_all
print(f"\nTotal inversion time: {time_all:.2f}s")
report_pde_totals("Total ")

eps_true = epsilon_true.cpu().numpy()
eps_init = epsilon_init.cpu().numpy()
eps_result = epsilon_inv.detach().cpu().numpy()
sig_true = sigma_true.cpu().numpy()
sig_init = sigma_init.cpu().numpy()
sig_result = sigma_fixed.detach().cpu().numpy()

models_plot = output_dir / "models_eps_sigma_true_init_inv.jpg"
save_six_models_plot(
    eps_true_arr=eps_true,
    sig_true_arr=sig_true,
    eps_init_arr=eps_init,
    sig_init_arr=sig_init,
    eps_inv_arr=eps_result,
    sig_inv_arr=sig_result,
    out_path=models_plot,
)

# 图2：正演数据（多偏移距） + loss曲线

# 图2：正演数据 + loss曲线
obs_np = observed_raw.detach().cpu().numpy()
mono_bscan = obs_np[:, :, 0]
vlim = max(np.percentile(np.abs(mono_bscan), 99.0), 1e-12)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
im = axes2[0].imshow(
    mono_bscan,
    cmap="seismic",
    vmin=-vlim,
    vmax=vlim,
    aspect="auto",
    origin="upper",
    extent=(0.0, float(n_shots - 1), (nt - 1) * dt * 1e9, 0.0),
)
axes2[0].set_title("Forward data (monostatic B-scan)")
axes2[0].set_xlabel("Shot index")
axes2[0].set_ylabel("Time (ns)")
fig2.colorbar(im, ax=axes2[0], label="Ey (V/m)")

axes2[1].semilogy(all_losses, "b-", linewidth=1.6)
axes2[1].set_title("Loss curve")
axes2[1].set_xlabel("Epoch")
axes2[1].set_ylabel("Loss")
axes2[1].grid(True)

data_loss_plot = output_dir / "forward_data_and_loss.jpg"
plt.savefig(data_loss_plot, dpi=160)
plt.close(fig2)
print(f"Saved: {data_loss_plot}")

# 保存可复现实验数据（六个模型 + loss + 正演数据 + 几何与参数）
bundle_path = output_dir / "repro_bundle_eps_sigma_loss.npz"
np.savez_compressed(
    bundle_path,
    eps_true=eps_true,
    sig_true=sig_true,
    eps_init=eps_init,
    sig_init=sig_init,
    eps_inv=eps_result,
    sig_inv=sig_result,
    loss=np.asarray(all_losses, dtype=np.float64),
    observed=obs_np,
    mono_bscan=mono_bscan,
    dt=dt,
    dx=dx,
    nt=nt,
    freq=base_forward_freq,
    n_shots=n_shots,
    n_receivers=n_receivers,
    pml_width=pml_width,
)
print(f"Saved reproducible bundle: {bundle_path}")

mask = ~(air_mask.cpu().numpy())
rms_init = np.sqrt(np.mean((eps_init[mask] - eps_true[mask]) ** 2))
rms_result = np.sqrt(np.mean((eps_result[mask] - eps_true[mask]) ** 2))
rms_sigma_init = np.sqrt(np.mean((sig_init[mask] - sig_true[mask]) ** 2))
rms_sigma_result = np.sqrt(np.mean((sig_result[mask] - sig_true[mask]) ** 2))

print(f"RMS Error εr (Initial):  {rms_init:.4f}")
print(f"RMS Error εr (Inverted): {rms_result:.4f}")
print(f"Improvement εr: {(1 - rms_result / rms_init) * 100:.1f}%")
print(f"RMS Error σ (Initial):  {rms_sigma_init:.6e}")
print(f"RMS Error σ (Inverted): {rms_sigma_result:.6e}")
if rms_sigma_init > 0:
    print(f"Improvement σ: {(1 - rms_sigma_result / rms_sigma_init) * 100:.1f}%")
else:
    print("Improvement σ: N/A (sigma fixed)")

print("\n=== Timing Summary ===")
print(f"Total inversion time: {time_all:.2f}s")
