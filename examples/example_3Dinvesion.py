from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import tide
try:
    from sotb_wrapper import interface as sotb_interface
except Exception:
    sotb_interface = None

THIS_DIR = Path(__file__).resolve().parent
THREED_DIR = (THIS_DIR.parent / "3D_forward").resolve()
if str(THREED_DIR) not in sys.path:
    sys.path.insert(0, str(THREED_DIR))

from hydro_forward_lines import (
    build_line_geometry,
    estimate_stable_dt,
    load_first_numeric_h5_dataset,
    make_ricker,
    resolve_device,
    to_solver_zyx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D多频带反演（固定sigma，仅反演epsilon）")
    parser.add_argument("--eps-path", type=str, default="../3D_forward/data/hyd_3D_model_1_resized.h5", help="真值epsilon模型H5路径")
    parser.add_argument("--eps-layout", type=str, default="yxz", choices=["zyx", "yxz"], help="epsilon数据轴顺序")

    parser.add_argument("--air-layers", type=int, default=5, help="空气层厚度（网格点）")
    parser.add_argument("--air-eps", type=float, default=1.0, help="空气层epsilon")
    parser.add_argument("--sigma-mean", type=float, default=1e-3, help="固定sigma常值")
    parser.add_argument("--ds", type=float, default=0.02, help="网格间距(m)")
    parser.add_argument("--nt", type=int, default=1200, help="时间采样点数")
    parser.add_argument("--pml-width", type=int, default=20, help="PML厚度")
    parser.add_argument("--stencil", type=int, default=4, choices=[2, 4, 6, 8], help="差分阶数")
    parser.add_argument("--dt", type=float, default=None, help="时间步长(s)，默认按CFL")
    parser.add_argument("--dt-safety", type=float, default=0.8, help="CFL安全系数")

    parser.add_argument("--n-lines", type=int, default=12, help="测线数量")
    parser.add_argument("--sources-per-line", type=int, default=12, help="每线激励源数")
    parser.add_argument("--receivers-per-line", type=int, default=70, help="每线接收点数")
    parser.add_argument("--edge-margin", type=int, default=2, help="边界保留网格")
    parser.add_argument("--shot-batch-size", type=int, default=1, help="反演时shot batch")
    parser.add_argument(
        "--model-gradient-sampling-interval",
        type=int,
        default=20,
        help="梯度时间采样间隔（用于反传相关正演）",
    )

    parser.add_argument("--base-freq", type=float, default=9e8, help="观测数据正演基频(Hz)")
    parser.add_argument(
        "--bands-mhz",
        type=str,
        default="400,600,900",
        help="多频带低通频率(MHz)，逗号分隔",
    )
    parser.add_argument(
        "--epochs-per-band",
        type=str,
        default="10,10,10",
        help="每个频带的PLBFGS迭代数，逗号分隔；可填一个数复用到全部频带",
    )
    parser.add_argument("--plbfgs-conv", type=float, default=1e-8, help="SOTB PLBFGS收敛阈值")
    parser.add_argument("--plbfgs-nls-max", type=int, default=20, help="SOTB PLBFGS线搜索最大次数")
    parser.add_argument("--plbfgs-l", type=int, default=5, help="SOTB PLBFGS历史长度")
    parser.add_argument("--eps-lb", type=float, default=1.0, help="epsilon下界")
    parser.add_argument("--eps-ub", type=float, default=25.0, help="epsilon上界")
    parser.add_argument("--init-smooth-sigma", type=float, default=3.0, help="初始模型高斯平滑sigma")
    parser.add_argument("--plot-interval", type=int, default=10, help="每隔多少个epoch绘制一次切片对比图；<=0表示关闭")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认自动按 bands-mhz 命名）",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        band_tag = "-".join(s.strip() for s in args.bands_mhz.split(",") if s.strip())
        smooth_tag = str(float(args.init_smooth_sigma)).replace(".", "p")
        args.output_dir = f"lbc_hydro_lp{band_tag}MHz_smooth{smooth_tag}"
    
    return args


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty list is not allowed.")
    return vals


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty list is not allowed.")
    if any(v <= 0 for v in vals):
        raise ValueError("All values must be positive integers.")
    return vals


def design_fir_filter(cutoff_hz: float, fs: float, numtaps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(numtaps, dtype=dtype, device=device)
    window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (numtaps - 1))
    x = n - (numtaps - 1) / 2
    sinc = torch.sin(2 * torch.pi * (cutoff_hz / fs) * x) / (torch.pi * x)
    sinc[(numtaps - 1) // 2] = 2 * cutoff_hz / fs
    h = window * sinc
    return h / h.sum()


def apply_fir_lowpass(data: torch.Tensor, dt: float, cutoff_hz: float) -> torch.Tensor:
    if cutoff_hz <= 0.0:
        return data
    if data.ndim != 3:
        raise ValueError(f"Expected [nt, n_shots, n_receivers], got {tuple(data.shape)}")

    fs = 1.0 / dt
    numtaps = max(3, int(fs / cutoff_hz))
    if numtaps % 2 == 0:
        numtaps += 1
    fir = design_fir_filter(cutoff_hz, fs, numtaps, device=data.device, dtype=data.dtype)

    nt, n_shots, n_rec = data.shape
    reshaped = data.permute(1, 2, 0).reshape(-1, 1, nt)
    padded = F.pad(reshaped, (numtaps - 1, 0), mode="reflect")
    filt = F.conv1d(padded, fir.view(1, 1, -1), padding=0)
    return filt.view(n_shots, n_rec, nt).permute(2, 0, 1).contiguous()


def forward_batched(
    *,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    source_amplitude_full: torch.Tensor,
    ds: float,
    dt: float,
    nt: int,
    pml_width: int,
    stencil: int,
    shot_batch_size: int,
    requires_grad: bool,
    model_gradient_sampling_interval: int,
) -> torch.Tensor:
    n_shots = source_points.shape[0]
    batch = max(1, int(shot_batch_size))

    # 部分 tide 后端会根据输入张量的 requires_grad 标志选择 Autograd Function。
    # 在纯推理/评估阶段（requires_grad=False）必须确保输入为 detached 张量，
    # 否则可能触发："Maxwell3DForwardFunc should only be used when gradients are required."。
    epsilon_run = epsilon if requires_grad else epsilon.detach()
    sigma_run = sigma if requires_grad else sigma.detach()
    mu_run = mu if requires_grad else mu.detach()

    rec_all = torch.from_numpy(receiver_points).to(device=epsilon.device, dtype=torch.long)
    rec_all = rec_all.unsqueeze(0).repeat(batch, 1, 1)

    out_list: list[torch.Tensor] = []
    for s0 in range(0, n_shots, batch):
        s1 = min(s0 + batch, n_shots)
        b = s1 - s0

        src_batch = torch.from_numpy(source_points[s0:s1]).to(device=epsilon.device, dtype=torch.long).unsqueeze(1)
        rec_batch = rec_all[:b]
        amp_batch = source_amplitude_full[s0:s1]

        out_b = tide.maxwell3d(
            epsilon=epsilon_run,
            sigma=sigma_run,
            mu=mu_run,
            grid_spacing=[ds, ds, ds],
            dt=dt,
            source_amplitude=amp_batch,
            source_location=src_batch,
            receiver_location=rec_batch,
            pml_width=pml_width,
            stencil=stencil,
            source_component="ey",
            receiver_component="ey",
            save_snapshots=requires_grad,
            model_gradient_sampling_interval=(
                int(model_gradient_sampling_interval) if requires_grad else 1
            ),
       
        )
        out_list.append(out_b[-1])

        if not requires_grad:
            del out_b
            if epsilon.device.type == "cuda":
                torch.cuda.empty_cache()

    return torch.cat(out_list, dim=1)


def forward_single_batch(
    *,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    source_amplitude_full: torch.Tensor,
    ds: float,
    dt: float,
    pml_width: int,
    stencil: int,
    model_gradient_sampling_interval: int,
    s0: int,
    s1: int,
) -> torch.Tensor:
    b = s1 - s0
    src_batch = torch.from_numpy(source_points[s0:s1]).to(device=epsilon.device, dtype=torch.long).unsqueeze(1)
    rec_batch = torch.from_numpy(receiver_points).to(device=epsilon.device, dtype=torch.long).unsqueeze(0).repeat(b, 1, 1)
    amp_batch = source_amplitude_full[s0:s1]

    out_b = tide.maxwell3d(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=[ds, ds, ds],
        dt=dt,
        source_amplitude=amp_batch,
        source_location=src_batch,
        receiver_location=rec_batch,
        pml_width=pml_width,
        stencil=stencil,
        source_component="ey",
        receiver_component="ey",
        save_snapshots=True,
        model_gradient_sampling_interval=int(model_gradient_sampling_interval),
       
    )
    return out_b[-1]


def save_stage_h5(
    out_h5: Path,
    epsilon_inv: np.ndarray,
    sigma_fixed: np.ndarray,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    record_syn: np.ndarray,
    ds: float,
    dt: float,
    cutoff_mhz: float,
) -> None:
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("epsilon_inverted", data=epsilon_inv, compression="gzip", compression_opts=4)
        f.create_dataset("sigma_fixed", data=sigma_fixed, compression="gzip", compression_opts=4)
        f.create_dataset("source_points_zyx", data=source_points, compression="gzip", compression_opts=4)
        f.create_dataset("receiver_points_zyx", data=receiver_points, compression="gzip", compression_opts=4)
        f.create_dataset("synthetic_stage", data=record_syn, compression="gzip", compression_opts=4)
        f.attrs["grid_spacing_m"] = ds
        f.attrs["dt_s"] = dt
        f.attrs["stage_lowpass_mhz"] = cutoff_mhz


def plot_model_translucent(
    out_png: Path,
    epsilon: np.ndarray,
    ds: float,
    air_layers: int,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    title: str,
) -> None:
    nz, ny, nx = epsilon.shape

    x = np.arange(nx, dtype=np.float64) * ds
    y = np.arange(ny, dtype=np.float64) * ds
    z = np.arange(nz, dtype=np.float64) * ds

    xx_top, yy_top = np.meshgrid(x, y, indexing="xy")
    zz_top = np.zeros_like(xx_top)
    val_top = epsilon[0, :, :]

    xx_front, zz_front = np.meshgrid(x, z, indexing="xy")
    yy_front = np.zeros_like(xx_front)
    val_front = epsilon[:, 0, :]

    yy_left, zz_left = np.meshgrid(y, z, indexing="xy")
    xx_left = np.zeros_like(yy_left)
    val_left = epsilon[:, :, 0]

    cmap = plt.get_cmap("jet")
    norm = colors.Normalize(vmin=float(np.nanmin(epsilon)), vmax=float(np.nanmax(epsilon)))

    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(xx_top, yy_top, zz_top, facecolors=cmap(norm(val_top)), linewidth=0.0, shade=False, alpha=0.45)
    ax.plot_surface(xx_front, yy_front, zz_front, facecolors=cmap(norm(val_front)), linewidth=0.0, shade=False, alpha=0.45)
    ax.plot_surface(xx_left, yy_left, zz_left, facecolors=cmap(norm(val_left)), linewidth=0.0, shade=False, alpha=0.45)

    ax.scatter(source_points[:, 2] * ds, source_points[:, 1] * ds, source_points[:, 0] * ds, c="red", s=18, marker="*", depthshade=False)
    ax.scatter(receiver_points[:, 2] * ds, receiver_points[:, 1] * ds, receiver_points[:, 0] * ds, c="lime", s=5, marker="o", alpha=0.8, depthshade=False)

    ax.set_title(title)
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    ax.set_xlim(0.0, x[-1] if nx > 1 else ds)
    ax.set_ylim(0.0, y[-1] if ny > 1 else ds)
    ax.set_zlim(0.0, z[-1] if nz > 1 else ds)
    ax.invert_zaxis()
    ax.view_init(elev=18, azim=-125)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("epsilon_r")

    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_model_slices_compare(
    out_png: Path,
    epsilon_inv: np.ndarray,
    epsilon_true: np.ndarray,
    epsilon_init: np.ndarray | None,
    title: str,
) -> None:
    if epsilon_inv.shape != epsilon_true.shape:
        raise ValueError(f"Shape mismatch: inv={epsilon_inv.shape}, true={epsilon_true.shape}")
    if epsilon_init is not None and epsilon_init.shape != epsilon_true.shape:
        raise ValueError(f"Shape mismatch: init={epsilon_init.shape}, true={epsilon_true.shape}")

    nz, ny, nx = epsilon_inv.shape
    iz, iy, ix = nz // 2, ny // 2, nx // 2

    slices_inv = [
        epsilon_inv[iz, :, :],
        epsilon_inv[:, iy, :],
        epsilon_inv[:, :, ix],
    ]
    slices_true = [
        epsilon_true[iz, :, :],
        epsilon_true[:, iy, :],
        epsilon_true[:, :, ix],
    ]
    slices_init = None
    if epsilon_init is not None:
        slices_init = [
            epsilon_init[iz, :, :],
            epsilon_init[:, iy, :],
            epsilon_init[:, :, ix],
        ]
    names = [f"XY (z={iz})", f"XZ (y={iy})", f"YZ (x={ix})"]

    vmin = 1
    vmax = 19

    n_rows = 3 if slices_init is not None else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.5 * n_rows), constrained_layout=True)
    for c in range(3):
        row0 = 0
        if slices_init is not None:
            im_init = axes[0, c].imshow(slices_init[c], cmap="jet", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
            axes[0, c].set_title(f"Initial | {names[c]}")
            axes[0, c].set_xlabel("X" if c < 2 else "Y")
            axes[0, c].set_ylabel("Y" if c == 0 else "Z")
            plt.colorbar(im_init, ax=axes[0, c], fraction=0.046, pad=0.02)
            row0 = 1

        im0 = axes[row0, c].imshow(slices_inv[c], cmap="jet", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
        axes[row0, c].set_title(f"Inverted | {names[c]}")
        axes[row0, c].set_xlabel("X" if c < 2 else "Y")
        axes[row0, c].set_ylabel("Y" if c == 0 else "Z")

        im1 = axes[row0 + 1, c].imshow(slices_true[c], cmap="jet", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
        axes[row0 + 1, c].set_title(f"True | {names[c]}")
        axes[row0 + 1, c].set_xlabel("X" if c < 2 else "Y")
        axes[row0 + 1, c].set_ylabel("Y" if c == 0 else "Z")

        plt.colorbar(im0, ax=axes[row0, c], fraction=0.046, pad=0.02)
        plt.colorbar(im1, ax=axes[row0 + 1, c], fraction=0.046, pad=0.02)

    fig.suptitle(title)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    device = resolve_device(args.device)

    eps_path = Path(args.eps_path)
    if not eps_path.is_absolute():
        eps_path = (Path(__file__).resolve().parent / eps_path).resolve()

    if args.output_dir is None:
        band_tag = "-".join(s.strip() for s in str(args.bands_mhz).split(",") if s.strip())
        smooth_tag = str(float(args.init_smooth_sigma)).replace(".", "p")
        args.output_dir = f"hydro_lp{band_tag}MHz_smooth{smooth_tag}"

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    band_mhz = parse_float_list(args.bands_mhz)
    band_hz = [v * 1e6 for v in band_mhz]
    epochs_list = parse_int_list(args.epochs_per_band)
    if len(epochs_list) == 1:
        epochs_list = epochs_list * len(band_hz)
    if len(epochs_list) != len(band_hz):
        raise ValueError("epochs-per-band must have 1 value or same length as bands-mhz")
    if args.model_gradient_sampling_interval < 1:
        raise ValueError("model-gradient-sampling-interval must be >= 1")

    epsilon_raw, eps_ds_name = load_first_numeric_h5_dataset(eps_path)
    epsilon_true_np = to_solver_zyx(epsilon_raw, args.eps_layout)

    nz, ny, nx = epsilon_true_np.shape
    if args.air_layers < 1 or args.air_layers >= nz:
        raise ValueError(f"air_layers must be in [1, {nz - 1}], got {args.air_layers}")

    epsilon_true_np = epsilon_true_np.copy()
    epsilon_true_np[: args.air_layers, :, :] = args.air_eps

    sigma_np = np.full_like(epsilon_true_np, fill_value=args.sigma_mean, dtype=np.float32)
    sigma_np[: args.air_layers, :, :] = 0.0
    mu_np = np.ones_like(epsilon_true_np, dtype=np.float32)

    source_points, receiver_points, _, _ = build_line_geometry(
        air_layers=args.air_layers,
        ny=ny,
        nx=nx,
        n_lines=args.n_lines,
        sources_per_line=args.sources_per_line,
        receivers_per_line=args.receivers_per_line,
        edge_margin=args.edge_margin,
    )

    auto_dt, dt_limit = estimate_stable_dt(epsilon_true_np, mu_np, args.ds, safety=args.dt_safety)
    dt = auto_dt if args.dt is None else args.dt
    if dt > dt_limit:
        raise ValueError(f"dt={dt:.3e}s exceeds CFL limit {dt_limit:.3e}s")

    epsilon_true = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
    sigma_fixed = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    mu_fixed = torch.tensor(mu_np, dtype=torch.float32, device=device)

    # 初始模型：对真值做平滑，确保不反演sigma
    epsilon_init_np = gaussian_filter(epsilon_true_np, sigma=float(args.init_smooth_sigma)).astype(np.float32)
    epsilon_init_np[: args.air_layers, :, :] = args.air_eps

    epsilon_inv = torch.tensor(epsilon_init_np, dtype=torch.float32, device=device, requires_grad=True)

    # 生成基础观测（基频）
    print("Generating observed data at base frequency...")
    wavelet_base = make_ricker(args.base_freq, args.nt, dt, device)
    n_shots = source_points.shape[0]
    source_amp_full_base = wavelet_base.view(1, 1, args.nt).repeat(n_shots, 1, 1)

    with torch.no_grad():
        observed_base = forward_batched(
            epsilon=epsilon_true,
            sigma=sigma_fixed,
            mu=mu_fixed,
            source_points=source_points,
            receiver_points=receiver_points,
            source_amplitude_full=source_amp_full_base,
            ds=args.ds,
            dt=dt,
            nt=args.nt,
            pml_width=args.pml_width,
            stencil=args.stencil,
            shot_batch_size=args.shot_batch_size,
            requires_grad=False,
            model_gradient_sampling_interval=args.model_gradient_sampling_interval,
        )

    observed_stages: list[torch.Tensor] = [apply_fir_lowpass(observed_base, dt=dt, cutoff_hz=f) for f in band_hz]

    print(f"Device: {device}")
    print(f"epsilon dataset: {eps_ds_name}")
    print(f"model shape={epsilon_true_np.shape}, n_shots={n_shots}, n_receivers={receiver_points.shape[0]}")
    print(f"base freq={args.base_freq / 1e6:.1f} MHz")
    print(f"bands (MHz)={band_mhz}")
    print(f"epochs={epochs_list}")
    print(f"model_gradient_sampling_interval={args.model_gradient_sampling_interval}")

    all_stage_meta: list[dict[str, float | int | str]] = []

    if sotb_interface is None:
        raise RuntimeError(
            "sotb-wrapper is not importable. Install via `uv pip install sotb-wrapper`."
        )

    n_param = int(epsilon_inv.numel())
    air_mask = torch.zeros_like(epsilon_inv, dtype=torch.bool)
    air_mask[: args.air_layers, :, :] = True
    air_mask_np = air_mask.detach().cpu().numpy().reshape(-1)

    lb_bounds = np.full(n_param, float(args.eps_lb), dtype=np.float32)
    ub_bounds = np.full(n_param, float(args.eps_ub), dtype=np.float32)
    lb_bounds[air_mask_np] = float(args.air_eps)
    ub_bounds[air_mask_np] = float(args.air_eps)

    def pack_eps_param() -> np.ndarray:
        return (
            epsilon_inv.detach()
            .contiguous()
            .view(-1)
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .astype(np.float32, copy=False)
        )

    def unpack_eps_param(x: np.ndarray) -> None:
        eps_vec = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            epsilon_inv.copy_(eps_vec.view_as(epsilon_inv))

    def pack_eps_grad() -> np.ndarray:
        if epsilon_inv.grad is None:
            grad = torch.zeros_like(epsilon_inv)
        else:
            grad = epsilon_inv.grad
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

    for stage_idx, (cut_hz, cut_mhz, stage_obs, n_epochs) in enumerate(zip(band_hz, band_mhz, observed_stages, epochs_list), start=1):
        print(f"\n=== Stage {stage_idx}/{len(band_hz)}: LP {cut_mhz:.1f} MHz, PLBFGS iters={n_epochs} ===")

        stage_t0 = time.perf_counter()
        x = pack_eps_param()

        def evaluate_from_x() -> tuple[float, np.ndarray, np.ndarray]:
            unpack_eps_param(x)
            with torch.no_grad():
                epsilon_inv.clamp_(float(args.eps_lb), float(args.eps_ub))
                epsilon_inv[: args.air_layers, :, :] = args.air_eps
            x[:] = pack_eps_param()

            if epsilon_inv.grad is not None:
                epsilon_inv.grad.zero_()

            total_loss = 0.0
            batch = max(1, int(args.shot_batch_size))
            for s0 in range(0, n_shots, batch):
                s1 = min(s0 + batch, n_shots)
                syn_b = forward_single_batch(
                    epsilon=epsilon_inv,
                    sigma=sigma_fixed,
                    mu=mu_fixed,
                    source_points=source_points,
                    receiver_points=receiver_points,
                    source_amplitude_full=source_amp_full_base,
                    ds=args.ds,
                    dt=dt,
                    pml_width=args.pml_width,
                    stencil=args.stencil,
                    model_gradient_sampling_interval=args.model_gradient_sampling_interval,
                    s0=s0,
                    s1=s1,
                )
                syn_b_filt = apply_fir_lowpass(syn_b, dt=dt, cutoff_hz=cut_hz)
                obs_b = stage_obs[:, s0:s1, :]
                loss_b = F.mse_loss(syn_b_filt, obs_b)
                loss_b.backward()
                total_loss += float(loss_b.item())

                del syn_b, syn_b_filt, obs_b, loss_b
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if epsilon_inv.grad is not None:
                epsilon_inv.grad[: args.air_layers, :, :] = 0.0
                epsilon_inv.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

            grad = pack_eps_grad()
            grad_preco = grad.copy()
            grad_preco[air_mask_np] = 0.0
            np.nan_to_num(grad_preco, copy=False)
            x[:] = pack_eps_param()
            return total_loss, grad, grad_preco

        sotb = sotb_interface.sotb_wrapper()
        sotb.udf = sotb_interface.UserDefined()

        fcost, grad, grad_preco = evaluate_from_x()
        sotb.set_inputs(
            fcost,
            niter_max=n_epochs,
            conv=float(args.plbfgs_conv),
            print_flag=0,
            nls_max=int(args.plbfgs_nls_max),
            l=int(args.plbfgs_l),
        )

        q_plb = np.zeros(n_param, dtype=np.float32)
        flag = 0
        eval_count = 0
        safety_max_evals = max(20, n_epochs * 80)
        last_eval_loss = float(fcost)
        last_logged_iter = int(sotb.udf.cpt_iter)
        last_plotted_iter = 0
        stage_tag = f"lp{int(round(cut_mhz))}MHz"
        progress_dir = out_dir / "epoch_slices" / stage_tag
        progress_dir.mkdir(parents=True, exist_ok=True)

        while flag not in (2, 4):
            flag = sotb.PLBFGS(
                n_param,
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
                print(f"  PLBFGS iter {curr_iter_after}/{n_epochs}  loss={last_eval_loss:.6e}")
                last_logged_iter = curr_iter_after

            if (
                args.plot_interval > 0
                and curr_iter_after > 0
                and curr_iter_after % int(args.plot_interval) == 0
                and curr_iter_after > last_plotted_iter
            ):
                unpack_eps_param(x)
                with torch.no_grad():
                    epsilon_inv.clamp_(float(args.eps_lb), float(args.eps_ub))
                    epsilon_inv[: args.air_layers, :, :] = args.air_eps
                x[:] = pack_eps_param()

                eps_iter_np = epsilon_inv.detach().cpu().numpy().astype(np.float32, copy=False)
                cmp_png = progress_dir / f"stage{stage_idx:02d}_{stage_tag}_iter{curr_iter_after:04d}_slices.png"
                plot_model_slices_compare(
                    cmp_png,
                    eps_iter_np,
                    epsilon_true_np,
                    epsilon_init_np,
                    title=f"Stage {stage_idx} | LP {cut_mhz:.1f} MHz | Iter {curr_iter_after}",
                )
                print(f"  Saved slice compare: {cmp_png}")
                last_plotted_iter = curr_iter_after

            if flag == 1:
                fcost, grad, grad_preco = evaluate_from_x()
                eval_count += 1
                last_eval_loss = float(fcost)
            elif flag == 5:
                np.nan_to_num(q_plb, copy=False)
                q_plb[air_mask_np] = 0.0
            elif flag not in (2, 3, 4):
                print(f"  [WARN] SOTB PLBFGS returned flag={flag}")

            if eval_count >= safety_max_evals:
                print(f"  [WARN] PLBFGS safety stop at eval_count={eval_count}, flag={flag}")
                break

        unpack_eps_param(x)
        with torch.no_grad():
            epsilon_inv.clamp_(float(args.eps_lb), float(args.eps_ub))
            epsilon_inv[: args.air_layers, :, :] = args.air_eps
        print(f"  PLBFGS stage finished with flag={flag}")

        stage_time = time.perf_counter() - stage_t0

        with torch.no_grad():
            syn_stage = forward_batched(
                epsilon=epsilon_inv,
                sigma=sigma_fixed,
                mu=mu_fixed,
                source_points=source_points,
                receiver_points=receiver_points,
                source_amplitude_full=source_amp_full_base,
                ds=args.ds,
                dt=dt,
                nt=args.nt,
                pml_width=args.pml_width,
                stencil=args.stencil,
                shot_batch_size=args.shot_batch_size,
                requires_grad=False,
                model_gradient_sampling_interval=args.model_gradient_sampling_interval,
            )

        eps_stage_np = epsilon_inv.detach().cpu().numpy().astype(np.float32, copy=False)
        syn_stage_np = syn_stage.detach().cpu().numpy().astype(np.float32, copy=False)

        npy_path = out_dir / f"epsilon_stage_{stage_tag}.npy"
        h5_path = out_dir / f"stage_{stage_tag}.h5"
        fig_path = out_dir / f"epsilon_stage_{stage_tag}_3d.png"

        np.save(npy_path, eps_stage_np)
        save_stage_h5(
            h5_path,
            eps_stage_np,
            sigma_np,
            source_points,
            receiver_points,
            syn_stage_np,
            args.ds,
            dt,
            cut_mhz,
        )
        plot_model_translucent(
            fig_path,
            eps_stage_np,
            args.ds,
            args.air_layers,
            source_points,
            receiver_points,
            title=f"Inversion result @ LP {cut_mhz:.1f} MHz",
        )

        all_stage_meta.append(
            {
                "stage": stage_idx,
                "lowpass_mhz": cut_mhz,
                "epochs": n_epochs,
                "time_s": stage_time,
                "epsilon_npy": str(npy_path),
                "stage_h5": str(h5_path),
                "stage_fig": str(fig_path),
            }
        )
        print(f"Saved stage epsilon: {npy_path}")
        print(f"Saved stage h5: {h5_path}")
        print(f"Saved stage figure: {fig_path}")

    summary_path = out_dir / "stage_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"stages": all_stage_meta}, f, indent=2, ensure_ascii=False)

    total_time = time.perf_counter() - t0
    print("\nAll stages done.")
    print(f"Saved summary: {summary_path}")
    print(f"Total time: {total_time:.2f} s")


if __name__ == "__main__":
    main()
