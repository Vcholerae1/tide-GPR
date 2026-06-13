from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch

import tide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D正演（12条测线：每条12源、70检）")
    parser.add_argument("--eps-path", type=str, default="data/eps_r_resized.h5", help="介电常数模型H5路径")
    parser.add_argument("--eps-layout", type=str, default="yxz", choices=["zyx", "yxz"], help="eps_r数据轴顺序")
    parser.add_argument("--air-layers", type=int, default=5, help="空气层厚度（网格点）")
    parser.add_argument("--air-eps", type=float, default=1.0, help="空气层介电常数")
    parser.add_argument("--sigma-mean", type=float, default=1e-3, help="地下电导率常值")
    parser.add_argument("--ds", type=float, default=0.02, help="网格间距(m)")
    parser.add_argument("--freq", type=float, default=4e8, help="Ricker主频(Hz)")
    parser.add_argument("--nt", type=int, default=1500, help="时间采样点数")
    parser.add_argument("--shot-batch-size", type=int, default=12, help="分批计算炮数")
    parser.add_argument("--pml-width", type=int, default=20, help="PML厚度")
    parser.add_argument("--stencil", type=int, default=4, choices=[2, 4, 6, 8], help="差分阶数")

    parser.add_argument("--n-lines", type=int, default=12, help="测线条数")
    parser.add_argument("--sources-per-line", type=int, default=12, help="每条测线激励源数")
    parser.add_argument("--receivers-per-line", type=int, default=70, help="每条测线接收点数")
    parser.add_argument("--edge-margin", type=int, default=2, help="边界保留网格数")

    parser.add_argument("--plot-shot-index", type=int, default=None, help="用于绘图的激励源编号，默认中间炮")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dt", type=float, default=None, help="时间步长(s)，默认按CFL自动估计")
    parser.add_argument("--dt-safety", type=float, default=0.8, help="自动估计dt时的CFL安全系数")
    parser.add_argument("--output-dir", type=str, default="outputs/water_f400MHz", help="输出目录")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _iter_h5_datasets(group: h5py.Group, prefix: str = ""):
    for key, value in group.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset):
            yield name, value
        elif isinstance(value, h5py.Group):
            yield from _iter_h5_datasets(value, prefix=name)


def load_first_numeric_h5_dataset(h5_path: Path) -> tuple[np.ndarray, str]:
    with h5py.File(h5_path, "r") as f:
        candidates: list[tuple[str, h5py.Dataset]] = []
        for name, ds in _iter_h5_datasets(f):
            if np.issubdtype(ds.dtype, np.number) and ds.ndim >= 2:
                candidates.append((name, ds))
        if not candidates:
            raise ValueError(f"No numeric dataset found in: {h5_path}")
        ds_name, ds = candidates[0]
        arr = ds[...]
    return np.asarray(arr, dtype=np.float32), ds_name


def to_solver_zyx(epsilon_src: np.ndarray, layout: str) -> np.ndarray:
    if epsilon_src.ndim != 3:
        raise ValueError(f"epsilon model must be 3D, got shape={epsilon_src.shape}")
    if layout == "zyx":
        return np.asarray(epsilon_src, dtype=np.float32)
    if layout == "yxz":
        return np.transpose(epsilon_src, (2, 0, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported eps layout: {layout}")


def estimate_stable_dt(epsilon: np.ndarray, mu: np.ndarray, ds: float, safety: float = 0.8) -> tuple[float, float]:
    if ds <= 0:
        raise ValueError(f"grid spacing ds must be positive, got {ds}")
    if not (0.0 < safety <= 1.0):
        raise ValueError(f"dt safety must be in (0, 1], got {safety}")

    min_eps_mu = float(np.min(epsilon * mu))
    if min_eps_mu <= 0.0:
        raise ValueError(f"epsilon * mu must stay positive, got min={min_eps_mu}")

    max_vel = tide.utils.C0 / math.sqrt(min_eps_mu)
    dt_limit = ds / (max_vel * math.sqrt(3.0))
    return safety * dt_limit, dt_limit


def make_ricker(freq: float, nt: int, dt: float, device: torch.device) -> torch.Tensor:
    return tide.ricker(freq, nt, dt, peak_time=1.2 / freq, dtype=torch.float32, device=device)


def shifted_linspace_int_indices(start: int, stop: int, count: int, shift_frac: float = 0.0) -> np.ndarray:
    if stop <= start:
        raise ValueError(f"Invalid index range: start={start}, stop={stop}")
    if count <= 1:
        return np.array([(start + stop - 1) // 2], dtype=np.int64)

    base = np.linspace(start, stop - 1, count, endpoint=True, dtype=np.float64)
    step = (stop - start - 1) / max(count - 1, 1)
    shifted = np.rint(base + shift_frac * step).astype(np.int64)
    return np.clip(shifted, start, stop - 1)


def build_line_geometry(
    *,
    air_layers: int,
    ny: int,
    nx: int,
    n_lines: int,
    sources_per_line: int,
    receivers_per_line: int,
    edge_margin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if edge_margin < 0:
        raise ValueError("edge_margin must be >= 0")
    if n_lines <= 0 or sources_per_line <= 0 or receivers_per_line <= 0:
        raise ValueError("n_lines, sources_per_line, receivers_per_line must be positive")

    y_start = edge_margin
    y_stop = ny - edge_margin
    x_start = edge_margin
    x_stop = nx - edge_margin
    if y_stop <= y_start or x_stop <= x_start:
        raise ValueError("edge_margin too large for current model size")

    z_plane = max(0, air_layers - 1)

    line_y = shifted_linspace_int_indices(y_start, y_stop, n_lines)
    src_x = shifted_linspace_int_indices(x_start, x_stop, sources_per_line, shift_frac=0.15)
    rec_x = shifted_linspace_int_indices(x_start, x_stop, receivers_per_line, shift_frac=-0.10)

    source_points = np.empty((n_lines * sources_per_line, 3), dtype=np.int64)
    receiver_points = np.empty((n_lines * receivers_per_line, 3), dtype=np.int64)
    receiver_line_id = np.empty((n_lines * receivers_per_line,), dtype=np.int64)

    s0 = 0
    r0 = 0
    for line_id, yv in enumerate(line_y):
        for xv in src_x:
            source_points[s0] = np.array([z_plane, int(yv), int(xv)], dtype=np.int64)
            s0 += 1
        for xv in rec_x:
            receiver_points[r0] = np.array([z_plane, int(yv), int(xv)], dtype=np.int64)
            receiver_line_id[r0] = line_id
            r0 += 1

    return source_points, receiver_points, receiver_line_id, line_y


def simulate_record_batched(
    *,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    wavelet: torch.Tensor,
    ds: float,
    dt: float,
    nt: int,
    pml_width: int,
    stencil: int,
    device: torch.device,
    shot_batch_size: int,
) -> np.ndarray:
    n_sources = source_points.shape[0]
    n_receivers = receiver_points.shape[0]
    batch = max(1, int(shot_batch_size))

    receiver_location_all = (
        torch.from_numpy(receiver_points)
        .to(device=device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(batch, 1, 1)
    )

    record = np.empty((nt, n_sources, n_receivers), dtype=np.float32)

    start = 0
    while start < n_sources:
        end = min(start + batch, n_sources)
        b = end - start
        print(f"[SIM] shots {start}:{end} / {n_sources} (batch={b})")

        src_batch = (
            torch.from_numpy(source_points[start:end])
            .to(device=device, dtype=torch.long)
            .unsqueeze(1)
        )
        rec_batch = receiver_location_all[:b]
        amp_batch = wavelet.view(1, 1, nt).repeat(b, 1, 1)

        try:
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
            )
        except torch.OutOfMemoryError:
            if device.type == "cuda" and b > 1:
                new_batch = max(1, b // 2)
                print(f"[WARN] CUDA OOM at batch={b}, reduce to {new_batch} and retry.")
                torch.cuda.empty_cache()
                batch = new_batch
                continue
            raise

        record[:, start:end, :] = out_b[-1].detach().cpu().numpy()
        del out_b, src_batch, rec_batch, amp_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

        start = end

    return record


def ensure_finite_record(record: np.ndarray) -> None:
    if np.isfinite(record).all():
        return
    bad_idx = np.argwhere(~np.isfinite(record))
    t_idx, shot_idx, rec_idx = bad_idx[0].tolist()
    raise RuntimeError(
        "Forward modeling produced non-finite values "
        f"at record[t={t_idx}, shot={shot_idx}, receiver={rec_idx}]."
    )


def save_h5(
    out_h5: Path,
    epsilon: np.ndarray,
    sigma: np.ndarray,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
    receiver_line_id: np.ndarray,
    line_y_index: np.ndarray,
    record: np.ndarray,
    ds: float,
    dt: float,
    freq: float,
    sim_time_s: float,
    total_time_s: float,
) -> None:
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("epsilon", data=epsilon, compression="gzip", compression_opts=4)
        f.create_dataset("sigma", data=sigma, compression="gzip", compression_opts=4)
        f.create_dataset("source_points_zyx", data=source_points, compression="gzip", compression_opts=4)
        f.create_dataset("receiver_points_zyx", data=receiver_points, compression="gzip", compression_opts=4)
        f.create_dataset("receiver_line_id", data=receiver_line_id, compression="gzip", compression_opts=4)
        f.create_dataset("receiver_line_y_index", data=line_y_index, compression="gzip", compression_opts=4)
        f.create_dataset("record", data=record, compression="gzip", compression_opts=4)

        f.attrs["grid_spacing_m"] = ds
        f.attrs["dt_s"] = dt
        f.attrs["freq_hz"] = freq
        f.attrs["n_sources"] = source_points.shape[0]
        f.attrs["n_receivers"] = receiver_points.shape[0]
        f.attrs["simulation_time_s"] = sim_time_s
        f.attrs["total_time_s"] = total_time_s


def plot_model(
    out_png: Path,
    epsilon: np.ndarray,
    ds: float,
    air_layers: int,
    source_points: np.ndarray,
    receiver_points: np.ndarray,
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

    ax.plot_surface(xx_top, yy_top, zz_top, facecolors=cmap(norm(val_top)), linewidth=0.0, shade=False, alpha=0.75)
    ax.plot_surface(xx_front, yy_front, zz_front, facecolors=cmap(norm(val_front)), linewidth=0.0, shade=False, alpha=0.75)
    ax.plot_surface(xx_left, yy_left, zz_left, facecolors=cmap(norm(val_left)), linewidth=0.0, shade=False, alpha=0.75)

    ax.scatter(source_points[:, 2] * ds, source_points[:, 1] * ds, source_points[:, 0] * ds, c="red", s=24, marker="*", depthshade=False, label="Sources")
    ax.scatter(receiver_points[:, 2] * ds, receiver_points[:, 1] * ds, receiver_points[:, 0] * ds, c="lime", s=6, marker="o", alpha=0.85, depthshade=False, label="Receivers")

    ax.set_title(f"3D epsilon_r model (air layers={air_layers})")
    ax.set_xlabel("X/m")
    ax.set_ylabel("Y/m")
    ax.set_zlabel("Z/m")
    ax.set_xlim(0.0, x[-1] if nx > 1 else ds)
    ax.set_ylim(0.0, y[-1] if ny > 1 else ds)
    ax.set_zlim(0.0, z[-1] if nz > 1 else ds)
    ax.invert_zaxis()
    ax.view_init(elev=18, azim=-125)
    ax.legend(loc="upper right", fontsize=8)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08)
    cbar.set_label("epsilon_r")

    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_single_shot_by_lines(
    out_png: Path,
    record: np.ndarray,
    receiver_line_id: np.ndarray,
    line_y_index: np.ndarray,
    ds: float,
    dt: float,
    shot_index: int,
) -> None:
    nt = record.shape[0]
    n_lines = line_y_index.size
    shot_gather = record[:, shot_index, :]

    line_indices: list[np.ndarray] = []
    for line_id in range(n_lines):
        idx = np.flatnonzero(receiver_line_id == line_id)
        if idx.size == 0:
            raise ValueError(f"Line {line_id} has no receivers.")
        line_indices.append(idx)

    max_abs = max(np.percentile(np.abs(shot_gather), 99.0), 1e-12)
    norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)
    cmap = plt.get_cmap("seismic")

    t_ns = np.arange(nt, dtype=np.float64) * dt * 1e9

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for line_id, idx in enumerate(line_indices):
        data_line = shot_gather[:, idx]  # [nt, nrec_line]
        nrec_line = data_line.shape[1]

        y_m = line_y_index[line_id] * ds
        yy = np.full((nt, nrec_line), y_m, dtype=np.float64)
        xx = np.tile(np.arange(nrec_line, dtype=np.float64), (nt, 1))
        zz = np.tile(t_ns[:, None], (1, nrec_line))

        ax.plot_surface(
            yy,
            xx,
            zz,
            facecolors=cmap(norm(data_line)),
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=0.95,
        )

    ax.set_title(f"Single-shot gather by survey lines (shot #{shot_index})")
    ax.set_xlabel("Y/m")
    ax.set_ylabel("Trace")
    ax.set_zlabel("Time/ns")
    ax.set_ylim(0.0, max((len(idx) for idx in line_indices), default=1) - 1)
    ax.set_zlim(float(t_ns[-1]), 0.0)
    ax.view_init(elev=18, azim=-118)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("E-field")

    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()

    device = resolve_device(args.device)
    eps_path = Path(args.eps_path)
    if not eps_path.is_absolute():
        eps_path = (Path(__file__).resolve().parent / eps_path).resolve()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    epsilon_raw, eps_ds_name = load_first_numeric_h5_dataset(eps_path)
    epsilon_np = to_solver_zyx(epsilon_raw, args.eps_layout)

    nz, ny, nx = epsilon_np.shape
    if args.air_layers < 1 or args.air_layers >= nz:
        raise ValueError(f"air_layers must be in [1, {nz - 1}], got {args.air_layers}")

    epsilon_np = epsilon_np.copy()
    epsilon_np[: args.air_layers, :, :] = args.air_eps
    sigma_np = np.full_like(epsilon_np, fill_value=args.sigma_mean, dtype=np.float32)
    sigma_np[: args.air_layers, :, :] = 0.0
    mu_np = np.ones_like(epsilon_np, dtype=np.float32)

    source_points, receiver_points, receiver_line_id, line_y_idx = build_line_geometry(
        air_layers=args.air_layers,
        ny=ny,
        nx=nx,
        n_lines=args.n_lines,
        sources_per_line=args.sources_per_line,
        receivers_per_line=args.receivers_per_line,
        edge_margin=args.edge_margin,
    )

    auto_dt, dt_limit = estimate_stable_dt(epsilon_np, mu_np, args.ds, safety=args.dt_safety)
    dt = auto_dt if args.dt is None else args.dt
    if dt > dt_limit:
        raise ValueError(
            f"dt={dt:.3e}s exceeds CFL limit {dt_limit:.3e}s for the current model. "
            "Reduce --dt or leave it unset for automatic estimation."
        )

    epsilon = torch.tensor(epsilon_np, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)

    wavelet = make_ricker(args.freq, args.nt, dt, device)

    n_sources = source_points.shape[0]
    n_receivers = receiver_points.shape[0]
    shot_index = n_sources // 2 if args.plot_shot_index is None else int(args.plot_shot_index)
    if shot_index < 0 or shot_index >= n_sources:
        raise ValueError(f"plot-shot-index must be in [0, {n_sources - 1}], got {shot_index}")

    print(f"Device: {device}")
    print(f"epsilon dataset: {eps_ds_name}")
    print(f"epsilon layout (input -> solver): {args.eps_layout} -> zyx")
    print(f"Model shape (z,y,x): {epsilon_np.shape}")
    print(f"line/source/receiver: {args.n_lines}/{args.sources_per_line}/{args.receivers_per_line}")
    print(f"n_sources={n_sources}, n_receivers={n_receivers}")
    print(f"CFL dt limit={dt_limit:.3e} s, auto dt={auto_dt:.3e} s, used dt={dt:.3e} s")

    t_sim0 = time.perf_counter()
    record = simulate_record_batched(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        source_points=source_points,
        receiver_points=receiver_points,
        wavelet=wavelet,
        ds=args.ds,
        dt=dt,
        nt=args.nt,
        pml_width=args.pml_width,
        stencil=args.stencil,
        device=device,
        shot_batch_size=args.shot_batch_size,
    )
    ensure_finite_record(record)
    sim_time_s = time.perf_counter() - t_sim0

    out_h5 = out_dir / "forward3d_multi_air_data.h5"
    model_png = out_dir / "epsilon_model_with_geometry.png"
    data_png = out_dir / "single_shot_all_receivers_by_line.png"

    total_time_s = time.perf_counter() - t0
    save_h5(
        out_h5=out_h5,
        epsilon=epsilon_np,
        sigma=sigma_np,
        source_points=source_points,
        receiver_points=receiver_points,
        receiver_line_id=receiver_line_id,
        line_y_index=line_y_idx,
        record=record,
        ds=args.ds,
        dt=dt,
        freq=args.freq,
        sim_time_s=sim_time_s,
        total_time_s=total_time_s,
    )
    plot_model(model_png, epsilon_np, args.ds, args.air_layers, source_points, receiver_points)
    plot_single_shot_by_lines(
        data_png,
        record,
        receiver_line_id,
        line_y_idx,
        args.ds,
        dt,
        shot_index,
    )

    print(f"Saved h5: {out_h5}")
    print(f"Saved model figure: {model_png}")
    print(f"Saved data figure: {data_png}")
    print(f"Simulation time: {sim_time_s:.3f} s")
    print(f"Total time: {total_time_s:.3f} s")


if __name__ == "__main__":
    main()
