from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tide

from hydro_forward_lines import (
    build_line_geometry,
    ensure_finite_record,
    estimate_stable_dt,
    load_first_numeric_h5_dataset,
    make_ricker,
    resolve_device,
    to_solver_zyx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="不同minibatch的显存与时间对比")
    parser.add_argument("--eps-path", type=str, default="data/eps_r_resized.h5", help="介电常数模型H5路径")
    parser.add_argument("--eps-layout", type=str, default="yxz", choices=["zyx", "yxz"], help="eps_r数据轴顺序")
    parser.add_argument("--air-layers", type=int, default=5, help="空气层厚度（网格点）")
    parser.add_argument("--air-eps", type=float, default=1.0, help="空气层介电常数")
    parser.add_argument("--sigma-mean", type=float, default=1e-3, help="地下电导率常值")
    parser.add_argument("--ds", type=float, default=0.02, help="网格间距(m)")
    parser.add_argument("--freq", type=float, default=4e8, help="Ricker主频(Hz)")
    parser.add_argument("--nt", type=int, default=1500, help="时间采样点数")
    parser.add_argument("--pml-width", type=int, default=20, help="PML厚度")
    parser.add_argument("--stencil", type=int, default=4, choices=[2, 4, 6, 8], help="差分阶数")

    parser.add_argument("--n-lines", type=int, default=12, help="测线条数")
    parser.add_argument("--sources-per-line", type=int, default=12, help="每条测线激励源数")
    parser.add_argument("--receivers-per-line", type=int, default=70, help="每条测线接收点数")
    parser.add_argument("--edge-margin", type=int, default=2, help="边界保留网格数")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dt", type=float, default=None, help="时间步长(s)，默认按CFL自动估计")
    parser.add_argument("--dt-safety", type=float, default=0.8, help="自动估计dt时的CFL安全系数")

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32,64",
        help="逗号分隔的batch size列表",
    )
    parser.add_argument("--repeat", type=int, default=1, help="每个batch重复次数")
    parser.add_argument("--warmup", action="store_true", help="正式计时前先做一次预热")

    parser.add_argument("--output-dir", type=str, default="outputs/test_minibatch", help="输出目录")
    parser.add_argument("--output-csv", type=str, default="minibatch_benchmark.csv", help="结果CSV文件名")
    parser.add_argument(
        "--output-benchmark-png",
        type=str,
        default="minibatch_time_memory_comparison.png",
        help="时间/显存对比图文件名",
    )
    parser.add_argument(
        "--output-result-png",
        type=str,
        default="minibatch_forward_results_comparison.png",
        help="不同batchsize正演结果对比图文件名",
    )

    return parser.parse_args()


def parse_batch_sizes(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    values = sorted(set(values))
    for v in values:
        if v <= 0:
            raise ValueError(f"Invalid batch size: {v}")
    return values


def benchmark_one_batch(
    *,
    batch_size: int,
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
    keep_record: bool = False,
) -> tuple[float, float | None, np.ndarray | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    record = simulate_record_batched_fixed_batch(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        source_points=source_points,
        receiver_points=receiver_points,
        wavelet=wavelet,
        ds=ds,
        dt=dt,
        nt=nt,
        pml_width=pml_width,
        stencil=stencil,
        device=device,
        shot_batch_size=batch_size,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - t0

    ensure_finite_record(record)

    peak_mem_mib: float | None = None
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_mib = peak_bytes / (1024.0**2)

    record_out = record if keep_record else None
    if not keep_record:
        del record
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return elapsed_s, peak_mem_mib, record_out


def simulate_record_batched_fixed_batch(
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
    """固定 batch 前向建模：不自动降批，OOM 直接抛出给上层记录失败。"""
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
        print(f"[SIM-fixed] shots {start}:{end} / {n_sources} (batch={b}, requested={batch})")

        src_batch = (
            torch.from_numpy(source_points[start:end])
            .to(device=device, dtype=torch.long)
            .unsqueeze(1)
        )
        rec_batch = receiver_location_all[:b]
        amp_batch = wavelet.view(1, 1, nt).repeat(b, 1, 1)

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

        record[:, start:end, :] = out_b[-1].detach().cpu().numpy()
        del out_b, src_batch, rec_batch, amp_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

        start = end

    return record


def plot_benchmark_summary(
    rows: list[dict[str, float | int | str]],
    out_png: Path,
    *,
    has_cuda_memory: bool,
) -> None:
    valid_rows = [r for r in rows if np.isfinite(float(r["time_mean_s"]))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax_t = axes[0]
    if valid_rows:
        batch = np.array([int(r["batch_size"]) for r in valid_rows], dtype=np.int64)
        time_mean = np.array([float(r["time_mean_s"]) for r in valid_rows], dtype=np.float64)
        bars = ax_t.bar(batch.astype(str), time_mean, color="#4C78A8", alpha=0.9)
        ax_t.set_title("Runtime comparison")
        ax_t.set_xlabel("Mini-batch size")
        ax_t.set_ylabel("Time (s)")
        for b in bars:
            h = b.get_height()
            ax_t.text(b.get_x() + b.get_width() / 2.0, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax_t.axis("off")
        ax_t.text(0.5, 0.5, "No successful runs", ha="center", va="center", fontsize=11)

    ax_m = axes[1]
    if has_cuda_memory and valid_rows:
        mem_rows = [r for r in valid_rows if np.isfinite(float(r["peak_mem_mean_mib"]))]
        if mem_rows:
            mem_batch = np.array([int(r["batch_size"]) for r in mem_rows], dtype=np.int64)
            mem_mean = np.array([float(r["peak_mem_mean_mib"]) for r in mem_rows], dtype=np.float64)
            bars_m = ax_m.bar(mem_batch.astype(str), mem_mean, color="#F58518", alpha=0.9)
            ax_m.set_title("Peak GPU memory comparison")
            ax_m.set_xlabel("Mini-batch size")
            ax_m.set_ylabel("Peak memory (MiB)")
            for b in bars_m:
                h = b.get_height()
                ax_m.text(b.get_x() + b.get_width() / 2.0, h, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
        else:
            ax_m.axis("off")
            ax_m.text(0.5, 0.5, "No valid GPU memory stats", ha="center", va="center", fontsize=11)
    else:
        ax_m.axis("off")
        ax_m.text(
            0.5,
            0.5,
            "Memory curve is unavailable on CPU\n(use --device cuda for GPU memory stats)",
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_forward_results_by_batch(
    records_by_batch: dict[int, np.ndarray],
    out_png: Path,
    *,
    shot_index: int,
    dt: float,
    receiver_line_id: np.ndarray,
) -> None:
    batch_sizes = sorted(records_by_batch.keys())
    if not batch_sizes:
        fig = plt.figure(figsize=(7, 4), constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "No successful forward records to compare", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=220)
        plt.close(fig)
        return

    ordered_idx = np.argsort(receiver_line_id, kind="stable")
    gathers: dict[int, np.ndarray] = {}
    all_abs: list[np.ndarray] = []
    for bs in batch_sizes:
        g = records_by_batch[bs][:, shot_index, :][:, ordered_idx]
        gathers[bs] = g
        all_abs.append(np.abs(g))

    vlim = max(np.percentile(np.concatenate([a.ravel() for a in all_abs]), 99.0), 1e-12)

    n = len(batch_sizes)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), constrained_layout=True)
    axes_arr = np.array(axes, dtype=object).reshape(-1)

    nt = next(iter(gathers.values())).shape[0]
    tmax_ns = (nt - 1) * dt * 1e9

    last_im = None
    for i, bs in enumerate(batch_sizes):
        ax = axes_arr[i]
        gather = gathers[bs]
        nrec = gather.shape[1]
        last_im = ax.imshow(
            gather,
            cmap="seismic",
            vmin=-vlim,
            vmax=vlim,
            origin="upper",
            aspect="auto",
            extent=(0, nrec - 1, tmax_ns, 0.0),
        )
        ax.set_title(f"batch={bs}")
        ax.set_xlabel("Receiver (ordered by line)")
        ax.set_ylabel("Time (ns)")

    for j in range(len(batch_sizes), len(axes_arr)):
        axes_arr[j].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes_arr.tolist(), shrink=0.9, label="E-field")
    fig.suptitle(f"Forward result comparison for middle source (shot #{shot_index})", fontsize=13)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    eps_path = Path(args.eps_path)
    if not eps_path.is_absolute():
        eps_path = (Path(__file__).resolve().parent / eps_path).resolve()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parent / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    if args.repeat < 1:
        raise ValueError("repeat must be >= 1")

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

    source_points, receiver_points, receiver_line_id, _ = build_line_geometry(
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

    print(f"Device: {device}")
    print(f"epsilon dataset: {eps_ds_name}")
    print(f"Model shape (z,y,x): {epsilon_np.shape}")
    print(f"n_sources={source_points.shape[0]}, n_receivers={receiver_points.shape[0]}")
    print(f"batch_sizes={batch_sizes}, repeat={args.repeat}")
    print("batch policy: fixed batch (auto-reduce on OOM disabled)")

    shot_index = source_points.shape[0] // 2
    print(f"example shot index (middle source) = {shot_index}")

    if args.warmup:
        print("[Warmup] running batch=1 once...")
        _ = benchmark_one_batch(
            batch_size=1,
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
            keep_record=False,
        )

    rows: list[dict[str, float | int | str]] = []
    records_by_batch: dict[int, np.ndarray] = {}
    for bs in batch_sizes:
        times: list[float] = []
        mems: list[float] = []
        failed: list[str] = []

        for r in range(args.repeat):
            print(f"[Benchmark] batch={bs}, run={r + 1}/{args.repeat}")
            try:
                elapsed_s, peak_mem_mib, maybe_record = benchmark_one_batch(
                    batch_size=bs,
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
                    keep_record=(r == 0),
                )
                times.append(elapsed_s)
                if peak_mem_mib is not None:
                    mems.append(peak_mem_mib)
                if maybe_record is not None:
                    records_by_batch[bs] = maybe_record
            except Exception as e:
                msg = str(e).strip().replace("\n", " | ")
                failed.append(msg if msg else e.__class__.__name__)
                print(f"[WARN] batch={bs}, run={r + 1} failed: {failed[-1]}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        n_success = len(times)
        n_fail = len(failed)
        status = "ok" if n_fail == 0 else ("partial" if n_success > 0 else "failed")

        row: dict[str, float | int | str] = {
            "batch_size": bs,
            "repeat": args.repeat,
            "status": status,
            "n_success": n_success,
            "n_fail": n_fail,
            "error_last": failed[-1] if failed else "",
        }
        if n_success > 0:
            row["time_mean_s"] = float(np.mean(times))
            row["time_std_s"] = float(np.std(times, ddof=0))
            row["time_min_s"] = float(np.min(times))
            row["time_max_s"] = float(np.max(times))
            if mems:
                row["peak_mem_mean_mib"] = float(np.mean(mems))
                row["peak_mem_max_mib"] = float(np.max(mems))
            else:
                row["peak_mem_mean_mib"] = float("nan")
                row["peak_mem_max_mib"] = float("nan")
        else:
            row["time_mean_s"] = float("nan")
            row["time_std_s"] = float("nan")
            row["time_min_s"] = float("nan")
            row["time_max_s"] = float("nan")
            row["peak_mem_mean_mib"] = float("nan")
            row["peak_mem_max_mib"] = float("nan")
        rows.append(row)

    csv_path = out_dir / args.output_csv
    rows = sorted(rows, key=lambda x: int(x["batch_size"]))

    fieldnames = [
        "batch_size",
        "repeat",
        "status",
        "n_success",
        "n_fail",
        "time_mean_s",
        "time_std_s",
        "time_min_s",
        "time_max_s",
        "peak_mem_mean_mib",
        "peak_mem_max_mib",
        "error_last",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    benchmark_png = out_dir / args.output_benchmark_png
    result_png = out_dir / args.output_result_png
    plot_benchmark_summary(rows, benchmark_png, has_cuda_memory=(device.type == "cuda"))
    plot_forward_results_by_batch(
        records_by_batch,
        result_png,
        shot_index=shot_index,
        dt=dt,
        receiver_line_id=receiver_line_id,
    )

    print("\n=== Minibatch Benchmark Summary ===")
    header = (
        f"{'batch':>6} {'repeat':>6} {'time_mean(s)':>12} {'time_std(s)':>11} "
        f"{'time_min(s)':>11} {'time_max(s)':>11} {'peak_mem_mean(MiB)':>18} {'peak_mem_max(MiB)':>17}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['batch_size']):6d} {int(row['repeat']):6d} "
            f"{float(row['time_mean_s']):12.4f} {float(row['time_std_s']):11.4f} "
            f"{float(row['time_min_s']):11.4f} {float(row['time_max_s']):11.4f} "
            f"{float(row['peak_mem_mean_mib']):18.2f} {float(row['peak_mem_max_mib']):17.2f} "
            f"[{row['status']}, ok={int(row['n_success'])}, fail={int(row['n_fail'])}]"
        )
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved benchmark figure: {benchmark_png}")
    print(f"Saved forward-result figure: {result_png}")


if __name__ == "__main__":
    main()
