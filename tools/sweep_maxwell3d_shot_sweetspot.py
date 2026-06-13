"""Sweep 3D Maxwell shot batching across gradient storage backends.

This wraps ``profile_maxwell3d_gradient_storage.py`` in subprocesses so CUDA
OOM at one point does not poison the rest of the sweep. Results are ranked by
fixed-total-shot wall time, while gradient trust is only granted when a same-shot
full fp32 reference is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - this tool is normally run via uv.
    torch = None  # type: ignore[assignment]


PROFILE_SCRIPT = Path(__file__).with_name("profile_maxwell3d_gradient_storage.py")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


@dataclass(slots=True)
class PointRun:
    shots: int
    modes: list[str]
    output_path: Path
    command: list[str]
    returncode: int
    stdout_tail: str
    stderr_tail: str
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_path.exists()

    @property
    def oom(self) -> bool:
        text = f"{self.stdout_tail}\n{self.stderr_tail}".lower()
        markers = (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "cuda_error_out_of_memory",
        )
        return any(marker in text for marker in markers)


def tail_text(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:]


_GRADIENT_COSINE_KEYS = (
    "epsilon_grad_inner_cosine_similarity",
    "sigma_grad_inner_cosine_similarity",
    "epsilon_grad_cosine_similarity",
    "sigma_grad_cosine_similarity",
)


def valid_json(args: argparse.Namespace, path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if args.trust_metric not in {"cosine", "both"}:
        return True
    for mode_result in data.get("modes", []):
        error = mode_result.get("error_vs_full")
        if mode_result.get("mode") == "full" or error is None:
            continue
        if not any(error.get(key) is not None for key in _GRADIENT_COSINE_KEYS):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Maxwell3D shots-per-batch sweet spots under memory and gradient-trust constraints."
    )
    parser.add_argument("--nz", type=positive_int, default=48)
    parser.add_argument("--ny", type=positive_int, default=48)
    parser.add_argument("--nx", type=positive_int, default=48)
    parser.add_argument("--nt", type=positive_int, default=300)
    parser.add_argument("--shots", type=positive_int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--sources", type=positive_int, default=1)
    parser.add_argument("--receivers", type=positive_int, default=16)
    parser.add_argument("--stencil", type=int, choices=(2, 4, 6, 8), default=2)
    parser.add_argument("--pml", type=nonnegative_int, default=8)
    parser.add_argument("--grid-spacing", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=4e-11)
    parser.add_argument("--freq", type=float, default=80e6)
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=2e-4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--homogeneous", dest="heterogeneous", action="store_false")
    parser.set_defaults(heterogeneous=True)
    parser.add_argument("--model-batched", action="store_true")
    parser.add_argument("--grad", choices=("epsilon", "sigma", "both"), default="both")
    parser.add_argument("--gradient-sampling-interval", type=positive_int, default=1)
    parser.add_argument("--storage-chunk-steps", type=nonnegative_int, default=0)
    parser.add_argument("--n-threads", type=int, default=256)
    parser.add_argument("--source-component", choices=("ex", "ey", "ez"), default="ey")
    parser.add_argument(
        "--receiver-component", choices=("ex", "ey", "ez"), default="ey"
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--iters", type=positive_int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "bf16",
            "eonly",
            "eonly_bf16",
            "checkpoint",
            "checkpoint_bf16",
            "revolve",
            "revolve_bf16",
            "direct",
            "direct_bf16",
        ),
        default=(
            "eonly",
            "checkpoint",
            "revolve",
            "eonly_bf16",
            "checkpoint_bf16",
            "revolve_bf16",
        ),
    )
    parser.add_argument("--device", type=nonnegative_int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-shots", type=positive_int, default=64)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--fp32-grad-rel-l2", type=float, default=2e-4)
    parser.add_argument("--bf16-grad-rel-l2", type=float, default=1e-3)
    parser.add_argument("--grad-cosine-min", type=float, default=0.999)
    parser.add_argument(
        "--trust-metric",
        choices=("rel_l2", "cosine", "both"),
        default="rel_l2",
        help="Gradient criterion used to mark a candidate backend as trusted.",
    )
    parser.add_argument("--receiver-rel-l2", type=float, default=1e-5)
    parser.add_argument("--balanced-slack", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/profiles/maxwell3d_shot_sweetspot.json"),
    )
    parser.add_argument("--point-dir", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid point JSON files in --point-dir instead of rerunning them.",
    )
    parser.add_argument(
        "--isolated-performance",
        action="store_true",
        help=(
            "Use full+candidate runs only for trust checks, then benchmark each "
            "candidate mode alone for timing and peak-memory results."
        ),
    )
    return parser.parse_args()


def device_memory_bytes(device: int) -> int | None:
    if torch is None or not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(device).total_memory)


def profile_command(
    args: argparse.Namespace, shots: int, modes: list[str], output: Path
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--nz",
        str(args.nz),
        "--ny",
        str(args.ny),
        "--nx",
        str(args.nx),
        "--nt",
        str(args.nt),
        "--shots",
        str(shots),
        "--sources",
        str(args.sources),
        "--receivers",
        str(args.receivers),
        "--stencil",
        str(args.stencil),
        "--pml",
        str(args.pml),
        "--grid-spacing",
        str(args.grid_spacing),
        "--dt",
        str(args.dt),
        "--freq",
        str(args.freq),
        "--epsilon",
        str(args.epsilon),
        "--sigma",
        str(args.sigma),
        "--mu",
        str(args.mu),
        "--grad",
        args.grad,
        "--gradient-sampling-interval",
        str(args.gradient_sampling_interval),
        "--storage-chunk-steps",
        str(args.storage_chunk_steps),
        "--n-threads",
        str(args.n_threads),
        "--source-component",
        args.source_component,
        "--receiver-component",
        args.receiver_component,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--modes",
        *modes,
        "--output",
        str(output),
    ]
    if not args.heterogeneous:
        cmd.append("--homogeneous")
    if args.model_batched:
        cmd.append("--model-batched")
    return cmd


def run_profile(
    args: argparse.Namespace, shots: int, modes: list[str], output: Path
) -> PointRun:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = profile_command(args, shots, modes, output)
    if args.resume and output.exists() and valid_json(args, output):
        return PointRun(
            shots=shots,
            modes=modes,
            output_path=output,
            command=cmd,
            returncode=0,
            stdout_tail="reused existing point output",
            stderr_tail="",
            elapsed_s=0.0,
        )
    if output.exists():
        output.unlink()
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return PointRun(
        shots=shots,
        modes=modes,
        output_path=output,
        command=cmd,
        returncode=proc.returncode,
        stdout_tail=tail_text(proc.stdout),
        stderr_tail=tail_text(proc.stderr),
        elapsed_s=time.perf_counter() - started,
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mode_threshold(mode: str, args: argparse.Namespace) -> float:
    return args.bf16_grad_rel_l2 if "bf16" in mode else args.fp32_grad_rel_l2


def worst_gradient_rel_l2(error: dict[str, Any] | None) -> float | None:
    if not error:
        return None
    values = [
        error.get("epsilon_grad_inner_rel_l2"),
        error.get("sigma_grad_inner_rel_l2"),
        error.get("epsilon_grad_rel_l2"),
        error.get("sigma_grad_rel_l2"),
    ]
    present = [float(value) for value in values if value is not None]
    return max(present) if present else None


def min_gradient_cosine_similarity(error: dict[str, Any] | None) -> float | None:
    if not error:
        return None
    values = [error.get(key) for key in _GRADIENT_COSINE_KEYS]
    present = [float(value) for value in values if value is not None]
    return min(present) if present else None


def receiver_rel_l2(error: dict[str, Any] | None) -> float | None:
    if not error:
        return None
    value = error.get("receiver_rel_l2")
    return None if value is None else float(value)


def flatten_mode_result(
    *,
    args: argparse.Namespace,
    shot_count: int,
    mode_result: dict[str, Any],
    reference_available: bool,
    total_memory: int | None,
    source_path: Path,
) -> dict[str, Any]:
    mode = str(mode_result["mode"])
    measurement = mode_result["measurement"]
    mean_s = float(measurement["mean_s"])
    peak_bytes = int(measurement["peak_memory_allocated_bytes"])
    fixed_batches = math.ceil(args.total_shots / shot_count)
    error = mode_result.get("error_vs_full")
    worst_grad = worst_gradient_rel_l2(error)
    min_grad_cosine = min_gradient_cosine_similarity(error)
    recv_error = receiver_rel_l2(error)
    receiver_ok = recv_error is None or recv_error <= args.receiver_rel_l2

    if mode == "full":
        trust_status = "reference"
        trusted = True
        threshold = 0.0
        min_grad_cosine = 1.0
    elif not reference_available:
        trust_status = "unverified_no_reference"
        trusted = False
        threshold = mode_threshold(mode, args)
    else:
        threshold = mode_threshold(mode, args)
        rel_l2_ok = worst_grad is not None and worst_grad <= threshold
        cosine_ok = (
            min_grad_cosine is not None and min_grad_cosine >= args.grad_cosine_min
        )
        if args.trust_metric == "rel_l2":
            trusted = rel_l2_ok and receiver_ok
            missing_metric = worst_grad is None
        elif args.trust_metric == "cosine":
            trusted = cosine_ok and receiver_ok
            missing_metric = min_grad_cosine is None
        else:
            trusted = rel_l2_ok and cosine_ok and receiver_ok
            missing_metric = worst_grad is None or min_grad_cosine is None
        if missing_metric:
            trust_status = "unverified_missing_error"
        else:
            trust_status = "trusted" if trusted else "gradient_mismatch"

    memory_fraction = None if total_memory is None else peak_bytes / total_memory
    memory_ok = (
        True if memory_fraction is None else memory_fraction <= args.memory_fraction
    )
    return {
        "status": "ok",
        "shots": shot_count,
        "mode": mode,
        "execution_backend": mode_result.get("execution_backend"),
        "storage_compression": mode_result.get("storage_compression"),
        "trusted": trusted,
        "trust_status": trust_status,
        "trust_metric": args.trust_metric,
        "gradient_threshold": threshold,
        "worst_gradient_rel_l2": worst_grad,
        "gradient_cosine_min_threshold": args.grad_cosine_min,
        "gradient_cosine_min": min_grad_cosine,
        "worst_gradient_cosine_distance": None
        if min_grad_cosine is None
        else 1.0 - min_grad_cosine,
        "receiver_rel_l2": recv_error,
        "mean_s": mean_s,
        "median_s": float(measurement["median"]),
        "min_s": float(measurement["min"]),
        "max_s": float(measurement["max"]),
        "peak_memory_allocated_bytes": peak_bytes,
        "memory_fraction": memory_fraction,
        "memory_ok": memory_ok,
        "shots_per_s": shot_count / mean_s,
        "fixed_total_shots": args.total_shots,
        "fixed_total_batches": fixed_batches,
        "fixed_total_time_s": fixed_batches * mean_s,
        "cell_steps_per_s": float(measurement["cell_steps_per_s"]),
        "snapshot_storage_estimate": mode_result.get("snapshot_storage_estimate", {}),
        "source_path": str(source_path),
    }


def failure_result(
    *,
    args: argparse.Namespace,
    shot_count: int,
    mode: str,
    run: PointRun,
    total_memory: int | None,
) -> dict[str, Any]:
    del total_memory
    return {
        "status": "oom" if run.oom else "failed",
        "shots": shot_count,
        "mode": mode,
        "trusted": False,
        "trust_status": "not_run",
        "trust_metric": args.trust_metric,
        "returncode": run.returncode,
        "elapsed_s": run.elapsed_s,
        "command": run.command,
        "stdout_tail": run.stdout_tail,
        "stderr_tail": run.stderr_tail,
        "source_path": str(run.output_path),
    }


def collect_from_output(
    *,
    args: argparse.Namespace,
    output_path: Path,
    shot_count: int,
    wanted_modes: set[str],
    reference_available: bool,
    total_memory: int | None,
) -> list[dict[str, Any]]:
    data = read_json(output_path)
    rows = []
    for mode_result in data["modes"]:
        if mode_result["mode"] not in wanted_modes:
            continue
        rows.append(
            flatten_mode_result(
                args=args,
                shot_count=shot_count,
                mode_result=mode_result,
                reference_available=reference_available,
                total_memory=total_memory,
                source_path=output_path,
            )
        )
    return rows


_PERFORMANCE_KEYS = (
    "mean_s",
    "median_s",
    "min_s",
    "max_s",
    "peak_memory_allocated_bytes",
    "memory_fraction",
    "memory_ok",
    "shots_per_s",
    "fixed_total_batches",
    "fixed_total_time_s",
    "cell_steps_per_s",
    "snapshot_storage_estimate",
)


def merge_isolated_performance(
    trust_row: dict[str, Any], performance_row: dict[str, Any]
) -> dict[str, Any]:
    row = dict(trust_row)
    for key in _PERFORMANCE_KEYS:
        row[key] = performance_row[key]
    row["trust_source_path"] = trust_row["source_path"]
    row["performance_source_path"] = performance_row["source_path"]
    row["source_path"] = performance_row["source_path"]
    return row


def collect_or_failure(
    *,
    args: argparse.Namespace,
    output_path: Path,
    shot_count: int,
    mode: str,
    reference_available: bool,
    total_memory: int | None,
    run: PointRun,
) -> list[dict[str, Any]]:
    if not run.ok:
        return [
            failure_result(
                args=args,
                shot_count=shot_count,
                mode=mode,
                run=run,
                total_memory=total_memory,
            )
        ]
    return collect_from_output(
        args=args,
        output_path=output_path,
        shot_count=shot_count,
        wanted_modes={mode},
        reference_available=reference_available,
        total_memory=total_memory,
    )


def collect_candidate_with_isolated_performance(
    *,
    args: argparse.Namespace,
    shot_count: int,
    mode: str,
    trust_rows: list[dict[str, Any]],
    point_dir: Path,
    total_memory: int | None,
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    performance_output = point_dir / f"shots{shot_count}_{mode}_solo_perf.json"
    performance_run = run_profile(args, shot_count, [mode], performance_output)
    runs.append(asdict(performance_run))
    performance_rows = collect_or_failure(
        args=args,
        output_path=performance_output,
        shot_count=shot_count,
        mode=mode,
        reference_available=False,
        total_memory=total_memory,
        run=performance_run,
    )
    if len(performance_rows) != 1 or performance_rows[0].get("status") != "ok":
        return performance_rows
    if not trust_rows or trust_rows[0].get("status") != "ok":
        return performance_rows
    return [merge_isolated_performance(trust_rows[0], performance_rows[0])]


def sweep_shots(
    args: argparse.Namespace,
    shot_count: int,
    point_dir: Path,
    total_memory: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    modes = list(dict.fromkeys(args.modes))
    combined_output = point_dir / f"shots{shot_count}_full_plus_candidates.json"
    combined = run_profile(args, shot_count, ["full", *modes], combined_output)
    runs.append(asdict(combined))
    if combined.ok:
        wanted = {"full", *modes}
        combined_rows = collect_from_output(
            args=args,
            output_path=combined_output,
            shot_count=shot_count,
            wanted_modes=wanted,
            reference_available=True,
            total_memory=total_memory,
        )
        if not args.isolated_performance:
            rows.extend(combined_rows)
            return rows, runs

        rows.extend(row for row in combined_rows if row["mode"] == "full")
        trust_by_mode = {row["mode"]: row for row in combined_rows}
        for mode in modes:
            rows.extend(
                collect_candidate_with_isolated_performance(
                    args=args,
                    shot_count=shot_count,
                    mode=mode,
                    trust_rows=[trust_by_mode[mode]],
                    point_dir=point_dir,
                    total_memory=total_memory,
                    runs=runs,
                )
            )
        return rows, runs

    reference_output = point_dir / f"shots{shot_count}_full_reference.json"
    reference = run_profile(args, shot_count, ["full"], reference_output)
    runs.append(asdict(reference))
    reference_available = reference.ok
    if reference.ok:
        rows.extend(
            collect_from_output(
                args=args,
                output_path=reference_output,
                shot_count=shot_count,
                wanted_modes={"full"},
                reference_available=True,
                total_memory=total_memory,
            )
        )
    else:
        rows.append(
            failure_result(
                args=args,
                shot_count=shot_count,
                mode="full",
                run=reference,
                total_memory=total_memory,
            )
        )

    for mode in modes:
        output = point_dir / (
            f"shots{shot_count}_{mode}_{'withref' if reference_available else 'solo'}.json"
        )
        run_modes = ["full", mode] if reference_available else [mode]
        run = run_profile(args, shot_count, run_modes, output)
        runs.append(asdict(run))
        trust_rows = collect_or_failure(
            args=args,
            output_path=output,
            shot_count=shot_count,
            mode=mode,
            reference_available=reference_available,
            total_memory=total_memory,
            run=run,
        )
        if args.isolated_performance and reference_available:
            rows.extend(
                collect_candidate_with_isolated_performance(
                    args=args,
                    shot_count=shot_count,
                    mode=mode,
                    trust_rows=trust_rows,
                    point_dir=point_dir,
                    total_memory=total_memory,
                    runs=runs,
                )
            )
        else:
            rows.extend(trust_rows)
    return rows, runs


def recommend(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    viable = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("trusted")
        and row.get("memory_ok", True)
        and row.get("mode") != "full"
    ]
    if not viable:
        return {
            "fastest": None,
            "balanced": None,
            "capacity": None,
            "reason": "no non-reference backend passed gradient trust and memory policy",
        }

    fastest = min(viable, key=lambda row: row["fixed_total_time_s"])
    cutoff = fastest["fixed_total_time_s"] * (1.0 + args.balanced_slack)
    near_fastest = [row for row in viable if row["fixed_total_time_s"] <= cutoff]
    balanced = min(
        near_fastest,
        key=lambda row: (row["peak_memory_allocated_bytes"], -row["shots_per_s"]),
    )
    capacity = max(
        viable,
        key=lambda row: (
            row["shots"],
            -row["fixed_total_time_s"],
            -row["peak_memory_allocated_bytes"],
        ),
    )
    return {"fastest": fastest, "balanced": balanced, "capacity": capacity}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "shots",
        "mode",
        "trusted",
        "trust_status",
        "trust_metric",
        "worst_gradient_rel_l2",
        "gradient_cosine_min",
        "worst_gradient_cosine_distance",
        "mean_s",
        "peak_memory_allocated_bytes",
        "memory_fraction",
        "shots_per_s",
        "fixed_total_time_s",
        "source_path",
        "trust_source_path",
        "performance_source_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], recommendations: dict[str, Any]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    print("\nSweep rows:")
    print(
        "shots mode              trust                    cos_min  mean_ms peak_MB fixedK_s shots/s"
    )
    for row in sorted(ok_rows, key=lambda item: (item["shots"], item["mode"])):
        cosine = row.get("gradient_cosine_min")
        cosine_text = "n/a" if cosine is None else f"{cosine:.6f}"
        print(
            f"{row['shots']:>5} {row['mode']:<17} {row['trust_status']:<24} "
            f"{cosine_text:>7} "
            f"{row['mean_s'] * 1000:>7.2f} "
            f"{row['peak_memory_allocated_bytes'] / 1e6:>7.1f} "
            f"{row['fixed_total_time_s']:>8.3f} "
            f"{row['shots_per_s']:>7.2f}"
        )

    failures = [row for row in rows if row.get("status") != "ok"]
    if failures:
        print("\nFailures:")
        for row in failures:
            print(f"shots={row['shots']} mode={row['mode']} status={row['status']}")

    print("\nRecommendations:")
    for name in ("fastest", "balanced", "capacity"):
        row = recommendations.get(name)
        if row is None:
            print(f"{name}: none")
        else:
            cosine = row.get("gradient_cosine_min")
            cosine_text = "n/a" if cosine is None else f"{cosine:.6f}"
            print(
                f"{name}: shots={row['shots']} mode={row['mode']} "
                f"fixedK={row['fixed_total_time_s']:.3f}s "
                f"peak={row['peak_memory_allocated_bytes'] / 1e6:.1f}MB "
                f"cos={cosine_text}"
            )


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    point_dir = args.point_dir or args.output.with_suffix("").parent / (
        args.output.with_suffix("").name + "_points_" + timestamp
    )
    total_memory = device_memory_bytes(args.device)

    all_rows: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    for shots in args.shots:
        rows, runs = sweep_shots(args, shots, point_dir, total_memory)
        all_rows.extend(rows)
        all_runs.extend(runs)

    recommendations = recommend(all_rows, args)
    result = {
        "workload": {
            "model_shape": [args.nz, args.ny, args.nx],
            "nt": args.nt,
            "shots_swept": args.shots,
            "modes_swept": list(args.modes),
            "total_shots_for_ranking": args.total_shots,
            "storage_chunk_steps": args.storage_chunk_steps,
            "isolated_performance": args.isolated_performance,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "device": {
            "index": args.device,
            "total_memory_bytes": total_memory,
            "memory_fraction_policy": args.memory_fraction,
        },
        "thresholds": {
            "fp32_grad_rel_l2": args.fp32_grad_rel_l2,
            "bf16_grad_rel_l2": args.bf16_grad_rel_l2,
            "grad_cosine_min": args.grad_cosine_min,
            "trust_metric": args.trust_metric,
            "receiver_rel_l2": args.receiver_rel_l2,
            "balanced_slack": args.balanced_slack,
        },
        "recommendations": recommendations,
        "rows": all_rows,
        "runs": all_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    )
    write_csv(args.output.with_suffix(".csv"), all_rows)
    print_summary(all_rows, recommendations)
    print(f"\nWrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
