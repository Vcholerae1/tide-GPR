from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def load_parser_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "profile" / "parse_ncu_metrics.py"
    spec = importlib.util.spec_from_file_location("parse_ncu_metrics", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_parser_memory_bound_classification(tmp_path: Path) -> None:
    module = load_parser_module()

    csv_path = tmp_path / "ncu_raw.csv"
    rows = [
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "gpu__time_duration.sum",
            "Metric Unit": "nsecond",
            "Metric Value": "2000000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "dram__bytes.sum",
            "Metric Unit": "byte",
            "Metric Value": "800000000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "Metric Unit": "inst",
            "Metric Value": "100000000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "Metric Unit": "inst",
            "Metric Value": "50000000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "Metric Unit": "inst",
            "Metric Value": "50000000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "Metric Unit": "pct",
            "Metric Value": "82",
        },
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "Metric Unit": "pct",
            "Metric Value": "30",
        },
        {
            "Kernel Name": "backward_kernel_lambda_e_with_grad",
            "Metric Name": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "Metric Unit": "pct",
            "Metric Value": "40",
        },
    ]
    write_csv(csv_path, rows)

    parsed, invocations = module.parse_ncu_csv(csv_path)
    table, summary = module.kernel_metrics_table(
        parsed,
        invocations,
        sustained_dram_peak_gbps=500.0,
        sustained_fp32_peak_gflops=10000.0,
        cells_per_launch=1e6,
    )

    assert len(table) == 1
    row = table[0]
    assert row["roofline_bound"] == "memory"
    assert row["stall_bound"] == "memory"
    assert row["evidence_consistent"] is True
    assert summary["consistency"]["has_conflict"] is False


def test_parser_detects_roofline_stall_conflict(tmp_path: Path) -> None:
    module = load_parser_module()

    csv_path = tmp_path / "ncu_raw_conflict.csv"
    rows = [
        {
            "Kernel Name": "backward_kernel_lambda_h",
            "Metric Name": "gpu__time_duration.sum",
            "Metric Unit": "nsecond",
            "Metric Value": "1500000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_h",
            "Metric Name": "dram__bytes.sum",
            "Metric Unit": "byte",
            "Metric Value": "500000000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_h",
            "Metric Name": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "Metric Unit": "inst",
            "Metric Value": "200000000",
        },
        {
            "Kernel Name": "backward_kernel_lambda_h",
            "Metric Name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "Metric Unit": "pct",
            "Metric Value": "90",
        },
        {
            "Kernel Name": "backward_kernel_lambda_h",
            "Metric Name": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "Metric Unit": "pct",
            "Metric Value": "25",
        },
        {
            "Kernel Name": "backward_kernel_lambda_h",
            "Metric Name": "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
            "Metric Unit": "pct",
            "Metric Value": "55",
        },
    ]
    write_csv(csv_path, rows)

    parsed, invocations = module.parse_ncu_csv(csv_path)
    _, summary = module.kernel_metrics_table(
        parsed,
        invocations,
        sustained_dram_peak_gbps=500.0,
        sustained_fp32_peak_gflops=10000.0,
        cells_per_launch=None,
    )

    assert summary["consistency"]["has_conflict"] is True
    assert summary["consistency"]["conflict_kernels"]
