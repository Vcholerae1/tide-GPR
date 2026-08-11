from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from numerical_utils import cosine_similarity, relative_l2, require_native_backend


@pytest.mark.slow
@pytest.mark.openmp
@pytest.mark.numerical
def test_openmp_thread_counts_preserve_traces_and_gradients(tmp_path: Path) -> None:
    require_native_backend()
    worker = Path(__file__).parent / "helpers" / "openmp_worker.py"
    results: dict[int, dict[str, object]] = {}
    for thread_count in (1, 2, 4):
        output = tmp_path / f"openmp-{thread_count}.pt"
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": str(thread_count),
                "OMP_DYNAMIC": "FALSE",
            }
        )
        subprocess.run(
            [sys.executable, str(worker), "--output", str(output)],
            check=True,
            env=environment,
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
        )
        results[thread_count] = torch.load(output, weights_only=True)

    reference = results[1]
    for thread_count in (2, 4):
        actual = results[thread_count]
        assert relative_l2(actual["receiver"], reference["receiver"]) < 1.0e-7
        for name in ("epsilon_gradient", "sigma_gradient"):
            assert relative_l2(actual[name], reference[name]) < 2.0e-6
            assert cosine_similarity(actual[name], reference[name]) > 0.999999
        assert float(actual["elapsed_seconds"]) > 0.0
