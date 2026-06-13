#!/usr/bin/env python
"""Benchmark the pure-Python LBFGS prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

import tide


def _make_problem(n_cell: int):
    eps_target = np.linspace(2.0, 5.0, n_cell, dtype=np.float32)
    sig_target = np.linspace(0.2, 1.1, n_cell, dtype=np.float32)
    target = np.concatenate([eps_target, sig_target]).astype(np.float32)
    eps_scale = np.linspace(1.0, 3.0, n_cell, dtype=np.float32)
    sig_scale = np.linspace(0.5, 2.0, n_cell, dtype=np.float32)
    cross_scale = np.full(n_cell, 0.15, dtype=np.float32)
    inv11 = 1.0 / eps_scale
    inv22 = 1.0 / sig_scale
    inv12 = -cross_scale / (eps_scale * sig_scale)
    residual = np.empty(2 * n_cell, dtype=np.float32)
    tmp = np.empty(n_cell, dtype=np.float32)

    def preconditioner(
        _x: np.ndarray,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        eps_vec = vector[:n_cell]
        sig_vec = vector[n_cell:]
        out_eps = out[:n_cell]
        out_sig = out[n_cell:]
        np.multiply(inv11, eps_vec, out=out_eps)
        np.multiply(inv12, sig_vec, out=tmp)
        out_eps += tmp
        np.multiply(inv12, eps_vec, out=out_sig)
        np.multiply(inv22, sig_vec, out=tmp)
        out_sig += tmp

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        eps = residual[:n_cell]
        sig = residual[n_cell:]
        grad_eps = grad[:n_cell]
        grad_sig = grad[n_cell:]
        np.subtract(x[:n_cell], eps_target, out=eps)
        np.subtract(x[n_cell:], sig_target, out=sig)
        np.multiply(eps_scale, eps, out=grad_eps)
        np.multiply(cross_scale, sig, out=tmp)
        grad_eps += tmp
        np.multiply(sig_scale, sig, out=grad_sig)
        np.multiply(cross_scale, eps, out=tmp)
        grad_sig += tmp
        f = 0.5 * float(np.dot(grad, residual))
        return f

    x0 = np.concatenate(
        [
            np.full(n_cell, 6.0, dtype=np.float32),
            np.full(n_cell, 1.8, dtype=np.float32),
        ]
    )
    return objective, preconditioner, x0, target


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "median_elapsed_s": median(item["elapsed_s"] for item in samples),
        "median_n_iter": median(item["n_iter"] for item in samples),
        "median_n_eval": median(item["n_eval"] for item in samples),
        "median_final_f": median(item["f"] for item in samples),
        "samples": samples,
    }


def _run_one(
    objective,
    preconditioner,
    x0: np.ndarray,
    options: tide.optim.LBFGSOptions,
) -> dict[str, Any]:
    result = tide.optim.lbfgs_minimize(
        objective,
        x0,
        preconditioner=preconditioner,
        options=options,
    )

    return {
        "backend": "python",
        "success": result.success,
        "status": result.status,
        "elapsed_s": result.elapsed_s,
        "n_iter": result.n_iter,
        "n_eval": result.n_eval,
        "n_prec": result.n_prec,
        "f": result.f,
    }


def run_benchmark(
    *,
    dims: list[int],
    repeats: int,
    history_size: int,
    max_iter: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"runs": []}
    options = tide.optim.LBFGSOptions(
        max_iter=max_iter,
        history_size=history_size,
        tolerance=1e-8,
        max_evaluations=max(100, max_iter * 40),
        record_trace=False,
    )
    for n_cell in dims:
        objective, preconditioner, x0, target = _make_problem(n_cell)
        per_backend: dict[str, Any] = {
            "n_cell": n_cell,
            "n_parameters": int(x0.size),
            "state_memory_mb": float(
                (x0.size * (2 * history_size + 6) * np.dtype(np.float32).itemsize)
            / 1e6
            ),
            "target_norm": float(np.linalg.norm(target)),
        }
        samples: list[dict[str, Any]] = []
        for _ in range(repeats):
            samples.append(_run_one(objective, preconditioner, x0, options))
        per_backend.update(_summarize(samples))
        payload["runs"].append(per_backend)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", type=int, default=[16, 1024, 16384])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    result = run_benchmark(
        dims=args.dims,
        repeats=args.repeats,
        history_size=args.history_size,
        max_iter=args.max_iter,
    )
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
