#!/usr/bin/env python
"""Speed benchmark for LBFGS on a joint_multi-style block problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

import tide


def make_joint_multi_style_problem(ny: int, nx: int, seed: int = 0):
    """Build an eps/sigma block quadratic with a dense vector shape like joint_multi."""

    rng = np.random.default_rng(seed)
    n_cell = ny * nx
    eps_target = rng.uniform(2.0, 6.0, size=n_cell).astype(np.float32)
    sigma_target = rng.uniform(0.0, 1.5, size=n_cell).astype(np.float32)

    h_ee = rng.lognormal(mean=0.0, sigma=0.35, size=n_cell).astype(np.float32)
    h_ss = rng.lognormal(mean=-0.4, sigma=0.45, size=n_cell).astype(np.float32)
    rho = rng.uniform(-0.35, 0.35, size=n_cell).astype(np.float32)
    h_es = rho * np.sqrt(h_ee * h_ss)

    det = np.maximum(h_ee * h_ss - h_es * h_es, 1e-5)
    inv11 = (h_ss / det).astype(np.float32)
    inv22 = (h_ee / det).astype(np.float32)
    inv12 = (-h_es / det).astype(np.float32)

    air_mask = np.zeros(n_cell, dtype=bool)
    air_mask[:nx * 3] = True
    eps_target[air_mask] = 1.0
    sigma_target[air_mask] = 0.0
    inv11[air_mask] = 0.0
    inv22[air_mask] = 0.0
    inv12[air_mask] = 0.0

    valid = ~air_mask
    diag_valid = np.concatenate([inv11[valid], inv22[valid]])
    median_diag = float(np.median(diag_valid)) if diag_valid.size else 1.0
    if not np.isfinite(median_diag) or median_diag <= 0.0:
        median_diag = 1.0
    inv11 = np.clip(inv11 / median_diag, 0.3, 3.0)
    inv22 = np.clip(inv22 / median_diag, 0.3, 3.0)
    inv12 = inv12 / median_diag
    cross_limit = 0.8 * np.sqrt(np.maximum(inv11 * inv22, 1e-12))
    inv12 = np.clip(inv12, -cross_limit, cross_limit)
    inv11 = (0.3 + 0.7 * inv11).astype(np.float32)
    inv22 = (0.3 + 0.7 * inv22).astype(np.float32)
    inv12 = (0.7 * inv12).astype(np.float32)
    inv11[air_mask] = 0.0
    inv22[air_mask] = 0.0
    inv12[air_mask] = 0.0

    target = np.concatenate([eps_target, sigma_target]).astype(np.float32)
    x0 = np.concatenate(
        [
            np.clip(eps_target + 1.5, 1.0, 9.0),
            np.clip(sigma_target + 0.25, 0.0, 2.0),
        ]
    ).astype(np.float32)
    lower_bounds = np.concatenate(
        [
            np.ones(n_cell, dtype=np.float32),
            np.zeros(n_cell, dtype=np.float32),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(n_cell, 9.0, dtype=np.float32),
            np.full(n_cell, 2.0, dtype=np.float32),
        ]
    )

    eps_residual = np.empty(n_cell, dtype=np.float32)
    sigma_residual = np.empty(n_cell, dtype=np.float32)
    tmp_eps = np.empty(n_cell, dtype=np.float32)
    tmp_sigma = np.empty(n_cell, dtype=np.float32)

    def preconditioner(
        _x: np.ndarray,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        eps_vec = vector[:n_cell]
        sigma_vec = vector[n_cell:]
        out_eps = out[:n_cell]
        out_sigma = out[n_cell:]

        np.multiply(inv11, eps_vec, out=out_eps)
        np.multiply(inv12, sigma_vec, out=tmp_eps)
        out_eps += tmp_eps

        np.multiply(inv12, eps_vec, out=out_sigma)
        np.multiply(inv22, sigma_vec, out=tmp_sigma)
        out_sigma += tmp_sigma

        out_eps[air_mask] = 0.0
        out_sigma[air_mask] = 0.0

    def objective(
        x: np.ndarray,
        grad: np.ndarray,
    ) -> float:
        eps = x[:n_cell]
        sigma = x[n_cell:]
        grad_eps = grad[:n_cell]
        grad_sigma = grad[n_cell:]

        np.subtract(eps, eps_target, out=eps_residual)
        np.subtract(sigma, sigma_target, out=sigma_residual)

        np.multiply(h_ee, eps_residual, out=grad_eps)
        np.multiply(h_es, sigma_residual, out=tmp_eps)
        grad_eps += tmp_eps

        np.multiply(h_es, eps_residual, out=grad_sigma)
        np.multiply(h_ss, sigma_residual, out=tmp_sigma)
        grad_sigma += tmp_sigma

        grad_eps[air_mask] = 0.0
        grad_sigma[air_mask] = 0.0
        f = 0.5 * (
            float(np.dot(grad_eps, eps_residual))
            + float(np.dot(grad_sigma, sigma_residual))
        )
        return f

    return (
        objective,
        preconditioner,
        x0,
        target,
        lower_bounds,
        upper_bounds,
    )


def _run_one(
    objective,
    preconditioner,
    x0: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    options: tide.optim.LBFGSOptions,
) -> dict[str, Any]:
    start = perf_counter()
    result = tide.optim.lbfgs_minimize(
        objective,
        x0,
        preconditioner=preconditioner,
        options=options,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    return {
        "backend": "python",
        "success": result.success,
        "status": result.status,
        "elapsed_s": result.elapsed_s,
        "outer_elapsed_s": perf_counter() - start,
        "n_iter": result.n_iter,
        "n_eval": result.n_eval,
        "n_prec": result.n_prec,
        "final_f": result.f,
    }


def compare_speed(
    *,
    ny: int,
    nx: int,
    repeats: int,
    history_size: int,
    max_iter: int,
    seed: int = 0,
) -> dict[str, Any]:
    (
        objective,
        preconditioner,
        x0,
        target,
        lower_bounds,
        upper_bounds,
    ) = make_joint_multi_style_problem(ny, nx, seed=seed)
    options = tide.optim.LBFGSOptions(
        max_iter=max_iter,
        history_size=history_size,
        max_line_search=20,
        tolerance=1e-8,
        max_evaluations=max(100, max_iter * 40),
        record_trace=False,
    )

    payload: dict[str, Any] = {
        "ny": ny,
        "nx": nx,
        "n_parameters": int(x0.size),
        "target_norm": float(np.linalg.norm(target)),
        "history_size": history_size,
        "max_iter": max_iter,
    }
    samples = [
        _run_one(
            objective,
            preconditioner,
            x0,
            lower_bounds,
            upper_bounds,
            options,
        )
        for _ in range(repeats)
    ]
    payload.update(
        {
            "median_elapsed_s": median(sample["elapsed_s"] for sample in samples),
            "median_outer_elapsed_s": median(
                sample["outer_elapsed_s"] for sample in samples
            ),
            "median_n_iter": median(sample["n_iter"] for sample in samples),
            "median_n_eval": median(sample["n_eval"] for sample in samples),
            "median_final_f": median(sample["final_f"] for sample in samples),
            "samples": samples,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    result = compare_speed(
        ny=args.ny,
        nx=args.nx,
        repeats=args.repeats,
        history_size=args.history_size,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
