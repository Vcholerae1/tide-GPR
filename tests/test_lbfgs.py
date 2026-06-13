import importlib.util
from pathlib import Path

import numpy as np

import tide


def _rosenbrock(x: np.ndarray, grad: np.ndarray) -> float:
    f = (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    grad[0] = 2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    grad[1] = 200.0 * (x[1] - x[0] ** 2)
    return float(f)


def test_lbfgs_converges_on_rosenbrock_without_preconditioner():
    result = tide.optim.lbfgs_minimize(
        _rosenbrock,
        np.array([1.5, 1.5], dtype=np.float32),
        options=tide.optim.LBFGSOptions(
            max_iter=1000,
            history_size=10,
            tolerance=1e-8,
            max_evaluations=10000,
        ),
    )

    assert result.success, result.status
    assert result.x.dtype == np.float32
    np.testing.assert_allclose(result.x, np.array([1.0, 1.0]), atol=1e-3)
    assert result.f < 1e-6
    assert result.n_prec == 0
    assert all(entry.x.dtype == np.float32 for entry in result.trace)
    assert {entry.metadata["line_search"] for entry in result.trace} == {"weak_wolfe"}


def test_lbfgs_hager_zhang_line_search_converges_on_rosenbrock():
    result = tide.optim.lbfgs_minimize(
        _rosenbrock,
        np.array([1.5, 1.5], dtype=np.float32),
        options=tide.optim.LBFGSOptions(
            max_iter=1000,
            history_size=10,
            tolerance=1e-8,
            max_evaluations=10000,
            line_search="hager_zhang",
        ),
    )

    assert result.success, result.status
    np.testing.assert_allclose(result.x, np.array([1.0, 1.0]), atol=2e-3)
    assert result.f < 1e-6
    assert {entry.metadata["line_search"] for entry in result.trace} == {
        "hager_zhang"
    }


def test_lbfgs_more_thuente_line_search_converges_on_rosenbrock():
    result = tide.optim.lbfgs_minimize(
        _rosenbrock,
        np.array([1.5, 1.5], dtype=np.float32),
        options=tide.optim.LBFGSOptions(
            max_iter=1000,
            history_size=10,
            tolerance=1e-8,
            max_evaluations=10000,
            line_search="more_thuente",
        ),
    )

    assert result.success, result.status
    np.testing.assert_allclose(result.x, np.array([1.0, 1.0]), atol=2e-3)
    assert result.f < 1e-6
    assert {entry.metadata["line_search"] for entry in result.trace} == {
        "more_thuente"
    }
    assert all(
        abs(entry.q) <= 0.9 * abs(entry.q0) + 1e-5 for entry in result.trace
    )


def test_lbfgs_uses_inplace_objective_and_preconditioner():
    scale = np.array([3.0, 0.5, 2.0], dtype=np.float32)
    target = np.array([1.0, -2.0, 0.25], dtype=np.float32)
    residual = np.empty_like(target)
    calls = {"objective": 0, "preconditioner": 0}

    def objective(x: np.ndarray, grad: np.ndarray) -> float:
        calls["objective"] += 1
        np.subtract(x, target, out=residual)
        np.multiply(scale, residual, out=grad)
        return 0.5 * float(np.dot(grad, x - target))

    def preconditioner(
        _x: np.ndarray,
        vector: np.ndarray,
        out: np.ndarray,
    ) -> None:
        calls["preconditioner"] += 1
        np.divide(vector, scale, out=out)

    x0 = np.array([5.0, -5.0, 2.0], dtype=np.float32)
    options = tide.optim.LBFGSOptions(
        max_iter=50,
        history_size=5,
        tolerance=1e-8,
        max_evaluations=400,
    )
    result = tide.optim.lbfgs_minimize(
        objective,
        x0,
        preconditioner=preconditioner,
        options=options,
    )

    assert result.success, result.status
    np.testing.assert_allclose(result.x, target, atol=1e-6)
    assert result.x.dtype == np.float32
    assert calls["objective"] == result.n_eval
    assert calls["preconditioner"] == result.n_prec
    assert result.n_prec == result.n_iter + 1


def test_lbfgs_can_disable_trace_storage_but_keep_callback_events():
    callback_events = []
    result = tide.optim.lbfgs_minimize(
        _rosenbrock,
        np.array([1.5, 1.5], dtype=np.float32),
        options=tide.optim.LBFGSOptions(
            max_iter=1000,
            history_size=10,
            tolerance=1e-8,
            max_evaluations=10000,
            record_trace=False,
        ),
        callback=callback_events.append,
    )

    assert result.success, result.status
    assert result.trace == []
    assert callback_events
    assert all(entry.x.dtype == np.float32 for entry in callback_events)


def test_lbfgs_solves_joint_style_block_quadratic_with_bounds():
    n_cell = 12
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
        return 0.5 * float(np.dot(grad, residual))

    def block_preconditioner(
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

    x0 = np.concatenate(
        [
            np.full(n_cell, 6.0, dtype=np.float32),
            np.full(n_cell, 1.8, dtype=np.float32),
        ]
    )
    lower = np.concatenate(
        [
            np.ones(n_cell, dtype=np.float32),
            np.zeros(n_cell, dtype=np.float32),
        ]
    )
    upper = np.concatenate(
        [
            np.full(n_cell, 8.0, dtype=np.float32),
            np.full(n_cell, 2.5, dtype=np.float32),
        ]
    )
    options = tide.optim.LBFGSOptions(
        max_iter=40,
        history_size=5,
        tolerance=1e-8,
        max_evaluations=400,
    )

    result = tide.optim.lbfgs_minimize(
        objective,
        x0,
        preconditioner=block_preconditioner,
        options=options,
        lower_bounds=lower,
        upper_bounds=upper,
    )

    assert result.success, result.status
    np.testing.assert_allclose(result.x, target, atol=5e-4)
    assert result.f < 1e-6
    assert result.n_prec == result.n_iter + 1
    assert np.all(result.x >= lower)
    assert np.all(result.x <= upper)


def test_lbfgs_history_rollover_remains_finite():
    options = tide.optim.LBFGSOptions(
        max_iter=12,
        history_size=2,
        tolerance=0.0,
        max_evaluations=1000,
        record_trace=False,
    )
    x0 = np.array([-1.2, 1.0], dtype=np.float32)

    result = tide.optim.lbfgs_minimize(_rosenbrock, x0, options=options)

    assert result.status == "max_iter"
    assert result.success
    assert result.n_iter == options.max_iter
    assert result.n_eval <= options.max_evaluations
    assert np.isfinite(result.f)
    assert np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(result.grad))


def test_joint_multi_speed_benchmark_runs_python_backend():
    path = Path("benchmarks/optim_lbfgs_joint_multi_speed.py")
    spec = importlib.util.spec_from_file_location("optim_lbfgs_joint_multi_speed", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.compare_speed(
        ny=8,
        nx=8,
        repeats=1,
        history_size=3,
        max_iter=8,
        seed=1,
    )

    assert result["n_parameters"] == 128
    assert result["samples"][0]["success"]
    assert result["median_elapsed_s"] > 0.0
