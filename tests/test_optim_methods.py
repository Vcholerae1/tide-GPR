import pytest
import torch

import tide


def _stopping(
    *,
    max_iter: int = 40,
    max_evaluations: int | None = 400,
    gtol: float = 1e-6,
) -> tide.optim.StoppingCriteria:
    return tide.optim.StoppingCriteria(
        max_iter=max_iter,
        max_evaluations=max_evaluations,
        gtol=gtol,
        ftol=1e-12,
        xtol=1e-12,
    )


def _quadratic_problem(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
):
    target = torch.tensor([1.0, -2.0, 0.25], dtype=dtype, device=device)
    scale = torch.tensor([3.0, 0.5, 2.0], dtype=dtype, device=device)

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x - target
        grad = scale * residual
        return 0.5 * torch.dot(grad.reshape(-1), residual.reshape(-1)), grad

    def preconditioner(_x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return vector / scale

    def hessian_vector(_x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return scale * vector

    x0 = torch.tensor([5.0, 5.0, 5.0], dtype=dtype, device=device)
    return objective, preconditioner, hessian_vector, x0, target


def test_public_api_is_torch_native() -> None:
    assert hasattr(tide.optim, "OptimizerStatus")
    assert hasattr(tide.optim, "StoppingCriteria")
    assert hasattr(tide.optim, "LineSearchOptions")


@pytest.mark.parametrize("method", ["sd", "nlcg", "lbfgs", "tn"])
def test_nonlinear_methods_converge_with_torch_tensors(method: str) -> None:
    objective, preconditioner, hessian_vector, x0, target = _quadratic_problem()
    if method == "sd":
        result = tide.optim.steepest_descent_minimize(
            objective,
            x0,
            preconditioner=preconditioner,
            options=tide.optim.SteepestDescentOptions(stopping=_stopping()),
        )
    elif method == "nlcg":
        result = tide.optim.nlcg_minimize(
            objective,
            x0,
            preconditioner=preconditioner,
            options=tide.optim.NLCGOptions(stopping=_stopping()),
        )
    elif method == "lbfgs":
        result = tide.optim.lbfgs_minimize(
            objective,
            x0,
            preconditioner=preconditioner,
            options=tide.optim.LBFGSOptions(stopping=_stopping()),
        )
    else:
        result = tide.optim.truncated_newton_minimize(
            objective,
            hessian_vector,
            x0,
            preconditioner=preconditioner,
            options=tide.optim.TruncatedNewtonOptions(stopping=_stopping()),
        )

    assert result.success, result.status
    assert isinstance(result.x, torch.Tensor)
    assert result.x.device == x0.device
    assert result.x.dtype == x0.dtype
    assert result.n_prec > 0
    torch.testing.assert_close(result.x, target, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lbfgs_preserves_shape_and_dtype(dtype: torch.dtype) -> None:
    target = torch.tensor([[1.0, 2.0], [-3.0, 0.5]], dtype=dtype)

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = x - target
        return 0.5 * grad.square().sum(), grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.zeros_like(target),
        options=tide.optim.LBFGSOptions(stopping=_stopping()),
    )

    assert result.success
    assert result.x.shape == target.shape
    assert result.x.dtype == dtype
    torch.testing.assert_close(result.x, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_lbfgs_keeps_state_on_cuda() -> None:
    objective, _, _, x0, target = _quadratic_problem(device="cuda")
    result = tide.optim.lbfgs_minimize(
        objective,
        x0,
        options=tide.optim.LBFGSOptions(stopping=_stopping()),
    )
    assert result.success
    assert result.x.is_cuda
    torch.testing.assert_close(result.x, target, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("method", ["sd", "nlcg", "lbfgs", "tn"])
def test_box_constrained_boundary_optimum_converges(method: str) -> None:
    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = x - 2.0
        return 0.5 * grad.square().sum(), grad

    def hessian_vector(_x: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        return vector

    x0 = torch.tensor([0.0])
    kwargs = {"lower_bounds": 0.0, "upper_bounds": 1.0}
    if method == "sd":
        result = tide.optim.steepest_descent_minimize(
            objective,
            x0,
            options=tide.optim.SteepestDescentOptions(stopping=_stopping()),
            **kwargs,
        )
    elif method == "nlcg":
        result = tide.optim.nlcg_minimize(
            objective,
            x0,
            options=tide.optim.NLCGOptions(stopping=_stopping()),
            **kwargs,
        )
    elif method == "lbfgs":
        result = tide.optim.lbfgs_minimize(
            objective,
            x0,
            options=tide.optim.LBFGSOptions(stopping=_stopping()),
            **kwargs,
        )
    else:
        result = tide.optim.truncated_newton_minimize(
            objective,
            hessian_vector,
            x0,
            options=tide.optim.TruncatedNewtonOptions(stopping=_stopping()),
            **kwargs,
        )

    assert result.status == tide.optim.OptimizerStatus.CONVERGED_GRADIENT
    assert result.success
    torch.testing.assert_close(result.x, torch.tensor([1.0]))


def test_max_iter_zero_does_not_take_a_step_or_report_success() -> None:
    calls = 0

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal calls
        calls += 1
        grad = x - 1.0
        return 0.5 * grad.square().sum(), grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([2.0]),
        options=tide.optim.LBFGSOptions(
            stopping=_stopping(max_iter=0, max_evaluations=1)
        ),
    )

    assert result.status == tide.optim.OptimizerStatus.MAX_ITERATIONS
    assert not result.success
    assert result.n_iter == 0
    assert result.n_eval == calls == 1
    torch.testing.assert_close(result.x, torch.tensor([2.0]))


def test_objective_budget_is_never_overshot() -> None:
    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = x - 1.0
        return 0.5 * grad.square().sum(), grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([2.0]),
        options=tide.optim.LBFGSOptions(stopping=_stopping(max_evaluations=1)),
    )

    assert result.status == tide.optim.OptimizerStatus.MAX_EVALUATIONS
    assert result.n_eval == 1


def test_trace_is_scalar_only_by_default_and_callbacks_receive_lifecycle() -> None:
    events: list[tide.optim.OptimizerEventType] = []

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = x - 1.0
        return 0.5 * grad.square().sum(), grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([2.0]),
        options=tide.optim.LBFGSOptions(
            stopping=_stopping(),
            trace=tide.optim.TraceOptions(record=True),
        ),
        callback=lambda event: events.append(event.event),
    )

    assert events[0] == tide.optim.OptimizerEventType.INITIAL
    assert events[-1] == tide.optim.OptimizerEventType.TERMINATED
    assert all(entry.x is None and entry.grad is None for entry in result.trace)


def test_cgnr_converges_and_preserves_exact_matvec_budget() -> None:
    matrix = torch.tensor([[2.0, 0.0, 1.0], [0.0, 1.5, -0.5], [1.0, -1.0, 2.0]])
    target = torch.tensor([1.0, -2.0, 0.25])
    data = matrix @ target

    def forward(x: torch.Tensor) -> torch.Tensor:
        return matrix @ x

    def adjoint(residual: torch.Tensor) -> torch.Tensor:
        return matrix.T @ residual

    result = tide.optim.cgnr_solve(
        forward,
        adjoint,
        data,
        torch.zeros_like(target),
        options=tide.optim.CGNROptions(max_iter=20, rtol=1e-6),
    )
    assert result.success, result.status
    torch.testing.assert_close(result.x, target, atol=2e-5, rtol=2e-5)

    with pytest.raises(ValueError, match="at least 2"):
        tide.optim.CGNROptions(max_matvec=1)

    for budget in (2, 3):
        limited = tide.optim.cgnr_solve(
            forward,
            adjoint,
            data,
            torch.zeros_like(target),
            options=tide.optim.CGNROptions(max_iter=20, max_matvec=budget, rtol=0.0),
        )
        assert limited.status == tide.optim.OptimizerStatus.MAX_EVALUATIONS
        assert limited.n_forward + limited.n_adjoint <= budget


def test_cgnr_rejects_zero_preconditioner() -> None:
    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    result = tide.optim.cgnr_solve(
        identity,
        identity,
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        preconditioner=lambda _x, vector: torch.zeros_like(vector),
    )

    assert result.status == tide.optim.OptimizerStatus.INVALID_PRECONDITIONER
    assert not result.success


def test_lbfgs_rejects_zero_preconditioner() -> None:
    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = x - 1.0
        return 0.5 * grad.square().sum(), grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([0.0]),
        preconditioner=lambda _x, vector: torch.zeros_like(vector),
    )

    assert result.status == tide.optim.OptimizerStatus.INVALID_PRECONDITIONER
    assert not result.success


def test_cgnr_converges_for_inconsistent_least_squares() -> None:
    matrix = torch.tensor([[1.0], [1.0]])
    data = torch.tensor([1.0, 2.0])
    result = tide.optim.cgnr_solve(
        lambda x: matrix @ x,
        lambda residual: matrix.T @ residual,
        data,
        torch.zeros(1),
    )

    assert result.status == tide.optim.OptimizerStatus.CONVERGED_GRADIENT
    torch.testing.assert_close(result.x, torch.tensor([1.5]))
    assert float(result.residual.norm()) > 0.0
    torch.testing.assert_close(
        result.normal_residual, torch.zeros_like(result.normal_residual)
    )


def test_lbfgs_converges_on_rosenbrock() -> None:
    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x[1] - x[0].square()
        offset = 1.0 - x[0]
        loss = 100.0 * residual.square() + offset.square()
        grad = torch.stack((-400.0 * x[0] * residual - 2.0 * offset, 200.0 * residual))
        return loss, grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([-1.2, 1.0]),
        options=tide.optim.LBFGSOptions(
            stopping=_stopping(max_iter=100, max_evaluations=500, gtol=1e-5)
        ),
    )

    assert result.success, result.status
    torch.testing.assert_close(result.x, torch.ones(2), atol=2e-4, rtol=2e-4)


def test_objective_may_reuse_its_gradient_buffer() -> None:
    gradient_buffer = torch.empty(1)

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gradient_buffer.copy_(100.0 * (x - 3.0))
        return 50.0 * (x - 3.0).square().sum(), gradient_buffer

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([0.0]),
        options=tide.optim.LBFGSOptions(
            stopping=_stopping(max_iter=20, max_evaluations=100)
        ),
    )

    assert result.success, result.status
    torch.testing.assert_close(result.x, torch.tensor([3.0]))
    torch.testing.assert_close(result.grad, torch.zeros(1), atol=1e-6, rtol=0)


def test_large_finite_float64_loss_is_not_misclassified() -> None:
    def objective(x: torch.Tensor) -> tuple[float, torch.Tensor]:
        return 1e40, torch.ones_like(x)

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.zeros(1, dtype=torch.float64),
        options=tide.optim.LBFGSOptions(
            stopping=_stopping(max_iter=0, max_evaluations=1)
        ),
    )

    assert result.status == tide.optim.OptimizerStatus.MAX_ITERATIONS
    assert result.f == 1e40


def test_truncated_newton_accepts_large_finite_float64_curvature() -> None:
    hessian = torch.tensor(1e40, dtype=torch.float64)

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = hessian * x
        return 0.5 * torch.dot(x, grad), grad

    result = tide.optim.truncated_newton_minimize(
        objective,
        lambda _x, vector: hessian * vector,
        torch.tensor([1e-20], dtype=torch.float64),
        options=tide.optim.TruncatedNewtonOptions(stopping=_stopping()),
    )

    assert result.success, result.status
    torch.testing.assert_close(result.x, torch.zeros_like(result.x))


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (tide.optim.LineSearchOptions, {"expansion": float("nan")}),
        (tide.optim.LineSearchOptions, {"zoom_tolerance": float("nan")}),
        (tide.optim.LBFGSOptions, {"curvature_tolerance": float("nan")}),
        (tide.optim.CGNROptions, {"rtol": float("nan")}),
        (tide.optim.CGNROptions, {"atol": float("inf")}),
    ],
)
def test_float_options_reject_nonfinite_values(factory, kwargs) -> None:
    with pytest.raises(ValueError, match="finite"):
        factory(**kwargs)


def test_line_search_accepts_large_finite_float64_step() -> None:
    options = tide.optim.LineSearchOptions(initial_step=1e40)
    assert options.initial_step == 1e40


def test_sotb_weak_wolfe_expands_a_projected_tiny_gradient_step() -> None:
    events: list[tide.optim.OptimizerEvent] = []

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = 1e-5 * (x - 1.0)
        return 0.5e-5 * (x - 1.0).square().sum(), grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([0.0]),
        lower_bounds=0.0,
        upper_bounds=2.0,
        options=tide.optim.LBFGSOptions(
            stopping=_stopping(max_iter=2, max_evaluations=20),
            line_search=tide.optim.LineSearchOptions(method="weak_wolfe"),
        ),
        callback=events.append,
    )

    steps = [event for event in events if event.event == tide.optim.OptimizerEventType.STEP]
    assert steps[0].alpha >= 10_000.0
    torch.testing.assert_close(result.x, torch.tensor([1.0]))


def test_preconditioned_lbfgs_history_keeps_sotb_gamma_scaling() -> None:
    from tide.optim.common import _EvaluationBudget
    from tide.optim.history import _LBFGSHistory

    history = _LBFGSHistory(size=2, curvature_tolerance=0.0)
    assert history.update(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0]))
    direction = history.direction(
        torch.zeros(2),
        torch.tensor([1.0, 3.0]),
        lambda _x, vector: torch.tensor([4.0, 5.0]) * vector,
        _EvaluationBudget(max_objective=None),
    )

    torch.testing.assert_close(direction, torch.tensor([2.0, -5.0]))


def test_lbfgs_supports_sotb_relative_objective_convergence() -> None:
    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = x
        return 0.5 * x.square().sum(), grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([2.0]),
        options=tide.optim.LBFGSOptions(
            stopping=_stopping(max_iter=10, max_evaluations=30, gtol=0.0),
            relative_objective_tolerance=0.1,
        ),
    )

    assert result.status == tide.optim.OptimizerStatus.CONVERGED_FUNCTION
    assert result.f / 2.0 < 0.1


def test_shifted_negative_objective_uses_gradient_convergence() -> None:
    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grad = x - 3.0
        return 0.5 * grad.square().sum() - 1000.0, grad

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([0.0]),
        options=tide.optim.LBFGSOptions(stopping=_stopping()),
    )
    assert result.status == tide.optim.OptimizerStatus.CONVERGED_GRADIENT
    torch.testing.assert_close(result.x, torch.tensor([3.0]))
