from __future__ import annotations

import pytest
import tide
import torch
from jaxtyping import TypeCheckError
from numerical_utils import MaxwellExample, make_maxwell3d_example, make_tm2d_example
from tide import backend_utils
from tide.workflow import (
    backward_shot_batches,
    block_preconditioner,
    curvature_preconditioner_block,
    curvature_preconditioner_diagonal,
    diagonal_preconditioner,
    expand_source_amplitude,
    gather_receiver_shards,
    index_shots,
    line_acquisition_2d,
    local_shot_positions,
    merge_receiver_batches,
    point_acquisition,
    receiver_gsot_loss,
    receiver_gsot_loss_shard,
    receiver_mse_loss,
    receiver_mse_loss_shard,
    receiver_sinkhorn_loss,
    receiver_sinkhorn_loss_shard,
    rank_shot_indices,
    split_rank_shots,
    split_shots,
    take_receiver_batch,
    take_receiver_shard_batch,
    take_shot_batch,
)
from tide.workflow.shots import _run_kernel_shot_batches as run_shot_batches

# --- test_optim_methods.py ---


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


def test_lbfgs_preserves_shape_and_float64_dtype() -> None:
    dtype = torch.float64
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

    steps = [
        event for event in events if event.event == tide.optim.OptimizerEventType.STEP
    ]
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


# --- test_workflow_shots.py ---


def test_split_shots_uses_long_indices_on_requested_device() -> None:
    batches = split_shots(5, 2, device=torch.device("cpu"))

    assert [batch.tolist() for batch in batches] == [[0, 1], [2, 3], [4]]
    assert all(batch.dtype == torch.long for batch in batches)
    assert all(batch.device.type == "cpu" for batch in batches)


def test_split_shots_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_shots"):
        split_shots(-1, 2)
    with pytest.raises(ValueError, match="batch_size"):
        split_shots(2, 0)


def test_rank_shot_indices_uses_round_robin_shards() -> None:
    shards = [
        rank_shot_indices(8, rank=rank, world_size=3).tolist() for rank in range(3)
    ]

    assert shards == [[0, 3, 6], [1, 4, 7], [2, 5]]
    assert [
        batch.tolist() for batch in split_rank_shots(10, 2, rank=1, world_size=3)
    ] == [[1, 4], [7]]


def test_local_shot_positions_maps_global_ids_to_shard_offsets() -> None:
    local = torch.tensor([1, 4, 7])
    positions = local_shot_positions(torch.tensor([4, 7]), local)

    torch.testing.assert_close(positions, torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="local shard"):
        local_shot_positions(torch.tensor([2]), local)


def test_index_shots_handles_shared_and_per_model_shot_axes() -> None:
    shared = torch.arange(4 * 2).reshape(4, 2)
    per_model = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    indices = torch.tensor([1, 3])

    torch.testing.assert_close(index_shots(shared, indices), shared[indices])
    torch.testing.assert_close(
        index_shots(per_model, indices, shot_dim=1),
        per_model[:, indices],
    )


def test_take_shot_batch_preserves_none_and_selects_all_locations() -> None:
    source_location = torch.arange(4 * 1 * 2).reshape(4, 1, 2)
    receiver_location = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
    indices = torch.tensor([0, 2])

    batch = take_shot_batch(
        source_amplitude=None,
        source_location=source_location,
        receiver_location=receiver_location,
        shot_indices=indices,
    )

    assert batch.source_amplitude is None
    torch.testing.assert_close(batch.source_location, source_location[indices])
    torch.testing.assert_close(batch.receiver_location, receiver_location[indices])


def test_point_acquisition_builds_shared_and_paired_receivers() -> None:
    source_points = torch.tensor([[1, 2], [1, 4], [1, 6]])
    shared_receivers = torch.tensor([[2, 3], [2, 5]])
    paired_receivers = torch.tensor([[2, 2], [2, 4], [2, 6]])

    shared = point_acquisition(source_points, shared_receivers, receiver_mode="shared")
    paired = point_acquisition(source_points, paired_receivers, receiver_mode="paired")

    assert shared.source_location.shape == (3, 1, 2)
    assert shared.receiver_location.shape == (3, 2, 2)
    assert paired.receiver_location.shape == (3, 1, 2)
    assert shared.n_shots == 3
    assert shared.n_receivers == 2
    assert shared.spatial_ndim == 2
    torch.testing.assert_close(shared.receiver_location[1], shared_receivers)
    torch.testing.assert_close(paired.receiver_location[:, 0], paired_receivers)


def test_line_acquisition_2d_builds_solver_locations() -> None:
    acquisition = line_acquisition_2d(
        torch.tensor([2, 4, 6]),
        torch.tensor([3, 5, 7]),
        source_depth=1,
        receiver_mode="paired",
    )

    expected_source = torch.tensor([[[1, 2]], [[1, 4]], [[1, 6]]])
    expected_receiver = torch.tensor([[[1, 3]], [[1, 5]], [[1, 7]]])
    torch.testing.assert_close(acquisition.source_location, expected_source)
    torch.testing.assert_close(acquisition.receiver_location, expected_receiver)


def test_expand_source_amplitude_handles_single_and_multi_source_wavelets() -> None:
    wavelet = torch.arange(4, dtype=torch.float32)
    multi_source = torch.stack([wavelet, wavelet + 1])

    single = expand_source_amplitude(wavelet, 3)
    multi = expand_source_amplitude(multi_source, 3, n_sources=2)

    assert single.shape == (3, 1, 4)
    assert multi.shape == (3, 2, 4)
    torch.testing.assert_close(single[2, 0], wavelet)
    torch.testing.assert_close(multi[1], multi_source)


def test_merge_receiver_batches_infers_tide_receiver_shot_axis() -> None:
    shared_chunks = [
        torch.full((3, 2, 1), 1.0),
        torch.full((3, 1, 1), 2.0),
    ]
    batched_model_chunks = [
        torch.full((3, 2, 2, 1), 1.0),
        torch.full((3, 2, 1, 1), 2.0),
    ]

    shared = merge_receiver_batches(shared_chunks)
    batched_model = merge_receiver_batches(batched_model_chunks)

    assert shared.shape == (3, 3, 1)
    assert batched_model.shape == (3, 2, 3, 1)
    torch.testing.assert_close(shared[:, :2], shared_chunks[0])
    torch.testing.assert_close(batched_model[:, :, :2], batched_model_chunks[0])


def test_receiver_batch_helpers_select_and_normalize_loss() -> None:
    observed = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    indices = torch.tensor([0, 2])
    predicted = observed[:, indices, :] + 1.0

    selected = take_receiver_batch(observed, indices)
    batch_loss = receiver_mse_loss(predicted, observed, indices)
    full_loss = receiver_mse_loss(
        predicted,
        observed,
        indices,
        normalization="all",
    )

    torch.testing.assert_close(selected, observed[:, indices, :])
    torch.testing.assert_close(batch_loss, torch.tensor(1.0))
    torch.testing.assert_close(
        full_loss, torch.tensor(predicted.numel() / observed.numel())
    )


def test_receiver_shard_batch_helpers_use_local_observed_columns() -> None:
    observed = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2)
    local_indices = torch.tensor([1, 4])
    global_batch = torch.tensor([4])
    predicted = observed[:, 1:2, :] + 2.0

    selected = take_receiver_shard_batch(observed, global_batch, local_indices)
    loss = receiver_mse_loss_shard(
        predicted,
        observed,
        global_batch,
        local_indices,
        global_observed_numel=4 * 5 * 2,
    )

    torch.testing.assert_close(selected, observed[:, 1:2, :])
    torch.testing.assert_close(
        loss, torch.tensor(predicted.numel() * 4.0 / (4 * 5 * 2))
    )


def test_receiver_sinkhorn_loss_matches_shots_and_backpropagates() -> None:
    pytest.importorskip("geomloss")
    observed = torch.zeros(8, 3, 1)
    observed[2, :, 0] = 1.0
    indices = torch.tensor([0, 2])
    predicted = torch.zeros(8, 2, 1, requires_grad=True)
    with torch.no_grad():
        predicted[3, :, 0] = 1.0

    loss = receiver_sinkhorn_loss(
        predicted,
        observed,
        indices,
        dt=0.1,
        p=1,
        blur=0.05,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_receiver_sinkhorn_loss_shard_selects_local_shots() -> None:
    pytest.importorskip("geomloss")
    observed = torch.zeros(8, 2, 1)
    observed[2, :, 0] = 1.0
    predicted = observed[:, 1:2].clone().requires_grad_()

    loss = receiver_sinkhorn_loss_shard(
        predicted,
        observed,
        torch.tensor([4]),
        torch.tensor([1, 4]),
        dt=0.1,
        blur=0.05,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_receiver_gsot_loss_uses_hard_assignment_and_backpropagates() -> None:
    observed = torch.tensor([0.0, 0.0, 1.0]).reshape(3, 1, 1)
    predicted = torch.tensor([1.0, 0.0, 0.0]).reshape(3, 1, 1).requires_grad_()

    loss = receiver_gsot_loss(
        predicted,
        observed,
        torch.tensor([0]),
        dt=1.0,
        p=2,
        max_time_shift=0.1,
        observed_energy_weighting=False,
    )
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(2.0))
    assert predicted.grad is not None
    torch.testing.assert_close(predicted.grad[:, 0, 0], torch.tensor([2.0, 0.0, -2.0]))


def test_receiver_gsot_loss_shard_selects_local_shots() -> None:
    observed = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2, 1)
    predicted = observed[:, 1:2].clone().requires_grad_()

    loss = receiver_gsot_loss_shard(
        predicted,
        observed,
        torch.tensor([4]),
        torch.tensor([1, 4]),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_gather_receiver_shards_noops_without_distributed_context() -> None:
    receiver = torch.arange(4 * 2 * 1, dtype=torch.float32).reshape(4, 2, 1)

    gathered = gather_receiver_shards(receiver, torch.tensor([0, 1]), 2)

    assert gathered is receiver


def test_backward_shot_batches_accumulates_full_gradient() -> None:
    x = torch.arange(1, 6, dtype=torch.float32)
    observed = 2.5 * x
    shot_batches = split_shots(x.numel(), 2)
    scale = torch.tensor(0.25, requires_grad=True)

    def clear_grad() -> None:
        scale.grad = None

    def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
        residual = scale * x[shot_indices] - observed[shot_indices]
        return residual.square().sum() / observed.numel()

    total_loss = backward_shot_batches(
        batch_loss,
        shot_batches,
        zero_grad=clear_grad,
    )

    expected_scale = torch.tensor(0.25, requires_grad=True)
    expected_loss = ((expected_scale * x - observed).square()).sum() / observed.numel()
    expected_loss.backward()

    torch.testing.assert_close(torch.tensor(total_loss), expected_loss.detach())
    assert scale.grad is not None
    assert expected_scale.grad is not None
    torch.testing.assert_close(scale.grad, expected_scale.grad)


def test_backward_shot_batches_can_inspect_per_batch_gradients() -> None:
    x = torch.arange(1, 5, dtype=torch.float32)
    observed = 3.0 * x
    shot_batches = split_shots(x.numel(), 2)
    scale = torch.tensor(1.0, requires_grad=True)
    per_batch_grads: list[torch.Tensor] = []

    def clear_grad() -> None:
        scale.grad = None

    def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
        residual = scale * x[shot_indices] - observed[shot_indices]
        return residual.square().sum()

    def record_grad(_shot_indices: torch.Tensor, _loss: torch.Tensor) -> None:
        assert scale.grad is not None
        per_batch_grads.append(scale.grad.detach().clone())

    total_loss = backward_shot_batches(
        batch_loss,
        shot_batches,
        zero_grad=clear_grad,
        zero_each_batch=True,
        after_backward=record_grad,
    )

    expected_grads = []
    expected_loss = 0.0
    for shot_indices in shot_batches:
        residual = scale.detach() * x[shot_indices] - observed[shot_indices]
        expected_loss += float(residual.square().sum())
        expected_grads.append((2.0 * residual * x[shot_indices]).sum())

    assert len(per_batch_grads) == len(expected_grads)
    torch.testing.assert_close(
        torch.stack(per_batch_grads), torch.stack(expected_grads)
    )
    torch.testing.assert_close(torch.tensor(total_loss), torch.tensor(expected_loss))


def test_curvature_preconditioner_diagonal_normalizes_clips_and_masks() -> None:
    curvature = torch.tensor(
        [
            [0.0, 1.0, 4.0],
            [float("nan"), float("inf"), 9.0],
        ],
        dtype=torch.float32,
    )
    inactive = torch.tensor(
        [
            [False, False, False],
            [True, True, False],
        ]
    )

    diagonal = curvature_preconditioner_diagonal(
        curvature,
        inactive_mask=inactive,
        damping=0.1,
        power=0.5,
        clip_min=0.5,
        clip_max=2.0,
        blend=0.75,
    )

    assert diagonal.shape == curvature.shape
    assert diagonal.dtype == curvature.dtype
    assert torch.all(torch.isfinite(diagonal))
    assert torch.all(diagonal[~inactive] >= 0.5)
    assert torch.all(diagonal[~inactive] <= 2.0)
    torch.testing.assert_close(diagonal[inactive], torch.zeros_like(diagonal[inactive]))


def test_curvature_preconditioner_diagonal_preserves_float64_dynamic_range() -> None:
    curvature = torch.tensor([1e40, 1e41], dtype=torch.float64)

    diagonal = curvature_preconditioner_diagonal(
        curvature,
        damping=0.0,
        power=0.5,
    )

    assert diagonal.dtype == torch.float64
    assert torch.all(torch.isfinite(diagonal))
    assert torch.all(diagonal > 0)
    assert diagonal[0] > diagonal[1]


def test_diagonal_preconditioner_matches_tide_optim_callback_contract() -> None:
    diagonal = torch.tensor([2.0, 0.5], dtype=torch.float32)
    preconditioner = diagonal_preconditioner(diagonal)
    x = torch.zeros(2)
    vector = torch.tensor([3.0, 4.0], dtype=torch.float32)

    out = preconditioner(x, vector)

    torch.testing.assert_close(out, torch.tensor([6.0, 2.0]))


def test_curvature_preconditioner_block_normalizes_clips_and_masks() -> None:
    curvature_11 = torch.tensor(
        [[[1.0, 4.0], [float("nan"), 9.0]]],
        dtype=torch.float32,
    )
    curvature_22 = torch.tensor([[[2.0, 8.0], [3.0, float("inf")]]])
    curvature_12 = torch.tensor([[[0.25, -0.5], [1.0, 2.0]]])
    inactive = torch.tensor([[[False, False], [True, False]]])

    block = curvature_preconditioner_block(
        curvature_11,
        curvature_22,
        curvature_12,
        inactive_mask=inactive,
        damping=0.1,
        power=0.5,
        clip_min=0.25,
        clip_max=4.0,
        blend=0.75,
    )

    assert block.diag11.shape == curvature_11.shape
    assert block.offdiag12.shape == curvature_11.shape
    assert block.diag22.shape == curvature_11.shape
    assert torch.all(torch.isfinite(block.diag11))
    assert torch.all(torch.isfinite(block.offdiag12))
    assert torch.all(torch.isfinite(block.diag22))
    assert torch.all(block.diag11[~inactive] >= 0.25)
    assert torch.all(block.diag22[~inactive] >= 0.25)
    assert torch.all(block.diag11[~inactive] <= 4.0)
    assert torch.all(block.diag22[~inactive] <= 4.0)
    torch.testing.assert_close(block.diag11[inactive], torch.zeros(1))
    torch.testing.assert_close(block.offdiag12[inactive], torch.zeros(1))
    torch.testing.assert_close(block.diag22[inactive], torch.zeros(1))


def test_block_preconditioner_matches_tide_optim_callback_contract() -> None:
    block = tide.workflow.BlockPreconditioner(
        diag11=torch.tensor([2.0, 3.0]),
        offdiag12=torch.tensor([0.5, -1.0]),
        diag22=torch.tensor([4.0, 5.0]),
    )
    preconditioner = block_preconditioner(block)
    x = torch.zeros(4)
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    out = preconditioner(x, vector)

    torch.testing.assert_close(
        out,
        torch.tensor([3.5, 2.0, 12.5, 18.0]),
    )


def test_run_shot_batches_preserves_autograd() -> None:
    source_amplitude = torch.arange(4 * 1 * 3, dtype=torch.float32).reshape(4, 1, 3)
    source_location = torch.zeros(4, 1, 2, dtype=torch.long)
    receiver_location = torch.zeros(4, 1, 2, dtype=torch.long)
    weight = torch.tensor(2.0, requires_grad=True)

    def solver(
        *,
        source_amplitude: torch.Tensor,
        source_location: torch.Tensor,
        receiver_location: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        assert source_location.shape[0] == receiver_location.shape[0]
        return source_amplitude[:, 0, :].transpose(0, 1).unsqueeze(-1) * weight

    receiver = run_shot_batches(
        solver,
        n_shots=4,
        batch_size=2,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        weight=weight,
    )
    loss = receiver.square().sum()
    loss.backward()

    assert receiver.shape == (3, 4, 1)
    assert weight.grad is not None
    assert float(weight.grad) > 0.0


def _tm_example() -> MaxwellExample:
    example = make_tm2d_example(
        shape=(7, 8),
        nt=6,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=80e6,
        sigma=1e-3,
        source_location=(3, 2),
        receiver_locations=((3, 5),),
        pml_width=1,
        stencil=2,
        python_backend=True,
    )
    return example.updated(
        source_amplitude=example.source_amplitude.repeat(3, 1, 1),
        source_location=torch.tensor([[[3, 2]], [[3, 3]], [[3, 4]]]),
        receiver_location=torch.tensor([[[3, 5]], [[3, 5]], [[3, 5]]]),
    )


def test_run_shot_batches_matches_full_maxwelltm_python_call() -> None:
    example = _tm_example()
    arguments = example.arguments()
    full = example.run()[-1]
    batched = run_shot_batches(
        example.solver,
        n_shots=3,
        batch_size=2,
        **arguments,
    )
    torch.testing.assert_close(batched, full)


def test_run_shot_batches_preserves_batched_model_receiver_shape() -> None:
    example = _tm_example()
    example = example.updated(
        epsilon=torch.stack([example.epsilon, example.epsilon * 1.1]),
        sigma=torch.stack([example.sigma, example.sigma * 1.2]),
        mu=torch.stack([example.mu, example.mu]),
    )
    arguments = example.arguments()
    full = example.run()[-1]
    batched = run_shot_batches(
        example.solver,
        n_shots=3,
        batch_size=1,
        **arguments,
    )
    assert batched.shape == (6, 2, 3, 1)
    torch.testing.assert_close(batched, full)


def _maxwell3d_example() -> MaxwellExample:
    example = make_maxwell3d_example(
        shape=(5, 6, 7),
        nt=5,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=70e6,
        sigma=1e-3,
        source_location=(2, 2, 2),
        receiver_locations=((2, 2, 4),),
        pml_width=1,
        python_backend=True,
    )
    return example.updated(
        source_amplitude=example.source_amplitude.repeat(3, 1, 1),
        source_location=torch.tensor([[[2, 2, 2]], [[2, 3, 2]], [[2, 4, 2]]]),
        receiver_location=torch.tensor([[[2, 2, 4]], [[2, 3, 4]], [[2, 4, 4]]]),
    )


def test_run_shot_batches_matches_full_maxwell3d_python_call() -> None:
    example = _maxwell3d_example()
    arguments = example.arguments()
    full = example.run()[-1]
    batched = run_shot_batches(
        example.solver,
        n_shots=3,
        batch_size=2,
        **arguments,
    )
    torch.testing.assert_close(batched, full)


# --- test_workflow_optim_integration.py ---


def _workflow_stopping(
    max_iter: int, max_evaluations: int
) -> tide.optim.StoppingCriteria:
    return tide.optim.StoppingCriteria(
        max_iter=max_iter,
        max_evaluations=max_evaluations,
        gtol=1e-6,
        ftol=1e-12,
        xtol=1e-12,
    )


def test_workflow_shot_batches_drive_torch_native_objective() -> None:
    n_shots = 5
    nt = 4
    batch_size = 2
    source_amplitude = torch.arange(
        n_shots * nt,
        dtype=torch.float32,
    ).reshape(n_shots, 1, nt)
    source_location = torch.zeros(n_shots, 1, 2, dtype=torch.long)
    receiver_location = torch.zeros(n_shots, 1, 2, dtype=torch.long)
    shot_batches = tide.workflow.split_shots(n_shots, batch_size)

    def solver(
        *,
        scale: torch.Tensor,
        source_amplitude: torch.Tensor,
        source_location: torch.Tensor,
        receiver_location: torch.Tensor,
    ) -> torch.Tensor:
        assert source_location.shape[0] == receiver_location.shape[0]
        return source_amplitude[:, 0, :].transpose(0, 1).unsqueeze(-1) * scale

    observed = tide.workflow.shots._run_kernel_shot_batches(
        solver,
        n_shots=n_shots,
        batch_size=batch_size,
        scale=torch.tensor(1.5),
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
    )

    def objective(x: torch.Tensor) -> tuple[float, torch.Tensor]:
        scale = x.detach().clone().requires_grad_(True)

        def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
            batch = tide.workflow.take_shot_batch(
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                shot_indices=shot_indices,
            )
            pred = solver(
                scale=scale,
                source_amplitude=batch.source_amplitude,
                source_location=batch.source_location,
                receiver_location=batch.receiver_location,
            )
            return tide.workflow.receiver_mse_loss(
                pred,
                observed,
                shot_indices,
                normalization="all",
            )

        total_loss = tide.workflow.backward_shot_batches(batch_loss, shot_batches)
        assert scale.grad is not None
        return total_loss, scale.grad.detach()

    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([0.25], dtype=torch.float32),
        options=tide.optim.LBFGSOptions(stopping=_workflow_stopping(20, 80)),
    )

    assert result.success, result.status
    torch.testing.assert_close(result.x, torch.tensor([1.5]), atol=1e-4, rtol=1e-4)


def test_workflow_diagonal_preconditioner_drives_tide_optim() -> None:
    target = torch.tensor([1.0, -2.0])
    hessian_diag = torch.tensor([10.0, 0.25])

    def objective(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x - target
        grad = hessian_diag * residual
        return 0.5 * torch.dot(residual, grad), grad

    preconditioner = tide.workflow.diagonal_preconditioner(1.0 / hessian_diag)
    result = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([8.0, 8.0]),
        preconditioner=preconditioner,
        options=tide.optim.LBFGSOptions(stopping=_workflow_stopping(10, 40)),
    )

    assert result.success, result.status
    assert result.n_prec > 0
    torch.testing.assert_close(result.x, target, atol=1e-4, rtol=1e-4)


# --- test_structured_batches.py ---


def _skip_if_no_backend() -> None:
    if not backend_utils.is_backend_available():
        pytest.skip("native backend unavailable")


def _structured_tm_example(device: torch.device) -> MaxwellExample:
    example = make_tm2d_example(
        shape=(8, 9),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=80e6,
        device=device,
        source_location=(4, 4),
        receiver_locations=((4, 6),),
        pml_width=2,
        stencil=2,
    )
    return example.updated(
        epsilon=torch.stack((example.epsilon, example.epsilon + 0.5)),
        sigma=torch.stack(
            (
                torch.full_like(example.sigma, 1e-3),
                torch.full_like(example.sigma, 2e-3),
            )
        ),
        mu=torch.stack((example.mu, example.mu)),
        source_amplitude=example.source_amplitude.repeat(2, 1, 1),
        source_location=torch.tensor(
            [[[4, 4]], [[4, 5]]],
            device=device,
        ),
        receiver_location=torch.tensor(
            [[[4, 6]], [[4, 7]]],
            device=device,
        ),
    )


def _structured_maxwell3d_example(device: torch.device) -> MaxwellExample:
    example = make_maxwell3d_example(
        shape=(5, 6, 7),
        nt=8,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=70e6,
        device=device,
        source_location=(2, 2, 2),
        receiver_locations=((2, 2, 4),),
        pml_width=1,
    )
    return example.updated(
        epsilon=torch.stack((example.epsilon, example.epsilon + 0.5)),
        sigma=torch.stack(
            (
                torch.full_like(example.sigma, 1e-3),
                torch.full_like(example.sigma, 2e-3),
            )
        ),
        mu=torch.stack((example.mu, example.mu)),
        source_amplitude=example.source_amplitude.repeat(2, 1, 1),
        source_location=torch.tensor(
            [[[2, 2, 2]], [[2, 3, 2]]],
            device=device,
        ),
        receiver_location=torch.tensor(
            [[[2, 2, 4]], [[2, 3, 4]]],
            device=device,
        ),
    )


def _assert_batched_forward_matches_loop(
    example: MaxwellExample,
    *,
    python_backend: bool,
) -> None:
    output = example.run(python_backend=python_backend)
    expected_receiver = torch.stack(
        [
            example.run(
                epsilon=example.epsilon[index],
                sigma=example.sigma[index],
                mu=example.mu[index],
                python_backend=python_backend,
            )[-1]
            for index in range(example.epsilon.shape[0])
        ],
        dim=1,
    )
    assert output[-1].shape == (
        example.source_amplitude.shape[-1],
        example.epsilon.shape[0],
        example.source_amplitude.shape[0],
        example.receiver_location.shape[1],
    )
    assert output[0].shape[:2] == (
        example.epsilon.shape[0],
        example.source_amplitude.shape[0],
    )
    torch.testing.assert_close(output[-1], expected_receiver)


def _assert_batched_backward_matches_loop(
    example: MaxwellExample,
    *,
    python_backend: bool,
) -> None:
    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    example.run(
        epsilon=epsilon,
        sigma=sigma,
        python_backend=python_backend,
    )[-1].square().sum().backward()
    assert epsilon.grad is not None

    gradients = []
    for index in range(epsilon.shape[0]):
        epsilon_i = epsilon.detach()[index].clone().requires_grad_(True)
        sigma_i = sigma.detach()[index].clone().requires_grad_(True)
        example.run(
            epsilon=epsilon_i,
            sigma=sigma_i,
            mu=example.mu[index],
            python_backend=python_backend,
        )[-1].square().sum().backward()
        assert epsilon_i.grad is not None
        gradients.append(epsilon_i.grad)
    torch.testing.assert_close(epsilon.grad, torch.stack(gradients))


def test_maxwelltm_batched_models_shared_shots_forward_matches_loop():
    _skip_if_no_backend()
    _assert_batched_forward_matches_loop(
        _structured_tm_example(torch.device("cpu")),
        python_backend=False,
    )


def test_maxwelltm_batched_models_per_model_shots_backward_matches_loop():
    _skip_if_no_backend()
    example = _structured_tm_example(torch.device("cpu"))
    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    source_amplitude = (
        example.source_amplitude.unsqueeze(0).expand(2, -1, -1, -1).clone()
    )
    source_amplitude[1] *= 0.75
    source_location = example.source_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    receiver_location = (
        example.receiver_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    )
    receiver_location[1, :, 0, 1] -= 1

    receivers = example.run(
        epsilon=epsilon,
        sigma=sigma,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        python_backend=False,
    )[-1]
    receivers.square().sum().backward()
    assert epsilon.grad is not None

    gradients = []
    for index in range(epsilon.shape[0]):
        epsilon_i = epsilon.detach()[index].clone().requires_grad_(True)
        sigma_i = sigma.detach()[index].clone().requires_grad_(True)
        receiver_i = example.run(
            epsilon=epsilon_i,
            sigma=sigma_i,
            mu=example.mu[index],
            source_amplitude=source_amplitude[index],
            source_location=source_location[index],
            receiver_location=receiver_location[index],
            python_backend=False,
        )[-1]
        receiver_i.square().sum().backward()
        assert epsilon_i.grad is not None
        gradients.append(epsilon_i.grad)

    torch.testing.assert_close(
        epsilon.grad,
        torch.stack(gradients),
        atol=2e-5,
        rtol=1e-5,
        equal_nan=True,
    )


def test_maxwelltm_batched_model_callbacks_expose_structured_shapes():
    _skip_if_no_backend()
    example = _structured_tm_example(torch.device("cpu"))
    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    seen: dict[str, tuple[int, ...]] = {}

    def forward_cb(state: tide.CallbackState) -> None:
        if "forward_wavefield" not in seen:
            seen["forward_wavefield"] = tuple(state.get_wavefield("Ey").shape)
            seen["forward_model"] = tuple(state.get_model("epsilon").shape)

    def backward_cb(state: tide.CallbackState) -> None:
        if "backward_gradient" not in seen:
            seen["backward_gradient"] = tuple(state.get_gradient("epsilon").shape)

    receivers = example.run(
        epsilon=epsilon,
        sigma=sigma,
        python_backend=False,
        forward_callback=forward_cb,
        backward_callback=backward_cb,
    )[-1]
    receivers.square().sum().backward()

    assert seen["forward_wavefield"] == (2, 2, 8, 9)
    assert seen["forward_model"] == (2, 8, 9)
    assert seen["backward_gradient"] == (2, 8, 9)


def test_maxwell3d_batched_models_shared_shots_forward_matches_loop():
    _skip_if_no_backend()
    _assert_batched_forward_matches_loop(
        _structured_maxwell3d_example(torch.device("cpu")),
        python_backend=False,
    )


def test_maxwell3d_batched_models_shared_shots_backward_matches_loop():
    _skip_if_no_backend()
    _assert_batched_backward_matches_loop(
        _structured_maxwell3d_example(torch.device("cpu")),
        python_backend=False,
    )


def test_maxwelltm_batched_models_python_backend_forward_matches_loop():
    _assert_batched_forward_matches_loop(
        _structured_tm_example(torch.device("cpu")),
        python_backend=True,
    )


def test_maxwell3d_batched_models_python_backend_backward_matches_loop():
    _assert_batched_backward_matches_loop(
        _structured_maxwell3d_example(torch.device("cpu")),
        python_backend=True,
    )


def test_batched_models_python_backend_callbacks_rejected():
    device = torch.device("cpu")
    for example in (
        _structured_tm_example(device),
        _structured_maxwell3d_example(device),
    ):
        with pytest.raises(NotImplementedError):
            example.run(
                python_backend=True,
                forward_callback=lambda state: None,
            )


def test_batched_models_validate_B_and_S_mismatch():
    example = _structured_tm_example(torch.device("cpu"))
    bad_source = example.source_amplitude.unsqueeze(0).expand(3, -1, -1, -1).clone()
    with pytest.raises((RuntimeError, TypeCheckError)):
        example.run(
            source_amplitude=bad_source,
            python_backend=False,
        )

    bad_receiver = example.receiver_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    bad_receiver = bad_receiver[:, :1]
    with pytest.raises((RuntimeError, TypeCheckError)):
        example.run(
            receiver_location=bad_receiver,
            python_backend=False,
        )
