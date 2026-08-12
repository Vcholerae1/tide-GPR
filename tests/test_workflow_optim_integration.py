import torch

import tide


def _stopping(max_iter: int, max_evaluations: int) -> tide.optim.StoppingCriteria:
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
        options=tide.optim.LBFGSOptions(stopping=_stopping(20, 80)),
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
        options=tide.optim.LBFGSOptions(stopping=_stopping(10, 40)),
    )

    assert result.success, result.status
    assert result.n_prec > 0
    torch.testing.assert_close(result.x, target, atol=1e-4, rtol=1e-4)
