import numpy as np
import torch

import tide


def test_workflow_shot_batches_drive_tide_optim_objective() -> None:
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

    observed = tide.workflow.run_shot_batches(
        solver,
        n_shots=n_shots,
        batch_size=batch_size,
        scale=torch.tensor(1.5),
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
    )

    def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
        scale = torch.tensor(float(x[0]), dtype=torch.float32, requires_grad=True)

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
        grad_out[0] = float(scale.grad.detach())
        return total_loss

    result = tide.optim.lbfgs_minimize(
        objective,
        np.array([0.25], dtype=np.float32),
        options=tide.optim.LBFGSOptions(
            max_iter=20,
            tolerance=1e-8,
            max_evaluations=80,
        ),
    )

    assert result.success, result.status
    np.testing.assert_allclose(result.x, np.array([1.5], dtype=np.float32), atol=1e-4)


def test_workflow_diagonal_preconditioner_drives_tide_optim() -> None:
    target = np.array([1.0, -2.0], dtype=np.float32)
    hessian_diag = np.array([10.0, 0.25], dtype=np.float32)

    def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
        residual = x - target
        grad_out[:] = hessian_diag * residual
        return float(0.5 * np.dot(residual, grad_out))

    preconditioner = tide.workflow.diagonal_preconditioner(1.0 / hessian_diag)
    result = tide.optim.lbfgs_minimize(
        objective,
        np.array([8.0, 8.0], dtype=np.float32),
        preconditioner=preconditioner,
        options=tide.optim.LBFGSOptions(max_iter=10, max_evaluations=40),
    )

    assert result.success, result.status
    assert result.n_prec > 0
    np.testing.assert_allclose(result.x, target, atol=1e-4)
