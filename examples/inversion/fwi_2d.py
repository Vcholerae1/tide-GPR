"""Minimal shot-batched inversion with tide.workflow and tide.optim."""

from __future__ import annotations

import torch

import tide


def main() -> None:
    device = torch.device("cpu")
    dtype = torch.float32

    ny, nx = 7, 8
    n_shots = 3
    nt = 6
    batch_size = 2
    dx = 0.02
    dt = 4e-11

    wavelet = tide.ricker(80e6, nt, dt, dtype=dtype, device=device)
    source_amplitude = tide.workflow.expand_source_amplitude(wavelet, n_shots)
    acquisition = tide.workflow.line_acquisition_2d(
        torch.tensor([2, 3, 4], device=device),
        torch.tensor([5, 5, 5], device=device),
        source_depth=3,
        receiver_mode="paired",
    )
    source_location = acquisition.source_location
    receiver_location = acquisition.receiver_location

    sigma = torch.full((ny, nx), 1e-3, dtype=dtype, device=device)
    mu = torch.ones((ny, nx), dtype=dtype, device=device)
    shot_batches = tide.workflow.split_shots(n_shots, batch_size, device=device)

    def maxwell_receivers(epsilon: torch.Tensor) -> torch.Tensor:
        return tide.workflow.shots._run_kernel_shot_batches(
            tide.maxwell._kernel_api.maxwelltm,
            n_shots=n_shots,
            batch_size=batch_size,
            epsilon=epsilon,
            sigma=sigma,
            mu=mu,
            grid_spacing=dx,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=1,
            stencil=2,
            python_backend=True,
        )

    with torch.no_grad():
        observed = maxwell_receivers(
            torch.full((ny, nx), 4.0, dtype=dtype, device=device)
        )

    def objective(x: torch.Tensor) -> tuple[float, torch.Tensor]:
        epsilon_value = x.detach().clone().requires_grad_(True)
        epsilon = epsilon_value.expand(ny, nx)

        def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
            batch = tide.workflow.take_shot_batch(
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                shot_indices=shot_indices,
            )
            predicted = tide.maxwell._kernel_api.maxwelltm(
                epsilon=epsilon,
                sigma=sigma,
                mu=mu,
                grid_spacing=dx,
                dt=dt,
                source_amplitude=batch.source_amplitude,
                source_location=batch.source_location,
                receiver_location=batch.receiver_location,
                pml_width=1,
                stencil=2,
                python_backend=True,
            )[-1]
            return tide.workflow.receiver_mse_loss(
                predicted,
                observed,
                shot_indices,
                normalization="all",
            )

        total_loss = tide.workflow.backward_shot_batches(batch_loss, shot_batches)

        if epsilon_value.grad is None:
            raise RuntimeError("objective did not produce a gradient")
        return total_loss, epsilon_value.grad.detach()

    result: tide.optim.OptimizerResult = tide.optim.lbfgs_minimize(
        objective,
        torch.tensor([3.0], dtype=dtype, device=device),
        lower_bounds=1.0,
        upper_bounds=9.0,
        options=tide.optim.LBFGSOptions(
            stopping=tide.optim.StoppingCriteria(
                max_iter=5,
                max_evaluations=20,
            )
        ),
    )

    print(f"estimated epsilon = {result.x[0].item():.4f}")
    print(f"final loss = {result.f:.6g}")


if __name__ == "__main__":
    main()
