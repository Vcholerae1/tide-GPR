"""Run a minimal structured 2-D TM forward model."""

from __future__ import annotations

import torch

import tide


def main() -> None:
    ny, nx = 32, 48
    epsilon = torch.full((ny, nx), 4.0)
    epsilon[18:, :] = 7.0
    model = tide.EMModel(
        epsilon=epsilon,
        sigma=torch.zeros_like(epsilon),
        mu=torch.ones_like(epsilon),
    )

    experiment = tide.Experiment(
        acquisition=tide.Acquisition(
            source_location=torch.tensor([[[8, 20]]], dtype=torch.long),
            receiver_location=torch.tensor([[[8, 28]]], dtype=torch.long),
        ),
        source_amplitude=tide.ricker(200e6, 160, 4e-11).reshape(1, 1, -1),
    )
    operator = tide.MaxwellTM(
        tide.Discretization(
            spacing=0.02,
            dt=4e-11,
            stencil=2,
            boundary=tide.CPML(width=4),
        ),
        experiment,
        execution=tide.ExecutionOptions(backend=tide.BackendPreference.REFERENCE),
    )

    result = operator(model)
    print(f"receiver data shape: {tuple(result.receiver_data.shape)}")


if __name__ == "__main__":
    main()
