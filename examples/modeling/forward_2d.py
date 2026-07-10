"""Run a minimal 2D TM forward model with the Python backend."""

from __future__ import annotations

import torch

import tide


def main() -> None:
    ny, nx = 32, 48
    epsilon = torch.full((ny, nx), 4.0)
    epsilon[18:, :] = 7.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_amplitude = tide.ricker(200e6, 160, 4e-11).reshape(1, 1, -1)
    source_location = torch.tensor([[[8, 20]]], dtype=torch.long)
    receiver_location = torch.tensor([[[8, 28]]], dtype=torch.long)

    receiver_data = tide.maxwelltm(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=4,
        stencil=2,
        python_backend=True,
    )[-1]
    print(f"receiver data shape: {tuple(receiver_data.shape)}")


if __name__ == "__main__":
    main()
