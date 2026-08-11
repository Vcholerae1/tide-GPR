from __future__ import annotations

import pytest
import torch

import tide


@pytest.fixture
def tm2d_numerical_case() -> dict[str, object]:
    dtype = torch.float64
    device = torch.device("cpu")
    ny, nx, nt = 28, 36, 180
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype, device=device)
    sigma = torch.full_like(epsilon, 2.0e-4)
    mu = torch.ones_like(epsilon)
    source_amplitude = tide.ricker(
        250e6,
        nt,
        3.0e-11,
        peak_time=2.5e-9,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    source_location = torch.tensor([[[ny // 2, 9]]], dtype=torch.long)
    receiver_location = torch.tensor([[[ny // 2, 14], [ny // 2, 20]]], dtype=torch.long)
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "grid_spacing": [0.018, 0.022],
        "dt": 3.0e-11,
        "pml_width": 4,
    }


@pytest.fixture
def em3d_numerical_case() -> dict[str, object]:
    dtype = torch.float32
    device = torch.device("cpu")
    nz, ny, nx, nt = 10, 11, 12, 80
    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype, device=device)
    sigma = torch.full_like(epsilon, 2.0e-4)
    mu = torch.ones_like(epsilon)
    source_amplitude = tide.ricker(
        350e6,
        nt,
        2.0e-11,
        peak_time=2.0e-9,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    source_location = torch.tensor([[[5, 5, 4]]], dtype=torch.long)
    receiver_location = torch.tensor([[[5, 5, 7], [5, 7, 7]]], dtype=torch.long)
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "grid_spacing": [0.016, 0.018, 0.022],
        "dt": 2.0e-11,
        "pml_width": 2,
    }
