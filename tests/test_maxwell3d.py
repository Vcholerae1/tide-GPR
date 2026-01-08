"""Tests for 3D Maxwell solver."""

import pytest
import torch

import tide
import tide.backend_utils as backend_utils


def _setup_3d(device: torch.device, dtype: torch.dtype) -> dict:
    nz, ny, nx = 6, 5, 4
    nt = 6

    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[nz // 2, ny // 2, nx // 2]]],
        device=device,
        dtype=torch.long,
    )
    receiver_location = torch.tensor(
        [[[nz // 2, ny // 2, nx // 2]]],
        device=device,
        dtype=torch.long,
    )

    source_amplitude = torch.zeros(1, 1, nt, device=device, dtype=dtype)
    source_amplitude[..., 0] = 1.0

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "nt": nt,
    }


def test_maxwell3d_forward_python_cpu():
    """Python backend should run and produce finite outputs."""
    device = torch.device("cpu")
    dtype = torch.float32

    s = _setup_3d(device, dtype)

    outputs = tide.maxwell3d(
        s["epsilon"],
        s["sigma"],
        s["mu"],
        grid_spacing=(0.02, 0.02, 0.02),
        dt=4e-11,
        source_amplitude=s["source_amplitude"],
        source_location=s["source_location"],
        receiver_location=s["receiver_location"],
        stencil=2,
        pml_width=0,
        nt=s["nt"],
        python_backend=True,
        source_component="Ez",
        receiver_component="Ez",
    )

    Ex, Ey, Ez, Hx, Hy, Hz = outputs[:6]
    receiver_amplitudes = outputs[-1]

    assert receiver_amplitudes.shape == (s["nt"], 1, 1)
    assert torch.isfinite(receiver_amplitudes).all()

    for field in (Ex, Ey, Ez, Hx, Hy, Hz):
        assert field.ndim == 4
        assert field.shape[0] == 1
        assert torch.isfinite(field).all()


def test_maxwell3d_forward_c_matches_python_cpu():
    """C backend (if available) should match Python backend on CPU."""
    device = torch.device("cpu")
    dtype = torch.float32

    if not backend_utils.is_backend_available():
        pytest.skip("C/CUDA backend not available")

    try:
        backend_utils.get_backend_function("maxwell_3d", "forward", 2, dtype, device)
    except (RuntimeError, AttributeError, TypeError):
        pytest.skip("maxwell_3d CPU backend not available in the shared library")

    s = _setup_3d(device, dtype)

    out_py = tide.maxwell3d(
        s["epsilon"],
        s["sigma"],
        s["mu"],
        grid_spacing=(0.02, 0.02, 0.02),
        dt=4e-11,
        source_amplitude=s["source_amplitude"],
        source_location=s["source_location"],
        receiver_location=s["receiver_location"],
        stencil=2,
        pml_width=0,
        nt=s["nt"],
        python_backend=True,
        source_component="Ez",
        receiver_component="Ez",
    )

    out_c = tide.maxwell3d(
        s["epsilon"],
        s["sigma"],
        s["mu"],
        grid_spacing=(0.02, 0.02, 0.02),
        dt=4e-11,
        source_amplitude=s["source_amplitude"],
        source_location=s["source_location"],
        receiver_location=s["receiver_location"],
        stencil=2,
        pml_width=0,
        nt=s["nt"],
        python_backend=False,
        source_component="Ez",
        receiver_component="Ez",
    )

    torch.testing.assert_close(
        out_c[-1], out_py[-1], rtol=1e-4, atol=1e-5
    )
