"""Numerical parity between the native CPU backend and the Python reference.

These tests gate the native C/C++ CPU path (built by CMake when CUDA is
absent) against the Python reference for forward and Born operators in 2D and
3D, plus the policy routing that must happen before any adapter runs.
"""

import pytest
import torch

import tide
from tide import backend_utils


def _requires_native_backend() -> None:
    if not backend_utils.is_backend_available():
        pytest.skip("native backend is not available")


def _case_2d(device: torch.device) -> dict[str, torch.Tensor]:
    dtype = torch.float64
    ny, nx, nt = 12, 12, 16
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype, device=device)
    epsilon[ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1] = 4.4
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_amplitude = tide.ricker(
        120e6, nt, 3.5e-11, peak_time=1.0 / 120e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    source_location = torch.tensor(
        [[[ny // 2, nx // 4]]], dtype=torch.long, device=device
    )
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
    )
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
    }


def _case_3d(device: torch.device) -> dict[str, torch.Tensor]:
    dtype = torch.float32
    nz, ny, nx, nt = 6, 6, 7, 10
    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype, device=device)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_amplitude = tide.ricker(
        90e6, nt, 4e-11, peak_time=1.0 / 90e6, dtype=dtype, device=device
    ).view(1, 1, nt)
    source_location = torch.tensor(
        [[[2, 2, 2]]], dtype=torch.long, device=device
    )
    receiver_location = torch.tensor(
        [[[2, 2, 4]]], dtype=torch.long, device=device
    )
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
    }


def test_native_cpu_tm2d_forward_matches_python() -> None:
    _requires_native_backend()
    case = _case_2d(torch.device("cpu"))
    out_py = tide.maxwelltm(
        **case, grid_spacing=0.02, dt=3.5e-11, pml_width=3, python_backend=True
    )
    out_native = tide.maxwelltm(
        **case, grid_spacing=0.02, dt=3.5e-11, pml_width=3, python_backend=False
    )
    for a, b in zip(out_py, out_native, strict=True):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)


def test_native_cpu_tm2d_born_matches_python() -> None:
    _requires_native_backend()
    case = _case_2d(torch.device("cpu"))
    depsilon = 0.03 * torch.randn_like(case["epsilon"])
    out_py = tide.borntm(
        **case,
        depsilon=depsilon,
        grid_spacing=0.02,
        dt=3.5e-11,
        pml_width=3,
        python_backend=True,
    )
    out_native = tide.borntm(
        **case,
        depsilon=depsilon,
        grid_spacing=0.02,
        dt=3.5e-11,
        pml_width=3,
        python_backend=False,
    )
    for a, b in zip(out_py, out_native, strict=True):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)


def test_native_cpu_em3d_forward_matches_python() -> None:
    _requires_native_backend()
    case = _case_3d(torch.device("cpu"))
    out_py = tide.maxwell3d(
        **case, grid_spacing=0.02, dt=4e-11, pml_width=1, python_backend=True
    )
    out_native = tide.maxwell3d(
        **case, grid_spacing=0.02, dt=4e-11, pml_width=1, python_backend=False
    )
    torch.testing.assert_close(out_native[-1], out_py[-1], atol=1e-4, rtol=1e-4)


def test_native_cpu_em3d_born_matches_python() -> None:
    _requires_native_backend()
    case = _case_3d(torch.device("cpu"))
    depsilon = 0.03 * torch.randn_like(case["epsilon"])
    out_py = tide.born3d(
        **case,
        depsilon=depsilon,
        grid_spacing=0.02,
        dt=4e-11,
        pml_width=1,
        python_backend=True,
    )
    out_native = tide.born3d(
        **case,
        depsilon=depsilon,
        grid_spacing=0.02,
        dt=4e-11,
        pml_width=1,
        python_backend=False,
    )
    torch.testing.assert_close(out_native[-1], out_py[-1], atol=1e-4, rtol=1e-4)


def test_native_cpu_born_model_grads_with_none_storage_route_to_python() -> None:
    _requires_native_backend()
    case = _case_2d(torch.device("cpu"))
    epsilon = case["epsilon"].clone().requires_grad_(True)
    out = tide.borntm(
        epsilon,
        case["sigma"],
        case["mu"],
        source_amplitude=case["source_amplitude"],
        source_location=case["source_location"],
        receiver_location=case["receiver_location"],
        grid_spacing=0.02,
        dt=3.5e-11,
        pml_width=3,
        storage_mode="none",
        fallback="reference",
    )
    grad = torch.autograd.grad(out[-1].square().sum(), epsilon)[0]
    assert torch.isfinite(grad).all()

    with pytest.raises(NotImplementedError, match="storage_mode='none'"):
        tide.borntm(
            epsilon.detach().requires_grad_(True),
            case["sigma"],
            case["mu"],
            source_amplitude=case["source_amplitude"],
            source_location=case["source_location"],
            receiver_location=case["receiver_location"],
            grid_spacing=0.02,
            dt=3.5e-11,
            pml_width=3,
            storage_mode="none",
            fallback="error",
        )
