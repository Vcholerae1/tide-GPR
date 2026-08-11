from __future__ import annotations

import pytest
import torch

import tide
from numerical_utils import (
    cosine_similarity,
    relative_l2,
    require_native_backend,
    signal_rms,
)


def _reflection_case(pml_width: int) -> torch.Tensor:
    dtype = torch.float64
    ny, nx, nt = 24, 50, 450
    dt = 2.0e-11
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(
        300e6,
        nt,
        dt,
        peak_time=1.5e-9,
        dtype=dtype,
    ).view(1, 1, nt)
    source_location = torch.tensor([[[ny // 2, 15]]], dtype=torch.long)
    receiver_location = torch.tensor([[[ny // 2, 20]]], dtype=torch.long)
    return tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        0.02,
        dt,
        source,
        source_location,
        receiver_location,
        stencil=2,
        pml_width=pml_width,
        python_backend=True,
    )[-1]


@pytest.mark.numerical
def test_cpml_preserves_early_trace_and_reduces_late_reflection() -> None:
    reflective = _reflection_case(0)
    early_stop = 100
    late_start = 280
    reflective_late_rms = signal_rms(reflective[late_start:])
    assert reflective_late_rms > 0.0

    for width in (4, 8, 12):
        absorbed = _reflection_case(width)
        assert relative_l2(absorbed[:early_stop], reflective[:early_stop]) < 1.0e-5
        assert signal_rms(absorbed[late_start:]) / reflective_late_rms < 0.30


def _tm_gradient(
    *, python_backend: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    ny, nx, nt = 14, 16, 48
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype, requires_grad=True)
    sigma = torch.full((ny, nx), 1.0e-4, dtype=dtype, requires_grad=True)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(250e6, nt, 2.5e-11, peak_time=1.0e-9, dtype=dtype).view(
        1, 1, nt
    )
    source_location = torch.tensor([[[7, 5]]], dtype=torch.long)
    receiver_location = torch.tensor([[[7, 10]]], dtype=torch.long)
    receiver = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        0.02,
        2.5e-11,
        source,
        source_location,
        receiver_location,
        stencil=4,
        pml_width=3,
        python_backend=python_backend,
        storage_compression=False,
    )[-1]
    receiver.square().sum().backward()
    assert epsilon.grad is not None
    assert sigma.grad is not None
    return receiver.detach(), epsilon.grad.detach(), sigma.grad.detach()


@pytest.mark.numerical
def test_cpml_native_foldback_matches_reference() -> None:
    require_native_backend()
    receiver_reference, epsilon_reference, sigma_reference = _tm_gradient(
        python_backend=True
    )
    receiver_native, epsilon_native, sigma_native = _tm_gradient(python_backend=False)
    assert relative_l2(receiver_native, receiver_reference) < 2.0e-4
    for actual, reference in (
        (epsilon_native, epsilon_reference),
        (sigma_native, sigma_reference),
    ):
        assert relative_l2(actual, reference) < 5.0e-3
        assert cosine_similarity(actual, reference) > 0.999
