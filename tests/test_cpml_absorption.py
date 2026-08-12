from __future__ import annotations

import pytest
import torch

from numerical_utils import (
    cosine_similarity,
    make_tm2d_example,
    relative_l2,
    require_native_backend,
    signal_rms,
)


def _reflection_response(pml_width: int) -> torch.Tensor:
    example = make_tm2d_example(
        shape=(24, 50),
        nt=450,
        grid_spacing=0.02,
        dt=2.0e-11,
        frequency=300e6,
        peak_time=1.5e-9,
        dtype=torch.float64,
        source_location=(12, 15),
        receiver_locations=((12, 20),),
        pml_width=pml_width,
        python_backend=True,
    )
    return example.run()[-1]


@pytest.mark.numerical
def test_cpml_preserves_early_trace_and_reduces_late_reflection() -> None:
    reflective = _reflection_response(0)
    early_stop = 100
    late_start = 280
    reflective_late_rms = signal_rms(reflective[late_start:])
    assert reflective_late_rms > 0.0

    for width in (4, 8, 12):
        absorbed = _reflection_response(width)
        assert relative_l2(absorbed[:early_stop], reflective[:early_stop]) < 1.0e-5
        assert signal_rms(absorbed[late_start:]) / reflective_late_rms < 0.30


def _tm_gradient(
    *, python_backend: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    example = make_tm2d_example(
        shape=(14, 16),
        nt=48,
        grid_spacing=0.02,
        dt=2.5e-11,
        frequency=250e6,
        peak_time=1.0e-9,
        dtype=torch.float64,
        sigma=1.0e-4,
        source_location=(7, 5),
        receiver_locations=((7, 10),),
        pml_width=3,
        stencil=4,
    )
    epsilon = example.epsilon.requires_grad_()
    sigma = example.sigma.requires_grad_()
    receiver = example.run(
        epsilon=epsilon,
        sigma=sigma,
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
