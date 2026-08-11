from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import tide
from numerical_utils import cosine_similarity, relative_l2, require_cuda_backend


def _run_tm(
    device: torch.device, stencil: int, python_backend: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    ny, nx, nt = 18, 20, 52
    epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype, requires_grad=True)
    sigma = torch.full_like(epsilon, 2.0e-4, requires_grad=True)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(
        400e6,
        nt,
        2.0e-11,
        peak_time=7.0e-10,
        device=device,
        dtype=dtype,
    ).view(1, 1, nt)
    source_location = torch.tensor([[[9, 6]]], device=device, dtype=torch.long)
    receiver_location = torch.tensor(
        [[[9, 12], [11, 12]]], device=device, dtype=torch.long
    )
    receiver = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        [0.018, 0.022],
        2.0e-11,
        source,
        source_location,
        receiver_location,
        stencil=stencil,
        pml_width=4,
        python_backend=python_backend,
        storage_compression=False,
    )[-1]
    residual = torch.linspace(-0.7, 1.1, nt, device=device, dtype=dtype).view(nt, 1, 1)
    (receiver * residual).sum().backward()
    assert epsilon.grad is not None and sigma.grad is not None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        receiver.detach().cpu(),
        epsilon.grad.detach().cpu(),
        sigma.grad.detach().cpu(),
    )


def _run_3d(
    device: torch.device, stencil: int, python_backend: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    nz, ny, nx, nt = 9, 10, 11, 40
    epsilon = torch.full(
        (nz, ny, nx), 4.0, device=device, dtype=dtype, requires_grad=True
    )
    sigma = torch.full_like(epsilon, 2.0e-4, requires_grad=True)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(
        500e6,
        nt,
        2.0e-11,
        peak_time=6.0e-10,
        device=device,
        dtype=dtype,
    ).view(1, 1, nt)
    source_location = torch.tensor([[[4, 5, 4]]], device=device, dtype=torch.long)
    receiver_location = torch.tensor(
        [[[4, 5, 7], [5, 7, 7]]], device=device, dtype=torch.long
    )
    receiver = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        [0.016, 0.018, 0.022],
        2.0e-11,
        source,
        source_location,
        receiver_location,
        source_component="ey",
        receiver_component="ey",
        stencil=stencil,
        pml_width=4,
        python_backend=python_backend,
        storage_compression=False,
    )[-1]
    residual = torch.linspace(-0.6, 1.0, nt, device=device, dtype=dtype).view(nt, 1, 1)
    (receiver * residual).sum().backward()
    assert epsilon.grad is not None and sigma.grad is not None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        receiver.detach().cpu(),
        epsilon.grad.detach().cpu(),
        sigma.grad.detach().cpu(),
    )


@pytest.mark.cuda
@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("runner", [_run_tm, _run_3d], ids=["tm2d", "maxwell3d"])
def test_cuda_matches_cpu_reference(
    runner: Callable[
        [torch.device, int, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    stencil: int,
) -> None:
    cuda_device = require_cuda_backend()
    reference = runner(torch.device("cpu"), stencil, True)
    actual = runner(cuda_device, stencil, False)
    assert relative_l2(actual[0], reference[0]) < 2.0e-4
    for gradient, reference_gradient in zip(actual[1:], reference[1:], strict=True):
        assert relative_l2(gradient, reference_gradient) < 5.0e-3
        assert cosine_similarity(gradient, reference_gradient) > 0.999
