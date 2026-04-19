import pytest
import torch

import tide
from tide.callbacks import CallbackState
from tide import backend_utils


def test_callback_state_views_for_3d():
    device = torch.device("cpu")
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 8
    pml_width = [2, 1, 1, 2, 1, 2]

    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor([[[2, 3, 2]]], dtype=torch.long, device=device)
    receiver_location = torch.tensor([[[2, 3, 5]]], dtype=torch.long, device=device)
    source_amplitude = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    ).view(1, 1, nt)

    seen = {"calls": 0}

    def cb(state: CallbackState):
        seen["calls"] += 1
        ey_inner = state.get_wavefield("Ey", view="inner")
        ey_pml = state.get_wavefield("Ey", view="pml")
        ey_full = state.get_wavefield("Ey", view="full")

        assert ey_inner.shape == (1, nz, ny, nx)
        assert ey_pml.shape == (
            1,
            nz + pml_width[0] + pml_width[1],
            ny + pml_width[2] + pml_width[3],
            nx + pml_width[4] + pml_width[5],
        )
        assert ey_full.ndim == 4
        assert ey_full.shape[-3] >= ey_pml.shape[-3]
        assert ey_full.shape[-2] >= ey_pml.shape[-2]
        assert ey_full.shape[-1] >= ey_pml.shape[-1]

    tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=[0.03, 0.02, 0.02],
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=pml_width,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
        forward_callback=cb,
        callback_frequency=2,
    )
    assert seen["calls"] > 0


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required for this test."
            ),
        ),
    ],
)
def test_native_3d_backward_callback_preserves_accumulated_gradients(
    device_type: str,
):
    if not backend_utils.is_backend_available():
        pytest.skip("native backend unavailable")

    device = torch.device(device_type)
    dtype = torch.float32
    nz, ny, nx = 6, 7, 8
    nt = 9
    pml_width = [2, 1, 1, 2, 1, 2]

    epsilon = torch.ones(nz, ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.ones_like(epsilon) * 2e-4
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[2, 3, 2]], [[2, 4, 2]]], dtype=torch.long, device=device
    )
    receiver_location = torch.tensor(
        [[[2, 3, 5]], [[2, 4, 5]]], dtype=torch.long, device=device
    )
    wavelet = tide.ricker(
        80e6, nt, 4e-11, peak_time=1.0 / 80e6, dtype=dtype, device=device
    )
    source_amplitude = wavelet.view(1, 1, nt).repeat(2, 1, 1)
    source_amplitude[1] *= 0.75

    def _run(*, with_callback: bool) -> tuple[torch.Tensor, torch.Tensor]:
        eps = epsilon.clone().detach().requires_grad_(True)
        sig = sigma.clone().detach().requires_grad_(True)
        seen_steps: list[int] = []

        kwargs = {}
        if with_callback:

            def backward_cb(state: CallbackState) -> None:
                seen_steps.append(state.step)

            kwargs["backward_callback"] = backward_cb
            kwargs["callback_frequency"] = 2

        receivers = tide.maxwell3d(
            eps,
            sig,
            mu,
            grid_spacing=[0.03, 0.02, 0.02],
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=pml_width,
            source_component="ey",
            receiver_component="ey",
            python_backend=False,
            **kwargs,
        )[-1]
        receivers.pow(2).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

        assert eps.grad is not None
        assert sig.grad is not None
        if with_callback:
            assert len(seen_steps) > 1
        return eps.grad.detach().clone(), sig.grad.detach().clone()

    grad_eps_base, grad_sig_base = _run(with_callback=False)
    grad_eps_cb, grad_sig_cb = _run(with_callback=True)

    torch.testing.assert_close(grad_eps_cb, grad_eps_base, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_sig_cb, grad_sig_base, rtol=1e-5, atol=1e-6)
