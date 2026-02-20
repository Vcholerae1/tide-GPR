import pytest
import torch

import tide

try:
    from tide import backend_utils
except Exception:  # pragma: no cover - import guard for environments without backend
    backend_utils = None  # type: ignore[assignment]


_MPS_AVAILABLE = torch.backends.mps.is_available()
_METAL_AVAILABLE = backend_utils is not None and backend_utils.metal_available()


pytestmark = pytest.mark.skipif(
    not (_MPS_AVAILABLE and _METAL_AVAILABLE),
    reason="MPS+Metal backend not available",
)


def _make_case(device: torch.device, *, nt: int = 24):
    dtype = torch.float32
    ny, nx = 20, 24
    epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    dt = 4e-11
    dx = 0.02
    source_locations = torch.tensor([[[ny // 2, nx // 3]]], dtype=torch.long, device=device)
    receiver_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
    wavelet = tide.ricker(200e6, nt, dt, peak_time=1.0 / 200e6, dtype=dtype, device=device)
    source_amplitude = wavelet.view(1, 1, nt)
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "dt": dt,
        "dx": dx,
        "source_amplitude": source_amplitude,
        "source_locations": source_locations,
        "receiver_locations": receiver_locations,
    }


def test_mps_forward_matches_cpu_reference_stencil2():
    mps = torch.device("mps")
    cpu = torch.device("cpu")

    c_mps = _make_case(mps)
    c_cpu = {
        k: (v.to(cpu) if isinstance(v, torch.Tensor) else v)
        for k, v in c_mps.items()
    }

    out_mps = tide.maxwelltm(
        c_mps["epsilon"],
        c_mps["sigma"],
        c_mps["mu"],
        grid_spacing=c_mps["dx"],
        dt=c_mps["dt"],
        source_amplitude=c_mps["source_amplitude"],
        source_location=c_mps["source_locations"],
        receiver_location=c_mps["receiver_locations"],
        pml_width=4,
        stencil=2,
    )[-1].cpu()

    out_cpu = tide.maxwelltm(
        c_cpu["epsilon"],
        c_cpu["sigma"],
        c_cpu["mu"],
        grid_spacing=c_cpu["dx"],
        dt=c_cpu["dt"],
        source_amplitude=c_cpu["source_amplitude"],
        source_location=c_cpu["source_locations"],
        receiver_location=c_cpu["receiver_locations"],
        pml_width=4,
        stencil=2,
    )[-1]

    torch.testing.assert_close(out_mps, out_cpu, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
def test_mps_forward_all_stencils_finite(stencil: int):
    mps = torch.device("mps")
    c = _make_case(mps)

    rec = tide.maxwelltm(
        c["epsilon"],
        c["sigma"],
        c["mu"],
        grid_spacing=c["dx"],
        dt=c["dt"],
        source_amplitude=c["source_amplitude"],
        source_location=c["source_locations"],
        receiver_location=c["receiver_locations"],
        pml_width=4,
        stencil=stencil,
    )[-1]

    assert torch.isfinite(rec).all()


def test_mps_autograd_epsilon_sigma_grad_finite_nonzero():
    mps = torch.device("mps")
    c = _make_case(mps)

    epsilon = c["epsilon"].clone().detach().requires_grad_(True)
    sigma = (torch.ones_like(c["sigma"]) * 1e-3).requires_grad_(True)

    rec = tide.maxwelltm(
        epsilon,
        sigma,
        c["mu"],
        grid_spacing=c["dx"],
        dt=c["dt"],
        source_amplitude=c["source_amplitude"],
        source_location=c["source_locations"],
        receiver_location=c["receiver_locations"],
        pml_width=4,
        stencil=2,
    )[-1]

    loss = rec.square().sum()
    loss.backward()

    assert epsilon.grad is not None
    assert sigma.grad is not None
    assert torch.isfinite(epsilon.grad).all()
    assert torch.isfinite(sigma.grad).all()
    assert epsilon.grad.abs().sum() > 0
    assert sigma.grad.abs().sum() > 0


def test_mps_gradient_sampling_interval_smoke():
    mps = torch.device("mps")
    c = _make_case(mps)

    eps1 = c["epsilon"].clone().detach().requires_grad_(True)
    out1 = tide.maxwelltm(
        eps1,
        c["sigma"],
        c["mu"],
        grid_spacing=c["dx"],
        dt=c["dt"],
        source_amplitude=c["source_amplitude"],
        source_location=c["source_locations"],
        receiver_location=c["receiver_locations"],
        pml_width=4,
        stencil=2,
        model_gradient_sampling_interval=1,
    )[-1]
    out1.pow(2).sum().backward()
    grad1 = eps1.grad

    eps2 = c["epsilon"].clone().detach().requires_grad_(True)
    out2 = tide.maxwelltm(
        eps2,
        c["sigma"],
        c["mu"],
        grid_spacing=c["dx"],
        dt=c["dt"],
        source_amplitude=c["source_amplitude"],
        source_location=c["source_locations"],
        receiver_location=c["receiver_locations"],
        pml_width=4,
        stencil=2,
        model_gradient_sampling_interval=3,
    )[-1]
    out2.pow(2).sum().backward()
    grad2 = eps2.grad

    assert grad1 is not None and grad2 is not None
    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()


def test_mps_callback_chunking_smoke():
    mps = torch.device("mps")
    c = _make_case(mps, nt=32)

    callback_steps: list[int] = []

    def callback(state: tide.CallbackState) -> None:
        callback_steps.append(state.step)

    rec = tide.maxwelltm(
        c["epsilon"],
        c["sigma"],
        c["mu"],
        grid_spacing=c["dx"],
        dt=c["dt"],
        source_amplitude=c["source_amplitude"],
        source_location=c["source_locations"],
        receiver_location=c["receiver_locations"],
        pml_width=4,
        stencil=2,
        callback_frequency=4,
        forward_callback=callback,
    )[-1]

    assert torch.isfinite(rec).all()
    assert len(callback_steps) > 0
