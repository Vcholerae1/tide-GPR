import pytest
import torch

import tide


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="3D fp16_io requires CUDA"
)


def _case(shots: int = 2, nt: int = 80):
    device = torch.device("cuda")
    shape = (24, 24, 24)
    dt = 1.6e-11
    source = torch.tensor([[[5, 12, 12]]] * shots, device=device)
    receiver = torch.tensor([[[5, 12, 16]]] * shots, device=device)
    amplitude = tide.ricker(
        160e6, nt, dt, peak_time=1.2 / 160e6, device=device
    ).view(1, 1, nt).expand(shots, -1, -1).contiguous()
    return dict(
        epsilon=torch.full(shape, 4.0, device=device),
        sigma=torch.zeros(shape, device=device),
        mu=torch.ones(shape, device=device),
        grid_spacing=0.02,
        dt=dt,
        source_amplitude=amplitude,
        source_location=source,
        receiver_location=receiver,
        pml_width=8,
        stencil=4,
        python_backend=False,
    )


def test_fp16_io_forward_matches_native():
    kwargs = _case()
    native = tide.maxwell3d(**kwargs, compute_mode="native")[-1]
    mixed = tide.maxwell3d(**kwargs, compute_mode="fp16_io")[-1]
    assert mixed.dtype == torch.float32
    assert torch.isfinite(mixed).all()
    torch.testing.assert_close(mixed, native, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
def test_fp16_io_half2_matches_scalar_fp16(monkeypatch, stencil):
    kwargs = _case(shots=2, nt=40)
    kwargs["stencil"] = stencil
    scalar = tide.maxwell3d(**kwargs, compute_mode="fp16_io")[-1]
    monkeypatch.setenv("TIDE_EM3D_FP16_HALF2", "1")
    packed = tide.maxwell3d(**kwargs, compute_mode="fp16_io")[-1]
    assert torch.isfinite(packed).all()
    torch.testing.assert_close(packed, scalar, rtol=0, atol=0)


def test_fp16_io_shot_batch_matches_individual_shots():
    kwargs = _case(shots=2)
    kwargs["source_amplitude"] = kwargs["source_amplitude"].clone()
    kwargs["source_amplitude"][1].mul_(0.03125)
    batched = tide.maxwell3d(**kwargs, compute_mode="fp16_io")[-1]
    individual = []
    for shot in range(2):
        shot_kwargs = dict(kwargs)
        for name in ("source_amplitude", "source_location", "receiver_location"):
            shot_kwargs[name] = kwargs[name][shot : shot + 1]
        individual.append(tide.maxwell3d(**shot_kwargs, compute_mode="fp16_io")[-1])
    torch.testing.assert_close(batched, torch.cat(individual, dim=1), rtol=0, atol=0)


def test_fp16_io_rejects_gradients():
    kwargs = _case(shots=1, nt=4)
    kwargs["epsilon"].requires_grad_()
    with pytest.raises(NotImplementedError, match="forward inference only"):
        tide.maxwell3d(**kwargs, compute_mode="fp16_io")
