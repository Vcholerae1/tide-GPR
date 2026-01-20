import math

import pytest
import torch

from tide.callbacks import CallbackState, create_callback_state
from tide.cfl import cfl_condition
from tide.padding import create_or_pad, reverse_pad, zero_interior
from tide.resampling import downsample, downsample_and_movedim, upsample
from tide.validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)
from tide.wavelets import ricker


def test_callback_state_views():
    ey = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    models = {"epsilon": torch.ones_like(ey)}
    state = CallbackState(
        dt=0.1,
        step=2,
        nt=5,
        wavefields={"Ey": ey},
        models=models,
        fd_pad=[1, 1, 1, 1],
        pml_width=[1, 1, 1, 1],
    )

    torch.testing.assert_close(state.get_wavefield("Ey", view="full"), ey)
    torch.testing.assert_close(state.get_wavefield("Ey", view="pml"), ey[1:-1, 1:-1])
    torch.testing.assert_close(state.get_wavefield("Ey", view="inner"), ey[2:-2, 2:-2])
    torch.testing.assert_close(
        state.get_model("epsilon", view="inner"),
        models["epsilon"][2:-2, 2:-2],
    )


def test_cfl_condition_warns_when_refining_dt():
    with pytest.warns(UserWarning):
        inner_dt, step_ratio = cfl_condition([0.1, 0.1], dt=0.1, max_vel=1.0)

    assert step_ratio >= 2
    assert math.isclose(inner_dt * step_ratio, 0.1)


def test_validate_freq_taper_frac_bounds():
    assert validate_freq_taper_frac(0.25) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        validate_freq_taper_frac(1.5)


def test_validate_time_pad_frac_bounds():
    assert validate_time_pad_frac(0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        validate_time_pad_frac(-0.1)


def test_validate_model_gradient_sampling_interval():
    assert validate_model_gradient_sampling_interval(0) == 0
    assert validate_model_gradient_sampling_interval(3) == 3
    with pytest.raises(TypeError):
        validate_model_gradient_sampling_interval(1.5)
    with pytest.raises(ValueError):
        validate_model_gradient_sampling_interval(-1)


def test_create_callback_state_factory():
    ey = torch.zeros((2, 3), dtype=torch.float32)
    models = {"epsilon": torch.ones_like(ey)}
    gradients = {"epsilon": torch.full_like(ey, 2.0)}
    state = create_callback_state(
        dt=0.2,
        step=3,
        nt=10,
        wavefields={"Ey": ey},
        models=models,
        gradients=gradients,
        fd_pad=[1, 1, 1, 1],
        pml_width=[2, 2, 2, 2],
        is_backward=True,
        grid_spacing=[0.1, 0.1],
    )

    assert state.dt == 0.2
    assert state.step == 3
    assert state.nt == 10
    assert state.is_backward is True
    assert state.wavefield_names == ["Ey"]
    assert state.model_names == ["epsilon"]
    assert state.gradient_names == ["epsilon"]


def test_reverse_pad_2d():
    assert reverse_pad([1, 2, 3, 4]) == [3, 4, 1, 2]


def test_create_or_pad_empty_and_constant():
    device = torch.device("cpu")
    dtype = torch.float32
    result = create_or_pad(torch.empty(0), 2, device, dtype, (2, 5, 6))
    assert result.shape == (2, 5, 6)
    assert torch.allclose(result, torch.zeros_like(result))

    base = torch.ones((2, 2), dtype=dtype, device=device)
    padded = create_or_pad(base, [1, 1, 1, 1], device, dtype, (4, 4))
    assert padded.shape == (4, 4)
    assert torch.allclose(padded[1:3, 1:3], base)
    assert padded[0, 0].item() == 0.0


def test_create_or_pad_replicate():
    device = torch.device("cpu")
    dtype = torch.float32
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
    padded = create_or_pad(base, [1, 1, 1, 1], device, dtype, (4, 4), mode="replicate")
    expected = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(padded, expected)


def test_zero_interior_y_and_x():
    tensor = torch.ones((1, 6, 6), dtype=torch.float32)
    fd_pad = [1, 1, 1, 1]
    pml_width = [1, 1, 1, 1]

    y_zeroed = zero_interior(tensor.clone(), fd_pad, pml_width, dim=0)
    assert torch.allclose(y_zeroed[:, 2:4, :], torch.zeros((1, 2, 6)))
    assert torch.all(y_zeroed[:, :2, :] == 1)
    assert torch.all(y_zeroed[:, 4:, :] == 1)

    x_zeroed = zero_interior(tensor.clone(), fd_pad, pml_width, dim=1)
    assert torch.allclose(x_zeroed[:, :, 2:4], torch.zeros((1, 6, 2)))
    assert torch.all(x_zeroed[:, :, :2] == 1)
    assert torch.all(x_zeroed[:, :, 4:] == 1)


def test_upsample_downsample_roundtrip_low_freq():
    device = torch.device("cpu")
    dtype = torch.float32
    step_ratio = 2
    n = 64
    t = torch.arange(n, device=device, dtype=dtype)
    signal = torch.sin(2.0 * math.pi * 4.0 * t / n)  # 4 cycles over length
    signal = signal[None, None, :]

    up = upsample(signal, step_ratio=step_ratio)
    down = downsample(up, step_ratio=step_ratio)
    torch.testing.assert_close(down, signal, atol=1e-4, rtol=1e-4)


def test_downsample_and_movedim_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float32
    step_ratio = 2
    receiver = torch.randn(6, 2, 3, device=device, dtype=dtype)
    expected = downsample(torch.movedim(receiver, 0, -1), step_ratio=step_ratio)
    actual = downsample_and_movedim(receiver, step_ratio=step_ratio)
    torch.testing.assert_close(actual, expected)


def test_ricker_wavelet_properties():
    freq = 2.0
    dt = 0.1
    length = 50

    wavelet = ricker(freq, length, dt, dtype=torch.float32)
    assert wavelet.shape == (length,)
    assert wavelet.dtype == torch.float32

    expected_peak_idx = int(round((1.0 / freq) / dt))
    assert abs(int(wavelet.abs().argmax()) - expected_peak_idx) <= 1

    with pytest.raises(ValueError):
        ricker(0.0, length, dt)
    with pytest.raises(ValueError):
        ricker(freq, length, 0.0)
