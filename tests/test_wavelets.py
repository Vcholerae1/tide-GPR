import pytest
import torch

from tide.wavelets import ricker


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
