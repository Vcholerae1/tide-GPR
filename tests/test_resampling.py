import math

import torch

from tide.resampling import downsample, downsample_and_movedim, upsample


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
