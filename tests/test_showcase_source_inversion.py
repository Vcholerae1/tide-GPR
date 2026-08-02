from __future__ import annotations

import torch

from experiments.papers.ifwi_showcase.inversion2d import (
    _direct_arrival_window,
    estimate_shared_source_wavelet,
)


def test_estimate_shared_source_wavelet_recovers_delayed_scaled_traces() -> None:
    nt = 64
    source = torch.zeros(nt)
    source[5:10] = torch.tensor([0.2, -0.5, 1.0, -0.5, 0.2])
    green = torch.zeros(nt, 3, 1)
    green[0, 0, 0] = 1.0
    green[2, 1, 0] = -0.75
    green[4, 2, 0] = 0.5
    observed = torch.zeros_like(green)
    observed[:, 0, 0] = source
    observed[2:, 1, 0] = -0.75 * source[:-2]
    observed[4:, 2, 0] = 0.5 * source[:-4]
    window = torch.ones(nt)

    estimated, predicted = estimate_shared_source_wavelet(
        green,
        observed,
        window_weights=window,
        waterlevel=0.0,
        dt=1.0,
    )

    torch.testing.assert_close(estimated, source, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(predicted, observed, atol=1.0e-6, rtol=1.0e-6)


def test_direct_arrival_window_has_tapered_exclusive_support() -> None:
    window = _direct_arrival_window(
        12,
        start_sample=2,
        end_sample=10,
        taper_samples=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.count_nonzero(window[:2]) == 0
    assert torch.count_nonzero(window[10:]) == 0
    assert 0.0 < float(window[2]) < float(window[3]) < 1.0
    assert float(window[4]) == 1.0
    assert 0.0 < float(window[9]) < float(window[8]) < 1.0


def test_direct_arrival_window_does_not_taper_physical_time_boundaries() -> None:
    window = _direct_arrival_window(
        12,
        start_sample=0,
        end_sample=10,
        taper_samples=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert float(window[0]) == 1.0
    assert float(window[1]) == 1.0
    assert 0.0 < float(window[9]) < float(window[8]) < 1.0
