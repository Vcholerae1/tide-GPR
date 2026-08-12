from __future__ import annotations


import pytest
import torch

from numerical_utils import MaxwellExample, relative_l2


@pytest.mark.numerical
def test_zero_source_produces_zero_response(
    maxwell_example: MaxwellExample,
) -> None:
    result = maxwell_example.run(
        source_amplitude=torch.zeros_like(maxwell_example.source_amplitude)
    )
    assert torch.count_nonzero(result[-1]) == 0
    for state in result[:-1]:
        assert torch.count_nonzero(state) == 0


@pytest.mark.numerical
def test_source_scaling_is_linear(maxwell_example: MaxwellExample) -> None:
    reference = maxwell_example.run()[-1]
    scaled = maxwell_example.run(
        source_amplitude=2.5 * maxwell_example.source_amplitude
    )[-1]
    assert relative_l2(scaled, 2.5 * reference) < 2.0e-6


@pytest.mark.numerical
def test_multiple_sources_superpose(tm2d_example: MaxwellExample) -> None:
    source = tm2d_example.source_amplitude
    shifted = torch.roll(source, shifts=9, dims=-1)
    source_locations = tm2d_example.source_location.repeat(1, 2, 1)
    source_locations[:, 1, 0] += 2
    combined = torch.cat((source, 0.4 * shifted), dim=1)

    both = tm2d_example.run(
        source_amplitude=combined,
        source_location=source_locations,
    )[-1]
    first = tm2d_example.run()[-1]
    second = tm2d_example.run(
        source_amplitude=0.4 * shifted,
        source_location=source_locations[:, 1:2],
    )[-1]
    assert relative_l2(both, first + second) < 1.0e-7


@pytest.mark.numerical
def test_tm2d_source_receiver_reciprocity(tm2d_example: MaxwellExample) -> None:
    sigma = torch.zeros_like(tm2d_example.sigma)
    receiver_location = tm2d_example.receiver_location[:, :1]
    forward = tm2d_example.run(
        sigma=sigma,
        receiver_location=receiver_location,
    )[-1]
    reverse = tm2d_example.run(
        sigma=sigma,
        source_location=receiver_location,
        receiver_location=tm2d_example.source_location,
    )[-1]
    assert relative_l2(forward, reverse) < 5.0e-4


@pytest.mark.numerical
def test_tm2d_state_continuation_matches_single_run(
    tm2d_example: MaxwellExample,
) -> None:
    source = tm2d_example.source_amplitude
    split = source.shape[-1] // 2
    whole = tm2d_example.run()
    first = tm2d_example.run(source_amplitude=source[..., :split])
    second = tm2d_example.run(
        source_amplitude=source[..., split:],
        Ey_0=first[0],
        Hx_0=first[1],
        Hz_0=first[2],
        m_Ey_x=first[3],
        m_Ey_z=first[4],
        m_Hx_z=first[5],
        m_Hz_x=first[6],
    )
    continued_receiver = torch.cat((first[-1], second[-1]), dim=0)
    assert relative_l2(continued_receiver, whole[-1]) < 5.0e-4
    for continued, uninterrupted in zip(second[:-1], whole[:-1], strict=True):
        assert relative_l2(continued, uninterrupted) < 5.0e-4
