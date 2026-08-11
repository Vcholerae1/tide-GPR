from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import tide
from numerical_utils import relative_l2


def _run_tm(case: dict[str, object], source: torch.Tensor, **overrides: object):
    kwargs = {
        "epsilon": case["epsilon"],
        "sigma": case["sigma"],
        "mu": case["mu"],
        "grid_spacing": case["grid_spacing"],
        "dt": case["dt"],
        "source_amplitude": source,
        "source_location": case["source_location"],
        "receiver_location": case["receiver_location"],
        "stencil": 4,
        "pml_width": case["pml_width"],
        "python_backend": True,
    }
    kwargs.update(overrides)
    return tide.maxwelltm(**kwargs)


def _run_3d(case: dict[str, object], source: torch.Tensor, **overrides: object):
    kwargs = {
        "epsilon": case["epsilon"],
        "sigma": case["sigma"],
        "mu": case["mu"],
        "grid_spacing": case["grid_spacing"],
        "dt": case["dt"],
        "source_amplitude": source,
        "source_location": case["source_location"],
        "receiver_location": case["receiver_location"],
        "source_component": "ey",
        "receiver_component": "ey",
        "stencil": 4,
        "pml_width": case["pml_width"],
        "python_backend": True,
    }
    kwargs.update(overrides)
    return tide.maxwell3d(**kwargs)


@pytest.mark.numerical
@pytest.mark.parametrize(
    "runner,fixture_name",
    [(_run_tm, "tm2d_numerical_case"), (_run_3d, "em3d_numerical_case")],
)
def test_zero_source_produces_zero_response(
    runner: Callable[..., tuple[torch.Tensor, ...]],
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    case = request.getfixturevalue(fixture_name)
    source = torch.zeros_like(case["source_amplitude"])
    result = runner(case, source)
    assert torch.count_nonzero(result[-1]) == 0
    for state in result[:-1]:
        assert torch.count_nonzero(state) == 0


@pytest.mark.numerical
@pytest.mark.parametrize(
    "runner,fixture_name",
    [(_run_tm, "tm2d_numerical_case"), (_run_3d, "em3d_numerical_case")],
)
def test_source_scaling_is_linear(
    runner: Callable[..., tuple[torch.Tensor, ...]],
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    case = request.getfixturevalue(fixture_name)
    source = case["source_amplitude"]
    reference = runner(case, source)[-1]
    scaled = runner(case, 2.5 * source)[-1]
    assert relative_l2(scaled, 2.5 * reference) < 2.0e-6


@pytest.mark.numerical
def test_multiple_sources_superpose(tm2d_numerical_case: dict[str, object]) -> None:
    case = tm2d_numerical_case
    source = case["source_amplitude"]
    shifted = torch.roll(source, shifts=9, dims=-1)
    source_locations = case["source_location"].repeat(1, 2, 1)
    source_locations[:, 1, 0] += 2
    combined = torch.cat((source, 0.4 * shifted), dim=1)

    both = _run_tm(case, combined, source_location=source_locations)[-1]
    first = _run_tm(case, source)[-1]
    second = _run_tm(
        case,
        0.4 * shifted,
        source_location=source_locations[:, 1:2],
    )[-1]
    assert relative_l2(both, first + second) < 1.0e-7


@pytest.mark.numerical
def test_tm2d_source_receiver_reciprocity(
    tm2d_numerical_case: dict[str, object],
) -> None:
    case = dict(tm2d_numerical_case)
    case["sigma"] = torch.zeros_like(case["sigma"])
    case["receiver_location"] = case["receiver_location"][:, :1]
    forward = _run_tm(case, case["source_amplitude"])[-1]

    reverse_case = dict(case)
    reverse_case["source_location"] = case["receiver_location"]
    reverse_case["receiver_location"] = case["source_location"]
    reverse = _run_tm(reverse_case, case["source_amplitude"])[-1]
    assert relative_l2(forward, reverse) < 5.0e-4


@pytest.mark.numerical
def test_tm2d_state_continuation_matches_single_run(
    tm2d_numerical_case: dict[str, object],
) -> None:
    case = tm2d_numerical_case
    source = case["source_amplitude"]
    split = source.shape[-1] // 2
    whole = _run_tm(case, source)
    first = _run_tm(case, source[..., :split])
    second = _run_tm(
        case,
        source[..., split:],
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
