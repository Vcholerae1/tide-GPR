from __future__ import annotations

import pytest
import torch

from numerical_utils import (
    MaxwellExample,
    make_maxwell3d_example,
    make_tm2d_example,
)


def _tm2d_example() -> MaxwellExample:
    return make_tm2d_example(
        shape=(28, 36),
        nt=180,
        grid_spacing=[0.018, 0.022],
        dt=3.0e-11,
        frequency=250e6,
        peak_time=2.5e-9,
        dtype=torch.float64,
        sigma=2.0e-4,
        source_location=(14, 9),
        receiver_locations=((14, 14), (14, 20)),
        pml_width=4,
        stencil=4,
        python_backend=True,
    )


def _em3d_example() -> MaxwellExample:
    return make_maxwell3d_example(
        shape=(10, 11, 12),
        nt=80,
        grid_spacing=[0.016, 0.018, 0.022],
        dt=2.0e-11,
        frequency=350e6,
        peak_time=2.0e-9,
        sigma=2.0e-4,
        source_location=(5, 5, 4),
        receiver_locations=((5, 5, 7), (5, 7, 7)),
        pml_width=2,
        stencil=4,
        python_backend=True,
    )


@pytest.fixture(params=(_tm2d_example, _em3d_example), ids=("tm2d", "maxwell3d"))
def maxwell_example(request: pytest.FixtureRequest) -> MaxwellExample:
    return request.param()


@pytest.fixture
def tm2d_example() -> MaxwellExample:
    return _tm2d_example()
