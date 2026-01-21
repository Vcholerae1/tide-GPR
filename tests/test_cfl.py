import math

import pytest

from tide.cfl import cfl_condition


def test_cfl_condition_warns_when_refining_dt():
    with pytest.warns(UserWarning):
        inner_dt, step_ratio = cfl_condition([0.1, 0.1], dt=0.1, max_vel=1.0)

    assert step_ratio >= 2
    assert math.isclose(inner_dt * step_ratio, 0.1)
