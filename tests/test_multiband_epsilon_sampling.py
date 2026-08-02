from __future__ import annotations

import numpy as np
import pytest

from experiments.inversion_3d.multiband_epsilon import (
    localize_shot_subset,
    random_shot_subset,
)


def test_random_shot_subset_is_exact_deterministic_quarter() -> None:
    first = random_shot_subset(144, 0.25, seed=17)
    second = random_shot_subset(144, 0.25, seed=17)
    different = random_shot_subset(144, 0.25, seed=18)

    assert first.shape == (36,)
    assert np.unique(first).size == first.size
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_localize_shot_subset_preserves_random_order_and_observation_positions() -> (
    None
):
    selected = np.array([7, 2, 9, 4, 0], dtype=np.int64)
    local = np.array([0, 2, 4, 6, 8], dtype=np.int64)

    local_selected, observation_positions = localize_shot_subset(selected, local)

    np.testing.assert_array_equal(local_selected, [2, 4, 0])
    np.testing.assert_array_equal(observation_positions, [1, 2, 0])


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_random_shot_subset_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        random_shot_subset(10, fraction, seed=0)
