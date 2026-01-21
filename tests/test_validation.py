import pytest

from tide.validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)


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
        validate_model_gradient_sampling_interval(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_model_gradient_sampling_interval(-1)
