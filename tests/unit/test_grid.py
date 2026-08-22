from __future__ import annotations

import math
import pytest
import torch
from tide import staggered, utils
from tide.grid_utils import _CompactCPMLLayout
from tide.cfl import cfl_condition
from tide.padding import create_or_pad, reverse_pad, zero_interior
from tide.resampling import downsample, downsample_and_movedim, upsample
from tide.wavelets import gaussian, gaussian_derivative, morlet, ricker, sine_burst

# --- test_staggered.py ---

"""Tests for staggered grid operations in tide.staggered module."""


@pytest.mark.parametrize(
    ("derivative", "shape", "axis", "slope", "half_grid"),
    [
        pytest.param(staggered.diffy1, (32, 32), 0, 2.0, False, id="diffy1"),
        pytest.param(staggered.diffx1, (32, 32), 1, 3.0, False, id="diffx1"),
        pytest.param(staggered.diffyh1, (32, 32), 0, 2.0, True, id="diffyh1"),
        pytest.param(staggered.diffxh1, (32, 32), 1, 3.0, True, id="diffxh1"),
        pytest.param(staggered.diffzh1, (16, 16, 16), 0, 2.0, True, id="diffzh1"),
    ],
)
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
def test_first_derivative_of_linear_function(
    derivative, shape, axis, slope, half_grid, stencil
):
    spacing = 0.1
    coordinate = torch.arange(shape[axis], dtype=torch.float32) * spacing
    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]
    field = (slope * coordinate).reshape(view_shape).expand(shape)

    result = derivative(field, stencil, torch.tensor(1.0 / spacing))

    pad = stencil // 2
    valid = [slice(None)] * len(shape)
    valid[axis] = (
        slice(pad, -pad) if half_grid else slice(pad, None if pad == 1 else 1 - pad)
    )
    assert result.shape == field.shape
    torch.testing.assert_close(
        result[tuple(valid)],
        torch.full_like(result[tuple(valid)], slope),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("derivative", "shape", "axis"),
    [
        pytest.param(staggered.diffy1, (64, 32), 0, id="diffy1"),
        pytest.param(staggered.diffx1, (32, 64), 1, id="diffx1"),
    ],
)
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
def test_first_derivative_of_sine_function(derivative, shape, axis, stencil):
    spacing = 0.05
    coordinate = torch.arange(shape[axis], dtype=torch.float32) * spacing
    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]
    field = torch.sin(2.0 * math.pi * coordinate / (shape[axis] * spacing))
    field = field.reshape(view_shape).expand(shape)

    result = derivative(field, stencil, torch.tensor(1.0 / spacing))

    length = shape[axis] * spacing
    expected = (2.0 * math.pi / length) * torch.cos(2.0 * math.pi * coordinate / length)
    pad = stencil // 2 + 2
    valid = [0] * len(shape)
    valid[axis] = slice(pad, -(pad - 1 if stencil > 2 else 1))
    torch.testing.assert_close(
        result[tuple(valid)],
        expected[valid[axis]],
        atol=0.15 if stencil == 2 else 0.12,
        rtol=1.0,
    )


def _pml_profiles(pml_width, *, accuracy=4, size=32):
    return staggered.set_pml_profiles(
        pml_width,
        accuracy,
        [accuracy // 2] * 4,
        1e-11,
        [0.01, 0.01],
        3e8,
        torch.float32,
        torch.device("cpu"),
        25.0,
        size,
        size,
    )


def test_set_pml_profiles_2d_shapes():
    profiles = _pml_profiles([4] * 4)
    ab_profiles, k_profiles = profiles[:8], profiles[8:]

    assert len(ab_profiles) == 8
    assert len(k_profiles) == 4
    assert ab_profiles[0].shape == (1, 32, 1)
    assert ab_profiles[2].shape == (1, 1, 32)
    assert k_profiles[0].shape == (1, 32, 1)
    assert k_profiles[2].shape == (1, 1, 32)


def test_set_pml_profiles_zero_width():
    profiles = _pml_profiles([0] * 4, accuracy=2, size=16)
    ab_profiles, k_profiles = profiles[:8], profiles[8:]

    assert all(torch.count_nonzero(profile) == 0 for profile in ab_profiles)
    assert all(torch.all(profile == 1.0) for profile in k_profiles)


def test_set_pml_profiles_coefficient_ranges():
    *_, by, byh, bx, bxh, ky, kyh, kx, kxh = _pml_profiles([8] * 4)

    for profile in (by, byh, bx, bxh):
        assert torch.all((0.0 <= profile) & (profile <= 1.0))
    for profile in (ky, kyh, kx, kxh):
        assert torch.all(profile >= 1.0)
        assert profile.max() > 1.0


def test_compact_cpml_layout_shapes() -> None:
    layout = _CompactCPMLLayout(
        2,
        (67, 68, 69),
        ((10, 58), (11, 59), (12, 60)),
    )
    assert tuple(layout.shape(axis) for axis in range(3)) == (
        (2, 20, 68, 69),
        (2, 67, 21, 69),
        (2, 67, 68, 22),
    )


def test_compact_cpml_layout_pack_unpack_roundtrip() -> None:
    layout = _CompactCPMLLayout(1, (6, 2, 2), ((2, 5), (0, 2), (0, 2)))
    full = torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2)
    packed = layout.pack(full, 0)
    restored = layout.unpack(packed, 0)
    assert torch.equal(restored[:, [0, 1, 4, 5]], full[:, [0, 1, 4, 5]])
    assert torch.count_nonzero(restored[:, 2:4]) == 0


# --- test_cfl.py ---


def test_cfl_condition_warns_when_refining_dt():
    with pytest.warns(UserWarning):
        inner_dt, step_ratio = cfl_condition([0.1, 0.1], dt=0.1, max_vel=1.0)

    assert step_ratio >= 2
    assert math.isclose(inner_dt * step_ratio, 0.1)


# --- test_padding.py ---


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


# --- test_resampling.py ---


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


# --- test_wavelets.py ---


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


@pytest.mark.parametrize("wavelet_fn", [ricker, gaussian, morlet, sine_burst])
def test_centered_wavelets_have_unit_peak(wavelet_fn):
    wavelet = wavelet_fn(2.0, 21, 0.05, peak_time=0.5, dtype=torch.float64)

    assert wavelet.shape == (21,)
    assert wavelet.dtype == torch.float64
    assert wavelet[10].item() == pytest.approx(1.0)


def test_gaussian_derivative_is_antisymmetric_and_normalized():
    wavelet = gaussian_derivative(2.0, 2001, 0.0005, peak_time=0.5)

    torch.testing.assert_close(wavelet, -wavelet.flip(0), atol=2e-5, rtol=0.0)
    assert wavelet.abs().max().item() == pytest.approx(1.0, rel=1e-5)


def test_sine_burst_has_finite_support():
    wavelet = sine_burst(2.0, 41, 0.05, peak_time=1.0, cycles=2.0)

    assert torch.count_nonzero(wavelet[:10]) == 0
    assert torch.count_nonzero(wavelet[31:]) == 0
    with pytest.raises(ValueError, match="cycles"):
        sine_burst(2.0, 41, 0.05, cycles=0.0)


# --- test_utils.py ---

"""Behavioral tests for PML profile construction."""


def _profiles(setup=utils.setup_pml, **overrides):
    args = {
        "pml_width": [6, 6],
        "pml_start": [8.0, 22.0],
        "max_pml": 0.12,
        "dt": 1e-11,
        "n": 30,
        "max_vel": 3e8,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "pml_freq": 25.0,
        "grid_spacing": 0.02,
    }
    args.update(overrides)
    return setup(**args)


@pytest.mark.parametrize("setup", [utils.setup_pml, utils.setup_pml_half])
def test_pml_profiles_have_valid_shape_and_ranges(setup) -> None:
    a, b, k = _profiles(setup)

    assert a.shape == b.shape == k.shape == (30,)
    assert torch.all((0.0 <= b) & (b <= 1.0))
    assert torch.all(k >= 1.0)
    assert k.max() > 1.0


@pytest.mark.parametrize("setup", [utils.setup_pml, utils.setup_pml_half])
def test_zero_width_disables_pml(setup) -> None:
    a, b, k = _profiles(
        setup,
        pml_width=[0, 0],
        pml_start=[4.0, 12.0],
        max_pml=0.0,
        n=16,
    )

    assert torch.count_nonzero(a) == 0
    assert torch.count_nonzero(b) == 0
    torch.testing.assert_close(k, torch.ones_like(k))


def test_pml_interior_is_unchanged() -> None:
    a, b, k = _profiles()
    interior = slice(11, 19)

    assert torch.count_nonzero(a[interior]) == 0
    assert torch.count_nonzero(b[interior]) == 0
    torch.testing.assert_close(k[interior], torch.ones_like(k[interior]))


def test_grid_spacing_changes_pml_profile() -> None:
    coarse = _profiles(grid_spacing=0.05)
    fine = _profiles(grid_spacing=0.01)

    assert not torch.equal(coarse[0], fine[0])
    assert not torch.equal(coarse[1], fine[1])


def test_custom_parameters_change_pml_profile() -> None:
    default = _profiles()
    custom = _profiles(r_val=1e-6, n_power=3, eps=1e-8)

    assert not torch.equal(custom[0], default[0])
    assert not torch.equal(custom[1], default[1])
    assert not torch.equal(custom[2], default[2])


def test_half_grid_profile_is_shifted() -> None:
    full = _profiles()
    half = _profiles(utils.setup_pml_half)

    for full_profile, half_profile in zip(full, half, strict=True):
        assert not torch.equal(full_profile, half_profile)
        assert full_profile.shape == half_profile.shape


def test_pml_supports_asymmetric_widths() -> None:
    _, b, k = _profiles(pml_width=[4, 8], pml_start=[6.0, 22.0], max_pml=0.16)

    assert torch.count_nonzero(k[:4] > 1.0) > 0
    assert torch.count_nonzero(k[-8:] > 1.0) > 0
    assert torch.all((0.0 <= b) & (b <= 1.0))


def test_pml_time_step_changes_decay_only() -> None:
    _, b1, k1 = _profiles(dt=1e-11)
    _, b2, k2 = _profiles(dt=2e-11)

    torch.testing.assert_close(k1, k2)
    assert b2.max() < b1.max()


def test_physical_constants_match_vacuum_values() -> None:
    assert utils.EP0 == pytest.approx(8.8541878128e-12)
    assert utils.MU0 == pytest.approx(1.2566370614359173e-6)
