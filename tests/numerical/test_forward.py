from __future__ import annotations

import math
import pytest
import tide
import torch
from numerical_utils import (
    MaxwellExample,
    relative_l2,
    assert_finite_nonzero,
    cosine_similarity,
    make_tm2d_example,
    require_native_backend,
    signal_rms,
)
from tide.cfl import cfl_condition
from tide.grid_utils import _normalize_grid_spacing_2d, _normalize_grid_spacing_3d

# --- test_forward_physics.py ---


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


# --- test_maxwell3d_python_forward.py ---


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


@pytest.mark.parametrize("device", _devices())
def test_maxwell3d_python_forward_long_nt_stability(device: torch.device):
    """Long propagation should remain finite and not show late-time blow-up."""
    dtype = torch.float32
    nz = ny = nx = 20
    nt = 1000
    dt = 1.6e-11
    dz = dy = dx = 0.01

    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)

    source_location = torch.tensor(
        [[[3, ny // 2, nx // 2]]],
        dtype=torch.long,
        device=device,
    )
    receiver_location = torch.tensor(
        [[[3, ny // 2, nx // 2 + 4]]],
        dtype=torch.long,
        device=device,
    )
    source_amplitude = tide.ricker(
        160e6,
        nt,
        dt,
        peak_time=1.2 / 160e6,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)

    wavefield_peaks: list[float] = []

    def cb(state):
        ey = state.get_wavefield("Ey", view="inner")
        wavefield_peaks.append(float(ey.abs().max().detach().cpu()))

    out = tide.maxwell._kernel_api.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=[dz, dy, dx],
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=6,
        stencil=4,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
        forward_callback=cb,
        callback_frequency=5,
    )

    receiver = out[-1]
    assert receiver.shape == (nt, 1, 1)
    assert torch.isfinite(receiver).all()
    assert receiver.abs().max() > 0

    peaks = torch.tensor(wavefield_peaks)
    assert peaks.numel() > 10
    assert torch.isfinite(peaks).all()
    assert peaks.max() > 0
    late_ratio = peaks[-20:].max() / peaks.max()
    assert late_ratio < 1e-2


# --- test_anisotropic_grid.py ---


@pytest.mark.numerical
def test_grid_spacing_normalization_preserves_axis_order() -> None:
    assert _normalize_grid_spacing_2d(0.02) == [0.02, 0.02]
    assert _normalize_grid_spacing_2d([0.018, 0.022]) == [0.018, 0.022]
    assert _normalize_grid_spacing_3d(0.02) == [0.02, 0.02, 0.02]
    assert _normalize_grid_spacing_3d([0.016, 0.018, 0.022]) == [
        0.016,
        0.018,
        0.022,
    ]


@pytest.mark.numerical
def test_cfl_uses_every_active_axis_spacing() -> None:
    dt = 1.0e-10
    velocity = 1.5e8
    with pytest.warns(UserWarning, match="CFL condition requires"):
        inner_dt, ratio = cfl_condition([0.01, 0.02, 0.04], dt, velocity)
    expected_max_dt = (
        1.0
        / math.sqrt(sum(1.0 / spacing**2 for spacing in (0.01, 0.02, 0.04)))
        / velocity
    )
    expected_ratio = math.ceil(dt / expected_max_dt)
    assert ratio == expected_ratio
    assert inner_dt == pytest.approx(dt / expected_ratio)


def _tm_source_field(grid_spacing: list[float]) -> torch.Tensor:
    dtype = torch.float64
    epsilon = torch.full((12, 14), 4.0, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source = torch.ones((1, 1, 1), dtype=dtype)
    source_location = torch.tensor([[[6, 7]]], dtype=torch.long)
    return tide.maxwell._kernel_api.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing,
        1.0e-11,
        source,
        source_location,
        None,
        nt=1,
        pml_width=0,
        python_backend=True,
    )[0]


def _em3d_source_field(grid_spacing: list[float]) -> torch.Tensor:
    dtype = torch.float64
    epsilon = torch.full((8, 9, 10), 4.0, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source = torch.ones((1, 1, 1), dtype=dtype)
    source_location = torch.tensor([[[4, 4, 5]]], dtype=torch.long)
    return tide.maxwell._kernel_api.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing,
        1.0e-11,
        source,
        source_location,
        None,
        nt=1,
        pml_width=0,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[1]


@pytest.mark.numerical
@pytest.mark.parametrize(
    "runner,coarse,fine",
    [
        (_tm_source_field, [0.04, 0.02], [0.02, 0.02]),
        (_em3d_source_field, [0.04, 0.02, 0.02], [0.02, 0.02, 0.02]),
    ],
    ids=["tm2d-area", "maxwell3d-volume"],
)
def test_source_injection_uses_cell_measure(runner, coarse, fine) -> None:
    coarse_peak = float(runner(coarse).abs().max())
    fine_peak = float(runner(fine).abs().max())
    expected_coarse = 1.0e-11 / (4.0 * 8.8541878128e-12 * math.prod(coarse))
    expected_fine = 1.0e-11 / (4.0 * 8.8541878128e-12 * math.prod(fine))
    assert coarse_peak == pytest.approx(expected_coarse, rel=1.0e-12)
    assert fine_peak == pytest.approx(expected_fine, rel=1.0e-12)


@pytest.mark.numerical
@pytest.mark.parametrize("dimension", [2, 3])
def test_anisotropic_forward_and_material_gradients_are_finite(dimension: int) -> None:
    dtype = torch.float64
    nt = 36
    if dimension == 2:
        shape = (16, 18)
        spacing = [0.018, 0.022]
        source_location = torch.tensor([[[8, 6]]], dtype=torch.long)
        receiver_location = torch.tensor([[[8, 12]]], dtype=torch.long)
        solver = tide.maxwell._kernel_api.maxwelltm
        component_kwargs = {}
    else:
        shape = (8, 9, 10)
        spacing = [0.016, 0.018, 0.022]
        source_location = torch.tensor([[[4, 4, 3]]], dtype=torch.long)
        receiver_location = torch.tensor([[[4, 4, 7]]], dtype=torch.long)
        solver = tide.maxwell._kernel_api.maxwell3d
        component_kwargs = {"source_component": "ey", "receiver_component": "ey"}

    epsilon = torch.full(shape, 4.0, dtype=dtype, requires_grad=True)
    sigma = torch.full(shape, 2.0e-4, dtype=dtype, requires_grad=True)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(450e6, nt, 2.0e-11, peak_time=5.0e-10, dtype=dtype).view(
        1, 1, nt
    )
    receiver = solver(
        epsilon,
        sigma,
        mu,
        spacing,
        2.0e-11,
        source,
        source_location,
        receiver_location,
        stencil=4,
        pml_width=3,
        python_backend=False,
        storage_compression=False,
        **component_kwargs,
    )[-1]
    receiver.square().sum().backward()
    assert epsilon.grad is not None and sigma.grad is not None
    assert_finite_nonzero(receiver, epsilon.grad, sigma.grad)


@pytest.mark.numerical
def test_scalar_and_equal_axis_spacing_match_exactly(
    tm2d_example: MaxwellExample,
) -> None:
    scalar = tm2d_example.run(grid_spacing=0.02)[-1]
    sequence = tm2d_example.run(grid_spacing=[0.02, 0.02])[-1]
    assert relative_l2(sequence, scalar) == 0.0


# --- test_cpml_absorption.py ---


def _reflection_response(pml_width: int) -> torch.Tensor:
    example = make_tm2d_example(
        shape=(24, 50),
        nt=450,
        grid_spacing=0.02,
        dt=2.0e-11,
        frequency=300e6,
        peak_time=1.5e-9,
        dtype=torch.float64,
        source_location=(12, 15),
        receiver_locations=((12, 20),),
        pml_width=pml_width,
        python_backend=True,
    )
    return example.run()[-1]


@pytest.mark.numerical
def test_cpml_preserves_early_trace_and_reduces_late_reflection() -> None:
    reflective = _reflection_response(0)
    early_stop = 100
    late_start = 280
    reflective_late_rms = signal_rms(reflective[late_start:])
    assert reflective_late_rms > 0.0

    for width in (4, 8, 12):
        absorbed = _reflection_response(width)
        assert relative_l2(absorbed[:early_stop], reflective[:early_stop]) < 1.0e-5
        assert signal_rms(absorbed[late_start:]) / reflective_late_rms < 0.30


def _tm_gradient(
    *, python_backend: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    example = make_tm2d_example(
        shape=(14, 16),
        nt=48,
        grid_spacing=0.02,
        dt=2.5e-11,
        frequency=250e6,
        peak_time=1.0e-9,
        dtype=torch.float64,
        sigma=1.0e-4,
        source_location=(7, 5),
        receiver_locations=((7, 10),),
        pml_width=3,
        stencil=4,
    )
    epsilon = example.epsilon.requires_grad_()
    sigma = example.sigma.requires_grad_()
    receiver = example.run(
        epsilon=epsilon,
        sigma=sigma,
        python_backend=python_backend,
        storage_compression=False,
    )[-1]
    receiver.square().sum().backward()
    assert epsilon.grad is not None
    assert sigma.grad is not None
    return receiver.detach(), epsilon.grad.detach(), sigma.grad.detach()


@pytest.mark.numerical
def test_cpml_native_foldback_matches_reference() -> None:
    require_native_backend()
    receiver_reference, epsilon_reference, sigma_reference = _tm_gradient(
        python_backend=True
    )
    receiver_native, epsilon_native, sigma_native = _tm_gradient(python_backend=False)
    assert relative_l2(receiver_native, receiver_reference) < 2.0e-4
    for actual, reference in (
        (epsilon_native, epsilon_reference),
        (sigma_native, sigma_reference),
    ):
        assert relative_l2(actual, reference) < 5.0e-3
        assert cosine_similarity(actual, reference) > 0.999
