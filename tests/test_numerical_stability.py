"""Tests for numerical stability of electromagnetic simulations."""

import pytest
import torch

import tide
from numerical_utils import assert_finite_nonzero, require_cuda_backend


class TestLongRunStability:
    """Tests for stability over long simulation times."""

    def test_long_run_stability_cpu(self):
        """Test that simulation remains stable over many time steps."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 16, 16
        nt = 150  # Longer simulation

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        # Add some conductivity to damp the simulation
        sigma = torch.ones(ny, nx, device=device, dtype=dtype) * 0.001
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=4,
            stencil=2,
        )[-1]

        # Output should remain finite throughout
        assert torch.isfinite(out).all(), "Simulation should remain stable"

        # With damping, signal should decay over time
        # Check that late-time signal is smaller than early peak
        peak_idx = out.abs().argmax()
        early_peak = out[: peak_idx + 10].abs().max()
        if peak_idx + 30 < nt:
            late_signal = out[peak_idx + 30 :].abs().max()
            # Late signal should be attenuated
            assert late_signal < early_peak * 1.5, (
                "Signal should not grow significantly"
            )

    def test_cfl_condition_stability(self):
        """Test that CFL condition is respected."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 20, 20
        nt = 100

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        # Use a time step that satisfies CFL
        # For epsilon=4, v = c/sqrt(4) = c/2 = 1.5e8
        # CFL: dt <= dx / (v * sqrt(2)) ≈ 0.02 / (1.5e8 * 1.414) ≈ 9e-11
        dt = 5e-11  # Conservative time step

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=dt,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=4,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()


class TestHighContrastStability:
    """Tests for stability with high material contrast."""

    def test_high_contrast_interface(self):
        """Test stability at high contrast material interface."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 20, 20
        nt = 80

        # Create a high contrast interface
        epsilon = torch.ones(ny, nx, device=device, dtype=dtype)
        epsilon[:, : nx // 2] = 1.0  # Left side: vacuum
        epsilon[:, nx // 2 :] = 80.0  # Right side: water-like

        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        # Source on the left side
        source_locations = torch.tensor(
            [[[ny // 2, nx // 4]]], dtype=torch.long, device=device
        )
        # Receiver on the right side
        receiver_locations = torch.tensor(
            [[[ny // 2, 3 * nx // 4]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=4,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_very_high_permittivity(self):
        """Test stability with very high permittivity."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 12
        nt = 50

        # Very high permittivity
        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 100.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        freq = 50e6  # Lower frequency for high permittivity
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=2e-11,  # Smaller dt for high epsilon
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=3,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()


class TestLossyMediaStability:
    """Tests for stability in lossy (conducting) media."""

    def test_high_conductivity(self):
        """Test stability with high conductivity."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 12
        nt = 60

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        # High conductivity (like good conductor)
        sigma = torch.ones(ny, nx, device=device, dtype=dtype) * 1.0
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=3,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_lossy_attenuation(self):
        """Test that lossy media attenuates the signal."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 20, 20
        nt = 80

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 4]]], dtype=torch.long, device=device
        )
        # Receiver far away
        receiver_locations = torch.tensor(
            [[[ny // 2, 3 * nx // 4]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        # Lossless case
        out_lossless = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=4,
            stencil=2,
        )[-1]

        # Lossy case
        sigma_lossy = torch.ones(ny, nx, device=device, dtype=dtype) * 0.1
        out_lossy = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma_lossy,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=4,
            stencil=2,
        )[-1]

        # Lossy signal should be attenuated (smaller amplitude)
        assert out_lossy.abs().max() < out_lossless.abs().max()


class TestDifferentStencilStability:
    """Tests for stability with different stencil orders."""

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_stencil_stability(self, stencil):
        """All stencil orders should be stable."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 14, 14
        nt = 80

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=3,
            stencil=stencil,
        )[-1]

        assert torch.isfinite(out).all()


class TestEnergyConservation:
    """Tests related to energy conservation in lossless media."""

    def test_lossless_energy_behavior(self):
        """Test that energy behaves reasonably in lossless media."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 12
        nt = 50

        # Lossless medium
        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=4,
            stencil=2,
        )[-1]

        # Output should decay after source stops (energy absorbed by PML)
        # Find where source ends
        source_end_idx = wavelet.abs().argmax()

        # Early signal (around peak)
        early_start = max(0, source_end_idx - 5)
        early_end = min(nt, source_end_idx + 10)
        early_signal = out[early_start:early_end].abs().max()

        # Late time signal (should be smaller due to PML absorption)
        if source_end_idx + 20 < nt:
            late_signal = out[source_end_idx + 20 :].abs().max()
        else:
            late_signal = torch.tensor(0.0)

        assert late_signal <= early_signal

        # With PML, late signal should be attenuated
        assert torch.isfinite(out).all()


@pytest.mark.slow
@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize(
    "dimension,device_kind",
    [
        (2, "cpu"),
        (3, "cpu"),
        pytest.param(2, "cuda", marks=pytest.mark.cuda),
        pytest.param(3, "cuda", marks=pytest.mark.cuda),
    ],
)
def test_native_long_run_forward_and_backward_stay_stable(
    dimension: int, device_kind: str, stencil: int
) -> None:
    device = require_cuda_backend() if device_kind == "cuda" else torch.device("cpu")
    dtype = torch.float32
    nt = 220
    dt = 2.0e-11
    if dimension == 2:
        shape = (24, 28)
        spacing = [0.018, 0.022]
        source_location = torch.tensor([[[12, 8]]], device=device, dtype=torch.long)
        receiver_location = torch.tensor(
            [[[12, 14], [14, 16]]], device=device, dtype=torch.long
        )
        solver = tide.maxwell._kernel_api.maxwelltm
        component_kwargs = {}
    else:
        shape = (12, 13, 14)
        spacing = [0.016, 0.018, 0.022]
        source_location = torch.tensor([[[6, 6, 4]]], device=device, dtype=torch.long)
        receiver_location = torch.tensor(
            [[[6, 6, 9], [7, 8, 9]]], device=device, dtype=torch.long
        )
        solver = tide.maxwell._kernel_api.maxwell3d
        component_kwargs = {"source_component": "ey", "receiver_component": "ey"}

    epsilon = torch.full(shape, 4.0, device=device, dtype=dtype)
    epsilon.reshape(-1)[epsilon.numel() // 3 : epsilon.numel() // 2] = 7.0
    epsilon.requires_grad_(True)
    sigma = torch.full(shape, 2.0e-4, device=device, dtype=dtype)
    sigma.reshape(-1)[sigma.numel() // 2 : 2 * sigma.numel() // 3] = 2.0e-3
    sigma.requires_grad_(True)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(
        350e6, nt, dt, peak_time=1.0e-9, device=device, dtype=dtype
    ).view(1, 1, nt)
    receiver = solver(
        epsilon,
        sigma,
        mu,
        spacing,
        dt,
        source,
        source_location,
        receiver_location,
        stencil=stencil,
        pml_width=4,
        python_backend=False,
        storage_compression=False,
        **component_kwargs,
    )[-1]
    receiver.square().mean().backward()
    assert epsilon.grad is not None and sigma.grad is not None
    assert_finite_nonzero(receiver, epsilon.grad, sigma.grad)
    middle_peak = receiver[55:150].abs().max()
    late_peak = receiver[175:].abs().max()
    assert late_peak <= 1.5 * middle_peak
