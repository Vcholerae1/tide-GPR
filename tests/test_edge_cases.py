"""Tests for edge cases and error handling."""

import pytest
import torch

import tide


class TestEdgeCaseGridSizes:
    """Tests for edge cases related to grid sizes."""

    def test_small_grid_cpu(self):
        """Test with very small grid size on CPU."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 4, 4
        nt = 5

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype)
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Should not raise an error for small grid
        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=1,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_single_cell_grid_cpu(self):
        """Test with single cell grid (minimum viable)."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 1, 1
        nt = 3

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype)
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[0, 0]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[0, 0]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Should handle single cell grid
        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=0,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_rectangular_grid(self):
        """Test with non-square (rectangular) grid."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 8, 16  # 2:1 aspect ratio
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype)
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 2]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=(0.02, 0.02),
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=2,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()


class TestEdgeCaseMaterialParameters:
    """Tests for edge cases related to material parameters."""

    def test_vacuum_parameters(self):
        """Test with vacuum (epsilon=1, sigma=0, mu=1)."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 10

        # Vacuum: all parameters are 1.0
        epsilon = torch.ones(ny, nx, device=device, dtype=dtype)
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=2,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_high_permittivity(self):
        """Test with high permittivity material."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 10

        # High permittivity (like water: ~80)
        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 80.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=2,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_lossy_material(self):
        """Test with lossy (conducting) material."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        # Conductivity like wet soil
        sigma = torch.ones(ny, nx, device=device, dtype=dtype) * 0.01
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=2,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_inhomogeneous_material(self):
        """Test with spatially varying material parameters."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 12
        nt = 12

        # Create a simple layered medium
        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        epsilon[ny // 2:, :] = 9.0  # Second half has higher permittivity

        sigma = torch.zeros(ny, nx, device=device, dtype=dtype)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 4, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 4, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=2,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()


class TestEdgeCaseSourceReceiver:
    """Tests for edge cases related to sources and receivers."""

    def test_source_at_boundary(self):
        """Test with source at domain boundary."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        # Source at corner
        source_locations = torch.tensor([[[0, 0]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=0,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()

    def test_no_source(self):
        """Test with no source (zero amplitude)."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt) * 0.0  # Zero amplitude

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=2,
            stencil=2,
        )[-1]

        # With no source, output should be very small (essentially zero)
        assert out.abs().max() < 1e-10

    def test_single_time_step(self):
        """Test with single time step."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 8, 8
        nt = 1

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=1,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()
        assert out.shape[0] == nt


class TestEdgeCasePML:
    """Tests for edge cases related to PML."""

    def test_no_pml(self):
        """Test without any PML (reflecting boundaries)."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 15

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=0,
            stencil=2,
        )[-1]

        # Should be finite (with reflections from boundaries)
        assert torch.isfinite(out).all()

    def test_large_pml(self):
        """Test with large PML width."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 12
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Large PML (almost half the domain)
        out = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=5,
            stencil=2,
        )[-1]

        assert torch.isfinite(out).all()


class TestEdgeCaseDifferentStencils:
    """Tests for different stencil orders."""

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_all_stencil_orders(self, stencil):
        """All stencil orders should work."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 10
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 2]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([[[ny // 2, nx // 2 + 1]]], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
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
            pml_width=2,
            stencil=stencil,
        )[-1]

        assert torch.isfinite(out).all()

