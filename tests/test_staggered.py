"""Tests for staggered grid operations in tide.staggered module."""

import math

import pytest
import torch

from tide import staggered


class TestDiffY1:
    """Tests for diffy1 - first y derivative at integer grid points."""

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_diffy1_linear_function(self, stencil):
        """First derivative of linear function should be constant."""
        ny, nx = 32, 32
        dy = 0.1
        rdy = torch.tensor(1.0 / dy, dtype=torch.float32)

        # f(y) = 2*y, so df/dy = 2
        y = torch.arange(ny, dtype=torch.float32) * dy
        f = 2.0 * y
        a = f.expand(nx, ny).T  # Shape: [ny, nx]

        result = staggered.diffy1(a, stencil, rdy)

        # The result should be approximately 2.0 in the valid region
        expected = torch.full((ny, nx), 2.0, dtype=torch.float32)

        # Valid region depends on stencil order:
        # stencil=2: [1:]  (1 row top padding)
        # stencil=4: [2:-1] (2 rows top, 1 row bottom)
        # stencil=6: [3:-2] (3 rows top, 2 rows bottom)
        # stencil=8: [4:-3] (4 rows top, 3 rows bottom)
        pad_top = stencil // 2
        pad_bottom = pad_top - 1 if stencil > 2 else 0
        actual = result[pad_top:ny-pad_bottom if pad_bottom > 0 else ny, :]
        expected_actual = expected[pad_top:ny-pad_bottom if pad_bottom > 0 else ny, :]

        torch.testing.assert_close(actual, expected_actual, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_diffy1_sine_function(self, stencil):
        """First derivative of sin should be cos."""
        ny, nx = 64, 32
        dy = 0.05
        rdy = torch.tensor(1.0 / dy, dtype=torch.float32)

        y = torch.arange(ny, dtype=torch.float32) * dy
        f = torch.sin(2.0 * math.pi * y / (ny * dy))
        a = f.expand(nx, ny).T

        result = staggered.diffy1(a, stencil, rdy)

        # Expected: df/dy = (2*pi/L) * cos(2*pi*y/L)
        L = ny * dy
        k = 2.0 * math.pi / L
        expected = k * torch.cos(2.0 * math.pi * y / L)

        # Check in the valid region (away from boundaries)
        pad_top = stencil // 2 + 2
        pad_bottom = pad_top - 1 if stencil > 2 else 1
        actual = result[pad_top:ny-pad_bottom, 0]
        expected_slice = expected[pad_top:ny-pad_bottom]

        # Use absolute tolerance mainly, as relative tolerance can blow up near zeros
        # Higher stencil orders should have better accuracy
        atol = 0.15 if stencil == 2 else 0.12
        torch.testing.assert_close(actual, expected_slice, atol=atol, rtol=1.0)

    def test_diffy1_output_shape(self):
        """Output shape should match input shape."""
        ny, nx = 16, 24
        a = torch.randn(ny, nx, dtype=torch.float32)
        rdy = torch.tensor(1.0, dtype=torch.float32)

        for stencil in [2, 4, 6, 8]:
            result = staggered.diffy1(a, stencil, rdy)
            assert result.shape == a.shape


class TestDiffX1:
    """Tests for diffx1 - first x derivative at integer grid points."""

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_diffx1_linear_function(self, stencil):
        """First derivative of linear function should be constant."""
        ny, nx = 32, 32
        dx = 0.1
        rdx = torch.tensor(1.0 / dx, dtype=torch.float32)

        # f(x) = 3*x, so df/dx = 3
        x = torch.arange(nx, dtype=torch.float32) * dx
        f = 3.0 * x
        a = f.expand(ny, nx)  # Shape: [ny, nx]

        result = staggered.diffx1(a, stencil, rdx)

        # The result should be approximately 3.0 in the valid region
        expected = torch.full((ny, nx), 3.0, dtype=torch.float32)

        # Valid region depends on stencil order
        pad_left = stencil // 2
        pad_right = pad_left - 1 if stencil > 2 else 0
        actual = result[:, pad_left:nx-pad_right if pad_right > 0 else nx]
        expected_actual = expected[:, pad_left:nx-pad_right if pad_right > 0 else nx]

        torch.testing.assert_close(actual, expected_actual, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_diffx1_sine_function(self, stencil):
        """First derivative of sin should be cos."""
        ny, nx = 32, 64
        dx = 0.05
        rdx = torch.tensor(1.0 / dx, dtype=torch.float32)

        x = torch.arange(nx, dtype=torch.float32) * dx
        f = torch.sin(2.0 * math.pi * x / (nx * dx))
        a = f.expand(ny, nx)

        result = staggered.diffx1(a, stencil, rdx)

        # Expected: df/dx = (2*pi/L) * cos(2*pi*x/L)
        L = nx * dx
        k = 2.0 * math.pi / L
        expected = k * torch.cos(2.0 * math.pi * x / L)

        pad_left = stencil // 2 + 2
        pad_right = pad_left - 1 if stencil > 2 else 1
        actual = result[0, pad_left:nx-pad_right]
        expected_slice = expected[pad_left:nx-pad_right]

        # Use absolute tolerance mainly, as relative tolerance can blow up near zeros
        atol = 0.15 if stencil == 2 else 0.12
        torch.testing.assert_close(actual, expected_slice, atol=atol, rtol=1.0)

    def test_diffx1_output_shape(self):
        """Output shape should match input shape."""
        ny, nx = 16, 24
        a = torch.randn(ny, nx, dtype=torch.float32)
        rdx = torch.tensor(1.0, dtype=torch.float32)

        for stencil in [2, 4, 6, 8]:
            result = staggered.diffx1(a, stencil, rdx)
            assert result.shape == a.shape


class TestDiffYH1:
    """Tests for diffyh1 - first y derivative at half grid points."""

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_diffyh1_linear_function(self, stencil):
        """First derivative of linear function should be constant."""
        ny, nx = 32, 32
        dy = 0.1
        rdy = torch.tensor(1.0 / dy, dtype=torch.float32)

        y = torch.arange(ny, dtype=torch.float32) * dy
        f = 2.0 * y
        a = f.expand(nx, ny).T

        result = staggered.diffyh1(a, stencil, rdy)

        expected = torch.full((ny, nx), 2.0, dtype=torch.float32)

        # For half-grid derivatives, valid region is:
        # stencil=2: [1:-1]  (1 row top, 1 row bottom)
        # stencil=4: [2:-2] (2 rows top, 2 rows bottom)
        # etc.
        pad = stencil // 2
        actual = result[pad:ny-pad, :]
        expected_actual = expected[pad:ny-pad, :]

        torch.testing.assert_close(actual, expected_actual, atol=1e-5, rtol=1e-5)

    def test_diffyh1_output_shape(self):
        """Output shape should match input shape."""
        ny, nx = 16, 24
        a = torch.randn(ny, nx, dtype=torch.float32)
        rdy = torch.tensor(1.0, dtype=torch.float32)

        for stencil in [2, 4, 6, 8]:
            result = staggered.diffyh1(a, stencil, rdy)
            assert result.shape == a.shape


class TestDiffXH1:
    """Tests for diffxh1 - first x derivative at half grid points."""

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_diffxh1_linear_function(self, stencil):
        """First derivative of linear function should be constant."""
        ny, nx = 32, 32
        dx = 0.1
        rdx = torch.tensor(1.0 / dx, dtype=torch.float32)

        x = torch.arange(nx, dtype=torch.float32) * dx
        f = 3.0 * x
        a = f.expand(ny, nx)

        result = staggered.diffxh1(a, stencil, rdx)

        expected = torch.full((ny, nx), 3.0, dtype=torch.float32)

        # For half-grid derivatives, valid region is symmetric
        pad = stencil // 2
        actual = result[:, pad:nx-pad]
        expected_actual = expected[:, pad:nx-pad]

        torch.testing.assert_close(actual, expected_actual, atol=1e-5, rtol=1e-5)

    def test_diffxh1_output_shape(self):
        """Output shape should match input shape."""
        ny, nx = 16, 24
        a = torch.randn(ny, nx, dtype=torch.float32)
        rdx = torch.tensor(1.0, dtype=torch.float32)

        for stencil in [2, 4, 6, 8]:
            result = staggered.diffxh1(a, stencil, rdx)
            assert result.shape == a.shape


class TestDiffZH1:
    """Tests for diffzh1 - first z derivative at half grid points (3D)."""

    @pytest.mark.parametrize("stencil", [2, 4, 6, 8])
    def test_diffzh1_linear_function(self, stencil):
        """First derivative of linear function should be constant."""
        nz, ny, nx = 16, 16, 16
        dz = 0.1
        rdz = torch.tensor(1.0 / dz, dtype=torch.float32)

        z = torch.arange(nz, dtype=torch.float32) * dz
        f = 2.0 * z
        a = f.reshape(nz, 1, 1).expand(nz, ny, nx)

        result = staggered.diffzh1(a, stencil, rdz)

        expected = torch.full((nz, ny, nx), 2.0, dtype=torch.float32)

        # For half-grid derivatives, valid region is symmetric
        pad = stencil // 2
        actual = result[pad:nz-pad, :, :]
        expected_actual = expected[pad:nz-pad, :, :]

        torch.testing.assert_close(actual, expected_actual, atol=1e-5, rtol=1e-5)

    def test_diffzh1_output_shape(self):
        """Output shape should match input shape."""
        nz, ny, nx = 8, 12, 16
        a = torch.randn(nz, ny, nx, dtype=torch.float32)
        rdz = torch.tensor(1.0, dtype=torch.float32)

        for stencil in [2, 4, 6, 8]:
            result = staggered.diffzh1(a, stencil, rdz)
            assert result.shape == a.shape


class TestSetPMLProfiles2D:
    """Tests for set_pml_profiles - 2D PML profile setup."""

    def test_set_pml_profiles_2d_shapes(self):
        """PML profiles should have correct shapes."""
        pml_width = [4, 4, 4, 4]  # [y0, y1, x0, x1]
        accuracy = 4
        fd_pad = [2, 2, 2, 2]
        dt = 1e-11
        grid_spacing = [0.01, 0.01]
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        ny, nx = 32, 32

        profiles = staggered.set_pml_profiles(
            pml_width, accuracy, fd_pad, dt, grid_spacing,
            max_vel, dtype, device, pml_freq, ny, nx
        )
        ab_profiles, k_profiles = profiles[:8], profiles[8:]

        # ab_profiles: [ay, ayh, ax, axh, by, byh, bx, bxh]
        # k_profiles: [ky, kyh, kx, kxh]
        assert len(ab_profiles) == 8
        assert len(k_profiles) == 4

        # Each profile should have length matching the dimension
        assert ab_profiles[0].shape == (1, ny, 1)  # ay
        assert ab_profiles[2].shape == (1, 1, nx)  # ax
        assert k_profiles[0].shape == (1, ny, 1)   # ky
        assert k_profiles[2].shape == (1, 1, nx)   # kx

    def test_set_pml_profiles_zero_width(self):
        """Zero PML width should produce valid (mostly zero) profiles."""
        pml_width = [0, 0, 0, 0]
        accuracy = 2
        fd_pad = [1, 1, 1, 1]
        dt = 1e-11
        grid_spacing = [0.01, 0.01]
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        ny, nx = 16, 16

        profiles = staggered.set_pml_profiles(
            pml_width, accuracy, fd_pad, dt, grid_spacing,
            max_vel, dtype, device, pml_freq, ny, nx
        )
        ab_profiles, k_profiles = profiles[:8], profiles[8:]

        # With zero PML width, a and b profiles should be all zeros
        # k profiles should be all ones
        for ab in ab_profiles:
            assert torch.all(ab == 0.0)
        for k in k_profiles:
            assert torch.all(k == 1.0)

    def test_set_pml_profiles_coefficient_ranges(self):
        """PML coefficients should be in physically reasonable ranges."""
        pml_width = [8, 8, 8, 8]
        accuracy = 4
        fd_pad = [2, 2, 2, 2]
        dt = 1e-11
        grid_spacing = [0.01, 0.01]
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        ny, nx = 32, 32
        profiles = staggered.set_pml_profiles(
            pml_width, accuracy, fd_pad, dt, grid_spacing,
            max_vel, dtype, device, pml_freq, ny, nx
        )
        
        ay, ayh, ax, axh, by, byh, bx, bxh,ky, kyh, kx, kxh = profiles
        # a coefficients can be negative or small (CPML recursive convolution)
        # b coefficients are decay factors, should be in [0, 1]
        for b in [by, byh, bx, bxh]:
            assert torch.all(b >= 0.0), "b coefficients should be non-negative"
            assert torch.all(b <= 1.0), "b coefficients should be <= 1"

        # k coefficients should be >= 1 (stretching factor)
        for k in [ky, kyh, kx, kxh]:
            assert torch.all(k >= 1.0), "k profiles should be >= 1"
            # k should be 1 in interior and > 1 in PML
            assert k.max() > 1.0, "k should be > 1 in PML region"
