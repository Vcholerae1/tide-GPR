"""Tests for tide.utils module - PML setup functions."""


import torch

from tide import utils


class TestSetupPML:
    """Tests for setup_pml function."""

    def test_setup_pml_output_shapes(self):
        """PML profiles should have correct shape."""
        pml_width = [4, 4]
        pml_start = [8.0, 20.0]
        max_pml = 0.08
        dt = 1e-11
        n = 32
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a, b, k = utils.setup_pml(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        assert a.shape == (n,)
        assert b.shape == (n,)
        assert k.shape == (n,)

    def test_setup_pml_zero_width(self):
        """Zero PML width should return zero profiles."""
        pml_width = [0, 0]
        pml_start = [4.0, 12.0]
        max_pml = 0.0
        dt = 1e-11
        n = 16
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.01

        a, b, k = utils.setup_pml(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        assert torch.all(a == 0.0)
        assert torch.all(b == 0.0)
        assert torch.all(k == 1.0)

    def test_setup_pml_coefficient_ranges(self):
        """PML coefficients should be in physically valid ranges."""
        pml_width = [8, 8]
        pml_start = [10.0, 26.0]
        max_pml = 0.16
        dt = 1e-11
        n = 40
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a, b, k = utils.setup_pml(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        # a coefficients can be negative or small
        # b coefficients should be in [0, 1] (decay factors)
        assert torch.all(b >= 0.0), "b coefficients should be non-negative"
        assert torch.all(b <= 1.0), "b coefficients should be <= 1"

        # k coefficients should be >= 1
        assert torch.all(k >= 1.0), "k coefficients should be >= 1"
        assert k.max() > 1.0, "k should be > 1 in PML region"

    def test_setup_pml_symmetry(self):
        """PML profiles should have similar behavior on both sides."""
        pml_width = [6, 6]
        pml_start = [8.0, 22.0]  # Symmetric around middle
        max_pml = 0.12
        dt = 1e-11
        n = 30
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a, b, k = utils.setup_pml(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        # Both sides should have k > 1 in the PML region
        left_k = k[:pml_width[0]]
        right_k = k[-pml_width[1]:]

        assert left_k.min() >= 1.0, "Left PML k should be >= 1"
        assert right_k.min() >= 1.0, "Right PML k should be >= 1"
        assert left_k.max() > 1.0, "Left PML should have k > 1"
        assert right_k.max() > 1.0, "Right PML should have k > 1"

    def test_setup_pml_interior_region(self):
        """Interior region should have k=1 and small a, b."""
        pml_width = [5, 5]
        pml_start = [7.0, 20.0]
        max_pml = 0.1
        dt = 1e-11
        n = 30
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a, b, k = utils.setup_pml(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        # In the deep interior (far from PML), k should be 1
        # Check a region well away from PML boundaries
        interior_start = pml_width[0] + 5
        interior_end = n - pml_width[1] - 5

        if interior_end > interior_start:
            interior_k = k[interior_start:interior_end]
            assert torch.allclose(interior_k, torch.ones_like(interior_k), atol=0.01), \
                "k should be 1 in deep interior region"

    def test_setup_pml_different_grid_spacings(self):
        """PML should handle different grid spacings."""
        pml_width = [4, 4]
        pml_start = [6.0, 14.0]
        max_pml = 0.08
        dt = 1e-11
        n = 20
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0

        for grid_spacing in [0.01, 0.02, 0.05]:
            a, b, k = utils.setup_pml(
                pml_width,
                pml_start,
                max_pml,
                dt,
                n,
                max_vel,
                dtype,
                device,
                pml_freq,
                grid_spacing=grid_spacing,
            )

            assert torch.all(b >= 0.0) and torch.all(b <= 1.0)
            assert torch.all(k >= 1.0)


class TestSetupPMLHalf:
    """Tests for setup_pml_half function."""

    def test_setup_pml_half_output_shapes(self):
        """Half-grid PML profiles should have correct shape."""
        pml_width = [4, 4]
        pml_start = [8.0, 20.0]
        max_pml = 0.08
        dt = 1e-11
        n = 32
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a, b, k = utils.setup_pml_half(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        assert a.shape == (n,)
        assert b.shape == (n,)
        assert k.shape == (n,)

    def test_setup_pml_half_zero_width(self):
        """Zero PML width should return zero profiles for half-grid."""
        pml_width = [0, 0]
        pml_start = [4.0, 12.0]
        max_pml = 0.0
        dt = 1e-11
        n = 16
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.01

        a, b, k = utils.setup_pml_half(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        assert torch.all(a == 0.0)
        assert torch.all(b == 0.0)
        assert torch.all(k == 1.0)

    def test_setup_pml_half_coefficient_ranges(self):
        """Half-grid PML coefficients should be in valid ranges."""
        pml_width = [8, 8]
        pml_start = [10.0, 26.0]
        max_pml = 0.16
        dt = 1e-11
        n = 40
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a, b, k = utils.setup_pml_half(
            pml_width,
            pml_start,
            max_pml,
            dt,
            n,
            max_vel,
            dtype,
            device,
            pml_freq,
            grid_spacing=grid_spacing,
        )

        assert torch.all(b >= 0.0), "b coefficients should be non-negative"
        assert torch.all(b <= 1.0), "b coefficients should be <= 1"
        assert torch.all(k >= 1.0), "k coefficients should be >= 1"


class TestPMLPhysicalConstants:
    """Tests for PML physical constants correctness."""

    def test_physical_constants_defined(self):
        """Physical constants should be properly defined."""
        # Check that EP0 and MU0 are defined with correct values
        assert hasattr(utils, 'EP0'), "EP0 (vacuum permittivity) should be defined"
        assert hasattr(utils, 'MU0'), "MU0 (vacuum permeability) should be defined"

        # EP0 should be approximately 8.854e-12 F/m
        assert 8.8e-12 < utils.EP0 < 8.9e-12, f"EP0 = {utils.EP0}, expected ~8.854e-12"

        # MU0 should be approximately 1.257e-6 H/m
        assert 1.25e-6 < utils.MU0 < 1.26e-6, f"MU0 = {utils.MU0}, expected ~1.257e-6"

    def test_pml_time_scaling(self):
        """PML coefficients should scale correctly with time step."""
        pml_width = [6, 6]
        pml_start = [8.0, 22.0]
        max_pml = 0.12
        n = 30
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        # Compute PML for two different time steps
        dt1 = 1e-11
        a1, b1, k1 = utils.setup_pml(
            pml_width, pml_start, max_pml, dt1, n, max_vel, dtype, device, pml_freq,
            grid_spacing=grid_spacing
        )

        dt2 = 2e-11
        a2, b2, k2 = utils.setup_pml(
            pml_width, pml_start, max_pml, dt2, n, max_vel, dtype, device, pml_freq,
            grid_spacing=grid_spacing
        )

        # k should be independent of dt
        torch.testing.assert_close(k1, k2, atol=1e-6, rtol=1e-6)

        # b should change with dt (b = exp(-something * dt))
        # Larger dt should give smaller b values (more decay per step)
        assert b2.max() < b1.max(), "b should decrease with larger dt"


class TestPMLIntegration:
    """Integration tests for PML with full setup."""

    def test_pml_full_vs_half_consistency(self):
        """Full and half-grid PML should be consistent."""
        pml_width = [6, 6]
        pml_start = [8.0, 22.0]
        max_pml = 0.12
        dt = 1e-11
        n = 30
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a_full, b_full, k_full = utils.setup_pml(
            pml_width, pml_start, max_pml, dt, n, max_vel, dtype, device, pml_freq,
            grid_spacing=grid_spacing
        )

        a_half, b_half, k_half = utils.setup_pml_half(
            pml_width, pml_start, max_pml, dt, n, max_vel, dtype, device, pml_freq,
            grid_spacing=grid_spacing
        )

        # Both should have the same range of values
        # (they're shifted by 0.5 grid point, but ranges should be similar)
        assert k_full.min() >= 1.0
        assert k_half.min() >= 1.0
        assert k_full.max() > 1.0
        assert k_half.max() > 1.0

    def test_pml_asymmetric_widths(self):
        """PML should handle asymmetric widths."""
        pml_width = [4, 8]  # Different left/right widths
        pml_start = [6.0, 22.0]
        max_pml = 0.16  # max is from the larger side
        dt = 1e-11
        n = 30
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        a, b, k = utils.setup_pml(
            pml_width, pml_start, max_pml, dt, n, max_vel, dtype, device, pml_freq,
            grid_spacing=grid_spacing
        )

        # Should still produce valid profiles
        assert torch.all(b >= 0.0) and torch.all(b <= 1.0)
        assert torch.all(k >= 1.0)

    def test_pml_custom_parameters(self):
        """PML should accept custom parameters."""
        pml_width = [5, 5]
        pml_start = [8.0, 20.0]
        max_pml = 0.1
        dt = 1e-11
        n = 28
        max_vel = 3e8
        dtype = torch.float32
        device = torch.device("cpu")
        pml_freq = 25.0
        grid_spacing = 0.02

        # Custom parameters
        custom_r_val = 1e-6
        custom_n_power = 3
        custom_eps = 1e-8

        a, b, k = utils.setup_pml(
            pml_width, pml_start, max_pml, dt, n, max_vel, dtype, device, pml_freq,
            grid_spacing=grid_spacing,
            r_val=custom_r_val,
            n_power=custom_n_power,
            eps=custom_eps,
        )

        # Should produce valid profiles
        assert torch.all(b >= 0.0) and torch.all(b <= 1.0)
        assert torch.all(k >= 1.0)
