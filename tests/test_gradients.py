"""Tests for gradient computation correctness and sampling interval."""

import pytest
import torch

import tide

class TestGradientAccuracy2D:
    """Tests for 2D MaxwellTM gradient accuracy."""

    @pytest.fixture
    def setup_2d(self):
        """Common setup for 2D tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        device = torch.device("cuda")
        dtype = torch.float32

        ny, nx = 20, 24
        nt = 30
        dx = 0.02
        dt = 4e-11
        pml_width = 4
        stencil = 2

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 4]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        freq = 200e6
        wavelet = tide.ricker(freq, nt, dt, peak_time=1.0 / freq, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        return {
            "device": device,
            "dtype": dtype,
            "epsilon": epsilon,
            "sigma": sigma,
            "mu": mu,
            "dx": dx,
            "dt": dt,
            "pml_width": pml_width,
            "stencil": stencil,
            "source_amplitude": source_amplitude,
            "source_locations": source_locations,
            "receiver_locations": receiver_locations,
        }

    def test_epsilon_gradient_finite_difference_2d(self, setup_2d):
        """Compare epsilon gradient with finite difference approximation."""
        s = setup_2d
        ny, nx = s["epsilon"].shape
        h = 1e-2

        # Forward with base epsilon
        eps_base = s["epsilon"].clone().detach().requires_grad_(True)
        out_base = tide.maxwelltm(
            eps_base,
            s["sigma"],
            s["mu"],
            grid_spacing=s["dx"],
            dt=s["dt"],
            source_amplitude=s["source_amplitude"],
            source_location=s["source_locations"],
            receiver_location=s["receiver_locations"],
            pml_width=s["pml_width"],
            stencil=s["stencil"],
        )[-1]
        loss_base = out_base.pow(2).sum()
        loss_base.backward()
        assert eps_base.grad is not None
        grad_autodiff = eps_base.grad.clone()

        # Finite difference: perturb epsilon at a single point
        eps_pert = s["epsilon"].clone()
        eps_pert[ny // 2, nx // 2] += h

        out_pert = tide.maxwelltm(
            eps_pert,
            s["sigma"],
            s["mu"],
            grid_spacing=s["dx"],
            dt=s["dt"],
            source_amplitude=s["source_amplitude"],
            source_location=s["source_locations"],
            receiver_location=s["receiver_locations"],
            pml_width=s["pml_width"],
            stencil=s["stencil"],
        )[-1]

        fd_approx = (out_pert.pow(2).sum() - loss_base.detach()) / h

        # Compare at the perturbed point
        grad_at_point = grad_autodiff[ny // 2, nx // 2]

        # The gradient should have the same sign and similar magnitude
        # Using a looser tolerance since FD is approximate
        assert torch.sign(grad_at_point) == torch.sign(fd_approx), "Gradient sign should match"
        rel_error = abs(grad_at_point - fd_approx) / (abs(fd_approx) + 1e-10)
        assert rel_error < 0.5, f"Gradient FD mismatch too large: {rel_error}"

    def test_sigma_gradient_is_nonzero_2d(self, setup_2d):
        """Sigma gradient should be non-zero for loss function."""
        s = setup_2d

        # Use non-zero sigma
        sigma = torch.ones_like(s["epsilon"]) * 1e-3
        sigma.requires_grad_(True)

        out = tide.maxwelltm(
            s["epsilon"],
            sigma,
            s["mu"],
            grid_spacing=s["dx"],
            dt=s["dt"],
            source_amplitude=s["source_amplitude"],
            source_location=s["source_locations"],
            receiver_location=s["receiver_locations"],
            pml_width=s["pml_width"],
            stencil=s["stencil"],
        )[-1]

        loss = out.pow(2).sum()
        loss.backward()

        assert sigma.grad is not None
        sigma_grad = sigma.grad

        # Gradient should be non-zero somewhere
        assert sigma_grad.abs().sum() > 0, "Sigma gradient should be non-zero"
        assert torch.isfinite(sigma_grad).all(), "Sigma gradient should be finite"


class TestGradientSamplingInterval:
    """Tests for model_gradient_sampling_interval parameter."""

    def test_gradient_sampling_interval_affects_gradient_cpu(self):
        """Test that gradient sampling interval affects gradient computation on CPU."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 16
        nt = 15

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 4]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Compute gradient with sampling interval 1
        eps1 = epsilon.clone().detach().requires_grad_(True)
        out1 = tide.maxwelltm(
            eps1,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=3,
            stencil=2,
            model_gradient_sampling_interval=1,
        )[-1]
        loss1 = out1.pow(2).sum()
        loss1.backward()
        assert eps1.grad is not None
        grad1 = eps1.grad.clone()

        # Compute gradient with sampling interval 3
        eps2 = epsilon.clone().detach().requires_grad_(True)
        out2 = tide.maxwelltm(
            eps2,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_locations,
            receiver_location=receiver_locations,
            pml_width=3,
            stencil=2,
            model_gradient_sampling_interval=3,
        )[-1]
        loss2 = out2.pow(2).sum()
        loss2.backward()
        assert eps2.grad is not None
        grad2 = eps2.grad.clone()

        # Gradients should be different (sampling_interval affects gradient computation)
        # Note: they might be similar if the simulation is short, so we just check they're not identical
        correlation = (grad1 * grad2).sum() / (torch.norm(grad1) * torch.norm(grad2) + 1e-10)
        # Correlation should be high (both approximate the same gradient) but not exactly 1
        assert 0.5 < correlation < 1.0, f"Unexpected gradient correlation: {correlation}"

    def test_gradient_sampling_interval_values(self):
        """Test various sampling interval values."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 10, 12
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 4]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Test that different sampling intervals work without error
        for interval in [1, 2, 5]:
            eps = epsilon.clone().detach().requires_grad_(True)
            out = tide.maxwelltm(
                eps,
                sigma,
                mu,
                grid_spacing=0.02,
                dt=4e-11,
                source_amplitude=source_amplitude,
                source_location=source_locations,
                receiver_location=receiver_locations,
                pml_width=2,
                stencil=2,
                model_gradient_sampling_interval=interval,
            )[-1]

            loss = out.pow(2).sum()
            loss.backward()

            assert eps.grad is not None
            grad = eps.grad

            assert torch.isfinite(grad).all(), f"Gradient should be finite for interval={interval}"


class TestGradientBoundaryConditions:
    """Tests for gradient computation with boundary conditions."""

    def test_gradient_with_pml(self):
        """Test gradient computation with PML boundaries."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 16
        nt = 12

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 4]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Test with PML
        eps = epsilon.clone().detach().requires_grad_(True)
        out = tide.maxwelltm(
            eps,
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

        loss = out.pow(2).sum()
        loss.backward()

        assert eps.grad is not None
        grad = eps.grad

        assert torch.isfinite(grad).all(), "Gradient should be finite with PML"
        assert grad.abs().sum() > 0, "Gradient should be non-zero with PML"

    def test_gradient_without_pml(self):
        """Test gradient computation without PML boundaries."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 12, 16
        nt = 10

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([[[ny // 2, nx // 4]]], dtype=torch.long, device=device)
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        source_amplitude = wavelet.view(1, 1, nt)

        # Test without PML
        eps = epsilon.clone().detach().requires_grad_(True)
        out = tide.maxwelltm(
            eps,
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

        loss = out.pow(2).sum()
        loss.backward()

        # Gradient might have reflections at boundaries, but should still be finite
        assert eps.grad is not None
        grad = eps.grad

        assert torch.isfinite(grad).all(), "Gradient should be finite without PML"


class TestGradientMultiSource:
    """Tests for gradient computation with multiple sources."""

    def test_gradient_multiple_sources(self):
        """Test gradient computation with multiple sources."""
        device = torch.device("cpu")
        dtype = torch.float32

        ny, nx = 14, 18
        nt = 12
        n_sources = 2

        epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor([
            [[ny // 3, nx // 3], [2 * ny // 3, 2 * nx // 3]]
        ], dtype=torch.long, device=device)
        receiver_locations = torch.tensor([
            [[ny // 2, nx // 2], [ny // 2, nx // 2 + 2]]
        ], dtype=torch.long, device=device)

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        # For multiple sources, use the same wavelet for each source
        source_amplitude = wavelet.view(1, 1, nt).expand(1, n_sources, nt)

        eps = epsilon.clone().detach().requires_grad_(True)
        out = tide.maxwelltm(
            eps,
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

        # out shape: [nt, n_shot, n_receiver]
        loss = out.pow(2).sum()
        loss.backward()

        assert eps.grad is not None
        grad = eps.grad

        assert torch.isfinite(grad).all(), "Gradient should be finite for multiple sources"
        assert grad.abs().sum() > 0, "Gradient should be non-zero for multiple sources"
