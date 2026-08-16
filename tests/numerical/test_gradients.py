from __future__ import annotations

import pytest
import tide
import torch
from numerical_utils import (
    MaxwellExample,
    convergence_orders,
    cosine_similarity,
    deterministic_direction,
    directional_derivative_errors,
    make_tm2d_example,
    relative_l2,
    taylor_remainders,
    make_maxwell3d_example,
)

# --- test_gradients.py ---

"""Tests for gradient computation correctness and sampling interval."""


class TestGradientAccuracy2D:
    """Tests for 2D MaxwellTM gradient accuracy."""

    @pytest.fixture
    def setup_2d(self) -> MaxwellExample:
        """Common setup for 2D tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        return make_tm2d_example(
            shape=(20, 24),
            nt=30,
            grid_spacing=0.02,
            dt=4e-11,
            frequency=200e6,
            device="cuda",
            source_location=(10, 6),
            receiver_locations=((10, 12),),
            pml_width=4,
            stencil=2,
        )

    def test_epsilon_gradient_finite_difference_2d(
        self,
        setup_2d: MaxwellExample,
    ):
        """Compare epsilon gradient with finite difference approximation."""
        example = setup_2d
        h = 1e-2
        epsilon = example.epsilon.clone().requires_grad_(True)
        receiver = example.run(epsilon=epsilon)[-1]
        loss = receiver.pow(2).sum()
        loss.backward()
        assert epsilon.grad is not None

        index = (example.epsilon.shape[0] // 2, example.epsilon.shape[1] // 2)
        perturbed = example.epsilon.clone()
        perturbed[index] += h
        finite_difference = (
            example.run(epsilon=perturbed)[-1].pow(2).sum() - loss.detach()
        ) / h
        gradient = epsilon.grad[index]

        assert torch.sign(gradient) == torch.sign(finite_difference), (
            "Gradient sign should match"
        )
        relative_error = abs(gradient - finite_difference) / (
            abs(finite_difference) + 1e-10
        )
        assert relative_error < 0.5, f"Gradient FD mismatch too large: {relative_error}"


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

        source_locations = torch.tensor(
            [[[ny // 2, nx // 4]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2]]], dtype=torch.long, device=device
        )

        freq = 100e6
        wavelet = tide.ricker(
            freq, nt, 4e-11, peak_time=1.0 / freq, dtype=dtype, device=device
        )
        source_amplitude = wavelet.view(1, 1, nt)

        # Compute gradient with sampling interval 1
        eps1 = epsilon.clone().detach().requires_grad_(True)
        out1 = tide.maxwell._kernel_api.maxwelltm(
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
        out2 = tide.maxwell._kernel_api.maxwelltm(
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
        correlation = (grad1 * grad2).sum() / (
            torch.norm(grad1) * torch.norm(grad2) + 1e-10
        )
        # Correlation should be high (both approximate the same gradient) but not exactly 1
        assert 0.5 < correlation < 1.0, (
            f"Unexpected gradient correlation: {correlation}"
        )


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

        source_locations = torch.tensor(
            [[[ny // 3, nx // 3], [2 * ny // 3, 2 * nx // 3]]],
            dtype=torch.long,
            device=device,
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2], [ny // 2, nx // 2 + 2]]],
            dtype=torch.long,
            device=device,
        )

        wavelet = tide.ricker(100e6, nt, 4e-11, dtype=dtype, device=device)
        # For multiple sources, use the same wavelet for each source
        source_amplitude = wavelet.view(1, 1, nt).expand(1, n_sources, nt)

        eps = epsilon.clone().detach().requires_grad_(True)
        out = tide.maxwell._kernel_api.maxwelltm(
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

        assert torch.isfinite(grad).all(), (
            "Gradient should be finite for multiple sources"
        )
        assert grad.abs().sum() > 0, "Gradient should be non-zero for multiple sources"


class TestGradientBackendConsistency:
    """Regression tests for eager vs native backend gradient consistency."""

    def test_eager_vs_native_source_and_model_gradients_cpu_no_pml_match_reference(
        self,
    ):
        try:
            from tide import backend_utils
        except Exception:  # pragma: no cover
            pytest.skip("backend_utils unavailable")

        if not backend_utils.is_backend_available():
            pytest.skip("native backend unavailable")

        device = torch.device("cpu")
        dtype = torch.float64
        ny, nx = 8, 9
        nt = 12

        epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype)
        epsilon[ny // 2 - 1 : ny // 2 + 1, nx // 2 - 1 : nx // 2 + 1] = 4.3
        sigma = torch.full((ny, nx), 5e-4, device=device, dtype=dtype)
        mu = torch.ones_like(epsilon)

        source_locations = torch.tensor(
            [[[ny // 2, nx // 3]]], dtype=torch.long, device=device
        )
        receiver_locations = torch.tensor(
            [[[ny // 2, nx // 2], [ny // 2, nx // 2 + 1]]],
            dtype=torch.long,
            device=device,
        )
        source_wavelet = tide.ricker(
            90e6,
            nt,
            4e-11,
            peak_time=1.0 / 90e6,
            dtype=dtype,
            device=device,
        ).view(1, 1, nt)

        def compute_grads(
            backend: bool | str,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            eps = epsilon.clone().detach().requires_grad_(True)
            sig = sigma.clone().detach().requires_grad_(True)
            src = source_wavelet.clone().detach().requires_grad_(True)
            rec = tide.maxwell._kernel_api.maxwelltm(
                eps,
                sig,
                mu,
                grid_spacing=0.02,
                dt=4e-11,
                source_amplitude=src,
                source_location=source_locations,
                receiver_location=receiver_locations,
                pml_width=0,
                stencil=2,
                python_backend=backend,
            )[-1]
            loss = 0.5 * rec.square().sum() + 0.01 * rec.sin().sum()
            loss.backward()
            assert eps.grad is not None
            assert sig.grad is not None
            assert src.grad is not None
            return (
                eps.grad.detach().clone(),
                sig.grad.detach().clone(),
                src.grad.detach().clone(),
            )

        g_eps_ref, g_sig_ref, g_src_ref = compute_grads("eager")
        g_eps_native, g_sig_native, g_src_native = compute_grads(False)

        cos_eps = cosine_similarity(g_eps_native, g_eps_ref)
        cos_sig = cosine_similarity(g_sig_native, g_sig_ref)
        cos_src = cosine_similarity(g_src_native, g_src_ref)
        assert cos_eps > 0.999, f"epsilon gradient cosine too low: {cos_eps:.6f}"
        assert cos_sig > 0.999, f"sigma gradient cosine too low: {cos_sig:.6f}"
        assert cos_src > 0.999, f"source gradient cosine too low: {cos_src:.6f}"

    @pytest.mark.parametrize("device_type", ["cpu", "cuda"])
    def test_eager_vs_native_source_only_gradient_without_snapshots(
        self, device_type: str
    ):
        try:
            from tide import backend_utils
        except Exception:  # pragma: no cover
            pytest.skip("backend_utils unavailable")

        if not backend_utils.is_backend_available():
            pytest.skip("native backend unavailable")
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")

        device = torch.device(device_type)
        dtype = torch.float64
        ny, nx = 8, 9
        nt = 12
        epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype)
        sigma = torch.full((ny, nx), 5e-4, device=device, dtype=dtype)
        mu = torch.ones_like(epsilon)
        source_location = torch.tensor(
            [[[ny // 2, nx // 3]]], dtype=torch.long, device=device
        )
        receiver_location = torch.tensor(
            [[[ny // 2, nx // 2], [ny // 2, nx // 2 + 1]]],
            dtype=torch.long,
            device=device,
        )
        source_wavelet = tide.ricker(
            90e6,
            nt,
            4e-11,
            peak_time=1.0 / 90e6,
            dtype=dtype,
            device=device,
        ).view(1, 1, nt)

        def source_gradient(backend: bool | str) -> torch.Tensor:
            source = source_wavelet.clone().detach().requires_grad_(True)
            receiver = tide.maxwell._kernel_api.maxwelltm(
                epsilon,
                sigma,
                mu,
                grid_spacing=0.02,
                dt=4e-11,
                source_amplitude=source,
                source_location=source_location,
                receiver_location=receiver_location,
                pml_width=0,
                stencil=2,
                python_backend=backend,
                storage_mode="none",
            )[-1]
            assert receiver.requires_grad
            weight = torch.linspace(
                0.5,
                1.5,
                receiver.numel(),
                device=device,
                dtype=dtype,
            ).reshape_as(receiver)
            (receiver * weight).sum().backward()
            assert source.grad is not None
            return source.grad.detach().clone()

        grad_reference = source_gradient("eager")
        grad_native = source_gradient(False)
        torch.testing.assert_close(
            grad_native,
            grad_reference,
            rtol=1e-10 if device_type == "cpu" else 1e-8,
            atol=1e-10 if device_type == "cpu" else 1e-8,
        )

    @pytest.mark.parametrize("device_type", ["cpu", "cuda"])
    def test_maxwell3d_eager_vs_native_source_only_gradient_without_snapshots(
        self, device_type: str
    ):
        try:
            from tide import backend_utils
        except Exception:  # pragma: no cover
            pytest.skip("backend_utils unavailable")

        if not backend_utils.is_backend_available():
            pytest.skip("native backend unavailable")
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")

        device = torch.device(device_type)
        dtype = torch.float32
        nz, ny, nx = 6, 6, 7
        nt = 10
        epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
        sigma = torch.full_like(epsilon, 1e-4)
        mu = torch.ones_like(epsilon)
        source_location = torch.tensor([[[2, 2, 2]]], dtype=torch.long, device=device)
        receiver_location = torch.tensor([[[2, 2, 4]]], dtype=torch.long, device=device)
        source_wavelet = tide.ricker(
            90e6,
            nt,
            4e-11,
            peak_time=1.0 / 90e6,
            dtype=dtype,
            device=device,
        ).view(1, 1, nt)

        def source_gradient(python_backend: bool) -> torch.Tensor:
            source = source_wavelet.clone().detach().requires_grad_(True)
            receiver = tide.maxwell._kernel_api.maxwell3d(
                epsilon,
                sigma,
                mu,
                grid_spacing=0.02,
                dt=4e-11,
                source_amplitude=source,
                source_location=source_location,
                receiver_location=receiver_location,
                pml_width=2,
                python_backend=python_backend,
                storage_mode="none",
            )[-1]
            assert receiver.requires_grad
            weight = torch.linspace(
                0.5,
                1.5,
                receiver.numel(),
                device=device,
                dtype=dtype,
            ).reshape_as(receiver)
            (receiver * weight).sum().backward()
            assert source.grad is not None
            return source.grad.detach().clone()

        grad_reference = source_gradient(True)
        grad_native = source_gradient(False)
        torch.testing.assert_close(
            grad_native,
            grad_reference,
            rtol=2e-4,
            atol=1e-3,
        )


def test_eager_vs_native_gradients_cuda_include_pml_foldback():
    try:
        from tide import backend_utils
    except Exception:  # pragma: no cover
        pytest.skip("backend_utils unavailable")

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not backend_utils.is_backend_available():
        pytest.skip("native backend unavailable")

    device = torch.device("cuda")
    dtype = torch.float32
    ny, nx = 18, 22
    nt = 28

    y = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    epsilon = (
        4.0 + 0.5 * torch.exp(-((xx - 0.15) ** 2 + (yy - 0.20) ** 2) / 0.02) + 0.3 * xx
    ).detach()
    sigma = (5e-4 + 8e-4 * yy).detach()
    mu = torch.ones_like(epsilon)

    source_locations = torch.tensor([[[1, 1]]], dtype=torch.long, device=device)
    receiver_locations = torch.tensor(
        [[[1, 3], [2, 2], [3, 4]]], dtype=torch.long, device=device
    )
    wavelet = tide.ricker(160e6, nt, 4e-11, dtype=dtype, device=device)
    source_amplitude = wavelet.view(1, 1, nt)

    def compute_grad(backend: bool | str) -> torch.Tensor:
        eps = epsilon.clone().detach().requires_grad_(True)
        rec = tide.maxwell._kernel_api.maxwelltm(
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
            python_backend=backend,
        )[-1]
        loss = 0.5 * rec.square().mean()
        loss.backward()
        assert eps.grad is not None
        return eps.grad.detach().clone()

    grad_ref = compute_grad("eager")
    grad_native = compute_grad(False)

    top_ref = grad_ref[0, :]
    top_native = grad_native[0, :]
    left_ref = grad_ref[:, 0]
    left_native = grad_native[:, 0]

    assert float(top_ref.abs().max()) > 1e-7
    assert float(left_ref.abs().max()) > 1e-7

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        value = (a.flatten() @ b.flatten()) / (torch.norm(a) * torch.norm(b) + 1e-12)
        return float(value)

    top_norm_ratio = float(torch.norm(top_native) / (torch.norm(top_ref) + 1e-12))
    left_norm_ratio = float(torch.norm(left_native) / (torch.norm(left_ref) + 1e-12))

    assert top_norm_ratio > 0.5, f"top-row foldback too small: {top_norm_ratio:.3f}"
    assert left_norm_ratio > 0.5, (
        f"left-column foldback too small: {left_norm_ratio:.3f}"
    )
    assert cosine(top_native, top_ref) > 0.75
    assert cosine(left_native, left_ref) > 0.95


@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("parameter", ["epsilon", "sigma"])
def test_tm2d_native_directional_derivative(parameter: str, stencil: int) -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    ny, nx, nt = 18, 20, 60
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype)
    sigma = torch.full_like(epsilon, 2.0e-4)
    mu = torch.ones_like(epsilon)
    source = tide.ricker(400e6, nt, 2.0e-11, peak_time=8.0e-10, dtype=dtype).view(
        1, 1, nt
    )
    source_location = torch.tensor([[[9, 7]]], dtype=torch.long)
    receiver_location = torch.tensor([[[9, 12], [11, 12]]], dtype=torch.long)
    residual = torch.linspace(-0.7, 1.1, nt, dtype=dtype).view(nt, 1, 1)

    def objective(value: torch.Tensor) -> torch.Tensor:
        epsilon_i = value if parameter == "epsilon" else epsilon
        sigma_i = value if parameter == "sigma" else sigma
        receiver = tide.maxwell._kernel_api.maxwelltm(
            epsilon_i,
            sigma_i,
            mu,
            [0.018, 0.022],
            2.0e-11,
            source,
            source_location,
            receiver_location,
            stencil=stencil,
            pml_width=4,
            python_backend=False,
            storage_compression=False,
        )[-1]
        return (receiver * residual).sum()

    base = (epsilon if parameter == "epsilon" else sigma).clone().requires_grad_(True)
    loss = objective(base)
    (gradient,) = torch.autograd.grad(loss, base)
    direction = deterministic_direction(
        base.shape, seed=7100 + stencil, device=device, dtype=dtype
    )
    scale = 1.0e-2 if parameter == "epsilon" else 1.0e-5
    steps = tuple(scale / 2**i for i in range(5))
    errors = directional_derivative_errors(
        objective,
        base.detach(),
        direction,
        gradient,
        steps,
    )
    zero_order, first_order = taylor_remainders(
        objective,
        base.detach(),
        direction,
        gradient,
        steps,
        base_value=loss,
    )
    assert min(errors) < 5.0e-3, errors
    zero_orders = convergence_orders(zero_order)
    first_orders = convergence_orders(first_order)
    assert all(0.9 <= order <= 1.1 for order in zero_orders[:2]), zero_orders
    assert all(order >= 1.7 for order in first_orders[:2]), first_orders
    assert first_order[-1] < zero_order[-1], (zero_order, first_order)


@pytest.mark.slow
@pytest.mark.numerical
@pytest.mark.parametrize("parameter", ["epsilon", "sigma"])
def test_tm2d_sampled_model_gradient_matches_central_difference(
    parameter: str,
) -> None:
    example = make_tm2d_example(
        shape=(6, 7),
        nt=18,
        grid_spacing=0.02,
        dt=4.0e-11,
        frequency=100e6,
        peak_time=4.0e-10,
        dtype=torch.float64,
        sigma=5.0e-4,
        source_location=(3, 2),
        receiver_locations=((3, 3), (3, 4)),
        pml_width=2,
        stencil=2,
        python_backend=True,
    )
    residual = torch.linspace(-0.5, 1.0, 18, dtype=torch.float64).view(18, 1, 1)

    def objective(value: torch.Tensor) -> torch.Tensor:
        receiver = example.run(
            epsilon=value if parameter == "epsilon" else example.epsilon,
            sigma=value if parameter == "sigma" else example.sigma,
        )[-1]
        return (receiver * residual).sum()

    base = (
        (example.epsilon if parameter == "epsilon" else example.sigma)
        .clone()
        .requires_grad_(True)
    )
    (gradient,) = torch.autograd.grad(objective(base), base)
    coordinates = ((0, 0), (0, 3), (1, 1), (3, 3), (4, 5), (5, 6))
    step = 1.0e-4 if parameter == "epsilon" else 1.0e-7
    finite_difference = []
    for coordinate in coordinates:
        perturbation = torch.zeros_like(base)
        perturbation[coordinate] = step
        finite_difference.append(
            float(
                (
                    objective(base.detach() + perturbation)
                    - objective(base.detach() - perturbation)
                )
                / (2.0 * step)
            )
        )
    actual = torch.stack([gradient[coordinate] for coordinate in coordinates])
    expected = torch.tensor(finite_difference, dtype=torch.float64)
    assert relative_l2(actual, expected) <= 2.0e-4
    assert cosine_similarity(actual, expected) >= 0.99999


# --- test_maxwell3d_gradients.py ---


def _maxwell3d_directional_metrics(
    parameter: str, stencil: int, *, python_backend: bool
) -> tuple[list[float], list[float], list[float]]:
    dtype = torch.float64
    example = make_maxwell3d_example(
        shape=(9, 10, 11),
        nt=45,
        grid_spacing=[0.016, 0.018, 0.022],
        dt=2.0e-11,
        frequency=500e6,
        peak_time=6.0e-10,
        dtype=dtype,
        sigma=2.0e-4,
        source_location=(4, 5, 4),
        receiver_locations=((4, 5, 7), (5, 7, 7)),
        pml_width=4,
        stencil=stencil,
        python_backend=python_backend,
    )
    residual = torch.linspace(-0.6, 1.0, 45, dtype=dtype).view(45, 1, 1)

    def objective(value: torch.Tensor) -> torch.Tensor:
        receiver = example.run(
            epsilon=value if parameter == "epsilon" else example.epsilon,
            sigma=value if parameter == "sigma" else example.sigma,
            storage_compression=False,
        )[-1]
        return (receiver * residual).sum()

    base = (
        (example.epsilon if parameter == "epsilon" else example.sigma)
        .clone()
        .requires_grad_(True)
    )
    loss = objective(base)
    (gradient,) = torch.autograd.grad(loss, base)
    direction = deterministic_direction(
        base.shape,
        seed=9100 + stencil,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    scale = 1.0e-2 if parameter == "epsilon" else 1.0e-5
    steps = tuple(scale / 2**i for i in range(5))
    errors = directional_derivative_errors(
        objective,
        base.detach(),
        direction,
        gradient,
        steps,
    )
    zero_order, first_order = taylor_remainders(
        objective,
        base.detach(),
        direction,
        gradient,
        steps,
        base_value=loss,
    )
    return errors, zero_order, first_order


@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("parameter", ["epsilon", "sigma"])
def test_maxwell3d_native_directional_derivative(parameter: str, stencil: int) -> None:
    errors, zero_order, first_order = _maxwell3d_directional_metrics(
        parameter, stencil, python_backend=False
    )
    assert min(errors) < 1.0e-3, errors
    assert first_order[-1] < zero_order[-1], (zero_order, first_order)


@pytest.mark.numerical
@pytest.mark.parametrize("stencil", [2, 4, 6, 8])
@pytest.mark.parametrize("parameter", ["epsilon", "sigma"])
def test_maxwell3d_reference_gradient_has_second_order_taylor_remainder(
    parameter: str, stencil: int
) -> None:
    errors, zero_order, first_order = _maxwell3d_directional_metrics(
        parameter, stencil, python_backend=True
    )
    assert min(errors) < 1.0e-5, errors
    zero_orders = convergence_orders(zero_order)
    first_orders = convergence_orders(first_order)
    assert all(0.9 <= order <= 1.1 for order in zero_orders[:2]), zero_orders
    assert all(order >= 1.8 for order in first_orders[:2]), first_orders
    assert first_order[-1] < zero_order[-1], (zero_order, first_order)


# --- test_tm2d_native_coeff_gradient.py ---


pytestmark = [pytest.mark.cuda, pytest.mark.numerical]


def _tm2d_examples(
    device: torch.device,
) -> tuple[MaxwellExample, MaxwellExample]:
    example = make_tm2d_example(
        shape=(24, 30),
        nt=64,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=180e6,
        device=device,
        sigma=1e-4,
        source_location=(6, 10),
        receiver_locations=((6, 15), (6, 20)),
        pml_width=4,
        stencil=2,
    )
    y = torch.arange(24, device=device, dtype=example.epsilon.dtype)[:, None]
    x = torch.arange(30, device=device, dtype=example.epsilon.dtype)[None, :]
    blob = torch.exp(-(((y - 14) / 4) ** 2 + ((x - 18) / 5) ** 2) * 0.5)
    truth = example.updated(
        epsilon=example.epsilon + 0.35 * blob,
        sigma=example.sigma + 2e-4 * blob,
    )
    return example, truth


def _gradient(
    example: MaxwellExample,
    truth: MaxwellExample,
    *,
    python_backend: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        observed = truth.run(
            python_backend=python_backend,
            storage_mode="none",
        )[-1]

    epsilon = example.epsilon.clone().requires_grad_(True)
    sigma = example.sigma.clone().requires_grad_(True)
    predicted = example.run(
        epsilon=epsilon,
        sigma=sigma,
        model_gradient_sampling_interval=1,
        save_snapshots=True,
        python_backend=python_backend,
        storage_mode="device",
    )[-1]
    (0.5 * (predicted - observed).square().sum()).backward()
    assert epsilon.grad is not None
    assert sigma.grad is not None
    return epsilon.grad.detach(), sigma.grad.detach()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av = a.reshape(-1).double()
    bv = b.reshape(-1).double()
    return (av @ bv) / (av.norm() * bv.norm())


def _relative_error(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    floor = torch.tensor(1e-30, device=lhs.device, dtype=lhs.dtype)
    denom = torch.maximum(torch.maximum(lhs.abs(), rhs.abs()), floor)
    return (lhs - rhs).abs() / denom


def test_tm2d_cuda_coeff_backward_default_matches_python_direction():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TM2D native coeff-gradient test.")

    example, truth = _tm2d_examples(torch.device("cuda"))
    reference_eps, reference_sig = _gradient(
        example,
        truth,
        python_backend=True,
    )
    native_eps, native_sig = _gradient(
        example,
        truth,
        python_backend=False,
    )

    assert _cosine(reference_eps, native_eps) > 0.98
    assert _cosine(reference_sig, native_sig) > 0.98


def test_tm2d_cuda_coeff_backward_default_dot_product_is_close_without_pml():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TM2D native coeff-gradient test.")

    torch.manual_seed(123)
    device = torch.device("cuda")
    dtype = torch.float64
    ny, nx, nt = 24, 30, 64
    dx = 0.02
    dt = 4e-11
    y = torch.arange(ny, device=device, dtype=dtype)[:, None]
    x = torch.arange(nx, device=device, dtype=dtype)[None, :]
    blob = torch.exp(-(((y - 14) / 4) ** 2 + ((x - 18) / 5) ** 2) * 0.5)
    epsilon = torch.ones(ny, nx, device=device, dtype=dtype) * 4.0 + 0.1 * blob
    sigma = torch.ones_like(epsilon) * 1e-4
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[ny // 4, nx // 3]]], device=device)
    receiver_location = torch.tensor(
        [[[ny // 4, nx // 2], [ny // 4, 2 * nx // 3]]], device=device
    )
    source_amplitude = tide.ricker(
        180e6,
        nt,
        dt,
        peak_time=1.0 / 180e6,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    depsilon = 0.05 * torch.randn_like(epsilon)
    dsigma = 1e-4 * torch.randn_like(sigma)
    data_weight = torch.randn(nt, 1, 2, device=device, dtype=dtype)

    born_data = tide.maxwell._kernel_api.borntm(
        epsilon,
        sigma,
        mu,
        dx,
        dt,
        source_amplitude,
        source_location,
        receiver_location,
        depsilon=depsilon,
        dsigma=dsigma,
        pml_width=0,
        stencil=2,
        linearize_source=True,
        python_backend=False,
        storage_mode="device",
    )[-1]
    lhs = torch.sum(born_data * data_weight)

    eps_req = epsilon.detach().clone().requires_grad_(True)
    sig_req = sigma.detach().clone().requires_grad_(True)
    predicted = tide.maxwell._kernel_api.maxwelltm(
        eps_req,
        sig_req,
        mu,
        dx,
        dt,
        source_amplitude,
        source_location,
        receiver_location,
        pml_width=0,
        stencil=2,
        model_gradient_sampling_interval=1,
        save_snapshots=True,
        python_backend=False,
        storage_mode="device",
    )[-1]
    grad_eps, grad_sig = torch.autograd.grad(
        torch.sum(predicted * data_weight),
        (eps_req, sig_req),
    )
    rhs = torch.sum(depsilon * grad_eps + dsigma * grad_sig)

    assert _relative_error(lhs, rhs) < 1e-2


def test_tm2d_cuda_coeff_backward_is_exact_at_high_side_pml():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TM2D native coeff-gradient test.")

    torch.manual_seed(41)
    device = torch.device("cuda")
    dtype = torch.float64
    ny, nx, nt = 18, 20, 180
    dx = 0.02
    dt = 2e-11
    epsilon = torch.full((ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor([[[ny - 2, nx - 2]]], device=device)
    receiver_location = torch.tensor(
        [[[ny - 2, nx - 1], [ny - 1, nx - 2], [ny - 3, nx - 3]]],
        device=device,
    )
    source_amplitude = tide.ricker(
        1.2e9,
        nt,
        dt,
        peak_time=7e-10,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt)
    depsilon = 0.08 * torch.randn_like(epsilon)
    dsigma = 5e-5 * torch.randn_like(sigma)
    data_weight = torch.randn(nt, 1, 3, device=device, dtype=dtype)
    common = {
        "grid_spacing": dx,
        "dt": dt,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "pml_width": [0, 6, 0, 6],
        "stencil": 2,
        "model_gradient_sampling_interval": 1,
        "save_snapshots": True,
    }

    def reference_forward(
        epsilon_arg: torch.Tensor, sigma_arg: torch.Tensor
    ) -> torch.Tensor:
        return tide.maxwell._kernel_api.maxwelltm(
            epsilon_arg,
            sigma_arg,
            mu,
            python_backend=True,
            storage_mode="none",
            **common,
        )[-1]

    _, jv = torch.autograd.functional.jvp(
        reference_forward,
        (epsilon, sigma),
        (depsilon, dsigma),
        strict=True,
    )
    lhs = torch.sum(jv * data_weight)

    epsilon_req = epsilon.clone().requires_grad_(True)
    sigma_req = sigma.clone().requires_grad_(True)
    predicted = tide.maxwell._kernel_api.maxwelltm(
        epsilon_req,
        sigma_req,
        mu,
        python_backend=False,
        storage_mode="device",
        **common,
    )[-1]
    grad_epsilon, grad_sigma = torch.autograd.grad(
        torch.sum(predicted * data_weight),
        (epsilon_req, sigma_req),
    )
    rhs = torch.sum(depsilon * grad_epsilon + dsigma * grad_sigma)

    assert _relative_error(lhs, rhs) < 1e-12
