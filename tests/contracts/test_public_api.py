from __future__ import annotations

import numpy as np
import pytest
import tide
import torch
from jaxtyping import TypeCheckError
from tide.validation import (
    validate_freq_taper_frac,
    validate_model_gradient_sampling_interval,
    validate_time_pad_frac,
)

# --- test_public_api.py ---

"""Intentional public API snapshot; additions require a documented review."""


EXPECTED_PUBLIC_NAMES = {
    "Acquisition",
    "BatchedModel2D",
    "BatchedModel3D",
    "BackendPreference",
    "CPML",
    "Callback",
    "CallbackState",
    "DebyeDispersion",
    "Field2DLike",
    "Field3DLike",
    "Discretization",
    "EM3DState",
    "EMDirection",
    "EMGradient",
    "EMModel",
    "ExecutionOptions",
    "Experiment",
    "FallbackPolicy",
    "ForwardResult",
    "Location2D",
    "Location3D",
    "MatrixF32",
    "Model2D",
    "Model2DLike",
    "Model3D",
    "Model3DLike",
    "ReceiverData",
    "ReceiverLocation2D",
    "ReceiverLocation3D",
    "SourceLocation2D",
    "SourceLocation3D",
    "LinearizedMaxwell3D",
    "LinearizedMaxwellTM",
    "Maxwell3D",
    "MaxwellTM",
    "Observers",
    "SourceConvention",
    "StorageMode",
    "StorageOptions",
    "TMState",
    "TangentResult",
    "VectorF32",
    "WaveletBatch",
    "callbacks",
    "cfl",
    "cfl_condition",
    "core",
    "create_or_pad",
    "downsample",
    "downsample_and_movedim",
    "gaussian",
    "gaussian_derivative",
    "maxwell",
    "morlet",
    "optim",
    "padding",
    "resampling",
    "reverse_pad",
    "ricker",
    "sine_burst",
    "runtime_typecheck",
    "staggered",
    "upsample",
    "utils",
    "validate_freq_taper_frac",
    "validate_model_gradient_sampling_interval",
    "validate_time_pad_frac",
    "validation",
    "wavelets",
    "workflow",
    "zero_interior",
}


def test_public_api_is_explicit() -> None:
    assert set(tide.__all__) == EXPECTED_PUBLIC_NAMES
    assert all(hasattr(tide, name) for name in tide.__all__)


# --- test_runtime_typecheck.py ---


def _tm_inputs() -> tuple[torch.Tensor, ...]:
    epsilon = torch.ones((6, 7), dtype=torch.float32)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_amplitude = torch.zeros((1, 1, 4), dtype=torch.float32)
    source_location = torch.tensor([[[3, 3]]], dtype=torch.long)
    receiver_location = torch.tensor([[[3, 4]]], dtype=torch.long)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def _em3d_inputs() -> tuple[torch.Tensor, ...]:
    epsilon = torch.ones((5, 6, 7), dtype=torch.float32)
    sigma = torch.zeros_like(epsilon)
    mu = torch.ones_like(epsilon)
    source_amplitude = torch.zeros((1, 1, 4), dtype=torch.float32)
    source_location = torch.tensor([[[2, 3, 3]]], dtype=torch.long)
    receiver_location = torch.tensor([[[2, 3, 4]]], dtype=torch.long)
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def test_numpy_shape_aliases_support_runtime_typechecking() -> None:
    @tide.runtime_typecheck
    def accept_vector(value: tide.VectorF32) -> tide.VectorF32:
        return value

    vector = np.zeros(3, dtype=np.float32)
    assert accept_vector(vector) is vector

    with pytest.raises(TypeCheckError):
        accept_vector(np.zeros(3, dtype=np.float64))


def test_maxwelltm_rejects_3d_coordinates() -> None:
    epsilon, sigma, mu, source_amplitude, _, receiver_location = _tm_inputs()
    bad_source_location = torch.tensor([[[2, 3, 4]]], dtype=torch.long)

    with pytest.raises(TypeCheckError):
        tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=bad_source_location,
            receiver_location=receiver_location,
            pml_width=1,
            python_backend=True,
        )


def test_maxwell3d_rejects_2d_coordinates() -> None:
    epsilon, sigma, mu, source_amplitude, source_location, _ = _em3d_inputs()
    bad_receiver_location = torch.tensor([[[3, 4]]], dtype=torch.long)

    with pytest.raises(TypeCheckError):
        tide.maxwell._kernel_api.maxwell3d(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=bad_receiver_location,
            pml_width=1,
            python_backend=True,
        )


def test_borntm_rejects_batched_perturbation() -> None:
    (
        epsilon,
        sigma,
        mu,
        source_amplitude,
        source_location,
        receiver_location,
    ) = _tm_inputs()
    bad_depsilon = torch.ones((1, *epsilon.shape), dtype=epsilon.dtype)

    with pytest.raises(TypeCheckError):
        tide.maxwell._kernel_api.borntm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            depsilon=bad_depsilon,
            pml_width=1,
            python_backend=True,
        )


def test_maxwelltm_operator_rejects_rank4_model() -> None:
    epsilon, _, _, source, source_location, receiver_location = _tm_inputs()
    bad_epsilon = torch.ones((1, 1, *epsilon.shape), dtype=epsilon.dtype)
    model = tide.EMModel(
        bad_epsilon,
        torch.zeros_like(bad_epsilon),
        torch.ones_like(bad_epsilon),
    )
    operator = tide.MaxwellTM(
        tide.Discretization(0.02, 1e-11, boundary=tide.CPML(1)),
        tide.Experiment(
            tide.Acquisition(source_location, receiver_location),
            source,
        ),
    )

    with pytest.raises(ValueError, match="2-D or batched 3-D"):
        operator(model)


# --- test_validation.py ---


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
