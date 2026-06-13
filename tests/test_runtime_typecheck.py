import pytest
import torch
from jaxtyping import TypeCheckError

import tide


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


def test_shape_aliases_are_public() -> None:
    for name in (
        "Model2D",
        "Model2DLike",
        "Model3D",
        "Model3DLike",
        "Location2D",
        "Location3D",
        "ReceiverLocation2D",
        "ReceiverLocation3D",
        "SourceLocation2D",
        "SourceLocation3D",
        "WaveletBatch",
        "runtime_typecheck",
    ):
        assert name in tide.__all__
        assert hasattr(tide, name)


def test_maxwelltm_rejects_3d_coordinates() -> None:
    epsilon, sigma, mu, source_amplitude, _, receiver_location = _tm_inputs()
    bad_source_location = torch.tensor([[[2, 3, 4]]], dtype=torch.long)

    with pytest.raises(TypeCheckError):
        tide.maxwelltm(
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
        tide.maxwell3d(
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
        tide.borntm(
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


def test_maxwelltm_constructor_rejects_rank4_model() -> None:
    epsilon, sigma, mu, *_ = _tm_inputs()
    bad_epsilon = torch.ones((1, 1, *epsilon.shape), dtype=epsilon.dtype)

    with pytest.raises(TypeCheckError):
        tide.MaxwellTM(
            bad_epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
        )
