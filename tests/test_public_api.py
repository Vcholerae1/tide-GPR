"""Intentional public API snapshot; additions require a documented review."""

import tide


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
    "create_callback_state",
    "create_or_pad",
    "downsample",
    "downsample_and_movedim",
    "maxwell",
    "optim",
    "padding",
    "resampling",
    "reverse_pad",
    "ricker",
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


def test_removed_tuple_api_is_not_public() -> None:
    removed = {
        "Born3D",
        "BornTM",
        "born3d",
        "borntm",
        "linearize_maxwelltm",
        "maxwell3d",
        "maxwell3d_hvp",
        "maxwelltm",
        "maxwelltm_hvp",
    }
    assert removed.isdisjoint(tide.__all__)
    assert all(not hasattr(tide, name) for name in removed)
