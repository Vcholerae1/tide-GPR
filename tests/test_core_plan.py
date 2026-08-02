from dataclasses import replace

import pytest
import torch

from tide import backend_utils
from tide.core import (
    BackendPreference,
    ComputeMode,
    Dimension,
    FallbackPolicy,
    Operation,
    compile_simulation_plan,
    normalize_backend_request,
    select_backend,
)


def test_compile_plan_normalizes_legacy_options() -> None:
    epsilon = torch.ones(4, 5, dtype=torch.float64)
    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        sigma=torch.zeros_like(epsilon),
        mu=torch.ones_like(epsilon),
        python_backend="eager",
        storage_mode="DEVICE",
        compute_mode="native",
        storage_chunk_steps=3,
    )

    assert plan.dimension is Dimension.TM2D
    assert plan.backend is BackendPreference.PYTHON
    assert plan.compute_mode is ComputeMode.NATIVE
    assert plan.storage.mode.value == "device"
    assert plan.storage.chunk_steps == 3


def test_compile_plan_detects_batched_3d_models() -> None:
    epsilon = torch.ones(2, 3, 4, 5)
    plan = compile_simulation_plan(dimension="em3d", epsilon=epsilon)
    assert plan.model_batched
    assert plan.source_component == "ey"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, BackendPreference.AUTO),
        (True, BackendPreference.PYTHON),
        ("compile", BackendPreference.PYTHON),
        ("native", BackendPreference.NATIVE),
    ],
)
def test_normalize_backend_request(
    value: bool | str, expected: BackendPreference
) -> None:
    assert normalize_backend_request(value) is expected


def test_select_backend_has_explicit_fallback_policy() -> None:
    epsilon = torch.ones(4, 4)
    plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        fallback=FallbackPolicy.REFERENCE.value,
    )
    decision = select_backend(plan, native_available=False)
    assert decision.selected is BackendPreference.PYTHON
    assert decision.fallback

    strict_plan = compile_simulation_plan(
        dimension="tm2d",
        epsilon=epsilon,
        fallback=FallbackPolicy.ERROR.value,
    )
    with pytest.raises(RuntimeError, match="native backend"):
        select_backend(strict_plan, native_available=False)


def test_hvp_capability_rejection_comes_from_backend_decision() -> None:
    epsilon = torch.ones(4, 4)
    plan = compile_simulation_plan(
        operation="hvp",
        dimension="tm2d",
        epsilon=epsilon,
        python_backend=True,
        model_gradient_sampling_interval=2,
    )

    assert plan.operation is Operation.HVP
    with pytest.raises(NotImplementedError, match="Python TM2D HVP currently"):
        select_backend(plan, native_available=True)


def test_linearization_decision_owns_background_reuse_capability() -> None:
    cpu_plan = compile_simulation_plan(
        operation="linearization",
        dimension="tm2d",
        epsilon=torch.ones(4, 4),
        storage_mode="device",
    )
    plan = replace(cpu_plan, device=torch.device("cuda"))
    decision = select_backend(plan, native_available=True)

    assert decision.selected is BackendPreference.NATIVE
    assert decision.can_reuse_background(plan)


def test_compile_plan_rejects_python_fp16_io() -> None:
    with pytest.raises(NotImplementedError, match="fp16_io"):
        compile_simulation_plan(
            dimension="tm2d",
            epsilon=torch.ones(4, 4),
            python_backend=True,
            compute_mode="fp16_io",
        )


def test_backend_templates_are_validated_and_named() -> None:
    backend_utils._validate_template_specs()
    signature = backend_utils.backend_signature("maxwell_tm", "forward")

    assert len(signature) == 54
    assert signature[:3] == ("ca", "cb", "cq")
    assert len(signature) == len(set(signature))


def test_backend_signature_rejects_unknown_template() -> None:
    with pytest.raises(KeyError):
        backend_utils.backend_signature("unknown", "forward")
