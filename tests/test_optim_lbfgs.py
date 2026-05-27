import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tide
from tide import backend_utils


def _optim_backend_available() -> bool:
    if not backend_utils.is_backend_available():
        return False
    try:
        dll = backend_utils.get_dll()
    except RuntimeError:
        return False
    return hasattr(dll, "tide_optim_create")


pytestmark = pytest.mark.skipif(
    not _optim_backend_available(), reason="native optim backend not available"
)


def _load_optim_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "optim_benchmark.py"
    spec = importlib.util.spec_from_file_location("tide_optim_benchmark", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_sotb_prototype_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "optim_sotb_python_prototype.py"
    )
    spec = importlib.util.spec_from_file_location("tide_optim_sotb_prototype", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cross_fwi_benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "optim_cross_fwi_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("tide_optim_cross_fwi", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_optim_public_api_exports_supported_policy_lists():
    expected_exports = {
        "AlphaGuessPolicy",
        "BoundsDiagnostics",
        "BoundsStrategy",
        "DirectionPolicy",
        "DirectionStatus",
        "DirectionDiagnostics",
        "GlobalizationPolicy",
        "InnerCgStatus",
        "InnerSolveDiagnostics",
        "LBFGS",
        "LbfgsUpdatePolicy",
        "LineSearchDiagnostics",
        "LineSearchAcceptance",
        "LineSearchPolicy",
        "LineSearchStatus",
        "NlcgBetaPolicy",
        "OptimStatus",
        "OptimizeResult",
        "OptimizerOptions",
        "OptimizerEvaluation",
        "OptimizerEvaluationStatus",
        "OptimizerRequest",
        "OptimizerRequestRequirements",
        "OptimizerSession",
        "OptionsValidation",
        "PairUpdateDiagnostics",
        "OptionsValidationCode",
        "PairStatus",
        "PreconditionerStatus",
        "RequestKind",
        "ResolvedPolicies",
        "StoppingDiagnostics",
        "SUPPORTED_COST_MODELS",
        "SUPPORTED_ALPHA_GUESSES",
        "SUPPORTED_DIRECTIONS",
        "SUPPORTED_GLOBALIZATIONS",
        "SUPPORTED_POLICIES",
        "StoppingPolicy",
        "TrustRegionDiagnostics",
        "TrustRegionStatus",
        "SUPPORTED_BOUNDS_STRATEGIES",
        "SUPPORTED_LBFGS_UPDATES",
        "SUPPORTED_LINE_SEARCHES",
        "SUPPORTED_METHODS",
        "SUPPORTED_NLCG_BETAS",
        "SUPPORTED_STOPPING_POLICIES",
        "SUPPORTED_TRACE_POLICIES",
        "TraceEntry",
        "TraceSummary",
        "WarningFlag",
        "get_include",
        "minimize",
        "resolve_policies",
        "validate_options",
    }

    assert expected_exports <= set(tide.optim.__all__)
    include_dir = Path(tide.optim.get_include())
    assert (include_dir / "optim" / "optim.h").exists()
    assert tide.optim.SUPPORTED_METHODS == (
        "lbfgs",
        "plbfgs",
        "pstd",
        "steepest_descent",
        "pnlcg",
        "trn",
        "ptrn",
    )
    assert tide.optim.SUPPORTED_DIRECTIONS == tide.optim.SUPPORTED_METHODS
    assert tide.optim.SUPPORTED_GLOBALIZATIONS == (
        "line_search",
        "trust_region",
    )
    assert tide.optim.SUPPORTED_POLICIES["method"] == tide.optim.SUPPORTED_METHODS
    assert tide.optim.SUPPORTED_POLICIES["direction"] == (
        tide.optim.SUPPORTED_DIRECTIONS
    )
    assert tide.optim.SUPPORTED_POLICIES["globalization"] == (
        tide.optim.SUPPORTED_GLOBALIZATIONS
    )
    assert tide.optim.SUPPORTED_POLICIES["line_search"] == (
        tide.optim.SUPPORTED_LINE_SEARCHES
    )
    assert "hager_zhang" in tide.optim.SUPPORTED_LINE_SEARCHES
    assert "more_thuente" in tide.optim.SUPPORTED_LINE_SEARCHES
    assert tide.optim.LineSearchPolicy.MORE_THUENTE.name == "MORE_THUENTE"
    assert tide.optim.SUPPORTED_ALPHA_GUESSES == (
        "initial",
        "previous",
        "barzilai_borwein",
    )
    assert tide.optim.SUPPORTED_COST_MODELS == (
        "balanced",
        "expensive_gradient",
        "joint_value_gradient",
    )
    assert tide.optim.SUPPORTED_STOPPING_POLICIES == (
        "standard",
        "gradient_only",
        "initial_relative_f",
    )
    assert tide.optim.SUPPORTED_NLCG_BETAS == (
        "dai_yuan",
        "fletcher_reeves",
        "polak_ribiere_plus",
        "hager_zhang",
    )
    assert tide.optim.SUPPORTED_LBFGS_UPDATES == (
        "skip",
        "regularize",
    )
    assert tide.optim.SUPPORTED_TRACE_POLICIES == (
        "all",
        "none",
        "last",
        "stride",
    )
    assert "legacy_weak_wolfe" in tide.optim.SUPPORTED_LINE_SEARCHES
    assert "static" in tide.optim.SUPPORTED_LINE_SEARCHES
    assert "projected_gradient" in tide.optim.SUPPORTED_BOUNDS_STRATEGIES
    assert tide.optim.OptionsValidationCode.TRACE_POLICY.value == 32
    assert tide.optim.OptionsValidationCode.TRACE_STRIDE.value == 33
    assert tide.optim.WarningFlag.LBFGS_PAIR_SKIPPED.value == 1 << 3
    assert (
        tide.optim.WarningFlag.NONFINITE_TRIAL
        | tide.optim.WarningFlag.LBFGS_PAIR_SKIPPED
    ) == tide.optim.WarningFlag(9)
    assert "compatibility" not in tide.optim.__doc__
    assert tide.optim.LBFGS is tide.optim.OptimizerOptions
    assert isinstance(tide.optim.LBFGS(), tide.optim.OptimizerOptions)


def test_native_optim_backend_exposes_generic_c_api_names():
    dll = backend_utils.get_dll()
    for name in (
        "tide_optim_create",
        "tide_optim_destroy",
        "tide_optim_start",
        "tide_optim_tell",
        "tide_optim_tell_value",
        "tide_optim_tell_preconditioner",
        "tide_optim_tell_hessian_vector",
        "tide_optim_current_x",
        "tide_optim_validate_options",
        "tide_optim_resolve_policies",
        "tide_optim_summarize_report",
        "tide_optim_summarize_request",
        "tide_optim_summarize_report_request",
        "tide_optim_validate_evaluation",
        "tide_optim_validate_report_evaluation",
        "tide_optim_get_session_snapshot",
        "tide_optim_request_kind_name",
        "tide_optim_request_expected_evaluation",
        "tide_optim_request_required_fields",
        "tide_optim_request_accepted_mapping_keys",
        "tide_optim_request_requires_evaluation",
        "tide_optim_request_is_error",
        "tide_optim_request_is_done",
        "tide_optim_request_needs_value",
        "tide_optim_request_needs_gradient",
        "tide_optim_request_needs_value_gradient",
        "tide_optim_request_needs_preconditioner",
        "tide_optim_request_needs_hessian_vector",
        "tide_optim_request_needs_vector_result",
        "tide_optim_status_name",
        "tide_optim_line_search_status_name",
        "tide_optim_line_search_policy_name",
        "tide_optim_alpha_guess_policy_name",
        "tide_optim_stopping_policy_name",
        "tide_optim_line_search_acceptance_name",
        "tide_optim_pair_status_name",
        "tide_optim_lbfgs_update_policy_name",
        "tide_optim_bounds_strategy_name",
        "tide_optim_cost_model_name",
        "tide_optim_direction_policy_name",
        "tide_optim_direction_method_name",
        "tide_optim_nlcg_beta_policy_name",
        "tide_optim_direction_status_name",
        "tide_optim_preconditioner_status_name",
        "tide_optim_inner_cg_status_name",
        "tide_optim_globalization_policy_name",
        "tide_optim_trust_region_status_name",
        "tide_optim_warning_flag_name",
        "tide_optim_options_validation_code_name",
    ):
        assert hasattr(dll, name)


def test_native_optim_backend_exposes_c_abi_semantic_helpers():
    from tide.optim import _backend as optim_backend

    assert optim_backend.request_kind_name(optim_backend.RequestKind.EVALUATE_FG) == (
        "EVALUATE_FG"
    )
    assert (
        optim_backend.request_expected_evaluation(
            optim_backend.RequestKind.EVALUATE_FG
        )
        == "value_gradient"
    )
    assert optim_backend.request_required_fields(
        optim_backend.RequestKind.EVALUATE_FG
    ) == ("f", "g")
    assert optim_backend.request_accepted_mapping_keys(
        optim_backend.RequestKind.EVALUATE_FG
    ) == ("f", "g", "gradient")
    assert optim_backend.request_requires_evaluation(
        optim_backend.RequestKind.EVALUATE_FG
    )
    assert not optim_backend.request_requires_evaluation(
        optim_backend.RequestKind.DONE
    )
    assert optim_backend.request_expected_evaluation(12345) == "none"
    assert optim_backend.request_required_fields(12345) == ()
    assert optim_backend.request_accepted_mapping_keys(12345) == ()
    assert optim_backend.request_kind_name(12345) == "UNKNOWN"
    assert optim_backend.request_needs_value(optim_backend.RequestKind.EVALUATE_FG)
    assert optim_backend.request_needs_value(optim_backend.RequestKind.EVALUATE_F)
    assert optim_backend.request_needs_gradient(optim_backend.RequestKind.EVALUATE_FG)
    assert optim_backend.request_needs_value_gradient(
        optim_backend.RequestKind.EVALUATE_FG
    )
    assert not optim_backend.request_needs_gradient(
        optim_backend.RequestKind.EVALUATE_F
    )
    assert optim_backend.request_needs_preconditioner(
        optim_backend.RequestKind.APPLY_PRECONDITIONER
    )
    assert optim_backend.request_needs_hessian_vector(
        optim_backend.RequestKind.EVALUATE_HV
    )
    assert optim_backend.request_needs_vector_result(
        optim_backend.RequestKind.APPLY_PRECONDITIONER
    )
    assert optim_backend.request_needs_vector_result(
        optim_backend.RequestKind.EVALUATE_HV
    )
    assert not optim_backend.request_needs_vector_result(
        optim_backend.RequestKind.EVALUATE_FG
    )
    assert optim_backend.request_is_done(optim_backend.RequestKind.DONE)
    assert optim_backend.request_is_error(optim_backend.RequestKind.ERROR)

    short_status = optim_backend.validate_evaluation(
        optim_backend.RequestKind.EVALUATE_FG,
        request_sequence=7,
        expected_gradient_size=2,
        has_value=True,
        has_gradient=True,
        gradient_size=1,
    )
    assert isinstance(short_status, optim_backend.OptimizerBackendEvaluationStatus)
    assert short_status.request == optim_backend.RequestKind.EVALUATE_FG
    assert short_status.request_sequence == 7
    assert short_status.request_name == "EVALUATE_FG"
    assert short_status.expected_evaluation == "value_gradient"
    assert short_status.required_fields == ("f", "g")
    assert short_status.accepted_mapping_keys == ("f", "g", "gradient")
    assert short_status.requires_evaluation
    assert short_status.has_value
    assert short_status.has_gradient
    assert not short_status.has_vector
    assert short_status.gradient_size == 1
    assert short_status.vector_size == 0
    assert short_status.expected_gradient_size == 2
    assert short_status.expected_vector_size == 0
    assert short_status.missing_fields == ()
    assert short_status.mismatched_fields == ("g",)
    assert not short_status.has_missing_fields
    assert short_status.gradient_size_mismatch
    assert short_status.has_size_mismatch
    assert not short_status.satisfied
    assert not short_status.valid
    short_status_payload = short_status.to_dict()
    assert short_status_payload["request"] == "EVALUATE_FG"
    assert short_status_payload["required_fields"] == ["f", "g"]
    assert short_status_payload["mismatched_fields"] == ["g"]

    ok_status = optim_backend.validate_evaluation(
        optim_backend.RequestKind.EVALUATE_FG,
        request_sequence=8,
        expected_gradient_size=2,
        has_value=True,
        has_gradient=True,
        gradient_size=2,
    )
    assert ok_status.satisfied
    assert ok_status.valid
    assert ok_status.missing_fields == ()
    assert ok_status.mismatched_fields == ()

    missing_vector_status = optim_backend.validate_evaluation(
        optim_backend.RequestKind.EVALUATE_HV,
        request_sequence=9,
        expected_vector_size=2,
    )
    assert missing_vector_status.expected_evaluation == "hessian_vector"
    assert missing_vector_status.required_fields == ("vector",)
    assert missing_vector_status.missing_fields == ("vector",)
    assert missing_vector_status.missing_vector
    assert missing_vector_status.has_missing_fields
    assert not missing_vector_status.satisfied

    done_status = optim_backend.validate_evaluation(
        optim_backend.RequestKind.DONE,
        has_value=True,
        has_gradient=True,
        gradient_size=2,
    )
    assert not done_status.requires_evaluation
    assert done_status.expected_evaluation == "none"
    assert done_status.missing_fields == ()
    assert done_status.mismatched_fields == ()
    assert not done_status.satisfied
    assert not done_status.valid

    request_summary = optim_backend.summarize_request(
        optim_backend.RequestKind.EVALUATE_HV
    )
    assert request_summary.valid
    assert request_summary.request == optim_backend.RequestKind.EVALUATE_HV
    assert request_summary.request_name == "EVALUATE_HV"
    assert request_summary.request_sequence == 0
    assert request_summary.requires_evaluation
    assert request_summary.expected_evaluation == "hessian_vector"
    assert request_summary.required_fields == ("vector",)
    assert request_summary.accepted_mapping_keys == ("vector", "hv")
    assert request_summary.needs_hessian_vector
    assert request_summary.needs_vector_result
    assert not request_summary.needs_value
    assert request_summary.to_dict()["request"] == "EVALUATE_HV"
    assert request_summary.to_dict()["expected_evaluation"] == "hessian_vector"
    assert request_summary.to_dict()["required_fields"] == ["vector"]
    assert request_summary.to_dict()["accepted_mapping_keys"] == [
        "vector",
        "hv",
    ]

    unknown_request = optim_backend.summarize_request(12345)
    assert not unknown_request.valid
    assert unknown_request.request == 12345
    assert unknown_request.request_name == "UNKNOWN"
    assert unknown_request.expected_evaluation == "none"
    assert unknown_request.required_fields == ()
    assert unknown_request.accepted_mapping_keys == ()
    assert not unknown_request.requires_evaluation
    assert unknown_request.to_dict()["request"] == 12345

    assert optim_backend.status_name(optim_backend.OptimStatus.MAX_EVAL) == "MAX_EVAL"
    assert (
        optim_backend.line_search_policy_name(
            optim_backend.LineSearchPolicy.HAGER_ZHANG
        )
        == "HAGER_ZHANG"
    )
    assert (
        optim_backend.direction_policy_name(
            optim_backend.DirectionPolicy.PRECONDITIONED_TRUNCATED_NEWTON
        )
        == "PRECONDITIONED_TRUNCATED_NEWTON"
    )
    assert (
        optim_backend.direction_method_name(
            optim_backend.DirectionPolicy.PRECONDITIONED_TRUNCATED_NEWTON
        )
        == "ptrn"
    )
    assert (
        optim_backend.options_validation_code_name(
            optim_backend.OptionsValidationCode.TRACE_STRIDE
        )
        == "TRACE_STRIDE"
    )
    assert (
        optim_backend.cost_model_name(optim_backend.CostModel.EXPENSIVE_GRADIENT)
        == "EXPENSIVE_GRADIENT"
    )

    resolved = optim_backend.resolve_backend_policies(
        optim_backend.OptimizerBackendOptions(
            n=3,
            direction_policy=optim_backend.DirectionPolicy.NLCG,
            line_search_policy=optim_backend.LineSearchPolicy.HAGER_ZHANG,
            nlcg_beta_policy=optim_backend.NlcgBetaPolicy.HAGER_ZHANG,
            bounds_strategy=optim_backend.BoundsStrategy.PROJECTED_GRADIENT,
        )
    )
    assert resolved.valid
    assert resolved.validation.ok
    assert resolved.method_name == "pnlcg"
    assert resolved.direction_policy == optim_backend.DirectionPolicy.NLCG
    assert resolved.line_search_policy == optim_backend.LineSearchPolicy.HAGER_ZHANG
    assert resolved.nlcg_beta_policy == optim_backend.NlcgBetaPolicy.HAGER_ZHANG
    assert resolved.bounds_strategy == optim_backend.BoundsStrategy.PROJECTED_GRADIENT
    assert resolved.cost_model == optim_backend.CostModel.BALANCED
    assert resolved.to_dict()["line_search_policy"] == "HAGER_ZHANG"
    assert resolved.to_dict()["cost_model"] == "BALANCED"

    null_resolved = optim_backend.resolve_backend_policies(None)
    assert not null_resolved.valid
    assert null_resolved.validation.code == (
        optim_backend.OptionsValidationCode.NULL_OPTIONS
    )
    assert null_resolved.validation.field == "options"


def test_native_optim_backend_summarizes_reports_through_c_abi():
    from tide.optim import _backend as optim_backend

    null_summary = optim_backend.summarize_report(None)
    assert null_summary.valid is False
    assert null_summary.request == optim_backend.RequestKind.ERROR
    assert null_summary.status == optim_backend.OptimStatus.INVALID_ARGUMENT
    assert null_summary.status_name == "INVALID_ARGUMENT"
    assert null_summary.failure_reason == "INVALID_ARGUMENT"
    assert null_summary.failed

    null_request = optim_backend.summarize_report_request(None)
    assert null_request.valid is False
    assert null_request.request == optim_backend.RequestKind.ERROR
    assert null_request.request_name == "ERROR"
    assert null_request.request_sequence == 0
    null_eval_status = optim_backend.validate_report_evaluation(None)
    assert null_eval_status.request == optim_backend.RequestKind.ERROR
    assert not null_eval_status.requires_evaluation
    assert not null_eval_status.satisfied

    x0 = np.array([1.0], dtype=np.float64)
    g0 = np.array([1.0], dtype=np.float64)
    x_request = np.empty_like(x0)
    options = optim_backend.OptimizerBackendOptions(n=1, gtol_abs=1e-12)
    with optim_backend.OptimizerBackend(options) as backend:
        report = backend.start(x0, 0.5, g0, x_request)
        summary = optim_backend.summarize_report(report)

        assert summary.valid
        assert summary.request == report.request
        assert summary.status == report.status
        assert summary.request_name == "EVALUATE_FG"
        assert summary.n == x0.size
        assert summary.expected_gradient_size == x0.size
        assert summary.expected_vector_size == 0
        assert summary.to_dict()["n"] == x0.size
        assert summary.to_dict()["expected_gradient_size"] == x0.size
        assert summary.expected_evaluation == "value_gradient"
        assert summary.required_fields == ("f", "g")
        assert summary.accepted_mapping_keys == ("f", "g", "gradient")
        assert summary.requires_evaluation
        assert summary.status_name == "RUNNING"
        assert summary.failure_reason is None
        assert summary.method_name == "lbfgs"
        assert summary.direction_policy_name == "LBFGS"
        assert summary.line_search_policy_name == "LEGACY_WEAK_WOLFE"
        assert summary.needs_value
        assert summary.needs_gradient
        assert summary.needs_value_gradient
        assert summary.request_sequence == report.request_sequence
        assert summary.request_sequence == 1
        assert not summary.needs_vector_result

        report_eval_status = optim_backend.validate_report_evaluation(
            report,
            has_value=True,
            has_gradient=True,
            gradient_size=x0.size,
        )
        assert report_eval_status.request == report.request
        assert report_eval_status.request_sequence == report.request_sequence
        assert report_eval_status.expected_gradient_size == x0.size
        assert report_eval_status.expected_vector_size == 0
        assert report_eval_status.satisfied
        short_report_eval_status = optim_backend.validate_report_evaluation(
            report,
            has_value=True,
            has_gradient=True,
            gradient_size=0,
        )
        assert short_report_eval_status.mismatched_fields == ("g",)
        assert not short_report_eval_status.satisfied

        report_request = optim_backend.summarize_report_request(report)
        assert report_request.valid
        assert report_request.request == report.request
        assert report_request.request_sequence == report.request_sequence
        assert report_request.expected_evaluation == "value_gradient"
        assert report_request.required_fields == ("f", "g")
        assert report_request.accepted_mapping_keys == ("f", "g", "gradient")
        assert report_request.requires_evaluation
        assert report_request.needs_value_gradient
        assert not report_request.needs_vector_result

        assert not summary.done
        assert not summary.success
        assert not summary.stopped
        assert not summary.failed
        assert summary.n_f == report.n_f
        assert summary.n_g == report.n_g
        assert summary.f == report.f
        assert summary.grad_norm == report.grad_norm

        done_report = backend.tell(0.0, np.array([0.0], dtype=np.float64), x_request)
        done_summary = optim_backend.summarize_report(done_report)

    assert done_summary.valid
    assert done_summary.done
    assert done_summary.success
    assert done_summary.stopped
    assert not done_summary.failed
    assert done_summary.status == optim_backend.OptimStatus.CONVERGED_GRADIENT
    assert done_summary.reason == "CONVERGED_GRADIENT"
    assert done_summary.failure_reason is None
    assert done_summary.request_name == "DONE"
    assert done_summary.request_sequence == done_report.request_sequence
    assert done_summary.request_sequence > summary.request_sequence
    assert done_summary.n_f == done_report.n_f
    assert done_summary.n_g == done_report.n_g
    assert done_summary.f == done_report.f


def test_native_optim_backend_exposes_session_snapshots_through_c_abi():
    from tide.optim import _backend as optim_backend

    invalid = optim_backend.get_session_snapshot(None)
    assert invalid.valid is False
    assert invalid.state_name == "INVALID"
    assert invalid.report.status == optim_backend.OptimStatus.INVALID_ARGUMENT
    assert invalid.report.failed

    x0 = np.array([1.0], dtype=np.float64)
    g0 = np.array([1.0], dtype=np.float64)
    x_request = np.empty_like(x0)
    options = optim_backend.OptimizerBackendOptions(n=1, gtol_abs=1e-12)
    with optim_backend.OptimizerBackend(options) as backend:
        initial = backend.snapshot()
        assert initial.valid
        assert initial.state_name == "NOT_STARTED"
        assert initial.n == 1
        assert not initial.started
        assert not initial.running
        assert not initial.done
        assert initial.report.request == optim_backend.RequestKind.ERROR
        assert initial.report.request_sequence == 0
        assert initial.report.method_name == "lbfgs"

        report = backend.start(x0, 0.5, g0, x_request)
        running = backend.snapshot()
        assert running.valid
        assert running.started
        assert running.running
        assert not running.done
        assert running.state_name == "RUNNING"
        assert running.report.request == report.request
        assert running.report.request_sequence == report.request_sequence
        assert running.report.request_sequence == 1
        assert running.report.status == optim_backend.OptimStatus.RUNNING
        assert running.report.needs_value
        assert running.report.needs_gradient
        assert running.awaiting_value_gradient
        assert not running.awaiting_preconditioner
        assert not running.awaiting_hessian_vector
        assert running.report.n_f == report.n_f
        assert running.report.n_g == report.n_g

        done_report = backend.tell(0.0, np.array([0.0], dtype=np.float64), x_request)
        done = backend.snapshot()
        assert done.valid
        assert done.started
        assert done.done
        assert not done.running
        assert done.state_name == "DONE"
        assert done.report.request == optim_backend.RequestKind.DONE
        assert done.report.request_sequence == done_report.request_sequence
        assert done.report.request_sequence > running.report.request_sequence
        assert done.report.status == optim_backend.OptimStatus.CONVERGED_GRADIENT
        assert done.report.success
        assert done.report.n_f == done_report.n_f
        assert done.report.n_g == done_report.n_g

    closed = backend.snapshot()
    assert closed.valid is False
    assert closed.state_name == "INVALID"


def test_native_optim_backend_validates_options_with_structured_reason():
    from tide.optim import _backend as optim_backend

    ok = optim_backend.validate_backend_options(
        optim_backend.OptimizerBackendOptions(n=1)
    )
    bad = optim_backend.validate_backend_options(
        optim_backend.OptimizerBackendOptions(n=1, bound_margin=-1.0)
    )

    assert ok.ok
    assert ok.code == optim_backend.OptionsValidationCode.OK
    assert ok.field == ""
    assert bad.ok is False
    assert bad.code == optim_backend.OptionsValidationCode.BOUND_MARGIN
    assert bad.field == "bound_margin"
    assert "bound_margin" in bad.message
    with pytest.raises(RuntimeError, match="bound_margin"):
        optim_backend.OptimizerBackend(
            optim_backend.OptimizerBackendOptions(n=1, bound_margin=-1.0)
        )


def test_public_optim_validate_options_reports_structured_resolution():
    ok = tide.optim.validate_options(
        method="lbfgs",
        options={"cost_model": "expensive_gradient"},
        n=4,
    )

    assert isinstance(ok, tide.optim.OptionsValidation)
    assert ok.ok
    assert ok.code == tide.optim.OptionsValidationCode.OK
    assert ok.field == ""
    assert ok.effective_options["cost_model"] == "expensive_gradient"
    assert ok.effective_options["line_search"] == "armijo_cubic"
    assert ok.backend_options["line_search_policy"] == "ARMIJO_CUBIC"
    assert ok.backend_options["n"] == 4
    assert ok.config_fingerprint is not None
    assert len(ok.config_fingerprint) == 64
    assert ok.config_signature["schema"] == "tide.optim.config.v1"
    assert ok.config_signature["backend_options"]["n"] == 4
    assert isinstance(ok.policy_resolution, tide.optim.ResolvedPolicies)
    assert ok.policy_resolution.line_search == tide.optim.LineSearchPolicy.ARMIJO_CUBIC
    assert ok.policy_resolution.direction == tide.optim.DirectionPolicy.LBFGS
    assert ok.to_dict()["ok"] is True
    assert ok.to_dict()["policy_resolution"]["line_search_policy"] == "ARMIJO_CUBIC"
    assert ok.to_dict()["config_fingerprint"] == ok.config_fingerprint
    repeat = tide.optim.validate_options(
        method="lbfgs",
        options={"cost_model": "expensive_gradient"},
        n=4,
    )
    assert repeat.config_fingerprint == ok.config_fingerprint
    changed = tide.optim.validate_options(
        method="lbfgs",
        options={"cost_model": "expensive_gradient", "line_search": "hager_zhang"},
        n=4,
    )
    assert changed.config_fingerprint != ok.config_fingerprint

    bad_bound = tide.optim.validate_options(options={"bound_margin": -1.0})
    assert bad_bound.ok is False
    assert bad_bound.code == tide.optim.OptionsValidationCode.BOUND_MARGIN
    assert bad_bound.field == "bound_margin"
    assert "bound_margin" in bad_bound.message

    bad_policy = tide.optim.validate_options(options={"line_search": "mystery"})
    assert bad_policy.ok is False
    assert bad_policy.code == tide.optim.OptionsValidationCode.LINE_SEARCH_POLICY
    assert bad_policy.field == "line_search"
    assert "line_search" in bad_policy.message

    bad_trace_policy = tide.optim.validate_options(
        options={"trace_policy": "log-to-random-file"}
    )
    assert bad_trace_policy.ok is False
    assert bad_trace_policy.code == tide.optim.OptionsValidationCode.TRACE_POLICY
    assert bad_trace_policy.field == "trace_policy"
    assert "trace_policy" in bad_trace_policy.message

    bad_trace_stride = tide.optim.validate_options(options={"trace_stride": 0})
    assert bad_trace_stride.ok is False
    assert bad_trace_stride.code == tide.optim.OptionsValidationCode.TRACE_STRIDE
    assert bad_trace_stride.field == "trace_stride"
    assert "trace_stride" in bad_trace_stride.message

    bad_method = tide.optim.validate_options(method="not_a_method")
    assert bad_method.ok is False
    assert bad_method.code == tide.optim.OptionsValidationCode.DIRECTION_POLICY
    assert bad_method.field == "method"

    pnlcg_options = tide.optim.OptimizerOptions.for_method("pnlcg")
    assert pnlcg_options.validate(method="pnlcg", n=3).ok
    assert pnlcg_options.validate(method="pnlcg", n=3).backend_options["n"] == 3


def test_public_optim_resolve_policies_reports_defaults_without_running():
    lbfgs = tide.optim.resolve_policies(method="lbfgs")
    assert isinstance(lbfgs, tide.optim.ResolvedPolicies)
    assert lbfgs.method == "lbfgs"
    assert lbfgs.cost_model == "balanced"
    assert lbfgs.direction == tide.optim.DirectionPolicy.LBFGS
    assert lbfgs.line_search == tide.optim.LineSearchPolicy.HAGER_ZHANG
    assert lbfgs.to_dict()["line_search_policy"] == "HAGER_ZHANG"

    expensive_gradient = tide.optim.resolve_policies(
        method="lbfgs",
        options={"cost_model": "expensive_gradient"},
        n=8,
    )
    assert expensive_gradient.line_search == tide.optim.LineSearchPolicy.ARMIJO_CUBIC
    assert expensive_gradient.cost_model == "expensive_gradient"

    pnlcg_options = tide.optim.OptimizerOptions.for_method(
        "pnlcg",
        nlcg_beta="hager_zhang",
    )
    pnlcg = pnlcg_options.resolve_policies(n=3)
    assert pnlcg.method == "pnlcg"
    assert pnlcg.direction == tide.optim.DirectionPolicy.NLCG
    assert pnlcg.nlcg_beta == tide.optim.NlcgBetaPolicy.HAGER_ZHANG

    with pytest.raises(ValueError, match="line_search"):
        tide.optim.resolve_policies(options={"line_search": "mystery"})


def test_cpp_optimizer_session_header_compiles(tmp_path: Path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is not available")
    source = tmp_path / "optim_session_smoke.cpp"
    source.write_text(
        r'''
#include <cassert>
#include "optim/optim.h"
#include <cstring>
#include <string>
#include <vector>

int main() {
  auto options = tide::optim::Options::pnlcg(2);
  options.line_search = tide::optim::LineSearchPolicy::HagerZhang;
  options.nlcg_beta = tide::optim::NlcgBetaPolicy::HagerZhang;
  auto more_thuente_options = options;
  more_thuente_options.line_search = tide::optim::LineSearchPolicy::MoreThuente;
  auto more_thuente_c_options = more_thuente_options.to_c_options();
  options.bounds = tide::optim::BoundsStrategy::ProjectedGradient;
  options.alpha_guess = tide::optim::AlphaGuessPolicy::Previous;
  auto expensive_gradient = tide::optim::Options::lbfgs(
      2, tide::optim::CostModel::ExpensiveGradient);
  auto joint_pstd = tide::optim::Options::pstd(
      2, tide::optim::CostModel::JointValueGradient);
  auto c_options = options.to_c_options();
  assert(c_options.direction_policy == TIDE_OPTIM_DIRECTION_NLCG);
  assert(c_options.line_search_policy == TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG);
  assert(more_thuente_c_options.line_search_policy ==
         TIDE_OPTIM_LINE_SEARCH_POLICY_MORE_THUENTE);
  assert(c_options.nlcg_beta_policy == TIDE_OPTIM_NLCG_BETA_HAGER_ZHANG);
  assert(c_options.bounds_strategy == TIDE_OPTIM_BOUNDS_PROJECTED_GRADIENT);
  assert(c_options.alpha_guess_policy == TIDE_OPTIM_ALPHA_GUESS_PREVIOUS);
  tide_optim_resolved_policies c_policies =
      tide_optim_resolve_policies(&c_options);
  assert(c_policies.valid);
  assert(c_policies.validation.code == TIDE_OPTIM_OPTIONS_VALIDATION_OK);
  assert(c_policies.direction_policy == TIDE_OPTIM_DIRECTION_NLCG);
  assert(c_policies.line_search_policy ==
         TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG);
  assert(c_policies.nlcg_beta_policy == TIDE_OPTIM_NLCG_BETA_HAGER_ZHANG);
  assert(c_policies.bounds_strategy == TIDE_OPTIM_BOUNDS_PROJECTED_GRADIENT);
  assert(std::strcmp(c_policies.method_name, "pnlcg") == 0);
  assert(std::strcmp(c_policies.line_search_policy_name, "HAGER_ZHANG") == 0);
  assert(std::strcmp(c_policies.cost_model_name, "BALANCED") == 0);
  tide_optim_resolved_policies null_policies =
      tide_optim_resolve_policies(nullptr);
  assert(!null_policies.valid);
  assert(null_policies.validation.code ==
         TIDE_OPTIM_OPTIONS_VALIDATION_NULL_OPTIONS);
  assert(std::strcmp(tide_optim_cost_model_name(1), "EXPENSIVE_GRADIENT") == 0);
  assert(expensive_gradient.line_search == tide::optim::LineSearchPolicy::ArmijoCubic);
  assert(joint_pstd.line_search == tide::optim::LineSearchPolicy::HagerZhang);
  assert(std::strcmp(
             tide::optim::name(tide::optim::CostModel::ExpensiveGradient),
             "EXPENSIVE_GRADIENT") == 0);
  assert(std::strcmp(
             tide::optim::name(tide::optim::LineSearchPolicy::MoreThuente),
             "MORE_THUENTE") == 0);
  assert(std::strcmp(
             tide::optim::method_name(tide::optim::DirectionPolicy::Nlcg),
             "pnlcg") == 0);
  assert(std::strcmp(
             tide::optim::method_name(
                 tide::optim::DirectionPolicy::PreconditionedTruncatedNewton),
             "ptrn") == 0);
  assert(std::strcmp(options.method_name(), "pnlcg") == 0);
  assert(std::strcmp(expensive_gradient.method_name(), "lbfgs") == 0);
  assert(std::strcmp(joint_pstd.method_name(), "pstd") == 0);
  tide_optim_session_snapshot invalid_snapshot =
      tide_optim_get_session_snapshot(nullptr);
  assert(!invalid_snapshot.valid);
  assert(std::strcmp(invalid_snapshot.state_name, "INVALID") == 0);
  assert(invalid_snapshot.report.status == TIDE_OPTIM_STATUS_INVALID_ARGUMENT);
  void *raw_handle = tide_optim_create(&c_options);
  assert(raw_handle != nullptr);
  tide_optim_session_snapshot raw_initial =
      tide_optim_get_session_snapshot(raw_handle);
  assert(raw_initial.valid);
  assert(!raw_initial.started);
  assert(!raw_initial.running);
  assert(!raw_initial.done);
  assert(raw_initial.n == 2);
  assert(std::strcmp(raw_initial.state_name, "NOT_STARTED") == 0);
  assert(raw_initial.report.request == TIDE_OPTIM_REQUEST_ERROR);
  assert(raw_initial.report.request_sequence == 0);
  assert(std::strcmp(raw_initial.report.method_name, "pnlcg") == 0);
  double raw_x[2] = {1.0, -1.0};
  double raw_g[2] = {2.0, -2.0};
  double raw_request_x[2] = {0.0, 0.0};
  tide_optim_report raw_report{};
  int32_t raw_request = tide_optim_start(
      raw_handle, raw_x, 2.0, raw_g, raw_request_x, &raw_report);
  tide_optim_session_snapshot raw_running =
      tide_optim_get_session_snapshot(raw_handle);
  assert(raw_running.valid);
  assert(raw_running.started);
  assert(raw_running.running);
  assert(!raw_running.done);
  assert(std::strcmp(raw_running.state_name, "RUNNING") == 0);
  assert(raw_running.report.request == raw_request);
  assert(raw_running.report.request_sequence == raw_report.request_sequence);
  assert(raw_running.report.request_sequence == 1);
  assert(raw_running.report.status == TIDE_OPTIM_STATUS_RUNNING);
  assert(raw_running.report.needs_value);
  assert(raw_running.report.needs_gradient);
  assert(raw_running.awaiting_value_gradient);
  assert(!raw_running.awaiting_preconditioner);
  assert(!raw_running.awaiting_hessian_vector);
  assert(raw_running.report.n_f == raw_report.n_f);
  assert(raw_running.report.n_g == raw_report.n_g);
  assert(raw_report.n == 2);
  assert(raw_running.report.n == 2);
  assert(raw_running.report.expected_gradient_size == 2);
  assert(raw_running.report.expected_vector_size == 0);
  tide_optim_request_summary raw_report_request =
      tide_optim_summarize_report_request(&raw_report);
  assert(raw_report_request.valid);
  assert(raw_report_request.request == raw_request);
  assert(raw_report_request.request_sequence == raw_report.request_sequence);
  assert(raw_report_request.requires_evaluation);
  assert(std::strcmp(raw_report_request.expected_evaluation,
                     "value_gradient") == 0);
  assert(std::strcmp(raw_report_request.required_fields, "f,g") == 0);
  assert(std::strcmp(raw_report_request.accepted_mapping_keys,
                     "f,g,gradient") == 0);
  assert(raw_report_request.needs_value);
  assert(raw_report_request.needs_gradient);
  tide_optim_evaluation_status raw_report_eval =
      tide_optim_validate_report_evaluation(&raw_report, 1, 1, 2, 0, 0);
  assert(raw_report_eval.request == raw_report.request);
  assert(raw_report_eval.request_sequence == raw_report.request_sequence);
  assert(raw_report_eval.expected_gradient_size == 2);
  assert(raw_report_eval.expected_vector_size == 0);
  assert(raw_report_eval.satisfied);
  tide_optim_evaluation_status raw_short_report_eval =
      tide_optim_validate_report_evaluation(&raw_report, 1, 1, 1, 0, 0);
  assert(!raw_short_report_eval.satisfied);
  assert(raw_short_report_eval.gradient_size_mismatch);
  assert(std::strcmp(raw_short_report_eval.mismatched_fields, "g") == 0);
  tide_optim_evaluation_status null_report_eval =
      tide_optim_validate_report_evaluation(nullptr, 0, 0, 0, 0, 0);
  assert(null_report_eval.request == TIDE_OPTIM_REQUEST_ERROR);
  assert(!null_report_eval.requires_evaluation);
  assert(!null_report_eval.satisfied);
  tide_optim_destroy(raw_handle);
  tide_optim_report_summary null_summary = tide_optim_summarize_report(nullptr);
  assert(!null_summary.valid);
  assert(null_summary.request == TIDE_OPTIM_REQUEST_ERROR);
  assert(null_summary.status == TIDE_OPTIM_STATUS_INVALID_ARGUMENT);
  assert(std::strcmp(null_summary.request_name, "ERROR") == 0);
  assert(std::strcmp(null_summary.status_name, "INVALID_ARGUMENT") == 0);
  assert(std::strcmp(null_summary.failure_reason, "INVALID_ARGUMENT") == 0);
  assert(null_summary.stopped);
  assert(null_summary.failed);
  assert(!null_summary.success);
  assert(std::strcmp(
             tide_optim_request_kind_name(TIDE_OPTIM_REQUEST_EVALUATE_FG),
             "EVALUATE_FG") == 0);
  assert(std::strcmp(tide_optim_request_expected_evaluation(
                         TIDE_OPTIM_REQUEST_EVALUATE_FG),
                     "value_gradient") == 0);
  assert(std::strcmp(tide_optim_request_required_fields(
                         TIDE_OPTIM_REQUEST_EVALUATE_FG),
                     "f,g") == 0);
  assert(std::strcmp(tide_optim_request_accepted_mapping_keys(
                         TIDE_OPTIM_REQUEST_EVALUATE_FG),
                     "f,g,gradient") == 0);
  assert(tide_optim_request_requires_evaluation(
      TIDE_OPTIM_REQUEST_EVALUATE_FG));
  assert(!tide_optim_request_requires_evaluation(TIDE_OPTIM_REQUEST_DONE));
  assert(tide_optim_request_needs_value(TIDE_OPTIM_REQUEST_EVALUATE_FG));
  assert(tide_optim_request_needs_gradient(TIDE_OPTIM_REQUEST_EVALUATE_FG));
  assert(tide_optim_request_needs_value_gradient(
      TIDE_OPTIM_REQUEST_EVALUATE_FG));
  assert(!tide_optim_request_needs_gradient(TIDE_OPTIM_REQUEST_EVALUATE_F));
  assert(tide_optim_request_needs_preconditioner(
      TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER));
  assert(tide_optim_request_needs_hessian_vector(TIDE_OPTIM_REQUEST_EVALUATE_HV));
  assert(tide_optim_request_needs_vector_result(
      TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER));
  assert(tide_optim_request_needs_vector_result(TIDE_OPTIM_REQUEST_EVALUATE_HV));
  assert(!tide_optim_request_needs_vector_result(TIDE_OPTIM_REQUEST_EVALUATE_FG));
  assert(tide_optim_request_is_done(TIDE_OPTIM_REQUEST_DONE));
  assert(tide_optim_request_is_error(TIDE_OPTIM_REQUEST_ERROR));
  assert(std::strcmp(tide_optim_request_kind_name(12345), "UNKNOWN") == 0);
  tide_optim_request_summary fg_request =
      tide_optim_summarize_request(TIDE_OPTIM_REQUEST_EVALUATE_FG);
  assert(fg_request.valid);
  assert(fg_request.request == TIDE_OPTIM_REQUEST_EVALUATE_FG);
  assert(fg_request.request_sequence == 0);
  assert(std::strcmp(fg_request.request_name, "EVALUATE_FG") == 0);
  assert(fg_request.requires_evaluation);
  assert(std::strcmp(fg_request.expected_evaluation, "value_gradient") == 0);
  assert(std::strcmp(fg_request.required_fields, "f,g") == 0);
  assert(std::strcmp(fg_request.accepted_mapping_keys,
                     "f,g,gradient") == 0);
  assert(fg_request.needs_value);
  assert(fg_request.needs_gradient);
  assert(fg_request.needs_value_gradient);
  assert(!fg_request.needs_vector_result);
  tide_optim_request_summary unknown_request =
      tide_optim_summarize_request(12345);
  assert(!unknown_request.valid);
  assert(std::strcmp(unknown_request.request_name, "UNKNOWN") == 0);
  assert(std::strcmp(unknown_request.expected_evaluation, "none") == 0);
  assert(std::strcmp(unknown_request.required_fields, "") == 0);
  assert(std::strcmp(unknown_request.accepted_mapping_keys, "") == 0);
  assert(!unknown_request.requires_evaluation);
  tide_optim_request_summary null_report_request =
      tide_optim_summarize_report_request(nullptr);
  assert(!null_report_request.valid);
  assert(null_report_request.request == TIDE_OPTIM_REQUEST_ERROR);
  assert(null_report_request.request_sequence == 0);
  assert(std::strcmp(null_report_request.request_name, "ERROR") == 0);
  tide_optim_evaluation_status short_eval =
      tide_optim_validate_evaluation(TIDE_OPTIM_REQUEST_EVALUATE_FG, 7, 2, 0,
                                     1, 1, 1, 0, 0);
  assert(short_eval.request == TIDE_OPTIM_REQUEST_EVALUATE_FG);
  assert(short_eval.request_sequence == 7);
  assert(std::strcmp(short_eval.request_name, "EVALUATE_FG") == 0);
  assert(std::strcmp(short_eval.expected_evaluation, "value_gradient") == 0);
  assert(std::strcmp(short_eval.required_fields, "f,g") == 0);
  assert(std::strcmp(short_eval.accepted_mapping_keys, "f,g,gradient") == 0);
  assert(short_eval.requires_evaluation);
  assert(short_eval.has_value);
  assert(short_eval.has_gradient);
  assert(!short_eval.has_vector);
  assert(short_eval.gradient_size == 1);
  assert(short_eval.expected_gradient_size == 2);
  assert(!short_eval.has_missing_fields);
  assert(std::strcmp(short_eval.missing_fields, "") == 0);
  assert(short_eval.gradient_size_mismatch);
  assert(short_eval.has_size_mismatch);
  assert(std::strcmp(short_eval.mismatched_fields, "g") == 0);
  assert(!short_eval.satisfied);
  assert(!short_eval.valid);
  tide_optim_evaluation_status ok_eval =
      tide_optim_validate_evaluation(TIDE_OPTIM_REQUEST_EVALUATE_FG, 8, 2, 0,
                                     1, 1, 2, 0, 0);
  assert(ok_eval.satisfied);
  assert(ok_eval.valid);
  assert(std::strcmp(ok_eval.missing_fields, "") == 0);
  assert(std::strcmp(ok_eval.mismatched_fields, "") == 0);
  tide_optim_evaluation_status missing_hv =
      tide_optim_validate_evaluation(TIDE_OPTIM_REQUEST_EVALUATE_HV, 9, 0, 2,
                                     0, 0, 0, 0, 0);
  assert(missing_hv.requires_evaluation);
  assert(missing_hv.missing_vector);
  assert(std::strcmp(missing_hv.missing_fields, "vector") == 0);
  assert(!missing_hv.satisfied);
  tide_optim_evaluation_status done_eval =
      tide_optim_validate_evaluation(TIDE_OPTIM_REQUEST_DONE, 10, 2, 2,
                                     1, 1, 2, 1, 2);
  assert(!done_eval.requires_evaluation);
  assert(std::strcmp(done_eval.expected_evaluation, "none") == 0);
  assert(std::strcmp(done_eval.missing_fields, "") == 0);
  assert(!done_eval.satisfied);
  assert(!done_eval.valid);
  assert(std::strcmp(
             tide_optim_status_name(TIDE_OPTIM_STATUS_INNER_CG_FAILED),
             "INNER_CG_FAILED") == 0);
  assert(std::strcmp(
             tide_optim_line_search_policy_name(
                 TIDE_OPTIM_LINE_SEARCH_POLICY_MORE_THUENTE),
             "MORE_THUENTE") == 0);
  assert(std::strcmp(
             tide_optim_direction_policy_name(
                 TIDE_OPTIM_DIRECTION_PRECONDITIONED_TRUNCATED_NEWTON),
             "PRECONDITIONED_TRUNCATED_NEWTON") == 0);
  assert(std::strcmp(
             tide_optim_direction_method_name(
                 TIDE_OPTIM_DIRECTION_PRECONDITIONED_TRUNCATED_NEWTON),
             "ptrn") == 0);
  assert(std::strcmp(
             tide_optim_options_validation_code_name(
                 TIDE_OPTIM_OPTIONS_VALIDATION_TRACE_STRIDE),
             "TRACE_STRIDE") == 0);
  assert(more_thuente_options.validate());
  tide::optim::OptionsValidation validation = options.validate();
  assert(validation);
  assert(validation.ok());
  assert(validation.code == tide::optim::OptionsValidationCode::Ok);
  assert(std::strcmp(validation.code_name(), "OK") == 0);
  tide::optim::TraceOptions validation_trace_options{};
  assert(validation_trace_options.validate());
  validation_trace_options.policy = tide::optim::TracePolicy::Stride;
  validation_trace_options.stride = 2;
  assert(validation_trace_options.valid());
  auto bad_trace_policy = validation_trace_options;
  bad_trace_policy.policy = static_cast<tide::optim::TracePolicy>(99);
  tide::optim::OptionsValidation bad_trace_policy_validation =
      bad_trace_policy.validate();
  assert(!bad_trace_policy_validation);
  assert(bad_trace_policy_validation.code ==
         tide::optim::OptionsValidationCode::TracePolicy);
  assert(std::strcmp(bad_trace_policy_validation.field, "trace_policy") == 0);
  assert(std::strcmp(bad_trace_policy_validation.code_name(), "TRACE_POLICY") == 0);
  auto bad_trace_stride = validation_trace_options;
  bad_trace_stride.stride = 0;
  tide::optim::OptionsValidation bad_trace_stride_validation =
      bad_trace_stride.validate();
  assert(!bad_trace_stride_validation);
  assert(bad_trace_stride_validation.code ==
         tide::optim::OptionsValidationCode::TraceStride);
  assert(std::strcmp(bad_trace_stride_validation.field, "trace_stride") == 0);
  assert(std::strcmp(bad_trace_stride_validation.code_name(), "TRACE_STRIDE") == 0);
  auto bad_alpha_bounds = options;
  bad_alpha_bounds.alpha_min = 1.0;
  bad_alpha_bounds.alpha_max = 1.0;
  tide::optim::OptionsValidation bad_alpha_validation =
      bad_alpha_bounds.validate();
  assert(!bad_alpha_validation);
  assert(!bad_alpha_validation.ok());
  assert(bad_alpha_validation.code ==
         tide::optim::OptionsValidationCode::AlphaBounds);
  assert(std::strcmp(bad_alpha_validation.field, "alpha_min/alpha_max") == 0);
  assert(std::strcmp(bad_alpha_validation.code_name(), "ALPHA_BOUNDS") == 0);
  auto bad_margin = options;
  bad_margin.bound_margin = -1.0;
  tide::optim::OptionsValidation bad_margin_validation =
      bad_margin.validate();
  assert(bad_margin_validation.code ==
         tide::optim::OptionsValidationCode::BoundMargin);
  tide::optim::Session bad_session(bad_margin);
  assert(!bad_session.valid());
  assert(bad_session.closed());
  assert(!bad_session.started());
  assert(!bad_session.done());
  assert(!bad_session.running());
  assert(std::strcmp(bad_session.state_name(), "CLOSED") == 0);

  tide::optim::Session session(options);
  assert(session.valid());
  assert(!session.closed());
  assert(!session.started());
  assert(!session.done());
  assert(!session.running());
  assert(std::strcmp(session.state_name(), "NOT_STARTED") == 0);
  assert(session.current_request().error());
  std::vector<double> x{1.0, -1.0};
  std::vector<double> g{2.0, -2.0};
  tide::optim::Request request = session.start(tide::optim::view(x), 2.0, tide::optim::view(g));
  assert(session.started());
  assert(session.running());
  assert(!session.done());
  assert(std::strcmp(session.state_name(), "RUNNING") == 0);
  tide::optim::RequestKind kind = request.kind;
  assert(!request.done());
  assert(!request.error());
  assert(request.needs_value());
  assert(request.needs_gradient());
  assert(request.needs_value_gradient());
  assert(!request.needs_vector_result());
  assert(!request.needs_preconditioner());
  assert(!request.needs_hessian_vector());
  assert(request.has_x());
  assert(request.x_size() == 2);
  assert(request.expected_gradient_size() == 2);
  assert(request.expected_vector_size() == 0);
  assert(!request.has_vector());
  assert(request.vector_size() == 0);
  assert(request.has_report());
  assert(request.sequence == 1);
  tide::optim::Request current_request = session.current_request();
  assert(current_request.kind == request.kind);
  assert(current_request.sequence == request.sequence);
  assert(current_request.x_size() == request.x_size());
  assert(current_request.expected_gradient_size() == request.expected_gradient_size());
  assert(current_request.expected_vector_size() == request.expected_vector_size());
  assert(current_request.has_report());
  assert(std::strcmp(request.kind_name(), tide::optim::name(kind)) == 0);
  assert(request.requires_evaluation());
  assert(std::strcmp(request.expected_evaluation(), "value_gradient") == 0);
  assert(std::strcmp(request.required_fields(), "f,g") == 0);
  assert(std::strcmp(request.accepted_mapping_keys(),
                     "f,g,gradient") == 0);
  tide::optim::RequestRequirements request_requirements =
      request.requirements();
  assert(request_requirements.kind == request.kind);
  assert(request_requirements.sequence == request.sequence);
  assert(request_requirements.needs_value_gradient());
  assert(std::strcmp(request_requirements.expected_evaluation(),
                     "value_gradient") == 0);
  tide::optim::Status status = session.current_x(tide::optim::view(x));
  tide::optim::ReportView report = request.report;
  tide_optim_report_summary request_summary =
      tide_optim_summarize_report(report.raw());
  assert(request_summary.valid);
  assert(request_summary.request == static_cast<int32_t>(kind));
  assert(request_summary.status == TIDE_OPTIM_STATUS_RUNNING);
  assert(request_summary.request_sequence == request.sequence);
  assert(request_summary.request_sequence == report.request_sequence());
  assert(request_summary.n == report.n());
  assert(request_summary.n == 2);
  assert(request_summary.expected_gradient_size == report.expected_gradient_size());
  assert(request_summary.expected_gradient_size == 2);
  assert(request_summary.expected_vector_size == report.expected_vector_size());
  assert(request_summary.expected_vector_size == 0);
  assert(std::strcmp(request_summary.request_name, report.request_name()) == 0);
  assert(std::strcmp(request_summary.expected_evaluation,
                     "value_gradient") == 0);
  assert(std::strcmp(request_summary.required_fields, "f,g") == 0);
  assert(std::strcmp(request_summary.accepted_mapping_keys,
                     "f,g,gradient") == 0);
  assert(request_summary.requires_evaluation);
  assert(report.requires_evaluation());
  assert(!report.error());
  assert(!report.done());
  assert(report.needs_value());
  assert(report.needs_gradient());
  assert(report.needs_value_gradient());
  assert(!report.needs_vector_result());
  assert(!report.needs_preconditioner());
  assert(!report.needs_hessian_vector());
  assert(std::strcmp(report.expected_evaluation(), "value_gradient") == 0);
  assert(std::strcmp(report.required_fields(), "f,g") == 0);
  assert(std::strcmp(report.accepted_mapping_keys(), "f,g,gradient") == 0);
  assert(std::strcmp(request_summary.status_name, report.status_name()) == 0);
  assert(std::strcmp(request_summary.method_name, "pnlcg") == 0);
  assert(std::strcmp(request_summary.direction_policy_name, "NLCG") == 0);
  assert(std::strcmp(request_summary.line_search_policy_name, "HAGER_ZHANG") == 0);
  assert(request_summary.needs_value);
  assert(request_summary.needs_gradient);
  assert(request_summary.needs_value_gradient);
  assert(!request_summary.needs_vector_result);
  assert(!request_summary.done);
  assert(!request_summary.success);
  assert(!request_summary.stopped);
  assert(!request_summary.failed);
  assert(request_summary.failure_reason == nullptr);
  assert(request_summary.n_f == report.n_f());
  assert(request_summary.n_g == report.n_g());
  assert(request_summary.f == report.f());
  assert(request_summary.grad_norm == report.grad_norm());
  tide::optim::EventCounts counts = report.event_counts();
  tide::optim::LineSearchDiagnostics line_search_diagnostics =
      report.line_search_diagnostics();
  tide::optim::PairUpdateDiagnostics pair_update_diagnostics =
      report.pair_update_diagnostics();
  tide::optim::InnerSolveDiagnostics inner_solve_diagnostics =
      report.inner_solve_diagnostics();
  tide::optim::TrustRegionDiagnostics trust_region_diagnostics =
      report.trust_region_diagnostics();
  tide::optim::StoppingDiagnostics stopping_diagnostics =
      report.stopping_diagnostics();
  tide::optim::DirectionDiagnostics direction_diagnostics =
      report.direction_diagnostics();
  tide::optim::LineSearchStatus line_search_status =
      report.line_search_status();
  tide::optim::PairStatus pair_status = report.pair_status();
  tide::optim::DirectionPolicy direction_policy = report.direction_policy();
  tide::optim::LineSearchPolicy line_search_policy = report.line_search_policy();
  tide::optim::NlcgBetaPolicy beta_policy = report.nlcg_beta_policy();
  tide::optim::ReportView last_report = session.last_report_view();
  int warning_bits = TIDE_OPTIM_WARNING_NONFINITE_TRIAL |
                     TIDE_OPTIM_WARNING_LBFGS_PAIR_SKIPPED;
  std::vector<tide::optim::WarningFlag> warning_flags =
      tide::optim::warning_flags(warning_bits);
  std::vector<char const*> warning_names =
      tide::optim::warning_names(warning_bits);
  assert(report.request_kind() == kind);
  assert(report.n() == 2);
  assert(report.expected_gradient_size() == 2);
  assert(report.expected_vector_size() == 0);
  assert(report.status() == tide::optim::Status::Running);
  assert(stopping_diagnostics.status == tide::optim::Status::Running);
  assert(stopping_diagnostics.policy == tide::optim::StoppingPolicy::Standard);
  assert(!stopping_diagnostics.stopped());
  assert(!stopping_diagnostics.failed());
  assert(stopping_diagnostics.grad_tolerance >= 0.0);
  assert(std::strcmp(stopping_diagnostics.policy_name(), "STANDARD") == 0);
  assert(direction_policy == tide::optim::DirectionPolicy::Nlcg);
  assert(std::strcmp(report.method_name(), "pnlcg") == 0);
  assert(direction_diagnostics.policy == direction_policy);
  assert(direction_diagnostics.uses_nlcg_beta());
  assert(direction_diagnostics.finite());
  assert(line_search_policy == tide::optim::LineSearchPolicy::HagerZhang);
  assert(line_search_diagnostics.policy == line_search_policy);
  assert(line_search_diagnostics.accept_count == counts.line_search_accept);
  assert(!line_search_diagnostics.failed());
  assert(std::strcmp(line_search_diagnostics.policy_name(), "HAGER_ZHANG") == 0);
  assert(pair_update_diagnostics.status == pair_status);
  assert(pair_update_diagnostics.stored_count == counts.pair_stored);
  assert(!pair_update_diagnostics.regularized());
  assert(inner_solve_diagnostics.inner_status == report.inner_status());
  assert(inner_solve_diagnostics.warning_count == counts.inner_warning);
  assert(!inner_solve_diagnostics.failed());
  assert(trust_region_diagnostics.status == report.trust_region_status());
  assert(trust_region_diagnostics.accept_count == counts.trust_region_accept);
  assert(!trust_region_diagnostics.failed());
  assert(beta_policy == tide::optim::NlcgBetaPolicy::HagerZhang);
  assert(last_report.raw() == report.raw());
  assert(report.warning_flags() == 0);
  assert(report.warnings().empty());
  assert(report.warning_names().empty());
  tide::optim::ResolvedPolicies option_policies = options.resolved_policies();
  assert(std::strcmp(option_policies.method_name(), "pnlcg") == 0);
  assert(option_policies.direction == tide::optim::DirectionPolicy::Nlcg);
  assert(option_policies.line_search ==
         tide::optim::LineSearchPolicy::HagerZhang);
  assert(option_policies.bounds ==
         tide::optim::BoundsStrategy::ProjectedGradient);
  tide::optim::ResolvedPolicies report_policies =
      tide::optim::ResolvedPolicies::from_report(report, &options);
  assert(report_policies.direction == tide::optim::DirectionPolicy::Nlcg);
  assert(report_policies.line_search ==
         tide::optim::LineSearchPolicy::HagerZhang);
  assert(report_policies.bounds ==
         tide::optim::BoundsStrategy::ProjectedGradient);
  assert(std::strcmp(report_policies.method_name(), "pnlcg") == 0);
  assert(std::strcmp(report_policies.line_search_name(), "HAGER_ZHANG") == 0);
  std::string config_signature =
      tide::optim::make_config_signature(options, report_policies);
  std::string config_fingerprint =
      tide::optim::make_config_fingerprint(config_signature);
  assert(options.config_signature() ==
         tide::optim::make_config_signature(options, option_policies));
  assert(options.config_fingerprint() ==
         tide::optim::make_config_fingerprint(options, option_policies));
  assert(config_signature.find("schema=tide.optim.config.v1") != std::string::npos);
  assert(config_signature.find("method=pnlcg") != std::string::npos);
  assert(config_signature.find("policy.method=pnlcg") != std::string::npos);
  assert(config_signature.find("policy.line_search=HAGER_ZHANG") != std::string::npos);
  assert(config_fingerprint.size() == 16);
  auto changed_options = options;
  changed_options.line_search = tide::optim::LineSearchPolicy::LegacyWeakWolfe;
  tide::optim::ResolvedPolicies changed_policies =
      tide::optim::ResolvedPolicies::from_options(changed_options);
  assert(changed_options.config_fingerprint() ==
         tide::optim::make_config_fingerprint(changed_options, changed_policies));
  assert(changed_options.config_fingerprint() != options.config_fingerprint());
  assert(tide::optim::make_config_fingerprint(changed_options, changed_policies) !=
         config_fingerprint);
  assert(tide::optim::failure_reason(
             tide::optim::Status::InnerCgFailed,
             tide::optim::LineSearchStatus::Started,
             tide::optim::InnerCgStatus::NonfiniteHvp,
             tide::optim::TrustRegionStatus::None) ==
         std::string("INNER_CG_NONFINITE_HVP"));
  assert(tide::optim::failure_reason(
             tide::optim::Status::TrustRegionFailed,
             tide::optim::LineSearchStatus::Started,
             tide::optim::InnerCgStatus::None,
             tide::optim::TrustRegionStatus::FailedPredictedReduction) ==
         std::string("TRUST_REGION_FAILED_PREDICTED_REDUCTION"));
  assert(warning_flags.size() == 2);
  assert(warning_flags[0] == tide::optim::WarningFlag::NonfiniteTrial);
  assert(warning_flags[1] == tide::optim::WarningFlag::LbfgsPairSkipped);
  assert(std::strcmp(warning_names[0], "NONFINITE_TRIAL") == 0);
  assert(std::strcmp(
             tide::optim::name(tide::optim::WarningFlag::LbfgsPairSkipped),
             "LBFGS_PAIR_SKIPPED") == 0);
  assert(tide::optim::has_warning(
      warning_bits, tide::optim::WarningFlag::LbfgsPairSkipped));
  assert(!tide::optim::has_warning(
      warning_bits, tide::optim::WarningFlag::InnerCg));
  tide::optim::Evaluation empty_evaluation{};
  assert(!empty_evaluation.has_value());
  assert(!empty_evaluation.has_gradient());
  assert(!empty_evaluation.has_vector());
  tide::optim::EvaluationStatus empty_status =
      empty_evaluation.status_for(request);
  tide::optim::EvaluationStatus empty_report_status =
      empty_evaluation.status_for(report);
  assert(!empty_report_status.satisfied());
  assert(empty_report_status.expected_gradient_size == report.expected_gradient_size());
  assert(empty_report_status.expected_vector_size == report.expected_vector_size());
  assert(std::strcmp(empty_report_status.missing_fields(), "f,g") == 0);
  assert(!empty_evaluation.valid_for(report));
  assert(!empty_status.satisfied());
  assert(!empty_status.valid());
  assert(empty_status.missing_value());
  assert(empty_status.missing_gradient());
  assert(!empty_status.missing_vector());
  assert(empty_status.has_missing_fields());
  assert(std::strcmp(empty_status.expected_evaluation(),
                     "value_gradient") == 0);
  assert(std::strcmp(empty_status.required_fields(), "f,g") == 0);
  assert(std::strcmp(empty_status.accepted_mapping_keys(),
                     "f,g,gradient") == 0);
  assert(!empty_status.has_size_mismatch());
  assert(!empty_status.gradient_size_mismatch());
  assert(!empty_status.vector_size_mismatch());
  assert(empty_status.expected_gradient_size == request.x_size());
  assert(empty_status.expected_vector_size == 0);
  assert(std::strcmp(empty_status.missing_fields(), "f,g") == 0);
  assert(std::strcmp(empty_status.mismatched_fields(), "") == 0);
  assert(std::strcmp(empty_evaluation.missing_fields(request), "f,g") == 0);
  assert(!empty_evaluation.satisfied_by(request));
  assert(!empty_evaluation.valid_for(request));
  tide::optim::Evaluation value_only_evaluation =
      tide::optim::Evaluation::value(2.0);
  tide::optim::EvaluationStatus value_only_status =
      value_only_evaluation.status_for(request);
  assert(!value_only_status.missing_value());
  assert(value_only_status.missing_gradient());
  assert(std::strcmp(value_only_status.missing_fields(), "g") == 0);
  assert(std::strcmp(value_only_status.mismatched_fields(), "") == 0);
  assert(!value_only_evaluation.valid_for(request));
  std::vector<double> short_g{1.0};
  tide::optim::Evaluation short_gradient_evaluation =
      tide::optim::Evaluation::value_gradient(2.0, tide::optim::view(short_g));
  tide::optim::EvaluationStatus short_gradient_status =
      short_gradient_evaluation.status_for(request);
  tide::optim::EvaluationStatus short_gradient_report_status =
      short_gradient_evaluation.status_for(report);
  assert(!short_gradient_report_status.satisfied());
  assert(short_gradient_report_status.gradient_size_mismatch());
  assert(short_gradient_report_status.expected_gradient_size == report.expected_gradient_size());
  assert(std::strcmp(short_gradient_report_status.mismatched_fields(), "g") == 0);
  assert(!short_gradient_evaluation.valid_for(report));
  assert(!short_gradient_status.satisfied());
  assert(!short_gradient_status.has_missing_fields());
  assert(short_gradient_status.has_size_mismatch());
  assert(short_gradient_status.gradient_size_mismatch());
  assert(!short_gradient_status.vector_size_mismatch());
  assert(short_gradient_status.gradient_size == 1);
  assert(short_gradient_status.expected_gradient_size == request.x_size());
  assert(std::strcmp(short_gradient_status.missing_fields(), "") == 0);
  assert(std::strcmp(short_gradient_status.mismatched_fields(), "g") == 0);
  assert(!short_gradient_evaluation.valid_for(request));
  tide::optim::Request short_gradient_response =
      session.respond(request, short_gradient_evaluation);
  assert(short_gradient_response.error());
  tide::optim::Request rejected_response =
      session.respond(request, empty_evaluation);
  assert(rejected_response.error());
  tide::optim::Evaluation fg_evaluation =
      tide::optim::Evaluation::value_gradient(2.0, tide::optim::view(g));
  assert(fg_evaluation.has_value());
  assert(fg_evaluation.has_gradient());
  assert(!fg_evaluation.has_vector());
  assert(fg_evaluation.gradient_size() == 2);
  tide::optim::EvaluationStatus fg_status = fg_evaluation.status_for(request);
  tide::optim::EvaluationStatus fg_report_status = fg_evaluation.status_for(report);
  assert(fg_report_status.satisfied());
  assert(fg_report_status.valid());
  assert(fg_report_status.expected_gradient_size == report.expected_gradient_size());
  assert(std::strcmp(fg_report_status.missing_fields(), "") == 0);
  assert(std::strcmp(fg_report_status.mismatched_fields(), "") == 0);
  assert(fg_evaluation.valid_for(report));
  assert(fg_status.satisfied());
  assert(fg_status.valid());
  assert(!fg_status.has_missing_fields());
  assert(!fg_status.has_size_mismatch());
  assert(!fg_status.gradient_size_mismatch());
  assert(fg_status.gradient_size == request.x_size());
  assert(fg_status.expected_gradient_size == request.x_size());
  assert(std::strcmp(fg_status.missing_fields(), "") == 0);
  assert(std::strcmp(fg_status.mismatched_fields(), "") == 0);
  assert(fg_evaluation.satisfied_by(request));
  assert(fg_evaluation.valid_for(request));
  tide::optim::EvaluationStatus done_status =
      fg_evaluation.status_for(tide::optim::RequestRequirements::from_kind(
          tide::optim::RequestKind::Done));
  assert(!done_status.satisfied());
  assert(!done_status.has_missing_fields());
  assert(std::strcmp(done_status.expected_evaluation(), "none") == 0);
  assert(std::strcmp(done_status.missing_fields(), "") == 0);
  tide::optim::Request previous_request = request;
  request = session.tell_evaluation(previous_request, fg_evaluation);
  assert(request.has_report());
  assert(request.sequence > previous_request.sequence);
  tide::optim::Request stale_response =
      session.respond(previous_request, fg_evaluation);
  assert(stale_response.error());
  assert(stale_response.sequence == 0);
  assert(session.running() || session.done());
  session.close();
  assert(session.closed());
  assert(!session.running());
  assert(std::strcmp(session.state_name(), "CLOSED") == 0);

  tide::optim::Objective objective{};
  objective.value_gradient = [](tide::optim::VectorView x,
                                tide::optim::MutableVectorView gradient) {
    gradient.data[0] = x.data[0];
    gradient.data[1] = x.data[1];
    return 0.5 * (x.data[0] * x.data[0] + x.data[1] * x.data[1]);
  };
  objective.value = [](tide::optim::VectorView x) {
    return 0.5 * (x.data[0] * x.data[0] + x.data[1] * x.data[1]);
  };
  objective.preconditioner = [](tide::optim::VectorView,
                                tide::optim::VectorView vector,
                                tide::optim::MutableVectorView out) {
    out.data[0] = vector.data[0];
    out.data[1] = vector.data[1];
  };
  objective.hessian_vector = objective.preconditioner;
  std::vector<double> lb{-2.0, -2.0};
  std::vector<double> ub{2.0, 2.0};
  tide::optim::Bounds bounds{tide::optim::view(lb), tide::optim::view(ub)};
  tide::optim::TraceOptions trace_options{};
  trace_options.policy = tide::optim::TracePolicy::Last;
  tide::optim::Result result = tide::optim::minimize(
      options, tide::optim::view(x), objective, bounds, trace_options,
      [](tide::optim::ReportView callback_report) {
        return callback_report.iter() > 10;
      });
  assert(result.report_view().raw() == &result.report);
  assert(result.trace_policy == tide::optim::TracePolicy::Last);
  assert(std::strcmp(result.method_name(), "pnlcg") == 0);
  assert(std::strcmp(result.line_search_policy_name(), "HAGER_ZHANG") == 0);
  assert(std::strcmp(result.globalization_policy_name(), "LINE_SEARCH") == 0);
  assert(std::strcmp(result.bounds_strategy_name(), "PROJECTED_GRADIENT") == 0);
  assert(std::strcmp(result.cost_model_name(), "BALANCED") == 0);
  assert(result.resolved_policies.direction == tide::optim::DirectionPolicy::Nlcg);
  assert(result.resolved_policies.line_search ==
         tide::optim::LineSearchPolicy::HagerZhang);
  assert(result.resolved_policies.alpha_guess ==
         tide::optim::AlphaGuessPolicy::Previous);
  assert(result.resolved_policies.bounds ==
         tide::optim::BoundsStrategy::ProjectedGradient);
  assert(std::strcmp(result.resolved_policies.bounds_name(),
                     "PROJECTED_GRADIENT") == 0);
  assert(result.config_signature ==
         tide::optim::make_config_signature(options, result.resolved_policies));
  assert(result.config_fingerprint ==
         tide::optim::make_config_fingerprint(result.config_signature));
  assert(result.config_fingerprint.size() == 16);
  assert(result.config_signature.find("n=2") != std::string::npos);
  assert(result.n_trace_events >= result.n_trace_stored());
  assert(result.trace_summary.n_reports == result.n_trace_events);
  assert(result.trace_summary.request_count(tide::optim::RequestKind::Done) == 1);
  assert(result.trace_summary.request_count(
             tide::optim::RequestKind::EvaluateFG) >= 1);
  assert(result.trace_summary.expected_gradient_requests ==
         result.trace_summary.request_count(
             tide::optim::RequestKind::EvaluateFG));
  assert(result.trace_summary.expected_vector_requests ==
         result.trace_summary.request_count(
             tide::optim::RequestKind::ApplyPreconditioner) +
             result.trace_summary.request_count(
                 tide::optim::RequestKind::EvaluateHv));
  assert(result.trace_summary.expected_gradient_elements ==
         result.trace_summary.expected_gradient_requests * 2);
  assert(result.trace_summary.expected_vector_elements ==
         result.trace_summary.expected_vector_requests * 2);
  assert(result.trace_summary.expected_total_vector_elements() ==
         result.trace_summary.expected_gradient_elements +
             result.trace_summary.expected_vector_elements);
  assert(result.trace_summary.status_count(tide::optim::Status::Running) > 0);
  assert(result.trace_summary.status_count(result.status) == 1);
  assert(result.trace_summary.failure_reason_count("null") >= 1);
  assert(result.trace_summary.success_count() == 1);
  assert(result.trace_summary.failed_count() == 0);
  assert(result.trace_summary.user_stopped_count() == 0);
  assert(result.trace_summary.line_search_status_count(
             tide::optim::LineSearchStatus::Started) > 0);
  assert(result.trace_summary.line_search_acceptance_count(
             tide::optim::LineSearchAcceptance::ApproximateWolfe) >= 1);
  assert(result.trace_summary.event_counts().line_search_accept ==
         result.event_counts().line_search_accept);
  tide::optim::EvaluationProfile evaluation_profile =
      result.evaluation_profile();
  assert(evaluation_profile.cost_model == tide::optim::CostModel::Balanced);
  assert(std::strcmp(evaluation_profile.cost_model_name(), "BALANCED") == 0);
  assert(evaluation_profile.n_f == result.n_f());
  assert(evaluation_profile.n_g == result.n_g());
  assert(evaluation_profile.n_value_gradient() == result.n_g());
  assert(evaluation_profile.n_value_only() ==
         (result.n_f() > result.n_g() ? result.n_f() - result.n_g() : 0));
  assert(evaluation_profile.n_total_requests() ==
         evaluation_profile.n_value_only() + result.n_g() +
             result.n_hvp() + result.n_prec());
  tide::optim::EvaluationCostEstimate cost_estimate =
      result.evaluation_cost_estimate();
  assert(cost_estimate.cost_model == tide::optim::CostModel::Balanced);
  assert(std::strcmp(cost_estimate.cost_model_name(), "BALANCED") == 0);
  assert(cost_estimate.weights.value_gradient == 2.0);
  assert(cost_estimate.n == 2);
  assert(cost_estimate.n_value_gradient == evaluation_profile.n_value_gradient());
  assert(cost_estimate.weighted_request_units() >=
         static_cast<double>(evaluation_profile.n_value_gradient()));
  assert(cost_estimate.expected_gradient_elements ==
         result.trace_summary.expected_gradient_elements);
  assert(cost_estimate.expected_vector_elements ==
         result.trace_summary.expected_vector_elements);
  assert(cost_estimate.expected_total_vector_elements() ==
         result.trace_summary.expected_total_vector_elements());
  tide::optim::StoppingDiagnostics result_stopping =
      result.stopping_diagnostics();
  assert(result_stopping.success());
  assert(result_stopping.status == result.status);
  assert(result_stopping.n_iter == result.n_iter());
  assert(result_stopping.n_f == result.n_f());
  assert(result_stopping.max_iter == options.max_iter);
  assert(result_stopping.max_eval == options.max_eval);
  assert(result_stopping.reason() != nullptr);
  assert(result_stopping.failure_reason() == nullptr);
  tide::optim::LineSearchDiagnostics result_line_search =
      result.line_search_diagnostics();
  assert(result_line_search.policy == tide::optim::LineSearchPolicy::HagerZhang);
  assert(result_line_search.accept_count ==
         result.event_counts().line_search_accept);
  assert(!result_line_search.failed());
  tide::optim::PairUpdateDiagnostics result_pair_update =
      result.pair_update_diagnostics();
  assert(result_pair_update.stored_count == result.event_counts().pair_stored);
  assert(result_pair_update.status_name() != nullptr);
  tide::optim::InnerSolveDiagnostics result_inner =
      result.inner_solve_diagnostics();
  assert(result_inner.n_hvp == result.n_hvp());
  assert(result_inner.n_prec == result.n_prec());
  assert(result_inner.inner_status_name() != nullptr);
  tide::optim::TrustRegionDiagnostics result_trust =
      result.trust_region_diagnostics();
  tide::optim::DirectionDiagnostics result_direction =
      result.direction_diagnostics();
  assert(result_direction.policy == tide::optim::DirectionPolicy::Nlcg);
  assert(result_direction.policy_name() != nullptr);
  assert(result_trust.accept_count == result.event_counts().trust_region_accept);
  assert(result_trust.status_name() != nullptr);
  assert(result.trace_summary.warning_names().empty());
  (void)session;
  (void)request;
  (void)kind;
  (void)status;
  (void)counts;
  (void)line_search_status;
  (void)pair_status;
  return 0;
}
''',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            tide.optim.get_include(),
            "-c",
            str(source),
            "-o",
            str(tmp_path / "optim_session_smoke.o"),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cpp_minimize_driver_links_and_solves_quadratic(tmp_path: Path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is not available")
    root = Path(__file__).resolve().parents[1]
    library_dir = root / "src" / "tide"
    library = library_dir / "libtide_C.so"
    if not library.exists():
        pytest.skip("native tide library is not built")

    source = tmp_path / "optim_minimize_smoke.cpp"
    executable = tmp_path / "optim_minimize_smoke"
    source.write_text(
        r'''
#include <cassert>
#include <cmath>
#include <cstring>
#include "optim/optim.h"
#include <vector>

int main() {
  auto options = tide::optim::Options::lbfgs(2);
  options.max_iter = 20;
  options.gtol_abs = 1e-10;
  options.line_search = tide::optim::LineSearchPolicy::HagerZhang;

  tide::optim::Objective objective{};
  objective.value_gradient = [](tide::optim::VectorView x,
                                tide::optim::MutableVectorView gradient) {
    gradient.data[0] = x.data[0];
    gradient.data[1] = x.data[1];
    return 0.5 * (x.data[0] * x.data[0] + x.data[1] * x.data[1]);
  };

  std::vector<double> x0{2.0, -3.0};
  int callbacks = 0;
  tide::optim::TraceOptions trace_options{};
  trace_options.policy = tide::optim::TracePolicy::Stride;
  trace_options.stride = 2;
  tide::optim::Result result = tide::optim::minimize(
      options, x0, objective, {}, trace_options,
      [&callbacks](tide::optim::ReportView report) {
        assert(report.direction_policy() == tide::optim::DirectionPolicy::Lbfgs);
        callbacks += 1;
        return false;
      });

  assert(result.success);
  assert(result.status == tide::optim::Status::ConvergedGradient);
  assert(std::strcmp(tide::optim::name(result.status), "CONVERGED_GRADIENT") == 0);
  assert(std::strcmp(result.status_name(), "CONVERGED_GRADIENT") == 0);
  assert(std::strcmp(result.reason(), "CONVERGED_GRADIENT") == 0);
  assert(result.failure_reason() == nullptr);
  assert(result.report_view().done());
  assert(std::strcmp(result.report_view().request_name(), "DONE") == 0);
  assert(std::strcmp(result.report_view().method_name(), "lbfgs") == 0);
  assert(std::strcmp(result.report_view().direction_policy_name(), "LBFGS") == 0);
  assert(std::strcmp(result.report_view().line_search_policy_name(), "HAGER_ZHANG") == 0);
  assert(std::strcmp(result.method_name(), "lbfgs") == 0);
  assert(result.event_counts().line_search_accept >= 1);
  assert(result.n_iter() >= 1);
  assert(result.n_f() >= 1);
  assert(result.n_g() >= 1);
  assert(callbacks == result.n_trace_events);
  assert(result.trace_policy == tide::optim::TracePolicy::Stride);
  assert(std::strcmp(result.trace_policy_name(), "STRIDE") == 0);
  assert(result.trace_stride == 2);
  assert(result.config_fingerprint.size() == 16);
  assert(result.config_signature.find("schema=tide.optim.config.v1") != std::string::npos);
  assert(result.n_trace_events >= 1);
  assert(result.n_trace_stored() >= 1);
  assert(result.n_trace_stored() <= result.n_trace_events);
  assert(result.last_trace_view().done());
  assert(result.trace_view(0));
  assert(std::abs(result.x[0]) < 1e-8);
  assert(std::abs(result.x[1]) < 1e-8);
  assert(result.f() < 1e-16);

  tide::optim::Session direct_session(options);
  assert(direct_session.valid());
  assert(!direct_session.started());
  assert(std::strcmp(direct_session.state_name(), "NOT_STARTED") == 0);
  std::vector<double> direct_g0(2, 0.0);
  double direct_f0 = objective.value_gradient(
      tide::optim::view(x0), tide::optim::view(direct_g0));
  tide::optim::Request direct_request = direct_session.start(
      tide::optim::view(x0), direct_f0, tide::optim::view(direct_g0));
  assert(direct_session.started());
  assert(direct_session.running());
  assert(std::strcmp(direct_session.state_name(), "RUNNING") == 0);
  while (!direct_request.done()) {
    assert(direct_request.kind == tide::optim::RequestKind::EvaluateFG);
    std::vector<double> direct_g(2, 0.0);
    double direct_f = objective.value_gradient(
        direct_request.x, tide::optim::view(direct_g));
    direct_request = direct_session.tell(direct_f, tide::optim::view(direct_g));
  }
  assert(direct_session.done());
  assert(!direct_session.running());
  assert(std::strcmp(direct_session.state_name(), "DONE") == 0);
  direct_session.close();
  assert(direct_session.closed());
  assert(std::strcmp(direct_session.state_name(), "CLOSED") == 0);

  int typed_callbacks = 0;
  tide::optim::TraceOptions stop_trace_options{};
  stop_trace_options.policy = tide::optim::TracePolicy::Last;
  tide::optim::Result stopped = tide::optim::minimize(
      options, x0, objective, {}, stop_trace_options,
      [&typed_callbacks](tide::optim::ReportView report)
          -> tide::optim::CallbackDecision {
        assert(report.status() == tide::optim::Status::Running);
        typed_callbacks += 1;
        return tide::optim::Status::UserStopped;
      });
  assert(typed_callbacks == 1);
  assert(!stopped.success);
  assert(stopped.status == tide::optim::Status::UserStopped);
  assert(std::strcmp(stopped.status_name(), "USER_STOPPED") == 0);
  assert(std::strcmp(stopped.failure_reason(), "USER_STOPPED") == 0);
  assert(stopped.trace_summary.n_reports == stopped.n_trace_events);
  assert(stopped.trace_summary.n_reports == 1);
  assert(stopped.trace_summary.status_count(
             tide::optim::Status::UserStopped) == 1);
  assert(stopped.trace_summary.status_count(tide::optim::Status::Running) == 0);
  assert(stopped.trace_summary.failure_reason_count("USER_STOPPED") == 1);
  assert(stopped.trace_summary.failure_reason_count("null") == 0);
  assert(stopped.trace_summary.user_stopped_count() == 1);
  assert(stopped.trace_summary.failed_count() == 0);
  assert(stopped.has_last_trace);
  assert(stopped.last_trace_view().status() == tide::optim::Status::UserStopped);
  assert(stopped.n_trace_stored() == 1);
  assert(stopped.trace_view(0).status() == tide::optim::Status::UserStopped);

  auto missing_prec_options = tide::optim::Options::plbfgs(2);
  missing_prec_options.max_iter = 20;
  missing_prec_options.gtol_abs = 1e-10;
  tide::optim::Result missing_prec = tide::optim::minimize(
      missing_prec_options, x0, objective, {}, stop_trace_options);
  assert(!missing_prec.success);
  assert(missing_prec.status == tide::optim::Status::InvalidArgument);
  assert(std::strcmp(missing_prec.failure_reason(), "INVALID_ARGUMENT") == 0);
  assert(missing_prec.report_view().request_kind() ==
         tide::optim::RequestKind::ApplyPreconditioner);
  assert(missing_prec.trace_summary.n_reports == missing_prec.n_trace_events);
  assert(missing_prec.trace_summary.n_reports == 1);
  assert(missing_prec.trace_summary.status_count(
             tide::optim::Status::InvalidArgument) == 1);
  assert(missing_prec.trace_summary.status_count(tide::optim::Status::Running) == 0);
  assert(missing_prec.trace_summary.failure_reason_count("INVALID_ARGUMENT") == 1);
  assert(missing_prec.trace_summary.failure_reason_count("null") == 0);
  assert(missing_prec.trace_summary.failed_count() == 1);
  assert(missing_prec.has_last_trace);
  assert(missing_prec.last_trace_view().status() ==
         tide::optim::Status::InvalidArgument);
  assert(missing_prec.n_trace_stored() == 1);
  assert(missing_prec.trace_view(0).status() ==
         tide::optim::Status::InvalidArgument);

  tide::optim::TraceOptions manual_trace_options{};
  manual_trace_options.policy = tide::optim::TracePolicy::Last;
  tide::optim::TelemetrySession manual(options, manual_trace_options);
  assert(manual.valid());
  assert(!manual.closed());
  assert(!manual.started());
  assert(!manual.done());
  assert(!manual.running());
  assert(std::strcmp(manual.state_name(), "NOT_STARTED") == 0);
  std::vector<double> manual_g0(2, 0.0);
  double manual_f0 = objective.value_gradient(
      tide::optim::view(x0), tide::optim::view(manual_g0));
  tide::optim::Request manual_request = manual.start(
      tide::optim::view(x0), manual_f0, tide::optim::view(manual_g0));
  assert(manual.started());
  assert(manual.running());
  assert(!manual.done());
  assert(std::strcmp(manual.state_name(), "RUNNING") == 0);
  int manual_reports = 1;
  while (!manual_request.done()) {
    assert(manual_request.report);
    assert(manual_request.kind == tide::optim::RequestKind::EvaluateFG);
    assert(manual_request.needs_value_gradient());
    assert(!manual_request.needs_vector_result());
    std::vector<double> g_request(2, 0.0);
    double f_request = objective.value_gradient(
        manual_request.x, tide::optim::view(g_request));
    manual_request = manual.tell(f_request, tide::optim::view(g_request));
    manual_reports += 1;
  }
  assert(manual_request.done());
  assert(manual.started());
  assert(manual.done());
  assert(!manual.running());
  assert(std::strcmp(manual.state_name(), "DONE") == 0);
  assert(!manual_request.needs_value());
  assert(!manual_request.needs_gradient());
  assert(!manual_request.needs_vector_result());
  assert(manual_request.has_x());
  assert(manual_request.x_size() == 2);
  assert(!manual_request.has_vector());
  assert(manual_request.vector_size() == 0);
  assert(manual_request.has_report());
  assert(std::strcmp(manual_request.kind_name(), "DONE") == 0);
  assert(manual.last_trace_view().done());
  assert(manual.trace_summary().request_count(tide::optim::RequestKind::Done) == 1);
  assert(manual.trace_summary().status_count(tide::optim::Status::ConvergedGradient) == 1);
  assert(manual.trace_summary().failure_reason_count("null") >= 1);
  assert(manual.n_trace_events() == manual_reports);
  assert(manual.n_trace_stored() == 1);
  assert(manual.trace().size() == 1);
  assert(manual.trace_view(0).done());
  tide::optim::Result manual_result = manual.result();
  assert(manual_result.success);
  assert(manual_result.status == tide::optim::Status::ConvergedGradient);
  assert(manual_result.n_trace_events == manual.n_trace_events());
  assert(manual_result.n_trace_stored() == manual.n_trace_stored());
  assert(manual_result.trace_summary.n_reports == manual.trace_summary().n_reports);
  assert(manual_result.config_fingerprint.size() == 16);
  assert(std::abs(manual_result.x[0]) < 1e-8);
  assert(std::abs(manual_result.x[1]) < 1e-8);

  tide::optim::TelemetrySession manual_stop(options, manual_trace_options);
  tide::optim::Request manual_stop_request = manual_stop.start(
      tide::optim::view(x0), manual_f0, tide::optim::view(manual_g0));
  assert(manual_stop_request.kind != tide::optim::RequestKind::Done);
  tide::optim::Result manual_stop_result = manual_stop.result(
      tide::optim::Status::UserStopped);
  assert(!manual_stop_result.success);
  assert(manual_stop_result.status == tide::optim::Status::UserStopped);
  assert(manual_stop_result.trace_summary.status_count(
             tide::optim::Status::UserStopped) == 1);
  assert(manual_stop_result.trace_summary.status_count(
             tide::optim::Status::Running) == 0);
  assert(manual_stop_result.trace_summary.failure_reason_count(
             "USER_STOPPED") == 1);
  assert(manual_stop_result.last_trace_view().status() ==
         tide::optim::Status::UserStopped);
  assert(manual_stop_result.n_trace_stored() == 1);

  auto invalid_manual_trace_options = manual_trace_options;
  invalid_manual_trace_options.stride = 0;
  tide::optim::TelemetrySession invalid_manual(
      options, invalid_manual_trace_options);
  assert(!invalid_manual.valid());
  assert(!invalid_manual.closed());
  assert(!invalid_manual.started());
  assert(!invalid_manual.done());
  assert(!invalid_manual.running());
  assert(std::strcmp(invalid_manual.state_name(), "INVALID") == 0);
  tide::optim::OptionsValidation invalid_trace_validation =
      invalid_manual.trace_options_validation();
  assert(invalid_trace_validation.code ==
         tide::optim::OptionsValidationCode::TraceStride);
  assert(std::strcmp(invalid_trace_validation.field, "trace_stride") == 0);
  assert(invalid_manual.clear_bounds() == tide::optim::Status::InvalidArgument);
  assert(invalid_manual.current_x(tide::optim::view(x0)) ==
         tide::optim::Status::InvalidArgument);
  tide::optim::Request invalid_manual_request = invalid_manual.start(
      tide::optim::view(x0), manual_f0, tide::optim::view(manual_g0));
  assert(invalid_manual_request.error());
  assert(!invalid_manual_request.has_report());
  tide::optim::Result invalid_manual_result = invalid_manual.result();
  assert(!invalid_manual_result.success);
  assert(invalid_manual_result.status == tide::optim::Status::InvalidArgument);
  assert(invalid_manual_result.n_trace_events == 0);
  assert(invalid_manual_result.n_trace_stored() == 0);
  return 0;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            tide.optim.get_include(),
            str(source),
            "-L",
            str(library_dir),
            "-Wl,-rpath," + str(library_dir),
            "-ltide_C",
            "-o",
            str(executable),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr


def test_cpp_minimize_driver_handles_preconditioner_and_hvp_requests(
    tmp_path: Path,
):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is not available")
    root = Path(__file__).resolve().parents[1]
    library_dir = root / "src" / "tide"
    library = library_dir / "libtide_C.so"
    if not library.exists():
        pytest.skip("native tide library is not built")

    source = tmp_path / "optim_minimize_pde_requests.cpp"
    executable = tmp_path / "optim_minimize_pde_requests"
    source.write_text(
        r'''
#include <cassert>
#include <cmath>
#include <cstring>
#include "optim/optim.h"
#include <vector>

namespace {

constexpr double kScale0 = 10.0;
constexpr double kScale1 = 1.0;
constexpr double kTarget0 = 1.0;
constexpr double kTarget1 = -2.0;

tide::optim::Objective scaled_quadratic_objective(int& n_prec, int& n_hvp) {
  tide::optim::Objective objective{};
  objective.value_gradient = [](tide::optim::VectorView x,
                                tide::optim::MutableVectorView gradient) {
    double const r0 = x.data[0] - kTarget0;
    double const r1 = x.data[1] - kTarget1;
    gradient.data[0] = kScale0 * r0;
    gradient.data[1] = kScale1 * r1;
    return 0.5 * (kScale0 * r0 * r0 + kScale1 * r1 * r1);
  };
  objective.preconditioner = [&n_prec](tide::optim::VectorView,
                                       tide::optim::VectorView vector,
                                       tide::optim::MutableVectorView out) {
    n_prec += 1;
    out.data[0] = vector.data[0] / kScale0;
    out.data[1] = vector.data[1] / kScale1;
  };
  objective.hessian_vector = [&n_hvp](tide::optim::VectorView,
                                      tide::optim::VectorView vector,
                                      tide::optim::MutableVectorView out) {
    n_hvp += 1;
    out.data[0] = kScale0 * vector.data[0];
    out.data[1] = kScale1 * vector.data[1];
  };
  return objective;
}

void assert_reaches_target(tide::optim::Result const& result) {
  assert(result.success);
  assert(std::abs(result.x[0] - kTarget0) < 1e-8);
  assert(std::abs(result.x[1] - kTarget1) < 1e-8);
  assert(result.f() < 1e-16);
}

}  // namespace

int main() {
  std::vector<double> x0{8.0, -6.0};

  int plbfgs_prec = 0;
  int plbfgs_hvp = 0;
  auto plbfgs_options = tide::optim::Options::plbfgs(2);
  plbfgs_options.max_iter = 20;
  plbfgs_options.gtol_abs = 1e-8;
  auto plbfgs_result = tide::optim::minimize(
      plbfgs_options, x0, scaled_quadratic_objective(plbfgs_prec, plbfgs_hvp));
  assert_reaches_target(plbfgs_result);
  assert(plbfgs_prec == plbfgs_result.n_prec());
  assert(plbfgs_hvp == 0);
  assert(plbfgs_result.n_prec() > 0);
  assert(std::strcmp(plbfgs_result.report_view().direction_policy_name(),
                     "PRECONDITIONED_LBFGS") == 0);
  assert(plbfgs_result.event_counts().preconditioner_skip == 0);
  assert(plbfgs_result.event_counts().pair_stored > 0);
  bool saw_plbfgs_preconditioner = false;
  for (std::size_t i = 0; i < plbfgs_result.trace.size(); ++i) {
    auto trace = plbfgs_result.trace_view(i);
    if (trace.request_kind() == tide::optim::RequestKind::ApplyPreconditioner) {
      saw_plbfgs_preconditioner = true;
    }
  }
  assert(saw_plbfgs_preconditioner);

  int ptrn_prec = 0;
  int ptrn_hvp = 0;
  auto ptrn_options = tide::optim::Options::ptrn(2);
  ptrn_options.max_iter = 20;
  ptrn_options.max_inner_iter = 4;
  ptrn_options.inner_rtol = 1e-12;
  ptrn_options.gtol_abs = 1e-10;
  auto ptrn_result = tide::optim::minimize(
      ptrn_options, x0, scaled_quadratic_objective(ptrn_prec, ptrn_hvp));
  assert_reaches_target(ptrn_result);
  assert(ptrn_prec == ptrn_result.n_prec());
  assert(ptrn_hvp == ptrn_result.n_hvp());
  assert(ptrn_result.n_prec() > 0);
  assert(ptrn_result.n_hvp() > 0);
  tide::optim::InnerSolveDiagnostics ptrn_inner =
      ptrn_result.inner_solve_diagnostics();
  assert(ptrn_inner.n_hvp == ptrn_result.n_hvp());
  assert(ptrn_inner.n_prec == ptrn_result.n_prec());
  assert(ptrn_inner.converged());
  assert(ptrn_inner.preconditioner_applied());
  assert(std::strcmp(ptrn_result.report_view().direction_policy_name(),
                     "PRECONDITIONED_TRUNCATED_NEWTON") == 0);
  bool saw_ptrn_preconditioner = false;
  bool saw_ptrn_hvp = false;
  bool saw_inner_solved = false;
  for (std::size_t i = 0; i < ptrn_result.trace.size(); ++i) {
    auto trace = ptrn_result.trace_view(i);
    if (trace.request_kind() == tide::optim::RequestKind::ApplyPreconditioner) {
      saw_ptrn_preconditioner = true;
    }
    if (trace.request_kind() == tide::optim::RequestKind::EvaluateHv) {
      saw_ptrn_hvp = true;
    }
    if (trace.inner_status() == tide::optim::InnerCgStatus::ForcingReached) {
      saw_inner_solved = true;
    }
  }
  assert(saw_ptrn_preconditioner);
  assert(saw_ptrn_hvp);
  assert(saw_inner_solved);
  return 0;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            tide.optim.get_include(),
            str(source),
            "-L",
            str(library_dir),
            "-Wl,-rpath," + str(library_dir),
            "-ltide_C",
            "-o",
            str(executable),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr


def test_cpp_minimize_driver_reports_projected_gradient_bounds(tmp_path: Path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is not available")
    root = Path(__file__).resolve().parents[1]
    library_dir = root / "src" / "tide"
    library = library_dir / "libtide_C.so"
    if not library.exists():
        pytest.skip("native tide library is not built")

    source = tmp_path / "optim_minimize_bounds_smoke.cpp"
    executable = tmp_path / "optim_minimize_bounds_smoke"
    source.write_text(
        r'''
#include <cassert>
#include <cmath>
#include <cstring>
#include "optim/optim.h"
#include <vector>

int main() {
  auto options = tide::optim::Options::pstd(1);
  options.bounds = tide::optim::BoundsStrategy::ProjectedGradient;
  options.line_search = tide::optim::LineSearchPolicy::ArmijoCubic;
  options.max_iter = 10;
  options.gtol_abs = 1e-12;
  options.initial_step = 1.0;

  tide::optim::Objective objective{};
  objective.value_gradient = [](tide::optim::VectorView x,
                                tide::optim::MutableVectorView gradient) {
    double const residual = x.data[0] + 1.0;
    gradient.data[0] = residual;
    return 0.5 * residual * residual;
  };
  objective.value = [](tide::optim::VectorView x) {
    double const residual = x.data[0] + 1.0;
    return 0.5 * residual * residual;
  };

  std::vector<double> x0{1.0};
  std::vector<double> lower{0.0};
  std::vector<double> upper{2.0};
  tide::optim::Bounds bounds{tide::optim::view(lower), tide::optim::view(upper)};
  tide::optim::Result result = tide::optim::minimize(options, x0, objective, bounds);

  assert(result.success);
  assert(result.status == tide::optim::Status::ConvergedGradient);
  assert(std::strcmp(result.status_name(), "CONVERGED_GRADIENT") == 0);
  assert(std::abs(result.x[0]) < 1e-12);
  assert(result.projected_grad_norm() == 0.0);
  tide::optim::BoundsDiagnostics bounds_diagnostics = result.bounds_diagnostics();
  assert(bounds_diagnostics.strategy == tide::optim::BoundsStrategy::ProjectedGradient);
  assert(std::strcmp(bounds_diagnostics.strategy_name(), "PROJECTED_GRADIENT") == 0);
  assert(bounds_diagnostics.projected_grad_norm == 0.0);
  assert(bounds_diagnostics.active_count() == 1);
  assert(bounds_diagnostics.has_active_bounds());
  assert(!bounds_diagnostics.has_kkt_violations());
  assert(bounds_diagnostics.has_trial_projections());
  assert(result.active_lower_count() == 1);
  assert(result.active_upper_count() == 0);
  assert(result.free_count() == 0);
  assert(result.kkt_violation_count() == 0);
  assert(result.lower_kkt_violation_count() == 0);
  assert(result.upper_kkt_violation_count() == 0);
  assert(result.free_gradient_count() == 0);
  assert(result.trial_projection_count() >= 1);
  assert(result.trial_lower_projection_count() >= 1);
  assert(result.trial_upper_projection_count() == 0);
  assert(result.last_trace_view().active_lower_count() == 1);
  assert(result.last_trace_view().projected_grad_norm() == 0.0);
  assert(result.last_trace_view()
             .bounds_diagnostics(tide::optim::BoundsStrategy::ProjectedGradient)
             .active_count() == 1);
  return 0;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            tide.optim.get_include(),
            str(source),
            "-L",
            str(library_dir),
            "-Wl,-rpath," + str(library_dir),
            "-ltide_C",
            "-o",
            str(executable),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr


def test_cpp_minimize_with_object_objective_links_and_solves(tmp_path: Path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is not available")
    root = Path(__file__).resolve().parents[1]
    library_dir = root / "src" / "tide"
    library = library_dir / "libtide_C.so"
    if not library.exists():
        pytest.skip("native tide library is not built")

    source = tmp_path / "optim_minimize_with_object.cpp"
    executable = tmp_path / "optim_minimize_with_object"
    source.write_text(
        r'''
#include <cassert>
#include <cmath>
#include <cstring>
#include "optim/optim.h"
#include <vector>

struct ScaledQuadraticObjective {
  int n_value_gradient = 0;
  int n_value = 0;
  int n_prec = 0;
  int n_hvp = 0;

  double value_gradient(tide::optim::VectorView x,
                        tide::optim::MutableVectorView gradient) {
    n_value_gradient += 1;
    double const r0 = x.data[0] - 1.0;
    double const r1 = x.data[1] + 2.0;
    gradient.data[0] = 10.0 * r0;
    gradient.data[1] = r1;
    return 0.5 * (10.0 * r0 * r0 + r1 * r1);
  }

  double value(tide::optim::VectorView x) {
    n_value += 1;
    double const r0 = x.data[0] - 1.0;
    double const r1 = x.data[1] + 2.0;
    return 0.5 * (10.0 * r0 * r0 + r1 * r1);
  }

  void preconditioner(tide::optim::VectorView,
                      tide::optim::VectorView vector,
                      tide::optim::MutableVectorView out) {
    n_prec += 1;
    out.data[0] = vector.data[0] / 10.0;
    out.data[1] = vector.data[1];
  }

  void hessian_vector(tide::optim::VectorView,
                      tide::optim::VectorView vector,
                      tide::optim::MutableVectorView out) {
    n_hvp += 1;
    out.data[0] = 10.0 * vector.data[0];
    out.data[1] = vector.data[1];
  }
};

void assert_reaches_target(tide::optim::Result const& result) {
  assert(result.success);
  assert(std::abs(result.x[0] - 1.0) < 1e-8);
  assert(std::abs(result.x[1] + 2.0) < 1e-8);
}

int main() {
  std::vector<double> x0{8.0, -6.0};

  ScaledQuadraticObjective pstd_objective{};
  auto pstd_options = tide::optim::Options::pstd(2);
  pstd_options.line_search = tide::optim::LineSearchPolicy::ArmijoCubic;
  pstd_options.max_iter = 40;
  pstd_options.gtol_abs = 1e-8;
  auto pstd_result =
      tide::optim::minimize_with(pstd_options, x0, pstd_objective);
  assert_reaches_target(pstd_result);
  assert(std::strcmp(pstd_result.report_view().line_search_policy_name(),
                     "ARMIJO_CUBIC") == 0);
  assert(pstd_objective.n_value_gradient == pstd_result.n_g());
  assert(pstd_objective.n_value > 0);

  ScaledQuadraticObjective ptrn_objective{};
  auto ptrn_options = tide::optim::Options::ptrn(2);
  ptrn_options.max_iter = 20;
  ptrn_options.max_inner_iter = 4;
  ptrn_options.inner_rtol = 1e-12;
  ptrn_options.gtol_abs = 1e-10;
  auto ptrn_result =
      tide::optim::minimize_with(ptrn_options, x0, ptrn_objective);
  assert_reaches_target(ptrn_result);
  assert(ptrn_objective.n_value_gradient == ptrn_result.n_g());
  assert(ptrn_objective.n_prec == ptrn_result.n_prec());
  assert(ptrn_objective.n_hvp == ptrn_result.n_hvp());
  assert(ptrn_result.n_prec() > 0);
  assert(ptrn_result.n_hvp() > 0);
  return 0;
}
''',
        encoding="utf-8",
    )

    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            tide.optim.get_include(),
            str(source),
            "-L",
            str(library_dir),
            "-Wl,-rpath," + str(library_dir),
            "-ltide_C",
            "-o",
            str(executable),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(executable)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr


def test_optimizer_session_exposes_typed_reverse_communication_requests():
    target = np.array([2.0, -1.0], dtype=np.float64)
    x0 = np.array([8.0, -6.0], dtype=np.float64)

    def value(x: np.ndarray) -> float:
        residual = x - target
        return float(0.5 * np.dot(residual, residual))

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    seen: list[tide.optim.RequestKind] = []
    stale_request_checked = False
    with tide.optim.OptimizerSession(
        x0,
        options={
            "line_search": "armijo_cubic",
            "initial_step": 10.0,
            "trace_policy": "none",
            "gtol_abs": 1e-8,
            "max_iter": 30,
        },
    ) as session:
        np.testing.assert_allclose(session.current_x(), x0)
        assert session.method == "lbfgs"
        assert not session.started
        assert not session.closed
        assert not session.done
        assert not session.running
        assert session.state == "not_started"
        initial_session_payload = session.to_dict()
        assert initial_session_payload["state"] == "not_started"
        assert initial_session_payload["backend_state"] == "NOT_STARTED"
        initial_backend_snapshot = initial_session_payload["backend_snapshot"]
        assert initial_backend_snapshot["valid"] is True
        assert initial_backend_snapshot["state_name"] == "NOT_STARTED"
        assert initial_backend_snapshot["started"] is False
        assert initial_backend_snapshot["running"] is False
        assert initial_backend_snapshot["done"] is False
        assert initial_backend_snapshot["n"] == x0.size
        assert initial_backend_snapshot["report"]["request"] == "ERROR"
        assert initial_session_payload["last_request"] is None
        assert initial_session_payload["current_request"] is None
        assert session.current_request is None
        assert initial_session_payload["n_trace_events"] == 0
        assert session.effective_options["line_search"] == "armijo_cubic"
        assert session.backend_options_payload["n"] == x0.size
        assert session.backend_options_payload["line_search_policy"] == "ARMIJO_CUBIC"
        assert isinstance(session.policy_resolution, tide.optim.ResolvedPolicies)
        assert session.policy_resolution.line_search == (
            tide.optim.LineSearchPolicy.ARMIJO_CUBIC
        )
        assert session.resolved_policies == session.policy_resolution.to_dict()
        assert len(session.config_fingerprint) == 64
        assert session.config_signature["schema"] == "tide.optim.config.v1"
        assert session.config_signature["method"] == "lbfgs"
        assert session.config_signature["backend_options"]["n"] == x0.size
        assert session.diagnostics["request_counts"] == {}
        f0, g0 = value_grad(session.current_x())
        request = session.start(f0, g0)
        assert session.started
        assert not session.closed
        assert not session.done
        assert session.running
        assert session.state == "running"
        running_session_payload = session.to_dict()
        assert running_session_payload["state"] == "running"
        assert running_session_payload["backend_state"] == "RUNNING"
        running_backend_snapshot = running_session_payload["backend_snapshot"]
        assert running_backend_snapshot["valid"] is True
        assert running_backend_snapshot["started"] is True
        assert running_backend_snapshot["running"] is True
        assert running_backend_snapshot["done"] is False
        assert running_backend_snapshot["report"]["request"] == request.kind.name
        assert running_backend_snapshot["report"]["needs_value"] is True
        assert running_backend_snapshot["report"]["needs_gradient"] is request.needs_gradient
        assert running_backend_snapshot["report"]["needs_vector_result"] is False
        assert running_session_payload["last_request"] == request.kind.name
        assert running_session_payload["current_request"] is not None
        assert running_session_payload["current_request"]["kind"] == request.kind.name
        assert running_session_payload["current_request"]["sequence"] == request.sequence
        session_request = session.current_request
        assert session_request is not None
        assert session_request.kind == request.kind
        assert session_request.sequence == request.sequence
        np.testing.assert_allclose(session_request.x, request.x)
        assert session_request.expected_gradient_shape == request.expected_gradient_shape
        assert session_request.expected_vector_shape == request.expected_vector_shape
        assert running_session_payload["n_trace_events"] == 1
        assert isinstance(request, tide.optim.OptimizerRequest)
        assert isinstance(
            request.requirements,
            tide.optim.OptimizerRequestRequirements,
        )
        assert request.sequence == 1
        assert request.requirements.kind == request.kind
        assert request.requirements.sequence == request.sequence
        assert request.requirements.requires_evaluation
        assert request.requirements.to_dict()["kind"] == request.kind.name
        assert request.required_fields == request.requirements.required_fields
        assert request.accepted_mapping_keys == (
            request.requirements.accepted_mapping_keys
        )
        empty_evaluation = tide.optim.OptimizerEvaluation()
        assert request.requirements.missing_from(empty_evaluation) == (
            request.required_fields
        )
        assert not request.requirements.satisfied_by(empty_evaluation)
        assert empty_evaluation.to_dict() == {
            "has_value": False,
            "has_gradient": False,
            "has_vector": False,
            "gradient_shape": None,
            "vector_shape": None,
        }
        empty_status = request.evaluation_status(empty_evaluation)
        assert isinstance(empty_status, tide.optim.OptimizerEvaluationStatus)
        assert empty_status.requirements == request.requirements
        assert empty_status.expected_evaluation == request.expected_evaluation
        assert empty_status.missing_fields == request.required_fields
        assert empty_status.mismatched_fields == ()
        assert empty_status.has_missing_fields
        assert not empty_status.has_shape_mismatch
        assert not empty_status.satisfied
        assert not empty_status.valid
        empty_status_payload = empty_status.to_dict()
        assert empty_status_payload["expected_evaluation"] == (
            request.expected_evaluation
        )
        assert empty_status_payload["missing_fields"] == list(
            request.required_fields
        )
        assert empty_status_payload["mismatched_fields"] == []
        assert empty_status_payload["satisfied"] is False
        with pytest.raises(ValueError, match=request.kind.name):
            session.respond(request, empty_evaluation)
        assert request.kind_name == request.kind.name
        assert not request.done
        assert not request.error
        request_payload = request.to_dict(include_trace=True)
        assert request_payload["kind"] == request.kind.name
        assert request_payload["sequence"] == request.sequence
        assert request_payload["done"] is False
        assert request_payload["error"] is False
        assert request_payload["x_shape"] == list(x0.shape)
        assert request_payload["vector_shape"] is None
        if request.needs_gradient:
            assert request_payload["expected_gradient_shape"] == list(x0.shape)
        else:
            assert request_payload["expected_gradient_shape"] is None
        assert request_payload["expected_vector_shape"] is None
        assert request_payload["has_vector"] is False
        assert request_payload["expected_evaluation"] == request.expected_evaluation
        assert request_payload["required_fields"] == list(request.required_fields)
        assert request_payload["requirements"]["kind"] == request.kind.name
        assert request_payload["requirements"]["sequence"] == request.sequence
        assert request_payload["trace"]["request"] == request.kind.name
        assert request_payload["trace"]["sequence"] == request.sequence
        while not request.done:
            seen.append(request.kind)
            assert isinstance(request.trace, tide.optim.TraceEntry)
            assert request.needs_value
            assert not request.needs_vector_result
            assert not request.has_vector
            assert request.x_shape == x0.shape
            assert request.vector_shape is None
            loop_session_request = session.current_request
            assert loop_session_request is not None
            assert loop_session_request.kind == request.kind
            assert loop_session_request.sequence == request.sequence
            np.testing.assert_allclose(loop_session_request.x, request.x)
            if request.needs_gradient:
                assert request.expected_gradient_shape == request.x_shape
            else:
                assert request.expected_gradient_shape is None
            assert request.expected_vector_shape is None
            if request.kind == tide.optim.RequestKind.EVALUATE_F:
                assert request.expected_evaluation == "value"
                assert request.required_fields == ("f",)
                assert request.accepted_mapping_keys == ("f",)
                assert not request.needs_gradient
                assert not request.needs_value_gradient
                assert request.vector is None
                assert request.x.shape == x0.shape
                f = value(request.x)
                evaluation = tide.optim.OptimizerEvaluation.value(f)
                assert request.requirements.satisfied_by({"f": f})
                assert request.satisfied_by({"f": f})
                assert request.requirements.validate({"f": f}).has_value
                assert request.validate_evaluation({"f": f}).has_value
                assert evaluation.has_value
                assert not evaluation.has_gradient
                assert evaluation.gradient_shape is None
                value_status = request.evaluation_status(evaluation)
                assert value_status.satisfied
                assert value_status.expected_gradient_shape is None
                assert value_status.expected_vector_shape is None
                assert not value_status.has_shape_mismatch
                previous_request = request
                request = session.respond(previous_request, evaluation)
                assert request.sequence > previous_request.sequence
                if not stale_request_checked:
                    with pytest.raises(RuntimeError, match="current optimizer request"):
                        session.respond(previous_request, evaluation)
                    stale_request_checked = True
            elif request.kind == tide.optim.RequestKind.EVALUATE_FG:
                assert request.expected_evaluation == "value_gradient"
                assert request.required_fields == ("f", "g")
                assert request.accepted_mapping_keys == (
                    "f",
                    "g",
                    "gradient",
                )
                assert request.needs_gradient
                assert request.needs_value_gradient
                assert request.expected_gradient_shape == request.x_shape
                assert request.expected_vector_shape is None
                assert request.vector is None
                f, g = value_grad(request.x)
                assert request.requirements.missing_from({"f": f}) == ("g",)
                assert request.requirements.satisfied_by({"f": f, "gradient": g})
                short_g = np.zeros(max(1, request.x.size - 1), dtype=np.float64)
                if short_g.shape == request.x_shape:
                    short_g = np.zeros(request.x.size + 1, dtype=np.float64)
                short_evaluation = tide.optim.OptimizerEvaluation.value_gradient(
                    f,
                    short_g,
                )
                short_status = request.evaluation_status(short_evaluation)
                assert short_status.missing_fields == ()
                assert short_status.mismatched_fields == ("g",)
                assert short_status.has_shape_mismatch
                assert short_status.gradient_shape == short_g.shape
                assert short_status.expected_gradient_shape == request.x_shape
                assert not short_status.satisfied
                assert not request.satisfied_by(short_evaluation)
                with pytest.raises(ValueError, match="mismatched"):
                    request.validate_evaluation(short_evaluation)
                with pytest.raises(ValueError, match="mismatched"):
                    session.respond(request, short_evaluation)
                evaluation = tide.optim.OptimizerEvaluation.value_gradient(f, g)
                assert evaluation.has_value
                assert evaluation.has_gradient
                assert evaluation.gradient_shape == x0.shape
                evaluation_status = request.evaluation_status(evaluation)
                assert evaluation_status.satisfied
                assert evaluation_status.mismatched_fields == ()
                assert evaluation_status.expected_gradient_shape == request.x_shape
                previous_request = request
                request = session.tell_evaluation(previous_request, evaluation)
                assert request.sequence > previous_request.sequence
                if not stale_request_checked:
                    with pytest.raises(RuntimeError, match="current optimizer request"):
                        session.respond(previous_request, evaluation)
                    stale_request_checked = True
            else:
                pytest.fail(f"unexpected request kind {request.kind!r}")
        assert request.done
        assert session.started
        assert not session.closed
        assert session.done
        assert not session.running
        assert session.state == "done"
        done_session_payload = session.to_dict()
        assert done_session_payload["state"] == "done"
        assert done_session_payload["backend_state"] == "DONE"
        assert done_session_payload["current_request"] is not None
        assert done_session_payload["current_request"]["kind"] == "DONE"
        assert session.current_request is not None
        assert session.current_request.done
        done_backend_snapshot = done_session_payload["backend_snapshot"]
        assert done_backend_snapshot["valid"] is True
        assert done_backend_snapshot["started"] is True
        assert done_backend_snapshot["running"] is False
        assert done_backend_snapshot["done"] is True
        assert done_backend_snapshot["report"]["request"] == "DONE"
        assert done_backend_snapshot["report"]["success"] is True
        assert done_session_payload["last_request"] == "DONE"
        assert request.kind_name == "DONE"
        assert request.sequence == session.trace_summary.n_reports
        assert stale_request_checked
        assert request.expected_evaluation == "none"
        assert request.required_fields == ()
        assert request.accepted_mapping_keys == ()
        assert not request.requirements.requires_evaluation
        assert not request.requirements.satisfied_by({})
        with pytest.raises(RuntimeError, match="DONE does not require"):
            request.requirements.validate({})
        assert not request.needs_value
        assert not request.needs_gradient
        assert not request.needs_vector_result
        done_payload = request.to_dict()
        assert done_payload["kind"] == "DONE"
        assert done_payload["done"] is True
        assert done_payload["x_shape"] == list(x0.shape)
        assert done_payload["vector_shape"] is None
        summary = session.trace_summary
        assert isinstance(summary, tide.optim.TraceSummary)
        assert summary.request_count(tide.optim.RequestKind.DONE) == 1
        assert summary.request_count(tide.optim.RequestKind.EVALUATE_F) > 0
        assert summary.line_search_rejection_count > 0
        assert session.diagnostics["request_counts"]["DONE"] == 1
        assert session.diagnostics["line_search_rejection_count"] > 0
        result = session.result()
        assert result.config_fingerprint == session.config_fingerprint
        assert result.config_signature == session.config_signature
        assert result.policy_resolution == session.policy_resolution
        assert result.diagnostics == session.diagnostics

    assert session.closed
    assert session.state == "closed"
    closed_payload = session.to_dict()
    assert closed_payload["state"] == "closed"
    assert closed_payload["backend_state"] == "INVALID"
    assert closed_payload["backend_snapshot"]["valid"] is False
    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-8)
    assert tide.optim.RequestKind.EVALUATE_F in seen
    assert tide.optim.RequestKind.EVALUATE_FG in seen
    assert result.trace == ()
    assert result.last_trace is not None
    assert result.last_trace.request == tide.optim.RequestKind.DONE
    assert result.diagnostics["line_search_rejection_count"] > 0


def test_optimizer_session_infers_method_from_options_direction():
    target = np.array([2.0], dtype=np.float64)
    x0 = np.array([8.0], dtype=np.float64)

    def value(x: np.ndarray) -> float:
        residual = x - target
        return float(0.5 * np.dot(residual, residual))

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    options = tide.optim.OptimizerOptions.for_method(
        "pnlcg",
        max_iter=10,
        gtol_abs=1e-10,
        trace_policy="last",
    )
    with tide.optim.OptimizerSession(x0, options=options) as session:
        assert session.method == "pnlcg"
        assert session.policy_resolution.method == "pnlcg"
        assert session.policy_resolution.direction == tide.optim.DirectionPolicy.NLCG
        assert session.resolved_policies["method"] == "pnlcg"
        assert session.config_signature["method"] == "pnlcg"
        assert session.backend_options_payload["direction_policy"] == "NLCG"
        f0, g0 = value_grad(session.current_x())
        request = session.start(f0, g0)
        while request.kind != tide.optim.RequestKind.DONE:
            if request.kind == tide.optim.RequestKind.EVALUATE_FG:
                f, g = value_grad(request.x)
                request = session.tell(f, g)
            elif request.kind == tide.optim.RequestKind.EVALUATE_F:
                request = session.tell_value(value(request.x))
            else:
                pytest.fail(f"unexpected request kind {request.kind!r}")
        result = session.result()

    assert result.success, result.reason
    assert result.method == "pnlcg"
    assert result.policy_resolution == session.policy_resolution
    assert result.config_signature == session.config_signature
    assert result.last_trace is not None
    assert result.last_trace.direction_policy == tide.optim.DirectionPolicy.NLCG
    np.testing.assert_allclose(result.x, target, atol=1e-10)


def test_optimizer_session_can_stop_manual_reverse_communication():
    target = np.array([2.0, -1.0], dtype=np.float64)
    x0 = np.array([8.0, -6.0], dtype=np.float64)

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    with tide.optim.OptimizerSession(
        x0,
        options={
            "store_trace": True,
            "trace_policy": "last",
            "gtol_abs": 0.0,
            "max_iter": 30,
        },
    ) as session:
        f0, g0 = value_grad(session.current_x())
        request = session.start(f0, g0)
        assert request.kind != tide.optim.RequestKind.DONE
        with pytest.raises(RuntimeError, match="Optimization is not done"):
            session.result()

        result = session.stop()

        assert result.config_fingerprint == session.config_fingerprint
        assert result.diagnostics == session.diagnostics

    assert result.status == tide.optim.OptimStatus.USER_STOPPED
    assert result.reason == "USER_STOPPED"
    assert not result.success
    assert result.last_trace is not None
    assert result.last_trace.status == tide.optim.OptimStatus.USER_STOPPED
    assert result.last_trace.request != tide.optim.RequestKind.DONE
    assert result.trace == (result.last_trace,)
    assert result.n_trace_events == 1
    assert result.trace_summary.n_reports == result.n_trace_events
    assert result.trace_summary.status_count(tide.optim.OptimStatus.USER_STOPPED) == 1
    assert result.trace_summary.failure_reason_count("USER_STOPPED") == 1
    assert result.trace_summary.success_count == 0
    assert result.trace_summary.failed_count == 0
    assert result.trace_summary.user_stopped_count == 1
    assert result.diagnostics["status_report_counts"] == {"USER_STOPPED": 1}
    assert result.diagnostics["failure_reason_report_counts"] == {
        "USER_STOPPED": 1
    }
    assert result.stopping_diagnostics.user_stopped
    assert not result.stopping_diagnostics.failed


def test_optimizer_session_handles_preconditioner_requests():
    scale = np.array([10.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -2.0], dtype=np.float64)
    x0 = np.array([8.0, -6.0], dtype=np.float64)

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    seen: list[tide.optim.RequestKind] = []
    with tide.optim.OptimizerSession(
        x0,
        method="plbfgs",
        options={
            "max_iter": 20,
            "gtol_abs": 1e-8,
            "trace_policy": "none",
        },
    ) as session:
        f0, g0 = value_grad(session.current_x())
        request = session.start(f0, g0)
        while request.kind != tide.optim.RequestKind.DONE:
            seen.append(request.kind)
            if request.kind == tide.optim.RequestKind.APPLY_PRECONDITIONER:
                assert request.expected_evaluation == "preconditioner"
                assert request.required_fields == ("vector",)
                assert request.accepted_mapping_keys == ("vector", "z")
                assert request.needs_preconditioner
                assert request.needs_vector_result
                assert not request.needs_value
                assert not request.needs_hessian_vector
                assert request.vector is not None
                assert request.has_vector
                assert request.x.shape == x0.shape
                assert request.vector.shape == x0.shape
                assert request.x_shape == x0.shape
                assert request.vector_shape == x0.shape
                vector_payload = request.to_dict()
                assert vector_payload["needs_preconditioner"] is True
                assert vector_payload["needs_hessian_vector"] is False
                assert vector_payload["needs_vector_result"] is True
                assert vector_payload["has_vector"] is True
                assert vector_payload["vector_shape"] == list(x0.shape)
                z = request.vector / scale
                evaluation = tide.optim.OptimizerEvaluation.preconditioner(z)
                assert request.requirements.satisfied_by({"z": z})
                assert evaluation.has_vector
                assert evaluation.vector_shape == x0.shape
                request = session.respond(request, evaluation)
            elif request.kind == tide.optim.RequestKind.EVALUATE_FG:
                assert request.needs_value_gradient
                assert not request.needs_vector_result
                assert request.vector is None
                assert not request.has_vector
                assert request.vector_shape is None
                f, g = value_grad(request.x)
                request = session.tell(f, g)
            else:
                pytest.fail(f"unexpected request kind {request.kind!r}")
        result = session.result()

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert tide.optim.RequestKind.APPLY_PRECONDITIONER in seen
    assert result.n_prec > 0
    assert result.diagnostics["backend_event_counts"]["preconditioner_skip"] == 0
    assert result.diagnostics["pair_stored_count"] > 0


def test_optimizer_session_handles_hvp_and_inner_preconditioner_requests():
    scale = np.array([10.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -2.0], dtype=np.float64)
    x0 = np.array([8.0, -6.0], dtype=np.float64)

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    seen: list[tide.optim.RequestKind] = []
    with tide.optim.OptimizerSession(
        x0,
        method="ptrn",
        options={
            "max_iter": 20,
            "max_inner_iter": 4,
            "inner_rtol": 1e-12,
            "gtol_abs": 1e-10,
            "trace_policy": "none",
        },
    ) as session:
        f0, g0 = value_grad(session.current_x())
        request = session.start(f0, g0)
        while request.kind != tide.optim.RequestKind.DONE:
            seen.append(request.kind)
            if request.kind == tide.optim.RequestKind.APPLY_PRECONDITIONER:
                assert request.expected_evaluation == "preconditioner"
                assert request.required_fields == ("vector",)
                assert request.accepted_mapping_keys == ("vector", "z")
                assert request.needs_preconditioner
                assert request.needs_vector_result
                assert not request.needs_hessian_vector
                assert request.vector is not None
                assert request.has_vector
                assert request.x_shape == x0.shape
                assert request.vector_shape == x0.shape
                vector_payload = request.to_dict()
                assert vector_payload["needs_preconditioner"] is True
                assert vector_payload["needs_vector_result"] is True
                assert vector_payload["vector_shape"] == list(x0.shape)
                request = session.respond(request, {"z": request.vector / scale})
            elif request.kind == tide.optim.RequestKind.EVALUATE_HV:
                assert request.expected_evaluation == "hessian_vector"
                assert request.required_fields == ("vector",)
                assert request.accepted_mapping_keys == ("vector", "hv")
                assert request.needs_hessian_vector
                assert request.needs_vector_result
                assert not request.needs_preconditioner
                assert request.vector is not None
                assert request.has_vector
                assert request.x.shape == x0.shape
                assert request.vector.shape == x0.shape
                assert request.x_shape == x0.shape
                assert request.vector_shape == x0.shape
                hvp_payload = request.to_dict()
                assert hvp_payload["needs_preconditioner"] is False
                assert hvp_payload["needs_hessian_vector"] is True
                assert hvp_payload["needs_vector_result"] is True
                assert hvp_payload["has_vector"] is True
                assert hvp_payload["vector_shape"] == list(x0.shape)
                hv = scale * request.vector
                assert request.requirements.satisfied_by({"hv": hv})
                request = session.respond(request, {"hv": hv})
            elif request.kind == tide.optim.RequestKind.EVALUATE_FG:
                assert request.needs_value_gradient
                assert not request.needs_vector_result
                assert request.vector is None
                f, g = value_grad(request.x)
                request = session.tell(f, g)
            else:
                pytest.fail(f"unexpected request kind {request.kind!r}")
        result = session.result()

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert tide.optim.RequestKind.APPLY_PRECONDITIONER in seen
    assert tide.optim.RequestKind.EVALUATE_HV in seen
    assert result.n_prec > 0
    assert result.n_hvp > 0
    assert result.diagnostics["inner_warning_count"] == 0


def test_optim_benchmark_smoke_payload_is_structured_json():
    benchmark = _load_optim_benchmark_module()

    payload = benchmark.run_benchmark(
        quick=True,
        case_names=("rosenbrock",),
        config_names=("lbfgs_hager_zhang",),
    )
    text = json.dumps(payload, sort_keys=True)
    result = payload["results"][0]

    assert '"benchmark": "tide.optim"' in text
    assert payload["schema_version"] == 1
    assert payload["metadata"]["runner"] == "python"
    assert payload["metadata"]["config_schema"] == "tide.optim.config.v1"
    assert payload["metadata"]["packages"]["tide"] == tide.__version__
    assert payload["metadata"]["packages"]["numpy"] == np.__version__
    manifest = payload["experiment_manifest"]
    assert manifest["schema"] == "tide.optim.benchmark.manifest.v1"
    assert manifest["benchmark"] == "tide.optim"
    assert manifest["config_schema"] == "tide.optim.config.v1"
    assert manifest["runner"] == "python"
    assert manifest["quick"] is True
    assert manifest["score_order"] == [
        "f",
        "projected_grad_norm",
        "weighted_request_units",
        "n_g",
        "n_hvp",
        "n_f",
        "n_prec",
    ]
    assert manifest["cost_model_weights"]["balanced"] == (
        tide.optim.cost_model_weights("balanced")
    )
    assert manifest["policy_catalog"] == payload["metadata"]["policy_catalog"]
    assert manifest["default_policy_resolution"] == (
        payload["metadata"]["default_policy_resolution"]
    )
    assert manifest["cases"] == [
        {
            "name": "rosenbrock",
            "n": 2,
            "shape": [2],
            "has_value": True,
            "has_hessian_vector": True,
            "has_bounds": False,
            "options": {"gtol_abs": 1e-7},
        }
    ]
    assert manifest["configs"] == [
        {
            "name": "lbfgs_hager_zhang",
            "method": "lbfgs",
            "line_search": "hager_zhang",
            "options": {"history_size": 10},
            "use_identity_preconditioner": False,
        }
    ]
    assert manifest["run_matrix"] == [
        {"case": "rosenbrock", "config": "lbfgs_hager_zhang"}
    ]
    assert payload["metadata"]["numeric"]["dtype"] == "float64"
    assert payload["metadata"]["numeric"]["sizeof_float64"] == 8
    assert payload["metadata"]["numeric"]["float64_bits"] == 64
    assert payload["metadata"]["numeric"]["float64_eps"] == float(np.finfo(np.float64).eps)
    assert payload["metadata"]["numeric"]["byteorder"] in {"little", "big"}
    assert payload["metadata"]["backend"]["available"] is True
    assert payload["metadata"]["backend"]["library_path"]
    assert payload["metadata"]["policy_catalog"]["line_search"] == list(
        tide.optim.SUPPORTED_LINE_SEARCHES
    )
    assert payload["metadata"]["policy_catalog"]["globalization"] == list(
        tide.optim.SUPPORTED_GLOBALIZATIONS
    )
    default_policies = payload["metadata"]["default_policy_resolution"]
    assert default_policies["balanced"]["lbfgs"]["line_search_policy"] == (
        "HAGER_ZHANG"
    )
    assert default_policies["expensive_gradient"]["lbfgs"][
        "line_search_policy"
    ] == "ARMIJO_CUBIC"
    assert default_policies["balanced"]["pnlcg"]["direction_policy"] == "NLCG"
    assert isinstance(payload["metadata"]["git"]["dirty"], bool)
    assert isinstance(payload["metadata"]["git"]["status_count"], int)
    assert payload["quick"] is True
    assert payload["cases"] == ["rosenbrock"]
    assert payload["configs"] == ["lbfgs_hager_zhang"]
    assert payload["summary"]["total_runs"] == 1
    assert payload["summary"]["success_count"] == 1
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["status_counts"] == {"CONVERGED_GRADIENT": 1}
    assert payload["summary"]["failure_reason_counts"] == {"null": 1}
    assert payload["summary"]["by_case"]["rosenbrock"]["runs"] == 1
    assert payload["summary"]["by_case"]["rosenbrock"]["successes"] == 1
    assert payload["summary"]["by_config"]["lbfgs_hager_zhang"]["runs"] == 1
    assert payload["summary"]["by_method"]["lbfgs"]["runs"] == 1
    assert payload["summary"]["by_method"]["lbfgs"]["successes"] == 1
    assert payload["summary"]["by_line_search"]["hager_zhang"]["runs"] == 1
    assert payload["summary"]["by_line_search_policy"]["HAGER_ZHANG"]["runs"] == 1
    assert payload["summary"]["by_globalization_policy"]["LINE_SEARCH"]["runs"] == 1
    assert payload["summary"]["by_bounds_strategy_policy"]["NONE"]["runs"] == 1
    assert payload["summary"]["by_cost_model_policy"]["BALANCED"]["runs"] == 1
    assert payload["summary"]["diagnostic_totals"]["line_search_failures"] == 0
    assert (
        payload["summary"]["diagnostic_totals"]["line_search_rejections"]
        == result["diagnostics"]["line_search_rejection_count"]
    )
    trace_totals = payload["summary"]["trace_summary_totals"]
    assert trace_totals["n_reports"] == result["trace_summary"]["n_reports"]
    assert trace_totals["request_counts"]["DONE"] == 1
    assert trace_totals["expected_evaluation_counts"] == (
        result["trace_summary"]["expected_evaluation_counts"]
    )
    for key in (
        "expected_gradient_requests",
        "expected_vector_requests",
        "expected_gradient_elements",
        "expected_vector_elements",
        "expected_total_vector_elements",
    ):
        assert trace_totals[key] == result["trace_summary"][key]
    assert trace_totals["expected_evaluation_counts"]["none"] == 1
    assert trace_totals["expected_evaluation_counts"]["value_gradient"] > 0
    cost_totals = payload["summary"]["cost_estimate_totals"]
    assert cost_totals["weighted_request_units"] == pytest.approx(
        result["cost_estimate"]["weighted_request_units"]
    )
    assert cost_totals["expected_gradient_elements"] == (
        result["cost_estimate"]["expected_gradient_elements"]
    )
    assert cost_totals["expected_vector_elements"] == (
        result["cost_estimate"]["expected_vector_elements"]
    )
    assert cost_totals["expected_total_vector_elements"] == (
        result["cost_estimate"]["expected_total_vector_elements"]
    )
    assert cost_totals["expected_vector_passes"] == pytest.approx(
        result["cost_estimate"]["expected_vector_passes"]
    )
    assert trace_totals["status_counts"]["CONVERGED_GRADIENT"] == 1
    assert trace_totals["failure_reason_counts"]["null"] >= 1
    assert trace_totals["success_count"] == 1
    assert trace_totals["failed_count"] == 0
    assert (
        payload["summary"]["by_case"]["rosenbrock"]["trace_summary_totals"]
        == trace_totals
    )
    assert (
        payload["summary"]["best_success_by_case"]["rosenbrock"]["config"]
        == "lbfgs_hager_zhang"
    )
    assert result["case"] == "rosenbrock"
    assert result["config"] == "lbfgs_hager_zhang"
    assert result["method"] == "lbfgs"
    assert result["effective_options"]["history_size"] == 10
    assert result["effective_options"]["trace_policy"] == "none"
    assert result["backend_options"]["line_search_policy"] == "HAGER_ZHANG"
    assert result["backend_options"]["direction_policy"] == "LBFGS"
    assert result["resolved_policies"]["method"] == "lbfgs"
    assert result["resolved_policies"]["cost_model"] == "balanced"
    assert result["resolved_policies"]["line_search_policy"] == "HAGER_ZHANG"
    assert result["resolved_policies"]["direction_policy"] == "LBFGS"
    assert result["policy_resolution"] == result["resolved_policies"]
    assert len(result["config_fingerprint"]) == 64
    assert result["config_signature"]["schema"] == "tide.optim.config.v1"
    assert result["config_signature"]["backend_options"]["n"] == 2
    assert (
        result["config_signature"]["policy_resolution"]
        == result["policy_resolution"]
    )
    best_rosenbrock = payload["summary"]["best_success_by_case"]["rosenbrock"]
    assert best_rosenbrock["config_fingerprint"] == result["config_fingerprint"]
    assert best_rosenbrock["method"] == "lbfgs"
    assert best_rosenbrock["line_search_policy"] == "HAGER_ZHANG"
    assert best_rosenbrock["globalization_policy"] == "LINE_SEARCH"
    assert best_rosenbrock["bounds_strategy_policy"] == "NONE"
    assert best_rosenbrock["cost_model_policy"] == "BALANCED"
    assert best_rosenbrock["resolved_policies"] == result["resolved_policies"]
    assert result["line_search_policy"] == "HAGER_ZHANG"
    assert result["globalization_policy"] == "LINE_SEARCH"
    assert result["bounds_strategy_policy"] == "NONE"
    assert result["cost_model"] == "balanced"
    assert result["cost_model_policy"] == "BALANCED"
    assert result["alpha_guess"] == "initial"
    assert result["alpha_guess_policy"] == "INITIAL"
    assert result["lbfgs_update"] == "skip"
    assert result["lbfgs_update_policy"] == "SKIP"
    assert result["nlcg_beta"] == "dai_yuan"
    assert result["nlcg_beta_policy"] == "DAI_YUAN"
    assert result["stopping"] == "standard"
    assert result["stopping_policy"] == "STANDARD"
    assert result["trace_policy"] == "none"
    assert result["n_trace_events"] > 0
    assert result["n_trace_stored"] == 0
    assert result["trace_summary"]["n_reports"] == result["n_trace_events"]
    assert result["trace_summary"]["request_counts"]["DONE"] == 1
    assert result["trace_summary"]["expected_evaluation_counts"]["none"] == 1
    assert (
        result["trace_summary"]["expected_evaluation_counts"]["value_gradient"]
        > 0
    )
    assert result["trace_summary"]["expected_gradient_requests"] == (
        result["trace_summary"]["expected_evaluation_counts"]["value_gradient"]
    )
    assert result["trace_summary"]["expected_gradient_elements"] == (
        result["trace_summary"]["expected_gradient_requests"] * result["n"]
    )
    assert result["trace_summary"]["expected_total_vector_elements"] == (
        result["trace_summary"]["expected_gradient_elements"]
        + result["trace_summary"]["expected_vector_elements"]
    )
    assert (
        result["trace_summary"]["line_search_status_counts"]["ACCEPTED"]
        > 0
    )
    assert result["trace_summary"]["backend_event_counts"] == (
        result["diagnostics"]["backend_event_counts"]
    )
    assert result["n_f"] >= result["n_g"]
    assert result["evaluation_profile"]["n_f"] == result["n_f"]
    assert result["evaluation_profile"]["n_g"] == result["n_g"]
    assert result["evaluation_profile"]["n_value_gradient"] == result["n_g"]
    assert result["evaluation_profile"]["n_value_only"] == (
        result["n_f"] - result["n_g"]
    )
    assert result["evaluation_profile"]["n_total_requests"] == (
        result["evaluation_profile"]["n_value_only"]
        + result["n_g"]
        + result["n_hvp"]
        + result["n_prec"]
    )
    assert result["cost_estimate"]["cost_model"] == "balanced"
    assert result["cost_estimate"]["n"] == result["n"]
    assert result["cost_estimate"]["weights"]["value_gradient"] == 2.0
    assert result["cost_estimate"]["weighted_request_units"] == pytest.approx(
        result["cost_estimate"]["n_value_only"]
        + 2.0 * result["cost_estimate"]["n_value_gradient"]
        + 2.0 * result["cost_estimate"]["n_hvp"]
        + 0.25 * result["cost_estimate"]["n_prec"]
    )
    assert result["cost_estimate"]["expected_total_vector_elements"] == (
        result["trace_summary"]["expected_total_vector_elements"]
    )
    assert payload["summary"]["evaluation_profile_totals"] == {
        key: value
        for key, value in result["evaluation_profile"].items()
        if key != "cost_model"
    }
    assert payload["summary"]["by_case"]["rosenbrock"][
        "evaluation_profile_totals"
    ] == payload["summary"]["evaluation_profile_totals"]
    assert result["n_iter"] >= 0
    assert isinstance(result["warnings"], list)


def test_cross_fwi_benchmark_presets_match_cross_common_geometry():
    benchmark = _load_cross_fwi_benchmark_module()

    common = benchmark.PRESETS["common"]
    ring = benchmark._boundary_points(
        common.ny, common.nx, margin=common.pml_width, n_side=common.n_side
    )
    eps, sigma = benchmark._build_cross_model(common.ny, common.nx)
    summary = benchmark._summary(
        [
            {
                "runner": "python_sotb_like",
                "success": False,
                "status": "MAX_ITER",
                "f": 1.0,
                "grad_norm": 2.0,
                "n_f": 1,
                "n_g": 1,
                "optimizer_wall_time_seconds": 10.0,
                "estimated_non_objective_seconds": 0.1,
                "objective": {
                    "value_grad_seconds": 9.9,
                    "total_pde_units": 2.0,
                },
            },
            {
                "runner": "tide_optim_backend",
                "success": False,
                "status": "MAX_ITER",
                "f": 1.0,
                "grad_norm": 2.0,
                "n_f": 1,
                "n_g": 1,
                "optimizer_wall_time_seconds": 20.0,
                "estimated_non_objective_seconds": 0.2,
                "objective": {
                    "value_grad_seconds": 19.8,
                    "total_pde_units": 2.0,
                },
            },
        ]
    )

    assert common.ny == 200
    assert common.nx == 200
    assert common.nt == 1200
    assert common.n_side == 30
    assert common.batch_size == 8
    assert ring.shape == (116, 2)
    assert eps.shape == (200, 200)
    assert sigma.shape == (200, 200)
    assert float(eps.min()) == 1.0
    assert float(eps.max()) == 9.0
    assert summary["python_over_backend_wall_time_ratio"] == pytest.approx(0.5)
    assert summary["python_over_backend_estimated_overhead_ratio"] == pytest.approx(
        0.5
    )


def test_sotb_python_prototype_benchmark_compares_matched_runners():
    benchmark = _load_sotb_prototype_benchmark_module()

    payload = benchmark.run_benchmark(quick=True, case_names=("quadratic_2d",))
    text = json.dumps(payload, sort_keys=True)
    results = {(item["case"], item["runner"]): item for item in payload["results"]}

    assert '"benchmark": "tide.optim.sotb_python_prototype"' in text
    assert payload["schema_version"] == 1
    assert payload["quick"] is True
    assert payload["cases"] == ["quadratic_2d"]
    assert payload["runners"] == ["python_sotb_like", "tide_optim_backend"]
    assert payload["summary"]["total_runs"] == 2
    assert payload["summary"]["failure_count"] == 0
    summary = payload["summary"]["by_case"]["quadratic_2d"]
    assert summary["both_success"] is True
    assert summary["runners"] == ["python_sotb_like", "tide_optim_backend"]
    assert summary["python_over_backend_wall_time_ratio"] is not None

    prototype = results[("quadratic_2d", "python_sotb_like")]
    backend = results[("quadratic_2d", "tide_optim_backend")]
    assert prototype["method"] == "lbfgs"
    assert backend["method"] == "lbfgs"
    assert prototype["line_search"] == "legacy_weak_wolfe"
    assert backend["line_search"] == "legacy_weak_wolfe"
    assert prototype["success"] is True
    assert backend["success"] is True
    assert prototype["status"] == "CONVERGED_GRADIENT"
    assert backend["status"] == "CONVERGED_GRADIENT"
    assert prototype["n_f"] > 0
    assert prototype["n_g"] > 0
    assert backend["n_f"] > 0
    assert backend["n_g"] > 0
    assert prototype["wall_time_seconds"] >= 0.0
    assert backend["wall_time_seconds"] >= 0.0
    assert prototype["f"] <= 1e-18
    assert backend["f"] <= 1e-18


def test_cpp_optim_benchmark_compiles_and_emits_diagnostics(tmp_path: Path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is not available")
    root = Path(__file__).resolve().parents[1]
    library_dir = root / "src" / "tide"
    library = library_dir / "libtide_C.so"
    if not library.exists():
        pytest.skip("native tide library is not built")

    source = root / "benchmarks" / "optim_cpp_benchmark.cpp"
    executable = tmp_path / "optim_cpp_benchmark"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            tide.optim.get_include(),
            str(source),
            "-L",
            str(library_dir),
            "-Wl,-rpath," + str(library_dir),
            "-ltide_C",
            "-o",
            str(executable),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    artifact_path = tmp_path / "optim_cpp_benchmark.json"
    run_result = subprocess.run(
        [str(executable), "--output", str(artifact_path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert run_result.returncode == 0, run_result.stderr
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == run_result.stdout

    payload = json.loads(run_result.stdout)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == payload
    results = {(item["case"], item["config"]): item for item in payload["results"]}

    assert payload["benchmark"] == "tide.optim.cpp"
    assert payload["schema_version"] == 1
    metadata = payload["metadata"]
    manifest = payload["experiment_manifest"]
    assert manifest["schema"] == "tide.optim.benchmark.manifest.v1"
    assert manifest["benchmark"] == "tide.optim.cpp"
    assert manifest["config_schema"] == "tide.optim.config.v1"
    assert manifest["runner"] == "cpp"
    assert manifest["quick"] is False
    assert manifest["score_order"] == [
        "f",
        "projected_grad_norm",
        "weighted_request_units",
        "n_g",
        "n_hvp",
        "n_f",
        "n_prec",
    ]
    assert manifest["cost_model_weights"]["BALANCED"] == {
        "value_only": 1.0,
        "value_gradient": 2.0,
        "hessian_vector": 2.0,
        "preconditioner": 0.25,
    }
    assert manifest["cases"][0]["name"] == "quadratic"
    assert manifest["cases"][0]["n"] == 2
    assert manifest["cases"][-1]["has_bounds"] is True
    assert manifest["configs"][0]["name"] == "lbfgs_hager_zhang"
    assert manifest["configs"][3]["use_identity_preconditioner"] is True
    assert manifest["run_matrix"] == [
        {"case": item["case"], "config": item["config"]}
        for item in payload["results"]
    ]
    metadata = payload["metadata"]
    assert metadata["backend"] == "cpp"
    assert metadata["config_schema"] == "tide.optim.config.v1"
    assert metadata["compiler"]["id"] in {"clang", "gcc", "msvc", "unknown"}
    assert isinstance(metadata["compiler"]["version"], str)
    assert metadata["compiler"]["version"]
    assert metadata["compiler"]["cplusplus"] >= 201703
    assert metadata["platform"]["os"] in {"linux", "macos", "windows", "unknown"}
    assert metadata["platform"]["architecture"] in {
        "x86_64",
        "aarch64",
        "x86",
        "unknown",
    }
    assert metadata["numeric"]["sizeof_double"] == 8
    assert metadata["numeric"]["sizeof_pointer"] in {4, 8}
    assert metadata["numeric"]["double_is_iec559"] is True
    assert metadata["numeric"]["double_epsilon"] == pytest.approx(
        float(np.finfo(np.float64).eps)
    )
    assert metadata["policy_catalog"]["line_search"] == [
        "HAGER_ZHANG",
        "LEGACY_WEAK_WOLFE",
        "ARMIJO_CUBIC",
        "STRONG_WOLFE",
        "MORE_THUENTE",
        "NONMONOTONE_ARMIJO",
        "STATIC",
    ]
    assert metadata["policy_catalog"]["globalization"] == [
        "LINE_SEARCH",
        "TRUST_REGION",
    ]
    default_policies = metadata["default_policy_resolution"]
    assert default_policies["BALANCED"]["lbfgs"]["line_search_policy"] == (
        "HAGER_ZHANG"
    )
    assert default_policies["EXPENSIVE_GRADIENT"]["lbfgs"][
        "line_search_policy"
    ] == "ARMIJO_CUBIC"
    assert default_policies["BALANCED"]["pnlcg"]["direction_policy"] == "NLCG"
    assert default_policies["BALANCED"]["pstd"]["line_search_policy"] == (
        "ARMIJO_CUBIC"
    )
    assert payload["cases"] == [
        "quadratic",
        "rosenbrock",
        "scaled_quadratic",
        "lower_bound_quadratic",
    ]
    assert payload["configs"] == [
        "lbfgs_hager_zhang",
        "lbfgs_more_thuente",
        "lbfgs_legacy_weak_wolfe",
        "ptrn_hager_zhang",
        "trn_trust_region",
        "pstd_projected_gradient",
    ]
    assert payload["summary"]["total_runs"] == 7
    assert payload["summary"]["success_count"] == 7
    assert payload["summary"]["failure_count"] == 0
    assert payload["summary"]["total_wall_time_seconds"] >= 0.0
    assert payload["summary"]["status_counts"] == {"CONVERGED_GRADIENT": 7}
    assert payload["summary"]["failure_reason_counts"] == {"null": 7}
    assert payload["summary"]["diagnostic_totals"]["line_search_failures"] == 0
    assert payload["summary"]["diagnostic_totals"]["line_search_rejections"] > 0
    assert payload["summary"]["diagnostic_totals"]["pair_stored"] > 1
    evaluation_totals = payload["summary"]["evaluation_profile_totals"]
    assert evaluation_totals["n_f"] == sum(
        item["evaluation_profile"]["n_f"] for item in payload["results"]
    )
    assert evaluation_totals["n_g"] == sum(
        item["evaluation_profile"]["n_g"] for item in payload["results"]
    )
    assert evaluation_totals["n_value_gradient"] == evaluation_totals["n_g"]
    assert evaluation_totals["n_total_requests"] == sum(
        item["evaluation_profile"]["n_total_requests"]
        for item in payload["results"]
    )
    cost_totals = payload["summary"]["cost_estimate_totals"]
    assert cost_totals["weighted_request_units"] == pytest.approx(
        sum(
            item["cost_estimate"]["weighted_request_units"]
            for item in payload["results"]
        )
    )
    assert cost_totals["expected_total_vector_elements"] == sum(
        item["cost_estimate"]["expected_total_vector_elements"]
        for item in payload["results"]
    )
    assert cost_totals["expected_vector_passes"] == pytest.approx(
        sum(
            item["cost_estimate"]["expected_vector_passes"]
            for item in payload["results"]
        )
    )
    assert payload["summary"]["trace_summary_totals"]["request_counts"]["DONE"] == 7
    trace_expected = payload["summary"]["trace_summary_totals"][
        "expected_evaluation_counts"
    ]
    assert trace_expected["none"] == 7
    assert trace_expected["value_gradient"] > 0
    assert trace_expected["preconditioner"] > 0
    assert trace_expected["hessian_vector"] > 0
    trace_workload = payload["summary"]["trace_summary_totals"]
    assert trace_workload["expected_gradient_requests"] == trace_expected[
        "value_gradient"
    ]
    assert trace_workload["expected_vector_requests"] == (
        trace_expected["preconditioner"] + trace_expected["hessian_vector"]
    )
    assert trace_workload["expected_total_vector_elements"] == (
        trace_workload["expected_gradient_elements"]
        + trace_workload["expected_vector_elements"]
    )
    assert payload["summary"]["trace_summary_totals"]["status_counts"][
        "CONVERGED_GRADIENT"
    ] == 7
    assert payload["summary"]["trace_summary_totals"]["success_count"] == 7
    assert payload["summary"]["trace_summary_totals"]["failed_count"] == 0
    assert payload["summary"]["by_case"]["quadratic"]["runs"] == 1
    assert payload["summary"]["by_case"]["quadratic"][
        "total_wall_time_seconds"
    ] >= 0.0
    assert payload["summary"]["by_case"]["rosenbrock"]["runs"] == 3
    assert payload["summary"]["by_case"]["scaled_quadratic"]["runs"] == 2
    assert payload["summary"]["by_config"]["lbfgs_more_thuente"]["runs"] == 1
    assert (
        payload["summary"]["by_config"]["lbfgs_legacy_weak_wolfe"]["runs"] == 1
    )
    assert payload["summary"]["by_config"]["ptrn_hager_zhang"]["total_n_hvp"] > 0
    assert payload["summary"]["by_config"]["ptrn_hager_zhang"][
        "evaluation_profile_totals"
    ]["n_hvp"] > 0
    assert payload["summary"]["by_method"]["lbfgs"]["runs"] == 4
    assert payload["summary"]["by_method"]["ptrn"]["total_n_hvp"] > 0
    assert payload["summary"]["by_method"]["trn"]["total_n_hvp"] > 0
    assert payload["summary"]["by_line_search"]["HAGER_ZHANG"]["runs"] == 4
    assert payload["summary"]["by_line_search"]["MORE_THUENTE"]["runs"] == 1
    assert payload["summary"]["by_line_search"]["ARMIJO_CUBIC"]["runs"] == 1
    assert payload["summary"]["by_line_search_policy"]["HAGER_ZHANG"]["runs"] == 4
    assert payload["summary"]["by_line_search_policy"]["MORE_THUENTE"]["runs"] == 1
    assert payload["summary"]["by_line_search_policy"]["ARMIJO_CUBIC"]["runs"] == 1
    assert payload["summary"]["by_globalization_policy"]["LINE_SEARCH"]["runs"] == 6
    assert payload["summary"]["by_globalization_policy"]["TRUST_REGION"]["runs"] == 1
    assert payload["summary"]["by_bounds_strategy_policy"]["NONE"]["runs"] == 6
    assert payload["summary"]["by_bounds_strategy_policy"]["PROJECTED_GRADIENT"]["runs"] == 1
    assert payload["summary"]["by_cost_model_policy"]["BALANCED"]["runs"] == 7
    assert (
        payload["summary"]["best_success_by_case"]["rosenbrock"]["config"]
        in {
            "lbfgs_hager_zhang",
            "lbfgs_more_thuente",
            "lbfgs_legacy_weak_wolfe",
        }
    )
    assert set(results) == {
        ("quadratic", "lbfgs_hager_zhang"),
        ("rosenbrock", "lbfgs_hager_zhang"),
        ("rosenbrock", "lbfgs_more_thuente"),
        ("rosenbrock", "lbfgs_legacy_weak_wolfe"),
        ("scaled_quadratic", "ptrn_hager_zhang"),
        ("scaled_quadratic", "trn_trust_region"),
        ("lower_bound_quadratic", "pstd_projected_gradient"),
    }
    quadratic_lbfgs = results[("quadratic", "lbfgs_hager_zhang")]
    rosenbrock_hz = results[("rosenbrock", "lbfgs_hager_zhang")]
    rosenbrock_more_thuente = results[("rosenbrock", "lbfgs_more_thuente")]
    rosenbrock_weak_wolfe = results[("rosenbrock", "lbfgs_legacy_weak_wolfe")]
    ptrn = results[("scaled_quadratic", "ptrn_hager_zhang")]
    trn_trust = results[("scaled_quadratic", "trn_trust_region")]
    projected = results[("lower_bound_quadratic", "pstd_projected_gradient")]

    assert quadratic_lbfgs["success"] is True
    assert quadratic_lbfgs["status"] == "CONVERGED_GRADIENT"
    assert len(quadratic_lbfgs["config_fingerprint"]) == 16
    assert "schema=tide.optim.config.v1" in quadratic_lbfgs["config_signature"]
    assert "method=lbfgs" in quadratic_lbfgs["config_signature"]
    assert "policy.method=lbfgs" in quadratic_lbfgs["config_signature"]
    assert "policy.line_search=HAGER_ZHANG" in quadratic_lbfgs["config_signature"]
    best_quadratic = payload["summary"]["best_success_by_case"]["quadratic"]
    assert best_quadratic["config_fingerprint"] == quadratic_lbfgs["config_fingerprint"]
    assert best_quadratic["method"] == "lbfgs"
    assert best_quadratic["line_search_policy"] == "HAGER_ZHANG"
    assert best_quadratic["globalization_policy"] == "LINE_SEARCH"
    assert best_quadratic["bounds_strategy_policy"] == "NONE"
    assert best_quadratic["cost_model_policy"] == "BALANCED"
    assert best_quadratic["resolved_policies"] == quadratic_lbfgs["resolved_policies"]
    assert best_quadratic["wall_time_seconds"] == quadratic_lbfgs["wall_time_seconds"]
    assert quadratic_lbfgs["wall_time_seconds"] >= 0.0
    assert quadratic_lbfgs["method"] == "lbfgs"
    assert quadratic_lbfgs["line_search"] == "HAGER_ZHANG"
    assert quadratic_lbfgs["bounds_strategy_policy"] == "NONE"
    assert quadratic_lbfgs["cost_model_policy"] == "BALANCED"
    assert quadratic_lbfgs["evaluation_profile"]["cost_model"] == "BALANCED"
    assert quadratic_lbfgs["evaluation_profile"]["n_f"] == quadratic_lbfgs["n_f"]
    assert quadratic_lbfgs["evaluation_profile"]["n_g"] == quadratic_lbfgs["n_g"]
    assert quadratic_lbfgs["evaluation_profile"]["n_value_gradient"] == (
        quadratic_lbfgs["n_g"]
    )
    assert quadratic_lbfgs["evaluation_profile"]["n_total_requests"] == (
        quadratic_lbfgs["evaluation_profile"]["n_value_only"]
        + quadratic_lbfgs["n_g"]
        + quadratic_lbfgs["n_hvp"]
        + quadratic_lbfgs["n_prec"]
    )
    assert quadratic_lbfgs["cost_estimate"]["cost_model"] == "BALANCED"
    assert quadratic_lbfgs["cost_estimate"]["n"] == quadratic_lbfgs["n"]
    assert quadratic_lbfgs["cost_estimate"]["weights"]["value_gradient"] == 2.0
    assert quadratic_lbfgs["cost_estimate"]["weighted_request_units"] == pytest.approx(
        quadratic_lbfgs["cost_estimate"]["n_value_only"]
        + 2.0 * quadratic_lbfgs["cost_estimate"]["n_value_gradient"]
        + 2.0 * quadratic_lbfgs["cost_estimate"]["n_hvp"]
        + 0.25 * quadratic_lbfgs["cost_estimate"]["n_prec"]
    )
    assert quadratic_lbfgs["cost_estimate"]["expected_total_vector_elements"] == (
        quadratic_lbfgs["trace_summary"]["expected_total_vector_elements"]
    )
    assert quadratic_lbfgs["direction_policy"] == "LBFGS"
    assert quadratic_lbfgs["stopping_diagnostics"]["policy"] == "STANDARD"
    assert quadratic_lbfgs["stopping_diagnostics"]["status"] == (
        quadratic_lbfgs["status"]
    )
    assert quadratic_lbfgs["stopping_diagnostics"]["success"] is True
    assert quadratic_lbfgs["stopping_diagnostics"]["gradient_satisfied"] is True
    assert quadratic_lbfgs["stopping_diagnostics"]["grad_tolerance"] >= 0.0
    assert quadratic_lbfgs["direction_diagnostics"]["policy"] == "LBFGS"
    assert quadratic_lbfgs["direction_diagnostics"]["descent_direction"] is True
    assert quadratic_lbfgs["direction_diagnostics"]["finite"] is True
    assert quadratic_lbfgs["resolved_policies"]["method"] == "lbfgs"
    assert quadratic_lbfgs["resolved_policies"]["direction_policy"] == "LBFGS"
    assert quadratic_lbfgs["resolved_policies"]["line_search_policy"] == (
        "HAGER_ZHANG"
    )
    assert quadratic_lbfgs["resolved_policies"]["alpha_guess_policy"] == "INITIAL"
    assert quadratic_lbfgs["resolved_policies"]["stopping_policy"] == "STANDARD"
    assert quadratic_lbfgs["resolved_policies"]["bounds_strategy"] == "NONE"
    assert quadratic_lbfgs["resolved_policies"]["cost_model"] == "BALANCED"
    assert payload["summary"]["total_wall_time_seconds"] == pytest.approx(
        sum(item["wall_time_seconds"] for item in payload["results"])
    )
    assert quadratic_lbfgs["trace_policy"] == "NONE"
    assert quadratic_lbfgs["n_trace_stored"] == 0
    assert quadratic_lbfgs["trace_summary"]["n_reports"] == (
        quadratic_lbfgs["n_trace_events"]
    )
    assert quadratic_lbfgs["trace_summary"]["request_counts"]["DONE"] == 1
    assert quadratic_lbfgs["trace_summary"]["expected_evaluation_counts"][
        "none"
    ] == 1
    assert quadratic_lbfgs["trace_summary"]["expected_evaluation_counts"][
        "value_gradient"
    ] >= 1
    assert quadratic_lbfgs["trace_summary"]["expected_gradient_requests"] == (
        quadratic_lbfgs["trace_summary"]["expected_evaluation_counts"][
            "value_gradient"
        ]
    )
    assert quadratic_lbfgs["trace_summary"]["expected_gradient_elements"] == (
        quadratic_lbfgs["trace_summary"]["expected_gradient_requests"]
        * quadratic_lbfgs["n"]
    )
    assert quadratic_lbfgs["trace_summary"]["expected_total_vector_elements"] == (
        quadratic_lbfgs["trace_summary"]["expected_gradient_elements"]
        + quadratic_lbfgs["trace_summary"]["expected_vector_elements"]
    )
    assert quadratic_lbfgs["trace_summary"]["status_counts"][
        "CONVERGED_GRADIENT"
    ] == 1
    assert quadratic_lbfgs["trace_summary"]["failure_reason_counts"][
        "null"
    ] >= 1
    assert quadratic_lbfgs["trace_summary"]["success_count"] == 1
    assert quadratic_lbfgs["trace_summary"]["failed_count"] == 0
    assert quadratic_lbfgs["trace_summary"]["line_search_status_counts"][
        "ACCEPTED"
    ] >= 1
    assert sum(
        count
        for acceptance, count in quadratic_lbfgs["trace_summary"][
            "line_search_acceptance_counts"
        ].items()
        if acceptance != "NONE"
    ) == quadratic_lbfgs["line_search_accept_count"]
    assert quadratic_lbfgs["trace_summary"]["pair_status_counts"]["STORED"] == 1
    assert quadratic_lbfgs["trace_summary"]["backend_event_counts"][
        "line_search_accept"
    ] == quadratic_lbfgs["line_search_accept_count"]
    assert quadratic_lbfgs["warning_flags"] == 0
    assert quadratic_lbfgs["warnings"] == []
    assert quadratic_lbfgs["line_search_failure_count"] == 0
    assert quadratic_lbfgs["line_search_diagnostics"]["policy"] == "HAGER_ZHANG"
    assert quadratic_lbfgs["line_search_diagnostics"]["accept_count"] == (
        quadratic_lbfgs["line_search_accept_count"]
    )
    assert quadratic_lbfgs["line_search_diagnostics"]["failed"] is False
    assert quadratic_lbfgs["pair_update_diagnostics"]["stored_count"] == (
        quadratic_lbfgs["pair_stored_count"]
    )
    assert quadratic_lbfgs["pair_update_diagnostics"]["history_updated"] is True
    assert quadratic_lbfgs["pair_stored_count"] == 1
    assert rosenbrock_hz["line_search_policy"] == "HAGER_ZHANG"
    assert rosenbrock_hz["line_search_rejection_count"] > 0
    assert rosenbrock_hz["line_search_diagnostics"]["rejected"] is True
    assert (
        rosenbrock_hz["config_fingerprint"]
        != rosenbrock_more_thuente["config_fingerprint"]
    )
    assert (
        rosenbrock_more_thuente["config_fingerprint"]
        != rosenbrock_weak_wolfe["config_fingerprint"]
    )
    assert rosenbrock_more_thuente["line_search_policy"] == "MORE_THUENTE"
    assert (
        rosenbrock_more_thuente["line_search_diagnostics"]["policy"]
        == "MORE_THUENTE"
    )
    assert rosenbrock_more_thuente["line_search_rejection_count"] > 0
    assert rosenbrock_weak_wolfe["line_search_policy"] == "LEGACY_WEAK_WOLFE"
    assert rosenbrock_weak_wolfe["line_search_rejection_count"] > 0
    assert ptrn["direction_policy"] == (
        "PRECONDITIONED_TRUNCATED_NEWTON"
    )
    assert ptrn["direction_diagnostics"]["policy"] == (
        "PRECONDITIONED_TRUNCATED_NEWTON"
    )
    assert ptrn["direction_diagnostics"]["descent_direction"] is True
    assert ptrn["resolved_policies"]["method"] == "ptrn"
    assert ptrn["resolved_policies"]["direction_policy"] == (
        "PRECONDITIONED_TRUNCATED_NEWTON"
    )
    assert ptrn["resolved_policies"]["globalization_policy"] == "LINE_SEARCH"
    assert ptrn["resolved_policies"]["cost_model"] == "BALANCED"
    assert ptrn["n_hvp"] > 0
    assert ptrn["n_prec"] > 0
    assert ptrn["inner_solve_diagnostics"]["n_hvp"] == ptrn["n_hvp"]
    assert ptrn["inner_solve_diagnostics"]["n_prec"] == ptrn["n_prec"]
    assert ptrn["inner_solve_diagnostics"]["converged"] is True
    assert ptrn["inner_solve_diagnostics"]["failed"] is False
    assert ptrn["trust_region_diagnostics"]["active"] is False
    assert ptrn["trace_summary"]["request_counts"]["EVALUATE_HV"] > 0
    assert ptrn["trace_summary"]["request_counts"]["APPLY_PRECONDITIONER"] > 0
    assert ptrn["trace_summary"]["expected_evaluation_counts"][
        "hessian_vector"
    ] > 0
    assert ptrn["trace_summary"]["expected_evaluation_counts"][
        "preconditioner"
    ] > 0
    assert ptrn["trace_summary"]["expected_vector_requests"] == (
        ptrn["trace_summary"]["expected_evaluation_counts"]["hessian_vector"]
        + ptrn["trace_summary"]["expected_evaluation_counts"]["preconditioner"]
    )
    assert ptrn["trace_summary"]["expected_vector_elements"] == (
        ptrn["trace_summary"]["expected_vector_requests"] * ptrn["n"]
    )
    assert ptrn["trace_summary"]["expected_total_vector_elements"] == (
        ptrn["trace_summary"]["expected_gradient_elements"]
        + ptrn["trace_summary"]["expected_vector_elements"]
    )
    assert ptrn["trace_summary"]["preconditioner_status_counts"]["APPLIED"] > 0
    assert (
        ptrn["trace_summary"]["inner_status_counts"]["FORCING_REACHED"] > 0
    )
    assert ptrn["inner_warning_count"] == 0
    assert trn_trust["direction_policy"] == "TRUNCATED_NEWTON"
    assert trn_trust["resolved_policies"]["method"] == "trn"
    assert trn_trust["resolved_policies"]["globalization_policy"] == (
        "TRUST_REGION"
    )
    assert trn_trust["trust_region_diagnostics"]["active"] is True
    assert trn_trust["trust_region_diagnostics"]["accepted"] is True
    assert trn_trust["trust_region_diagnostics"]["failed"] is False
    assert trn_trust["trust_region_diagnostics"]["accept_count"] > 0
    assert trn_trust["trace_summary"]["trust_region_status_counts"]["ACCEPTED"] > 0
    assert projected["direction_policy"] == "STEEPEST_DESCENT"
    assert projected["resolved_policies"]["method"] == "pstd"
    assert projected["line_search_policy"] == "ARMIJO_CUBIC"
    assert projected["projected_grad_norm"] == 0.0
    assert projected["bounds_diagnostics"]["strategy"] == "PROJECTED_GRADIENT"
    assert projected["bounds_diagnostics"]["active_count"] == 1
    assert projected["bounds_diagnostics"]["has_trial_projections"] is True
    assert projected["active_lower_count"] == 1


def test_optimize_result_to_dict_is_json_serializable_and_compact_by_default():
    target = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual.ravel(), residual.ravel())), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([[4.0, -5.0], [1.5, 8.0]], dtype=np.float64),
        options={
            "trace_policy": "stride",
            "trace_stride": 2,
            "line_search": "armijo_cubic",
            "initial_step": 1.0,
            "gtol_abs": 1e-9,
            "max_iter": 20,
        },
    )
    payload = result.to_dict()

    json.dumps(payload, sort_keys=True)
    assert result.success, result.reason
    assert payload["success"] is True
    assert payload["status"] == result.status.name
    assert payload["method"] == "lbfgs"
    assert payload["effective_options"]["line_search"] == "armijo_cubic"
    assert payload["effective_options"]["trace_policy"] == "stride"
    assert payload["effective_options"]["trace_stride"] == 2
    assert payload["backend_options"]["line_search_policy"] == "ARMIJO_CUBIC"
    assert payload["backend_options"]["n"] == 4
    assert payload["resolved_policies"]["method"] == "lbfgs"
    assert payload["resolved_policies"]["line_search_policy"] == "ARMIJO_CUBIC"
    assert payload["resolved_policies"]["bounds_strategy"] == "NONE"
    assert isinstance(result.stopping_diagnostics, tide.optim.StoppingDiagnostics)
    assert payload["stopping_diagnostics"]["policy"] == "STANDARD"
    assert payload["stopping_diagnostics"]["status"] == result.status.name
    assert payload["stopping_diagnostics"]["reason"] == result.reason
    assert payload["stopping_diagnostics"]["success"] is True
    assert payload["stopping_diagnostics"]["grad_tolerance"] >= 0.0
    assert payload["stopping_diagnostics"]["n_iter"] == result.n_iter
    assert payload["stopping_diagnostics"]["max_iter"] == 20
    assert payload["trace_summary"]["status_counts"]["CONVERGED_GRADIENT"] == 1
    assert payload["trace_summary"]["failure_reason_counts"]["null"] >= 1
    assert payload["trace_summary"]["success_count"] == 1
    assert payload["trace_summary"]["failed_count"] == 0
    assert result.trace_summary.status_count(result.status) == 1
    assert result.trace_summary.failure_reason_count(None) >= 1
    assert payload["direction_diagnostics"]["policy"] == "LBFGS"
    assert payload["direction_diagnostics"]["descent_direction"] is True
    assert isinstance(result.direction_diagnostics, tide.optim.DirectionDiagnostics)
    assert payload["bounds_diagnostics"]["strategy"] == "NONE"
    assert payload["bounds_diagnostics"]["active_count"] == 0
    assert isinstance(result.line_search_diagnostics, tide.optim.LineSearchDiagnostics)
    assert result.line_search_diagnostics.policy == tide.optim.LineSearchPolicy.ARMIJO_CUBIC
    assert payload["line_search_diagnostics"]["policy"] == "ARMIJO_CUBIC"
    assert payload["line_search_diagnostics"]["accept_count"] == (
        result.trace_summary.line_search_accept_count
    )
    assert payload["line_search_diagnostics"]["failed"] is False
    assert isinstance(result.pair_update_diagnostics, tide.optim.PairUpdateDiagnostics)
    assert payload["pair_update_diagnostics"]["status"] == (
        result.pair_update_diagnostics.status.name
    )
    assert payload["pair_update_diagnostics"]["stored_count"] == (
        result.trace_summary.pair_stored_count
    )
    assert isinstance(result.inner_solve_diagnostics, tide.optim.InnerSolveDiagnostics)
    assert payload["inner_solve_diagnostics"]["inner_status"] == (
        result.inner_solve_diagnostics.inner_status.name
    )
    assert payload["inner_solve_diagnostics"]["n_hvp"] == result.n_hvp
    assert isinstance(result.trust_region_diagnostics, tide.optim.TrustRegionDiagnostics)
    assert payload["trust_region_diagnostics"]["status"] == (
        result.trust_region_diagnostics.status.name
    )
    assert payload["trust_region_diagnostics"]["active"] is False
    assert isinstance(result.evaluation_profile, tide.optim.EvaluationProfile)
    assert payload["evaluation_profile"] == result.evaluation_profile.to_dict()
    assert payload["evaluation_profile"]["cost_model"] == "balanced"
    assert payload["evaluation_profile"]["n_f"] == result.n_f
    assert payload["evaluation_profile"]["n_g"] == result.n_g
    assert payload["evaluation_profile"]["n_value_only"] == max(
        0,
        result.n_f - result.n_g,
    )
    assert payload["evaluation_profile"]["n_value_gradient"] == result.n_g
    assert payload["evaluation_profile"]["n_total_requests"] == (
        payload["evaluation_profile"]["n_value_only"]
        + result.n_g
        + result.n_hvp
        + result.n_prec
    )
    assert isinstance(result.cost_estimate, tide.optim.EvaluationCostEstimate)
    assert payload["cost_estimate"] == result.cost_estimate.to_dict()
    assert payload["cost_estimate"]["cost_model"] == "balanced"
    assert payload["cost_estimate"]["n"] == result.n
    assert payload["cost_estimate"]["weights"]["value_gradient"] == 2.0
    assert payload["cost_estimate"]["weighted_request_units"] == pytest.approx(
        payload["cost_estimate"]["n_value_only"]
        + 2.0 * payload["cost_estimate"]["n_value_gradient"]
        + 2.0 * payload["cost_estimate"]["n_hvp"]
        + 0.25 * payload["cost_estimate"]["n_prec"]
    )
    assert payload["cost_estimate"]["expected_total_vector_elements"] == (
        result.trace_summary.expected_total_vector_elements
    )
    assert payload["policy_resolution"] == payload["resolved_policies"]
    assert payload["config_fingerprint"] == result.config_fingerprint
    assert len(result.config_fingerprint) == 64
    assert result.config_signature["backend_options"]["n"] == 4
    assert payload["config_signature"] == result.config_signature
    assert payload["x_shape"] == [2, 2]
    assert payload["x_norm"] == pytest.approx(float(np.linalg.norm(result.x)))
    assert "x" not in payload
    assert "trace" not in payload
    assert payload["diagnostics"]["request_counts"]["DONE"] == 1
    assert (
        payload["diagnostics"]["line_search_status_report_counts"]["ACCEPTED"]
        > 0
    )
    assert payload["line_search_policy"] == result.last_trace.line_search_policy.name
    assert payload["line_search_acceptance"] == (
        result.last_trace.line_search_acceptance.name
    )
    assert isinstance(result.last_trace.warning_flags, tide.optim.WarningFlag)
    assert result.last_trace.warning_flags == tide.optim.WarningFlag(0)
    assert result.last_trace.warnings == ()
    assert isinstance(
        result.last_trace.request_requirements,
        tide.optim.OptimizerRequestRequirements,
    )
    assert result.last_trace.expected_evaluation == "none"
    assert result.last_trace.required_fields == ()
    assert result.trace_summary.expected_evaluation_count("none") >= 1
    assert payload["trace_summary"]["expected_evaluation_counts"]["none"] >= 1
    assert payload["warnings"] == list(result.warnings)


def test_optimize_result_to_dict_can_include_x_and_trace():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - np.array([2.0, -1.0], dtype=np.float64)
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        options={
            "trace_policy": "last",
            "line_search": "armijo_cubic",
            "initial_step": 1.0,
            "gtol_abs": 1e-8,
            "max_iter": 20,
        },
    )
    payload = result.to_dict(include_x=True, include_trace=True)

    json.dumps(payload, sort_keys=True)
    assert result.success, result.reason
    assert payload["x"] == result.x.tolist()
    assert len(payload["trace"]) == result.n_trace_stored
    assert payload["trace"][-1]["request"] == result.trace[-1].request.name
    assert payload["trace"][-1]["status"] == result.status.name
    assert payload["trace"][-1]["line_search_policy"] == "ARMIJO_CUBIC"
    assert payload["trace"][-1]["expected_evaluation"] == (
        result.trace[-1].expected_evaluation
    )
    assert payload["trace"][-1]["request_requirements"]["kind"] == (
        result.trace[-1].request.name
    )
    assert payload["trace"][-1]["request_requirements"]["sequence"] == (
        result.trace[-1].sequence
    )
    assert payload["trace"][-1]["stopping_diagnostics"]["status"] == (
        result.trace[-1].status.name
    )
    assert payload["trace"][-1]["stopping_diagnostics"]["max_iter"] == 20
    assert payload["trace"][-1]["direction_diagnostics"]["policy"] == (
        result.trace[-1].direction_policy.name
    )
    assert payload["trace"][-1]["line_search_diagnostics"]["policy"] == (
        result.trace[-1].line_search_policy.name
    )
    assert payload["trace"][-1]["line_search_diagnostics"]["status"] == (
        result.trace[-1].line_search_status.name
    )
    assert payload["trace"][-1]["pair_update_diagnostics"]["status"] == (
        result.trace[-1].pair_status.name
    )
    assert payload["trace"][-1]["inner_solve_diagnostics"]["inner_status"] == (
        result.trace[-1].inner_status.name
    )
    assert payload["trace"][-1]["trust_region_diagnostics"]["status"] == (
        result.trace[-1].trust_region_status.name
    )
    assert payload["trace"][-1]["warning_flags"] == int(
        result.trace[-1].warning_flags
    )
    assert payload["trace"][-1]["warnings"] == list(result.trace[-1].warnings)


def test_trace_policy_none_keeps_aggregate_diagnostics():
    target = np.array([2.0, -1.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        options={
            "trace_policy": "none",
            "line_search": "armijo_cubic",
            "initial_step": 10.0,
            "max_iter": 30,
            "gtol_abs": 1e-8,
        },
    )
    diagnostics = result.diagnostics
    summary = result.trace_summary
    payload = result.to_dict()

    assert result.success, result.reason
    assert result.trace == ()
    assert result.n_trace_stored == 0
    assert isinstance(summary, tide.optim.TraceSummary)
    assert summary.n_reports == result.n_trace_events
    assert summary.request_count(tide.optim.RequestKind.EVALUATE_F) > 0
    assert summary.request_count(tide.optim.RequestKind.DONE) == 1
    assert summary.expected_evaluation_count("value") == summary.request_count(
        tide.optim.RequestKind.EVALUATE_F
    )
    assert summary.expected_evaluation_count("none") >= 1
    assert summary.expected_gradient_requests == summary.expected_evaluation_count(
        "value_gradient"
    )
    assert summary.expected_vector_requests == 0
    assert summary.expected_gradient_elements == (
        summary.expected_gradient_requests * target.size
    )
    assert summary.expected_vector_elements == 0
    assert summary.expected_total_vector_elements == (
        summary.expected_gradient_elements
    )
    assert summary.status_count(tide.optim.OptimStatus.RUNNING) > 0
    assert summary.status_count(tide.optim.OptimStatus.CONVERGED_GRADIENT) == 1
    assert summary.failure_reason_count(None) >= 1
    assert summary.success_count == 1
    assert summary.failed_count == 0
    assert summary.user_stopped_count == 0
    assert (
        summary.line_search_status_count(
            tide.optim.LineSearchStatus.REJECTED_ARMIJO
        )
        > 0
    )
    assert (
        summary.line_search_status_count(tide.optim.LineSearchStatus.ACCEPTED)
        > 0
    )
    assert (
        summary.line_search_acceptance_count(
            tide.optim.LineSearchAcceptance.ARMIJO
        )
        > 0
    )
    assert summary.pair_status_count(tide.optim.PairStatus.STORED) > 0
    assert summary.warning_flags == tide.optim.WarningFlag(0)
    assert summary.warnings == ()
    assert summary.line_search_rejection_count > 0
    assert summary.line_search_accept_count == result.n_iter
    assert summary.pair_stored_count > 0
    assert summary.backend_event_counts["line_search_rejection"] == (
        diagnostics["line_search_rejection_count"]
    )
    assert diagnostics["request_counts"]["EVALUATE_F"] > 0
    assert diagnostics["request_counts"]["DONE"] == 1
    assert diagnostics["status_report_counts"]["CONVERGED_GRADIENT"] == 1
    assert diagnostics["failure_reason_report_counts"]["null"] >= 1
    assert diagnostics["line_search_status_report_counts"]["REJECTED_ARMIJO"] > 0
    assert diagnostics["line_search_status_report_counts"]["ACCEPTED"] > 0
    assert diagnostics["line_search_rejection_count"] > 0
    assert diagnostics["line_search_accept_count"] == result.n_iter
    assert diagnostics["pair_stored_count"] > 0
    assert diagnostics["expected_gradient_requests"] == (
        summary.expected_gradient_requests
    )
    assert diagnostics["expected_gradient_elements"] == (
        summary.expected_gradient_elements
    )
    assert diagnostics["expected_vector_requests"] == 0
    assert diagnostics["expected_vector_elements"] == 0
    assert diagnostics["backend_event_counts"]["line_search_rejection"] == (
        diagnostics["line_search_rejection_count"]
    )
    assert payload["trace_summary"]["request_counts"]["DONE"] == 1
    assert payload["trace_summary"]["expected_evaluation_counts"]["value"] == (
        summary.expected_evaluation_count("value")
    )
    assert payload["trace_summary"]["expected_evaluation_counts"]["none"] >= 1
    assert payload["trace_summary"]["expected_gradient_requests"] == (
        summary.expected_gradient_requests
    )
    assert payload["trace_summary"]["expected_gradient_elements"] == (
        summary.expected_gradient_elements
    )
    assert payload["trace_summary"]["expected_total_vector_elements"] == (
        summary.expected_total_vector_elements
    )
    assert payload["trace_summary"]["status_counts"]["CONVERGED_GRADIENT"] == 1
    assert payload["trace_summary"]["failure_reason_counts"]["null"] >= 1
    assert (
        payload["trace_summary"]["line_search_status_counts"]["REJECTED_ARMIJO"]
        > 0
    )
    assert payload["trace_summary"]["warning_flags"] == 0
    assert payload["diagnostics"] == diagnostics
    json.dumps(payload, sort_keys=True)


def test_optim_policy_aliases_are_documented_by_canonical_lists():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    result = tide.optim.minimize(
        quadratic,
        np.array([1.0], dtype=np.float64),
        method="steepest_descent",
        options={
            "bounds_strategy": "projected",
            "line_search": "hz",
            "max_iter": 1,
            "gtol_abs": 0.0,
        },
        bounds=(np.array([-2.0]), np.array([2.0])),
    )

    assert result.last_trace is not None
    assert result.last_trace.direction_policy == tide.optim.DirectionPolicy.STEEPEST_DESCENT
    assert result.last_trace.line_search_policy == tide.optim.LineSearchPolicy.HAGER_ZHANG
    assert "hager_zhang" in tide.optim.SUPPORTED_LINE_SEARCHES
    assert "projected_gradient" in tide.optim.SUPPORTED_BOUNDS_STRATEGIES


def test_optimizer_options_is_generic_public_options_type():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    options = tide.optim.OptimizerOptions(
        line_search="armijo_cubic",
        max_iter=10,
        gtol_abs=1e-10,
    )
    result = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        method="pstd",
        options=options,
    )
    pstd_options = tide.optim.OptimizerOptions.for_method(
        "pstd",
        line_search="armijo_cubic",
        max_iter=10,
        gtol_abs=1e-10,
    )
    pstd_direct = pstd_options.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
    )
    pnlcg_options = tide.optim.OptimizerOptions.for_method(
        "pnlcg",
        max_iter=10,
        gtol_abs=1e-10,
    )
    pnlcg_validation = pnlcg_options.validate(n=1)
    pnlcg_direct = pnlcg_options.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
    )
    pnlcg_top_level_validation = tide.optim.validate_options(
        options=pnlcg_options,
        n=1,
    )
    pnlcg_top_level = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        options=pnlcg_options,
    )
    pnlcg_mapping = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        options={
            "direction": "pnlcg",
            "max_iter": 10,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    assert result.effective_options["direction"] == "pstd"
    assert result.config_signature["effective_options"]["direction"] == "pstd"
    assert result.last_trace is not None
    assert result.last_trace.direction_policy == tide.optim.DirectionPolicy.STEEPEST_DESCENT
    assert result.last_trace.line_search_policy == tide.optim.LineSearchPolicy.ARMIJO_CUBIC
    assert pstd_direct.success, pstd_direct.reason
    assert pstd_direct.method == "pstd"
    assert pstd_direct.resolved_policies["method"] == "pstd"
    assert pstd_direct.config_signature["method"] == "pstd"
    assert pstd_direct.last_trace is not None
    assert pstd_direct.last_trace.direction_policy == (
        tide.optim.DirectionPolicy.STEEPEST_DESCENT
    )
    assert pnlcg_validation.ok
    assert pnlcg_validation.policy_resolution is not None
    assert pnlcg_validation.policy_resolution.method == "pnlcg"
    assert pnlcg_validation.policy_resolution.direction == tide.optim.DirectionPolicy.NLCG
    assert pnlcg_validation.config_signature["method"] == "pnlcg"
    assert pnlcg_direct.success, pnlcg_direct.reason
    assert pnlcg_direct.method == "pnlcg"
    assert pnlcg_direct.resolved_policies["method"] == "pnlcg"
    assert pnlcg_direct.last_trace is not None
    assert pnlcg_direct.last_trace.direction_policy == tide.optim.DirectionPolicy.NLCG
    assert pnlcg_top_level_validation.ok
    assert pnlcg_top_level_validation.policy_resolution is not None
    assert pnlcg_top_level_validation.policy_resolution.method == "pnlcg"
    assert pnlcg_top_level_validation.config_signature["method"] == "pnlcg"
    assert pnlcg_top_level.success, pnlcg_top_level.reason
    assert pnlcg_top_level.method == "pnlcg"
    assert pnlcg_top_level.resolved_policies["direction_policy"] == "NLCG"
    assert pnlcg_mapping.success, pnlcg_mapping.reason
    assert pnlcg_mapping.method == "pnlcg"
    assert pnlcg_mapping.resolved_policies["direction_policy"] == "NLCG"


def test_explicit_method_rejects_conflicting_direction_policy():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    validation = tide.optim.validate_options(
        method="lbfgs",
        options={"direction": "pnlcg"},
        n=1,
    )

    assert not validation.ok
    assert validation.code == tide.optim.OptionsValidationCode.DIRECTION_POLICY
    assert validation.field == "direction"
    assert "conflicts with explicit method" in validation.message
    assert validation.effective_options == {}
    assert validation.config_signature == {}

    with pytest.raises(ValueError, match="conflicts with explicit method"):
        tide.optim.minimize(
            quadratic,
            np.array([8.0], dtype=np.float64),
            method="lbfgs",
            options={"direction": "pnlcg"},
        )

    with pytest.raises(ValueError, match="conflicts with explicit method"):
        tide.optim.OptimizerSession(
            np.array([8.0], dtype=np.float64),
            method="lbfgs",
            options={"direction": "pnlcg"},
        )


def test_cost_model_selects_default_line_search_without_overriding_explicit_policy():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    expensive_gradient = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        method="lbfgs",
        options={
            "cost_model": "expensive_gradient",
            "gtol_abs": 1e-10,
            "max_iter": 10,
        },
    )
    explicit = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        method="lbfgs",
        options={
            "cost_model": "expensive_gradient",
            "line_search": "hager_zhang",
            "gtol_abs": 1e-10,
            "max_iter": 10,
        },
    )
    joint_pstd_options = tide.optim.OptimizerOptions.for_method(
        "pstd",
        cost_model="joint_value_gradient",
        gtol_abs=1e-10,
        max_iter=10,
    )
    joint_pstd = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        method="pstd",
        options=joint_pstd_options,
    )

    assert expensive_gradient.success, expensive_gradient.reason
    assert explicit.success, explicit.reason
    assert joint_pstd.success, joint_pstd.reason
    assert expensive_gradient.effective_options["cost_model"] == (
        "expensive_gradient"
    )
    assert expensive_gradient.resolved_policies["cost_model"] == (
        "expensive_gradient"
    )
    assert expensive_gradient.resolved_policies["line_search_policy"] == (
        "ARMIJO_CUBIC"
    )
    assert isinstance(expensive_gradient.policy_resolution, tide.optim.ResolvedPolicies)
    assert expensive_gradient.policy_resolution.line_search == (
        tide.optim.LineSearchPolicy.ARMIJO_CUBIC
    )
    assert expensive_gradient.policy_resolution.cost_model == "expensive_gradient"
    assert expensive_gradient.to_dict()["policy_resolution"] == (
        expensive_gradient.policy_resolution.to_dict()
    )
    assert expensive_gradient.last_trace is not None
    assert expensive_gradient.last_trace.line_search_policy == (
        tide.optim.LineSearchPolicy.ARMIJO_CUBIC
    )
    assert explicit.last_trace is not None
    assert explicit.last_trace.line_search_policy == (
        tide.optim.LineSearchPolicy.HAGER_ZHANG
    )
    assert joint_pstd.effective_options["cost_model"] == "joint_value_gradient"
    assert joint_pstd.resolved_policies["cost_model"] == "joint_value_gradient"
    assert joint_pstd.resolved_policies["direction_policy"] == "STEEPEST_DESCENT"
    assert joint_pstd.last_trace is not None
    assert joint_pstd.last_trace.line_search_policy == (
        tide.optim.LineSearchPolicy.HAGER_ZHANG
    )
    assert tide.optim.cost_model_weights("balanced") == {
        "value_only": 1.0,
        "value_gradient": 2.0,
        "hessian_vector": 2.0,
        "preconditioner": 0.25,
    }
    assert tide.optim.cost_model_weights("expensive_gradient")[
        "value_gradient"
    ] == 4.0
    assert tide.optim.cost_model_weights("joint_value_gradient")[
        "value_gradient"
    ] == 1.0


def test_alpha_guess_previous_reuses_last_accepted_step():
    scale = np.array([1.0, 10.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(0.5 * np.dot(scale * x, x)), scale * x

    result = tide.optim.minimize(
        quadratic,
        np.array([4.0, 4.0], dtype=np.float64),
        method="pstd",
        options={
            "line_search": "armijo_cubic",
            "alpha_guess": "previous",
            "initial_step": 2.0,
            "max_iter": 4,
            "gtol_abs": 0.0,
        },
    )
    started = [
        entry
        for entry in result.trace
        if entry.request == tide.optim.RequestKind.EVALUATE_F
        and entry.line_search_status == tide.optim.LineSearchStatus.STARTED
    ]

    assert len(started) >= 2
    assert started[0].alpha == pytest.approx(2.0)
    assert started[1].alpha < started[0].alpha
    assert all(
        entry.alpha_guess_policy == tide.optim.AlphaGuessPolicy.PREVIOUS
        for entry in result.trace
    )


def test_alpha_guess_barzilai_borwein_uses_last_step_curvature():
    scale = np.array([1.0, 10.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(0.5 * np.dot(scale * x, x)), scale * x

    result = tide.optim.minimize(
        quadratic,
        np.array([4.0, 4.0], dtype=np.float64),
        method="pstd",
        options={
            "line_search": "armijo_cubic",
            "alpha_guess": "barzilai_borwein",
            "initial_step": 2.0,
            "max_iter": 4,
            "gtol_abs": 0.0,
        },
    )
    started = [
        entry
        for entry in result.trace
        if entry.request == tide.optim.RequestKind.EVALUATE_F
        and entry.line_search_status == tide.optim.LineSearchStatus.STARTED
    ]

    assert len(started) >= 2
    assert started[1].sy > 0.0
    assert started[1].yy > 0.0
    assert started[1].alpha == pytest.approx(started[1].sy / started[1].yy)
    assert all(
        entry.alpha_guess_policy
        == tide.optim.AlphaGuessPolicy.BARZILAI_BORWEIN
        for entry in result.trace
    )


def test_minimize_preserves_multidimensional_model_shape():
    target = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    seen = {"fun": 0, "value": 0}

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        seen["fun"] += 1
        assert x.shape == target.shape
        residual = x - target
        return float(0.5 * np.sum(residual * residual)), residual

    def value(x: np.ndarray) -> float:
        seen["value"] += 1
        assert x.shape == target.shape
        residual = x - target
        return float(0.5 * np.sum(residual * residual))

    result = tide.optim.minimize(
        quadratic,
        np.array([[8.0, -6.0], [2.0, 0.0]], dtype=np.float64),
        method="pstd",
        value=value,
        options={
            "line_search": "armijo_cubic",
            "initial_step": 1.0,
            "max_iter": 10,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    assert result.x.shape == target.shape
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert seen["fun"] > 0
    assert seen["value"] > 0


def test_bounds_accept_multidimensional_shapes_and_preserve_result_shape():
    target = np.array([[0.25, 0.8], [0.5, 0.1]], dtype=np.float64)
    lower = np.zeros_like(target)
    upper = np.ones_like(target)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        assert x.shape == target.shape
        residual = x - target
        return float(0.5 * np.sum(residual * residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([[0.0, 0.9], [0.2, 0.7]], dtype=np.float64),
        method="pstd",
        bounds=(lower, upper),
        options={
            "bounds_strategy": "projected_gradient",
            "line_search": "armijo_cubic",
            "initial_step": 1.0,
            "max_iter": 10,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    assert result.x.shape == target.shape
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert np.all(result.x >= lower)
    assert np.all(result.x <= upper)


def test_ptrn_hvp_and_preconditioner_receive_multidimensional_vectors():
    scale = np.array([[10.0, 1.0], [4.0, 2.0]], dtype=np.float64)
    target = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float64)
    calls = {"hvp": 0, "preconditioner": 0}

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        assert x.shape == target.shape
        residual = x - target
        return float(0.5 * np.sum(scale * residual * residual)), scale * residual

    def hessian_vector(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        calls["hvp"] += 1
        assert x.shape == target.shape
        assert vector.shape == target.shape
        return scale * vector

    def preconditioner(x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        calls["preconditioner"] += 1
        assert x.shape == target.shape
        assert vector.shape == target.shape
        return vector / scale

    result = tide.optim.minimize(
        quadratic,
        np.array([[8.0, -6.0], [2.0, 0.0]], dtype=np.float64),
        method="ptrn",
        hessian_vector=hessian_vector,
        preconditioner=preconditioner,
        options={
            "max_iter": 20,
            "max_inner_iter": 8,
            "inner_rtol": 1e-12,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    assert result.x.shape == target.shape
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert calls["hvp"] == result.n_hvp
    assert calls["preconditioner"] == result.n_prec
    assert result.n_hvp > 0
    assert result.n_prec > 0
    assert result.inner_solve_diagnostics.n_hvp == result.n_hvp
    assert result.inner_solve_diagnostics.n_prec == result.n_prec
    assert result.inner_solve_diagnostics.converged
    assert result.inner_solve_diagnostics.preconditioner_applied


def test_optim_unsupported_method_error_lists_public_methods():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(NotImplementedError) as error:
        tide.optim.minimize(
            quadratic,
            np.array([1.0], dtype=np.float64),
            method="adam",
        )

    message = str(error.value)
    for method in tide.optim.SUPPORTED_METHODS:
        assert f"method={method!r}" in message


def test_optim_unsupported_alpha_guess_error_lists_option_name():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(NotImplementedError, match="alpha_guess"):
        tide.optim.minimize(
            quadratic,
            np.array([1.0], dtype=np.float64),
            options={"alpha_guess": "line-search-oracle"},
        )


def test_optim_unsupported_stopping_policy_error_lists_option_name():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(NotImplementedError, match="stopping"):
        tide.optim.minimize(
            quadratic,
            np.array([1.0], dtype=np.float64),
            options={"stopping": "paper-over-failures"},
        )


def test_optim_unsupported_nlcg_beta_error_lists_option_name():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(NotImplementedError, match="nlcg_beta"):
        tide.optim.minimize(
            quadratic,
            np.array([1.0], dtype=np.float64),
            method="pnlcg",
            options={"nlcg_beta": "mystery-conjugacy"},
        )


def test_optim_unsupported_lbfgs_update_error_lists_option_name():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(NotImplementedError, match="lbfgs_update"):
        tide.optim.minimize(
            quadratic,
            np.array([1.0], dtype=np.float64),
            options={"lbfgs_update": "invent-curvature"},
        )


def test_optim_unsupported_trace_policy_error_lists_option_name():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(NotImplementedError, match="trace_policy"):
        tide.optim.minimize(
            quadratic,
            np.array([1.0], dtype=np.float64),
            options={"trace_policy": "log-to-random-file"},
        )


@pytest.mark.parametrize(
    ("bad_options", "message"),
    [
        ({"history_size": 0}, "history_size"),
        ({"max_line_search": 0}, "max_line_search"),
        ({"max_iter": -1}, "max_iter"),
        ({"max_eval": -1}, "max_eval"),
        ({"max_inner_iter": -1}, "max_inner_iter"),
        ({"initial_step": 0.0}, "initial_step"),
        ({"c1": 0.9, "c2": 0.1}, "0 < c1 < c2 < 1"),
        ({"growth": 1.0}, "growth"),
        ({"alpha_min": 1.0, "alpha_max": 1.0}, "alpha_min"),
        ({"gamma_min": 0.0}, "gamma_min"),
        ({"gtol_abs": -1.0}, "gtol_abs"),
        ({"x_rtol": float("nan")}, "x_rtol"),
        ({"initial_trust_radius": 0.0}, "initial_trust_radius"),
        ({"max_trust_radius": 0.5, "initial_trust_radius": 1.0}, "trust_radius"),
        ({"trust_eta": 1.0}, "trust_eta"),
        ({"trust_shrink": 1.0}, "trust_shrink"),
        ({"trust_grow": 1.0}, "trust_grow"),
        ({"armijo_shrink_min": 0.8, "armijo_shrink_max": 0.5}, "Armijo"),
        ({"bound_margin": -1.0}, "bound_margin"),
        ({"trace_stride": 0}, "trace_stride"),
    ],
)
def test_optimizer_options_are_validated_before_backend_creation(
    bad_options: dict[str, float | int],
    message: str,
):
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(ValueError, match=message):
        tide.optim.minimize(
            quadratic,
            np.array([1.0], dtype=np.float64),
            options=bad_options,
        )


def _rosenbrock(x: np.ndarray) -> tuple[float, np.ndarray]:
    f = (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
    g = np.array(
        [
            2.0 * (x[0] - 1.0) - 400.0 * x[0] * (x[1] - x[0] ** 2),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=np.float64,
    )
    return float(f), g


def test_lbfgs_minimize_rosenbrock_converges():
    result = tide.optim.minimize(
        _rosenbrock,
        np.array([1.5, 1.5], dtype=np.float64),
        options={
            "max_iter": 200,
            "history_size": 20,
            "gtol_abs": 1e-5,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, np.array([1.0, 1.0]), atol=5e-4)
    assert result.f < 1e-8
    assert result.n_f >= result.n_iter
    assert any(entry.pair_status.name == "STORED" for entry in result.trace)


def test_lbfgs_projected_trial_bounds():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 0.4
        return float(np.dot(residual, residual)), 2.0 * residual

    result = tide.optim.minimize(
        quadratic,
        np.array([0.0, 0.9], dtype=np.float64),
        bounds=(np.zeros(2), np.ones(2)),
        options={
            "bounds_strategy": "projected_trial",
            "max_iter": 50,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    assert np.all(result.x >= 0.0)
    assert np.all(result.x <= 1.0)
    np.testing.assert_allclose(result.x, np.array([0.4, 0.4]), atol=1e-6)


def test_projected_gradient_bounds_use_kkt_residual_for_convergence():
    def lower_bound_solution(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x + 1.0
        return float(np.dot(residual, residual)), 2.0 * residual

    result = tide.optim.minimize(
        lower_bound_solution,
        np.array([1.0], dtype=np.float64),
        bounds=(np.array([0.0]), np.array([2.0])),
        options={
            "bounds_strategy": "projected_gradient",
            "line_search": "armijo_cubic",
            "max_iter": 20,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    assert result.status == tide.optim.OptimStatus.CONVERGED_GRADIENT
    np.testing.assert_allclose(result.x, np.array([0.0]), atol=1e-12)
    assert result.f == pytest.approx(1.0)
    assert result.grad_norm == pytest.approx(0.0)
    assert result.projected_grad_norm == pytest.approx(0.0)
    assert isinstance(result.bounds_diagnostics, tide.optim.BoundsDiagnostics)
    assert result.bounds_diagnostics.strategy == (
        tide.optim.BoundsStrategy.PROJECTED_GRADIENT
    )
    assert result.bounds_diagnostics.projected_grad_norm == pytest.approx(0.0)
    assert result.bounds_diagnostics.active_count == 1
    assert result.bounds_diagnostics.has_active_bounds
    assert result.bounds_diagnostics.has_trial_projections
    assert not result.bounds_diagnostics.has_kkt_violations
    assert result.bounds_diagnostics.to_dict()["strategy"] == "PROJECTED_GRADIENT"
    assert result.active_lower_count == 1
    assert result.active_upper_count == 0
    assert result.free_count == 0
    assert result.kkt_violation_count == 0
    assert result.lower_kkt_violation_count == 0
    assert result.upper_kkt_violation_count == 0
    assert result.free_gradient_count == 0
    assert result.trial_projection_count == 1
    assert result.trial_lower_projection_count == 1
    assert result.trial_upper_projection_count == 0
    assert result.trace[-1].projected_grad_norm == pytest.approx(0.0)
    assert result.trace[-1].bounds_diagnostics.strategy is None
    assert result.trace[-1].bounds_diagnostics.active_count == 1
    assert result.trace[-1].active_lower_count == 1
    assert result.trace[-1].kkt_violation_count == 0
    assert result.trace[-1].trial_lower_projection_count == 1


def test_projected_gradient_bounds_report_active_set_kkt_violations():
    def lower_bound_not_optimal(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 1.0
        return float(np.dot(residual, residual)), 2.0 * residual

    result = tide.optim.minimize(
        lower_bound_not_optimal,
        np.array([0.0], dtype=np.float64),
        bounds=(np.array([0.0]), np.array([2.0])),
        options={
            "bounds_strategy": "projected_gradient",
            "max_iter": 0,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.MAX_ITER
    assert result.projected_grad_norm == pytest.approx(2.0)
    assert result.bounds_diagnostics.has_kkt_violations
    assert result.bounds_diagnostics.kkt_violation_count == 1
    assert result.active_lower_count == 1
    assert result.active_upper_count == 0
    assert result.free_count == 0
    assert result.kkt_violation_count == 1
    assert result.lower_kkt_violation_count == 1
    assert result.upper_kkt_violation_count == 0
    assert result.free_gradient_count == 0


def test_lbfgs_skips_curvature_pair_when_bounds_project_trial():
    def lower_bound_solution(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x + 1.0
        return float(np.dot(residual, residual)), 2.0 * residual

    result = tide.optim.minimize(
        lower_bound_solution,
        np.array([1.0], dtype=np.float64),
        bounds=(np.array([0.0]), np.array([2.0])),
        options={
            "bounds_strategy": "projected_gradient",
            "line_search": "armijo_cubic",
            "max_iter": 20,
            "gtol_abs": 1e-10,
        },
    )

    skipped = [
        entry
        for entry in result.trace
        if entry.pair_status == tide.optim.PairStatus.SKIPPED_BOUNDS_PROJECTION
    ]
    assert skipped
    assert all(entry.trial_projection_count > 0 for entry in skipped)
    assert all(entry.direction_diagnostics.finite for entry in result.trace)
    assert all(entry.history_size == 0 for entry in result.trace)
    assert "lbfgs_pair_skipped" in result.warnings


def test_lbfgs_update_skip_bad_curvature_pair_by_default():
    def linear_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(x[0]), np.array([1.0], dtype=np.float64)

    result = tide.optim.minimize(
        linear_objective,
        np.array([0.0], dtype=np.float64),
        method="lbfgs",
        options={
            "line_search": "static",
            "initial_step": 1e-3,
            "max_iter": 1,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.MAX_ITER
    assert result.last_trace is not None
    assert result.last_trace.lbfgs_update_policy == tide.optim.LbfgsUpdatePolicy.SKIP
    assert (
        result.last_trace.pair_status
        == tide.optim.PairStatus.SKIPPED_BAD_CURVATURE
    )
    assert result.last_trace.history_size == 0
    assert result.last_trace.pair_update_diagnostics.bad_curvature
    assert result.last_trace.pair_update_diagnostics.skipped
    assert not result.last_trace.pair_update_diagnostics.history_updated
    assert result.pair_update_diagnostics.skipped
    assert result.pair_update_diagnostics.skip_count > 0
    assert result.last_trace.has_warning(tide.optim.WarningFlag.LBFGS_PAIR_SKIPPED)
    assert "lbfgs_pair_skipped" in result.warnings


def test_lbfgs_update_regularize_stores_bad_curvature_pair():
    def linear_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(x[0]), np.array([1.0], dtype=np.float64)

    result = tide.optim.minimize(
        linear_objective,
        np.array([0.0], dtype=np.float64),
        method="lbfgs",
        options={
            "line_search": "static",
            "initial_step": 1e-3,
            "lbfgs_update": "regularize",
            "max_iter": 1,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.MAX_ITER
    assert result.last_trace is not None
    assert (
        result.last_trace.lbfgs_update_policy
        == tide.optim.LbfgsUpdatePolicy.REGULARIZE
    )
    assert result.last_trace.pair_status == tide.optim.PairStatus.REGULARIZED_STORED
    assert result.last_trace.history_size == 1
    assert result.last_trace.sy > 0.0
    assert result.last_trace.yy > 0.0
    assert np.isfinite(result.last_trace.gamma)
    assert result.last_trace.pair_update_diagnostics.regularized
    assert result.last_trace.pair_update_diagnostics.history_updated
    assert result.pair_update_diagnostics.regularized
    assert result.pair_update_diagnostics.regularized_count > 0
    assert "lbfgs_pair_regularized" in result.warnings
    assert "lbfgs_pair_skipped" not in result.warnings


def test_lbfgs_trace_and_callback_are_diagnostic():
    seen = []

    def callback(entry: tide.optim.TraceEntry) -> bool:
        seen.append((entry.iter, entry.line_search_status.name, entry.grad_norm))
        return False

    result = tide.optim.minimize(
        _rosenbrock,
        np.array([1.2, 1.2], dtype=np.float64),
        options={"max_iter": 20, "gtol_abs": 1e-4},
        callback=callback,
    )

    assert result.trace
    assert seen
    assert result.trace[-1].status == result.status
    assert all(np.isfinite(entry.f) for entry in result.trace)
    assert all(isinstance(entry.request, tide.optim.RequestKind) for entry in result.trace)
    assert all(
        isinstance(entry.line_search_policy, tide.optim.LineSearchPolicy)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.alpha_guess_policy, tide.optim.AlphaGuessPolicy)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.stopping_policy, tide.optim.StoppingPolicy)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.stopping_diagnostics, tide.optim.StoppingDiagnostics)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.line_search_acceptance, tide.optim.LineSearchAcceptance)
        for entry in result.trace
    )
    assert all(
        isinstance(
            entry.request_requirements,
            tide.optim.OptimizerRequestRequirements,
        )
        for entry in result.trace
    )
    assert all(
        entry.expected_evaluation == entry.request_requirements.expected_evaluation
        for entry in result.trace
    )
    assert any(
        entry.expected_evaluation == "value_gradient"
        for entry in result.trace
    )
    assert all(
        isinstance(entry.direction_policy, tide.optim.DirectionPolicy)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.nlcg_beta_policy, tide.optim.NlcgBetaPolicy)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.lbfgs_update_policy, tide.optim.LbfgsUpdatePolicy)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.direction_status, tide.optim.DirectionStatus)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.preconditioner_status, tide.optim.PreconditionerStatus)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.inner_status, tide.optim.InnerCgStatus)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.globalization_policy, tide.optim.GlobalizationPolicy)
        for entry in result.trace
    )
    assert all(
        isinstance(entry.trust_region_status, tide.optim.TrustRegionStatus)
        for entry in result.trace
    )


def test_lbfgs_store_trace_false_keeps_callback_and_last_trace():
    seen = []
    target = np.array([2.0, -1.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    def callback(entry: tide.optim.TraceEntry) -> bool:
        seen.append(entry)
        return False

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        options={
            "store_trace": False,
            "gtol_abs": 1e-8,
        },
        callback=callback,
    )

    assert result.success, result.reason
    assert result.trace == ()
    assert result.trace_policy == "none"
    assert result.n_trace_events == len(seen)
    assert result.n_trace_stored == 0
    assert seen
    assert result.last_trace is not None
    assert result.last_trace.status == result.status
    assert result.last_trace.request == tide.optim.RequestKind.DONE
    assert result.last_trace.n_f == result.n_f
    assert seen[-1].status == result.status


def test_trace_policy_last_keeps_only_final_trace_snapshot():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        options={
            "trace_policy": "last",
            "line_search": "armijo_cubic",
            "initial_step": 0.25,
            "max_iter": 10,
            "gtol_abs": 5e-1,
        },
    )

    assert result.success, result.reason
    assert result.trace_policy == "last"
    assert result.n_trace_events > 1
    assert result.n_trace_stored == 1
    assert result.trace == (result.last_trace,)
    assert result.trace[0].request == tide.optim.RequestKind.DONE


def test_trace_policy_stride_samples_trace_and_keeps_final_snapshot():
    seen = []

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    def callback(entry: tide.optim.TraceEntry) -> bool:
        seen.append(entry)
        return False

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        options={
            "trace_policy": "stride",
            "trace_stride": 3,
            "line_search": "armijo_cubic",
            "initial_step": 0.25,
            "max_iter": 20,
            "gtol_abs": 5e-2,
        },
        callback=callback,
    )

    assert result.success, result.reason
    assert result.trace_policy == "stride"
    assert result.trace_stride == 3
    assert result.n_trace_events == len(seen)
    assert result.n_trace_stored == len(result.trace)
    assert 1 < result.n_trace_stored < result.n_trace_events
    assert result.trace[-1] == result.last_trace
    assert result.trace[-1].request == tide.optim.RequestKind.DONE


def test_lbfgs_initial_callback_can_stop_without_storing_trace():
    seen = []

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    def stop_immediately(entry: tide.optim.TraceEntry) -> bool:
        seen.append(entry)
        return True

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        options={
            "store_trace": False,
            "gtol_abs": 0.0,
        },
        callback=stop_immediately,
    )

    assert result.status == tide.optim.OptimStatus.USER_STOPPED
    assert result.reason == "USER_STOPPED"
    assert result.trace == ()
    assert seen
    assert result.last_trace is not None
    assert result.last_trace.status == tide.optim.OptimStatus.USER_STOPPED
    assert result.last_trace.request != tide.optim.RequestKind.DONE
    assert result.trace_summary.status_count(tide.optim.OptimStatus.USER_STOPPED) == 1
    assert result.trace_summary.failure_reason_count("USER_STOPPED") == 1
    assert result.trace_summary.user_stopped_count == 1
    assert result.trace_summary.failed_count == 0
    assert result.diagnostics["status_report_counts"] == {"USER_STOPPED": 1}
    assert result.diagnostics["failure_reason_report_counts"] == {
        "USER_STOPPED": 1
    }


def test_lbfgs_callback_can_stop_with_explicit_status():
    def stop_after_first(
        entry: tide.optim.TraceEntry,
    ) -> tide.optim.OptimStatus | None:
        if entry.iter >= 1:
            return tide.optim.OptimStatus.USER_STOPPED
        return None

    result = tide.optim.minimize(
        _rosenbrock,
        np.array([1.2, 1.2], dtype=np.float64),
        options={"max_iter": 20},
        callback=stop_after_first,
    )

    assert result.status == tide.optim.OptimStatus.USER_STOPPED
    assert result.reason == "USER_STOPPED"
    assert result.trace_summary.status_count(tide.optim.OptimStatus.USER_STOPPED) == 1
    assert result.trace_summary.failure_reason_count("USER_STOPPED") == 1
    assert result.diagnostics["status_report_counts"]["USER_STOPPED"] == 1
    assert result.diagnostics["failure_reason_report_counts"]["USER_STOPPED"] == 1


def test_lbfgs_callback_accepts_string_stop_status():
    def stop_immediately(_entry: tide.optim.TraceEntry) -> str:
        return "user_stopped"

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        options={"gtol_abs": 0.0},
        callback=stop_immediately,
    )

    assert result.status == tide.optim.OptimStatus.USER_STOPPED
    assert result.reason == "USER_STOPPED"
    assert result.trace_summary.user_stopped_count == 1


def test_lbfgs_callback_rejects_running_stop_status():
    def keep_running(_entry: tide.optim.TraceEntry) -> tide.optim.OptimStatus:
        return tide.optim.OptimStatus.RUNNING

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    with pytest.raises(ValueError, match="terminal status"):
        tide.optim.minimize(
            quadratic,
            np.array([8.0], dtype=np.float64),
            options={"gtol_abs": 0.0},
            callback=keep_running,
        )


def test_lbfgs_stops_when_max_eval_budget_is_reached():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 2.0
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0], dtype=np.float64),
        options={
            "max_eval": 1,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.MAX_EVAL
    assert result.reason == "MAX_EVAL"
    assert result.failure_reason == "MAX_EVAL"
    assert result.line_search_failure is None
    assert result.n_f == 1
    assert result.n_iter == 0
    assert result.trace[-1].request == tide.optim.RequestKind.DONE


def test_lbfgs_xtol_convergence_reports_step_tolerance():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 10.0
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([0.0], dtype=np.float64),
        options={
            "line_search": "armijo_cubic",
            "initial_step": 1e-4,
            "max_iter": 10,
            "gtol_abs": 0.0,
            "x_atol": 1e-2,
        },
    )

    assert result.success, result.reason
    assert result.status == tide.optim.OptimStatus.CONVERGED_XTOL
    assert result.reason == "CONVERGED_XTOL"
    assert result.failure_reason is None
    assert result.line_search_failure is None
    assert result.trace[-1].step_norm <= result.trace[-1].step_tolerance
    assert result.trace[-1].step_tolerance == pytest.approx(1e-2)
    assert result.grad_norm > 1.0


def test_stopping_gradient_only_ignores_xtol_convergence():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 10.0
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([0.0], dtype=np.float64),
        options={
            "line_search": "armijo_cubic",
            "initial_step": 1e-4,
            "max_iter": 3,
            "gtol_abs": 0.0,
            "x_atol": 1e-2,
            "stopping": "gradient_only",
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.MAX_ITER
    assert result.trace[-1].stopping_policy == tide.optim.StoppingPolicy.GRADIENT_ONLY
    assert result.trace[-1].step_norm <= result.trace[-1].step_tolerance
    assert result.grad_norm > 1.0


def test_stopping_initial_relative_f_uses_initial_objective_ratio():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(0.5 * np.dot(x, x)), x.copy()

    result = tide.optim.minimize(
        quadratic,
        np.array([10.0], dtype=np.float64),
        method="pstd",
        options={
            "line_search": "armijo_cubic",
            "initial_step": 0.25,
            "max_iter": 10,
            "gtol_abs": 0.0,
            "f_rtol": 0.5,
            "stopping": "initial_relative_f",
        },
    )

    assert result.success, result.reason
    assert result.status == tide.optim.OptimStatus.CONVERGED_FTOL
    assert result.f <= 25.0
    assert result.grad_norm > 1.0
    assert all(
        entry.stopping_policy == tide.optim.StoppingPolicy.INITIAL_RELATIVE_F
        for entry in result.trace
    )


def test_lbfgs_line_search_maxls_failure_is_reported_on_result():
    def flat_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 1.0, np.ones_like(x)

    result = tide.optim.minimize(
        flat_objective,
        np.array([0.0], dtype=np.float64),
        options={
            "line_search": "armijo_cubic",
            "max_line_search": 1,
            "max_iter": 5,
            "gtol_abs": 0.0,
            "accept_decrease_after_maxls": False,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.LINE_SEARCH_FAILED
    assert result.reason == "LINE_SEARCH_FAILED"
    assert result.failure_reason == "LINE_SEARCH_FAILED_MAXLS"
    assert result.line_search_failure == tide.optim.LineSearchStatus.FAILED_MAXLS
    assert result.line_search_diagnostics.failed
    assert result.line_search_diagnostics.status == (
        tide.optim.LineSearchStatus.FAILED_MAXLS
    )
    assert result.to_dict()["line_search_diagnostics"]["failed"] is True


def test_lbfgs_line_search_alpha_bounds_failure_is_reported_on_result():
    def flat_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        return 1.0, np.ones_like(x)

    result = tide.optim.minimize(
        flat_objective,
        np.array([0.0], dtype=np.float64),
        options={
            "line_search": "armijo_cubic",
            "initial_step": 1.0,
            "alpha_min": 0.75,
            "alpha_max": 1.25,
            "max_iter": 5,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.LINE_SEARCH_FAILED
    assert result.failure_reason == "LINE_SEARCH_FAILED_ALPHA_BOUNDS"
    assert (
        result.line_search_failure
        == tide.optim.LineSearchStatus.FAILED_ALPHA_BOUNDS
    )


def test_lbfgs_nonfinite_accepted_trial_failure_is_reported_on_result():
    def value(x: np.ndarray) -> float:
        return -1.0

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        if x[0] == 0.0:
            return 1.0, np.ones_like(x)
        return float("nan"), np.ones_like(x)

    result = tide.optim.minimize(
        value_grad,
        np.array([0.0], dtype=np.float64),
        value=value,
        options={
            "line_search": "armijo_cubic",
            "max_iter": 5,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.NONFINITE
    assert result.reason == "NONFINITE"
    assert result.failure_reason == "NONFINITE_TRIAL"
    assert result.line_search_failure == tide.optim.LineSearchStatus.FAILED_NONFINITE
    assert "nonfinite_trial" in result.warnings


def test_lbfgs_armijo_cubic_policy_is_selectable_and_diagnostic():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - np.array([2.0, -1.0])
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        options={
            "line_search": "armijo_cubic",
            "initial_step": 10.0,
            "max_iter": 30,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, np.array([2.0, -1.0]), atol=1e-6)
    assert any(
        entry.line_search_status == tide.optim.LineSearchStatus.REJECTED_ARMIJO
        for entry in result.trace
    )


def test_lbfgs_armijo_cubic_can_use_value_only_rejected_trials():
    counts = {"value": 0, "value_grad": 0}
    target = np.array([2.0, -1.0], dtype=np.float64)

    def value(x: np.ndarray) -> float:
        counts["value"] += 1
        residual = x - target
        return float(0.5 * np.dot(residual, residual))

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        counts["value_grad"] += 1
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        value_grad,
        np.array([8.0, -6.0], dtype=np.float64),
        value=value,
        options={
            "line_search": "armijo_cubic",
            "initial_step": 10.0,
            "max_iter": 30,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-6)
    assert counts["value"] > 0
    assert counts["value_grad"] == result.n_g
    assert result.n_f > result.n_g
    assert any(entry.request == tide.optim.RequestKind.EVALUATE_F for entry in result.trace)
    rejected = [
        entry
        for entry in result.trace
        if entry.line_search_status == tide.optim.LineSearchStatus.REJECTED_ARMIJO
    ]
    accepted = [
        entry
        for entry in result.trace
        if entry.line_search_status == tide.optim.LineSearchStatus.ACCEPTED
    ]
    assert rejected
    assert accepted
    assert all(entry.trial_f > entry.line_search_armijo_rhs for entry in rejected)
    assert all(entry.trial_f <= entry.line_search_armijo_rhs for entry in accepted)
    assert all(
        entry.line_search_acceptance == tide.optim.LineSearchAcceptance.ARMIJO
        for entry in accepted
    )


def test_lbfgs_strong_wolfe_policy_converges_with_gradient_trials():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - np.array([2.0, -1.0])
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        options={
            "line_search": "strong_wolfe",
            "initial_step": 10.0,
            "max_iter": 30,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, np.array([2.0, -1.0]), atol=1e-6)
    assert result.n_f == result.n_g
    assert any(
        entry.line_search_status
        in {
            tide.optim.LineSearchStatus.REJECTED_ARMIJO,
            tide.optim.LineSearchStatus.REJECTED_CURVATURE,
        }
        for entry in result.trace
    )


def test_lbfgs_more_thuente_policy_converges_with_gradient_trials():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - np.array([2.0, -1.0])
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        options={
            "line_search": "more_thuente",
            "initial_step": 10.0,
            "max_iter": 30,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, np.array([2.0, -1.0]), atol=1e-6)
    assert result.n_f == result.n_g
    assert result.last_trace.line_search_policy == (
        tide.optim.LineSearchPolicy.MORE_THUENTE
    )
    assert result.backend_options["line_search_policy"] == "MORE_THUENTE"
    assert any(
        entry.line_search_acceptance == tide.optim.LineSearchAcceptance.STRONG_WOLFE
        for entry in result.trace
    )
    assert any(
        entry.line_search_status
        in {
            tide.optim.LineSearchStatus.REJECTED_ARMIJO,
            tide.optim.LineSearchStatus.REJECTED_CURVATURE,
        }
        for entry in result.trace
    )


def test_lbfgs_hager_zhang_policy_converges_with_gradient_trials():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - np.array([2.0, -1.0])
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        options={
            "line_search": "hager_zhang",
            "initial_step": 10.0,
            "max_iter": 30,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, np.array([2.0, -1.0]), atol=1e-6)
    assert result.n_f == result.n_g
    assert all(
        entry.line_search_policy == tide.optim.LineSearchPolicy.HAGER_ZHANG
        for entry in result.trace
    )
    assert result.trace[-1].line_search_acceptance in {
        tide.optim.LineSearchAcceptance.WEAK_WOLFE,
        tide.optim.LineSearchAcceptance.APPROXIMATE_WOLFE,
    }
    assert any(
        entry.line_search_status
        in {
            tide.optim.LineSearchStatus.REJECTED_ARMIJO,
            tide.optim.LineSearchStatus.REJECTED_CURVATURE,
        }
        for entry in result.trace
    )


def test_lbfgs_hager_zhang_reports_approximate_wolfe_acceptance():
    def synthetic_roundoff_case(x: np.ndarray) -> tuple[float, np.ndarray]:
        gradient = np.array([-1.0 if x[0] < 0.5 else 0.0], dtype=np.float64)
        return 1000.0, gradient

    result = tide.optim.minimize(
        synthetic_roundoff_case,
        np.array([0.0], dtype=np.float64),
        options={
            "line_search": "hager_zhang",
            "initial_step": 1.0,
            "max_iter": 2,
            "gtol_abs": 1e-12,
        },
    )

    assert result.success, result.reason
    assert result.trace[-1].line_search_status == tide.optim.LineSearchStatus.ACCEPTED
    assert (
        result.trace[-1].line_search_acceptance
        == tide.optim.LineSearchAcceptance.APPROXIMATE_WOLFE
    )
    assert result.trace[-1].trial_f > result.trace[-1].line_search_armijo_rhs


def test_static_line_search_accepts_single_gradient_trial():
    target = np.array([2.0, -1.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        method="pstd",
        options={
            "line_search": "static",
            "initial_step": 1.0,
            "max_iter": 5,
            "gtol_abs": 1e-12,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-12)
    assert result.n_iter == 1
    assert result.n_f == result.n_g == 2
    assert result.n_trace_events == 2
    assert result.last_trace is not None
    assert result.last_trace.line_search_policy == tide.optim.LineSearchPolicy.STATIC
    assert (
        result.last_trace.line_search_acceptance
        == tide.optim.LineSearchAcceptance.STATIC
    )
    assert result.last_trace.line_search_iter == 0
    assert any(
        entry.request == tide.optim.RequestKind.EVALUATE_FG
        for entry in result.trace
    )


def test_static_line_search_reports_nonfinite_trial():
    def nonfinite_trial(x: np.ndarray) -> tuple[float, np.ndarray]:
        if x[0] == 0.0:
            return 1.0, np.array([-1.0], dtype=np.float64)
        return float("nan"), np.array([0.0], dtype=np.float64)

    result = tide.optim.minimize(
        nonfinite_trial,
        np.array([0.0], dtype=np.float64),
        method="pstd",
        options={
            "line_search": "static",
            "initial_step": 1.0,
            "max_iter": 5,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.NONFINITE
    assert result.failure_reason == "NONFINITE_TRIAL"
    assert result.line_search_failure == tide.optim.LineSearchStatus.FAILED_NONFINITE
    assert "nonfinite_trial" in result.warnings
    assert result.last_trace is not None
    assert result.last_trace.line_search_policy == tide.optim.LineSearchPolicy.STATIC


def test_lbfgs_nonmonotone_armijo_policy_uses_value_only_trials():
    counts = {"value": 0, "value_grad": 0}
    target = np.array([2.0, -1.0], dtype=np.float64)

    def value(x: np.ndarray) -> float:
        counts["value"] += 1
        residual = x - target
        return float(0.5 * np.dot(residual, residual))

    def value_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        counts["value_grad"] += 1
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        value_grad,
        np.array([8.0, -6.0], dtype=np.float64),
        value=value,
        options={
            "line_search": "nonmonotone_armijo",
            "nonmonotone_window": 4,
            "initial_step": 10.0,
            "max_iter": 30,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-6)
    assert counts["value"] > 0
    assert counts["value_grad"] == result.n_g
    assert result.n_f > result.n_g
    assert all(
        entry.line_search_policy == tide.optim.LineSearchPolicy.NONMONOTONE_ARMIJO
        for entry in result.trace
    )
    assert any(
        entry.line_search_reference >= entry.f
        for entry in result.trace
        if entry.request == tide.optim.RequestKind.EVALUATE_F
    )
    assert any(
        entry.line_search_status == tide.optim.LineSearchStatus.REJECTED_ARMIJO
        for entry in result.trace
    )


def test_pstd_uses_steepest_descent_direction_without_lbfgs_pairs():
    target = np.array([2.0, -1.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(residual, residual)), residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        method="pstd",
        options={
            "line_search": "armijo_cubic",
            "initial_step": 1.0,
            "max_iter": 10,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert all(
        entry.direction_policy == tide.optim.DirectionPolicy.STEEPEST_DESCENT
        for entry in result.trace
    )
    assert all(entry.direction_diagnostics.descent_direction for entry in result.trace)
    assert all(entry.history_size == 0 for entry in result.trace)
    assert not any(
        entry.pair_status == tide.optim.PairStatus.STORED
        for entry in result.trace
    )


def test_pnlcg_dai_yuan_direction_updates_without_lbfgs_pairs():
    scale = np.array([10.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -2.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        method="pnlcg",
        options={
            "max_iter": 80,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-7)
    assert all(
        entry.direction_policy == tide.optim.DirectionPolicy.NLCG
        for entry in result.trace
    )
    assert all(
        entry.nlcg_beta_policy == tide.optim.NlcgBetaPolicy.DAI_YUAN
        for entry in result.trace
    )
    assert all(
        entry.line_search_policy == tide.optim.LineSearchPolicy.HAGER_ZHANG
        for entry in result.trace
    )
    assert any(
        entry.direction_status == tide.optim.DirectionStatus.UPDATE
        for entry in result.trace
    )
    assert any(entry.direction_diagnostics.updated for entry in result.trace)
    assert any(entry.direction_diagnostics.uses_nlcg_beta for entry in result.trace)
    assert any(entry.direction_beta > 0.0 for entry in result.trace)
    assert all(entry.history_size == 0 for entry in result.trace)
    assert not any(
        entry.pair_status == tide.optim.PairStatus.STORED
        for entry in result.trace
    )


@pytest.mark.parametrize(
    ("nlcg_beta", "expected_policy"),
    [
        ("fletcher_reeves", tide.optim.NlcgBetaPolicy.FLETCHER_REEVES),
        ("polak_ribiere_plus", tide.optim.NlcgBetaPolicy.POLAK_RIBIERE_PLUS),
        ("hager_zhang", tide.optim.NlcgBetaPolicy.HAGER_ZHANG),
    ],
)
def test_pnlcg_beta_policy_is_selectable_and_diagnostic(
    nlcg_beta: str,
    expected_policy: tide.optim.NlcgBetaPolicy,
):
    scale = np.array([20.0, 2.0, 0.5], dtype=np.float64)
    target = np.array([1.0, -2.0, 0.25], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0, 2.0], dtype=np.float64),
        method="pnlcg",
        options={
            "nlcg_beta": nlcg_beta,
            "max_iter": 120,
            "gtol_abs": 1e-6,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-5)
    assert all(entry.nlcg_beta_policy == expected_policy for entry in result.trace)
    assert any(
        entry.direction_status == tide.optim.DirectionStatus.UPDATE
        for entry in result.trace
    )
    if expected_policy == tide.optim.NlcgBetaPolicy.HAGER_ZHANG:
        assert any(entry.direction_beta < 0.0 for entry in result.trace)
    assert all(entry.history_size == 0 for entry in result.trace)


def test_trn_uses_typed_hessian_vector_request_and_inner_cg_diagnostics():
    scale = np.array([10.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -2.0], dtype=np.float64)
    calls = {"hvp": 0}

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    def hessian_vector(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        calls["hvp"] += 1
        return scale * vector

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        method="trn",
        hessian_vector=hessian_vector,
        options={
            "max_iter": 20,
            "max_inner_iter": 4,
            "inner_rtol": 1e-12,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert calls["hvp"] == result.n_hvp
    assert result.n_hvp > 0
    assert any(
        entry.request == tide.optim.RequestKind.EVALUATE_HV
        for entry in result.trace
    )
    assert all(
        entry.direction_policy == tide.optim.DirectionPolicy.TRUNCATED_NEWTON
        for entry in result.trace
    )
    assert any(
        entry.inner_status == tide.optim.InnerCgStatus.FORCING_REACHED
        for entry in result.trace
    )
    assert max(entry.inner_iter for entry in result.trace) > 0
    assert result.inner_solve_diagnostics.n_hvp == result.n_hvp
    assert result.inner_solve_diagnostics.converged
    assert result.inner_solve_diagnostics.failed is False
    assert any(entry.inner_solve_diagnostics.active for entry in result.trace)
    assert all(entry.history_size == 0 for entry in result.trace)


def test_trn_nonfinite_hvp_reports_inner_cg_failure_reason():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(0.5 * np.dot(x, x)), x.copy()

    def nonfinite_hvp(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return np.full_like(vector, np.nan)

    result = tide.optim.minimize(
        quadratic,
        np.array([2.0, -1.0], dtype=np.float64),
        method="trn",
        hessian_vector=nonfinite_hvp,
        options={
            "max_iter": 5,
            "max_inner_iter": 4,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.INNER_CG_FAILED
    assert result.failure_reason == "INNER_CG_NONFINITE_HVP"
    assert result.last_trace is not None
    assert result.last_trace.inner_status == tide.optim.InnerCgStatus.NONFINITE_HVP
    assert result.last_trace.inner_solve_diagnostics.failed
    assert result.last_trace.inner_solve_diagnostics.hvp_failed
    assert result.inner_solve_diagnostics.failed
    assert result.inner_solve_diagnostics.hvp_failed
    assert result.last_trace.has_warning(tide.optim.WarningFlag.INNER_CG)
    assert "inner_cg_warning" in result.warnings


def test_trn_trust_region_globalization_reports_radius_and_ratio():
    scale = np.array([10.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -2.0], dtype=np.float64)

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    def hessian_vector(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return scale * vector

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        method="trn",
        hessian_vector=hessian_vector,
        options={
            "globalization": "trust_region",
            "initial_trust_radius": 0.5,
            "max_trust_radius": 8.0,
            "max_iter": 50,
            "max_inner_iter": 4,
            "inner_rtol": 1e-12,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert result.last_trace is not None
    assert result.last_trace.globalization_policy == tide.optim.GlobalizationPolicy.TRUST_REGION
    assert result.last_trace.trust_region_status == tide.optim.TrustRegionStatus.ACCEPTED
    assert result.last_trace.trust_radius > 0.5
    assert result.last_trace.trust_ratio == pytest.approx(1.0)
    assert result.last_trace.predicted_reduction > 0.0
    assert result.last_trace.actual_reduction == pytest.approx(
        result.last_trace.predicted_reduction
    )
    assert result.trust_region_diagnostics.active
    assert result.trust_region_diagnostics.accepted
    assert not result.trust_region_diagnostics.failed
    assert result.trust_region_diagnostics.has_model_reduction
    assert result.last_trace.trust_region_diagnostics.accepted
    assert result.last_trace.trust_region_diagnostics.reduction_matches_model
    assert any(
        entry.inner_status == tide.optim.InnerCgStatus.TRUST_BOUNDARY
        for entry in result.trace
    )
    assert any(entry.inner_solve_diagnostics.trust_boundary for entry in result.trace)
    assert all(
        entry.globalization_policy == tide.optim.GlobalizationPolicy.TRUST_REGION
        for entry in result.trace
    )


def test_trn_trust_region_nonfinite_trial_reports_failure_reason():
    initial = np.array([2.0], dtype=np.float64)

    def nonfinite_trial_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        if np.allclose(x, initial):
            return float(0.5 * np.dot(x, x)), x.copy()
        return float("nan"), np.zeros_like(x)

    def hessian_vector(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return vector.copy()

    result = tide.optim.minimize(
        nonfinite_trial_objective,
        initial,
        method="trn",
        hessian_vector=hessian_vector,
        options={
            "globalization": "trust_region",
            "max_line_search": 1,
            "max_inner_iter": 4,
            "gtol_abs": 0.0,
        },
    )

    assert not result.success
    assert result.status == tide.optim.OptimStatus.TRUST_REGION_FAILED
    assert result.failure_reason == "TRUST_REGION_FAILED_NONFINITE"
    assert result.last_trace is not None
    assert (
        result.last_trace.trust_region_status
        == tide.optim.TrustRegionStatus.FAILED_NONFINITE
    )
    assert result.trust_region_diagnostics.failed
    assert result.last_trace.trust_region_diagnostics.failed
    assert "nonfinite_trial" in result.warnings


def test_trust_region_globalization_is_limited_to_trn_methods():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 1.0
        return float(0.5 * np.dot(residual, residual)), residual

    with pytest.raises(NotImplementedError, match="trust_region"):
        tide.optim.minimize(
            quadratic,
            np.array([2.0], dtype=np.float64),
            method="lbfgs",
            options={"globalization": "trust_region"},
        )


def test_ptrn_uses_preconditioner_inside_inner_cg():
    scale = np.array([10.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -2.0], dtype=np.float64)
    calls = {"hvp": 0, "preconditioner": 0}

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    def hessian_vector(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        calls["hvp"] += 1
        return scale * vector

    def preconditioner(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        calls["preconditioner"] += 1
        return vector / scale

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        method="ptrn",
        hessian_vector=hessian_vector,
        preconditioner=preconditioner,
        options={
            "max_iter": 20,
            "max_inner_iter": 4,
            "inner_rtol": 1e-12,
            "gtol_abs": 1e-10,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert calls["hvp"] == result.n_hvp
    assert calls["preconditioner"] == result.n_prec
    assert result.n_hvp > 0
    assert result.n_prec > 0
    assert result.inner_solve_diagnostics.n_hvp == result.n_hvp
    assert result.inner_solve_diagnostics.n_prec == result.n_prec
    assert result.inner_solve_diagnostics.converged
    assert result.inner_solve_diagnostics.preconditioner_applied
    assert any(
        entry.request == tide.optim.RequestKind.APPLY_PRECONDITIONER
        for entry in result.trace
    )
    assert any(
        entry.preconditioner_status == tide.optim.PreconditionerStatus.APPLIED
        for entry in result.trace
    )
    assert all(
        entry.direction_policy
        == tide.optim.DirectionPolicy.PRECONDITIONED_TRUNCATED_NEWTON
        for entry in result.trace
    )
    assert any(
        entry.inner_status == tide.optim.InnerCgStatus.FORCING_REACHED
        for entry in result.trace
    )


def test_trn_requires_hessian_vector_callable():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 1.0
        return float(0.5 * np.dot(residual, residual)), residual

    with pytest.raises(ValueError, match="hessian_vector callable"):
        tide.optim.minimize(
            quadratic,
            np.array([2.0], dtype=np.float64),
            method="trn",
            options={"max_iter": 2},
        )


def test_plbfgs_uses_typed_preconditioner_request_and_stores_pairs():
    scale = np.array([10.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -2.0], dtype=np.float64)
    calls = {"preconditioner": 0}

    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(0.5 * np.dot(scale * residual, residual)), scale * residual

    def preconditioner(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        calls["preconditioner"] += 1
        return vector / scale

    result = tide.optim.minimize(
        quadratic,
        np.array([8.0, -6.0], dtype=np.float64),
        method="plbfgs",
        preconditioner=preconditioner,
        options={
            "max_iter": 20,
            "gtol_abs": 1e-8,
        },
    )

    assert result.success, result.reason
    np.testing.assert_allclose(result.x, target, atol=1e-10)
    assert calls["preconditioner"] == result.n_prec
    assert result.n_prec > 0
    assert any(
        entry.request == tide.optim.RequestKind.APPLY_PRECONDITIONER
        for entry in result.trace
    )
    assert any(
        entry.preconditioner_status == tide.optim.PreconditionerStatus.APPLIED
        for entry in result.trace
    )
    assert all(
        entry.direction_policy == tide.optim.DirectionPolicy.PRECONDITIONED_LBFGS
        for entry in result.trace
    )
    assert any(
        entry.pair_status == tide.optim.PairStatus.STORED
        for entry in result.trace
    )
    assert max(entry.preconditioner_dot for entry in result.trace) > 0.0


def test_plbfgs_skips_pair_when_preconditioner_is_not_positive():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 1.0
        return float(0.5 * np.dot(residual, residual)), residual

    def non_spd_preconditioner(_x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return -vector

    result = tide.optim.minimize(
        quadratic,
        np.array([2.0], dtype=np.float64),
        method="plbfgs",
        preconditioner=non_spd_preconditioner,
        options={
            "max_iter": 5,
            "gtol_abs": 1e-12,
        },
    )

    assert result.success, result.reason
    assert any(
        entry.preconditioner_status
        == tide.optim.PreconditionerStatus.SKIPPED_NOT_POSITIVE
        for entry in result.trace
    )
    assert any(
        entry.pair_status == tide.optim.PairStatus.SKIPPED_PRECONDITIONER
        for entry in result.trace
    )
    assert all(entry.history_size == 0 for entry in result.trace)
    assert "preconditioner_skipped" in result.warnings
    assert "lbfgs_pair_skipped" in result.warnings


def test_plbfgs_skips_pair_when_preconditioner_is_nonfinite():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - 1.0
        return float(0.5 * np.dot(residual, residual)), residual

    def nonfinite_preconditioner(
        _x: np.ndarray, vector: np.ndarray
    ) -> np.ndarray:
        return np.full_like(vector, np.nan)

    result = tide.optim.minimize(
        quadratic,
        np.array([2.0], dtype=np.float64),
        method="plbfgs",
        preconditioner=nonfinite_preconditioner,
        options={
            "max_iter": 5,
            "gtol_abs": 1e-12,
        },
    )

    assert result.success, result.reason
    assert any(
        entry.preconditioner_status
        == tide.optim.PreconditionerStatus.SKIPPED_NONFINITE
        for entry in result.trace
    )
    assert any(
        entry.pair_status == tide.optim.PairStatus.SKIPPED_PRECONDITIONER
        for entry in result.trace
    )
    assert all(entry.history_size == 0 for entry in result.trace)
    assert "preconditioner_skipped" in result.warnings
    assert "lbfgs_pair_skipped" in result.warnings


def test_plbfgs_requires_preconditioner_callable():
    def quadratic(x: np.ndarray) -> tuple[float, np.ndarray]:
        return float(np.dot(x, x)), 2.0 * x

    with pytest.raises(ValueError, match="preconditioner callable"):
        tide.optim.minimize(
            quadratic,
            np.array([1.0, -1.0], dtype=np.float64),
            method="plbfgs",
            options={"max_iter": 2},
        )
