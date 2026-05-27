#ifndef TIDE_OPTIM_OPTIM_H
#define TIDE_OPTIM_OPTIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  TIDE_OPTIM_REQUEST_ERROR = -1,
  TIDE_OPTIM_REQUEST_EVALUATE_FG = 1,
  TIDE_OPTIM_REQUEST_DONE = 2,
  TIDE_OPTIM_REQUEST_EVALUATE_F = 3,
  TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER = 4,
  TIDE_OPTIM_REQUEST_EVALUATE_HV = 5,
};

enum {
  TIDE_OPTIM_STATUS_RUNNING = 0,
  TIDE_OPTIM_STATUS_CONVERGED_GRADIENT = 1,
  TIDE_OPTIM_STATUS_CONVERGED_FTOL = 2,
  TIDE_OPTIM_STATUS_MAX_ITER = 3,
  TIDE_OPTIM_STATUS_LINE_SEARCH_FAILED = 4,
  TIDE_OPTIM_STATUS_NONFINITE = 5,
  TIDE_OPTIM_STATUS_NON_DESCENT_DIRECTION = 6,
  TIDE_OPTIM_STATUS_INVALID_ARGUMENT = 7,
  TIDE_OPTIM_STATUS_USER_STOPPED = 8,
  TIDE_OPTIM_STATUS_CONVERGED_XTOL = 9,
  TIDE_OPTIM_STATUS_MAX_EVAL = 10,
  TIDE_OPTIM_STATUS_INNER_CG_FAILED = 11,
  TIDE_OPTIM_STATUS_TRUST_REGION_FAILED = 12,
};

enum {
  TIDE_OPTIM_LINE_SEARCH_STARTED = 0,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTED = 1,
  TIDE_OPTIM_LINE_SEARCH_REJECTED_ARMIJO = 2,
  TIDE_OPTIM_LINE_SEARCH_REJECTED_CURVATURE = 3,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTED_DECREASE_AFTER_MAXLS = 4,
  TIDE_OPTIM_LINE_SEARCH_FAILED_MAXLS = 5,
  TIDE_OPTIM_LINE_SEARCH_FAILED_NONFINITE = 6,
  TIDE_OPTIM_LINE_SEARCH_FAILED_ALPHA_BOUNDS = 7,
};

enum {
  TIDE_OPTIM_LINE_SEARCH_POLICY_LEGACY_WEAK_WOLFE = 0,
  TIDE_OPTIM_LINE_SEARCH_POLICY_ARMIJO_CUBIC = 1,
  TIDE_OPTIM_LINE_SEARCH_POLICY_STRONG_WOLFE = 2,
  TIDE_OPTIM_LINE_SEARCH_POLICY_MORE_THUENTE = 3,
  TIDE_OPTIM_LINE_SEARCH_POLICY_NONMONOTONE_ARMIJO = 4,
  TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG = 5,
  TIDE_OPTIM_LINE_SEARCH_POLICY_STATIC = 6,
};

enum {
  TIDE_OPTIM_ALPHA_GUESS_INITIAL = 0,
  TIDE_OPTIM_ALPHA_GUESS_PREVIOUS = 1,
  TIDE_OPTIM_ALPHA_GUESS_BARZILAI_BORWEIN = 2,
};

enum {
  TIDE_OPTIM_STOPPING_STANDARD = 0,
  TIDE_OPTIM_STOPPING_GRADIENT_ONLY = 1,
  TIDE_OPTIM_STOPPING_INITIAL_RELATIVE_F = 2,
};

enum {
  TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE = 0,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_ARMIJO = 1,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_WEAK_WOLFE = 2,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_STRONG_WOLFE = 3,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_APPROXIMATE_WOLFE = 4,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_DECREASE_FALLBACK = 5,
  TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_STATIC = 6,
};

enum {
  TIDE_OPTIM_PAIR_NONE = 0,
  TIDE_OPTIM_PAIR_STORED = 1,
  TIDE_OPTIM_PAIR_SKIPPED_BAD_CURVATURE = 2,
  TIDE_OPTIM_PAIR_SKIPPED_NONFINITE = 3,
  TIDE_OPTIM_PAIR_SKIPPED_LINE_SEARCH_FALLBACK = 4,
  TIDE_OPTIM_PAIR_SKIPPED_BOUNDS_PROJECTION = 5,
  TIDE_OPTIM_PAIR_SKIPPED_PRECONDITIONER = 6,
  TIDE_OPTIM_PAIR_REGULARIZED_STORED = 7,
};

enum {
  TIDE_OPTIM_LBFGS_UPDATE_SKIP = 0,
  TIDE_OPTIM_LBFGS_UPDATE_REGULARIZE = 1,
};

enum {
  TIDE_OPTIM_BOUNDS_NONE = 0,
  TIDE_OPTIM_BOUNDS_PROJECTED_TRIAL = 1,
  TIDE_OPTIM_BOUNDS_PROJECTED_GRADIENT = 2,
};

enum {
  TIDE_OPTIM_DIRECTION_STEEPEST_DESCENT = 0,
  TIDE_OPTIM_DIRECTION_LBFGS = 1,
  TIDE_OPTIM_DIRECTION_NLCG = 2,
  TIDE_OPTIM_DIRECTION_NLCG_DAI_YUAN = 2,
  TIDE_OPTIM_DIRECTION_PRECONDITIONED_LBFGS = 3,
  TIDE_OPTIM_DIRECTION_TRUNCATED_NEWTON = 4,
  TIDE_OPTIM_DIRECTION_PRECONDITIONED_TRUNCATED_NEWTON = 5,
};

enum {
  TIDE_OPTIM_NLCG_BETA_DAI_YUAN = 0,
  TIDE_OPTIM_NLCG_BETA_FLETCHER_REEVES = 1,
  TIDE_OPTIM_NLCG_BETA_POLAK_RIBIERE_PLUS = 2,
  TIDE_OPTIM_NLCG_BETA_HAGER_ZHANG = 3,
};

enum {
  TIDE_OPTIM_DIRECTION_STATUS_INITIAL = 0,
  TIDE_OPTIM_DIRECTION_STATUS_UPDATE = 1,
  TIDE_OPTIM_DIRECTION_STATUS_RESTART_DENOMINATOR = 2,
  TIDE_OPTIM_DIRECTION_STATUS_RESTART_NONFINITE = 3,
  TIDE_OPTIM_DIRECTION_STATUS_RESTART_NON_DESCENT = 4,
};

enum {
  TIDE_OPTIM_PRECONDITIONER_NONE = 0,
  TIDE_OPTIM_PRECONDITIONER_APPLIED = 1,
  TIDE_OPTIM_PRECONDITIONER_SKIPPED_NONFINITE = 2,
  TIDE_OPTIM_PRECONDITIONER_SKIPPED_NOT_POSITIVE = 3,
};

enum {
  TIDE_OPTIM_INNER_CG_NONE = 0,
  TIDE_OPTIM_INNER_CG_STARTED = 1,
  TIDE_OPTIM_INNER_CG_FORCING_REACHED = 2,
  TIDE_OPTIM_INNER_CG_NEGATIVE_CURVATURE = 3,
  TIDE_OPTIM_INNER_CG_ZERO_CURVATURE = 4,
  TIDE_OPTIM_INNER_CG_MAX_ITER = 5,
  TIDE_OPTIM_INNER_CG_NONFINITE_HVP = 6,
  TIDE_OPTIM_INNER_CG_PRECONDITIONER_BREAKDOWN = 7,
  TIDE_OPTIM_INNER_CG_TRUST_BOUNDARY = 8,
};

enum {
  TIDE_OPTIM_GLOBALIZATION_LINE_SEARCH = 0,
  TIDE_OPTIM_GLOBALIZATION_TRUST_REGION = 1,
};

enum {
  TIDE_OPTIM_TRUST_REGION_NONE = 0,
  TIDE_OPTIM_TRUST_REGION_STARTED = 1,
  TIDE_OPTIM_TRUST_REGION_ACCEPTED = 2,
  TIDE_OPTIM_TRUST_REGION_REJECTED = 3,
  TIDE_OPTIM_TRUST_REGION_FAILED_NONFINITE = 4,
  TIDE_OPTIM_TRUST_REGION_FAILED_PREDICTED_REDUCTION = 5,
};

enum {
  TIDE_OPTIM_WARNING_NONFINITE_TRIAL = 1 << 0,
  TIDE_OPTIM_WARNING_NON_DESCENT_DIRECTION_RESET = 1 << 1,
  TIDE_OPTIM_WARNING_ACCEPTED_DECREASE_AFTER_MAX_LINE_SEARCH = 1 << 2,
  TIDE_OPTIM_WARNING_LBFGS_PAIR_SKIPPED = 1 << 3,
  TIDE_OPTIM_WARNING_PRECONDITIONER_SKIPPED = 1 << 4,
  TIDE_OPTIM_WARNING_INNER_CG = 1 << 5,
  TIDE_OPTIM_WARNING_LBFGS_PAIR_REGULARIZED = 1 << 6,
};

enum {
  TIDE_OPTIM_OPTIONS_VALIDATION_OK = 0,
  TIDE_OPTIM_OPTIONS_VALIDATION_DIMENSION = 1,
  TIDE_OPTIM_OPTIONS_VALIDATION_HISTORY_SIZE = 2,
  TIDE_OPTIM_OPTIONS_VALIDATION_MAX_ITER = 3,
  TIDE_OPTIM_OPTIONS_VALIDATION_MAX_LINE_SEARCH = 4,
  TIDE_OPTIM_OPTIONS_VALIDATION_MAX_EVAL = 5,
  TIDE_OPTIM_OPTIONS_VALIDATION_MAX_INNER_ITER = 6,
  TIDE_OPTIM_OPTIONS_VALIDATION_NONMONOTONE_WINDOW = 7,
  TIDE_OPTIM_OPTIONS_VALIDATION_INITIAL_STEP = 8,
  TIDE_OPTIM_OPTIONS_VALIDATION_WOLFE_PARAMETERS = 9,
  TIDE_OPTIM_OPTIONS_VALIDATION_GROWTH = 10,
  TIDE_OPTIM_OPTIONS_VALIDATION_LINE_SEARCH_POLICY = 11,
  TIDE_OPTIM_OPTIONS_VALIDATION_DIRECTION_POLICY = 12,
  TIDE_OPTIM_OPTIONS_VALIDATION_NLCG_BETA_POLICY = 13,
  TIDE_OPTIM_OPTIONS_VALIDATION_LBFGS_UPDATE_POLICY = 14,
  TIDE_OPTIM_OPTIONS_VALIDATION_GLOBALIZATION_POLICY = 15,
  TIDE_OPTIM_OPTIONS_VALIDATION_ALPHA_GUESS_POLICY = 16,
  TIDE_OPTIM_OPTIONS_VALIDATION_STOPPING_POLICY = 17,
  TIDE_OPTIM_OPTIONS_VALIDATION_COST_MODEL = 18,
  TIDE_OPTIM_OPTIONS_VALIDATION_ALPHA_BOUNDS = 19,
  TIDE_OPTIM_OPTIONS_VALIDATION_INNER_TOLERANCE = 20,
  TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_RADIUS = 21,
  TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_ETA = 22,
  TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_SHRINK = 23,
  TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_GROW = 24,
  TIDE_OPTIM_OPTIONS_VALIDATION_CURVATURE_EPS = 25,
  TIDE_OPTIM_OPTIONS_VALIDATION_GAMMA_BOUNDS = 26,
  TIDE_OPTIM_OPTIONS_VALIDATION_TOLERANCES = 27,
  TIDE_OPTIM_OPTIONS_VALIDATION_ARMIJO_SHRINK = 28,
  TIDE_OPTIM_OPTIONS_VALIDATION_BOUND_MARGIN = 29,
  TIDE_OPTIM_OPTIONS_VALIDATION_BOUNDS_STRATEGY = 30,
  TIDE_OPTIM_OPTIONS_VALIDATION_NULL_OPTIONS = 31,
  TIDE_OPTIM_OPTIONS_VALIDATION_TRACE_POLICY = 32,
  TIDE_OPTIM_OPTIONS_VALIDATION_TRACE_STRIDE = 33,
};

typedef struct {
  int64_t n;
  int64_t history_size;
  int64_t max_iter;
  int64_t max_line_search;
  int64_t max_eval;
  int64_t max_inner_iter;
  int64_t nonmonotone_window;
  int32_t line_search_policy;
  int32_t direction_policy;
  int32_t nlcg_beta_policy;
  int32_t lbfgs_update_policy;
  int32_t globalization_policy;
  int32_t alpha_guess_policy;
  int32_t stopping_policy;
  double initial_step;
  double c1;
  double c2;
  double growth;
  double alpha_min;
  double alpha_max;
  double gtol_abs;
  double gtol_rel;
  double f_atol;
  double f_rtol;
  double x_atol;
  double x_rtol;
  double inner_rtol;
  double inner_atol;
  double initial_trust_radius;
  double max_trust_radius;
  double trust_eta;
  double trust_shrink;
  double trust_grow;
  double curvature_eps;
  double gamma_min;
  double gamma_max;
  double armijo_shrink_min;
  double armijo_shrink_max;
  double bound_margin;
  int32_t bounds_strategy;
  int32_t accept_decrease_after_maxls;
} tide_optim_lbfgs_options;

typedef tide_optim_lbfgs_options tide_optim_options;

typedef struct {
  int32_t code;
  char const *field;
  char const *message;
} tide_optim_options_validation;

typedef struct {
  int32_t valid;
  tide_optim_options_validation validation;
  int32_t direction_policy;
  int32_t line_search_policy;
  int32_t alpha_guess_policy;
  int32_t stopping_policy;
  int32_t nlcg_beta_policy;
  int32_t lbfgs_update_policy;
  int32_t globalization_policy;
  int32_t bounds_strategy;
  int32_t cost_model;
  char const *method_name;
  char const *direction_policy_name;
  char const *line_search_policy_name;
  char const *alpha_guess_policy_name;
  char const *stopping_policy_name;
  char const *nlcg_beta_policy_name;
  char const *lbfgs_update_policy_name;
  char const *globalization_policy_name;
  char const *bounds_strategy_name;
  char const *cost_model_name;
} tide_optim_resolved_policies;

typedef struct {
  int32_t request;
  int32_t status;
  int32_t line_search_status;
  int32_t line_search_acceptance;
  int32_t pair_status;
  int32_t warning_flags;
  int32_t line_search_policy;
  int32_t direction_policy;
  int32_t nlcg_beta_policy;
  int32_t lbfgs_update_policy;
  int32_t preconditioner_status;
  int32_t inner_status;
  int32_t globalization_policy;
  int32_t trust_region_status;
  int32_t alpha_guess_policy;
  int32_t stopping_policy;
  int64_t request_sequence;
  int64_t n;
  int64_t iter;
  int64_t n_f;
  int64_t n_g;
  int64_t n_hvp;
  int64_t n_prec;
  int64_t line_search_iter;
  int64_t inner_iter;
  int64_t history_size;
  double f;
  double grad_norm;
  double alpha;
  double step_norm;
  double step_tolerance;
  double directional_derivative_initial;
  double directional_derivative_trial;
  double sy;
  double yy;
  double gamma;
  double direction_beta;
  int32_t direction_status;
  double preconditioner_dot;
  double inner_residual_norm;
  double inner_forcing_tolerance;
  double inner_curvature;
  double trust_radius;
  double trust_ratio;
  double predicted_reduction;
  double actual_reduction;
  double trial_alpha;
  double trial_f;
  double line_search_reference;
  double line_search_armijo_rhs;
  double projected_grad_norm;
  int64_t active_lower_count;
  int64_t active_upper_count;
  int64_t free_count;
  int64_t kkt_violation_count;
  int64_t lower_kkt_violation_count;
  int64_t upper_kkt_violation_count;
  int64_t free_gradient_count;
  int64_t trial_projection_count;
  int64_t trial_lower_projection_count;
  int64_t trial_upper_projection_count;
  int64_t line_search_accept_count;
  int64_t line_search_rejection_count;
  int64_t line_search_failure_count;
  int64_t line_search_fallback_accept_count;
  int64_t nonfinite_trial_count;
  int64_t pair_skip_count;
  int64_t pair_stored_count;
  int64_t pair_regularized_count;
  int64_t preconditioner_skip_count;
  int64_t inner_warning_count;
  int64_t trust_region_accept_count;
  int64_t trust_region_rejection_count;
  int64_t trust_region_failure_count;
  double grad_tolerance;
  double f_change;
  double f_tolerance;
  double initial_f;
  double initial_grad_norm;
  int64_t max_iter;
  int64_t max_eval;
} tide_optim_report;

typedef struct {
  int32_t valid;
  int32_t request;
  int64_t request_sequence;
  char const *request_name;
  char const *expected_evaluation;
  char const *required_fields;
  char const *accepted_mapping_keys;
  int32_t requires_evaluation;
  int32_t error;
  int32_t done;
  int32_t needs_value;
  int32_t needs_gradient;
  int32_t needs_value_gradient;
  int32_t needs_preconditioner;
  int32_t needs_hessian_vector;
  int32_t needs_vector_result;
} tide_optim_request_summary;

typedef struct {
  int32_t request;
  int64_t request_sequence;
  char const *request_name;
  char const *expected_evaluation;
  char const *required_fields;
  char const *accepted_mapping_keys;
  int32_t requires_evaluation;
  int32_t has_value;
  int32_t has_gradient;
  int32_t has_vector;
  int64_t gradient_size;
  int64_t vector_size;
  int64_t expected_gradient_size;
  int64_t expected_vector_size;
  int32_t missing_value;
  int32_t missing_gradient;
  int32_t missing_vector;
  int32_t has_missing_fields;
  char const *missing_fields;
  int32_t gradient_size_mismatch;
  int32_t vector_size_mismatch;
  int32_t has_size_mismatch;
  char const *mismatched_fields;
  int32_t satisfied;
  int32_t valid;
} tide_optim_evaluation_status;

typedef struct {
  int32_t valid;
  int32_t request;
  int32_t status;
  char const *request_name;
  char const *expected_evaluation;
  char const *required_fields;
  char const *accepted_mapping_keys;
  int32_t requires_evaluation;
  char const *status_name;
  char const *reason;
  char const *failure_reason;
  char const *method_name;
  char const *line_search_status_name;
  char const *line_search_acceptance_name;
  char const *pair_status_name;
  char const *direction_policy_name;
  char const *line_search_policy_name;
  char const *globalization_policy_name;
  char const *preconditioner_status_name;
  char const *inner_status_name;
  char const *trust_region_status_name;
  int32_t done;
  int32_t success;
  int32_t stopped;
  int32_t failed;
  int32_t user_stopped;
  int32_t needs_value;
  int32_t needs_gradient;
  int32_t needs_value_gradient;
  int32_t needs_preconditioner;
  int32_t needs_hessian_vector;
  int32_t needs_vector_result;
  int32_t line_search_failed;
  int32_t inner_cg_failed;
  int32_t trust_region_failed;
  int32_t nonfinite;
  int32_t has_warnings;
  int32_t warning_flags;
  int64_t request_sequence;
  int64_t n;
  int64_t expected_gradient_size;
  int64_t expected_vector_size;
  int64_t iter;
  int64_t n_f;
  int64_t n_g;
  int64_t n_hvp;
  int64_t n_prec;
  double f;
  double grad_norm;
  double projected_grad_norm;
  double alpha;
} tide_optim_report_summary;

typedef struct {
  int32_t valid;
  int32_t started;
  int32_t done;
  int32_t running;
  int32_t has_bounds;
  char const *state_name;
  int64_t n;
  tide_optim_report_summary report;
  int32_t awaiting_value_gradient;
  int32_t awaiting_preconditioner;
  int32_t awaiting_hessian_vector;
  int32_t awaiting_trust_region_trial;
  int32_t awaiting_accepted_gradient;
} tide_optim_session_snapshot;

tide_optim_options_validation
tide_optim_lbfgs_validate_options(tide_optim_lbfgs_options const *options);

void *tide_optim_lbfgs_create(tide_optim_lbfgs_options const *options);
void tide_optim_lbfgs_destroy(void *handle);

int32_t tide_optim_lbfgs_set_bounds(void *handle, double const *lb,
                                    double const *ub);

int32_t tide_optim_lbfgs_start(void *handle, double const *x, double f,
                               double const *g, double *x_request,
                               tide_optim_report *report);

int32_t tide_optim_lbfgs_tell(void *handle, double f, double const *g,
                              double *x_request,
                              tide_optim_report *report);

int32_t tide_optim_lbfgs_tell_value(void *handle, double f, double *x_request,
                                    tide_optim_report *report);

int32_t tide_optim_lbfgs_tell_preconditioner(void *handle, double const *z,
                                             double *x_request,
                                             tide_optim_report *report);

int32_t tide_optim_lbfgs_tell_hessian_vector(void *handle, double const *hv,
                                             double *x_request,
                                             tide_optim_report *report);

int32_t tide_optim_lbfgs_current_x(void *handle, double *x_out);

tide_optim_options_validation
tide_optim_validate_options(tide_optim_options const *options);

tide_optim_resolved_policies
tide_optim_resolve_policies(tide_optim_options const *options);

void *tide_optim_create(tide_optim_options const *options);
void tide_optim_destroy(void *handle);

int32_t tide_optim_set_bounds(void *handle, double const *lb,
                              double const *ub);

int32_t tide_optim_start(void *handle, double const *x, double f,
                         double const *g, double *x_request,
                         tide_optim_report *report);

int32_t tide_optim_tell(void *handle, double f, double const *g,
                        double *x_request, tide_optim_report *report);

int32_t tide_optim_tell_value(void *handle, double f, double *x_request,
                              tide_optim_report *report);

int32_t tide_optim_tell_preconditioner(void *handle, double const *z,
                                       double *x_request,
                                       tide_optim_report *report);

int32_t tide_optim_tell_hessian_vector(void *handle, double const *hv,
                                       double *x_request,
                                       tide_optim_report *report);

int32_t tide_optim_current_x(void *handle, double *x_out);

tide_optim_report_summary
tide_optim_summarize_report(tide_optim_report const *report);

tide_optim_request_summary tide_optim_summarize_request(int32_t request);

tide_optim_request_summary
tide_optim_summarize_report_request(tide_optim_report const *report);

tide_optim_evaluation_status tide_optim_validate_evaluation(
    int32_t request, int64_t request_sequence, int64_t expected_gradient_size,
    int64_t expected_vector_size, int32_t has_value, int32_t has_gradient,
    int64_t gradient_size, int32_t has_vector, int64_t vector_size);

tide_optim_evaluation_status tide_optim_validate_report_evaluation(
    tide_optim_report const *report, int32_t has_value, int32_t has_gradient,
    int64_t gradient_size, int32_t has_vector, int64_t vector_size);

tide_optim_session_snapshot tide_optim_get_session_snapshot(void *handle);

char const *tide_optim_request_kind_name(int32_t request);
char const *tide_optim_request_expected_evaluation(int32_t request);
char const *tide_optim_request_required_fields(int32_t request);
char const *tide_optim_request_accepted_mapping_keys(int32_t request);
int32_t tide_optim_request_requires_evaluation(int32_t request);
int32_t tide_optim_request_is_error(int32_t request);
int32_t tide_optim_request_is_done(int32_t request);
int32_t tide_optim_request_needs_value(int32_t request);
int32_t tide_optim_request_needs_gradient(int32_t request);
int32_t tide_optim_request_needs_value_gradient(int32_t request);
int32_t tide_optim_request_needs_preconditioner(int32_t request);
int32_t tide_optim_request_needs_hessian_vector(int32_t request);
int32_t tide_optim_request_needs_vector_result(int32_t request);

char const *tide_optim_status_name(int32_t status);
char const *tide_optim_line_search_status_name(int32_t status);
char const *tide_optim_line_search_policy_name(int32_t policy);
char const *tide_optim_alpha_guess_policy_name(int32_t policy);
char const *tide_optim_stopping_policy_name(int32_t policy);
char const *tide_optim_line_search_acceptance_name(int32_t acceptance);
char const *tide_optim_pair_status_name(int32_t status);
char const *tide_optim_lbfgs_update_policy_name(int32_t policy);
char const *tide_optim_bounds_strategy_name(int32_t strategy);
char const *tide_optim_cost_model_name(int32_t cost_model);
char const *tide_optim_direction_policy_name(int32_t policy);
char const *tide_optim_direction_method_name(int32_t policy);
char const *tide_optim_nlcg_beta_policy_name(int32_t policy);
char const *tide_optim_direction_status_name(int32_t status);
char const *tide_optim_preconditioner_status_name(int32_t status);
char const *tide_optim_inner_cg_status_name(int32_t status);
char const *tide_optim_globalization_policy_name(int32_t policy);
char const *tide_optim_trust_region_status_name(int32_t status);
char const *tide_optim_warning_flag_name(int32_t flag);
char const *tide_optim_options_validation_code_name(int32_t code);

#ifdef __cplusplus
}

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tide::optim {

struct ResolvedPolicies;

enum class RequestKind : int32_t {
  Error = TIDE_OPTIM_REQUEST_ERROR,
  EvaluateFG = TIDE_OPTIM_REQUEST_EVALUATE_FG,
  Done = TIDE_OPTIM_REQUEST_DONE,
  EvaluateF = TIDE_OPTIM_REQUEST_EVALUATE_F,
  ApplyPreconditioner = TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER,
  EvaluateHv = TIDE_OPTIM_REQUEST_EVALUATE_HV,
};

enum class Status : int32_t {
  Running = TIDE_OPTIM_STATUS_RUNNING,
  ConvergedGradient = TIDE_OPTIM_STATUS_CONVERGED_GRADIENT,
  ConvergedFtol = TIDE_OPTIM_STATUS_CONVERGED_FTOL,
  MaxIter = TIDE_OPTIM_STATUS_MAX_ITER,
  LineSearchFailed = TIDE_OPTIM_STATUS_LINE_SEARCH_FAILED,
  Nonfinite = TIDE_OPTIM_STATUS_NONFINITE,
  NonDescentDirection = TIDE_OPTIM_STATUS_NON_DESCENT_DIRECTION,
  InvalidArgument = TIDE_OPTIM_STATUS_INVALID_ARGUMENT,
  UserStopped = TIDE_OPTIM_STATUS_USER_STOPPED,
  ConvergedXtol = TIDE_OPTIM_STATUS_CONVERGED_XTOL,
  MaxEval = TIDE_OPTIM_STATUS_MAX_EVAL,
  InnerCgFailed = TIDE_OPTIM_STATUS_INNER_CG_FAILED,
  TrustRegionFailed = TIDE_OPTIM_STATUS_TRUST_REGION_FAILED,
};

enum class LineSearchStatus : int32_t {
  Started = TIDE_OPTIM_LINE_SEARCH_STARTED,
  Accepted = TIDE_OPTIM_LINE_SEARCH_ACCEPTED,
  RejectedArmijo = TIDE_OPTIM_LINE_SEARCH_REJECTED_ARMIJO,
  RejectedCurvature = TIDE_OPTIM_LINE_SEARCH_REJECTED_CURVATURE,
  AcceptedDecreaseAfterMaxLineSearch =
      TIDE_OPTIM_LINE_SEARCH_ACCEPTED_DECREASE_AFTER_MAXLS,
  FailedMaxLineSearch = TIDE_OPTIM_LINE_SEARCH_FAILED_MAXLS,
  FailedNonfinite = TIDE_OPTIM_LINE_SEARCH_FAILED_NONFINITE,
  FailedAlphaBounds = TIDE_OPTIM_LINE_SEARCH_FAILED_ALPHA_BOUNDS,
};

enum class LineSearchPolicy : int32_t {
  LegacyWeakWolfe = TIDE_OPTIM_LINE_SEARCH_POLICY_LEGACY_WEAK_WOLFE,
  ArmijoCubic = TIDE_OPTIM_LINE_SEARCH_POLICY_ARMIJO_CUBIC,
  StrongWolfe = TIDE_OPTIM_LINE_SEARCH_POLICY_STRONG_WOLFE,
  MoreThuente = TIDE_OPTIM_LINE_SEARCH_POLICY_MORE_THUENTE,
  NonmonotoneArmijo = TIDE_OPTIM_LINE_SEARCH_POLICY_NONMONOTONE_ARMIJO,
  HagerZhang = TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG,
  Static = TIDE_OPTIM_LINE_SEARCH_POLICY_STATIC,
};

enum class AlphaGuessPolicy : int32_t {
  Initial = TIDE_OPTIM_ALPHA_GUESS_INITIAL,
  Previous = TIDE_OPTIM_ALPHA_GUESS_PREVIOUS,
  BarzilaiBorwein = TIDE_OPTIM_ALPHA_GUESS_BARZILAI_BORWEIN,
};

enum class StoppingPolicy : int32_t {
  Standard = TIDE_OPTIM_STOPPING_STANDARD,
  GradientOnly = TIDE_OPTIM_STOPPING_GRADIENT_ONLY,
  InitialRelativeF = TIDE_OPTIM_STOPPING_INITIAL_RELATIVE_F,
};

enum class CostModel : int32_t {
  Balanced = 0,
  ExpensiveGradient = 1,
  JointValueGradient = 2,
};

enum class LineSearchAcceptance : int32_t {
  None = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE,
  Armijo = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_ARMIJO,
  WeakWolfe = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_WEAK_WOLFE,
  StrongWolfe = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_STRONG_WOLFE,
  ApproximateWolfe = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_APPROXIMATE_WOLFE,
  DecreaseFallback = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_DECREASE_FALLBACK,
  Static = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_STATIC,
};

enum class PairStatus : int32_t {
  None = TIDE_OPTIM_PAIR_NONE,
  Stored = TIDE_OPTIM_PAIR_STORED,
  SkippedBadCurvature = TIDE_OPTIM_PAIR_SKIPPED_BAD_CURVATURE,
  SkippedNonfinite = TIDE_OPTIM_PAIR_SKIPPED_NONFINITE,
  SkippedLineSearchFallback = TIDE_OPTIM_PAIR_SKIPPED_LINE_SEARCH_FALLBACK,
  SkippedBoundsProjection = TIDE_OPTIM_PAIR_SKIPPED_BOUNDS_PROJECTION,
  SkippedPreconditioner = TIDE_OPTIM_PAIR_SKIPPED_PRECONDITIONER,
  RegularizedStored = TIDE_OPTIM_PAIR_REGULARIZED_STORED,
};

enum class LbfgsUpdatePolicy : int32_t {
  Skip = TIDE_OPTIM_LBFGS_UPDATE_SKIP,
  Regularize = TIDE_OPTIM_LBFGS_UPDATE_REGULARIZE,
};

enum class BoundsStrategy : int32_t {
  None = TIDE_OPTIM_BOUNDS_NONE,
  ProjectedTrial = TIDE_OPTIM_BOUNDS_PROJECTED_TRIAL,
  ProjectedGradient = TIDE_OPTIM_BOUNDS_PROJECTED_GRADIENT,
};

enum class DirectionPolicy : int32_t {
  SteepestDescent = TIDE_OPTIM_DIRECTION_STEEPEST_DESCENT,
  Lbfgs = TIDE_OPTIM_DIRECTION_LBFGS,
  Nlcg = TIDE_OPTIM_DIRECTION_NLCG,
  PreconditionedLbfgs = TIDE_OPTIM_DIRECTION_PRECONDITIONED_LBFGS,
  TruncatedNewton = TIDE_OPTIM_DIRECTION_TRUNCATED_NEWTON,
  PreconditionedTruncatedNewton =
      TIDE_OPTIM_DIRECTION_PRECONDITIONED_TRUNCATED_NEWTON,
};

enum class NlcgBetaPolicy : int32_t {
  DaiYuan = TIDE_OPTIM_NLCG_BETA_DAI_YUAN,
  FletcherReeves = TIDE_OPTIM_NLCG_BETA_FLETCHER_REEVES,
  PolakRibierePlus = TIDE_OPTIM_NLCG_BETA_POLAK_RIBIERE_PLUS,
  HagerZhang = TIDE_OPTIM_NLCG_BETA_HAGER_ZHANG,
};

enum class DirectionStatus : int32_t {
  Initial = TIDE_OPTIM_DIRECTION_STATUS_INITIAL,
  Update = TIDE_OPTIM_DIRECTION_STATUS_UPDATE,
  RestartDenominator = TIDE_OPTIM_DIRECTION_STATUS_RESTART_DENOMINATOR,
  RestartNonfinite = TIDE_OPTIM_DIRECTION_STATUS_RESTART_NONFINITE,
  RestartNonDescent = TIDE_OPTIM_DIRECTION_STATUS_RESTART_NON_DESCENT,
};

enum class PreconditionerStatus : int32_t {
  None = TIDE_OPTIM_PRECONDITIONER_NONE,
  Applied = TIDE_OPTIM_PRECONDITIONER_APPLIED,
  SkippedNonfinite = TIDE_OPTIM_PRECONDITIONER_SKIPPED_NONFINITE,
  SkippedNotPositive = TIDE_OPTIM_PRECONDITIONER_SKIPPED_NOT_POSITIVE,
};

enum class InnerCgStatus : int32_t {
  None = TIDE_OPTIM_INNER_CG_NONE,
  Started = TIDE_OPTIM_INNER_CG_STARTED,
  ForcingReached = TIDE_OPTIM_INNER_CG_FORCING_REACHED,
  NegativeCurvature = TIDE_OPTIM_INNER_CG_NEGATIVE_CURVATURE,
  ZeroCurvature = TIDE_OPTIM_INNER_CG_ZERO_CURVATURE,
  MaxIter = TIDE_OPTIM_INNER_CG_MAX_ITER,
  NonfiniteHvp = TIDE_OPTIM_INNER_CG_NONFINITE_HVP,
  PreconditionerBreakdown = TIDE_OPTIM_INNER_CG_PRECONDITIONER_BREAKDOWN,
  TrustBoundary = TIDE_OPTIM_INNER_CG_TRUST_BOUNDARY,
};

enum class GlobalizationPolicy : int32_t {
  LineSearch = TIDE_OPTIM_GLOBALIZATION_LINE_SEARCH,
  TrustRegion = TIDE_OPTIM_GLOBALIZATION_TRUST_REGION,
};

enum class TrustRegionStatus : int32_t {
  None = TIDE_OPTIM_TRUST_REGION_NONE,
  Started = TIDE_OPTIM_TRUST_REGION_STARTED,
  Accepted = TIDE_OPTIM_TRUST_REGION_ACCEPTED,
  Rejected = TIDE_OPTIM_TRUST_REGION_REJECTED,
  FailedNonfinite = TIDE_OPTIM_TRUST_REGION_FAILED_NONFINITE,
  FailedPredictedReduction =
      TIDE_OPTIM_TRUST_REGION_FAILED_PREDICTED_REDUCTION,
};

enum class WarningFlag : int32_t {
  NonfiniteTrial = TIDE_OPTIM_WARNING_NONFINITE_TRIAL,
  NonDescentDirectionReset =
      TIDE_OPTIM_WARNING_NON_DESCENT_DIRECTION_RESET,
  AcceptedDecreaseAfterMaxLineSearch =
      TIDE_OPTIM_WARNING_ACCEPTED_DECREASE_AFTER_MAX_LINE_SEARCH,
  LbfgsPairSkipped = TIDE_OPTIM_WARNING_LBFGS_PAIR_SKIPPED,
  PreconditionerSkipped = TIDE_OPTIM_WARNING_PRECONDITIONER_SKIPPED,
  InnerCg = TIDE_OPTIM_WARNING_INNER_CG,
  LbfgsPairRegularized = TIDE_OPTIM_WARNING_LBFGS_PAIR_REGULARIZED,
};

enum class OptionsValidationCode : int32_t {
  Ok = TIDE_OPTIM_OPTIONS_VALIDATION_OK,
  Dimension = TIDE_OPTIM_OPTIONS_VALIDATION_DIMENSION,
  HistorySize = TIDE_OPTIM_OPTIONS_VALIDATION_HISTORY_SIZE,
  MaxIter = TIDE_OPTIM_OPTIONS_VALIDATION_MAX_ITER,
  MaxLineSearch = TIDE_OPTIM_OPTIONS_VALIDATION_MAX_LINE_SEARCH,
  MaxEval = TIDE_OPTIM_OPTIONS_VALIDATION_MAX_EVAL,
  MaxInnerIter = TIDE_OPTIM_OPTIONS_VALIDATION_MAX_INNER_ITER,
  NonmonotoneWindow = TIDE_OPTIM_OPTIONS_VALIDATION_NONMONOTONE_WINDOW,
  InitialStep = TIDE_OPTIM_OPTIONS_VALIDATION_INITIAL_STEP,
  WolfeParameters = TIDE_OPTIM_OPTIONS_VALIDATION_WOLFE_PARAMETERS,
  Growth = TIDE_OPTIM_OPTIONS_VALIDATION_GROWTH,
  LineSearchPolicy = TIDE_OPTIM_OPTIONS_VALIDATION_LINE_SEARCH_POLICY,
  DirectionPolicy = TIDE_OPTIM_OPTIONS_VALIDATION_DIRECTION_POLICY,
  NlcgBetaPolicy = TIDE_OPTIM_OPTIONS_VALIDATION_NLCG_BETA_POLICY,
  LbfgsUpdatePolicy = TIDE_OPTIM_OPTIONS_VALIDATION_LBFGS_UPDATE_POLICY,
  GlobalizationPolicy = TIDE_OPTIM_OPTIONS_VALIDATION_GLOBALIZATION_POLICY,
  AlphaGuessPolicy = TIDE_OPTIM_OPTIONS_VALIDATION_ALPHA_GUESS_POLICY,
  StoppingPolicy = TIDE_OPTIM_OPTIONS_VALIDATION_STOPPING_POLICY,
  CostModel = TIDE_OPTIM_OPTIONS_VALIDATION_COST_MODEL,
  AlphaBounds = TIDE_OPTIM_OPTIONS_VALIDATION_ALPHA_BOUNDS,
  InnerTolerance = TIDE_OPTIM_OPTIONS_VALIDATION_INNER_TOLERANCE,
  TrustRadius = TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_RADIUS,
  TrustEta = TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_ETA,
  TrustShrink = TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_SHRINK,
  TrustGrow = TIDE_OPTIM_OPTIONS_VALIDATION_TRUST_GROW,
  CurvatureEps = TIDE_OPTIM_OPTIONS_VALIDATION_CURVATURE_EPS,
  GammaBounds = TIDE_OPTIM_OPTIONS_VALIDATION_GAMMA_BOUNDS,
  Tolerances = TIDE_OPTIM_OPTIONS_VALIDATION_TOLERANCES,
  ArmijoShrink = TIDE_OPTIM_OPTIONS_VALIDATION_ARMIJO_SHRINK,
  BoundMargin = TIDE_OPTIM_OPTIONS_VALIDATION_BOUND_MARGIN,
  BoundsStrategy = TIDE_OPTIM_OPTIONS_VALIDATION_BOUNDS_STRATEGY,
  NullOptions = TIDE_OPTIM_OPTIONS_VALIDATION_NULL_OPTIONS,
  TracePolicy = TIDE_OPTIM_OPTIONS_VALIDATION_TRACE_POLICY,
  TraceStride = TIDE_OPTIM_OPTIONS_VALIDATION_TRACE_STRIDE,
};

inline char const *name(RequestKind value) {
  switch (value) {
  case RequestKind::Error:
    return "ERROR";
  case RequestKind::EvaluateFG:
    return "EVALUATE_FG";
  case RequestKind::Done:
    return "DONE";
  case RequestKind::EvaluateF:
    return "EVALUATE_F";
  case RequestKind::ApplyPreconditioner:
    return "APPLY_PRECONDITIONER";
  case RequestKind::EvaluateHv:
    return "EVALUATE_HV";
  }
  return "UNKNOWN";
}

inline bool requires_evaluation(RequestKind value) {
  switch (value) {
  case RequestKind::EvaluateF:
  case RequestKind::EvaluateFG:
  case RequestKind::ApplyPreconditioner:
  case RequestKind::EvaluateHv:
    return true;
  case RequestKind::Error:
  case RequestKind::Done:
    return false;
  }
  return false;
}

inline char const *expected_evaluation(RequestKind value) {
  switch (value) {
  case RequestKind::EvaluateF:
    return "value";
  case RequestKind::EvaluateFG:
    return "value_gradient";
  case RequestKind::ApplyPreconditioner:
    return "preconditioner";
  case RequestKind::EvaluateHv:
    return "hessian_vector";
  case RequestKind::Error:
  case RequestKind::Done:
    return "none";
  }
  return "none";
}

inline char const *required_fields(RequestKind value) {
  switch (value) {
  case RequestKind::EvaluateF:
    return "f";
  case RequestKind::EvaluateFG:
    return "f,g";
  case RequestKind::ApplyPreconditioner:
  case RequestKind::EvaluateHv:
    return "vector";
  case RequestKind::Error:
  case RequestKind::Done:
    return "";
  }
  return "";
}

inline char const *accepted_mapping_keys(RequestKind value) {
  switch (value) {
  case RequestKind::EvaluateF:
    return "f";
  case RequestKind::EvaluateFG:
    return "f,g,gradient";
  case RequestKind::ApplyPreconditioner:
    return "vector,z";
  case RequestKind::EvaluateHv:
    return "vector,hv";
  case RequestKind::Error:
  case RequestKind::Done:
    return "";
  }
  return "";
}

inline char const *name(Status value) {
  switch (value) {
  case Status::Running:
    return "RUNNING";
  case Status::ConvergedGradient:
    return "CONVERGED_GRADIENT";
  case Status::ConvergedFtol:
    return "CONVERGED_FTOL";
  case Status::MaxIter:
    return "MAX_ITER";
  case Status::LineSearchFailed:
    return "LINE_SEARCH_FAILED";
  case Status::Nonfinite:
    return "NONFINITE";
  case Status::NonDescentDirection:
    return "NON_DESCENT_DIRECTION";
  case Status::InvalidArgument:
    return "INVALID_ARGUMENT";
  case Status::UserStopped:
    return "USER_STOPPED";
  case Status::ConvergedXtol:
    return "CONVERGED_XTOL";
  case Status::MaxEval:
    return "MAX_EVAL";
  case Status::InnerCgFailed:
    return "INNER_CG_FAILED";
  case Status::TrustRegionFailed:
    return "TRUST_REGION_FAILED";
  }
  return "UNKNOWN";
}

inline char const *name(LineSearchStatus value) {
  switch (value) {
  case LineSearchStatus::Started:
    return "STARTED";
  case LineSearchStatus::Accepted:
    return "ACCEPTED";
  case LineSearchStatus::RejectedArmijo:
    return "REJECTED_ARMIJO";
  case LineSearchStatus::RejectedCurvature:
    return "REJECTED_CURVATURE";
  case LineSearchStatus::AcceptedDecreaseAfterMaxLineSearch:
    return "ACCEPTED_DECREASE_AFTER_MAXLS";
  case LineSearchStatus::FailedMaxLineSearch:
    return "FAILED_MAXLS";
  case LineSearchStatus::FailedNonfinite:
    return "FAILED_NONFINITE";
  case LineSearchStatus::FailedAlphaBounds:
    return "FAILED_ALPHA_BOUNDS";
  }
  return "UNKNOWN";
}

inline char const *name(LineSearchPolicy value) {
  switch (value) {
  case LineSearchPolicy::LegacyWeakWolfe:
    return "LEGACY_WEAK_WOLFE";
  case LineSearchPolicy::ArmijoCubic:
    return "ARMIJO_CUBIC";
  case LineSearchPolicy::StrongWolfe:
    return "STRONG_WOLFE";
  case LineSearchPolicy::MoreThuente:
    return "MORE_THUENTE";
  case LineSearchPolicy::NonmonotoneArmijo:
    return "NONMONOTONE_ARMIJO";
  case LineSearchPolicy::HagerZhang:
    return "HAGER_ZHANG";
  case LineSearchPolicy::Static:
    return "STATIC";
  }
  return "UNKNOWN";
}

inline char const *name(AlphaGuessPolicy value) {
  switch (value) {
  case AlphaGuessPolicy::Initial:
    return "INITIAL";
  case AlphaGuessPolicy::Previous:
    return "PREVIOUS";
  case AlphaGuessPolicy::BarzilaiBorwein:
    return "BARZILAI_BORWEIN";
  }
  return "UNKNOWN";
}

inline char const *name(StoppingPolicy value) {
  switch (value) {
  case StoppingPolicy::Standard:
    return "STANDARD";
  case StoppingPolicy::GradientOnly:
    return "GRADIENT_ONLY";
  case StoppingPolicy::InitialRelativeF:
    return "INITIAL_RELATIVE_F";
  }
  return "UNKNOWN";
}

inline char const *name(CostModel value) {
  switch (value) {
  case CostModel::Balanced:
    return "BALANCED";
  case CostModel::ExpensiveGradient:
    return "EXPENSIVE_GRADIENT";
  case CostModel::JointValueGradient:
    return "JOINT_VALUE_GRADIENT";
  }
  return "UNKNOWN";
}

inline char const *name(LineSearchAcceptance value) {
  switch (value) {
  case LineSearchAcceptance::None:
    return "NONE";
  case LineSearchAcceptance::Armijo:
    return "ARMIJO";
  case LineSearchAcceptance::WeakWolfe:
    return "WEAK_WOLFE";
  case LineSearchAcceptance::StrongWolfe:
    return "STRONG_WOLFE";
  case LineSearchAcceptance::ApproximateWolfe:
    return "APPROXIMATE_WOLFE";
  case LineSearchAcceptance::DecreaseFallback:
    return "DECREASE_FALLBACK";
  case LineSearchAcceptance::Static:
    return "STATIC";
  }
  return "UNKNOWN";
}

inline char const *name(PairStatus value) {
  switch (value) {
  case PairStatus::None:
    return "NONE";
  case PairStatus::Stored:
    return "STORED";
  case PairStatus::SkippedBadCurvature:
    return "SKIPPED_BAD_CURVATURE";
  case PairStatus::SkippedNonfinite:
    return "SKIPPED_NONFINITE";
  case PairStatus::SkippedLineSearchFallback:
    return "SKIPPED_LINE_SEARCH_FALLBACK";
  case PairStatus::SkippedBoundsProjection:
    return "SKIPPED_BOUNDS_PROJECTION";
  case PairStatus::SkippedPreconditioner:
    return "SKIPPED_PRECONDITIONER";
  case PairStatus::RegularizedStored:
    return "REGULARIZED_STORED";
  }
  return "UNKNOWN";
}

inline char const *name(LbfgsUpdatePolicy value) {
  switch (value) {
  case LbfgsUpdatePolicy::Skip:
    return "SKIP";
  case LbfgsUpdatePolicy::Regularize:
    return "REGULARIZE";
  }
  return "UNKNOWN";
}

inline char const *name(BoundsStrategy value) {
  switch (value) {
  case BoundsStrategy::None:
    return "NONE";
  case BoundsStrategy::ProjectedTrial:
    return "PROJECTED_TRIAL";
  case BoundsStrategy::ProjectedGradient:
    return "PROJECTED_GRADIENT";
  }
  return "UNKNOWN";
}

inline char const *name(DirectionPolicy value) {
  switch (value) {
  case DirectionPolicy::SteepestDescent:
    return "STEEPEST_DESCENT";
  case DirectionPolicy::Lbfgs:
    return "LBFGS";
  case DirectionPolicy::Nlcg:
    return "NLCG";
  case DirectionPolicy::PreconditionedLbfgs:
    return "PRECONDITIONED_LBFGS";
  case DirectionPolicy::TruncatedNewton:
    return "TRUNCATED_NEWTON";
  case DirectionPolicy::PreconditionedTruncatedNewton:
    return "PRECONDITIONED_TRUNCATED_NEWTON";
  }
  return "UNKNOWN";
}

inline char const *method_name(DirectionPolicy value) {
  switch (value) {
  case DirectionPolicy::SteepestDescent:
    return "pstd";
  case DirectionPolicy::Lbfgs:
    return "lbfgs";
  case DirectionPolicy::Nlcg:
    return "pnlcg";
  case DirectionPolicy::PreconditionedLbfgs:
    return "plbfgs";
  case DirectionPolicy::TruncatedNewton:
    return "trn";
  case DirectionPolicy::PreconditionedTruncatedNewton:
    return "ptrn";
  }
  return "unknown";
}

inline char const *name(NlcgBetaPolicy value) {
  switch (value) {
  case NlcgBetaPolicy::DaiYuan:
    return "DAI_YUAN";
  case NlcgBetaPolicy::FletcherReeves:
    return "FLETCHER_REEVES";
  case NlcgBetaPolicy::PolakRibierePlus:
    return "POLAK_RIBIERE_PLUS";
  case NlcgBetaPolicy::HagerZhang:
    return "HAGER_ZHANG";
  }
  return "UNKNOWN";
}

inline char const *name(DirectionStatus value) {
  switch (value) {
  case DirectionStatus::Initial:
    return "INITIAL";
  case DirectionStatus::Update:
    return "UPDATE";
  case DirectionStatus::RestartDenominator:
    return "RESTART_DENOMINATOR";
  case DirectionStatus::RestartNonfinite:
    return "RESTART_NONFINITE";
  case DirectionStatus::RestartNonDescent:
    return "RESTART_NON_DESCENT";
  }
  return "UNKNOWN";
}

inline char const *name(PreconditionerStatus value) {
  switch (value) {
  case PreconditionerStatus::None:
    return "NONE";
  case PreconditionerStatus::Applied:
    return "APPLIED";
  case PreconditionerStatus::SkippedNonfinite:
    return "SKIPPED_NONFINITE";
  case PreconditionerStatus::SkippedNotPositive:
    return "SKIPPED_NOT_POSITIVE";
  }
  return "UNKNOWN";
}

inline char const *name(InnerCgStatus value) {
  switch (value) {
  case InnerCgStatus::None:
    return "NONE";
  case InnerCgStatus::Started:
    return "STARTED";
  case InnerCgStatus::ForcingReached:
    return "FORCING_REACHED";
  case InnerCgStatus::NegativeCurvature:
    return "NEGATIVE_CURVATURE";
  case InnerCgStatus::ZeroCurvature:
    return "ZERO_CURVATURE";
  case InnerCgStatus::MaxIter:
    return "MAX_ITER";
  case InnerCgStatus::NonfiniteHvp:
    return "NONFINITE_HVP";
  case InnerCgStatus::PreconditionerBreakdown:
    return "PRECONDITIONER_BREAKDOWN";
  case InnerCgStatus::TrustBoundary:
    return "TRUST_BOUNDARY";
  }
  return "UNKNOWN";
}

inline char const *name(GlobalizationPolicy value) {
  switch (value) {
  case GlobalizationPolicy::LineSearch:
    return "LINE_SEARCH";
  case GlobalizationPolicy::TrustRegion:
    return "TRUST_REGION";
  }
  return "UNKNOWN";
}

inline char const *name(TrustRegionStatus value) {
  switch (value) {
  case TrustRegionStatus::None:
    return "NONE";
  case TrustRegionStatus::Started:
    return "STARTED";
  case TrustRegionStatus::Accepted:
    return "ACCEPTED";
  case TrustRegionStatus::Rejected:
    return "REJECTED";
  case TrustRegionStatus::FailedNonfinite:
    return "FAILED_NONFINITE";
  case TrustRegionStatus::FailedPredictedReduction:
    return "FAILED_PREDICTED_REDUCTION";
  }
  return "UNKNOWN";
}

inline char const *name(WarningFlag value) {
  switch (value) {
  case WarningFlag::NonfiniteTrial:
    return "NONFINITE_TRIAL";
  case WarningFlag::NonDescentDirectionReset:
    return "NON_DESCENT_DIRECTION_RESET";
  case WarningFlag::AcceptedDecreaseAfterMaxLineSearch:
    return "ACCEPTED_DECREASE_AFTER_MAX_LINE_SEARCH";
  case WarningFlag::LbfgsPairSkipped:
    return "LBFGS_PAIR_SKIPPED";
  case WarningFlag::PreconditionerSkipped:
    return "PRECONDITIONER_SKIPPED";
  case WarningFlag::InnerCg:
    return "INNER_CG";
  case WarningFlag::LbfgsPairRegularized:
    return "LBFGS_PAIR_REGULARIZED";
  }
  return "UNKNOWN";
}

inline char const *name(OptionsValidationCode value) {
  switch (value) {
  case OptionsValidationCode::Ok:
    return "OK";
  case OptionsValidationCode::Dimension:
    return "DIMENSION";
  case OptionsValidationCode::HistorySize:
    return "HISTORY_SIZE";
  case OptionsValidationCode::MaxIter:
    return "MAX_ITER";
  case OptionsValidationCode::MaxLineSearch:
    return "MAX_LINE_SEARCH";
  case OptionsValidationCode::MaxEval:
    return "MAX_EVAL";
  case OptionsValidationCode::MaxInnerIter:
    return "MAX_INNER_ITER";
  case OptionsValidationCode::NonmonotoneWindow:
    return "NONMONOTONE_WINDOW";
  case OptionsValidationCode::InitialStep:
    return "INITIAL_STEP";
  case OptionsValidationCode::WolfeParameters:
    return "WOLFE_PARAMETERS";
  case OptionsValidationCode::Growth:
    return "GROWTH";
  case OptionsValidationCode::LineSearchPolicy:
    return "LINE_SEARCH_POLICY";
  case OptionsValidationCode::DirectionPolicy:
    return "DIRECTION_POLICY";
  case OptionsValidationCode::NlcgBetaPolicy:
    return "NLCG_BETA_POLICY";
  case OptionsValidationCode::LbfgsUpdatePolicy:
    return "LBFGS_UPDATE_POLICY";
  case OptionsValidationCode::GlobalizationPolicy:
    return "GLOBALIZATION_POLICY";
  case OptionsValidationCode::AlphaGuessPolicy:
    return "ALPHA_GUESS_POLICY";
  case OptionsValidationCode::StoppingPolicy:
    return "STOPPING_POLICY";
  case OptionsValidationCode::CostModel:
    return "COST_MODEL";
  case OptionsValidationCode::AlphaBounds:
    return "ALPHA_BOUNDS";
  case OptionsValidationCode::InnerTolerance:
    return "INNER_TOLERANCE";
  case OptionsValidationCode::TrustRadius:
    return "TRUST_RADIUS";
  case OptionsValidationCode::TrustEta:
    return "TRUST_ETA";
  case OptionsValidationCode::TrustShrink:
    return "TRUST_SHRINK";
  case OptionsValidationCode::TrustGrow:
    return "TRUST_GROW";
  case OptionsValidationCode::CurvatureEps:
    return "CURVATURE_EPS";
  case OptionsValidationCode::GammaBounds:
    return "GAMMA_BOUNDS";
  case OptionsValidationCode::Tolerances:
    return "TOLERANCES";
  case OptionsValidationCode::ArmijoShrink:
    return "ARMIJO_SHRINK";
  case OptionsValidationCode::BoundMargin:
    return "BOUND_MARGIN";
  case OptionsValidationCode::BoundsStrategy:
    return "BOUNDS_STRATEGY";
  case OptionsValidationCode::NullOptions:
    return "NULL_OPTIONS";
  case OptionsValidationCode::TracePolicy:
    return "TRACE_POLICY";
  case OptionsValidationCode::TraceStride:
    return "TRACE_STRIDE";
  }
  return "UNKNOWN";
}

inline bool has_warning(int32_t flags, WarningFlag warning) {
  return (flags & static_cast<int32_t>(warning)) != 0;
}

inline std::vector<WarningFlag> warning_flags(int32_t flags) {
  std::vector<WarningFlag> result{};
  for (WarningFlag const warning : {
           WarningFlag::NonfiniteTrial,
           WarningFlag::NonDescentDirectionReset,
           WarningFlag::AcceptedDecreaseAfterMaxLineSearch,
           WarningFlag::LbfgsPairSkipped,
           WarningFlag::PreconditionerSkipped,
           WarningFlag::InnerCg,
           WarningFlag::LbfgsPairRegularized,
       }) {
    if (has_warning(flags, warning)) {
      result.push_back(warning);
    }
  }
  return result;
}

inline std::vector<char const *> warning_names(int32_t flags) {
  std::vector<char const *> result{};
  for (WarningFlag const warning : warning_flags(flags)) {
    result.push_back(name(warning));
  }
  return result;
}

inline char const *reason(Status status) { return name(status); }

inline char const *failure_reason(Status status,
                                  LineSearchStatus line_search_status,
                                  InnerCgStatus inner_status,
                                  TrustRegionStatus trust_region_status);

inline char const *failure_reason(Status status,
                                  LineSearchStatus line_search_status) {
  return failure_reason(status, line_search_status, InnerCgStatus::None,
                        TrustRegionStatus::None);
}

inline char const *failure_reason(Status status,
                                  LineSearchStatus line_search_status,
                                  InnerCgStatus inner_status,
                                  TrustRegionStatus trust_region_status) {
  switch (status) {
  case Status::ConvergedGradient:
  case Status::ConvergedFtol:
  case Status::ConvergedXtol:
  case Status::Running:
    return nullptr;
  case Status::LineSearchFailed:
    switch (line_search_status) {
    case LineSearchStatus::FailedMaxLineSearch:
      return "LINE_SEARCH_FAILED_MAXLS";
    case LineSearchStatus::FailedNonfinite:
      return "LINE_SEARCH_FAILED_NONFINITE";
    case LineSearchStatus::FailedAlphaBounds:
      return "LINE_SEARCH_FAILED_ALPHA_BOUNDS";
    default:
      return "LINE_SEARCH_FAILED";
    }
  case Status::Nonfinite:
    if (line_search_status == LineSearchStatus::FailedNonfinite) {
      return "NONFINITE_TRIAL";
    }
    return "NONFINITE";
  case Status::InnerCgFailed:
    switch (inner_status) {
    case InnerCgStatus::NonfiniteHvp:
      return "INNER_CG_NONFINITE_HVP";
    case InnerCgStatus::PreconditionerBreakdown:
      return "INNER_CG_PRECONDITIONER_BREAKDOWN";
    case InnerCgStatus::MaxIter:
      return "INNER_CG_MAX_ITER";
    case InnerCgStatus::NegativeCurvature:
      return "INNER_CG_NEGATIVE_CURVATURE";
    case InnerCgStatus::ZeroCurvature:
      return "INNER_CG_ZERO_CURVATURE";
    case InnerCgStatus::TrustBoundary:
      return "INNER_CG_TRUST_BOUNDARY";
    default:
      return "INNER_CG_FAILED";
    }
  case Status::TrustRegionFailed:
    switch (trust_region_status) {
    case TrustRegionStatus::FailedNonfinite:
      return "TRUST_REGION_FAILED_NONFINITE";
    case TrustRegionStatus::FailedPredictedReduction:
      return "TRUST_REGION_FAILED_PREDICTED_REDUCTION";
    case TrustRegionStatus::Rejected:
      return "TRUST_REGION_REJECTED";
    default:
      return "TRUST_REGION_FAILED";
    }
  default:
    return name(status);
  }
}

struct OptionsValidation {
  OptionsValidationCode code = OptionsValidationCode::Ok;
  char const *field = "";
  char const *message = "OK";

  explicit operator bool() const { return code == OptionsValidationCode::Ok; }
  bool ok() const { return code == OptionsValidationCode::Ok; }
  char const *code_name() const { return name(code); }

  static OptionsValidation valid() { return OptionsValidation{}; }

  static OptionsValidation invalid(OptionsValidationCode code_in,
                                   char const *field_in,
                                   char const *message_in) {
    return OptionsValidation{code_in, field_in, message_in};
  }
};

struct Options {
  int64_t n = 0;
  int64_t history_size = 10;
  int64_t max_iter = 100;
  int64_t max_line_search = 20;
  int64_t max_eval = 0;
  int64_t max_inner_iter = 0;
  int64_t nonmonotone_window = 10;
  LineSearchPolicy line_search = LineSearchPolicy::HagerZhang;
  DirectionPolicy direction = DirectionPolicy::Lbfgs;
  NlcgBetaPolicy nlcg_beta = NlcgBetaPolicy::DaiYuan;
  LbfgsUpdatePolicy lbfgs_update = LbfgsUpdatePolicy::Skip;
  GlobalizationPolicy globalization = GlobalizationPolicy::LineSearch;
  AlphaGuessPolicy alpha_guess = AlphaGuessPolicy::Initial;
  StoppingPolicy stopping = StoppingPolicy::Standard;
  CostModel cost_model = CostModel::Balanced;
  double initial_step = 1.0;
  double c1 = 1e-4;
  double c2 = 0.9;
  double growth = 10.0;
  double alpha_min = 1e-20;
  double alpha_max = 1e20;
  double gtol_abs = 1e-6;
  double gtol_rel = 0.0;
  double f_atol = 0.0;
  double f_rtol = 0.0;
  double x_atol = 0.0;
  double x_rtol = 0.0;
  double inner_rtol = 0.1;
  double inner_atol = 0.0;
  double initial_trust_radius = 1.0;
  double max_trust_radius = 1e20;
  double trust_eta = 1e-4;
  double trust_shrink = 0.25;
  double trust_grow = 2.0;
  double curvature_eps = 1e-12;
  double gamma_min = 1e-20;
  double gamma_max = 1e20;
  double armijo_shrink_min = 0.1;
  double armijo_shrink_max = 0.5;
  BoundsStrategy bounds = BoundsStrategy::None;
  double bound_margin = 0.0;
  bool accept_decrease_after_maxls = true;

  static LineSearchPolicy default_line_search(DirectionPolicy direction,
                                              CostModel cost_model) {
    if (cost_model == CostModel::JointValueGradient) {
      return LineSearchPolicy::HagerZhang;
    }
    if (cost_model == CostModel::ExpensiveGradient &&
        (direction == DirectionPolicy::Lbfgs ||
         direction == DirectionPolicy::PreconditionedLbfgs ||
         direction == DirectionPolicy::SteepestDescent ||
         direction == DirectionPolicy::TruncatedNewton ||
         direction == DirectionPolicy::PreconditionedTruncatedNewton)) {
      return LineSearchPolicy::ArmijoCubic;
    }
    if (direction == DirectionPolicy::SteepestDescent) {
      return LineSearchPolicy::ArmijoCubic;
    }
    return LineSearchPolicy::HagerZhang;
  }

  static Options with_direction(int64_t dimension, DirectionPolicy direction,
                                CostModel cost_model = CostModel::Balanced) {
    Options options{};
    options.n = dimension;
    options.direction = direction;
    options.cost_model = cost_model;
    options.line_search = default_line_search(direction, cost_model);
    return options;
  }

  static Options lbfgs(int64_t dimension,
                       CostModel cost_model = CostModel::Balanced) {
    return with_direction(dimension, DirectionPolicy::Lbfgs, cost_model);
  }

  static Options plbfgs(int64_t dimension,
                        CostModel cost_model = CostModel::Balanced) {
    return with_direction(dimension, DirectionPolicy::PreconditionedLbfgs,
                          cost_model);
  }

  static Options pstd(int64_t dimension,
                      CostModel cost_model = CostModel::Balanced) {
    return with_direction(dimension, DirectionPolicy::SteepestDescent,
                          cost_model);
  }

  static Options pnlcg(int64_t dimension,
                       CostModel cost_model = CostModel::Balanced) {
    return with_direction(dimension, DirectionPolicy::Nlcg, cost_model);
  }

  static Options trn(int64_t dimension,
                     CostModel cost_model = CostModel::Balanced) {
    return with_direction(dimension, DirectionPolicy::TruncatedNewton,
                          cost_model);
  }

  static Options ptrn(int64_t dimension,
                      CostModel cost_model = CostModel::Balanced) {
    return with_direction(dimension,
                          DirectionPolicy::PreconditionedTruncatedNewton,
                          cost_model);
  }

  tide_optim_options to_c_options() const {
    tide_optim_options options{};
    options.n = n;
    options.history_size = history_size;
    options.max_iter = max_iter;
    options.max_line_search = max_line_search;
    options.max_eval = max_eval;
    options.max_inner_iter = max_inner_iter;
    options.nonmonotone_window = nonmonotone_window;
    options.line_search_policy = static_cast<int32_t>(line_search);
    options.direction_policy = static_cast<int32_t>(direction);
    options.nlcg_beta_policy = static_cast<int32_t>(nlcg_beta);
    options.lbfgs_update_policy = static_cast<int32_t>(lbfgs_update);
    options.globalization_policy = static_cast<int32_t>(globalization);
    options.alpha_guess_policy = static_cast<int32_t>(alpha_guess);
    options.stopping_policy = static_cast<int32_t>(stopping);
    options.initial_step = initial_step;
    options.c1 = c1;
    options.c2 = c2;
    options.growth = growth;
    options.alpha_min = alpha_min;
    options.alpha_max = alpha_max;
    options.gtol_abs = gtol_abs;
    options.gtol_rel = gtol_rel;
    options.f_atol = f_atol;
    options.f_rtol = f_rtol;
    options.x_atol = x_atol;
    options.x_rtol = x_rtol;
    options.inner_rtol = inner_rtol;
    options.inner_atol = inner_atol;
    options.initial_trust_radius = initial_trust_radius;
    options.max_trust_radius = max_trust_radius;
    options.trust_eta = trust_eta;
    options.trust_shrink = trust_shrink;
    options.trust_grow = trust_grow;
    options.curvature_eps = curvature_eps;
    options.gamma_min = gamma_min;
    options.gamma_max = gamma_max;
    options.armijo_shrink_min = armijo_shrink_min;
    options.armijo_shrink_max = armijo_shrink_max;
    options.bound_margin = bound_margin;
    options.bounds_strategy = static_cast<int32_t>(bounds);
    options.accept_decrease_after_maxls =
        accept_decrease_after_maxls ? 1 : 0;
    return options;
  }

  OptionsValidation validate() const {
    if (n <= 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::Dimension, "n", "n must be positive.");
    }
    if (history_size <= 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::HistorySize, "history_size",
          "history_size must be positive.");
    }
    if (max_iter < 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::MaxIter, "max_iter",
          "max_iter must be non-negative.");
    }
    if (max_line_search <= 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::MaxLineSearch, "max_line_search",
          "max_line_search must be positive.");
    }
    if (max_eval < 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::MaxEval, "max_eval",
          "max_eval must be non-negative.");
    }
    if (max_inner_iter < 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::MaxInnerIter, "max_inner_iter",
          "max_inner_iter must be non-negative.");
    }
    if (nonmonotone_window <= 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::NonmonotoneWindow, "nonmonotone_window",
          "nonmonotone_window must be positive.");
    }
    if (!std::isfinite(initial_step) || initial_step <= 0.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::InitialStep, "initial_step",
          "initial_step must be finite and positive.");
    }
    if (!std::isfinite(c1) || !std::isfinite(c2) ||
        !(0.0 < c1 && c1 < c2 && c2 < 1.0)) {
      return OptionsValidation::invalid(
          OptionsValidationCode::WolfeParameters, "c1/c2",
          "Wolfe parameters must satisfy 0 < c1 < c2 < 1.");
    }
    if (!std::isfinite(growth) || growth <= 1.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::Growth, "growth",
          "growth must be finite and greater than 1.");
    }
    switch (line_search) {
    case LineSearchPolicy::LegacyWeakWolfe:
    case LineSearchPolicy::ArmijoCubic:
    case LineSearchPolicy::StrongWolfe:
    case LineSearchPolicy::MoreThuente:
    case LineSearchPolicy::NonmonotoneArmijo:
    case LineSearchPolicy::HagerZhang:
    case LineSearchPolicy::Static:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::LineSearchPolicy, "line_search",
          "line_search policy is not recognized.");
    }
    switch (direction) {
    case DirectionPolicy::SteepestDescent:
    case DirectionPolicy::Lbfgs:
    case DirectionPolicy::Nlcg:
    case DirectionPolicy::PreconditionedLbfgs:
    case DirectionPolicy::TruncatedNewton:
    case DirectionPolicy::PreconditionedTruncatedNewton:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::DirectionPolicy, "direction",
          "direction policy is not recognized.");
    }
    switch (nlcg_beta) {
    case NlcgBetaPolicy::DaiYuan:
    case NlcgBetaPolicy::FletcherReeves:
    case NlcgBetaPolicy::PolakRibierePlus:
    case NlcgBetaPolicy::HagerZhang:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::NlcgBetaPolicy, "nlcg_beta",
          "NLCG beta policy is not recognized.");
    }
    switch (lbfgs_update) {
    case LbfgsUpdatePolicy::Skip:
    case LbfgsUpdatePolicy::Regularize:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::LbfgsUpdatePolicy, "lbfgs_update",
          "LBFGS update policy is not recognized.");
    }
    switch (globalization) {
    case GlobalizationPolicy::LineSearch:
    case GlobalizationPolicy::TrustRegion:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::GlobalizationPolicy, "globalization",
          "globalization policy is not recognized.");
    }
    switch (alpha_guess) {
    case AlphaGuessPolicy::Initial:
    case AlphaGuessPolicy::Previous:
    case AlphaGuessPolicy::BarzilaiBorwein:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::AlphaGuessPolicy, "alpha_guess",
          "alpha guess policy is not recognized.");
    }
    switch (stopping) {
    case StoppingPolicy::Standard:
    case StoppingPolicy::GradientOnly:
    case StoppingPolicy::InitialRelativeF:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::StoppingPolicy, "stopping",
          "stopping policy is not recognized.");
    }
    switch (cost_model) {
    case CostModel::Balanced:
    case CostModel::ExpensiveGradient:
    case CostModel::JointValueGradient:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::CostModel, "cost_model",
          "cost model is not recognized.");
    }
    if (!std::isfinite(alpha_min) || !std::isfinite(alpha_max) ||
        !(0.0 < alpha_min && alpha_min < alpha_max)) {
      return OptionsValidation::invalid(
          OptionsValidationCode::AlphaBounds, "alpha_min/alpha_max",
          "alpha bounds must satisfy 0 < alpha_min < alpha_max.");
    }
    if (!std::isfinite(inner_rtol) || !std::isfinite(inner_atol) ||
        inner_rtol < 0.0 || inner_atol < 0.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::InnerTolerance, "inner_rtol/inner_atol",
          "inner tolerances must be finite and non-negative.");
    }
    if (!std::isfinite(initial_trust_radius) ||
        !std::isfinite(max_trust_radius) ||
        !(0.0 < initial_trust_radius &&
          initial_trust_radius <= max_trust_radius)) {
      return OptionsValidation::invalid(
          OptionsValidationCode::TrustRadius,
          "initial_trust_radius/max_trust_radius",
          "trust radius bounds must satisfy 0 < initial <= max.");
    }
    if (!std::isfinite(trust_eta) || trust_eta < 0.0 || trust_eta >= 1.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::TrustEta, "trust_eta",
          "trust_eta must satisfy 0 <= trust_eta < 1.");
    }
    if (!std::isfinite(trust_shrink) || trust_shrink <= 0.0 ||
        trust_shrink >= 1.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::TrustShrink, "trust_shrink",
          "trust_shrink must satisfy 0 < trust_shrink < 1.");
    }
    if (!std::isfinite(trust_grow) || trust_grow <= 1.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::TrustGrow, "trust_grow",
          "trust_grow must be finite and greater than 1.");
    }
    if (!std::isfinite(curvature_eps) || curvature_eps < 0.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::CurvatureEps, "curvature_eps",
          "curvature_eps must be finite and non-negative.");
    }
    if (!std::isfinite(gamma_min) || !std::isfinite(gamma_max) ||
        !(0.0 < gamma_min && gamma_min <= gamma_max)) {
      return OptionsValidation::invalid(
          OptionsValidationCode::GammaBounds, "gamma_min/gamma_max",
          "gamma bounds must satisfy 0 < gamma_min <= gamma_max.");
    }
    if (!std::isfinite(gtol_abs) || !std::isfinite(gtol_rel) ||
        !std::isfinite(f_atol) || !std::isfinite(f_rtol) ||
        !std::isfinite(x_atol) || !std::isfinite(x_rtol) ||
        gtol_abs < 0.0 || gtol_rel < 0.0 || f_atol < 0.0 ||
        f_rtol < 0.0 || x_atol < 0.0 || x_rtol < 0.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::Tolerances,
          "gtol/f_tol/x_tol",
          "stopping tolerances must be finite and non-negative.");
    }
    if (!std::isfinite(armijo_shrink_min) ||
        !std::isfinite(armijo_shrink_max) || armijo_shrink_min <= 0.0 ||
        armijo_shrink_min > armijo_shrink_max ||
        armijo_shrink_max >= 1.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::ArmijoShrink,
          "armijo_shrink_min/armijo_shrink_max",
          "Armijo shrink factors must satisfy 0 < min <= max < 1.");
    }
    if (!std::isfinite(bound_margin) || bound_margin < 0.0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::BoundMargin, "bound_margin",
          "bound_margin must be finite and non-negative.");
    }
    switch (bounds) {
    case BoundsStrategy::None:
    case BoundsStrategy::ProjectedTrial:
    case BoundsStrategy::ProjectedGradient:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::BoundsStrategy, "bounds",
          "bounds strategy is not recognized.");
    }
    return OptionsValidation::valid();
  }

  bool valid() const { return validate().ok(); }
  char const *method_name() const { return tide::optim::method_name(direction); }
  ResolvedPolicies resolved_policies() const;
  std::string config_signature() const;
  std::string config_fingerprint() const;
};

struct EventCounts {
  int64_t line_search_accept = 0;
  int64_t line_search_rejection = 0;
  int64_t line_search_failure = 0;
  int64_t line_search_fallback_accept = 0;
  int64_t nonfinite_trial = 0;
  int64_t pair_skip = 0;
  int64_t pair_stored = 0;
  int64_t pair_regularized = 0;
  int64_t preconditioner_skip = 0;
  int64_t inner_warning = 0;
  int64_t trust_region_accept = 0;
  int64_t trust_region_rejection = 0;
  int64_t trust_region_failure = 0;
};

struct StoppingDiagnostics {
  StoppingPolicy policy = StoppingPolicy::Standard;
  Status status = Status::Running;
  LineSearchStatus line_search_status = LineSearchStatus::Started;
  InnerCgStatus inner_status = InnerCgStatus::None;
  TrustRegionStatus trust_region_status = TrustRegionStatus::None;
  double f = 0.0;
  double initial_f = 0.0;
  double f_change = 0.0;
  double f_tolerance = 0.0;
  double grad_norm = 0.0;
  double projected_grad_norm = 0.0;
  double initial_grad_norm = 0.0;
  double grad_tolerance = 0.0;
  double step_norm = 0.0;
  double step_tolerance = 0.0;
  int64_t n_iter = 0;
  int64_t n_f = 0;
  int64_t n_g = 0;
  int64_t n_hvp = 0;
  int64_t n_prec = 0;
  int64_t max_iter = 0;
  int64_t max_eval = 0;

  bool success() const {
    return status == Status::ConvergedGradient ||
           status == Status::ConvergedFtol ||
           status == Status::ConvergedXtol;
  }
  bool converged_gradient() const {
    return status == Status::ConvergedGradient;
  }
  bool converged_ftol() const { return status == Status::ConvergedFtol; }
  bool converged_xtol() const { return status == Status::ConvergedXtol; }
  bool stopped() const { return status != Status::Running; }
  bool user_stopped() const { return status == Status::UserStopped; }
  bool failed() const { return stopped() && !success() && !user_stopped(); }
  bool line_search_failed() const {
    return status == Status::LineSearchFailed;
  }
  bool inner_cg_failed() const { return status == Status::InnerCgFailed; }
  bool trust_region_failed() const {
    return status == Status::TrustRegionFailed;
  }
  bool nonfinite() const { return status == Status::Nonfinite; }
  bool gradient_satisfied() const {
    return policy != StoppingPolicy::InitialRelativeF &&
           std::isfinite(grad_norm) && std::isfinite(grad_tolerance) &&
           grad_norm <= grad_tolerance;
  }
  bool ftol_satisfied() const {
    if (policy == StoppingPolicy::GradientOnly ||
        !std::isfinite(f_tolerance) || f_tolerance <= 0.0) {
      return false;
    }
    if (policy == StoppingPolicy::InitialRelativeF) {
      return std::isfinite(f) && f <= f_tolerance;
    }
    return std::isfinite(f_change) && f_change <= f_tolerance;
  }
  bool xtol_satisfied() const {
    return policy == StoppingPolicy::Standard && n_iter > 0 &&
           std::isfinite(step_norm) && std::isfinite(step_tolerance) &&
           step_tolerance > 0.0 && step_norm <= step_tolerance;
  }
  bool max_iter_reached() const {
    return status == Status::MaxIter || (max_iter > 0 && n_iter >= max_iter);
  }
  bool max_eval_reached() const {
    return status == Status::MaxEval || (max_eval > 0 && n_f >= max_eval);
  }
  bool budget_exhausted() const {
    return max_iter_reached() || max_eval_reached();
  }
  char const *policy_name() const { return name(policy); }
  char const *status_name() const { return name(status); }
  char const *reason() const { return tide::optim::reason(status); }
  char const *failure_reason() const {
    return tide::optim::failure_reason(status, line_search_status,
                                       inner_status, trust_region_status);
  }
};

struct DirectionDiagnostics {
  DirectionPolicy policy = DirectionPolicy::SteepestDescent;
  DirectionStatus status = DirectionStatus::Initial;
  NlcgBetaPolicy nlcg_beta = NlcgBetaPolicy::DaiYuan;
  double beta = 0.0;
  double directional_derivative_initial = 0.0;
  double directional_derivative_trial = 0.0;
  int64_t history_size = 0;

  bool beta_finite() const { return std::isfinite(beta); }
  bool derivatives_finite() const {
    return std::isfinite(directional_derivative_initial) &&
           std::isfinite(directional_derivative_trial);
  }
  bool descent_direction() const {
    return derivatives_finite() && directional_derivative_initial < 0.0;
  }
  bool non_descent_direction() const {
    return std::isfinite(directional_derivative_initial) &&
           directional_derivative_initial >= 0.0;
  }
  bool restarted() const {
    return status == DirectionStatus::RestartDenominator ||
           status == DirectionStatus::RestartNonfinite ||
           status == DirectionStatus::RestartNonDescent;
  }
  bool updated() const { return status == DirectionStatus::Update; }
  bool uses_nlcg_beta() const {
    return policy == DirectionPolicy::Nlcg;
  }
  bool uses_history() const {
    return (policy == DirectionPolicy::Lbfgs ||
            policy == DirectionPolicy::PreconditionedLbfgs) &&
           history_size > 0;
  }
  bool finite() const { return beta_finite() && derivatives_finite(); }
  char const *policy_name() const { return name(policy); }
  char const *status_name() const { return name(status); }
  char const *nlcg_beta_name() const { return name(nlcg_beta); }
};

struct BoundsDiagnostics {
  BoundsStrategy strategy = BoundsStrategy::None;
  double projected_grad_norm = 0.0;
  int64_t active_lower_count = 0;
  int64_t active_upper_count = 0;
  int64_t free_count = 0;
  int64_t kkt_violation_count = 0;
  int64_t lower_kkt_violation_count = 0;
  int64_t upper_kkt_violation_count = 0;
  int64_t free_gradient_count = 0;
  int64_t trial_projection_count = 0;
  int64_t trial_lower_projection_count = 0;
  int64_t trial_upper_projection_count = 0;

  int64_t active_count() const {
    return active_lower_count + active_upper_count;
  }
  bool has_active_bounds() const { return active_count() > 0; }
  bool has_kkt_violations() const { return kkt_violation_count > 0; }
  bool has_trial_projections() const { return trial_projection_count > 0; }
  char const *strategy_name() const { return name(strategy); }
};

struct LineSearchDiagnostics {
  LineSearchPolicy policy = LineSearchPolicy::LegacyWeakWolfe;
  LineSearchStatus status = LineSearchStatus::Started;
  LineSearchAcceptance acceptance = LineSearchAcceptance::None;
  double alpha = 0.0;
  double trial_alpha = 0.0;
  double trial_f = 0.0;
  int64_t line_search_iter = 0;
  double reference = 0.0;
  double armijo_rhs = 0.0;
  int64_t accept_count = 0;
  int64_t rejection_count = 0;
  int64_t failure_count = 0;
  int64_t fallback_accept_count = 0;
  int64_t nonfinite_trial_count = 0;

  bool accepted() const {
    return status == LineSearchStatus::Accepted ||
           status == LineSearchStatus::AcceptedDecreaseAfterMaxLineSearch;
  }
  bool failed() const {
    return failure_count > 0 ||
           status == LineSearchStatus::FailedMaxLineSearch ||
           status == LineSearchStatus::FailedNonfinite ||
           status == LineSearchStatus::FailedAlphaBounds;
  }
  bool rejected() const {
    return rejection_count > 0 ||
           status == LineSearchStatus::RejectedArmijo ||
           status == LineSearchStatus::RejectedCurvature;
  }
  bool fallback_accepted() const {
    return fallback_accept_count > 0 ||
           acceptance == LineSearchAcceptance::DecreaseFallback ||
           status == LineSearchStatus::AcceptedDecreaseAfterMaxLineSearch;
  }
  char const *policy_name() const { return name(policy); }
  char const *status_name() const { return name(status); }
  char const *acceptance_name() const { return name(acceptance); }
};

struct PairUpdateDiagnostics {
  PairStatus status = PairStatus::None;
  double sy = 0.0;
  double yy = 0.0;
  double gamma = 0.0;
  int64_t history_size = 0;
  int64_t skip_count = 0;
  int64_t stored_count = 0;
  int64_t regularized_count = 0;

  bool stored() const {
    return status == PairStatus::Stored ||
           status == PairStatus::RegularizedStored;
  }
  bool skipped() const {
    return skip_count > 0 || status == PairStatus::SkippedBadCurvature ||
           status == PairStatus::SkippedNonfinite ||
           status == PairStatus::SkippedLineSearchFallback ||
           status == PairStatus::SkippedBoundsProjection ||
           status == PairStatus::SkippedPreconditioner;
  }
  bool regularized() const {
    return regularized_count > 0 ||
           status == PairStatus::RegularizedStored;
  }
  bool bad_curvature() const {
    return status == PairStatus::SkippedBadCurvature;
  }
  bool history_updated() const { return stored(); }
  char const *status_name() const { return name(status); }
};

struct InnerSolveDiagnostics {
  PreconditionerStatus preconditioner_status = PreconditionerStatus::None;
  InnerCgStatus inner_status = InnerCgStatus::None;
  int64_t inner_iter = 0;
  double residual_norm = 0.0;
  double forcing_tolerance = 0.0;
  double curvature = 0.0;
  double preconditioner_dot = 0.0;
  int64_t n_hvp = 0;
  int64_t n_prec = 0;
  int64_t preconditioner_skip_count = 0;
  int64_t warning_count = 0;

  bool active() const {
    return inner_status != InnerCgStatus::None || inner_iter > 0;
  }
  bool converged() const {
    return inner_status == InnerCgStatus::ForcingReached;
  }
  bool failed() const {
    return inner_status == InnerCgStatus::MaxIter ||
           inner_status == InnerCgStatus::NonfiniteHvp ||
           inner_status == InnerCgStatus::PreconditionerBreakdown;
  }
  bool warned() const { return warning_count > 0 || failed(); }
  bool negative_curvature() const {
    return inner_status == InnerCgStatus::NegativeCurvature;
  }
  bool trust_boundary() const {
    return inner_status == InnerCgStatus::TrustBoundary;
  }
  bool preconditioner_applied() const {
    return preconditioner_status == PreconditionerStatus::Applied;
  }
  bool preconditioner_failed() const {
    return preconditioner_status == PreconditionerStatus::SkippedNonfinite ||
           preconditioner_status == PreconditionerStatus::SkippedNotPositive ||
           inner_status == InnerCgStatus::PreconditionerBreakdown ||
           preconditioner_skip_count > 0;
  }
  bool hvp_failed() const {
    return inner_status == InnerCgStatus::NonfiniteHvp;
  }
  bool residual_satisfied() const {
    return residual_norm <= forcing_tolerance && forcing_tolerance >= 0.0;
  }
  char const *preconditioner_status_name() const {
    return name(preconditioner_status);
  }
  char const *inner_status_name() const { return name(inner_status); }
};

struct TrustRegionDiagnostics {
  GlobalizationPolicy globalization = GlobalizationPolicy::LineSearch;
  TrustRegionStatus status = TrustRegionStatus::None;
  double radius = 0.0;
  double ratio = 0.0;
  double predicted_reduction = 0.0;
  double actual_reduction = 0.0;
  int64_t accept_count = 0;
  int64_t rejection_count = 0;
  int64_t failure_count = 0;

  bool active() const {
    return globalization == GlobalizationPolicy::TrustRegion ||
           status != TrustRegionStatus::None;
  }
  bool accepted() const {
    return status == TrustRegionStatus::Accepted || accept_count > 0;
  }
  bool rejected() const {
    return status == TrustRegionStatus::Rejected || rejection_count > 0;
  }
  bool failed() const {
    return failure_count > 0 || status == TrustRegionStatus::FailedNonfinite ||
           status == TrustRegionStatus::FailedPredictedReduction;
  }
  bool has_model_reduction() const { return predicted_reduction > 0.0; }
  bool ratio_is_finite() const { return std::isfinite(ratio); }
  bool reduction_matches_model() const {
    double const expected = predicted_reduction * ratio;
    double const scale = 1.0 + std::abs(actual_reduction) + std::abs(expected);
    return has_model_reduction() && ratio_is_finite() &&
           std::abs(actual_reduction - expected) <= 1e-10 * scale;
  }
  char const *globalization_name() const { return name(globalization); }
  char const *status_name() const { return name(status); }
};

struct ReportView {
  tide_optim_report const *data = nullptr;

  explicit operator bool() const { return data != nullptr; }
  tide_optim_report const *raw() const { return data; }

  RequestKind request_kind() const {
    return data == nullptr ? RequestKind::Error
                           : static_cast<RequestKind>(data->request);
  }
  char const *request_name() const { return name(request_kind()); }
  int64_t request_sequence() const {
    return data == nullptr ? 0 : data->request_sequence;
  }
  int64_t n() const { return data == nullptr ? 0 : data->n; }
  char const *expected_evaluation() const {
    return tide::optim::expected_evaluation(request_kind());
  }
  char const *required_fields() const {
    return tide::optim::required_fields(request_kind());
  }
  char const *accepted_mapping_keys() const {
    return tide::optim::accepted_mapping_keys(request_kind());
  }
  bool requires_evaluation() const {
    return tide::optim::requires_evaluation(request_kind());
  }
  bool error() const { return request_kind() == RequestKind::Error; }
  bool needs_value() const {
    RequestKind const request = request_kind();
    return request == RequestKind::EvaluateF ||
           request == RequestKind::EvaluateFG;
  }
  bool needs_gradient() const {
    return request_kind() == RequestKind::EvaluateFG;
  }
  bool needs_value_gradient() const {
    return request_kind() == RequestKind::EvaluateFG;
  }
  bool needs_preconditioner() const {
    return request_kind() == RequestKind::ApplyPreconditioner;
  }
  bool needs_hessian_vector() const {
    return request_kind() == RequestKind::EvaluateHv;
  }
  bool needs_vector_result() const {
    RequestKind const request = request_kind();
    return request == RequestKind::ApplyPreconditioner ||
           request == RequestKind::EvaluateHv;
  }
  int64_t expected_gradient_size() const {
    return needs_gradient() ? n() : 0;
  }
  int64_t expected_vector_size() const {
    return needs_vector_result() ? n() : 0;
  }

  Status status() const {
    return data == nullptr ? Status::InvalidArgument
                           : static_cast<Status>(data->status);
  }
  char const *status_name() const { return name(status()); }
  char const *reason() const { return tide::optim::reason(status()); }
  char const *failure_reason() const {
    return tide::optim::failure_reason(status(), line_search_status(),
                                       inner_status(), trust_region_status());
  }

  LineSearchStatus line_search_status() const {
    return data == nullptr
               ? LineSearchStatus::FailedNonfinite
               : static_cast<LineSearchStatus>(data->line_search_status);
  }
  char const *line_search_status_name() const {
    return name(line_search_status());
  }

  LineSearchAcceptance line_search_acceptance() const {
    return data == nullptr
               ? LineSearchAcceptance::None
               : static_cast<LineSearchAcceptance>(
                     data->line_search_acceptance);
  }
  char const *line_search_acceptance_name() const {
    return name(line_search_acceptance());
  }

  PairStatus pair_status() const {
    return data == nullptr ? PairStatus::None
                           : static_cast<PairStatus>(data->pair_status);
  }
  char const *pair_status_name() const { return name(pair_status()); }

  LineSearchPolicy line_search_policy() const {
    return data == nullptr
               ? LineSearchPolicy::LegacyWeakWolfe
               : static_cast<LineSearchPolicy>(data->line_search_policy);
  }
  char const *line_search_policy_name() const {
    return name(line_search_policy());
  }

  DirectionPolicy direction_policy() const {
    return data == nullptr ? DirectionPolicy::SteepestDescent
                           : static_cast<DirectionPolicy>(
                                 data->direction_policy);
  }
  char const *direction_policy_name() const { return name(direction_policy()); }
  char const *method_name() const {
    return tide::optim::method_name(direction_policy());
  }

  NlcgBetaPolicy nlcg_beta_policy() const {
    return data == nullptr
               ? NlcgBetaPolicy::DaiYuan
               : static_cast<NlcgBetaPolicy>(data->nlcg_beta_policy);
  }
  char const *nlcg_beta_policy_name() const {
    return name(nlcg_beta_policy());
  }

  LbfgsUpdatePolicy lbfgs_update_policy() const {
    return data == nullptr
               ? LbfgsUpdatePolicy::Skip
               : static_cast<LbfgsUpdatePolicy>(data->lbfgs_update_policy);
  }
  char const *lbfgs_update_policy_name() const {
    return name(lbfgs_update_policy());
  }

  PreconditionerStatus preconditioner_status() const {
    return data == nullptr
               ? PreconditionerStatus::None
               : static_cast<PreconditionerStatus>(
                     data->preconditioner_status);
  }
  char const *preconditioner_status_name() const {
    return name(preconditioner_status());
  }

  InnerCgStatus inner_status() const {
    return data == nullptr ? InnerCgStatus::None
                           : static_cast<InnerCgStatus>(data->inner_status);
  }
  char const *inner_status_name() const { return name(inner_status()); }

  GlobalizationPolicy globalization_policy() const {
    return data == nullptr
               ? GlobalizationPolicy::LineSearch
               : static_cast<GlobalizationPolicy>(data->globalization_policy);
  }
  char const *globalization_policy_name() const {
    return name(globalization_policy());
  }

  TrustRegionStatus trust_region_status() const {
    return data == nullptr
               ? TrustRegionStatus::None
               : static_cast<TrustRegionStatus>(data->trust_region_status);
  }
  char const *trust_region_status_name() const {
    return name(trust_region_status());
  }

  AlphaGuessPolicy alpha_guess_policy() const {
    return data == nullptr
               ? AlphaGuessPolicy::Initial
               : static_cast<AlphaGuessPolicy>(data->alpha_guess_policy);
  }
  char const *alpha_guess_policy_name() const {
    return name(alpha_guess_policy());
  }

  StoppingPolicy stopping_policy() const {
    return data == nullptr ? StoppingPolicy::Standard
                           : static_cast<StoppingPolicy>(
                                 data->stopping_policy);
  }
  char const *stopping_policy_name() const { return name(stopping_policy()); }

  DirectionStatus direction_status() const {
    return data == nullptr
               ? DirectionStatus::Initial
               : static_cast<DirectionStatus>(data->direction_status);
  }
  char const *direction_status_name() const { return name(direction_status()); }

  int32_t warning_flags() const {
    return data == nullptr ? 0 : data->warning_flags;
  }
  bool has_warning(WarningFlag warning) const {
    return tide::optim::has_warning(warning_flags(), warning);
  }
  std::vector<WarningFlag> warnings() const {
    return tide::optim::warning_flags(warning_flags());
  }
  std::vector<char const *> warning_names() const {
    return tide::optim::warning_names(warning_flags());
  }

  bool done() const { return request_kind() == RequestKind::Done; }
  bool line_search_failed() const {
    return status() == Status::LineSearchFailed ||
           line_search_status() == LineSearchStatus::FailedMaxLineSearch ||
           line_search_status() == LineSearchStatus::FailedNonfinite ||
           line_search_status() == LineSearchStatus::FailedAlphaBounds;
  }

  int64_t iter() const { return data == nullptr ? 0 : data->iter; }
  int64_t n_f() const { return data == nullptr ? 0 : data->n_f; }
  int64_t n_g() const { return data == nullptr ? 0 : data->n_g; }
  int64_t n_hvp() const { return data == nullptr ? 0 : data->n_hvp; }
  int64_t n_prec() const { return data == nullptr ? 0 : data->n_prec; }
  int64_t max_iter() const { return data == nullptr ? 0 : data->max_iter; }
  int64_t max_eval() const { return data == nullptr ? 0 : data->max_eval; }
  int64_t line_search_iter() const {
    return data == nullptr ? 0 : data->line_search_iter;
  }
  int64_t inner_iter() const { return data == nullptr ? 0 : data->inner_iter; }
  int64_t history_size() const {
    return data == nullptr ? 0 : data->history_size;
  }

  double f() const { return data == nullptr ? 0.0 : data->f; }
  double grad_norm() const {
    return data == nullptr ? 0.0 : data->grad_norm;
  }
  double alpha() const { return data == nullptr ? 0.0 : data->alpha; }
  double trial_alpha() const {
    return data == nullptr ? 0.0 : data->trial_alpha;
  }
  double trial_f() const { return data == nullptr ? 0.0 : data->trial_f; }
  double line_search_reference() const {
    return data == nullptr ? 0.0 : data->line_search_reference;
  }
  double line_search_armijo_rhs() const {
    return data == nullptr ? 0.0 : data->line_search_armijo_rhs;
  }
  double step_norm() const {
    return data == nullptr ? 0.0 : data->step_norm;
  }
  double step_tolerance() const {
    return data == nullptr ? 0.0 : data->step_tolerance;
  }
  double grad_tolerance() const {
    return data == nullptr ? 0.0 : data->grad_tolerance;
  }
  double f_change() const {
    return data == nullptr ? 0.0 : data->f_change;
  }
  double f_tolerance() const {
    return data == nullptr ? 0.0 : data->f_tolerance;
  }
  double initial_f() const {
    return data == nullptr ? 0.0 : data->initial_f;
  }
  double initial_grad_norm() const {
    return data == nullptr ? 0.0 : data->initial_grad_norm;
  }
  double direction_beta() const {
    return data == nullptr ? 0.0 : data->direction_beta;
  }
  double directional_derivative_initial() const {
    return data == nullptr ? 0.0 : data->directional_derivative_initial;
  }
  double directional_derivative_trial() const {
    return data == nullptr ? 0.0 : data->directional_derivative_trial;
  }
  double sy() const { return data == nullptr ? 0.0 : data->sy; }
  double yy() const { return data == nullptr ? 0.0 : data->yy; }
  double gamma() const { return data == nullptr ? 0.0 : data->gamma; }
  double preconditioner_dot() const {
    return data == nullptr ? 0.0 : data->preconditioner_dot;
  }
  double inner_residual_norm() const {
    return data == nullptr ? 0.0 : data->inner_residual_norm;
  }
  double inner_forcing_tolerance() const {
    return data == nullptr ? 0.0 : data->inner_forcing_tolerance;
  }
  double inner_curvature() const {
    return data == nullptr ? 0.0 : data->inner_curvature;
  }
  double trust_radius() const {
    return data == nullptr ? 0.0 : data->trust_radius;
  }
  double trust_ratio() const {
    return data == nullptr ? 0.0 : data->trust_ratio;
  }
  double predicted_reduction() const {
    return data == nullptr ? 0.0 : data->predicted_reduction;
  }
  double actual_reduction() const {
    return data == nullptr ? 0.0 : data->actual_reduction;
  }
  double projected_grad_norm() const {
    return data == nullptr ? 0.0 : data->projected_grad_norm;
  }
  int64_t active_lower_count() const {
    return data == nullptr ? 0 : data->active_lower_count;
  }
  int64_t active_upper_count() const {
    return data == nullptr ? 0 : data->active_upper_count;
  }
  int64_t free_count() const { return data == nullptr ? 0 : data->free_count; }
  int64_t kkt_violation_count() const {
    return data == nullptr ? 0 : data->kkt_violation_count;
  }
  int64_t lower_kkt_violation_count() const {
    return data == nullptr ? 0 : data->lower_kkt_violation_count;
  }
  int64_t upper_kkt_violation_count() const {
    return data == nullptr ? 0 : data->upper_kkt_violation_count;
  }
  int64_t free_gradient_count() const {
    return data == nullptr ? 0 : data->free_gradient_count;
  }
  int64_t trial_projection_count() const {
    return data == nullptr ? 0 : data->trial_projection_count;
  }
  int64_t trial_lower_projection_count() const {
    return data == nullptr ? 0 : data->trial_lower_projection_count;
  }
  int64_t trial_upper_projection_count() const {
    return data == nullptr ? 0 : data->trial_upper_projection_count;
  }

  StoppingDiagnostics stopping_diagnostics() const {
    return StoppingDiagnostics{
        stopping_policy(),
        status(),
        line_search_status(),
        inner_status(),
        trust_region_status(),
        f(),
        initial_f(),
        f_change(),
        f_tolerance(),
        grad_norm(),
        projected_grad_norm(),
        initial_grad_norm(),
        grad_tolerance(),
        step_norm(),
        step_tolerance(),
        iter(),
        n_f(),
        n_g(),
        n_hvp(),
        n_prec(),
        max_iter(),
        max_eval(),
    };
  }

  DirectionDiagnostics direction_diagnostics() const {
    return DirectionDiagnostics{
        direction_policy(),
        direction_status(),
        nlcg_beta_policy(),
        direction_beta(),
        directional_derivative_initial(),
        directional_derivative_trial(),
        history_size(),
    };
  }

  BoundsDiagnostics bounds_diagnostics(
      BoundsStrategy strategy = BoundsStrategy::None) const {
    return BoundsDiagnostics{
        strategy,
        projected_grad_norm(),
        active_lower_count(),
        active_upper_count(),
        free_count(),
        kkt_violation_count(),
        lower_kkt_violation_count(),
        upper_kkt_violation_count(),
        free_gradient_count(),
        trial_projection_count(),
        trial_lower_projection_count(),
        trial_upper_projection_count(),
    };
  }

  LineSearchDiagnostics line_search_diagnostics() const {
    EventCounts const counts = event_counts();
    return LineSearchDiagnostics{
        line_search_policy(),
        line_search_status(),
        line_search_acceptance(),
        alpha(),
        trial_alpha(),
        trial_f(),
        line_search_iter(),
        line_search_reference(),
        line_search_armijo_rhs(),
        counts.line_search_accept,
        counts.line_search_rejection,
        counts.line_search_failure,
        counts.line_search_fallback_accept,
        counts.nonfinite_trial,
    };
  }

  PairUpdateDiagnostics pair_update_diagnostics() const {
    EventCounts const counts = event_counts();
    return PairUpdateDiagnostics{
        pair_status(),
        sy(),
        yy(),
        gamma(),
        history_size(),
        counts.pair_skip,
        counts.pair_stored,
        counts.pair_regularized,
    };
  }

  InnerSolveDiagnostics inner_solve_diagnostics() const {
    EventCounts const counts = event_counts();
    return InnerSolveDiagnostics{
        preconditioner_status(),
        inner_status(),
        inner_iter(),
        inner_residual_norm(),
        inner_forcing_tolerance(),
        inner_curvature(),
        preconditioner_dot(),
        n_hvp(),
        n_prec(),
        counts.preconditioner_skip,
        counts.inner_warning,
    };
  }

  TrustRegionDiagnostics trust_region_diagnostics() const {
    EventCounts const counts = event_counts();
    return TrustRegionDiagnostics{
        globalization_policy(),
        trust_region_status(),
        trust_radius(),
        trust_ratio(),
        predicted_reduction(),
        actual_reduction(),
        counts.trust_region_accept,
        counts.trust_region_rejection,
        counts.trust_region_failure,
    };
  }

  EventCounts event_counts() const {
    if (data == nullptr) {
      return EventCounts{};
    }
    return EventCounts{
        data->line_search_accept_count,
        data->line_search_rejection_count,
        data->line_search_failure_count,
        data->line_search_fallback_accept_count,
        data->nonfinite_trial_count,
        data->pair_skip_count,
        data->pair_stored_count,
        data->pair_regularized_count,
        data->preconditioner_skip_count,
        data->inner_warning_count,
        data->trust_region_accept_count,
        data->trust_region_rejection_count,
        data->trust_region_failure_count,
    };
  }
};

struct ResolvedPolicies {
  DirectionPolicy direction = DirectionPolicy::Lbfgs;
  LineSearchPolicy line_search = LineSearchPolicy::HagerZhang;
  AlphaGuessPolicy alpha_guess = AlphaGuessPolicy::Initial;
  StoppingPolicy stopping = StoppingPolicy::Standard;
  NlcgBetaPolicy nlcg_beta = NlcgBetaPolicy::DaiYuan;
  LbfgsUpdatePolicy lbfgs_update = LbfgsUpdatePolicy::Skip;
  GlobalizationPolicy globalization = GlobalizationPolicy::LineSearch;
  BoundsStrategy bounds = BoundsStrategy::None;
  CostModel cost_model = CostModel::Balanced;

  static ResolvedPolicies from_options(Options const &options) {
    ResolvedPolicies policies{};
    policies.direction = options.direction;
    policies.line_search = options.line_search;
    policies.alpha_guess = options.alpha_guess;
    policies.stopping = options.stopping;
    policies.nlcg_beta = options.nlcg_beta;
    policies.lbfgs_update = options.lbfgs_update;
    policies.globalization = options.globalization;
    policies.bounds = options.bounds;
    policies.cost_model = options.cost_model;
    return policies;
  }

  static ResolvedPolicies from_report(ReportView report,
                                      Options const *options = nullptr) {
    ResolvedPolicies policies =
        options == nullptr ? ResolvedPolicies{} : from_options(*options);
    if (report) {
      policies.direction = report.direction_policy();
      policies.line_search = report.line_search_policy();
      policies.alpha_guess = report.alpha_guess_policy();
      policies.stopping = report.stopping_policy();
      policies.nlcg_beta = report.nlcg_beta_policy();
      policies.lbfgs_update = report.lbfgs_update_policy();
      policies.globalization = report.globalization_policy();
    }
    return policies;
  }

  char const *method_name() const { return tide::optim::method_name(direction); }
  char const *direction_name() const { return name(direction); }
  char const *line_search_name() const { return name(line_search); }
  char const *alpha_guess_name() const { return name(alpha_guess); }
  char const *stopping_name() const { return name(stopping); }
  char const *nlcg_beta_name() const { return name(nlcg_beta); }
  char const *lbfgs_update_name() const { return name(lbfgs_update); }
  char const *globalization_name() const { return name(globalization); }
  char const *bounds_name() const { return name(bounds); }
  char const *cost_model_name() const { return name(cost_model); }
};

inline std::string make_config_signature(Options const &options,
                                         ResolvedPolicies const &policies) {
  std::ostringstream out;
  out << std::setprecision(17)
      << "schema=tide.optim.config.v1"
      << ";backend=cpp"
      << ";method=" << options.method_name()
      << ";n=" << options.n
      << ";history_size=" << options.history_size
      << ";max_iter=" << options.max_iter
      << ";max_line_search=" << options.max_line_search
      << ";max_eval=" << options.max_eval
      << ";max_inner_iter=" << options.max_inner_iter
      << ";nonmonotone_window=" << options.nonmonotone_window
      << ";line_search=" << name(options.line_search)
      << ";direction=" << name(options.direction)
      << ";nlcg_beta=" << name(options.nlcg_beta)
      << ";lbfgs_update=" << name(options.lbfgs_update)
      << ";globalization=" << name(options.globalization)
      << ";alpha_guess=" << name(options.alpha_guess)
      << ";stopping=" << name(options.stopping)
      << ";cost_model=" << name(options.cost_model)
      << ";initial_step=" << options.initial_step
      << ";c1=" << options.c1
      << ";c2=" << options.c2
      << ";growth=" << options.growth
      << ";alpha_min=" << options.alpha_min
      << ";alpha_max=" << options.alpha_max
      << ";gtol_abs=" << options.gtol_abs
      << ";gtol_rel=" << options.gtol_rel
      << ";f_atol=" << options.f_atol
      << ";f_rtol=" << options.f_rtol
      << ";x_atol=" << options.x_atol
      << ";x_rtol=" << options.x_rtol
      << ";inner_rtol=" << options.inner_rtol
      << ";inner_atol=" << options.inner_atol
      << ";initial_trust_radius=" << options.initial_trust_radius
      << ";max_trust_radius=" << options.max_trust_radius
      << ";trust_eta=" << options.trust_eta
      << ";trust_shrink=" << options.trust_shrink
      << ";trust_grow=" << options.trust_grow
      << ";curvature_eps=" << options.curvature_eps
      << ";gamma_min=" << options.gamma_min
      << ";gamma_max=" << options.gamma_max
      << ";armijo_shrink_min=" << options.armijo_shrink_min
      << ";armijo_shrink_max=" << options.armijo_shrink_max
      << ";bounds=" << name(options.bounds)
      << ";bound_margin=" << options.bound_margin
      << ";accept_decrease_after_maxls="
      << (options.accept_decrease_after_maxls ? "true" : "false")
      << ";policy.method=" << policies.method_name()
      << ";policy.direction=" << policies.direction_name()
      << ";policy.line_search=" << policies.line_search_name()
      << ";policy.alpha_guess=" << policies.alpha_guess_name()
      << ";policy.stopping=" << policies.stopping_name()
      << ";policy.nlcg_beta=" << policies.nlcg_beta_name()
      << ";policy.lbfgs_update=" << policies.lbfgs_update_name()
      << ";policy.globalization=" << policies.globalization_name()
      << ";policy.bounds=" << policies.bounds_name()
      << ";policy.cost_model=" << policies.cost_model_name();
  return out.str();
}

inline std::string make_config_fingerprint(std::string const &signature) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char ch : signature) {
    hash ^= static_cast<uint64_t>(ch);
    hash *= 1099511628211ull;
  }
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setw(16)
      << std::setfill('0') << hash;
  return out.str();
}

inline std::string make_config_fingerprint(
    Options const &options, ResolvedPolicies const &policies) {
  return make_config_fingerprint(make_config_signature(options, policies));
}

inline ResolvedPolicies Options::resolved_policies() const {
  return ResolvedPolicies::from_options(*this);
}

inline std::string Options::config_signature() const {
  return make_config_signature(*this, resolved_policies());
}

inline std::string Options::config_fingerprint() const {
  return make_config_fingerprint(config_signature());
}

struct VectorView {
  double const *data = nullptr;
  int64_t size = 0;
};

struct MutableVectorView {
  double *data = nullptr;
  int64_t size = 0;

  operator VectorView() const { return VectorView{data, size}; }
};

inline VectorView view(std::vector<double> const &values) {
  return VectorView{values.data(), static_cast<int64_t>(values.size())};
}

inline MutableVectorView view(std::vector<double> &values) {
  return MutableVectorView{values.data(),
                           static_cast<int64_t>(values.size())};
}

struct RequestRequirements {
  RequestKind kind = RequestKind::Error;
  int64_t sequence = 0;

  static RequestRequirements from_kind(RequestKind kind_in,
                                       int64_t sequence_in = 0) {
    return RequestRequirements{kind_in, sequence_in};
  }

  char const *kind_name() const { return name(kind); }
  bool requires_evaluation() const { return tide::optim::requires_evaluation(kind); }
  char const *expected_evaluation() const {
    return tide::optim::expected_evaluation(kind);
  }
  char const *required_fields() const {
    return tide::optim::required_fields(kind);
  }
  char const *accepted_mapping_keys() const {
    return tide::optim::accepted_mapping_keys(kind);
  }
  bool needs_value() const {
    return kind == RequestKind::EvaluateF || kind == RequestKind::EvaluateFG;
  }
  bool needs_gradient() const { return kind == RequestKind::EvaluateFG; }
  bool needs_value_gradient() const { return kind == RequestKind::EvaluateFG; }
  bool needs_preconditioner() const {
    return kind == RequestKind::ApplyPreconditioner;
  }
  bool needs_hessian_vector() const { return kind == RequestKind::EvaluateHv; }
  bool needs_vector_result() const {
    return needs_preconditioner() || needs_hessian_vector();
  }
};

struct Request {
  RequestKind kind = RequestKind::Error;
  VectorView x{};
  VectorView vector{};
  ReportView report{};
  int64_t sequence = 0;

  RequestRequirements requirements() const {
    return RequestRequirements::from_kind(kind, sequence);
  }
  char const *kind_name() const { return name(kind); }
  char const *expected_evaluation() const {
    return requirements().expected_evaluation();
  }
  char const *required_fields() const { return requirements().required_fields(); }
  char const *accepted_mapping_keys() const {
    return requirements().accepted_mapping_keys();
  }
  bool requires_evaluation() const {
    return requirements().requires_evaluation();
  }
  bool error() const { return kind == RequestKind::Error; }
  bool done() const { return kind == RequestKind::Done; }
  bool needs_value() const { return requirements().needs_value(); }
  bool needs_gradient() const { return requirements().needs_gradient(); }
  bool needs_value_gradient() const {
    return requirements().needs_value_gradient();
  }
  bool needs_preconditioner() const {
    return requirements().needs_preconditioner();
  }
  bool needs_hessian_vector() const {
    return requirements().needs_hessian_vector();
  }
  bool needs_vector_result() const {
    return requirements().needs_vector_result();
  }
  bool has_x() const { return x.data != nullptr; }
  bool has_vector() const { return vector.data != nullptr; }
  bool has_report() const { return static_cast<bool>(report); }
  int64_t x_size() const { return x.size; }
  int64_t vector_size() const { return vector.size; }
  int64_t expected_gradient_size() const {
    return needs_gradient() ? x_size() : 0;
  }
  int64_t expected_vector_size() const {
    return needs_vector_result() ? vector_size() : 0;
  }
};

struct EvaluationStatus {
  RequestRequirements requirements{};
  bool has_value = false;
  bool has_gradient = false;
  bool has_vector = false;
  int64_t gradient_size = 0;
  int64_t vector_size = 0;
  int64_t expected_gradient_size = 0;
  int64_t expected_vector_size = 0;

  bool missing_value() const {
    return requirements.needs_value() && !has_value;
  }
  bool missing_gradient() const {
    return requirements.needs_gradient() && !has_gradient;
  }
  bool missing_vector() const {
    return requirements.needs_vector_result() && !has_vector;
  }
  bool has_missing_fields() const {
    return missing_value() || missing_gradient() || missing_vector();
  }
  bool gradient_size_mismatch() const {
    return requirements.needs_gradient() && has_gradient &&
           expected_gradient_size > 0 && gradient_size != expected_gradient_size;
  }
  bool vector_size_mismatch() const {
    return requirements.needs_vector_result() && has_vector &&
           expected_vector_size > 0 && vector_size != expected_vector_size;
  }
  bool has_size_mismatch() const {
    return gradient_size_mismatch() || vector_size_mismatch();
  }
  bool satisfied() const {
    return requirements.requires_evaluation() && !has_missing_fields() &&
           !has_size_mismatch();
  }
  bool valid() const { return satisfied(); }
  char const *expected_evaluation() const {
    return requirements.expected_evaluation();
  }
  char const *required_fields() const {
    return requirements.required_fields();
  }
  char const *accepted_mapping_keys() const {
    return requirements.accepted_mapping_keys();
  }
  char const *missing_fields() const {
    if (missing_value() && missing_gradient()) {
      return "f,g";
    }
    if (missing_value()) {
      return "f";
    }
    if (missing_gradient()) {
      return "g";
    }
    if (missing_vector()) {
      return "vector";
    }
    return "";
  }
  char const *mismatched_fields() const {
    if (gradient_size_mismatch()) {
      return "g";
    }
    if (vector_size_mismatch()) {
      return "vector";
    }
    return "";
  }
};

struct Evaluation {
  double f = 0.0;
  VectorView gradient{};
  VectorView vector{};
  bool value_set = false;
  bool gradient_set = false;
  bool vector_set = false;

  static Evaluation value(double f_in) {
    Evaluation evaluation{};
    evaluation.f = f_in;
    evaluation.value_set = true;
    return evaluation;
  }

  static Evaluation value_gradient(double f_in, VectorView gradient_in) {
    Evaluation evaluation = value(f_in);
    evaluation.gradient = gradient_in;
    evaluation.gradient_set = true;
    return evaluation;
  }

  static Evaluation vector_result(VectorView vector_in) {
    Evaluation evaluation{};
    evaluation.vector = vector_in;
    evaluation.vector_set = true;
    return evaluation;
  }

  static Evaluation preconditioner(VectorView z) { return vector_result(z); }
  static Evaluation hessian_vector(VectorView hv) { return vector_result(hv); }

  bool has_value() const { return value_set; }
  bool has_gradient() const { return gradient_set && gradient.data != nullptr; }
  bool has_vector() const { return vector_set && vector.data != nullptr; }
  int64_t gradient_size() const { return has_gradient() ? gradient.size : 0; }
  int64_t vector_size() const { return has_vector() ? vector.size : 0; }

  EvaluationStatus status_for(Request const &request) const {
    return EvaluationStatus{
        request.requirements(),
        has_value(),
        has_gradient(),
        has_vector(),
        gradient_size(),
        vector_size(),
        request.needs_gradient() ? request.x_size() : 0,
        request.needs_vector_result() ? request.vector_size() : 0,
    };
  }

  EvaluationStatus status_for(RequestRequirements requirements) const {
    return EvaluationStatus{requirements, has_value(), has_gradient(),
                            has_vector(), gradient_size(), vector_size(),
                            0, 0};
  }

  EvaluationStatus status_for(ReportView report) const {
    return EvaluationStatus{
        RequestRequirements::from_kind(report.request_kind(),
                                       report.request_sequence()),
        has_value(),
        has_gradient(),
        has_vector(),
        gradient_size(),
        vector_size(),
        report.expected_gradient_size(),
        report.expected_vector_size(),
    };
  }

  char const *missing_fields(Request const &request) const {
    return status_for(request).missing_fields();
  }

  char const *missing_fields(ReportView report) const {
    return status_for(report).missing_fields();
  }

  bool satisfied_by(Request const &request) const {
    return status_for(request).satisfied();
  }

  bool satisfied_by(ReportView report) const {
    return status_for(report).satisfied();
  }

  bool valid_for(Request const &request) const {
    return satisfied_by(request);
  }

  bool valid_for(ReportView report) const { return satisfied_by(report); }
};

class Session {
public:
  explicit Session(Options const &options) : Session(options.to_c_options()) {}

  explicit Session(tide_optim_options const &options)
      : handle_(tide_optim_create(&options)), n_(options.n),
        request_x_(allocate_work(options.n)),
        current_x_(allocate_work(options.n)) {}

  ~Session() { close(); }

  Session(Session const &) = delete;
  Session &operator=(Session const &) = delete;

  Session(Session &&other) noexcept
      : handle_(std::exchange(other.handle_, nullptr)), n_(other.n_),
        request_x_(std::move(other.request_x_)),
        current_x_(std::move(other.current_x_)), report_(other.report_),
        started_(std::exchange(other.started_, false)) {}

  Session &operator=(Session &&other) noexcept {
    if (this != &other) {
      close();
      handle_ = std::exchange(other.handle_, nullptr);
      n_ = other.n_;
      request_x_ = std::move(other.request_x_);
      current_x_ = std::move(other.current_x_);
      report_ = other.report_;
      started_ = std::exchange(other.started_, false);
    }
    return *this;
  }

  bool valid() const { return handle_ != nullptr; }
  bool closed() const { return handle_ == nullptr; }
  bool started() const { return started_; }
  bool done() const {
    return started_ && last_report_view().request_kind() == RequestKind::Done;
  }
  bool running() const { return valid() && started_ && !done(); }
  char const *state_name() const {
    if (closed()) {
      return "CLOSED";
    }
    if (!started()) {
      return "NOT_STARTED";
    }
    if (done()) {
      return "DONE";
    }
    return "RUNNING";
  }

  void close() {
    if (handle_ != nullptr) {
      tide_optim_destroy(handle_);
      handle_ = nullptr;
    }
  }

  Status set_bounds(double const *lb, double const *ub) {
    if (handle_ == nullptr) {
      return Status::InvalidArgument;
    }
    return static_cast<Status>(tide_optim_set_bounds(handle_, lb, ub));
  }

  Status set_bounds(VectorView lb, VectorView ub) {
    if (!valid_view(lb) || !valid_view(ub)) {
      return Status::InvalidArgument;
    }
    return set_bounds(lb.data, ub.data);
  }

  Status clear_bounds() { return set_bounds(nullptr, nullptr); }

  Request start(double const *x, double f, double const *g) {
    return start(VectorView{x, n_}, f, VectorView{g, n_});
  }

  Request start(VectorView x, double f, VectorView g) {
    if (handle_ == nullptr || !valid_view(x) || !valid_view(g)) {
      return error_request();
    }
    tide_optim_start(handle_, x.data, f, g.data, request_x_.data(),
                     &report_);
    started_ = true;
    return make_request();
  }

  Request tell(double f, double const *g) {
    return tell(f, VectorView{g, n_});
  }

  Request tell(double f, VectorView g) {
    if (handle_ == nullptr || !valid_view(g)) {
      return error_request();
    }
    tide_optim_tell(handle_, f, g.data, request_x_.data(), &report_);
    return make_request();
  }

  Request tell_value(double f) {
    if (handle_ == nullptr) {
      return error_request();
    }
    tide_optim_tell_value(handle_, f, request_x_.data(), &report_);
    return make_request();
  }

  Request tell_preconditioner(double const *z) {
    return tell_preconditioner(VectorView{z, n_});
  }

  Request tell_preconditioner(VectorView z) {
    if (handle_ == nullptr || !valid_view(z)) {
      return error_request();
    }
    tide_optim_tell_preconditioner(handle_, z.data, request_x_.data(),
                                    &report_);
    return make_request();
  }

  Request tell_hessian_vector(double const *hv) {
    return tell_hessian_vector(VectorView{hv, n_});
  }

  Request tell_hessian_vector(VectorView hv) {
    if (handle_ == nullptr || !valid_view(hv)) {
      return error_request();
    }
    tide_optim_tell_hessian_vector(handle_, hv.data, request_x_.data(),
                                    &report_);
    return make_request();
  }

  Request respond(Request const &request, Evaluation const &evaluation) {
    ReportView const current = last_report_view();
    if (handle_ == nullptr || request.kind != current.request_kind() ||
        request.sequence != current.request_sequence() ||
        !evaluation.valid_for(current)) {
      return error_request();
    }
    switch (request.kind) {
    case RequestKind::EvaluateF:
      return tell_value(evaluation.f);
    case RequestKind::EvaluateFG:
      return tell(evaluation.f, evaluation.gradient);
    case RequestKind::ApplyPreconditioner:
      return tell_preconditioner(evaluation.vector);
    case RequestKind::EvaluateHv:
      return tell_hessian_vector(evaluation.vector);
    case RequestKind::Error:
    case RequestKind::Done:
      break;
    }
    return error_request();
  }

  Request tell_evaluation(Request const &request, Evaluation const &evaluation) {
    return respond(request, evaluation);
  }

  Status current_x(double *out) const {
    return current_x(MutableVectorView{out, n_});
  }

  Status current_x(MutableVectorView out) const {
    if (handle_ == nullptr || !valid_mutable_view(out)) {
      return Status::InvalidArgument;
    }
    return static_cast<Status>(tide_optim_current_x(handle_, out.data));
  }

  Request current_request() {
    if (handle_ == nullptr || !started_) {
      return error_request();
    }
    return make_request();
  }

  tide_optim_report const &last_report() const { return report_; }
  ReportView last_report_view() const { return ReportView{&report_}; }
  int64_t size() const { return n_; }

private:
  static std::vector<double> allocate_work(int64_t n) {
    return std::vector<double>(n > 0 ? static_cast<std::size_t>(n) : 0U,
                               0.0);
  }

  bool valid_view(VectorView view) const {
    return view.data != nullptr && view.size == n_;
  }

  bool valid_mutable_view(MutableVectorView view) const {
    return view.data != nullptr && view.size == n_;
  }

  Request error_request() const {
    return Request{RequestKind::Error, VectorView{}, VectorView{},
                   ReportView{&report_}};
  }

  Request make_request() {
    RequestKind const kind = static_cast<RequestKind>(report_.request);
    int64_t const sequence = last_report_view().request_sequence();
    if (kind == RequestKind::EvaluateF || kind == RequestKind::EvaluateFG) {
      return Request{kind, VectorView{request_x_.data(), n_}, VectorView{},
                     ReportView{&report_}, sequence};
    }
    if (handle_ != nullptr) {
      tide_optim_current_x(handle_, current_x_.data());
    }
    VectorView const x{current_x_.data(), n_};
    if (kind == RequestKind::ApplyPreconditioner ||
        kind == RequestKind::EvaluateHv) {
      return Request{kind, x, VectorView{request_x_.data(), n_},
                     ReportView{&report_}, sequence};
    }
    return Request{kind, x, VectorView{}, ReportView{&report_}, sequence};
  }

  void *handle_ = nullptr;
  int64_t n_ = 0;
  std::vector<double> request_x_{};
  std::vector<double> current_x_{};
  tide_optim_report report_{};
  bool started_ = false;
};

struct Bounds {
  VectorView lower{};
  VectorView upper{};

  bool enabled() const { return lower.data != nullptr || upper.data != nullptr; }
  bool valid(int64_t n) const {
    return !enabled() ||
           (lower.data != nullptr && upper.data != nullptr &&
            lower.size == n && upper.size == n);
  }
};

using ValueGradientFunction =
    std::function<double(VectorView x, MutableVectorView gradient)>;
using ValueFunction = std::function<double(VectorView x)>;
using LinearOperatorFunction =
    std::function<void(VectorView x, VectorView vector, MutableVectorView out)>;
struct CallbackDecision {
  Status status = Status::Running;

  CallbackDecision() = default;
  CallbackDecision(std::nullptr_t) : status(Status::Running) {}
  CallbackDecision(bool stop)
      : status(stop ? Status::UserStopped : Status::Running) {}
  CallbackDecision(Status status_in) : status(status_in) {}

  bool stop() const { return status != Status::Running; }
};

using TraceCallback = std::function<CallbackDecision(ReportView report)>;

struct Objective {
  ValueGradientFunction value_gradient{};
  ValueFunction value{};
  LinearOperatorFunction preconditioner{};
  LinearOperatorFunction hessian_vector{};
};

namespace detail {

template <typename ObjectiveT, typename = void>
struct has_value : std::false_type {};

template <typename ObjectiveT>
struct has_value<
    ObjectiveT,
    std::void_t<decltype(std::declval<ObjectiveT &>().value(
        std::declval<VectorView>()))>> : std::true_type {};

template <typename ObjectiveT, typename = void>
struct has_preconditioner : std::false_type {};

template <typename ObjectiveT>
struct has_preconditioner<
    ObjectiveT,
    std::void_t<decltype(std::declval<ObjectiveT &>().preconditioner(
        std::declval<VectorView>(), std::declval<VectorView>(),
        std::declval<MutableVectorView>()))>> : std::true_type {};

template <typename ObjectiveT, typename = void>
struct has_hessian_vector : std::false_type {};

template <typename ObjectiveT>
struct has_hessian_vector<
    ObjectiveT,
    std::void_t<decltype(std::declval<ObjectiveT &>().hessian_vector(
        std::declval<VectorView>(), std::declval<VectorView>(),
        std::declval<MutableVectorView>()))>> : std::true_type {};

} // namespace detail

enum class TracePolicy : int32_t {
  All = 0,
  None = 1,
  Last = 2,
  Stride = 3,
};

inline char const *name(TracePolicy value) {
  switch (value) {
  case TracePolicy::All:
    return "ALL";
  case TracePolicy::None:
    return "NONE";
  case TracePolicy::Last:
    return "LAST";
  case TracePolicy::Stride:
    return "STRIDE";
  }
  return "UNKNOWN";
}

struct TraceOptions {
  TracePolicy policy = TracePolicy::All;
  int64_t stride = 1;

  OptionsValidation validate() const {
    switch (policy) {
    case TracePolicy::All:
    case TracePolicy::None:
    case TracePolicy::Last:
    case TracePolicy::Stride:
      break;
    default:
      return OptionsValidation::invalid(
          OptionsValidationCode::TracePolicy, "trace_policy",
          "trace_policy must be ALL, NONE, LAST, or STRIDE.");
    }
    if (stride <= 0) {
      return OptionsValidation::invalid(
          OptionsValidationCode::TraceStride, "trace_stride",
          "trace_stride must be a positive integer.");
    }
    return OptionsValidation::valid();
  }

  bool valid() const { return validate().ok(); }
};

struct TraceSummary {
  int64_t n_reports = 0;
  int32_t warning_flags_seen = 0;
  EventCounts last_event_counts{};

  int64_t request_error = 0;
  int64_t request_evaluate_fg = 0;
  int64_t request_done = 0;
  int64_t request_evaluate_f = 0;
  int64_t request_apply_preconditioner = 0;
  int64_t request_evaluate_hv = 0;

  int64_t expected_gradient_requests = 0;
  int64_t expected_vector_requests = 0;
  int64_t expected_gradient_elements = 0;
  int64_t expected_vector_elements = 0;

  int64_t status_running = 0;
  int64_t status_converged_gradient = 0;
  int64_t status_converged_ftol = 0;
  int64_t status_max_iter = 0;
  int64_t status_line_search_failed = 0;
  int64_t status_nonfinite = 0;
  int64_t status_non_descent_direction = 0;
  int64_t status_invalid_argument = 0;
  int64_t status_user_stopped = 0;
  int64_t status_converged_xtol = 0;
  int64_t status_max_eval = 0;
  int64_t status_inner_cg_failed = 0;
  int64_t status_trust_region_failed = 0;

  int64_t failure_reason_null = 0;
  int64_t failure_reason_max_iter = 0;
  int64_t failure_reason_max_eval = 0;
  int64_t failure_reason_line_search_failed = 0;
  int64_t failure_reason_line_search_failed_maxls = 0;
  int64_t failure_reason_line_search_failed_nonfinite = 0;
  int64_t failure_reason_line_search_failed_alpha_bounds = 0;
  int64_t failure_reason_nonfinite = 0;
  int64_t failure_reason_nonfinite_trial = 0;
  int64_t failure_reason_non_descent_direction = 0;
  int64_t failure_reason_invalid_argument = 0;
  int64_t failure_reason_user_stopped = 0;
  int64_t failure_reason_inner_cg_failed = 0;
  int64_t failure_reason_inner_cg_nonfinite_hvp = 0;
  int64_t failure_reason_inner_cg_preconditioner_breakdown = 0;
  int64_t failure_reason_inner_cg_max_iter = 0;
  int64_t failure_reason_inner_cg_negative_curvature = 0;
  int64_t failure_reason_inner_cg_zero_curvature = 0;
  int64_t failure_reason_inner_cg_trust_boundary = 0;
  int64_t failure_reason_trust_region_failed = 0;
  int64_t failure_reason_trust_region_failed_nonfinite = 0;
  int64_t failure_reason_trust_region_failed_predicted_reduction = 0;
  int64_t failure_reason_trust_region_rejected = 0;
  int64_t failure_reason_other = 0;

  int64_t line_search_started = 0;
  int64_t line_search_accepted = 0;
  int64_t line_search_rejected_armijo = 0;
  int64_t line_search_rejected_curvature = 0;
  int64_t line_search_accepted_decrease_after_maxls = 0;
  int64_t line_search_failed_maxls = 0;
  int64_t line_search_failed_nonfinite = 0;
  int64_t line_search_failed_alpha_bounds = 0;

  int64_t line_search_acceptance_none = 0;
  int64_t line_search_acceptance_armijo = 0;
  int64_t line_search_acceptance_weak_wolfe = 0;
  int64_t line_search_acceptance_strong_wolfe = 0;
  int64_t line_search_acceptance_approximate_wolfe = 0;
  int64_t line_search_acceptance_decrease_fallback = 0;
  int64_t line_search_acceptance_static = 0;

  int64_t pair_none = 0;
  int64_t pair_stored = 0;
  int64_t pair_skipped_bad_curvature = 0;
  int64_t pair_skipped_nonfinite = 0;
  int64_t pair_skipped_line_search_fallback = 0;
  int64_t pair_skipped_bounds_projection = 0;
  int64_t pair_skipped_preconditioner = 0;
  int64_t pair_regularized_stored = 0;

  int64_t direction_initial = 0;
  int64_t direction_update = 0;
  int64_t direction_restart_denominator = 0;
  int64_t direction_restart_nonfinite = 0;
  int64_t direction_restart_non_descent = 0;

  int64_t preconditioner_none = 0;
  int64_t preconditioner_applied = 0;
  int64_t preconditioner_skipped_nonfinite = 0;
  int64_t preconditioner_skipped_not_positive = 0;

  int64_t inner_none = 0;
  int64_t inner_started = 0;
  int64_t inner_forcing_reached = 0;
  int64_t inner_negative_curvature = 0;
  int64_t inner_zero_curvature = 0;
  int64_t inner_max_iter = 0;
  int64_t inner_nonfinite_hvp = 0;
  int64_t inner_preconditioner_breakdown = 0;
  int64_t inner_trust_boundary = 0;

  int64_t trust_region_none = 0;
  int64_t trust_region_started = 0;
  int64_t trust_region_accepted = 0;
  int64_t trust_region_rejected = 0;
  int64_t trust_region_failed_nonfinite = 0;
  int64_t trust_region_failed_predicted_reduction = 0;

  void record(ReportView report) {
    if (!report) {
      return;
    }
    n_reports += 1;
    warning_flags_seen |= report.warning_flags();
    last_event_counts = report.event_counts();

    switch (report.request_kind()) {
    case RequestKind::Error:
      request_error += 1;
      break;
    case RequestKind::EvaluateFG:
      request_evaluate_fg += 1;
      break;
    case RequestKind::Done:
      request_done += 1;
      break;
    case RequestKind::EvaluateF:
      request_evaluate_f += 1;
      break;
    case RequestKind::ApplyPreconditioner:
      request_apply_preconditioner += 1;
      break;
    case RequestKind::EvaluateHv:
      request_evaluate_hv += 1;
      break;
    }

    if (report.needs_gradient()) {
      expected_gradient_requests += 1;
      int64_t const size = report.expected_gradient_size();
      if (size > 0) {
        expected_gradient_elements += size;
      }
    }
    if (report.needs_vector_result()) {
      expected_vector_requests += 1;
      int64_t const size = report.expected_vector_size();
      if (size > 0) {
        expected_vector_elements += size;
      }
    }

    switch (report.status()) {
    case Status::Running:
      status_running += 1;
      break;
    case Status::ConvergedGradient:
      status_converged_gradient += 1;
      break;
    case Status::ConvergedFtol:
      status_converged_ftol += 1;
      break;
    case Status::MaxIter:
      status_max_iter += 1;
      break;
    case Status::LineSearchFailed:
      status_line_search_failed += 1;
      break;
    case Status::Nonfinite:
      status_nonfinite += 1;
      break;
    case Status::NonDescentDirection:
      status_non_descent_direction += 1;
      break;
    case Status::InvalidArgument:
      status_invalid_argument += 1;
      break;
    case Status::UserStopped:
      status_user_stopped += 1;
      break;
    case Status::ConvergedXtol:
      status_converged_xtol += 1;
      break;
    case Status::MaxEval:
      status_max_eval += 1;
      break;
    case Status::InnerCgFailed:
      status_inner_cg_failed += 1;
      break;
    case Status::TrustRegionFailed:
      status_trust_region_failed += 1;
      break;
    }
    record_failure_reason(report.failure_reason());

    switch (report.line_search_status()) {
    case LineSearchStatus::Started:
      line_search_started += 1;
      break;
    case LineSearchStatus::Accepted:
      line_search_accepted += 1;
      break;
    case LineSearchStatus::RejectedArmijo:
      line_search_rejected_armijo += 1;
      break;
    case LineSearchStatus::RejectedCurvature:
      line_search_rejected_curvature += 1;
      break;
    case LineSearchStatus::AcceptedDecreaseAfterMaxLineSearch:
      line_search_accepted_decrease_after_maxls += 1;
      break;
    case LineSearchStatus::FailedMaxLineSearch:
      line_search_failed_maxls += 1;
      break;
    case LineSearchStatus::FailedNonfinite:
      line_search_failed_nonfinite += 1;
      break;
    case LineSearchStatus::FailedAlphaBounds:
      line_search_failed_alpha_bounds += 1;
      break;
    }

    switch (report.line_search_acceptance()) {
    case LineSearchAcceptance::None:
      line_search_acceptance_none += 1;
      break;
    case LineSearchAcceptance::Armijo:
      line_search_acceptance_armijo += 1;
      break;
    case LineSearchAcceptance::WeakWolfe:
      line_search_acceptance_weak_wolfe += 1;
      break;
    case LineSearchAcceptance::StrongWolfe:
      line_search_acceptance_strong_wolfe += 1;
      break;
    case LineSearchAcceptance::ApproximateWolfe:
      line_search_acceptance_approximate_wolfe += 1;
      break;
    case LineSearchAcceptance::DecreaseFallback:
      line_search_acceptance_decrease_fallback += 1;
      break;
    case LineSearchAcceptance::Static:
      line_search_acceptance_static += 1;
      break;
    }

    switch (report.pair_status()) {
    case PairStatus::None:
      pair_none += 1;
      break;
    case PairStatus::Stored:
      pair_stored += 1;
      break;
    case PairStatus::SkippedBadCurvature:
      pair_skipped_bad_curvature += 1;
      break;
    case PairStatus::SkippedNonfinite:
      pair_skipped_nonfinite += 1;
      break;
    case PairStatus::SkippedLineSearchFallback:
      pair_skipped_line_search_fallback += 1;
      break;
    case PairStatus::SkippedBoundsProjection:
      pair_skipped_bounds_projection += 1;
      break;
    case PairStatus::SkippedPreconditioner:
      pair_skipped_preconditioner += 1;
      break;
    case PairStatus::RegularizedStored:
      pair_regularized_stored += 1;
      break;
    }

    switch (report.direction_status()) {
    case DirectionStatus::Initial:
      direction_initial += 1;
      break;
    case DirectionStatus::Update:
      direction_update += 1;
      break;
    case DirectionStatus::RestartDenominator:
      direction_restart_denominator += 1;
      break;
    case DirectionStatus::RestartNonfinite:
      direction_restart_nonfinite += 1;
      break;
    case DirectionStatus::RestartNonDescent:
      direction_restart_non_descent += 1;
      break;
    }

    switch (report.preconditioner_status()) {
    case PreconditionerStatus::None:
      preconditioner_none += 1;
      break;
    case PreconditionerStatus::Applied:
      preconditioner_applied += 1;
      break;
    case PreconditionerStatus::SkippedNonfinite:
      preconditioner_skipped_nonfinite += 1;
      break;
    case PreconditionerStatus::SkippedNotPositive:
      preconditioner_skipped_not_positive += 1;
      break;
    }

    switch (report.inner_status()) {
    case InnerCgStatus::None:
      inner_none += 1;
      break;
    case InnerCgStatus::Started:
      inner_started += 1;
      break;
    case InnerCgStatus::ForcingReached:
      inner_forcing_reached += 1;
      break;
    case InnerCgStatus::NegativeCurvature:
      inner_negative_curvature += 1;
      break;
    case InnerCgStatus::ZeroCurvature:
      inner_zero_curvature += 1;
      break;
    case InnerCgStatus::MaxIter:
      inner_max_iter += 1;
      break;
    case InnerCgStatus::NonfiniteHvp:
      inner_nonfinite_hvp += 1;
      break;
    case InnerCgStatus::PreconditionerBreakdown:
      inner_preconditioner_breakdown += 1;
      break;
    case InnerCgStatus::TrustBoundary:
      inner_trust_boundary += 1;
      break;
    }

    switch (report.trust_region_status()) {
    case TrustRegionStatus::None:
      trust_region_none += 1;
      break;
    case TrustRegionStatus::Started:
      trust_region_started += 1;
      break;
    case TrustRegionStatus::Accepted:
      trust_region_accepted += 1;
      break;
    case TrustRegionStatus::Rejected:
      trust_region_rejected += 1;
      break;
    case TrustRegionStatus::FailedNonfinite:
      trust_region_failed_nonfinite += 1;
      break;
    case TrustRegionStatus::FailedPredictedReduction:
      trust_region_failed_predicted_reduction += 1;
      break;
    }
  }

  EventCounts event_counts() const { return last_event_counts; }

  bool has_warning(WarningFlag warning) const {
    return tide::optim::has_warning(warning_flags_seen, warning);
  }

  std::vector<char const *> warning_names() const {
    return tide::optim::warning_names(warning_flags_seen);
  }

  int64_t request_count(RequestKind kind) const {
    switch (kind) {
    case RequestKind::Error:
      return request_error;
    case RequestKind::EvaluateFG:
      return request_evaluate_fg;
    case RequestKind::Done:
      return request_done;
    case RequestKind::EvaluateF:
      return request_evaluate_f;
    case RequestKind::ApplyPreconditioner:
      return request_apply_preconditioner;
    case RequestKind::EvaluateHv:
      return request_evaluate_hv;
    }
    return 0;
  }

  int64_t status_count(Status status) const {
    switch (status) {
    case Status::Running:
      return status_running;
    case Status::ConvergedGradient:
      return status_converged_gradient;
    case Status::ConvergedFtol:
      return status_converged_ftol;
    case Status::MaxIter:
      return status_max_iter;
    case Status::LineSearchFailed:
      return status_line_search_failed;
    case Status::Nonfinite:
      return status_nonfinite;
    case Status::NonDescentDirection:
      return status_non_descent_direction;
    case Status::InvalidArgument:
      return status_invalid_argument;
    case Status::UserStopped:
      return status_user_stopped;
    case Status::ConvergedXtol:
      return status_converged_xtol;
    case Status::MaxEval:
      return status_max_eval;
    case Status::InnerCgFailed:
      return status_inner_cg_failed;
    case Status::TrustRegionFailed:
      return status_trust_region_failed;
    }
    return 0;
  }

  int64_t success_count() const {
    return status_converged_gradient + status_converged_ftol +
           status_converged_xtol;
  }

  int64_t failed_count() const {
    return status_max_iter + status_line_search_failed + status_nonfinite +
           status_non_descent_direction + status_invalid_argument +
           status_max_eval + status_inner_cg_failed +
           status_trust_region_failed;
  }

  int64_t user_stopped_count() const { return status_user_stopped; }

  int64_t expected_total_vector_elements() const {
    return expected_gradient_elements + expected_vector_elements;
  }

  void adjust_status(Status status, int64_t delta) {
    switch (status) {
    case Status::Running:
      status_running += delta;
      break;
    case Status::ConvergedGradient:
      status_converged_gradient += delta;
      break;
    case Status::ConvergedFtol:
      status_converged_ftol += delta;
      break;
    case Status::MaxIter:
      status_max_iter += delta;
      break;
    case Status::LineSearchFailed:
      status_line_search_failed += delta;
      break;
    case Status::Nonfinite:
      status_nonfinite += delta;
      break;
    case Status::NonDescentDirection:
      status_non_descent_direction += delta;
      break;
    case Status::InvalidArgument:
      status_invalid_argument += delta;
      break;
    case Status::UserStopped:
      status_user_stopped += delta;
      break;
    case Status::ConvergedXtol:
      status_converged_xtol += delta;
      break;
    case Status::MaxEval:
      status_max_eval += delta;
      break;
    case Status::InnerCgFailed:
      status_inner_cg_failed += delta;
      break;
    case Status::TrustRegionFailed:
      status_trust_region_failed += delta;
      break;
    }
  }

  void adjust_failure_reason(char const *reason, int64_t delta) {
    if (reason == nullptr) {
      failure_reason_null += delta;
      return;
    }
    std::string const key(reason);
    if (key == "MAX_ITER") {
      failure_reason_max_iter += delta;
    } else if (key == "MAX_EVAL") {
      failure_reason_max_eval += delta;
    } else if (key == "LINE_SEARCH_FAILED") {
      failure_reason_line_search_failed += delta;
    } else if (key == "LINE_SEARCH_FAILED_MAXLS") {
      failure_reason_line_search_failed_maxls += delta;
    } else if (key == "LINE_SEARCH_FAILED_NONFINITE") {
      failure_reason_line_search_failed_nonfinite += delta;
    } else if (key == "LINE_SEARCH_FAILED_ALPHA_BOUNDS") {
      failure_reason_line_search_failed_alpha_bounds += delta;
    } else if (key == "NONFINITE") {
      failure_reason_nonfinite += delta;
    } else if (key == "NONFINITE_TRIAL") {
      failure_reason_nonfinite_trial += delta;
    } else if (key == "NON_DESCENT_DIRECTION") {
      failure_reason_non_descent_direction += delta;
    } else if (key == "INVALID_ARGUMENT") {
      failure_reason_invalid_argument += delta;
    } else if (key == "USER_STOPPED") {
      failure_reason_user_stopped += delta;
    } else if (key == "INNER_CG_FAILED") {
      failure_reason_inner_cg_failed += delta;
    } else if (key == "INNER_CG_NONFINITE_HVP") {
      failure_reason_inner_cg_nonfinite_hvp += delta;
    } else if (key == "INNER_CG_PRECONDITIONER_BREAKDOWN") {
      failure_reason_inner_cg_preconditioner_breakdown += delta;
    } else if (key == "INNER_CG_MAX_ITER") {
      failure_reason_inner_cg_max_iter += delta;
    } else if (key == "INNER_CG_NEGATIVE_CURVATURE") {
      failure_reason_inner_cg_negative_curvature += delta;
    } else if (key == "INNER_CG_ZERO_CURVATURE") {
      failure_reason_inner_cg_zero_curvature += delta;
    } else if (key == "INNER_CG_TRUST_BOUNDARY") {
      failure_reason_inner_cg_trust_boundary += delta;
    } else if (key == "TRUST_REGION_FAILED") {
      failure_reason_trust_region_failed += delta;
    } else if (key == "TRUST_REGION_FAILED_NONFINITE") {
      failure_reason_trust_region_failed_nonfinite += delta;
    } else if (key == "TRUST_REGION_FAILED_PREDICTED_REDUCTION") {
      failure_reason_trust_region_failed_predicted_reduction += delta;
    } else if (key == "TRUST_REGION_REJECTED") {
      failure_reason_trust_region_rejected += delta;
    } else {
      failure_reason_other += delta;
    }
  }

  void record_failure_reason(char const *reason) {
    adjust_failure_reason(reason, 1);
  }

  void finalize_stopping(ReportView old_report, ReportView new_report) {
    if (!old_report || !new_report) {
      return;
    }
    adjust_status(old_report.status(), -1);
    adjust_status(new_report.status(), 1);
    adjust_failure_reason(old_report.failure_reason(), -1);
    adjust_failure_reason(new_report.failure_reason(), 1);
    last_event_counts = new_report.event_counts();
  }

  int64_t failure_reason_count(char const *reason) const {
    if (reason == nullptr || std::string(reason) == "null") {
      return failure_reason_null;
    }
    std::string const key(reason);
    if (key == "MAX_ITER") return failure_reason_max_iter;
    if (key == "MAX_EVAL") return failure_reason_max_eval;
    if (key == "LINE_SEARCH_FAILED") return failure_reason_line_search_failed;
    if (key == "LINE_SEARCH_FAILED_MAXLS") return failure_reason_line_search_failed_maxls;
    if (key == "LINE_SEARCH_FAILED_NONFINITE") return failure_reason_line_search_failed_nonfinite;
    if (key == "LINE_SEARCH_FAILED_ALPHA_BOUNDS") return failure_reason_line_search_failed_alpha_bounds;
    if (key == "NONFINITE") return failure_reason_nonfinite;
    if (key == "NONFINITE_TRIAL") return failure_reason_nonfinite_trial;
    if (key == "NON_DESCENT_DIRECTION") return failure_reason_non_descent_direction;
    if (key == "INVALID_ARGUMENT") return failure_reason_invalid_argument;
    if (key == "USER_STOPPED") return failure_reason_user_stopped;
    if (key == "INNER_CG_FAILED") return failure_reason_inner_cg_failed;
    if (key == "INNER_CG_NONFINITE_HVP") return failure_reason_inner_cg_nonfinite_hvp;
    if (key == "INNER_CG_PRECONDITIONER_BREAKDOWN") return failure_reason_inner_cg_preconditioner_breakdown;
    if (key == "INNER_CG_MAX_ITER") return failure_reason_inner_cg_max_iter;
    if (key == "INNER_CG_NEGATIVE_CURVATURE") return failure_reason_inner_cg_negative_curvature;
    if (key == "INNER_CG_ZERO_CURVATURE") return failure_reason_inner_cg_zero_curvature;
    if (key == "INNER_CG_TRUST_BOUNDARY") return failure_reason_inner_cg_trust_boundary;
    if (key == "TRUST_REGION_FAILED") return failure_reason_trust_region_failed;
    if (key == "TRUST_REGION_FAILED_NONFINITE") return failure_reason_trust_region_failed_nonfinite;
    if (key == "TRUST_REGION_FAILED_PREDICTED_REDUCTION") return failure_reason_trust_region_failed_predicted_reduction;
    if (key == "TRUST_REGION_REJECTED") return failure_reason_trust_region_rejected;
    if (key == "other") return failure_reason_other;
    return 0;
  }

  int64_t line_search_status_count(LineSearchStatus status) const {
    switch (status) {
    case LineSearchStatus::Started:
      return line_search_started;
    case LineSearchStatus::Accepted:
      return line_search_accepted;
    case LineSearchStatus::RejectedArmijo:
      return line_search_rejected_armijo;
    case LineSearchStatus::RejectedCurvature:
      return line_search_rejected_curvature;
    case LineSearchStatus::AcceptedDecreaseAfterMaxLineSearch:
      return line_search_accepted_decrease_after_maxls;
    case LineSearchStatus::FailedMaxLineSearch:
      return line_search_failed_maxls;
    case LineSearchStatus::FailedNonfinite:
      return line_search_failed_nonfinite;
    case LineSearchStatus::FailedAlphaBounds:
      return line_search_failed_alpha_bounds;
    }
    return 0;
  }

  int64_t line_search_acceptance_count(LineSearchAcceptance acceptance) const {
    switch (acceptance) {
    case LineSearchAcceptance::None:
      return line_search_acceptance_none;
    case LineSearchAcceptance::Armijo:
      return line_search_acceptance_armijo;
    case LineSearchAcceptance::WeakWolfe:
      return line_search_acceptance_weak_wolfe;
    case LineSearchAcceptance::StrongWolfe:
      return line_search_acceptance_strong_wolfe;
    case LineSearchAcceptance::ApproximateWolfe:
      return line_search_acceptance_approximate_wolfe;
    case LineSearchAcceptance::DecreaseFallback:
      return line_search_acceptance_decrease_fallback;
    case LineSearchAcceptance::Static:
      return line_search_acceptance_static;
    }
    return 0;
  }

  int64_t pair_status_count(PairStatus status) const {
    switch (status) {
    case PairStatus::None:
      return pair_none;
    case PairStatus::Stored:
      return pair_stored;
    case PairStatus::SkippedBadCurvature:
      return pair_skipped_bad_curvature;
    case PairStatus::SkippedNonfinite:
      return pair_skipped_nonfinite;
    case PairStatus::SkippedLineSearchFallback:
      return pair_skipped_line_search_fallback;
    case PairStatus::SkippedBoundsProjection:
      return pair_skipped_bounds_projection;
    case PairStatus::SkippedPreconditioner:
      return pair_skipped_preconditioner;
    case PairStatus::RegularizedStored:
      return pair_regularized_stored;
    }
    return 0;
  }

  int64_t direction_status_count(DirectionStatus status) const {
    switch (status) {
    case DirectionStatus::Initial:
      return direction_initial;
    case DirectionStatus::Update:
      return direction_update;
    case DirectionStatus::RestartDenominator:
      return direction_restart_denominator;
    case DirectionStatus::RestartNonfinite:
      return direction_restart_nonfinite;
    case DirectionStatus::RestartNonDescent:
      return direction_restart_non_descent;
    }
    return 0;
  }

  int64_t preconditioner_status_count(PreconditionerStatus status) const {
    switch (status) {
    case PreconditionerStatus::None:
      return preconditioner_none;
    case PreconditionerStatus::Applied:
      return preconditioner_applied;
    case PreconditionerStatus::SkippedNonfinite:
      return preconditioner_skipped_nonfinite;
    case PreconditionerStatus::SkippedNotPositive:
      return preconditioner_skipped_not_positive;
    }
    return 0;
  }

  int64_t inner_status_count(InnerCgStatus status) const {
    switch (status) {
    case InnerCgStatus::None:
      return inner_none;
    case InnerCgStatus::Started:
      return inner_started;
    case InnerCgStatus::ForcingReached:
      return inner_forcing_reached;
    case InnerCgStatus::NegativeCurvature:
      return inner_negative_curvature;
    case InnerCgStatus::ZeroCurvature:
      return inner_zero_curvature;
    case InnerCgStatus::MaxIter:
      return inner_max_iter;
    case InnerCgStatus::NonfiniteHvp:
      return inner_nonfinite_hvp;
    case InnerCgStatus::PreconditionerBreakdown:
      return inner_preconditioner_breakdown;
    case InnerCgStatus::TrustBoundary:
      return inner_trust_boundary;
    }
    return 0;
  }

  int64_t trust_region_status_count(TrustRegionStatus status) const {
    switch (status) {
    case TrustRegionStatus::None:
      return trust_region_none;
    case TrustRegionStatus::Started:
      return trust_region_started;
    case TrustRegionStatus::Accepted:
      return trust_region_accepted;
    case TrustRegionStatus::Rejected:
      return trust_region_rejected;
    case TrustRegionStatus::FailedNonfinite:
      return trust_region_failed_nonfinite;
    case TrustRegionStatus::FailedPredictedReduction:
      return trust_region_failed_predicted_reduction;
    }
    return 0;
  }
};

struct TraceRecorder {
  explicit TraceRecorder(TraceOptions options_in = {})
      : options(options_in), stride(options_in.stride > 0 ? options_in.stride
                                                          : 1) {}

  void record(ReportView report) {
    if (!report) {
      return;
    }
    n_seen += 1;
    last = *report.raw();
    has_last = true;
    summary.record(report);

    switch (options.policy) {
    case TracePolicy::None:
      return;
    case TracePolicy::All:
      entries.push_back(last);
      return;
    case TracePolicy::Last:
      if (entries.empty()) {
        entries.push_back(last);
      } else {
        entries[0] = last;
      }
      return;
    case TracePolicy::Stride:
      if (n_seen == 1 || n_seen % stride == 0 || report.done()) {
        entries.push_back(last);
      }
      return;
    }
  }

  void finalize(Status status) {
    if (!has_last) {
      return;
    }
    tide_optim_report const old = last;
    last.status = static_cast<int32_t>(status);
    summary.finalize_stopping(ReportView{&old}, ReportView{&last});

    switch (options.policy) {
    case TracePolicy::None:
      return;
    case TracePolicy::All:
      if (!entries.empty()) {
        entries.back() = last;
      }
      return;
    case TracePolicy::Last:
      if (entries.empty()) {
        entries.push_back(last);
      } else {
        entries[0] = last;
      }
      return;
    case TracePolicy::Stride:
      if (!entries.empty() && same_report_event(entries.back(), old)) {
        entries.back() = last;
      } else {
        entries.push_back(last);
      }
      return;
    }
  }

  static bool same_report_event(tide_optim_report const &a,
                                tide_optim_report const &b) {
    return a.request_sequence == b.request_sequence &&
           a.request == b.request && a.iter == b.iter &&
           a.n_f == b.n_f && a.n_g == b.n_g &&
           a.n_hvp == b.n_hvp && a.n_prec == b.n_prec;
  }

  TraceOptions options{};
  int64_t stride = 1;
  int64_t n_seen = 0;
  TraceSummary summary{};
  std::vector<tide_optim_report> entries{};
  tide_optim_report last{};
  bool has_last = false;
};

struct EvaluationProfile {
  CostModel cost_model = CostModel::Balanced;
  int64_t n_f = 0;
  int64_t n_g = 0;
  int64_t n_hvp = 0;
  int64_t n_prec = 0;

  char const *cost_model_name() const { return name(cost_model); }
  int64_t n_value_gradient() const { return n_g; }
  int64_t n_value_only() const { return n_f > n_g ? n_f - n_g : 0; }
  int64_t n_objective() const { return n_f; }
  int64_t n_gradient_required() const { return n_value_gradient() + n_hvp; }
  int64_t n_vector_operator() const { return n_hvp + n_prec; }
  int64_t n_total_requests() const {
    return n_value_only() + n_value_gradient() + n_hvp + n_prec;
  }
};


struct EvaluationCostWeights {
  double value_only = 1.0;
  double value_gradient = 2.0;
  double hessian_vector = 2.0;
  double preconditioner = 0.25;
};

inline EvaluationCostWeights cost_weights(CostModel cost_model) {
  switch (cost_model) {
  case CostModel::Balanced:
    return EvaluationCostWeights{1.0, 2.0, 2.0, 0.25};
  case CostModel::ExpensiveGradient:
    return EvaluationCostWeights{1.0, 4.0, 4.0, 0.25};
  case CostModel::JointValueGradient:
    return EvaluationCostWeights{1.0, 1.0, 2.0, 0.25};
  }
  return EvaluationCostWeights{};
}

struct EvaluationCostEstimate {
  CostModel cost_model = CostModel::Balanced;
  int64_t n = 0;
  int64_t n_value_only = 0;
  int64_t n_value_gradient = 0;
  int64_t n_hvp = 0;
  int64_t n_prec = 0;
  EvaluationCostWeights weights{};
  int64_t expected_gradient_elements = 0;
  int64_t expected_vector_elements = 0;

  char const *cost_model_name() const { return name(cost_model); }
  double weighted_request_units() const {
    return static_cast<double>(n_value_only) * weights.value_only +
           static_cast<double>(n_value_gradient) * weights.value_gradient +
           static_cast<double>(n_hvp) * weights.hessian_vector +
           static_cast<double>(n_prec) * weights.preconditioner;
  }
  int64_t expected_total_vector_elements() const {
    return expected_gradient_elements + expected_vector_elements;
  }
  double expected_vector_passes() const {
    if (n <= 0) {
      return 0.0;
    }
    return static_cast<double>(expected_total_vector_elements()) /
           static_cast<double>(n);
  }
};

struct Result {
  Status status = Status::InvalidArgument;
  bool success = false;
  std::vector<double> x{};
  tide_optim_report report{};
  TracePolicy trace_policy = TracePolicy::All;
  int64_t trace_stride = 1;
  int64_t n_trace_events = 0;
  TraceSummary trace_summary{};
  ResolvedPolicies resolved_policies{};
  std::string config_signature{};
  std::string config_fingerprint{};
  std::vector<tide_optim_report> trace{};
  tide_optim_report last_trace{};
  bool has_last_trace = false;

  ReportView report_view() const { return ReportView{&report}; }
  EventCounts event_counts() const { return report_view().event_counts(); }
  ReportView last_trace_view() const {
    return has_last_trace ? ReportView{&last_trace} : ReportView{};
  }
  ReportView trace_view(std::size_t index) const {
    return index < trace.size() ? ReportView{&trace[index]} : ReportView{};
  }
  int64_t n_iter() const { return report.iter; }
  int64_t n_f() const { return report.n_f; }
  int64_t n_g() const { return report.n_g; }
  int64_t n_hvp() const { return report.n_hvp; }
  int64_t n_prec() const { return report.n_prec; }
  int64_t n_trace_stored() const {
    return static_cast<int64_t>(trace.size());
  }
  double f() const { return report.f; }
  double grad_norm() const { return report.grad_norm; }
  char const *status_name() const { return name(status); }
  char const *reason() const { return tide::optim::reason(status); }
  char const *failure_reason() const {
    ReportView const view = report_view();
    return tide::optim::failure_reason(status, view.line_search_status(),
                                       view.inner_status(),
                                       view.trust_region_status());
  }
  char const *trace_policy_name() const { return name(trace_policy); }
  char const *method_name() const { return resolved_policies.method_name(); }
  char const *line_search_policy_name() const {
    return resolved_policies.line_search_name();
  }
  char const *globalization_policy_name() const {
    return resolved_policies.globalization_name();
  }
  char const *bounds_strategy_name() const {
    return resolved_policies.bounds_name();
  }
  char const *cost_model_name() const { return resolved_policies.cost_model_name(); }
  double projected_grad_norm() const { return report.projected_grad_norm; }
  int64_t active_lower_count() const { return report.active_lower_count; }
  int64_t active_upper_count() const { return report.active_upper_count; }
  int64_t free_count() const { return report.free_count; }
  int64_t kkt_violation_count() const { return report.kkt_violation_count; }
  int64_t lower_kkt_violation_count() const {
    return report.lower_kkt_violation_count;
  }
  int64_t upper_kkt_violation_count() const {
    return report.upper_kkt_violation_count;
  }
  int64_t free_gradient_count() const { return report.free_gradient_count; }
  int64_t trial_projection_count() const {
    return report.trial_projection_count;
  }
  int64_t trial_lower_projection_count() const {
    return report.trial_lower_projection_count;
  }
  int64_t trial_upper_projection_count() const {
    return report.trial_upper_projection_count;
  }
  StoppingDiagnostics stopping_diagnostics() const {
    return report_view().stopping_diagnostics();
  }
  DirectionDiagnostics direction_diagnostics() const {
    return report_view().direction_diagnostics();
  }
  BoundsDiagnostics bounds_diagnostics() const {
    return report_view().bounds_diagnostics(resolved_policies.bounds);
  }
  LineSearchDiagnostics line_search_diagnostics() const {
    return report_view().line_search_diagnostics();
  }
  PairUpdateDiagnostics pair_update_diagnostics() const {
    return report_view().pair_update_diagnostics();
  }
  InnerSolveDiagnostics inner_solve_diagnostics() const {
    return report_view().inner_solve_diagnostics();
  }
  TrustRegionDiagnostics trust_region_diagnostics() const {
    return report_view().trust_region_diagnostics();
  }
  EvaluationProfile evaluation_profile() const {
    return EvaluationProfile{resolved_policies.cost_model, n_f(), n_g(),
                             n_hvp(), n_prec()};
  }
  EvaluationCostEstimate evaluation_cost_estimate() const {
    EvaluationProfile const profile = evaluation_profile();
    return EvaluationCostEstimate{
        profile.cost_model,
        report.n,
        profile.n_value_only(),
        profile.n_value_gradient(),
        profile.n_hvp,
        profile.n_prec,
        cost_weights(profile.cost_model),
        trace_summary.expected_gradient_elements,
        trace_summary.expected_vector_elements,
    };
  }
};

inline bool successful_status(Status status) {
  return status == Status::ConvergedGradient ||
         status == Status::ConvergedFtol || status == Status::ConvergedXtol;
}

inline void apply_trace(Result &result, TraceRecorder const *trace) {
  if (trace == nullptr) {
    return;
  }
  result.trace_policy = trace->options.policy;
  result.trace_stride = trace->stride;
  result.n_trace_events = trace->n_seen;
  result.trace_summary = trace->summary;
  result.trace = trace->entries;
  result.has_last_trace = trace->has_last;
  if (trace->has_last) {
    result.last_trace = trace->last;
  }
}

inline void apply_config_identity(Result &result, Options const *options) {
  if (options == nullptr) {
    return;
  }
  result.config_signature =
      make_config_signature(*options, result.resolved_policies);
  result.config_fingerprint = make_config_fingerprint(result.config_signature);
}

inline Result result_from_session(Session &session, Status status,
                                  TraceRecorder const *trace = nullptr,
                                  Options const *options = nullptr) {
  Result result{};
  result.status = status;
  result.success = successful_status(status);
  result.report = session.last_report();
  result.report.status = static_cast<int32_t>(status);
  result.resolved_policies =
      ResolvedPolicies::from_report(result.report_view(), options);
  apply_config_identity(result, options);
  result.x.assign(static_cast<std::size_t>(session.size()), 0.0);
  session.current_x(view(result.x));
  apply_trace(result, trace);
  return result;
}

inline Result finalized_result_from_session(Session &session, Status status,
                                            TraceRecorder *trace,
                                            Options const *options = nullptr) {
  if (trace != nullptr) {
    ReportView const report = session.last_report_view();
    if (report && report.status() != status) {
      trace->finalize(status);
    }
  }
  return result_from_session(session, status, trace, options);
}

class TelemetrySession {
public:
  explicit TelemetrySession(Options const &options,
                            TraceOptions trace_options = {})
      : session_(options), trace_(trace_options),
        trace_validation_(trace_options.validate()), options_(options),
        has_options_(true) {}

  explicit TelemetrySession(tide_optim_options const &options,
                            TraceOptions trace_options = {})
      : session_(options), trace_(trace_options),
        trace_validation_(trace_options.validate()) {}

  bool valid() const { return session_.valid() && trace_validation_.ok(); }
  bool closed() const { return !session_.valid(); }
  bool started() const { return trace_.n_seen > 0; }
  bool done() const {
    ReportView const report = last_trace_view();
    return report && report.done();
  }
  bool running() const { return valid() && started() && !done(); }
  char const *state_name() const {
    if (closed()) {
      return "CLOSED";
    }
    if (!trace_validation_.ok()) {
      return "INVALID";
    }
    if (!started()) {
      return "NOT_STARTED";
    }
    if (done()) {
      return "DONE";
    }
    return "RUNNING";
  }
  OptionsValidation trace_options_validation() const { return trace_validation_; }
  void close() { session_.close(); }
  int64_t size() const { return session_.size(); }

  Status set_bounds(double const *lb, double const *ub) {
    return valid() ? session_.set_bounds(lb, ub) : Status::InvalidArgument;
  }
  Status set_bounds(VectorView lb, VectorView ub) {
    return valid() ? session_.set_bounds(lb, ub) : Status::InvalidArgument;
  }
  Status clear_bounds() {
    return valid() ? session_.clear_bounds() : Status::InvalidArgument;
  }

  Request start(double const *x, double f, double const *g) {
    return valid() ? record(session_.start(x, f, g)) : invalid_request();
  }
  Request start(VectorView x, double f, VectorView g) {
    return valid() ? record(session_.start(x, f, g)) : invalid_request();
  }
  Request tell(double f, double const *g) {
    return valid() ? record(session_.tell(f, g)) : invalid_request();
  }
  Request tell(double f, VectorView g) {
    return valid() ? record(session_.tell(f, g)) : invalid_request();
  }
  Request tell_value(double f) {
    return valid() ? record(session_.tell_value(f)) : invalid_request();
  }
  Request tell_preconditioner(double const *z) {
    return valid() ? record(session_.tell_preconditioner(z)) : invalid_request();
  }
  Request tell_preconditioner(VectorView z) {
    return valid() ? record(session_.tell_preconditioner(z)) : invalid_request();
  }
  Request tell_hessian_vector(double const *hv) {
    return valid() ? record(session_.tell_hessian_vector(hv)) : invalid_request();
  }
  Request tell_hessian_vector(VectorView hv) {
    return valid() ? record(session_.tell_hessian_vector(hv)) : invalid_request();
  }

  Request respond(Request const &request, Evaluation const &evaluation) {
    return valid() ? record(session_.respond(request, evaluation))
                   : invalid_request();
  }

  Request tell_evaluation(Request const &request, Evaluation const &evaluation) {
    return respond(request, evaluation);
  }

  Status current_x(double *out) const {
    return valid() ? session_.current_x(out) : Status::InvalidArgument;
  }
  Status current_x(MutableVectorView out) const {
    return valid() ? session_.current_x(out) : Status::InvalidArgument;
  }

  tide_optim_report const &last_report() const {
    return session_.last_report();
  }
  ReportView last_report_view() const { return session_.last_report_view(); }

  TraceRecorder const &trace_recorder() const { return trace_; }
  TraceSummary const &trace_summary() const { return trace_.summary; }
  int64_t n_trace_events() const { return trace_.n_seen; }
  int64_t n_trace_stored() const {
    return static_cast<int64_t>(trace_.entries.size());
  }
  std::vector<tide_optim_report> const &trace() const {
    return trace_.entries;
  }
  ReportView last_trace_view() const {
    return trace_.has_last ? ReportView{&trace_.last} : ReportView{};
  }
  ReportView trace_view(std::size_t index) const {
    return index < trace_.entries.size() ? ReportView{&trace_.entries[index]}
                                         : ReportView{};
  }

  Result result() {
    ReportView const report = last_report_view();
    return result(report ? report.status() : Status::InvalidArgument);
  }
  Result result(Status status) {
    if (!trace_validation_.ok()) {
      status = Status::InvalidArgument;
    }
    ReportView const report = last_report_view();
    if (report && report.status() != status) {
      trace_.finalize(status);
    }
    return result_from_session(session_, status, &trace_,
                               has_options_ ? &options_ : nullptr);
  }

  Session &session() { return session_; }
  Session const &session() const { return session_; }

private:
  Request invalid_request() const {
    return Request{RequestKind::Error, VectorView{}, VectorView{}, ReportView{}};
  }

  Request record(Request request) {
    trace_.record(request.report);
    return request;
  }

  Session session_;
  TraceRecorder trace_;
  OptionsValidation trace_validation_{};
  Options options_{};
  bool has_options_ = false;
};

inline Result invalid_result(VectorView x0, int64_t n, Status status,
                             Options const *options = nullptr) {
  Result result{};
  result.status = status;
  result.success = false;
  result.report.request = TIDE_OPTIM_REQUEST_ERROR;
  result.report.status = static_cast<int32_t>(status);
  result.resolved_policies =
      ResolvedPolicies::from_report(result.report_view(), options);
  apply_config_identity(result, options);
  if (x0.data != nullptr && x0.size == n && n > 0) {
    result.x.assign(x0.data, x0.data + n);
  }
  return result;
}

inline Result minimize(Options const &options, VectorView x0,
                       Objective const &objective, Bounds bounds = {},
                       TraceOptions trace_options = {},
                       TraceCallback callback = {}) {
  int64_t const n = options.n;
  if (n <= 0 || x0.data == nullptr || x0.size != n ||
      !objective.value_gradient || !bounds.valid(n)) {
    return invalid_result(x0, n, Status::InvalidArgument, &options);
  }
  if (!trace_options.validate()) {
    return invalid_result(x0, n, Status::InvalidArgument, &options);
  }

  Session session(options);
  if (!session.valid()) {
    return invalid_result(x0, n, Status::InvalidArgument, &options);
  }
  if (bounds.enabled()) {
    Status const bounds_status = session.set_bounds(bounds.lower, bounds.upper);
    if (bounds_status != Status::Running) {
      return result_from_session(session, bounds_status, nullptr, &options);
    }
  }

  std::vector<double> gradient(static_cast<std::size_t>(n), 0.0);
  std::vector<double> output(static_cast<std::size_t>(n), 0.0);
  double f = objective.value_gradient(x0, view(gradient));
  Request request = session.start(x0, f, view(gradient));
  TraceRecorder trace(trace_options);

  while (true) {
    trace.record(request.report);
    if (callback) {
      CallbackDecision const decision = callback(request.report);
      if (decision.stop()) {
        return finalized_result_from_session(session, decision.status,
                                             &trace, &options);
      }
    }

    switch (request.kind) {
    case RequestKind::Done:
      return result_from_session(session, request.report.status(), &trace,
                                 &options);
    case RequestKind::EvaluateFG:
      f = objective.value_gradient(request.x, view(gradient));
      request = session.tell(f, view(gradient));
      break;
    case RequestKind::EvaluateF:
      if (objective.value) {
        f = objective.value(request.x);
      } else {
        f = objective.value_gradient(request.x, view(gradient));
      }
      request = session.tell_value(f);
      break;
    case RequestKind::ApplyPreconditioner:
      if (!objective.preconditioner) {
        return finalized_result_from_session(session, Status::InvalidArgument,
                                             &trace, &options);
      }
      objective.preconditioner(request.x, request.vector, view(output));
      request = session.tell_preconditioner(view(output));
      break;
    case RequestKind::EvaluateHv:
      if (!objective.hessian_vector) {
        return finalized_result_from_session(session, Status::InvalidArgument,
                                             &trace, &options);
      }
      objective.hessian_vector(request.x, request.vector, view(output));
      request = session.tell_hessian_vector(view(output));
      break;
    case RequestKind::Error:
    default:
      return finalized_result_from_session(session, Status::InvalidArgument,
                                           &trace, &options);
    }
  }
}

inline Result minimize(Options const &options, VectorView x0,
                       Objective const &objective, Bounds bounds,
                       TraceCallback callback) {
  return minimize(options, x0, objective, bounds, TraceOptions{},
                  std::move(callback));
}

inline Result minimize(Options const &options, std::vector<double> const &x0,
                       Objective const &objective, Bounds bounds = {},
                       TraceOptions trace_options = {},
                       TraceCallback callback = {}) {
  return minimize(options, view(x0), objective, bounds, trace_options,
                  std::move(callback));
}

inline Result minimize(Options const &options, std::vector<double> const &x0,
                       Objective const &objective, Bounds bounds,
                       TraceCallback callback) {
  return minimize(options, view(x0), objective, bounds, TraceOptions{},
                  std::move(callback));
}

template <typename ObjectiveT>
Result minimize_with(Options const &options, VectorView x0,
                     ObjectiveT &objective, Bounds bounds = {},
                     TraceOptions trace_options = {},
                     TraceCallback callback = {}) {
  int64_t const n = options.n;
  if (n <= 0 || x0.data == nullptr || x0.size != n || !bounds.valid(n)) {
    return invalid_result(x0, n, Status::InvalidArgument, &options);
  }
  if (!trace_options.validate()) {
    return invalid_result(x0, n, Status::InvalidArgument, &options);
  }

  Session session(options);
  if (!session.valid()) {
    return invalid_result(x0, n, Status::InvalidArgument, &options);
  }
  if (bounds.enabled()) {
    Status const bounds_status = session.set_bounds(bounds.lower, bounds.upper);
    if (bounds_status != Status::Running) {
      return result_from_session(session, bounds_status, nullptr, &options);
    }
  }

  std::vector<double> gradient(static_cast<std::size_t>(n), 0.0);
  std::vector<double> output(static_cast<std::size_t>(n), 0.0);
  double f = objective.value_gradient(x0, view(gradient));
  Request request = session.start(x0, f, view(gradient));
  TraceRecorder trace(trace_options);

  while (true) {
    trace.record(request.report);
    if (callback) {
      CallbackDecision const decision = callback(request.report);
      if (decision.stop()) {
        return finalized_result_from_session(session, decision.status,
                                             &trace, &options);
      }
    }

    switch (request.kind) {
    case RequestKind::Done:
      return result_from_session(session, request.report.status(), &trace,
                                 &options);
    case RequestKind::EvaluateFG:
      f = objective.value_gradient(request.x, view(gradient));
      request = session.tell(f, view(gradient));
      break;
    case RequestKind::EvaluateF:
      if constexpr (detail::has_value<ObjectiveT>::value) {
        f = objective.value(request.x);
      } else {
        f = objective.value_gradient(request.x, view(gradient));
      }
      request = session.tell_value(f);
      break;
    case RequestKind::ApplyPreconditioner:
      if constexpr (detail::has_preconditioner<ObjectiveT>::value) {
        objective.preconditioner(request.x, request.vector, view(output));
        request = session.tell_preconditioner(view(output));
      } else {
        return finalized_result_from_session(session, Status::InvalidArgument,
                                             &trace, &options);
      }
      break;
    case RequestKind::EvaluateHv:
      if constexpr (detail::has_hessian_vector<ObjectiveT>::value) {
        objective.hessian_vector(request.x, request.vector, view(output));
        request = session.tell_hessian_vector(view(output));
      } else {
        return finalized_result_from_session(session, Status::InvalidArgument,
                                             &trace, &options);
      }
      break;
    case RequestKind::Error:
    default:
      return finalized_result_from_session(session, Status::InvalidArgument,
                                           &trace, &options);
    }
  }
}

template <typename ObjectiveT>
Result minimize_with(Options const &options, std::vector<double> const &x0,
                     ObjectiveT &objective, Bounds bounds = {},
                     TraceOptions trace_options = {},
                     TraceCallback callback = {}) {
  return minimize_with(options, view(x0), objective, bounds, trace_options,
                       std::move(callback));
}

template <typename ObjectiveT>
Result minimize_with(Options const &options, VectorView x0,
                     ObjectiveT &objective, Bounds bounds,
                     TraceCallback callback) {
  return minimize_with(options, x0, objective, bounds, TraceOptions{},
                       std::move(callback));
}

template <typename ObjectiveT>
Result minimize_with(Options const &options, std::vector<double> const &x0,
                     ObjectiveT &objective, Bounds bounds,
                     TraceCallback callback) {
  return minimize_with(options, view(x0), objective, bounds, TraceOptions{},
                       std::move(callback));
}

} // namespace tide::optim
#endif

#endif
