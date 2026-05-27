#include "optim.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <vector>

namespace {

constexpr int kWarnNonfiniteTrial = TIDE_OPTIM_WARNING_NONFINITE_TRIAL;
constexpr int kWarnNonDescentReset =
    TIDE_OPTIM_WARNING_NON_DESCENT_DIRECTION_RESET;
constexpr int kWarnFallbackAccepted =
    TIDE_OPTIM_WARNING_ACCEPTED_DECREASE_AFTER_MAX_LINE_SEARCH;
constexpr int kWarnPairSkipped = TIDE_OPTIM_WARNING_LBFGS_PAIR_SKIPPED;
constexpr int kWarnPreconditionerSkipped =
    TIDE_OPTIM_WARNING_PRECONDITIONER_SKIPPED;
constexpr int kWarnInnerCg = TIDE_OPTIM_WARNING_INNER_CG;
constexpr int kWarnPairRegularized =
    TIDE_OPTIM_WARNING_LBFGS_PAIR_REGULARIZED;

struct LbfgsState {
  tide_optim_lbfgs_options opt{};
  std::vector<double> x;
  std::vector<double> g;
  std::vector<double> base_x;
  std::vector<double> base_g;
  std::vector<double> trial_x;
  std::vector<double> direction;
  std::vector<double> q;
  std::vector<double> preconditioner_input;
  std::vector<double> preconditioned_q;
  std::vector<double> inner_p;
  std::vector<double> inner_r;
  std::vector<double> inner_z;
  std::vector<double> inner_d;
  std::vector<double> inner_hv;
  std::vector<double> inner_hp;
  std::vector<double> alpha_work;
  std::vector<double> s_hist;
  std::vector<double> y_hist;
  std::vector<double> rho_hist;
  std::vector<double> f_window;
  std::vector<double> lb;
  std::vector<double> ub;
  bool has_bounds = false;
  bool initialized = false;
  bool fallback_accept = false;
  bool awaiting_accepted_gradient = false;
  bool pending_fallback_accept = false;
  bool awaiting_preconditioner = false;
  bool awaiting_inner_preconditioner = false;
  bool awaiting_hvp = false;
  bool awaiting_trust_region_trial = false;
  int64_t hist_count = 0;
  int64_t hist_start = 0;
  int64_t f_window_count = 0;
  int64_t f_window_start = 0;
  int64_t iter = 0;
  int64_t n_f = 0;
  int64_t n_g = 0;
  int64_t n_hvp = 0;
  int64_t n_prec = 0;
  int64_t line_search_iter = 0;
  int64_t inner_iter = 0;
  int64_t request_sequence = 0;
  double f = 0.0;
  double initial_f = 0.0;
  double base_f = 0.0;
  double last_f_change = 0.0;
  double initial_grad_norm = 0.0;
  double alpha = 1.0;
  double last_accepted_alpha = 1.0;
  double alpha_l = 0.0;
  double alpha_r = 0.0;
  double armijo_prev_alpha = 0.0;
  double armijo_prev_f = 0.0;
  double strong_wolfe_prev_alpha = 0.0;
  double strong_wolfe_prev_f = 0.0;
  double strong_wolfe_lo = 0.0;
  double strong_wolfe_hi = 0.0;
  double strong_wolfe_lo_f = 0.0;
  double q0 = 0.0;
  double q_trial = 0.0;
  double last_trial_alpha = 0.0;
  double last_trial_f = std::numeric_limits<double>::quiet_NaN();
  double last_line_search_reference = 0.0;
  double last_line_search_armijo_rhs = 0.0;
  double last_sy = 0.0;
  double last_yy = 0.0;
  double last_gamma = 1.0;
  double last_step_norm = 0.0;
  double last_direction_beta = 0.0;
  double last_preconditioner_dot = 0.0;
  double inner_rz = 0.0;
  double inner_residual_norm = 0.0;
  double inner_forcing_tolerance = 0.0;
  double inner_curvature = 0.0;
  double trust_radius = 0.0;
  double trust_ratio = 0.0;
  double predicted_reduction = 0.0;
  double actual_reduction = 0.0;
  int64_t trial_projection_count = 0;
  int64_t trial_lower_projection_count = 0;
  int64_t trial_upper_projection_count = 0;
  int64_t line_search_accept_count = 0;
  int64_t line_search_rejection_count = 0;
  int64_t line_search_failure_count = 0;
  int64_t line_search_fallback_accept_count = 0;
  int64_t nonfinite_trial_count = 0;
  int64_t pair_skip_count = 0;
  int64_t pair_stored_count = 0;
  int64_t pair_regularized_count = 0;
  int64_t preconditioner_skip_count = 0;
  int64_t inner_warning_count = 0;
  int64_t trust_region_accept_count = 0;
  int64_t trust_region_rejection_count = 0;
  int64_t trust_region_failure_count = 0;
  int32_t status = TIDE_OPTIM_STATUS_RUNNING;
  int32_t line_search_status = TIDE_OPTIM_LINE_SEARCH_STARTED;
  int32_t line_search_acceptance = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE;
  int32_t pair_status = TIDE_OPTIM_PAIR_NONE;
  int32_t direction_status = TIDE_OPTIM_DIRECTION_STATUS_INITIAL;
  int32_t preconditioner_status = TIDE_OPTIM_PRECONDITIONER_NONE;
  int32_t inner_status = TIDE_OPTIM_INNER_CG_NONE;
  int32_t trust_region_status = TIDE_OPTIM_TRUST_REGION_NONE;
  int32_t warning_flags = 0;
  bool armijo_has_prev = false;
  bool strong_wolfe_zoom = false;

  explicit LbfgsState(tide_optim_lbfgs_options const &options) : opt(options) {
    int64_t const n = opt.n;
    int64_t const h = opt.history_size;
    x.assign(n, 0.0);
    g.assign(n, 0.0);
    base_x.assign(n, 0.0);
    base_g.assign(n, 0.0);
    trial_x.assign(n, 0.0);
    direction.assign(n, 0.0);
    q.assign(n, 0.0);
    preconditioner_input.assign(n, 0.0);
    preconditioned_q.assign(n, 0.0);
    inner_p.assign(n, 0.0);
    inner_r.assign(n, 0.0);
    inner_z.assign(n, 0.0);
    inner_d.assign(n, 0.0);
    inner_hv.assign(n, 0.0);
    inner_hp.assign(n, 0.0);
    alpha_work.assign(h, 0.0);
    s_hist.assign(n * h, 0.0);
    y_hist.assign(n * h, 0.0);
    rho_hist.assign(h, 0.0);
    f_window.assign(opt.nonmonotone_window, 0.0);
  }
};

struct ProjectedGradientStats {
  double norm = 0.0;
  int64_t active_lower_count = 0;
  int64_t active_upper_count = 0;
  int64_t free_count = 0;
  int64_t kkt_violation_count = 0;
  int64_t lower_kkt_violation_count = 0;
  int64_t upper_kkt_violation_count = 0;
  int64_t free_gradient_count = 0;
};

tide::optim::Options options_from_c(tide_optim_lbfgs_options const &opt) {
  tide::optim::Options options{};
  options.n = opt.n;
  options.history_size = opt.history_size;
  options.max_iter = opt.max_iter;
  options.max_line_search = opt.max_line_search;
  options.max_eval = opt.max_eval;
  options.max_inner_iter = opt.max_inner_iter;
  options.nonmonotone_window = opt.nonmonotone_window;
  options.line_search =
      static_cast<tide::optim::LineSearchPolicy>(opt.line_search_policy);
  options.direction =
      static_cast<tide::optim::DirectionPolicy>(opt.direction_policy);
  options.nlcg_beta =
      static_cast<tide::optim::NlcgBetaPolicy>(opt.nlcg_beta_policy);
  options.lbfgs_update =
      static_cast<tide::optim::LbfgsUpdatePolicy>(opt.lbfgs_update_policy);
  options.globalization =
      static_cast<tide::optim::GlobalizationPolicy>(opt.globalization_policy);
  options.alpha_guess =
      static_cast<tide::optim::AlphaGuessPolicy>(opt.alpha_guess_policy);
  options.stopping =
      static_cast<tide::optim::StoppingPolicy>(opt.stopping_policy);
  options.initial_step = opt.initial_step;
  options.c1 = opt.c1;
  options.c2 = opt.c2;
  options.growth = opt.growth;
  options.alpha_min = opt.alpha_min;
  options.alpha_max = opt.alpha_max;
  options.gtol_abs = opt.gtol_abs;
  options.gtol_rel = opt.gtol_rel;
  options.f_atol = opt.f_atol;
  options.f_rtol = opt.f_rtol;
  options.x_atol = opt.x_atol;
  options.x_rtol = opt.x_rtol;
  options.inner_rtol = opt.inner_rtol;
  options.inner_atol = opt.inner_atol;
  options.initial_trust_radius = opt.initial_trust_radius;
  options.max_trust_radius = opt.max_trust_radius;
  options.trust_eta = opt.trust_eta;
  options.trust_shrink = opt.trust_shrink;
  options.trust_grow = opt.trust_grow;
  options.curvature_eps = opt.curvature_eps;
  options.gamma_min = opt.gamma_min;
  options.gamma_max = opt.gamma_max;
  options.armijo_shrink_min = opt.armijo_shrink_min;
  options.armijo_shrink_max = opt.armijo_shrink_max;
  options.bound_margin = opt.bound_margin;
  options.bounds =
      static_cast<tide::optim::BoundsStrategy>(opt.bounds_strategy);
  options.accept_decrease_after_maxls = opt.accept_decrease_after_maxls != 0;
  return options;
}

tide_optim_options_validation
validation_from_cpp(tide::optim::OptionsValidation const &validation) {
  return tide_optim_options_validation{
      static_cast<int32_t>(validation.code), validation.field,
      validation.message};
}

tide_optim_options_validation
validate_options_detail(tide_optim_lbfgs_options const *options) {
  if (options == nullptr) {
    return tide_optim_options_validation{
        TIDE_OPTIM_OPTIONS_VALIDATION_NULL_OPTIONS, "options",
        "options must not be null."};
  }
  return validation_from_cpp(options_from_c(*options).validate());
}

tide_optim_resolved_policies resolved_policies_detail(
    tide_optim_lbfgs_options const *options) {
  tide_optim_options_validation const validation =
      validate_options_detail(options);
  tide::optim::Options cpp_options{};
  if (options != nullptr) {
    cpp_options = options_from_c(*options);
  }
  tide::optim::ResolvedPolicies const policies =
      cpp_options.resolved_policies();
  return tide_optim_resolved_policies{
      validation.code == TIDE_OPTIM_OPTIONS_VALIDATION_OK ? 1 : 0,
      validation,
      static_cast<int32_t>(policies.direction),
      static_cast<int32_t>(policies.line_search),
      static_cast<int32_t>(policies.alpha_guess),
      static_cast<int32_t>(policies.stopping),
      static_cast<int32_t>(policies.nlcg_beta),
      static_cast<int32_t>(policies.lbfgs_update),
      static_cast<int32_t>(policies.globalization),
      static_cast<int32_t>(policies.bounds),
      static_cast<int32_t>(policies.cost_model),
      policies.method_name(),
      policies.direction_name(),
      policies.line_search_name(),
      policies.alpha_guess_name(),
      policies.stopping_name(),
      policies.nlcg_beta_name(),
      policies.lbfgs_update_name(),
      policies.globalization_name(),
      policies.bounds_name(),
      policies.cost_model_name()};
}

bool valid_options(tide_optim_lbfgs_options const &opt) {
  return validate_options_detail(&opt).code == TIDE_OPTIM_OPTIONS_VALIDATION_OK;
}

LbfgsState *as_state(void *handle) {
  return static_cast<LbfgsState *>(handle);
}

double dot(std::vector<double> const &a, std::vector<double> const &b) {
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

double dot_ptr(std::vector<double> const &a, double const *b) {
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

double norm2(std::vector<double> const &a) {
  return std::sqrt(dot(a, a));
}

double norm_inf(std::vector<double> const &a) {
  double result = 0.0;
  for (double const value : a) {
    result = std::max(result, std::abs(value));
  }
  return result;
}

bool finite_value(double value) { return std::isfinite(value); }

bool finite_vector(double const *x, int64_t n) {
  if (x == nullptr) {
    return false;
  }
  for (int64_t i = 0; i < n; ++i) {
    if (!std::isfinite(x[i])) {
      return false;
    }
  }
  return true;
}

bool finite_vector(std::vector<double> const &x) {
  for (double const value : x) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

void note_nonfinite_trial(LbfgsState &state) {
  state.warning_flags |= kWarnNonfiniteTrial;
  state.nonfinite_trial_count += 1;
}

void note_line_search_rejection(LbfgsState &state, int32_t status) {
  state.line_search_status = status;
  state.line_search_rejection_count += 1;
}

void note_line_search_failure(LbfgsState &state, int32_t status) {
  state.line_search_status = status;
  state.line_search_failure_count += 1;
}

void note_pair_skip(LbfgsState &state, int32_t status) {
  state.pair_status = status;
  state.pair_skip_count += 1;
  state.warning_flags |= kWarnPairSkipped;
}

void note_pair_stored(LbfgsState &state, int32_t status) {
  state.pair_status = status;
  state.pair_stored_count += 1;
  if (status == TIDE_OPTIM_PAIR_REGULARIZED_STORED) {
    state.pair_regularized_count += 1;
    state.warning_flags |= kWarnPairRegularized;
  }
}

void note_preconditioner_skip(LbfgsState &state, int32_t status) {
  state.preconditioner_status = status;
  state.preconditioner_skip_count += 1;
  state.warning_flags |= kWarnPreconditionerSkipped;
}

void note_inner_warning(LbfgsState &state) {
  state.warning_flags |= kWarnInnerCg;
  state.inner_warning_count += 1;
}

void note_trust_region_failure(LbfgsState &state, int32_t status) {
  state.trust_region_status = status;
  state.trust_region_failure_count += 1;
}

void copy_from(double const *src, std::vector<double> &dst) {
  std::copy(src, src + static_cast<std::ptrdiff_t>(dst.size()), dst.begin());
}

void copy_to(std::vector<double> const &src, double *dst) {
  std::copy(src.begin(), src.end(), dst);
}

bool uses_projected_trials(LbfgsState const &state) {
  return state.has_bounds &&
         (state.opt.bounds_strategy == TIDE_OPTIM_BOUNDS_PROJECTED_TRIAL ||
          state.opt.bounds_strategy == TIDE_OPTIM_BOUNDS_PROJECTED_GRADIENT);
}

bool uses_projected_gradient_convergence(LbfgsState const &state) {
  return state.has_bounds &&
         state.opt.bounds_strategy == TIDE_OPTIM_BOUNDS_PROJECTED_GRADIENT;
}

ProjectedGradientStats projected_gradient_stats(LbfgsState const &state) {
  ProjectedGradientStats stats{};
  if (!state.has_bounds) {
    stats.norm = norm_inf(state.g);
    stats.free_count = state.opt.n;
    return stats;
  }

  double const active_tol = std::max(1e-12, state.opt.bound_margin);
  double const violation_tol = std::max(1e-12, state.opt.gtol_abs);
  for (int64_t i = 0; i < state.opt.n; ++i) {
    bool const active_lower = state.x[i] <= state.lb[i] + active_tol;
    bool const active_upper = state.x[i] >= state.ub[i] - active_tol;
    if (active_lower) {
      stats.active_lower_count += 1;
    }
    if (active_upper) {
      stats.active_upper_count += 1;
    }
    if (!active_lower && !active_upper) {
      stats.free_count += 1;
    }

    bool const lower_kkt_satisfied = active_lower && state.g[i] >= 0.0;
    bool const upper_kkt_satisfied = active_upper && state.g[i] <= 0.0;
    double const residual =
        (lower_kkt_satisfied || upper_kkt_satisfied) ? 0.0 : state.g[i];
    stats.norm = std::max(stats.norm, std::abs(residual));
    if (std::abs(residual) > violation_tol) {
      stats.kkt_violation_count += 1;
      if (active_lower) {
        stats.lower_kkt_violation_count += 1;
      } else if (active_upper) {
        stats.upper_kkt_violation_count += 1;
      } else {
        stats.free_gradient_count += 1;
      }
    }
  }
  return stats;
}

double convergence_grad_norm(LbfgsState const &state) {
  if (uses_projected_gradient_convergence(state)) {
    return projected_gradient_stats(state).norm;
  }
  return norm_inf(state.g);
}

double step_tolerance(LbfgsState const &state) {
  if (state.opt.x_atol <= 0.0 && state.opt.x_rtol <= 0.0) {
    return 0.0;
  }
  return state.opt.x_atol +
         state.opt.x_rtol * std::max(1.0, norm2(state.x));
}

double grad_tolerance(LbfgsState const &state) {
  return state.opt.gtol_abs +
         state.opt.gtol_rel * std::max(1.0, state.initial_grad_norm);
}

double f_tolerance(LbfgsState const &state) {
  if (state.opt.stopping_policy == TIDE_OPTIM_STOPPING_GRADIENT_ONLY) {
    return 0.0;
  }
  if (state.opt.stopping_policy == TIDE_OPTIM_STOPPING_INITIAL_RELATIVE_F) {
    if (state.opt.f_rtol <= 0.0 || !finite_value(state.initial_f) ||
        state.initial_f <= 0.0) {
      return 0.0;
    }
    return state.opt.f_rtol * state.initial_f;
  }
  if (state.opt.f_atol <= 0.0 && state.opt.f_rtol <= 0.0) {
    return 0.0;
  }
  return state.opt.f_atol +
         state.opt.f_rtol * std::max(1.0, std::abs(state.f));
}

void fill_report(LbfgsState const &state, int32_t request,
                 tide_optim_report *report) {
  if (report == nullptr) {
    return;
  }
  ProjectedGradientStats const projected = projected_gradient_stats(state);
  report->request = request;
  report->status = state.status;
  report->line_search_status = state.line_search_status;
  report->line_search_acceptance = state.line_search_acceptance;
  report->pair_status = state.pair_status;
  report->warning_flags = state.warning_flags;
  report->line_search_policy = state.opt.line_search_policy;
  report->direction_policy = state.opt.direction_policy;
  report->nlcg_beta_policy = state.opt.nlcg_beta_policy;
  report->lbfgs_update_policy = state.opt.lbfgs_update_policy;
  report->preconditioner_status = state.preconditioner_status;
  report->inner_status = state.inner_status;
  report->globalization_policy = state.opt.globalization_policy;
  report->trust_region_status = state.trust_region_status;
  report->alpha_guess_policy = state.opt.alpha_guess_policy;
  report->stopping_policy = state.opt.stopping_policy;
  report->request_sequence = state.request_sequence;
  report->n = state.opt.n;
  report->iter = state.iter;
  report->n_f = state.n_f;
  report->n_g = state.n_g;
  report->n_hvp = state.n_hvp;
  report->n_prec = state.n_prec;
  report->line_search_iter = state.line_search_iter;
  report->inner_iter = state.inner_iter;
  report->history_size = state.hist_count;
  report->f = state.f;
  report->grad_norm = uses_projected_gradient_convergence(state)
                          ? projected.norm
                          : norm_inf(state.g);
  report->alpha = state.alpha;
  report->step_norm = state.last_step_norm;
  report->step_tolerance = step_tolerance(state);
  report->directional_derivative_initial = state.q0;
  report->directional_derivative_trial = state.q_trial;
  report->sy = state.last_sy;
  report->yy = state.last_yy;
  report->gamma = state.last_gamma;
  report->direction_beta = state.last_direction_beta;
  report->direction_status = state.direction_status;
  report->preconditioner_dot = state.last_preconditioner_dot;
  report->inner_residual_norm = state.inner_residual_norm;
  report->inner_forcing_tolerance = state.inner_forcing_tolerance;
  report->inner_curvature = state.inner_curvature;
  report->trust_radius = state.trust_radius;
  report->trust_ratio = state.trust_ratio;
  report->predicted_reduction = state.predicted_reduction;
  report->actual_reduction = state.actual_reduction;
  report->trial_alpha = state.last_trial_alpha;
  report->trial_f = state.last_trial_f;
  report->line_search_reference = state.last_line_search_reference;
  report->line_search_armijo_rhs = state.last_line_search_armijo_rhs;
  report->projected_grad_norm = projected.norm;
  report->active_lower_count = projected.active_lower_count;
  report->active_upper_count = projected.active_upper_count;
  report->free_count = projected.free_count;
  report->kkt_violation_count = projected.kkt_violation_count;
  report->lower_kkt_violation_count = projected.lower_kkt_violation_count;
  report->upper_kkt_violation_count = projected.upper_kkt_violation_count;
  report->free_gradient_count = projected.free_gradient_count;
  report->trial_projection_count = state.trial_projection_count;
  report->trial_lower_projection_count = state.trial_lower_projection_count;
  report->trial_upper_projection_count = state.trial_upper_projection_count;
  report->line_search_accept_count = state.line_search_accept_count;
  report->line_search_rejection_count = state.line_search_rejection_count;
  report->line_search_failure_count = state.line_search_failure_count;
  report->line_search_fallback_accept_count =
      state.line_search_fallback_accept_count;
  report->nonfinite_trial_count = state.nonfinite_trial_count;
  report->pair_skip_count = state.pair_skip_count;
  report->pair_stored_count = state.pair_stored_count;
  report->pair_regularized_count = state.pair_regularized_count;
  report->preconditioner_skip_count = state.preconditioner_skip_count;
  report->inner_warning_count = state.inner_warning_count;
  report->trust_region_accept_count = state.trust_region_accept_count;
  report->trust_region_rejection_count = state.trust_region_rejection_count;
  report->trust_region_failure_count = state.trust_region_failure_count;
  report->grad_tolerance = grad_tolerance(state);
  report->f_change = state.last_f_change;
  report->f_tolerance = f_tolerance(state);
  report->initial_f = state.initial_f;
  report->initial_grad_norm = state.initial_grad_norm;
  report->max_iter = state.opt.max_iter;
  report->max_eval = state.opt.max_eval;
}

void emit_report(LbfgsState &state, int32_t request,
                 tide_optim_report *report) {
  state.request_sequence += 1;
  fill_report(state, request, report);
}

int32_t line_search_request_kind(LbfgsState const &state) {
  if (state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_ARMIJO_CUBIC ||
      state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_NONMONOTONE_ARMIJO) {
    return TIDE_OPTIM_REQUEST_EVALUATE_F;
  }
  return TIDE_OPTIM_REQUEST_EVALUATE_FG;
}

int32_t current_request_kind(LbfgsState const &state) {
  if (!state.initialized) {
    return TIDE_OPTIM_REQUEST_ERROR;
  }
  if (state.status != TIDE_OPTIM_STATUS_RUNNING) {
    return TIDE_OPTIM_REQUEST_DONE;
  }
  if (state.awaiting_preconditioner || state.awaiting_inner_preconditioner) {
    return TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER;
  }
  if (state.awaiting_hvp) {
    return TIDE_OPTIM_REQUEST_EVALUATE_HV;
  }
  if (state.awaiting_trust_region_trial || state.awaiting_accepted_gradient) {
    return TIDE_OPTIM_REQUEST_EVALUATE_FG;
  }
  return line_search_request_kind(state);
}

char const *session_state_name(LbfgsState const *state) {
  if (state == nullptr) {
    return "INVALID";
  }
  if (!state->initialized) {
    return "NOT_STARTED";
  }
  if (state->status != TIDE_OPTIM_STATUS_RUNNING) {
    return "DONE";
  }
  return "RUNNING";
}

int32_t done(LbfgsState &state, int32_t status, tide_optim_report *report) {
  state.status = status;
  emit_report(state, TIDE_OPTIM_REQUEST_DONE, report);
  return TIDE_OPTIM_REQUEST_DONE;
}

void project_trial(LbfgsState &state, std::vector<double> &x) {
  if (!uses_projected_trials(state)) {
    return;
  }
  double const margin = std::max(0.0, state.opt.bound_margin);
  for (int64_t i = 0; i < state.opt.n; ++i) {
    if (x[i] < state.lb[i]) {
      x[i] = state.lb[i] + margin;
      state.trial_lower_projection_count += 1;
      state.trial_projection_count += 1;
    } else if (x[i] > state.ub[i]) {
      x[i] = state.ub[i] - margin;
      state.trial_upper_projection_count += 1;
      state.trial_projection_count += 1;
    }
  }
}

void make_trial(LbfgsState &state) {
  state.trial_projection_count = 0;
  state.trial_lower_projection_count = 0;
  state.trial_upper_projection_count = 0;
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.trial_x[i] = state.base_x[i] + state.alpha * state.direction[i];
  }
  project_trial(state, state.trial_x);
}

int64_t hist_slot(LbfgsState const &state, int64_t logical_index) {
  int64_t const h = state.opt.history_size;
  return (state.hist_start + logical_index) % h;
}

int64_t f_window_slot(LbfgsState const &state, int64_t logical_index) {
  int64_t const w = state.opt.nonmonotone_window;
  return (state.f_window_start + logical_index) % w;
}

void record_f_value(LbfgsState &state, double f) {
  int64_t slot = 0;
  if (state.f_window_count < state.opt.nonmonotone_window) {
    slot = f_window_slot(state, state.f_window_count);
    state.f_window_count += 1;
  } else {
    slot = state.f_window_start;
    state.f_window_start =
        (state.f_window_start + 1) % state.opt.nonmonotone_window;
  }
  state.f_window[slot] = f;
}

double armijo_reference(LbfgsState const &state) {
  if (state.opt.line_search_policy !=
      TIDE_OPTIM_LINE_SEARCH_POLICY_NONMONOTONE_ARMIJO) {
    return state.base_f;
  }
  double reference = state.base_f;
  for (int64_t i = 0; i < state.f_window_count; ++i) {
    reference = std::max(reference, state.f_window[f_window_slot(state, i)]);
  }
  return reference;
}

double *hist_vec(std::vector<double> &storage, int64_t n, int64_t slot) {
  return storage.data() + slot * n;
}

double const *hist_vec(std::vector<double> const &storage, int64_t n,
                       int64_t slot) {
  return storage.data() + slot * n;
}

double dot_hist(double const *a, std::vector<double> const &b, int64_t n) {
  double result = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

void lbfgs_first_loop(LbfgsState &state) {
  int64_t const n = state.opt.n;
  std::copy(state.g.begin(), state.g.end(), state.q.begin());

  for (int64_t rev = state.hist_count - 1; rev >= 0; --rev) {
    int64_t const slot = hist_slot(state, rev);
    double const *s = hist_vec(state.s_hist, n, slot);
    double const *y = hist_vec(state.y_hist, n, slot);
    double const alpha = state.rho_hist[slot] * dot_hist(s, state.q, n);
    state.alpha_work[rev] = alpha;
    for (int64_t i = 0; i < n; ++i) {
      state.q[i] -= alpha * y[i];
    }
  }
}

void lbfgs_second_loop(LbfgsState &state,
                       std::vector<double> const &initial_direction) {
  int64_t const n = state.opt.n;
  for (int64_t i = 0; i < n; ++i) {
    state.direction[i] = initial_direction[i];
  }

  for (int64_t idx = 0; idx < state.hist_count; ++idx) {
    int64_t const slot = hist_slot(state, idx);
    double const *s = hist_vec(state.s_hist, n, slot);
    double const *y = hist_vec(state.y_hist, n, slot);
    double const beta = state.rho_hist[slot] * dot_hist(y, state.direction, n);
    double const scale = state.alpha_work[idx] - beta;
    for (int64_t i = 0; i < n; ++i) {
      state.direction[i] += scale * s[i];
    }
  }

  for (double &value : state.direction) {
    value = -value;
  }
}

void compute_lbfgs_direction(LbfgsState &state) {
  int64_t const n = state.opt.n;
  lbfgs_first_loop(state);

  double gamma = 1.0;
  if (state.hist_count > 0) {
    gamma = std::min(std::max(state.last_gamma, state.opt.gamma_min),
                     state.opt.gamma_max);
  }
  for (int64_t i = 0; i < n; ++i) {
    state.preconditioned_q[i] = gamma * state.q[i];
  }
  lbfgs_second_loop(state, state.preconditioned_q);
}

void prepare_preconditioned_lbfgs_direction(LbfgsState &state) {
  lbfgs_first_loop(state);
  state.preconditioner_input = state.q;
}

void compute_steepest_descent_direction(LbfgsState &state) {
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.direction[i] = -state.g[i];
  }
  state.last_direction_beta = 0.0;
  state.direction_status = TIDE_OPTIM_DIRECTION_STATUS_INITIAL;
}

void compute_nlcg_direction(LbfgsState &state) {
  int64_t const n = state.opt.n;
  if (state.iter == 0) {
    compute_steepest_descent_direction(state);
    return;
  }

  std::vector<double> y(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    y[i] = state.g[i] - state.base_g[i];
  }

  double const gg = dot(state.g, state.g);
  double const previous_gg = dot(state.base_g, state.base_g);
  double const gy = dot(state.g, y);
  double const dy = dot(state.direction, y);
  double const dg = dot(state.direction, state.g);
  double const yy = dot(y, y);
  double numerator = gg;
  double denominator = dy;
  if (state.opt.nlcg_beta_policy ==
      TIDE_OPTIM_NLCG_BETA_FLETCHER_REEVES) {
    denominator = previous_gg;
  } else if (state.opt.nlcg_beta_policy ==
             TIDE_OPTIM_NLCG_BETA_POLAK_RIBIERE_PLUS) {
    numerator = gy;
    denominator = previous_gg;
  } else if (state.opt.nlcg_beta_policy ==
             TIDE_OPTIM_NLCG_BETA_HAGER_ZHANG) {
    numerator = gy - 2.0 * yy * dg / dy;
    denominator = dy;
  }
  double const direction_norm = norm2(state.direction);
  double const y_norm = norm2(y);
  double const previous_g_norm = norm2(state.base_g);
  double const min_denominator =
      (state.opt.nlcg_beta_policy == TIDE_OPTIM_NLCG_BETA_DAI_YUAN ||
       state.opt.nlcg_beta_policy == TIDE_OPTIM_NLCG_BETA_HAGER_ZHANG)
          ? state.opt.curvature_eps * std::max(1.0, direction_norm * y_norm)
          : state.opt.curvature_eps *
                std::max(1.0, previous_g_norm * previous_g_norm);

  if (!finite_value(numerator) || !finite_value(denominator) ||
      !finite_value(gg) || !finite_value(previous_gg) ||
      !finite_value(gy) || !finite_value(dy) || !finite_value(dg) ||
      !finite_value(yy) ||
      !finite_value(direction_norm) || !finite_value(y_norm) ||
      !finite_value(previous_g_norm)) {
    compute_steepest_descent_direction(state);
    state.direction_status = TIDE_OPTIM_DIRECTION_STATUS_RESTART_NONFINITE;
    return;
  }
  if (denominator <= min_denominator) {
    compute_steepest_descent_direction(state);
    state.direction_status = TIDE_OPTIM_DIRECTION_STATUS_RESTART_DENOMINATOR;
    return;
  }

  double beta = numerator / denominator;
  if (state.opt.nlcg_beta_policy ==
      TIDE_OPTIM_NLCG_BETA_POLAK_RIBIERE_PLUS) {
    beta = std::max(0.0, beta);
  }
  if (!finite_value(beta) ||
      (beta < 0.0 &&
       state.opt.nlcg_beta_policy != TIDE_OPTIM_NLCG_BETA_HAGER_ZHANG)) {
    compute_steepest_descent_direction(state);
    state.direction_status = TIDE_OPTIM_DIRECTION_STATUS_RESTART_NONFINITE;
    return;
  }

  for (int64_t i = 0; i < n; ++i) {
    state.direction[i] = -state.g[i] + beta * state.direction[i];
  }
  state.last_direction_beta = beta;
  state.direction_status = TIDE_OPTIM_DIRECTION_STATUS_UPDATE;

  double const descent = dot(state.g, state.direction);
  if (!finite_value(descent) ||
      descent >= -state.opt.curvature_eps * std::max(1.0, gg)) {
    compute_steepest_descent_direction(state);
    state.direction_status = TIDE_OPTIM_DIRECTION_STATUS_RESTART_NON_DESCENT;
  }
}

void update_step_diagnostics_without_pair(LbfgsState &state) {
  state.pair_status = TIDE_OPTIM_PAIR_NONE;
  state.last_sy = 0.0;
  state.last_yy = 0.0;
  state.last_step_norm = 0.0;
  double step_norm_sq = 0.0;
  double sy = 0.0;
  double yy = 0.0;
  for (int64_t i = 0; i < state.opt.n; ++i) {
    double const step = state.x[i] - state.base_x[i];
    double const y = state.g[i] - state.base_g[i];
    step_norm_sq += step * step;
    sy += step * y;
    yy += y * y;
  }
  state.last_step_norm = std::sqrt(step_norm_sq);
  state.last_sy = sy;
  state.last_yy = yy;
  if (finite_value(sy) && finite_value(yy) && yy > 0.0 && sy > 0.0) {
    state.last_gamma = std::min(std::max(sy / yy, state.opt.gamma_min),
                                state.opt.gamma_max);
  }
}

void store_pair(LbfgsState &state, bool skip_for_fallback) {
  int64_t const n = state.opt.n;
  state.pair_status = TIDE_OPTIM_PAIR_NONE;
  state.last_sy = 0.0;
  state.last_yy = 0.0;
  state.last_step_norm = 0.0;

  std::vector<double> s(static_cast<std::size_t>(n));
  std::vector<double> y(static_cast<std::size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    s[i] = state.x[i] - state.base_x[i];
    y[i] = state.g[i] - state.base_g[i];
  }
  double sy = dot(s, y);
  double yy = dot(y, y);
  double const sn = norm2(s);
  double yn = std::sqrt(yy);
  state.last_sy = sy;
  state.last_yy = yy;
  state.last_step_norm = sn;

  if (skip_for_fallback) {
    note_pair_skip(state, TIDE_OPTIM_PAIR_SKIPPED_LINE_SEARCH_FALLBACK);
    return;
  }
  if (state.trial_projection_count > 0) {
    note_pair_skip(state, TIDE_OPTIM_PAIR_SKIPPED_BOUNDS_PROJECTION);
    return;
  }
  if (state.opt.direction_policy ==
          TIDE_OPTIM_DIRECTION_PRECONDITIONED_LBFGS &&
      state.preconditioner_status != TIDE_OPTIM_PRECONDITIONER_APPLIED) {
    note_pair_skip(state, TIDE_OPTIM_PAIR_SKIPPED_PRECONDITIONER);
    return;
  }
  if (!finite_value(sy) || !finite_value(yy) || !finite_value(sn) ||
      !finite_value(yn)) {
    note_pair_skip(state, TIDE_OPTIM_PAIR_SKIPPED_NONFINITE);
    return;
  }
  double const min_curvature = state.opt.curvature_eps * sn * yn;
  if (sn == 0.0 || yn == 0.0 || sy <= min_curvature) {
    if (state.opt.lbfgs_update_policy != TIDE_OPTIM_LBFGS_UPDATE_REGULARIZE ||
        sn == 0.0 || !finite_value(sn)) {
      note_pair_skip(state, TIDE_OPTIM_PAIR_SKIPPED_BAD_CURVATURE);
      return;
    }
    double const ss = sn * sn;
    double target_sy = state.opt.curvature_eps * std::max(1.0, ss);
    if (finite_value(yn) && yn > 0.0) {
      target_sy = std::max(target_sy, state.opt.curvature_eps * sn * yn);
    }
    double const tau = (target_sy - sy) / ss;
    if (!finite_value(ss) || ss <= 0.0 || !finite_value(target_sy) ||
        !finite_value(tau)) {
      note_pair_skip(state, TIDE_OPTIM_PAIR_SKIPPED_BAD_CURVATURE);
      return;
    }
    for (int64_t i = 0; i < n; ++i) {
      y[i] += tau * s[i];
    }
    sy = dot(s, y);
    yy = dot(y, y);
    yn = std::sqrt(yy);
    state.last_sy = sy;
    state.last_yy = yy;
    if (!finite_value(sy) || !finite_value(yy) || !finite_value(yn) ||
        yn == 0.0 || sy <= 0.0) {
      note_pair_skip(state, TIDE_OPTIM_PAIR_SKIPPED_BAD_CURVATURE);
      return;
    }
    note_pair_stored(state, TIDE_OPTIM_PAIR_REGULARIZED_STORED);
  }

  int64_t slot = 0;
  if (state.hist_count < state.opt.history_size) {
    slot = hist_slot(state, state.hist_count);
    state.hist_count += 1;
  } else {
    slot = state.hist_start;
    state.hist_start = (state.hist_start + 1) % state.opt.history_size;
  }
  double *s_dst = hist_vec(state.s_hist, n, slot);
  double *y_dst = hist_vec(state.y_hist, n, slot);
  for (int64_t i = 0; i < n; ++i) {
    s_dst[i] = s[i];
    y_dst[i] = y[i];
  }
  state.rho_hist[slot] = 1.0 / sy;
  state.last_gamma = std::min(std::max(sy / yy, state.opt.gamma_min),
                              state.opt.gamma_max);
  if (state.pair_status != TIDE_OPTIM_PAIR_REGULARIZED_STORED) {
    note_pair_stored(state, TIDE_OPTIM_PAIR_STORED);
  }
}

bool gradient_converged(LbfgsState const &state) {
  if (state.opt.stopping_policy ==
      TIDE_OPTIM_STOPPING_INITIAL_RELATIVE_F) {
    return false;
  }
  return convergence_grad_norm(state) <= grad_tolerance(state);
}

bool ftol_converged(LbfgsState const &state, double previous_f) {
  if (state.opt.stopping_policy == TIDE_OPTIM_STOPPING_GRADIENT_ONLY) {
    return false;
  }
  if (state.opt.stopping_policy ==
      TIDE_OPTIM_STOPPING_INITIAL_RELATIVE_F) {
    double const threshold = f_tolerance(state);
    return threshold > 0.0 && state.f <= threshold;
  }
  if (state.opt.f_atol <= 0.0 && state.opt.f_rtol <= 0.0) {
    return false;
  }
  double const threshold = f_tolerance(state);
  return threshold > 0.0 && std::abs(previous_f - state.f) <= threshold;
}

bool xtol_converged(LbfgsState const &state) {
  if (state.opt.stopping_policy != TIDE_OPTIM_STOPPING_STANDARD) {
    return false;
  }
  if (state.iter == 0 ||
      (state.opt.x_atol <= 0.0 && state.opt.x_rtol <= 0.0)) {
    return false;
  }
  return state.last_step_norm <= step_tolerance(state);
}

bool max_eval_reached(LbfgsState const &state) {
  return state.opt.max_eval > 0 && state.n_f >= state.opt.max_eval;
}

double clipped_alpha_guess(LbfgsState const &state, double alpha) {
  if (!finite_value(alpha) || alpha <= 0.0) {
    alpha = state.opt.initial_step;
  }
  return std::min(std::max(alpha, state.opt.alpha_min), state.opt.alpha_max);
}

double initial_line_search_alpha(LbfgsState const &state) {
  if (state.opt.alpha_guess_policy == TIDE_OPTIM_ALPHA_GUESS_PREVIOUS &&
      state.iter > 0) {
    return clipped_alpha_guess(state, state.last_accepted_alpha);
  }
  if (state.opt.alpha_guess_policy ==
          TIDE_OPTIM_ALPHA_GUESS_BARZILAI_BORWEIN &&
      state.iter > 0 && state.last_sy > 0.0 && state.last_yy > 0.0) {
    return clipped_alpha_guess(state, state.last_sy / state.last_yy);
  }
  return clipped_alpha_guess(state, state.opt.initial_step);
}

bool uses_trust_region(LbfgsState const &state);
double trust_predicted_reduction(LbfgsState const &state);

int32_t begin_trial_after_direction(LbfgsState &state, double *x_request,
                                    tide_optim_report *report) {
  state.base_f = state.f;
  state.base_x = state.x;
  state.base_g = state.g;

  state.q0 = dot(state.g, state.direction);
  if (!finite_value(state.q0) || state.q0 >= 0.0) {
    state.warning_flags |= kWarnNonDescentReset;
    for (int64_t i = 0; i < state.opt.n; ++i) {
      state.direction[i] = -state.g[i];
    }
    state.q0 = dot(state.g, state.direction);
  }
  if (!finite_value(state.q0) || state.q0 >= 0.0) {
    return done(state, TIDE_OPTIM_STATUS_NON_DESCENT_DIRECTION, report);
  }

  state.last_line_search_reference = armijo_reference(state);
  state.last_line_search_armijo_rhs =
      state.last_line_search_reference + state.opt.c1 * state.alpha * state.q0;
  make_trial(state);
  copy_to(state.trial_x, x_request);
  int32_t const request = line_search_request_kind(state);
  emit_report(state, request, report);
  return request;
}

int32_t begin_trust_region_trial_after_direction(LbfgsState &state,
                                                 double *x_request,
                                                 tide_optim_report *report) {
  state.base_f = state.f;
  state.base_x = state.x;
  state.base_g = state.g;
  state.alpha = 1.0;
  state.last_line_search_reference = state.base_f;
  state.last_line_search_armijo_rhs =
      std::numeric_limits<double>::quiet_NaN();
  state.q0 = dot(state.g, state.direction);
  state.predicted_reduction = trust_predicted_reduction(state);
  state.actual_reduction = 0.0;
  state.trust_ratio = 0.0;
  state.trust_region_status = TIDE_OPTIM_TRUST_REGION_STARTED;

  if (!finite_value(state.q0) || !finite_value(state.predicted_reduction) ||
      state.predicted_reduction <= 0.0) {
    state.trust_region_status =
        TIDE_OPTIM_TRUST_REGION_FAILED_PREDICTED_REDUCTION;
    return done(state, TIDE_OPTIM_STATUS_TRUST_REGION_FAILED, report);
  }

  make_trial(state);
  copy_to(state.trial_x, x_request);
  emit_report(state, TIDE_OPTIM_REQUEST_EVALUATE_FG, report);
  state.awaiting_trust_region_trial = true;
  return TIDE_OPTIM_REQUEST_EVALUATE_FG;
}

int32_t begin_globalization_after_direction(LbfgsState &state,
                                            double *x_request,
                                            tide_optim_report *report) {
  if (uses_trust_region(state)) {
    return begin_trust_region_trial_after_direction(state, x_request, report);
  }
  return begin_trial_after_direction(state, x_request, report);
}

bool uses_truncated_newton(LbfgsState const &state) {
  return state.opt.direction_policy == TIDE_OPTIM_DIRECTION_TRUNCATED_NEWTON ||
         state.opt.direction_policy ==
             TIDE_OPTIM_DIRECTION_PRECONDITIONED_TRUNCATED_NEWTON;
}

bool uses_preconditioned_truncated_newton(LbfgsState const &state) {
  return state.opt.direction_policy ==
         TIDE_OPTIM_DIRECTION_PRECONDITIONED_TRUNCATED_NEWTON;
}

bool uses_trust_region(LbfgsState const &state) {
  return state.opt.globalization_policy ==
         TIDE_OPTIM_GLOBALIZATION_TRUST_REGION;
}

int64_t inner_iter_limit(LbfgsState const &state) {
  if (state.opt.max_inner_iter > 0) {
    return state.opt.max_inner_iter;
  }
  return state.opt.n;
}

void set_steepest_direction_from_inner_fallback(LbfgsState &state) {
  double const grad_norm = norm2(state.g);
  double scale = 1.0;
  if (uses_trust_region(state) && grad_norm > state.trust_radius &&
      grad_norm > 0.0) {
    scale = state.trust_radius / grad_norm;
  }
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.direction[i] = -scale * state.g[i];
    state.inner_hp[i] = 0.0;
  }
}

void set_newton_direction_from_inner_p(LbfgsState &state) {
  double p_norm = 0.0;
  for (int64_t i = 0; i < state.opt.n; ++i) {
    p_norm += state.inner_p[i] * state.inner_p[i];
  }
  if (p_norm == 0.0 || !finite_value(p_norm)) {
    set_steepest_direction_from_inner_fallback(state);
    return;
  }
  state.direction = state.inner_p;
}

double tau_to_trust_boundary(LbfgsState const &state,
                             std::vector<double> const &p,
                             std::vector<double> const &d) {
  double const a = dot(d, d);
  double const b = 2.0 * dot(p, d);
  double const c = dot(p, p) - state.trust_radius * state.trust_radius;
  double const disc = b * b - 4.0 * a * c;
  if (!finite_value(a) || !finite_value(b) || !finite_value(c) ||
      !finite_value(disc) || a <= 0.0 || disc < 0.0) {
    return 0.0;
  }
  return (-b + std::sqrt(disc)) / (2.0 * a);
}

void add_inner_step(LbfgsState &state, double step_scale) {
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.inner_p[i] += step_scale * state.inner_d[i];
    state.inner_hp[i] += step_scale * state.inner_hv[i];
  }
}

double trust_predicted_reduction(LbfgsState const &state) {
  double const linear = dot(state.g, state.direction);
  double const quadratic = dot(state.direction, state.inner_hp);
  return -(linear + 0.5 * quadratic);
}

int32_t request_hvp_for_inner_cg(LbfgsState &state, double *x_request,
                                 tide_optim_report *report) {
  state.awaiting_hvp = true;
  copy_to(state.inner_d, x_request);
  emit_report(state, TIDE_OPTIM_REQUEST_EVALUATE_HV, report);
  return TIDE_OPTIM_REQUEST_EVALUATE_HV;
}

int32_t request_preconditioner_for_inner_cg(LbfgsState &state,
                                            double *x_request,
                                            tide_optim_report *report) {
  state.awaiting_inner_preconditioner = true;
  state.preconditioner_input = state.inner_r;
  copy_to(state.preconditioner_input, x_request);
  emit_report(state, TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER, report);
  return TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER;
}

void update_inner_direction_from_z(LbfgsState &state, bool first_direction) {
  double rz = dot(state.inner_r, state.inner_z);
  if (!finite_value(rz) || rz <= 0.0) {
    state.inner_status = TIDE_OPTIM_INNER_CG_PRECONDITIONER_BREAKDOWN;
    note_inner_warning(state);
    state.inner_z = state.inner_r;
    rz = dot(state.inner_r, state.inner_z);
  }

  if (first_direction || !finite_value(state.inner_rz) ||
      state.inner_rz <= 0.0) {
    state.inner_rz = rz;
    state.inner_d = state.inner_z;
    return;
  }

  double const beta = rz / state.inner_rz;
  if (!finite_value(beta)) {
    state.inner_status = TIDE_OPTIM_INNER_CG_PRECONDITIONER_BREAKDOWN;
    note_inner_warning(state);
    state.inner_z = state.inner_r;
    state.inner_rz = dot(state.inner_r, state.inner_z);
    state.inner_d = state.inner_z;
    return;
  }
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.inner_d[i] = state.inner_z[i] + beta * state.inner_d[i];
  }
  state.inner_rz = rz;
}

int32_t continue_inner_cg_after_z(LbfgsState &state, bool first_direction,
                                  double *x_request,
                                  tide_optim_report *report) {
  update_inner_direction_from_z(state, first_direction);
  return request_hvp_for_inner_cg(state, x_request, report);
}

int32_t start_inner_cg(LbfgsState &state, double *x_request,
                       tide_optim_report *report) {
  std::fill(state.inner_p.begin(), state.inner_p.end(), 0.0);
  std::fill(state.inner_hp.begin(), state.inner_hp.end(), 0.0);
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.inner_r[i] = -state.g[i];
  }
  state.inner_iter = 0;
  state.inner_status = TIDE_OPTIM_INNER_CG_STARTED;
  state.inner_curvature = 0.0;
  state.inner_residual_norm = norm2(state.inner_r);
  state.inner_forcing_tolerance =
      state.opt.inner_atol + state.opt.inner_rtol * norm2(state.g);
  state.preconditioner_status = TIDE_OPTIM_PRECONDITIONER_NONE;
  state.last_preconditioner_dot = 0.0;

  if (!finite_value(state.inner_residual_norm) ||
      !finite_value(state.inner_forcing_tolerance)) {
    state.inner_status = TIDE_OPTIM_INNER_CG_NONFINITE_HVP;
    note_inner_warning(state);
    return done(state, TIDE_OPTIM_STATUS_INNER_CG_FAILED, report);
  }
  if (state.inner_residual_norm <= state.inner_forcing_tolerance) {
    state.inner_status = TIDE_OPTIM_INNER_CG_FORCING_REACHED;
    set_steepest_direction_from_inner_fallback(state);
    return begin_globalization_after_direction(state, x_request, report);
  }

  if (uses_preconditioned_truncated_newton(state)) {
    return request_preconditioner_for_inner_cg(state, x_request, report);
  }

  state.inner_z = state.inner_r;
  state.inner_rz = dot(state.inner_r, state.inner_z);
  state.inner_d = state.inner_z;
  return request_hvp_for_inner_cg(state, x_request, report);
}

int32_t finish_inner_cg_with_current_p(LbfgsState &state, int32_t inner_status,
                                       double *x_request,
                                       tide_optim_report *report) {
  state.inner_status = inner_status;
  set_newton_direction_from_inner_p(state);
  return begin_globalization_after_direction(state, x_request, report);
}

int32_t continue_inner_cg_after_hvp(LbfgsState &state, double const *hv,
                                    double *x_request,
                                    tide_optim_report *report) {
  state.n_hvp += 1;
  state.awaiting_hvp = false;

  if (!finite_vector(hv, state.opt.n)) {
    state.inner_status = TIDE_OPTIM_INNER_CG_NONFINITE_HVP;
    note_inner_warning(state);
    return done(state, TIDE_OPTIM_STATUS_INNER_CG_FAILED, report);
  }
  copy_from(hv, state.inner_hv);
  double const d_hd = dot(state.inner_d, state.inner_hv);
  double const d_norm = norm2(state.inner_d);
  double const hv_norm = norm2(state.inner_hv);
  state.inner_curvature = d_hd;
  double const min_curvature = state.opt.curvature_eps * d_norm * hv_norm;

  if (!finite_value(d_hd) || !finite_value(d_norm) || !finite_value(hv_norm)) {
    state.inner_status = TIDE_OPTIM_INNER_CG_NONFINITE_HVP;
    note_inner_warning(state);
    return done(state, TIDE_OPTIM_STATUS_INNER_CG_FAILED, report);
  }
  if (d_hd <= min_curvature) {
    if (uses_trust_region(state)) {
      double const tau =
          tau_to_trust_boundary(state, state.inner_p, state.inner_d);
      add_inner_step(state, tau);
      state.inner_status = tau > 0.0
                               ? TIDE_OPTIM_INNER_CG_TRUST_BOUNDARY
                               : (d_hd <= 0.0
                                      ? TIDE_OPTIM_INNER_CG_NEGATIVE_CURVATURE
                                      : TIDE_OPTIM_INNER_CG_ZERO_CURVATURE);
      set_newton_direction_from_inner_p(state);
      return begin_globalization_after_direction(state, x_request, report);
    }
    if (state.inner_iter == 0) {
      set_steepest_direction_from_inner_fallback(state);
    } else {
      set_newton_direction_from_inner_p(state);
    }
    int32_t const status = d_hd <= 0.0
                               ? TIDE_OPTIM_INNER_CG_NEGATIVE_CURVATURE
                               : TIDE_OPTIM_INNER_CG_ZERO_CURVATURE;
    state.inner_status = status;
    return begin_globalization_after_direction(state, x_request, report);
  }

  double const alpha_cg = state.inner_rz / d_hd;
  if (!finite_value(alpha_cg)) {
    state.inner_status = TIDE_OPTIM_INNER_CG_NONFINITE_HVP;
    note_inner_warning(state);
    return done(state, TIDE_OPTIM_STATUS_INNER_CG_FAILED, report);
  }
  if (uses_trust_region(state)) {
    std::vector<double> next_p = state.inner_p;
    for (int64_t i = 0; i < state.opt.n; ++i) {
      next_p[i] += alpha_cg * state.inner_d[i];
    }
    if (norm2(next_p) >= state.trust_radius) {
      double const tau =
          tau_to_trust_boundary(state, state.inner_p, state.inner_d);
      add_inner_step(state, tau);
      state.inner_iter += 1;
      state.inner_status = TIDE_OPTIM_INNER_CG_TRUST_BOUNDARY;
      set_newton_direction_from_inner_p(state);
      return begin_globalization_after_direction(state, x_request, report);
    }
  }

  add_inner_step(state, alpha_cg);
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.inner_r[i] -= alpha_cg * state.inner_hv[i];
  }
  state.inner_iter += 1;
  state.inner_residual_norm = norm2(state.inner_r);

  if (!finite_value(state.inner_residual_norm)) {
    state.inner_status = TIDE_OPTIM_INNER_CG_NONFINITE_HVP;
    note_inner_warning(state);
    return done(state, TIDE_OPTIM_STATUS_INNER_CG_FAILED, report);
  }
  if (state.inner_residual_norm <= state.inner_forcing_tolerance) {
    return finish_inner_cg_with_current_p(
        state, TIDE_OPTIM_INNER_CG_FORCING_REACHED, x_request, report);
  }
  if (state.inner_iter >= inner_iter_limit(state)) {
    return finish_inner_cg_with_current_p(
        state, TIDE_OPTIM_INNER_CG_MAX_ITER, x_request, report);
  }

  if (uses_preconditioned_truncated_newton(state)) {
    return request_preconditioner_for_inner_cg(state, x_request, report);
  }

  double const rz_old = state.inner_rz;
  state.inner_z = state.inner_r;
  state.inner_rz = dot(state.inner_r, state.inner_z);
  double const beta = state.inner_rz / rz_old;
  if (!finite_value(beta)) {
    return finish_inner_cg_with_current_p(
        state, TIDE_OPTIM_INNER_CG_PRECONDITIONER_BREAKDOWN, x_request,
        report);
  }
  for (int64_t i = 0; i < state.opt.n; ++i) {
    state.inner_d[i] = state.inner_z[i] + beta * state.inner_d[i];
  }
  return request_hvp_for_inner_cg(state, x_request, report);
}

int32_t begin_line_search(LbfgsState &state, double *x_request,
                          tide_optim_report *report) {
  if (gradient_converged(state)) {
    return done(state, TIDE_OPTIM_STATUS_CONVERGED_GRADIENT, report);
  }
  if (max_eval_reached(state)) {
    return done(state, TIDE_OPTIM_STATUS_MAX_EVAL, report);
  }
  if (state.iter >= state.opt.max_iter) {
    return done(state, TIDE_OPTIM_STATUS_MAX_ITER, report);
  }

  state.warning_flags = 0;
  state.pair_status = TIDE_OPTIM_PAIR_NONE;
  state.line_search_status = TIDE_OPTIM_LINE_SEARCH_STARTED;
  state.line_search_acceptance = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE;
  state.line_search_iter = 0;
  state.awaiting_accepted_gradient = false;
  state.pending_fallback_accept = false;
  state.awaiting_preconditioner = false;
  state.awaiting_inner_preconditioner = false;
  state.awaiting_hvp = false;
  state.awaiting_trust_region_trial = false;
  state.last_trial_f = std::numeric_limits<double>::quiet_NaN();
  state.alpha = initial_line_search_alpha(state);
  state.last_trial_alpha = state.alpha;
  state.alpha_l = 0.0;
  state.alpha_r = 0.0;
  state.armijo_has_prev = false;
  state.armijo_prev_alpha = 0.0;
  state.armijo_prev_f = 0.0;
  state.strong_wolfe_zoom = false;
  state.strong_wolfe_prev_alpha = 0.0;
  state.strong_wolfe_prev_f = state.f;
  state.strong_wolfe_lo = 0.0;
  state.strong_wolfe_hi = 0.0;
  state.strong_wolfe_lo_f = state.f;
  state.preconditioner_status = TIDE_OPTIM_PRECONDITIONER_NONE;
  state.last_preconditioner_dot = 0.0;
  state.inner_status = TIDE_OPTIM_INNER_CG_NONE;
  state.inner_iter = 0;
  state.inner_residual_norm = 0.0;
  state.inner_forcing_tolerance = 0.0;
  state.inner_curvature = 0.0;
  state.trust_region_status = TIDE_OPTIM_TRUST_REGION_NONE;
  state.trust_ratio = 0.0;
  state.predicted_reduction = 0.0;
  state.actual_reduction = 0.0;

  if (state.opt.direction_policy ==
      TIDE_OPTIM_DIRECTION_PRECONDITIONED_LBFGS) {
    prepare_preconditioned_lbfgs_direction(state);
    state.awaiting_preconditioner = true;
    copy_to(state.preconditioner_input, x_request);
    emit_report(state, TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER, report);
    return TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER;
  }

  if (uses_truncated_newton(state)) {
    return start_inner_cg(state, x_request, report);
  }

  if (state.opt.direction_policy == TIDE_OPTIM_DIRECTION_LBFGS &&
      state.hist_count > 0) {
    compute_lbfgs_direction(state);
  } else if (state.opt.direction_policy == TIDE_OPTIM_DIRECTION_NLCG_DAI_YUAN) {
    compute_nlcg_direction(state);
  } else {
    compute_steepest_descent_direction(state);
  }

  return begin_trial_after_direction(state, x_request, report);
}

bool update_alpha_legacy_weak_wolfe(LbfgsState &state, bool armijo_failure) {
  if (armijo_failure) {
    state.alpha_r = state.alpha;
    state.alpha = (state.alpha_l + state.alpha_r) * 0.5;
  } else {
    state.alpha_l = state.alpha;
    if (state.alpha_r == 0.0) {
      state.alpha = state.opt.growth * state.alpha;
    } else {
      state.alpha = (state.alpha_l + state.alpha_r) * 0.5;
    }
  }
  return std::isfinite(state.alpha) && state.alpha >= state.opt.alpha_min &&
         state.alpha <= state.opt.alpha_max;
}

bool set_next_alpha(LbfgsState &state, double candidate) {
  if (!std::isfinite(candidate) || candidate < state.opt.alpha_min ||
      candidate > state.opt.alpha_max || candidate == state.alpha) {
    return false;
  }
  state.alpha = candidate;
  return true;
}

double clipped_shrink(LbfgsState const &state, double candidate) {
  double const lo = state.opt.armijo_shrink_min * state.alpha;
  double const hi = state.opt.armijo_shrink_max * state.alpha;
  if (!std::isfinite(candidate) || candidate <= 0.0) {
    return hi;
  }
  return std::min(std::max(candidate, lo), hi);
}

bool update_alpha_armijo_cubic(LbfgsState &state, double f,
                               bool nonfinite_trial) {
  double candidate = state.alpha * state.opt.armijo_shrink_max;
  if (!nonfinite_trial && !state.armijo_has_prev) {
    double const denom =
        2.0 * (f - state.base_f - state.q0 * state.alpha);
    if (std::isfinite(denom) && denom > 0.0) {
      candidate = -(state.q0 * state.alpha * state.alpha) / denom;
    }
  } else if (!nonfinite_trial) {
    double const alpha0 = state.armijo_prev_alpha;
    double const alpha1 = state.alpha;
    double const d0 = state.armijo_prev_f - state.base_f - state.q0 * alpha0;
    double const d1 = f - state.base_f - state.q0 * alpha1;
    double const denom = alpha0 * alpha0 * alpha1 * alpha1 * (alpha1 - alpha0);
    if (std::isfinite(denom) && denom != 0.0) {
      double const a = (alpha0 * alpha0 * d1 - alpha1 * alpha1 * d0) / denom;
      double const b =
          (-alpha0 * alpha0 * alpha0 * d1 + alpha1 * alpha1 * alpha1 * d0) /
          denom;
      double const disc = b * b - 3.0 * a * state.q0;
      if (std::isfinite(a) && std::isfinite(b) && std::isfinite(disc) &&
          a != 0.0 && disc >= 0.0) {
        candidate = (-b + std::sqrt(disc)) / (3.0 * a);
      }
    }
  }
  if (!nonfinite_trial) {
    state.armijo_prev_alpha = state.alpha;
    state.armijo_prev_f = f;
    state.armijo_has_prev = true;
  }
  state.alpha = clipped_shrink(state, candidate);
  return std::isfinite(state.alpha) && state.alpha >= state.opt.alpha_min &&
         state.alpha <= state.opt.alpha_max;
}

double bracket_midpoint(double a, double b) { return a + 0.5 * (b - a); }

bool update_alpha_strong_wolfe(LbfgsState &state, double f, double q_trial,
                               bool armijo_failure, bool nonfinite_trial) {
  if (nonfinite_trial) {
    double const lo = state.strong_wolfe_prev_alpha;
    if (lo > 0.0 && lo < state.alpha) {
      state.strong_wolfe_zoom = true;
      state.strong_wolfe_lo = lo;
      state.strong_wolfe_hi = state.alpha;
      state.strong_wolfe_lo_f = state.strong_wolfe_prev_f;
      return set_next_alpha(state,
                            bracket_midpoint(state.strong_wolfe_lo,
                                             state.strong_wolfe_hi));
    }
    return set_next_alpha(state, 0.5 * state.alpha);
  }

  if (!state.strong_wolfe_zoom) {
    bool const past_first_trial = state.line_search_iter > 0;
    if (armijo_failure ||
        (past_first_trial && f >= state.strong_wolfe_prev_f)) {
      state.strong_wolfe_zoom = true;
      state.strong_wolfe_lo = state.strong_wolfe_prev_alpha;
      state.strong_wolfe_hi = state.alpha;
      state.strong_wolfe_lo_f = state.strong_wolfe_prev_f;
      return set_next_alpha(state,
                            bracket_midpoint(state.strong_wolfe_lo,
                                             state.strong_wolfe_hi));
    }
    if (q_trial >= 0.0) {
      state.strong_wolfe_zoom = true;
      state.strong_wolfe_lo = state.alpha;
      state.strong_wolfe_hi = state.strong_wolfe_prev_alpha;
      state.strong_wolfe_lo_f = f;
      return set_next_alpha(state,
                            bracket_midpoint(state.strong_wolfe_lo,
                                             state.strong_wolfe_hi));
    }
    state.strong_wolfe_prev_alpha = state.alpha;
    state.strong_wolfe_prev_f = f;
    double const candidate = std::min(state.opt.alpha_max,
                                      state.opt.growth * state.alpha);
    return set_next_alpha(state, candidate);
  }

  if (armijo_failure || f >= state.strong_wolfe_lo_f) {
    state.strong_wolfe_hi = state.alpha;
  } else {
    if (q_trial * (state.strong_wolfe_hi - state.strong_wolfe_lo) >= 0.0) {
      state.strong_wolfe_hi = state.strong_wolfe_lo;
    }
    state.strong_wolfe_lo = state.alpha;
    state.strong_wolfe_lo_f = f;
  }
  return set_next_alpha(state,
                        bracket_midpoint(state.strong_wolfe_lo,
                                         state.strong_wolfe_hi));
}

double hager_zhang_epsilon(LbfgsState const &state) {
  return 1e-12 * std::max(1.0, std::abs(state.base_f));
}

bool hager_zhang_approximate_decrease(LbfgsState const &state, double f) {
  return f <= state.base_f + hager_zhang_epsilon(state);
}

bool hager_zhang_upper_curvature(LbfgsState const &state, double q_trial) {
  return q_trial <= (2.0 * state.opt.c1 - 1.0) * state.q0;
}

int32_t line_search_acceptance_kind(LbfgsState const &state, bool armijo,
                                    bool weak_curvature,
                                    bool strong_curvature, double f,
                                    double q_trial) {
  if (state.opt.line_search_policy ==
      TIDE_OPTIM_LINE_SEARCH_POLICY_STATIC) {
    return TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_STATIC;
  }
  if (state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_ARMIJO_CUBIC ||
      state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_NONMONOTONE_ARMIJO) {
    return armijo ? TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_ARMIJO
                  : TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE;
  }
  if (state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_STRONG_WOLFE ||
      state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_MORE_THUENTE) {
    return (armijo && strong_curvature)
               ? TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_STRONG_WOLFE
               : TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE;
  }
  if (state.opt.line_search_policy ==
      TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG) {
    if (armijo && weak_curvature) {
      return TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_WEAK_WOLFE;
    }
    if (hager_zhang_approximate_decrease(state, f) && weak_curvature &&
        hager_zhang_upper_curvature(state, q_trial)) {
      return TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_APPROXIMATE_WOLFE;
    }
    return TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE;
  }
  return (armijo && weak_curvature)
             ? TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_WEAK_WOLFE
             : TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE;
}

bool update_alpha_hager_zhang(LbfgsState &state, double f, double q_trial,
                              bool decrease_failure,
                              bool nonfinite_trial) {
  return update_alpha_strong_wolfe(state, f, q_trial, decrease_failure,
                                   nonfinite_trial);
}

bool update_alpha_after_reject(LbfgsState &state, double f,
                               double q_trial, bool decrease_failure,
                               bool nonfinite_trial) {
  if (state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_ARMIJO_CUBIC ||
      state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_NONMONOTONE_ARMIJO) {
    return update_alpha_armijo_cubic(state, f, nonfinite_trial);
  }
  if (state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_STRONG_WOLFE ||
      state.opt.line_search_policy ==
          TIDE_OPTIM_LINE_SEARCH_POLICY_MORE_THUENTE) {
    return update_alpha_strong_wolfe(state, f, q_trial, decrease_failure,
                                     nonfinite_trial);
  }
  if (state.opt.line_search_policy ==
      TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG) {
    return update_alpha_hager_zhang(state, f, q_trial, decrease_failure,
                                    nonfinite_trial);
  }
  return update_alpha_legacy_weak_wolfe(state, decrease_failure);
}

int32_t finish_trust_region_trial(LbfgsState &state, double f,
                                  double const *g, bool nonfinite,
                                  double *x_request,
                                  tide_optim_report *report) {
  state.awaiting_trust_region_trial = false;
  state.last_trial_alpha = 1.0;
  state.last_trial_f = f;
  if (nonfinite) {
    note_nonfinite_trial(state);
    note_trust_region_failure(state,
                              TIDE_OPTIM_TRUST_REGION_FAILED_NONFINITE);
    state.trust_radius *= state.opt.trust_shrink;
    state.line_search_iter += 1;
    if (state.line_search_iter >= state.opt.max_line_search ||
        state.trust_radius <= state.opt.alpha_min) {
      state.trust_region_failure_count += 1;
      return done(state, TIDE_OPTIM_STATUS_TRUST_REGION_FAILED, report);
    }
    return start_inner_cg(state, x_request, report);
  }

  if (!finite_value(state.predicted_reduction) ||
      state.predicted_reduction <= 0.0) {
    note_trust_region_failure(
        state, TIDE_OPTIM_TRUST_REGION_FAILED_PREDICTED_REDUCTION);
    return done(state, TIDE_OPTIM_STATUS_TRUST_REGION_FAILED, report);
  }

  double const previous_f = state.base_f;
  state.actual_reduction = state.base_f - f;
  state.last_f_change = std::abs(previous_f - f);
  state.trust_ratio = state.actual_reduction / state.predicted_reduction;
  if (!finite_value(state.trust_ratio)) {
    note_trust_region_failure(
        state, TIDE_OPTIM_TRUST_REGION_FAILED_PREDICTED_REDUCTION);
    return done(state, TIDE_OPTIM_STATUS_TRUST_REGION_FAILED, report);
  }

  if (state.trust_ratio < 0.25) {
    state.trust_radius *= state.opt.trust_shrink;
  }

  double const trial_step_norm = norm2(state.direction);
  bool const accept = state.trust_ratio >= state.opt.trust_eta;
  if (!accept) {
    state.trust_region_status = TIDE_OPTIM_TRUST_REGION_REJECTED;
    state.trust_region_rejection_count += 1;
    state.line_search_iter += 1;
    if (state.line_search_iter >= state.opt.max_line_search ||
        state.trust_radius <= state.opt.alpha_min) {
      return done(state, TIDE_OPTIM_STATUS_TRUST_REGION_FAILED, report);
    }
    return start_inner_cg(state, x_request, report);
  }

  if (state.trust_ratio > 0.75 &&
      trial_step_norm >= 0.8 * state.trust_radius) {
    state.trust_radius =
        std::min(state.opt.max_trust_radius,
                 state.opt.trust_grow * state.trust_radius);
  }

  state.trust_region_status = TIDE_OPTIM_TRUST_REGION_ACCEPTED;
  state.trust_region_accept_count += 1;
  state.f = f;
  copy_from(g, state.g);
  state.x = state.trial_x;
  state.last_accepted_alpha = state.alpha;
  state.iter += 1;
  record_f_value(state, f);
  update_step_diagnostics_without_pair(state);

  if (gradient_converged(state)) {
    return done(state, TIDE_OPTIM_STATUS_CONVERGED_GRADIENT, report);
  }
  if (ftol_converged(state, previous_f)) {
    return done(state, TIDE_OPTIM_STATUS_CONVERGED_FTOL, report);
  }
  if (xtol_converged(state)) {
    return done(state, TIDE_OPTIM_STATUS_CONVERGED_XTOL, report);
  }
  if (max_eval_reached(state)) {
    return done(state, TIDE_OPTIM_STATUS_MAX_EVAL, report);
  }
  if (state.iter >= state.opt.max_iter) {
    return done(state, TIDE_OPTIM_STATUS_MAX_ITER, report);
  }
  return begin_line_search(state, x_request, report);
}

int32_t finish_accepted_trial(LbfgsState &state, double f, double const *g,
                              bool fallback_accept, double *x_request,
                              tide_optim_report *report) {
  (void)x_request;
  double const previous_f = state.base_f;
  state.last_f_change = std::abs(previous_f - f);
  state.f = f;
  copy_from(g, state.g);
  state.x = state.trial_x;
  state.last_accepted_alpha = state.alpha;
  state.iter += 1;
  record_f_value(state, f);
  state.fallback_accept = fallback_accept;
  state.awaiting_accepted_gradient = false;
  state.pending_fallback_accept = false;
  state.line_search_status =
      fallback_accept ? TIDE_OPTIM_LINE_SEARCH_ACCEPTED_DECREASE_AFTER_MAXLS
                      : TIDE_OPTIM_LINE_SEARCH_ACCEPTED;
  state.line_search_accept_count += 1;
  if (fallback_accept) {
    state.line_search_acceptance =
        TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_DECREASE_FALLBACK;
    state.warning_flags |= kWarnFallbackAccepted;
    state.line_search_fallback_accept_count += 1;
  }
  if (state.opt.direction_policy == TIDE_OPTIM_DIRECTION_LBFGS ||
      state.opt.direction_policy ==
          TIDE_OPTIM_DIRECTION_PRECONDITIONED_LBFGS) {
    store_pair(state, fallback_accept);
  } else {
    update_step_diagnostics_without_pair(state);
  }

  if (gradient_converged(state)) {
    return done(state, TIDE_OPTIM_STATUS_CONVERGED_GRADIENT, report);
  }
  if (ftol_converged(state, previous_f)) {
    return done(state, TIDE_OPTIM_STATUS_CONVERGED_FTOL, report);
  }
  if (xtol_converged(state)) {
    return done(state, TIDE_OPTIM_STATUS_CONVERGED_XTOL, report);
  }
  if (max_eval_reached(state)) {
    return done(state, TIDE_OPTIM_STATUS_MAX_EVAL, report);
  }
  if (state.iter >= state.opt.max_iter) {
    return done(state, TIDE_OPTIM_STATUS_MAX_ITER, report);
  }
  return begin_line_search(state, x_request, report);
}

int32_t request_gradient_for_accepted_trial(LbfgsState &state, double f,
                                            bool fallback_accept,
                                            double *x_request,
                                            tide_optim_report *report) {
  state.f = f;
  state.x = state.trial_x;
  state.fallback_accept = fallback_accept;
  state.awaiting_accepted_gradient = true;
  state.pending_fallback_accept = fallback_accept;
  state.line_search_status =
      fallback_accept ? TIDE_OPTIM_LINE_SEARCH_ACCEPTED_DECREASE_AFTER_MAXLS
                      : TIDE_OPTIM_LINE_SEARCH_ACCEPTED;
  if (fallback_accept) {
    state.line_search_acceptance =
        TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_DECREASE_FALLBACK;
    state.warning_flags |= kWarnFallbackAccepted;
  } else {
    state.line_search_acceptance = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_ARMIJO;
  }
  copy_to(state.trial_x, x_request);
  emit_report(state, TIDE_OPTIM_REQUEST_EVALUATE_FG, report);
  return TIDE_OPTIM_REQUEST_EVALUATE_FG;
}

} // namespace

static bool known_request_kind(int32_t request) {
  switch (request) {
  case TIDE_OPTIM_REQUEST_ERROR:
  case TIDE_OPTIM_REQUEST_EVALUATE_FG:
  case TIDE_OPTIM_REQUEST_DONE:
  case TIDE_OPTIM_REQUEST_EVALUATE_F:
  case TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER:
  case TIDE_OPTIM_REQUEST_EVALUATE_HV:
    return true;
  default:
    return false;
  }
}

static tide_optim_request_summary request_summary_from_values(int32_t request,
                                                        int64_t sequence,
                                                        bool valid) {
  tide::optim::RequestKind const kind =
      static_cast<tide::optim::RequestKind>(request);
  return tide_optim_request_summary{
      valid && known_request_kind(request) ? 1 : 0,
      request,
      sequence,
      tide::optim::name(kind),
      tide::optim::expected_evaluation(kind),
      tide::optim::required_fields(kind),
      tide::optim::accepted_mapping_keys(kind),
      tide::optim::requires_evaluation(kind) && known_request_kind(request) ? 1 : 0,
      kind == tide::optim::RequestKind::Error ? 1 : 0,
      kind == tide::optim::RequestKind::Done ? 1 : 0,
      kind == tide::optim::RequestKind::EvaluateF ||
              kind == tide::optim::RequestKind::EvaluateFG
          ? 1
          : 0,
      kind == tide::optim::RequestKind::EvaluateFG ? 1 : 0,
      kind == tide::optim::RequestKind::EvaluateFG ? 1 : 0,
      kind == tide::optim::RequestKind::ApplyPreconditioner ? 1 : 0,
      kind == tide::optim::RequestKind::EvaluateHv ? 1 : 0,
      kind == tide::optim::RequestKind::ApplyPreconditioner ||
              kind == tide::optim::RequestKind::EvaluateHv
          ? 1
          : 0,
  };
}

static char const *missing_fields_name(bool missing_value,
                                       bool missing_gradient,
                                       bool missing_vector) {
  if (missing_value && missing_gradient) {
    return "f,g";
  }
  if (missing_value) {
    return "f";
  }
  if (missing_gradient) {
    return "g";
  }
  if (missing_vector) {
    return "vector";
  }
  return "";
}

static char const *mismatched_fields_name(bool gradient_mismatch,
                                          bool vector_mismatch) {
  if (gradient_mismatch) {
    return "g";
  }
  if (vector_mismatch) {
    return "vector";
  }
  return "";
}

extern "C" tide_optim_evaluation_status tide_optim_validate_evaluation(
    int32_t request, int64_t request_sequence, int64_t expected_gradient_size,
    int64_t expected_vector_size, int32_t has_value, int32_t has_gradient,
    int64_t gradient_size, int32_t has_vector, int64_t vector_size) {
  tide::optim::RequestKind const kind =
      static_cast<tide::optim::RequestKind>(request);
  bool const known = known_request_kind(request);
  bool const needs_value =
      kind == tide::optim::RequestKind::EvaluateF ||
      kind == tide::optim::RequestKind::EvaluateFG;
  bool const needs_gradient = kind == tide::optim::RequestKind::EvaluateFG;
  bool const needs_vector =
      kind == tide::optim::RequestKind::ApplyPreconditioner ||
      kind == tide::optim::RequestKind::EvaluateHv;
  bool const requires_evaluation =
      known && tide::optim::requires_evaluation(kind);
  bool const has_value_bool = has_value != 0;
  bool const has_gradient_bool = has_gradient != 0;
  bool const has_vector_bool = has_vector != 0;
  bool const missing_value = needs_value && !has_value_bool;
  bool const missing_gradient = needs_gradient && !has_gradient_bool;
  bool const missing_vector = needs_vector && !has_vector_bool;
  bool const has_missing =
      missing_value || missing_gradient || missing_vector;
  bool const gradient_mismatch =
      needs_gradient && has_gradient_bool && expected_gradient_size > 0 &&
      gradient_size != expected_gradient_size;
  bool const vector_mismatch =
      needs_vector && has_vector_bool && expected_vector_size > 0 &&
      vector_size != expected_vector_size;
  bool const has_mismatch = gradient_mismatch || vector_mismatch;
  bool const satisfied =
      requires_evaluation && !has_missing && !has_mismatch;

  return tide_optim_evaluation_status{
      request,
      request_sequence,
      tide::optim::name(kind),
      tide::optim::expected_evaluation(kind),
      tide::optim::required_fields(kind),
      tide::optim::accepted_mapping_keys(kind),
      requires_evaluation ? 1 : 0,
      has_value_bool ? 1 : 0,
      has_gradient_bool ? 1 : 0,
      has_vector_bool ? 1 : 0,
      has_gradient_bool ? gradient_size : 0,
      has_vector_bool ? vector_size : 0,
      needs_gradient ? expected_gradient_size : 0,
      needs_vector ? expected_vector_size : 0,
      missing_value ? 1 : 0,
      missing_gradient ? 1 : 0,
      missing_vector ? 1 : 0,
      has_missing ? 1 : 0,
      missing_fields_name(missing_value, missing_gradient, missing_vector),
      gradient_mismatch ? 1 : 0,
      vector_mismatch ? 1 : 0,
      has_mismatch ? 1 : 0,
      mismatched_fields_name(gradient_mismatch, vector_mismatch),
      satisfied ? 1 : 0,
      satisfied ? 1 : 0,
  };
}

static int64_t expected_gradient_size_for_report(tide_optim_report const *report) {
  if (report == nullptr || report->request != TIDE_OPTIM_REQUEST_EVALUATE_FG) {
    return 0;
  }
  return report->n;
}

static int64_t expected_vector_size_for_report(tide_optim_report const *report) {
  if (report == nullptr) {
    return 0;
  }
  if (report->request == TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER ||
      report->request == TIDE_OPTIM_REQUEST_EVALUATE_HV) {
    return report->n;
  }
  return 0;
}

extern "C" tide_optim_evaluation_status
tide_optim_validate_report_evaluation(tide_optim_report const *report,
                                      int32_t has_value,
                                      int32_t has_gradient,
                                      int64_t gradient_size,
                                      int32_t has_vector,
                                      int64_t vector_size) {
  if (report == nullptr) {
    return tide_optim_validate_evaluation(TIDE_OPTIM_REQUEST_ERROR, 0, 0, 0,
                                          has_value, has_gradient,
                                          gradient_size, has_vector,
                                          vector_size);
  }
  return tide_optim_validate_evaluation(
      report->request, report->request_sequence,
      expected_gradient_size_for_report(report),
      expected_vector_size_for_report(report), has_value, has_gradient,
      gradient_size, has_vector, vector_size);
}

extern "C" tide_optim_report_summary
tide_optim_summarize_report(tide_optim_report const *report) {
  tide::optim::ReportView const view{report};
  tide::optim::Status const status = view.status();
  tide::optim::RequestKind const request = view.request_kind();
  bool const success = status == tide::optim::Status::ConvergedGradient ||
                       status == tide::optim::Status::ConvergedFtol ||
                       status == tide::optim::Status::ConvergedXtol;
  bool const stopped = status != tide::optim::Status::Running;
  bool const user_stopped = status == tide::optim::Status::UserStopped;
  return tide_optim_report_summary{
      report != nullptr ? 1 : 0,
      static_cast<int32_t>(request),
      static_cast<int32_t>(status),
      view.request_name(),
      tide::optim::expected_evaluation(request),
      tide::optim::required_fields(request),
      tide::optim::accepted_mapping_keys(request),
      tide::optim::requires_evaluation(request) ? 1 : 0,
      view.status_name(),
      view.reason(),
      view.failure_reason(),
      view.method_name(),
      view.line_search_status_name(),
      view.line_search_acceptance_name(),
      view.pair_status_name(),
      view.direction_policy_name(),
      view.line_search_policy_name(),
      view.globalization_policy_name(),
      view.preconditioner_status_name(),
      view.inner_status_name(),
      view.trust_region_status_name(),
      view.done() ? 1 : 0,
      success ? 1 : 0,
      stopped ? 1 : 0,
      stopped && !success && !user_stopped ? 1 : 0,
      user_stopped ? 1 : 0,
      request == tide::optim::RequestKind::EvaluateF ||
              request == tide::optim::RequestKind::EvaluateFG
          ? 1
          : 0,
      request == tide::optim::RequestKind::EvaluateFG ? 1 : 0,
      request == tide::optim::RequestKind::EvaluateFG ? 1 : 0,
      request == tide::optim::RequestKind::ApplyPreconditioner ? 1 : 0,
      request == tide::optim::RequestKind::EvaluateHv ? 1 : 0,
      request == tide::optim::RequestKind::ApplyPreconditioner ||
              request == tide::optim::RequestKind::EvaluateHv
          ? 1
          : 0,
      view.line_search_failed() ? 1 : 0,
      status == tide::optim::Status::InnerCgFailed ? 1 : 0,
      status == tide::optim::Status::TrustRegionFailed ? 1 : 0,
      status == tide::optim::Status::Nonfinite ? 1 : 0,
      view.warning_flags() != 0 ? 1 : 0,
      view.warning_flags(),
      view.request_sequence(),
      view.n(),
      view.expected_gradient_size(),
      view.expected_vector_size(),
      view.iter(),
      view.n_f(),
      view.n_g(),
      view.n_hvp(),
      view.n_prec(),
      view.f(),
      view.grad_norm(),
      view.projected_grad_norm(),
      view.alpha(),
  };
}

extern "C" tide_optim_request_summary
tide_optim_summarize_request(int32_t request) {
  return request_summary_from_values(request, 0, true);
}

extern "C" tide_optim_request_summary
tide_optim_summarize_report_request(tide_optim_report const *report) {
  if (report == nullptr) {
    return request_summary_from_values(TIDE_OPTIM_REQUEST_ERROR, 0, false);
  }
  return request_summary_from_values(report->request, report->request_sequence,
                                     true);
}

extern "C" tide_optim_session_snapshot
tide_optim_get_session_snapshot(void *handle) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr) {
    return tide_optim_session_snapshot{
        0, 0, 0, 0, 0, "INVALID", 0, tide_optim_summarize_report(nullptr),
        0, 0, 0, 0, 0};
  }

  int32_t const request = current_request_kind(*state);
  tide_optim_report report{};
  fill_report(*state, request, &report);
  bool const started = state->initialized;
  bool const done = started && state->status != TIDE_OPTIM_STATUS_RUNNING;
  bool const running = started && state->status == TIDE_OPTIM_STATUS_RUNNING;
  return tide_optim_session_snapshot{
      1,
      started ? 1 : 0,
      done ? 1 : 0,
      running ? 1 : 0,
      state->has_bounds ? 1 : 0,
      session_state_name(state),
      state->opt.n,
      tide_optim_summarize_report(&report),
      request == TIDE_OPTIM_REQUEST_EVALUATE_FG ? 1 : 0,
      request == TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER ? 1 : 0,
      request == TIDE_OPTIM_REQUEST_EVALUATE_HV ? 1 : 0,
      state->awaiting_trust_region_trial ? 1 : 0,
      state->awaiting_accepted_gradient ? 1 : 0,
  };
}

extern "C" char const *tide_optim_request_kind_name(int32_t request) {
  return tide::optim::name(static_cast<tide::optim::RequestKind>(request));
}

extern "C" char const *
tide_optim_request_expected_evaluation(int32_t request) {
  return tide::optim::expected_evaluation(
      static_cast<tide::optim::RequestKind>(request));
}

extern "C" char const *tide_optim_request_required_fields(int32_t request) {
  return tide::optim::required_fields(
      static_cast<tide::optim::RequestKind>(request));
}

extern "C" char const *
tide_optim_request_accepted_mapping_keys(int32_t request) {
  return tide::optim::accepted_mapping_keys(
      static_cast<tide::optim::RequestKind>(request));
}

extern "C" int32_t tide_optim_request_requires_evaluation(int32_t request) {
  return tide::optim::requires_evaluation(
             static_cast<tide::optim::RequestKind>(request))
             ? 1
             : 0;
}

extern "C" int32_t tide_optim_request_is_error(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_ERROR ? 1 : 0;
}

extern "C" int32_t tide_optim_request_is_done(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_DONE ? 1 : 0;
}

extern "C" int32_t tide_optim_request_needs_value(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_EVALUATE_F ||
                 request == TIDE_OPTIM_REQUEST_EVALUATE_FG
             ? 1
             : 0;
}

extern "C" int32_t tide_optim_request_needs_gradient(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_EVALUATE_FG ? 1 : 0;
}

extern "C" int32_t
tide_optim_request_needs_value_gradient(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_EVALUATE_FG ? 1 : 0;
}

extern "C" int32_t
tide_optim_request_needs_preconditioner(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER ? 1 : 0;
}

extern "C" int32_t
tide_optim_request_needs_hessian_vector(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_EVALUATE_HV ? 1 : 0;
}

extern "C" int32_t
tide_optim_request_needs_vector_result(int32_t request) {
  return request == TIDE_OPTIM_REQUEST_APPLY_PRECONDITIONER ||
                 request == TIDE_OPTIM_REQUEST_EVALUATE_HV
             ? 1
             : 0;
}

extern "C" char const *tide_optim_status_name(int32_t status) {
  return tide::optim::name(static_cast<tide::optim::Status>(status));
}

extern "C" char const *tide_optim_line_search_status_name(int32_t status) {
  return tide::optim::name(
      static_cast<tide::optim::LineSearchStatus>(status));
}

extern "C" char const *tide_optim_line_search_policy_name(int32_t policy) {
  return tide::optim::name(static_cast<tide::optim::LineSearchPolicy>(policy));
}

extern "C" char const *tide_optim_alpha_guess_policy_name(int32_t policy) {
  return tide::optim::name(static_cast<tide::optim::AlphaGuessPolicy>(policy));
}

extern "C" char const *tide_optim_stopping_policy_name(int32_t policy) {
  return tide::optim::name(static_cast<tide::optim::StoppingPolicy>(policy));
}

extern "C" char const *
tide_optim_line_search_acceptance_name(int32_t acceptance) {
  return tide::optim::name(
      static_cast<tide::optim::LineSearchAcceptance>(acceptance));
}

extern "C" char const *tide_optim_pair_status_name(int32_t status) {
  return tide::optim::name(static_cast<tide::optim::PairStatus>(status));
}

extern "C" char const *tide_optim_lbfgs_update_policy_name(int32_t policy) {
  return tide::optim::name(static_cast<tide::optim::LbfgsUpdatePolicy>(policy));
}

extern "C" char const *tide_optim_bounds_strategy_name(int32_t strategy) {
  return tide::optim::name(static_cast<tide::optim::BoundsStrategy>(strategy));
}

extern "C" char const *tide_optim_cost_model_name(int32_t cost_model) {
  return tide::optim::name(static_cast<tide::optim::CostModel>(cost_model));
}

extern "C" char const *tide_optim_direction_policy_name(int32_t policy) {
  return tide::optim::name(static_cast<tide::optim::DirectionPolicy>(policy));
}

extern "C" char const *tide_optim_direction_method_name(int32_t policy) {
  return tide::optim::method_name(
      static_cast<tide::optim::DirectionPolicy>(policy));
}

extern "C" char const *tide_optim_nlcg_beta_policy_name(int32_t policy) {
  return tide::optim::name(static_cast<tide::optim::NlcgBetaPolicy>(policy));
}

extern "C" char const *tide_optim_direction_status_name(int32_t status) {
  return tide::optim::name(static_cast<tide::optim::DirectionStatus>(status));
}

extern "C" char const *
tide_optim_preconditioner_status_name(int32_t status) {
  return tide::optim::name(
      static_cast<tide::optim::PreconditionerStatus>(status));
}

extern "C" char const *tide_optim_inner_cg_status_name(int32_t status) {
  return tide::optim::name(static_cast<tide::optim::InnerCgStatus>(status));
}

extern "C" char const *tide_optim_globalization_policy_name(int32_t policy) {
  return tide::optim::name(static_cast<tide::optim::GlobalizationPolicy>(policy));
}

extern "C" char const *tide_optim_trust_region_status_name(int32_t status) {
  return tide::optim::name(static_cast<tide::optim::TrustRegionStatus>(status));
}

extern "C" char const *tide_optim_warning_flag_name(int32_t flag) {
  return tide::optim::name(static_cast<tide::optim::WarningFlag>(flag));
}

extern "C" char const *
tide_optim_options_validation_code_name(int32_t code) {
  return tide::optim::name(
      static_cast<tide::optim::OptionsValidationCode>(code));
}

extern "C" tide_optim_options_validation
tide_optim_lbfgs_validate_options(tide_optim_lbfgs_options const *options) {
  return validate_options_detail(options);
}

extern "C" void *
tide_optim_lbfgs_create(tide_optim_lbfgs_options const *options) {
  if (options == nullptr || !valid_options(*options)) {
    return nullptr;
  }
  try {
    return new LbfgsState(*options);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void tide_optim_lbfgs_destroy(void *handle) {
  delete as_state(handle);
}

extern "C" int32_t tide_optim_lbfgs_set_bounds(void *handle, double const *lb,
                                                double const *ub) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr) {
    return TIDE_OPTIM_STATUS_INVALID_ARGUMENT;
  }
  if (lb == nullptr || ub == nullptr) {
    state->has_bounds = false;
    state->lb.clear();
    state->ub.clear();
    return TIDE_OPTIM_STATUS_RUNNING;
  }
  if (!finite_vector(lb, state->opt.n) || !finite_vector(ub, state->opt.n)) {
    return TIDE_OPTIM_STATUS_NONFINITE;
  }
  state->lb.assign(lb, lb + state->opt.n);
  state->ub.assign(ub, ub + state->opt.n);
  for (int64_t i = 0; i < state->opt.n; ++i) {
    if (state->lb[i] > state->ub[i]) {
      state->has_bounds = false;
      return TIDE_OPTIM_STATUS_INVALID_ARGUMENT;
    }
  }
  state->has_bounds = true;
  return TIDE_OPTIM_STATUS_RUNNING;
}

extern "C" int32_t tide_optim_lbfgs_start(void *handle, double const *x,
                                           double f, double const *g,
                                           double *x_request,
                                           tide_optim_report *report) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr || x_request == nullptr ||
      !finite_vector(x, state ? state->opt.n : 0) ||
      !finite_vector(g, state ? state->opt.n : 0) || !finite_value(f)) {
    if (state != nullptr) {
      return done(*state, TIDE_OPTIM_STATUS_INVALID_ARGUMENT, report);
    }
    return TIDE_OPTIM_REQUEST_ERROR;
  }
  copy_from(x, state->x);
  copy_from(g, state->g);
  state->f = f;
  state->initial_f = f;
  state->n_f = 1;
  state->n_g = 1;
  state->n_hvp = 0;
  state->n_prec = 0;
  state->line_search_accept_count = 0;
  state->line_search_rejection_count = 0;
  state->line_search_failure_count = 0;
  state->line_search_fallback_accept_count = 0;
  state->nonfinite_trial_count = 0;
  state->pair_skip_count = 0;
  state->pair_stored_count = 0;
  state->pair_regularized_count = 0;
  state->preconditioner_skip_count = 0;
  state->inner_warning_count = 0;
  state->trust_region_accept_count = 0;
  state->trust_region_rejection_count = 0;
  state->trust_region_failure_count = 0;
  state->iter = 0;
  state->hist_count = 0;
  state->hist_start = 0;
  state->f_window_count = 0;
  state->f_window_start = 0;
  state->last_gamma = 1.0;
  state->last_f_change = 0.0;
  state->last_direction_beta = 0.0;
  state->last_preconditioner_dot = 0.0;
  state->last_accepted_alpha = state->opt.initial_step;
  state->direction_status = TIDE_OPTIM_DIRECTION_STATUS_INITIAL;
  state->preconditioner_status = TIDE_OPTIM_PRECONDITIONER_NONE;
  state->awaiting_accepted_gradient = false;
  state->pending_fallback_accept = false;
  state->awaiting_preconditioner = false;
  state->awaiting_inner_preconditioner = false;
  state->awaiting_hvp = false;
  state->awaiting_trust_region_trial = false;
  state->inner_status = TIDE_OPTIM_INNER_CG_NONE;
  state->inner_iter = 0;
  state->inner_residual_norm = 0.0;
  state->inner_forcing_tolerance = 0.0;
  state->inner_curvature = 0.0;
  state->trust_radius = state->opt.initial_trust_radius;
  state->trust_region_status = TIDE_OPTIM_TRUST_REGION_NONE;
  state->trust_ratio = 0.0;
  state->predicted_reduction = 0.0;
  state->actual_reduction = 0.0;
  state->initial_grad_norm = convergence_grad_norm(*state);
  state->initialized = true;
  state->status = TIDE_OPTIM_STATUS_RUNNING;
  record_f_value(*state, f);
  return begin_line_search(*state, x_request, report);
}

extern "C" int32_t tide_optim_lbfgs_tell(void *handle, double f,
                                          double const *g, double *x_request,
                                          tide_optim_report *report) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr || x_request == nullptr || !state->initialized ||
      state->awaiting_preconditioner ||
      state->awaiting_inner_preconditioner || state->awaiting_hvp) {
    return TIDE_OPTIM_REQUEST_ERROR;
  }
  state->n_f += 1;
  state->n_g += 1;
  bool const finite_g = finite_vector(g, state->opt.n);
  state->q_trial =
      finite_g ? dot_ptr(state->direction, g)
               : std::numeric_limits<double>::quiet_NaN();

  bool const nonfinite = !finite_value(f) || !finite_g;
  state->last_trial_alpha = state->alpha;
  state->last_trial_f = f;
  state->last_line_search_reference = armijo_reference(*state);
  state->last_line_search_armijo_rhs =
      state->last_line_search_reference +
      state->opt.c1 * state->alpha * state->q0;
  if (state->awaiting_trust_region_trial && uses_trust_region(*state)) {
    return finish_trust_region_trial(*state, f, g, nonfinite, x_request,
                                     report);
  }
  state->awaiting_trust_region_trial = false;
  if (state->awaiting_accepted_gradient) {
    if (nonfinite) {
      note_nonfinite_trial(*state);
      note_line_search_failure(*state,
                               TIDE_OPTIM_LINE_SEARCH_FAILED_NONFINITE);
      state->awaiting_accepted_gradient = false;
      state->pending_fallback_accept = false;
      return done(*state, TIDE_OPTIM_STATUS_NONFINITE, report);
    }
    return finish_accepted_trial(*state, f, g, state->pending_fallback_accept,
                                 x_request, report);
  }

  if (nonfinite) {
    note_nonfinite_trial(*state);
    state->line_search_status = TIDE_OPTIM_LINE_SEARCH_FAILED_NONFINITE;
    if (state->opt.line_search_policy == TIDE_OPTIM_LINE_SEARCH_POLICY_STATIC) {
      return done(*state, TIDE_OPTIM_STATUS_NONFINITE, report);
    }
  } else {
    bool const armijo = f <= state->last_line_search_armijo_rhs;
    bool const weak_curvature = state->q_trial >= state->opt.c2 * state->q0;
    bool const strong_curvature =
        std::abs(state->q_trial) <= -state->opt.c2 * state->q0;
    int32_t const acceptance =
        line_search_acceptance_kind(*state, armijo, weak_curvature,
                                    strong_curvature, f, state->q_trial);
    if (acceptance != TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE) {
      state->line_search_acceptance = acceptance;
      return finish_accepted_trial(*state, f, g, false, x_request, report);
    }
    state->line_search_acceptance = TIDE_OPTIM_LINE_SEARCH_ACCEPTANCE_NONE;
    bool const decrease_ok =
        armijo || (state->opt.line_search_policy ==
                       TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG &&
                   hager_zhang_approximate_decrease(*state, f));
    note_line_search_rejection(
        *state,
        decrease_ok && !((state->opt.line_search_policy ==
                              TIDE_OPTIM_LINE_SEARCH_POLICY_STRONG_WOLFE ||
                          state->opt.line_search_policy ==
                              TIDE_OPTIM_LINE_SEARCH_POLICY_MORE_THUENTE) &&
                         state->line_search_iter > 0 &&
                         f >= state->strong_wolfe_prev_f)
            ? TIDE_OPTIM_LINE_SEARCH_REJECTED_CURVATURE
            : TIDE_OPTIM_LINE_SEARCH_REJECTED_ARMIJO);
  }

  if (state->line_search_iter >= state->opt.max_line_search) {
    if (!nonfinite && state->opt.accept_decrease_after_maxls &&
        f < state->base_f) {
      return finish_accepted_trial(*state, f, g, true, x_request, report);
    }
    note_line_search_failure(*state, TIDE_OPTIM_LINE_SEARCH_FAILED_MAXLS);
    return done(*state, TIDE_OPTIM_STATUS_LINE_SEARCH_FAILED, report);
  }
  if (max_eval_reached(*state)) {
    return done(*state, TIDE_OPTIM_STATUS_MAX_EVAL, report);
  }

  bool const decrease_failure =
      nonfinite ||
      (state->opt.line_search_policy == TIDE_OPTIM_LINE_SEARCH_POLICY_HAGER_ZHANG
           ? !hager_zhang_approximate_decrease(*state, f)
           : f > state->last_line_search_armijo_rhs);
  if (!update_alpha_after_reject(*state, f, state->q_trial, decrease_failure,
                                 nonfinite)) {
    note_line_search_failure(*state,
                             TIDE_OPTIM_LINE_SEARCH_FAILED_ALPHA_BOUNDS);
    return done(*state, TIDE_OPTIM_STATUS_LINE_SEARCH_FAILED, report);
  }
  state->line_search_iter += 1;
  make_trial(*state);
  copy_to(state->trial_x, x_request);
  int32_t const request = line_search_request_kind(*state);
  emit_report(*state, request, report);
  return request;
}

extern "C" int32_t tide_optim_lbfgs_tell_value(void *handle, double f,
                                                double *x_request,
                                                tide_optim_report *report) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr || x_request == nullptr || !state->initialized ||
      state->awaiting_preconditioner ||
      state->awaiting_inner_preconditioner || state->awaiting_hvp ||
      state->awaiting_trust_region_trial ||
      state->awaiting_accepted_gradient ||
      (state->opt.line_search_policy !=
           TIDE_OPTIM_LINE_SEARCH_POLICY_ARMIJO_CUBIC &&
       state->opt.line_search_policy !=
           TIDE_OPTIM_LINE_SEARCH_POLICY_NONMONOTONE_ARMIJO)) {
    return TIDE_OPTIM_REQUEST_ERROR;
  }

  state->n_f += 1;
  state->q_trial = std::numeric_limits<double>::quiet_NaN();
  state->last_trial_alpha = state->alpha;
  state->last_trial_f = f;
  state->last_line_search_reference = armijo_reference(*state);
  state->last_line_search_armijo_rhs =
      state->last_line_search_reference +
      state->opt.c1 * state->alpha * state->q0;

  bool const nonfinite = !finite_value(f);
  if (nonfinite) {
    note_nonfinite_trial(*state);
    state->line_search_status = TIDE_OPTIM_LINE_SEARCH_FAILED_NONFINITE;
  } else {
    bool const armijo = f <= state->last_line_search_armijo_rhs;
    if (armijo) {
      return request_gradient_for_accepted_trial(*state, f, false, x_request,
                                                 report);
    }
    note_line_search_rejection(*state,
                               TIDE_OPTIM_LINE_SEARCH_REJECTED_ARMIJO);
  }

  if (state->line_search_iter >= state->opt.max_line_search) {
    if (!nonfinite && state->opt.accept_decrease_after_maxls &&
        f < state->base_f) {
      return request_gradient_for_accepted_trial(*state, f, true, x_request,
                                                 report);
    }
    note_line_search_failure(*state, TIDE_OPTIM_LINE_SEARCH_FAILED_MAXLS);
    return done(*state, TIDE_OPTIM_STATUS_LINE_SEARCH_FAILED, report);
  }
  if (max_eval_reached(*state)) {
    return done(*state, TIDE_OPTIM_STATUS_MAX_EVAL, report);
  }

  bool const armijo_failure =
      nonfinite || f > state->last_line_search_armijo_rhs;
  if (!update_alpha_after_reject(*state, f, state->q_trial, armijo_failure,
                                 nonfinite)) {
    note_line_search_failure(*state,
                             TIDE_OPTIM_LINE_SEARCH_FAILED_ALPHA_BOUNDS);
    return done(*state, TIDE_OPTIM_STATUS_LINE_SEARCH_FAILED, report);
  }
  state->line_search_iter += 1;
  make_trial(*state);
  copy_to(state->trial_x, x_request);
  emit_report(*state, TIDE_OPTIM_REQUEST_EVALUATE_F, report);
  return TIDE_OPTIM_REQUEST_EVALUATE_F;
}

extern "C" int32_t tide_optim_lbfgs_tell_preconditioner(
    void *handle, double const *z, double *x_request,
    tide_optim_report *report) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr || x_request == nullptr || !state->initialized ||
      (!state->awaiting_preconditioner &&
       !state->awaiting_inner_preconditioner)) {
    return TIDE_OPTIM_REQUEST_ERROR;
  }

  state->n_prec += 1;
  bool const inner_preconditioner = state->awaiting_inner_preconditioner;
  state->awaiting_preconditioner = false;
  state->awaiting_inner_preconditioner = false;

  bool const finite_z = finite_vector(z, state->opt.n);
  double z_norm_sq = 0.0;
  if (finite_z) {
    for (int64_t i = 0; i < state->opt.n; ++i) {
      z_norm_sq += z[i] * z[i];
    }
  }
  double const input_norm = norm2(state->preconditioner_input);
  double const z_norm = std::sqrt(z_norm_sq);
  double const pz = finite_z ? dot_ptr(state->preconditioner_input, z)
                             : std::numeric_limits<double>::quiet_NaN();
  state->last_preconditioner_dot = pz;
  double const min_pz = state->opt.curvature_eps * input_norm * z_norm;

  if (!finite_z || !finite_value(pz) || !finite_value(z_norm)) {
    note_preconditioner_skip(
        *state, TIDE_OPTIM_PRECONDITIONER_SKIPPED_NONFINITE);
    state->preconditioned_q = state->preconditioner_input;
  } else if (pz <= min_pz) {
    note_preconditioner_skip(
        *state, TIDE_OPTIM_PRECONDITIONER_SKIPPED_NOT_POSITIVE);
    state->preconditioned_q = state->preconditioner_input;
  } else {
    state->preconditioner_status = TIDE_OPTIM_PRECONDITIONER_APPLIED;
    copy_from(z, state->preconditioned_q);
  }

  if (inner_preconditioner) {
    if (state->preconditioner_status == TIDE_OPTIM_PRECONDITIONER_APPLIED) {
      state->inner_z = state->preconditioned_q;
    } else {
      state->inner_status = TIDE_OPTIM_INNER_CG_PRECONDITIONER_BREAKDOWN;
      note_inner_warning(*state);
      state->inner_z = state->inner_r;
    }
    bool const first_direction = state->inner_iter == 0;
    return continue_inner_cg_after_z(*state, first_direction, x_request,
                                     report);
  }

  lbfgs_second_loop(*state, state->preconditioned_q);
  return begin_trial_after_direction(*state, x_request, report);
}

extern "C" int32_t tide_optim_lbfgs_tell_hessian_vector(
    void *handle, double const *hv, double *x_request,
    tide_optim_report *report) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr || x_request == nullptr || !state->initialized ||
      !state->awaiting_hvp) {
    return TIDE_OPTIM_REQUEST_ERROR;
  }
  return continue_inner_cg_after_hvp(*state, hv, x_request, report);
}

extern "C" int32_t tide_optim_lbfgs_current_x(void *handle, double *x_out) {
  LbfgsState *state = as_state(handle);
  if (state == nullptr || x_out == nullptr || !state->initialized) {
    return TIDE_OPTIM_STATUS_INVALID_ARGUMENT;
  }
  copy_to(state->x, x_out);
  return state->status;
}

extern "C" tide_optim_options_validation
tide_optim_validate_options(tide_optim_options const *options) {
  return tide_optim_lbfgs_validate_options(options);
}

extern "C" tide_optim_resolved_policies
tide_optim_resolve_policies(tide_optim_options const *options) {
  return resolved_policies_detail(options);
}

extern "C" void *tide_optim_create(tide_optim_options const *options) {
  return tide_optim_lbfgs_create(options);
}

extern "C" void tide_optim_destroy(void *handle) {
  tide_optim_lbfgs_destroy(handle);
}

extern "C" int32_t tide_optim_set_bounds(void *handle, double const *lb,
                                          double const *ub) {
  return tide_optim_lbfgs_set_bounds(handle, lb, ub);
}

extern "C" int32_t tide_optim_start(void *handle, double const *x, double f,
                                     double const *g, double *x_request,
                                     tide_optim_report *report) {
  return tide_optim_lbfgs_start(handle, x, f, g, x_request, report);
}

extern "C" int32_t tide_optim_tell(void *handle, double f, double const *g,
                                    double *x_request,
                                    tide_optim_report *report) {
  return tide_optim_lbfgs_tell(handle, f, g, x_request, report);
}

extern "C" int32_t tide_optim_tell_value(void *handle, double f,
                                          double *x_request,
                                          tide_optim_report *report) {
  return tide_optim_lbfgs_tell_value(handle, f, x_request, report);
}

extern "C" int32_t tide_optim_tell_preconditioner(
    void *handle, double const *z, double *x_request,
    tide_optim_report *report) {
  return tide_optim_lbfgs_tell_preconditioner(handle, z, x_request, report);
}

extern "C" int32_t tide_optim_tell_hessian_vector(
    void *handle, double const *hv, double *x_request,
    tide_optim_report *report) {
  return tide_optim_lbfgs_tell_hessian_vector(handle, hv, x_request, report);
}

extern "C" int32_t tide_optim_current_x(void *handle, double *x_out) {
  return tide_optim_lbfgs_current_x(handle, x_out);
}
