/*
 * Maxwell 3D CPU backend implementation (forward path).
 *
 * This implementation mirrors the 3D Python backend update equations
 * (staggered-grid FDTD + CPML) for inference use.
 *
 * Notes:
 * - forward_with_storage currently reuses forward behavior.
 * - backward implements the native 3D adjoint used by the autograd path.
 */

#undef DIFFZ1
#undef DIFFY1
#undef DIFFX1
#undef DIFFZH1
#undef DIFFYH1
#undef DIFFXH1
#undef FD_PAD
#undef DIFFZ1_ADJ
#undef DIFFY1_ADJ
#undef DIFFX1_ADJ
#undef DIFFZH1_ADJ
#undef DIFFYH1_ADJ
#undef DIFFXH1_ADJ

#ifdef STAGGERED_GRID_H
#undef STAGGERED_GRID_H
#endif
#define TIDE_STAGGERED_GRID_3D 1
#include "staggered_grid.h"
#undef TIDE_STAGGERED_GRID_3D

namespace FUNC(Inst) {
template <typename T> static inline T tide_max(T a, T b) {
  return a > b ? a : b;
}

template <typename T> static inline T tide_min(T a, T b) {
  return a < b ? a : b;
}

/* Vacuum permittivity (F/m): convert dL/d(epsilon_abs) -> dL/d(epsilon_r). */
constexpr TIDE_DTYPE kEp0 = (TIDE_DTYPE)8.8541878128e-12;

#define IDX(z, y, x) (((z) * ny + (y)) * nx + (x))
#define IDX_SHOT(shot, z, y, x) ((shot) * shot_numel + IDX(z, y, x))

#define EX(dz, dy, dx) ex[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define EY(dz, dy, dx) ey[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define EZ(dz, dy, dx) ez[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define HX(dz, dy, dx) hx[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define HY(dz, dy, dx) hy[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define HZ(dz, dy, dx) hz[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define DEX(dz, dy, dx) dex[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DEY(dz, dy, dx) dey[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DEZ(dz, dy, dx) dez[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DHX(dz, dy, dx) dhx[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DHY(dz, dy, dx) dhy[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DHZ(dz, dy, dx) dhz[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define CA_AT(dz, dy, dx) \
  (ca_batched ? ca[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))] \
              : ca[IDX(z + (dz), y + (dy), x + (dx))])
#define CB_AT(dz, dy, dx) \
  (cb_batched ? cb[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))] \
              : cb[IDX(z + (dz), y + (dy), x + (dx))])
#define CQ_AT(dz, dy, dx) \
  (cq_batched ? cq[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))] \
              : cq[IDX(z + (dz), y + (dy), x + (dx))])

#define M_HZ_Y(dz, dy, dx) m_hz_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HY_Z(dz, dy, dx) m_hy_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HX_Z(dz, dy, dx) m_hx_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HZ_X(dz, dy, dx) m_hz_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HY_X(dz, dy, dx) m_hy_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HX_Y(dz, dy, dx) m_hx_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define M_EY_Z(dz, dy, dx) m_ey_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EZ_Y(dz, dy, dx) m_ez_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EZ_X(dz, dy, dx) m_ez_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EX_Z(dz, dy, dx) m_ex_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EX_Y(dz, dy, dx) m_ex_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EY_X(dz, dy, dx) m_ey_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define DM_HZ_Y(dz, dy, dx) \
  dm_hz_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_HY_Z(dz, dy, dx) \
  dm_hy_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_HX_Z(dz, dy, dx) \
  dm_hx_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_HZ_X(dz, dy, dx) \
  dm_hz_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_HY_X(dz, dy, dx) \
  dm_hy_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_HX_Y(dz, dy, dx) \
  dm_hx_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define DM_EY_Z(dz, dy, dx) \
  dm_ey_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_EZ_Y(dz, dy, dx) \
  dm_ez_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_EZ_X(dz, dy, dx) \
  dm_ez_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_EX_Z(dz, dy, dx) \
  dm_ex_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_EX_Y(dz, dy, dx) \
  dm_ex_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define DM_EY_X(dz, dy, dx) \
  dm_ey_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define DCA_AT(dz, dy, dx) \
  dca[IDX(z + (dz), y + (dy), x + (dx))]
#define DCB_AT(dz, dy, dx) \
  dcb[IDX(z + (dz), y + (dy), x + (dx))]

#define LAMBDA_EX(dz, dy, dx) \
  lambda_ex[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LAMBDA_EY(dz, dy, dx) \
  lambda_ey[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LAMBDA_EZ(dz, dy, dx) \
  lambda_ez[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LAMBDA_HX(dz, dy, dx) \
  lambda_hx[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LAMBDA_HY(dz, dy, dx) \
  lambda_hy[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LAMBDA_HZ(dz, dy, dx) \
  lambda_hz[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define M_LAMBDA_EY_Z(dz, dy, dx) \
  m_lambda_ey_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_EZ_Y(dz, dy, dx) \
  m_lambda_ez_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_EZ_X(dz, dy, dx) \
  m_lambda_ez_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_EX_Z(dz, dy, dx) \
  m_lambda_ex_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_EX_Y(dz, dy, dx) \
  m_lambda_ex_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_EY_X(dz, dy, dx) \
  m_lambda_ey_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define M_LAMBDA_HZ_Y(dz, dy, dx) \
  m_lambda_hz_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_HY_Z(dz, dy, dx) \
  m_lambda_hy_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_HX_Z(dz, dy, dx) \
  m_lambda_hx_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_HZ_X(dz, dy, dx) \
  m_lambda_hz_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_HY_X(dz, dy, dx) \
  m_lambda_hy_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_LAMBDA_HX_Y(dz, dy, dx) \
  m_lambda_hx_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define ETA_EX(dz, dy, dx) \
  eta_ex[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define ETA_EY(dz, dy, dx) \
  eta_ey[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define ETA_EZ(dz, dy, dx) \
  eta_ez[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define ETA_HX(dz, dy, dx) \
  eta_hx[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define ETA_HY(dz, dy, dx) \
  eta_hy[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define ETA_HZ(dz, dy, dx) \
  eta_hz[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define M_ETA_EY_Z(dz, dy, dx) \
  m_eta_ey_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_EZ_Y(dz, dy, dx) \
  m_eta_ez_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_EZ_X(dz, dy, dx) \
  m_eta_ez_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_EX_Z(dz, dy, dx) \
  m_eta_ex_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_EX_Y(dz, dy, dx) \
  m_eta_ex_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_EY_X(dz, dy, dx) \
  m_eta_ey_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

#define M_ETA_HZ_Y(dz, dy, dx) \
  m_eta_hz_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_HY_Z(dz, dy, dx) \
  m_eta_hy_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_HX_Z(dz, dy, dx) \
  m_eta_hx_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_HZ_X(dz, dy, dx) \
  m_eta_hz_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_HY_X(dz, dy, dx) \
  m_eta_hy_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_ETA_HX_Y(dz, dy, dx) \
  m_eta_hx_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

static inline void tide_zero_if_not_null(void *ptr, size_t bytes) {
  if (ptr != NULL && bytes > 0) {
    memset(ptr, 0, bytes);
  }
}

static void combine_grad_shot_3d(
    TIDE_DTYPE *__restrict const grad,
    TIDE_DTYPE const *__restrict const grad_shot,
    int64_t const n_shots,
    int64_t const shot_numel) {
  if (grad == NULL || grad_shot == NULL || n_shots <= 0 || shot_numel <= 0) {
    return;
  }

  TIDE_OMP_INDEX idx;
TIDE_OMP_PARALLEL_FOR
  for (idx = 0; idx < shot_numel; ++idx) {
    TIDE_DTYPE sum = 0;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      sum += grad_shot[shot_idx * shot_numel + idx];
    }
    grad[idx] += sum;
  }
}

static void add_sources_component(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i,
    int64_t const time_offset,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_sources_per_shot) {
  if (field == NULL || f == NULL || sources_i == NULL || n_sources_per_shot <= 0) {
    return;
  }

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
      int64_t k = shot_idx * n_sources_per_shot + source_idx;
      int64_t const src = sources_i[k];
      if (src >= 0) {
        field[shot_idx * shot_numel + src] += f[time_offset + k];
      }
    }
  }
}

static void record_receivers_component(
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE const *__restrict const field,
    int64_t const *__restrict const receivers_i,
    int64_t const time_offset,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_receivers_per_shot) {
  if (r == NULL || field == NULL || receivers_i == NULL || n_receivers_per_shot <= 0) {
    return;
  }

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t receiver_idx = 0; receiver_idx < n_receivers_per_shot; ++receiver_idx) {
      int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
      int64_t const rec = receivers_i[k];
      r[time_offset + k] = (rec >= 0) ? field[shot_idx * shot_numel + rec] : (TIDE_DTYPE)0;
    }
  }
}

static void update_polarization_debye_3d(
    TIDE_DTYPE const *__restrict const prev_field,
    TIDE_DTYPE const *__restrict const field,
    TIDE_DTYPE const *__restrict const debye_a,
    TIDE_DTYPE const *__restrict const debye_b,
    TIDE_DTYPE *__restrict const polarization,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_poles) {
  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_off = (int64_t)shot_idx * shot_numel;
    for (int64_t pole = 0; pole < n_poles; ++pole) {
      int64_t const coeff_off = pole * shot_numel;
      int64_t const pol_off = ((int64_t)shot_idx * n_poles + pole) * shot_numel;
      for (int64_t idx = 0; idx < shot_numel; ++idx) {
        polarization[pol_off + idx] =
            debye_a[coeff_off + idx] * polarization[pol_off + idx] +
            debye_b[coeff_off + idx] *
                (field[shot_off + idx] + prev_field[shot_off + idx]);
      }
    }
  }
}

static void convert_grad_ca_cb_to_eps_sigma_3d(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const grad_ca,
    TIDE_DTYPE const *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE const dt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    bool const ca_batched,
    bool const cb_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {
  if ((grad_eps == NULL && grad_sigma == NULL) ||
      (!ca_requires_grad && !cb_requires_grad)) {
    return;
  }

  int64_t const shot_numel = nz * ny * nx;
  TIDE_DTYPE const inv_dt = (TIDE_DTYPE)1 / dt;

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < (ca_batched ? n_shots : 1); ++shot_idx) {
    int64_t const shot_off = (int64_t)shot_idx * shot_numel;
    for (int64_t z = 0; z < nz; ++z) {
      for (int64_t y = 0; y < ny; ++y) {
        for (int64_t x = 0; x < nx; ++x) {
          int64_t const idx = IDX(z, y, x);
          int64_t const idx_shot = shot_off + idx;
          int64_t const out_idx = ca_batched ? idx_shot : idx;
          int64_t const ca_idx = ca_batched ? idx_shot : idx;
          int64_t const cb_idx = cb_batched ? idx_shot : idx;

          TIDE_DTYPE const ca_val = ca[ca_idx];
          TIDE_DTYPE const cb_val = cb[cb_idx];
          TIDE_DTYPE const cb_sq = cb_val * cb_val;

          TIDE_DTYPE grad_ca_val = 0;
          TIDE_DTYPE grad_cb_val = 0;
          if (ca_requires_grad && grad_ca != NULL) {
            grad_ca_val = grad_ca[out_idx];
          }
          if (cb_requires_grad && grad_cb != NULL) {
            grad_cb_val = grad_cb[out_idx];
          }

          TIDE_DTYPE const dca_de =
              ((TIDE_DTYPE)1 - ca_val) * cb_val * inv_dt;
          TIDE_DTYPE const dcb_de = -cb_sq * inv_dt;
          TIDE_DTYPE const dca_ds =
              -((TIDE_DTYPE)0.5) * ((TIDE_DTYPE)1 + ca_val) * cb_val;
          TIDE_DTYPE const dcb_ds = -((TIDE_DTYPE)0.5) * cb_sq;

          if (grad_eps != NULL) {
            TIDE_DTYPE const grad_e = grad_ca_val * dca_de + grad_cb_val * dcb_de;
            grad_eps[out_idx] = grad_e * kEp0;
          }
          if (grad_sigma != NULL) {
            grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
          }
        }
      }
    }
  }
}

TIDE_EXTERN_C TIDE_EXPORT void FUNC(forward)(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const f,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hy,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE *__restrict const m_ez_y,
    TIDE_DTYPE *__restrict const m_ez_x,
    TIDE_DTYPE *__restrict const m_ex_z,
    TIDE_DTYPE *__restrict const m_ex_y,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE const *__restrict const debye_a,
    TIDE_DTYPE const *__restrict const debye_b,
    TIDE_DTYPE const *__restrict const debye_cp,
    TIDE_DTYPE *__restrict const pol_ex,
    TIDE_DTYPE *__restrict const pol_ey,
    TIDE_DTYPE *__restrict const pol_ez,
    TIDE_DTYPE *__restrict const ex_prev,
    TIDE_DTYPE *__restrict const ey_prev,
    TIDE_DTYPE *__restrict const ez_prev,
    TIDE_DTYPE *__restrict const r,
    int64_t const n_poles,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    bool const has_dispersion,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_z0,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_z1,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle) {
  (void)dt;
  (void)step_ratio;
  (void)device;
  (void)execution_backend;
  (void)compute_stream_handle;

#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = nz * ny * nx;

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    TIDE_OMP_INDEX shot_idx;

    /* H update (uses half-grid derivatives of E). */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const cq_val = CQ_AT(0, 0, 0);

            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dEy_dz_pml = 0;
            TIDE_DTYPE dEz_dy_pml = 0;
            TIDE_DTYPE dEz_dx_pml = 0;
            TIDE_DTYPE dEx_dz_pml = 0;
            TIDE_DTYPE dEx_dy_pml = 0;
            TIDE_DTYPE dEy_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEy_dz = DIFFZH1(EY);
              if (pml_z_h) {
                M_EY_Z(0, 0, 0) = bzh[z] * M_EY_Z(0, 0, 0) + azh[z] * dEy_dz;
                dEy_dz = dEy_dz / kzh[z] + M_EY_Z(0, 0, 0);
              }
              dEy_dz_pml = dEy_dz;
            }

            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEz_dy = DIFFYH1(EZ);
              if (pml_y_h) {
                M_EZ_Y(0, 0, 0) = byh[y] * M_EZ_Y(0, 0, 0) + ayh[y] * dEz_dy;
                dEz_dy = dEz_dy / kyh[y] + M_EZ_Y(0, 0, 0);
              }
              dEz_dy_pml = dEz_dy;
            }

            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEz_dx = DIFFXH1(EZ);
              if (pml_x_h) {
                M_EZ_X(0, 0, 0) = bxh[x] * M_EZ_X(0, 0, 0) + axh[x] * dEz_dx;
                dEz_dx = dEz_dx / kxh[x] + M_EZ_X(0, 0, 0);
              }
              dEz_dx_pml = dEz_dx;
            }

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEx_dz = DIFFZH1(EX);
              if (pml_z_h) {
                M_EX_Z(0, 0, 0) = bzh[z] * M_EX_Z(0, 0, 0) + azh[z] * dEx_dz;
                dEx_dz = dEx_dz / kzh[z] + M_EX_Z(0, 0, 0);
              }
              dEx_dz_pml = dEx_dz;
            }

            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEx_dy = DIFFYH1(EX);
              if (pml_y_h) {
                M_EX_Y(0, 0, 0) = byh[y] * M_EX_Y(0, 0, 0) + ayh[y] * dEx_dy;
                dEx_dy = dEx_dy / kyh[y] + M_EX_Y(0, 0, 0);
              }
              dEx_dy_pml = dEx_dy;
            }

            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEy_dx = DIFFXH1(EY);
              if (pml_x_h) {
                M_EY_X(0, 0, 0) = bxh[x] * M_EY_X(0, 0, 0) + axh[x] * dEy_dx;
                dEy_dx = dEy_dx / kxh[x] + M_EY_X(0, 0, 0);
              }
              dEy_dx_pml = dEy_dx;
            }

            HX(0, 0, 0) -= cq_val * (dEy_dz_pml - dEz_dy_pml);
            HY(0, 0, 0) -= cq_val * (dEz_dx_pml - dEx_dz_pml);
            HZ(0, 0, 0) -= cq_val * (dEx_dy_pml - dEy_dx_pml);
          }
        }
      }
    }

    if (has_dispersion) {
      int64_t const total_cells = n_shots * shot_numel;
TIDE_OMP_PARALLEL_FOR
      for (shot_idx = 0; shot_idx < total_cells; ++shot_idx) {
        ex_prev[shot_idx] = ex[shot_idx];
        ey_prev[shot_idx] = ey[shot_idx];
        ez_prev[shot_idx] = ez[shot_idx];
      }
    }

    /* E update (uses integer-grid derivatives of H). */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            TIDE_DTYPE const cb_val = CB_AT(0, 0, 0);

            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dHy_dz = DIFFZ1(HY);
            TIDE_DTYPE dHz_dy = DIFFY1(HZ);
            TIDE_DTYPE dHz_dx = DIFFX1(HZ);
            TIDE_DTYPE dHx_dz = DIFFZ1(HX);
            TIDE_DTYPE dHx_dy = DIFFY1(HX);
            TIDE_DTYPE dHy_dx = DIFFX1(HY);

            if (pml_z) {
              M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
              dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);

              M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
              dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
            }

            if (pml_y) {
              M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
              dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);

              M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
              dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
            }

            if (pml_x) {
              M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
              dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);

              M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
              dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
            }

            EX(0, 0, 0) = ca_val * EX(0, 0, 0) + cb_val * (dHy_dz - dHz_dy);
            EY(0, 0, 0) = ca_val * EY(0, 0, 0) + cb_val * (dHz_dx - dHx_dz);
            EZ(0, 0, 0) = ca_val * EZ(0, 0, 0) + cb_val * (dHx_dy - dHy_dx);

            if (has_dispersion) {
              int64_t const idx = IDX(0, 0, 0);
              TIDE_DTYPE pol_term_x = 0;
              TIDE_DTYPE pol_term_y = 0;
              TIDE_DTYPE pol_term_z = 0;
              for (int64_t pole = 0; pole < n_poles; ++pole) {
                int64_t const coeff_idx = pole * shot_numel + idx;
                int64_t const pol_idx =
                    ((int64_t)shot_idx * n_poles + pole) * shot_numel + idx;
                TIDE_DTYPE const cp = debye_cp[coeff_idx];
                pol_term_x += cp * pol_ex[pol_idx];
                pol_term_y += cp * pol_ey[pol_idx];
                pol_term_z += cp * pol_ez[pol_idx];
              }
              EX(0, 0, 0) += pol_term_x;
              EY(0, 0, 0) += pol_term_y;
              EZ(0, 0, 0) += pol_term_z;
            }
          }
        }
      }
    }

    int64_t const source_time_offset = t * n_shots * n_sources_per_shot;
    if (n_sources_per_shot > 0 && f != NULL && sources_i != NULL) {
      if (source_component == 0) {
        add_sources_component(ex, f, sources_i, source_time_offset, n_shots, shot_numel,
                              n_sources_per_shot);
      } else if (source_component == 2) {
        add_sources_component(ez, f, sources_i, source_time_offset, n_shots, shot_numel,
                              n_sources_per_shot);
      } else {
        add_sources_component(ey, f, sources_i, source_time_offset, n_shots, shot_numel,
                              n_sources_per_shot);
      }
    }

    if (has_dispersion) {
      update_polarization_debye_3d(
          ex_prev, ex, debye_a, debye_b, pol_ex, n_shots, shot_numel, n_poles);
      update_polarization_debye_3d(
          ey_prev, ey, debye_a, debye_b, pol_ey, n_shots, shot_numel, n_poles);
      update_polarization_debye_3d(
          ez_prev, ez, debye_a, debye_b, pol_ez, n_shots, shot_numel, n_poles);
    }

    int64_t const recv_time_offset = t * n_shots * n_receivers_per_shot;
    if (n_receivers_per_shot > 0 && r != NULL && receivers_i != NULL) {
      if (receiver_component == 0) {
        record_receivers_component(r, ex, receivers_i, recv_time_offset, n_shots,
                                   shot_numel, n_receivers_per_shot);
      } else if (receiver_component == 2) {
        record_receivers_component(r, ez, receivers_i, recv_time_offset, n_shots,
                                   shot_numel, n_receivers_per_shot);
      } else {
        record_receivers_component(r, ey, receivers_i, recv_time_offset, n_shots,
                                   shot_numel, n_receivers_per_shot);
      }
    }
  }

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

TIDE_EXTERN_C TIDE_EXPORT void FUNC(forward_with_storage)(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const f,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hy,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE *__restrict const m_ez_y,
    TIDE_DTYPE *__restrict const m_ez_x,
    TIDE_DTYPE *__restrict const m_ex_z,
    TIDE_DTYPE *__restrict const m_ex_y,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE *__restrict const store_1,
    TIDE_DTYPE *__restrict const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *__restrict const store_3,
    TIDE_DTYPE *__restrict const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *__restrict const store_5,
    TIDE_DTYPE *__restrict const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *__restrict const store_7,
    TIDE_DTYPE *__restrict const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *__restrict const store_9,
    TIDE_DTYPE *__restrict const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *__restrict const store_11,
    TIDE_DTYPE *__restrict const store_12,
    char **store_filenames_6,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    int64_t const storage_mode,
    int64_t const storage_format,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_z0,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_z1,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)dt;
  (void)device;
  (void)execution_backend;
  (void)compute_stream_handle;
  (void)storage_stream_handle;
  (void)storage_format;
  (void)store_2;
  (void)store_filenames_1;
  (void)store_4;
  (void)store_filenames_2;
  (void)store_6;
  (void)store_filenames_3;
  (void)store_8;
  (void)store_filenames_4;
  (void)store_10;
  (void)store_filenames_5;
  (void)store_12;
  (void)store_filenames_6;

#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = nz * ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  int64_t const step_ratio_eff = step_ratio > 0 ? step_ratio : 1;
  bool const can_store =
      (storage_mode == STORAGE_DEVICE) &&
      (storage_format == STORAGE_FORMAT_FULL) &&
      (shot_bytes_uncomp == (int64_t)(shot_numel * (int64_t)sizeof(TIDE_DTYPE))) &&
      ((ca_requires_grad && store_1 != NULL && store_3 != NULL && store_5 != NULL) ||
       (cb_requires_grad && store_7 != NULL && store_9 != NULL && store_11 != NULL));

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    bool const do_store = can_store && ((t % step_ratio_eff) == 0);
    int64_t const store_idx = do_store ? (t / step_ratio_eff) : 0;
    int64_t const store_offset = do_store ? (store_idx * store_size) : 0;

    TIDE_DTYPE *__restrict const ex_store =
        (do_store && ca_requires_grad) ? (store_1 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const ey_store =
        (do_store && ca_requires_grad) ? (store_3 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const ez_store =
        (do_store && ca_requires_grad) ? (store_5 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const curl_x_store =
        (do_store && cb_requires_grad) ? (store_7 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const curl_y_store =
        (do_store && cb_requires_grad) ? (store_9 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const curl_z_store =
        (do_store && cb_requires_grad) ? (store_11 + store_offset) : NULL;

    TIDE_OMP_INDEX shot_idx;

    /* H update (uses half-grid derivatives of E). */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const cq_val = CQ_AT(0, 0, 0);

            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dEy_dz_pml = 0;
            TIDE_DTYPE dEz_dy_pml = 0;
            TIDE_DTYPE dEz_dx_pml = 0;
            TIDE_DTYPE dEx_dz_pml = 0;
            TIDE_DTYPE dEx_dy_pml = 0;
            TIDE_DTYPE dEy_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEy_dz = DIFFZH1(EY);
              if (pml_z_h) {
                M_EY_Z(0, 0, 0) = bzh[z] * M_EY_Z(0, 0, 0) + azh[z] * dEy_dz;
                dEy_dz = dEy_dz / kzh[z] + M_EY_Z(0, 0, 0);
              }
              dEy_dz_pml = dEy_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEz_dy = DIFFYH1(EZ);
              if (pml_y_h) {
                M_EZ_Y(0, 0, 0) = byh[y] * M_EZ_Y(0, 0, 0) + ayh[y] * dEz_dy;
                dEz_dy = dEz_dy / kyh[y] + M_EZ_Y(0, 0, 0);
              }
              dEz_dy_pml = dEz_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEz_dx = DIFFXH1(EZ);
              if (pml_x_h) {
                M_EZ_X(0, 0, 0) = bxh[x] * M_EZ_X(0, 0, 0) + axh[x] * dEz_dx;
                dEz_dx = dEz_dx / kxh[x] + M_EZ_X(0, 0, 0);
              }
              dEz_dx_pml = dEz_dx;
            }
            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEx_dz = DIFFZH1(EX);
              if (pml_z_h) {
                M_EX_Z(0, 0, 0) = bzh[z] * M_EX_Z(0, 0, 0) + azh[z] * dEx_dz;
                dEx_dz = dEx_dz / kzh[z] + M_EX_Z(0, 0, 0);
              }
              dEx_dz_pml = dEx_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEx_dy = DIFFYH1(EX);
              if (pml_y_h) {
                M_EX_Y(0, 0, 0) = byh[y] * M_EX_Y(0, 0, 0) + ayh[y] * dEx_dy;
                dEx_dy = dEx_dy / kyh[y] + M_EX_Y(0, 0, 0);
              }
              dEx_dy_pml = dEx_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEy_dx = DIFFXH1(EY);
              if (pml_x_h) {
                M_EY_X(0, 0, 0) = bxh[x] * M_EY_X(0, 0, 0) + axh[x] * dEy_dx;
                dEy_dx = dEy_dx / kxh[x] + M_EY_X(0, 0, 0);
              }
              dEy_dx_pml = dEy_dx;
            }

            HX(0, 0, 0) -= cq_val * (dEy_dz_pml - dEz_dy_pml);
            HY(0, 0, 0) -= cq_val * (dEz_dx_pml - dEx_dz_pml);
            HZ(0, 0, 0) -= cq_val * (dEx_dy_pml - dEy_dx_pml);
          }
        }
      }
    }

    /* E update (store Ex/Ey/Ez and curl(H) before update when requested). */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            int64_t const idx = IDX(z, y, x);
            int64_t const store_linear = shot_idx * shot_numel + idx;
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            TIDE_DTYPE const cb_val = CB_AT(0, 0, 0);

            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dHy_dz = DIFFZ1(HY);
            TIDE_DTYPE dHz_dy = DIFFY1(HZ);
            TIDE_DTYPE dHz_dx = DIFFX1(HZ);
            TIDE_DTYPE dHx_dz = DIFFZ1(HX);
            TIDE_DTYPE dHx_dy = DIFFY1(HX);
            TIDE_DTYPE dHy_dx = DIFFX1(HY);

            if (pml_z) {
              M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
              dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
              M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
              dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
              dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
              M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
              dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
              dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
              M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
              dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
            TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
            TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

            if (ex_store != NULL) {
              ex_store[store_linear] = EX(0, 0, 0);
              ey_store[store_linear] = EY(0, 0, 0);
              ez_store[store_linear] = EZ(0, 0, 0);
            }
            if (curl_x_store != NULL) {
              curl_x_store[store_linear] = curl_x;
              curl_y_store[store_linear] = curl_y;
              curl_z_store[store_linear] = curl_z;
            }

            EX(0, 0, 0) = ca_val * EX(0, 0, 0) + cb_val * curl_x;
            EY(0, 0, 0) = ca_val * EY(0, 0, 0) + cb_val * curl_y;
            EZ(0, 0, 0) = ca_val * EZ(0, 0, 0) + cb_val * curl_z;
          }
        }
      }
    }

    int64_t const source_time_offset = t * n_shots * n_sources_per_shot;
    if (n_sources_per_shot > 0 && f != NULL && sources_i != NULL) {
      if (source_component == 0) {
        add_sources_component(
            ex, f, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      } else if (source_component == 2) {
        add_sources_component(
            ez, f, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      } else {
        add_sources_component(
            ey, f, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      }
    }

    int64_t const recv_time_offset = t * n_shots * n_receivers_per_shot;
    if (n_receivers_per_shot > 0 && r != NULL && receivers_i != NULL) {
      if (receiver_component == 0) {
        record_receivers_component(
            r, ex, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      } else if (receiver_component == 2) {
        record_receivers_component(
            r, ez, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      } else {
        record_receivers_component(
            r, ey, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      }
    }
  }

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

TIDE_EXTERN_C TIDE_EXPORT void FUNC(born_forward)(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const f0,
    TIDE_DTYPE const *__restrict const df,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hy,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE *__restrict const m_ez_y,
    TIDE_DTYPE *__restrict const m_ez_x,
    TIDE_DTYPE *__restrict const m_ex_z,
    TIDE_DTYPE *__restrict const m_ex_y,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const dex,
    TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const dez,
    TIDE_DTYPE *__restrict const dhx,
    TIDE_DTYPE *__restrict const dhy,
    TIDE_DTYPE *__restrict const dhz,
    TIDE_DTYPE *__restrict const dm_hz_y,
    TIDE_DTYPE *__restrict const dm_hy_z,
    TIDE_DTYPE *__restrict const dm_hx_z,
    TIDE_DTYPE *__restrict const dm_hz_x,
    TIDE_DTYPE *__restrict const dm_hy_x,
    TIDE_DTYPE *__restrict const dm_hx_y,
    TIDE_DTYPE *__restrict const dm_ey_z,
    TIDE_DTYPE *__restrict const dm_ez_y,
    TIDE_DTYPE *__restrict const dm_ez_x,
    TIDE_DTYPE *__restrict const dm_ex_z,
    TIDE_DTYPE *__restrict const dm_ex_y,
    TIDE_DTYPE *__restrict const dm_ey_x,
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_z0,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_z1,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle) {
  (void)dt;
  (void)step_ratio;
  (void)device;
  (void)execution_backend;
  (void)compute_stream_handle;

#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = nz * ny * nx;
  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    TIDE_OMP_INDEX shot_idx;

TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const cq_val = CQ_AT(0, 0, 0);
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dEy_dz_pml = 0;
            TIDE_DTYPE dEz_dy_pml = 0;
            TIDE_DTYPE dEz_dx_pml = 0;
            TIDE_DTYPE dEx_dz_pml = 0;
            TIDE_DTYPE dEx_dy_pml = 0;
            TIDE_DTYPE dEy_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEy_dz = DIFFZH1(EY);
              if (pml_z_h) {
                M_EY_Z(0, 0, 0) = bzh[z] * M_EY_Z(0, 0, 0) + azh[z] * dEy_dz;
                dEy_dz = dEy_dz / kzh[z] + M_EY_Z(0, 0, 0);
              }
              dEy_dz_pml = dEy_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEz_dy = DIFFYH1(EZ);
              if (pml_y_h) {
                M_EZ_Y(0, 0, 0) = byh[y] * M_EZ_Y(0, 0, 0) + ayh[y] * dEz_dy;
                dEz_dy = dEz_dy / kyh[y] + M_EZ_Y(0, 0, 0);
              }
              dEz_dy_pml = dEz_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEz_dx = DIFFXH1(EZ);
              if (pml_x_h) {
                M_EZ_X(0, 0, 0) = bxh[x] * M_EZ_X(0, 0, 0) + axh[x] * dEz_dx;
                dEz_dx = dEz_dx / kxh[x] + M_EZ_X(0, 0, 0);
              }
              dEz_dx_pml = dEz_dx;
            }
            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEx_dz = DIFFZH1(EX);
              if (pml_z_h) {
                M_EX_Z(0, 0, 0) = bzh[z] * M_EX_Z(0, 0, 0) + azh[z] * dEx_dz;
                dEx_dz = dEx_dz / kzh[z] + M_EX_Z(0, 0, 0);
              }
              dEx_dz_pml = dEx_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEx_dy = DIFFYH1(EX);
              if (pml_y_h) {
                M_EX_Y(0, 0, 0) = byh[y] * M_EX_Y(0, 0, 0) + ayh[y] * dEx_dy;
                dEx_dy = dEx_dy / kyh[y] + M_EX_Y(0, 0, 0);
              }
              dEx_dy_pml = dEx_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEy_dx = DIFFXH1(EY);
              if (pml_x_h) {
                M_EY_X(0, 0, 0) = bxh[x] * M_EY_X(0, 0, 0) + axh[x] * dEy_dx;
                dEy_dx = dEy_dx / kxh[x] + M_EY_X(0, 0, 0);
              }
              dEy_dx_pml = dEy_dx;
            }

            HX(0, 0, 0) -= cq_val * (dEy_dz_pml - dEz_dy_pml);
            HY(0, 0, 0) -= cq_val * (dEz_dx_pml - dEx_dz_pml);
            HZ(0, 0, 0) -= cq_val * (dEx_dy_pml - dEy_dx_pml);
          }
        }
      }
    }

TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const cq_val = CQ_AT(0, 0, 0);
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dDEy_dz_pml = 0;
            TIDE_DTYPE dDEz_dy_pml = 0;
            TIDE_DTYPE dDEz_dx_pml = 0;
            TIDE_DTYPE dDEx_dz_pml = 0;
            TIDE_DTYPE dDEx_dy_pml = 0;
            TIDE_DTYPE dDEy_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dDEy_dz = DIFFZH1(DEY);
              if (pml_z_h) {
                DM_EY_Z(0, 0, 0) =
                    bzh[z] * DM_EY_Z(0, 0, 0) + azh[z] * dDEy_dz;
                dDEy_dz = dDEy_dz / kzh[z] + DM_EY_Z(0, 0, 0);
              }
              dDEy_dz_pml = dDEy_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dDEz_dy = DIFFYH1(DEZ);
              if (pml_y_h) {
                DM_EZ_Y(0, 0, 0) =
                    byh[y] * DM_EZ_Y(0, 0, 0) + ayh[y] * dDEz_dy;
                dDEz_dy = dDEz_dy / kyh[y] + DM_EZ_Y(0, 0, 0);
              }
              dDEz_dy_pml = dDEz_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dDEz_dx = DIFFXH1(DEZ);
              if (pml_x_h) {
                DM_EZ_X(0, 0, 0) =
                    bxh[x] * DM_EZ_X(0, 0, 0) + axh[x] * dDEz_dx;
                dDEz_dx = dDEz_dx / kxh[x] + DM_EZ_X(0, 0, 0);
              }
              dDEz_dx_pml = dDEz_dx;
            }
            if (z < nz - FD_PAD) {
              TIDE_DTYPE dDEx_dz = DIFFZH1(DEX);
              if (pml_z_h) {
                DM_EX_Z(0, 0, 0) =
                    bzh[z] * DM_EX_Z(0, 0, 0) + azh[z] * dDEx_dz;
                dDEx_dz = dDEx_dz / kzh[z] + DM_EX_Z(0, 0, 0);
              }
              dDEx_dz_pml = dDEx_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dDEx_dy = DIFFYH1(DEX);
              if (pml_y_h) {
                DM_EX_Y(0, 0, 0) =
                    byh[y] * DM_EX_Y(0, 0, 0) + ayh[y] * dDEx_dy;
                dDEx_dy = dDEx_dy / kyh[y] + DM_EX_Y(0, 0, 0);
              }
              dDEx_dy_pml = dDEx_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dDEy_dx = DIFFXH1(DEY);
              if (pml_x_h) {
                DM_EY_X(0, 0, 0) =
                    bxh[x] * DM_EY_X(0, 0, 0) + axh[x] * dDEy_dx;
                dDEy_dx = dDEy_dx / kxh[x] + DM_EY_X(0, 0, 0);
              }
              dDEy_dx_pml = dDEy_dx;
            }

            DHX(0, 0, 0) -= cq_val * (dDEy_dz_pml - dDEz_dy_pml);
            DHY(0, 0, 0) -= cq_val * (dDEz_dx_pml - dDEx_dz_pml);
            DHZ(0, 0, 0) -= cq_val * (dDEx_dy_pml - dDEy_dx_pml);
          }
        }
      }
    }

TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            TIDE_DTYPE const cb_val = CB_AT(0, 0, 0);
            TIDE_DTYPE const dca_val = DCA_AT(0, 0, 0);
            TIDE_DTYPE const dcb_val = DCB_AT(0, 0, 0);
            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dHy_dz = DIFFZ1(HY);
            TIDE_DTYPE dHz_dy = DIFFY1(HZ);
            TIDE_DTYPE dHz_dx = DIFFX1(HZ);
            TIDE_DTYPE dHx_dz = DIFFZ1(HX);
            TIDE_DTYPE dHx_dy = DIFFY1(HX);
            TIDE_DTYPE dHy_dx = DIFFX1(HY);

            if (pml_z) {
              M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
              dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
              M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
              dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
              dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
              M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
              dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
              dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
              M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
              dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
            TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
            TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

            TIDE_DTYPE ddHy_dz = DIFFZ1(DHY);
            TIDE_DTYPE ddHz_dy = DIFFY1(DHZ);
            TIDE_DTYPE ddHz_dx = DIFFX1(DHZ);
            TIDE_DTYPE ddHx_dz = DIFFZ1(DHX);
            TIDE_DTYPE ddHx_dy = DIFFY1(DHX);
            TIDE_DTYPE ddHy_dx = DIFFX1(DHY);

            if (pml_z) {
              DM_HY_Z(0, 0, 0) =
                  bz[z] * DM_HY_Z(0, 0, 0) + az[z] * ddHy_dz;
              ddHy_dz = ddHy_dz / kz[z] + DM_HY_Z(0, 0, 0);
              DM_HX_Z(0, 0, 0) =
                  bz[z] * DM_HX_Z(0, 0, 0) + az[z] * ddHx_dz;
              ddHx_dz = ddHx_dz / kz[z] + DM_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              DM_HZ_Y(0, 0, 0) =
                  by[y] * DM_HZ_Y(0, 0, 0) + ay[y] * ddHz_dy;
              ddHz_dy = ddHz_dy / ky[y] + DM_HZ_Y(0, 0, 0);
              DM_HX_Y(0, 0, 0) =
                  by[y] * DM_HX_Y(0, 0, 0) + ay[y] * ddHx_dy;
              ddHx_dy = ddHx_dy / ky[y] + DM_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              DM_HZ_X(0, 0, 0) =
                  bx[x] * DM_HZ_X(0, 0, 0) + ax[x] * ddHz_dx;
              ddHz_dx = ddHz_dx / kx[x] + DM_HZ_X(0, 0, 0);
              DM_HY_X(0, 0, 0) =
                  bx[x] * DM_HY_X(0, 0, 0) + ax[x] * ddHy_dx;
              ddHy_dx = ddHy_dx / kx[x] + DM_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const dcurl_x = ddHy_dz - ddHz_dy;
            TIDE_DTYPE const dcurl_y = ddHz_dx - ddHx_dz;
            TIDE_DTYPE const dcurl_z = ddHx_dy - ddHy_dx;

            TIDE_DTYPE const ex_old = EX(0, 0, 0);
            TIDE_DTYPE const ey_old = EY(0, 0, 0);
            TIDE_DTYPE const ez_old = EZ(0, 0, 0);

            DEX(0, 0, 0) =
                ca_val * DEX(0, 0, 0) + cb_val * dcurl_x + dca_val * ex_old +
                dcb_val * curl_x;
            DEY(0, 0, 0) =
                ca_val * DEY(0, 0, 0) + cb_val * dcurl_y + dca_val * ey_old +
                dcb_val * curl_y;
            DEZ(0, 0, 0) =
                ca_val * DEZ(0, 0, 0) + cb_val * dcurl_z + dca_val * ez_old +
                dcb_val * curl_z;

            EX(0, 0, 0) = ca_val * EX(0, 0, 0) + cb_val * curl_x;
            EY(0, 0, 0) = ca_val * EY(0, 0, 0) + cb_val * curl_y;
            EZ(0, 0, 0) = ca_val * EZ(0, 0, 0) + cb_val * curl_z;
          }
        }
      }
    }

    int64_t const source_time_offset = t * n_shots * n_sources_per_shot;
    if (n_sources_per_shot > 0 && sources_i != NULL) {
      if (source_component == 0) {
        add_sources_component(
            ex, f0, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
        add_sources_component(
            dex, df, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      } else if (source_component == 2) {
        add_sources_component(
            ez, f0, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
        add_sources_component(
            dez, df, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      } else {
        add_sources_component(
            ey, f0, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
        add_sources_component(
            dey, df, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      }
    }

    int64_t const recv_time_offset = t * n_shots * n_receivers_per_shot;
    if (n_receivers_per_shot > 0 && r != NULL && receivers_i != NULL) {
      if (receiver_component == 0) {
        record_receivers_component(
            r, dex, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      } else if (receiver_component == 2) {
        record_receivers_component(
            r, dez, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      } else {
        record_receivers_component(
            r, dey, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      }
    }
  }

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

TIDE_EXTERN_C TIDE_EXPORT void FUNC(born_forward_with_storage)(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const f0,
    TIDE_DTYPE const *__restrict const df,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hy,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE *__restrict const m_ez_y,
    TIDE_DTYPE *__restrict const m_ez_x,
    TIDE_DTYPE *__restrict const m_ex_z,
    TIDE_DTYPE *__restrict const m_ex_y,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const dex,
    TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const dez,
    TIDE_DTYPE *__restrict const dhx,
    TIDE_DTYPE *__restrict const dhy,
    TIDE_DTYPE *__restrict const dhz,
    TIDE_DTYPE *__restrict const dm_hz_y,
    TIDE_DTYPE *__restrict const dm_hy_z,
    TIDE_DTYPE *__restrict const dm_hx_z,
    TIDE_DTYPE *__restrict const dm_hz_x,
    TIDE_DTYPE *__restrict const dm_hy_x,
    TIDE_DTYPE *__restrict const dm_hx_y,
    TIDE_DTYPE *__restrict const dm_ey_z,
    TIDE_DTYPE *__restrict const dm_ez_y,
    TIDE_DTYPE *__restrict const dm_ez_x,
    TIDE_DTYPE *__restrict const dm_ex_z,
    TIDE_DTYPE *__restrict const dm_ex_y,
    TIDE_DTYPE *__restrict const dm_ey_x,
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE *__restrict const store_1,
    TIDE_DTYPE *__restrict const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *__restrict const store_3,
    TIDE_DTYPE *__restrict const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *__restrict const store_5,
    TIDE_DTYPE *__restrict const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *__restrict const store_7,
    TIDE_DTYPE *__restrict const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *__restrict const store_9,
    TIDE_DTYPE *__restrict const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *__restrict const store_11,
    TIDE_DTYPE *__restrict const store_12,
    char **store_filenames_6,
    TIDE_DTYPE *__restrict const dstore_ex,
    TIDE_DTYPE *__restrict const dstore_ey,
    TIDE_DTYPE *__restrict const dstore_ez,
    TIDE_DTYPE *__restrict const dstore_curl_x,
    TIDE_DTYPE *__restrict const dstore_curl_y,
    TIDE_DTYPE *__restrict const dstore_curl_z,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    int64_t const storage_mode,
    int64_t const storage_format,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_z0,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_z1,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)dt;
  (void)device;
  (void)execution_backend;
  (void)compute_stream_handle;
  (void)storage_stream_handle;
  (void)storage_format;
  (void)store_2;
  (void)store_filenames_1;
  (void)store_4;
  (void)store_filenames_2;
  (void)store_6;
  (void)store_filenames_3;
  (void)store_8;
  (void)store_filenames_4;
  (void)store_10;
  (void)store_filenames_5;
  (void)store_12;
  (void)store_filenames_6;

#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = nz * ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  int64_t const step_ratio_eff = step_ratio > 0 ? step_ratio : 1;
  bool const can_store =
      (storage_mode == STORAGE_DEVICE) &&
      (storage_format == STORAGE_FORMAT_FULL) &&
      (shot_bytes_uncomp == (int64_t)(shot_numel * (int64_t)sizeof(TIDE_DTYPE))) &&
      ((ca_requires_grad && store_1 != NULL && store_3 != NULL && store_5 != NULL) ||
       (cb_requires_grad && store_7 != NULL && store_9 != NULL && store_11 != NULL) ||
       (dstore_ex != NULL && dstore_ey != NULL && dstore_ez != NULL) ||
       (dstore_curl_x != NULL && dstore_curl_y != NULL &&
        dstore_curl_z != NULL));

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    bool const do_store = can_store && ((t % step_ratio_eff) == 0);
    int64_t const store_idx = do_store ? (t / step_ratio_eff) : 0;
    int64_t const store_offset = do_store ? (store_idx * store_size) : 0;

    TIDE_DTYPE *__restrict const ex_store =
        (do_store && ca_requires_grad) ? (store_1 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const ey_store =
        (do_store && ca_requires_grad) ? (store_3 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const ez_store =
        (do_store && ca_requires_grad) ? (store_5 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const curl_x_store =
        (do_store && cb_requires_grad) ? (store_7 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const curl_y_store =
        (do_store && cb_requires_grad) ? (store_9 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const curl_z_store =
        (do_store && cb_requires_grad) ? (store_11 + store_offset) : NULL;
    TIDE_DTYPE *__restrict const dex_store =
        (do_store && dstore_ex != NULL) ? (dstore_ex + store_offset) : NULL;
    TIDE_DTYPE *__restrict const dey_store =
        (do_store && dstore_ey != NULL) ? (dstore_ey + store_offset) : NULL;
    TIDE_DTYPE *__restrict const dez_store =
        (do_store && dstore_ez != NULL) ? (dstore_ez + store_offset) : NULL;
    TIDE_DTYPE *__restrict const dcurl_x_store =
        (do_store && dstore_curl_x != NULL) ? (dstore_curl_x + store_offset) : NULL;
    TIDE_DTYPE *__restrict const dcurl_y_store =
        (do_store && dstore_curl_y != NULL) ? (dstore_curl_y + store_offset) : NULL;
    TIDE_DTYPE *__restrict const dcurl_z_store =
        (do_store && dstore_curl_z != NULL) ? (dstore_curl_z + store_offset) : NULL;

    TIDE_OMP_INDEX shot_idx;

TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const cq_val = CQ_AT(0, 0, 0);
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dEy_dz_pml = 0;
            TIDE_DTYPE dEz_dy_pml = 0;
            TIDE_DTYPE dEz_dx_pml = 0;
            TIDE_DTYPE dEx_dz_pml = 0;
            TIDE_DTYPE dEx_dy_pml = 0;
            TIDE_DTYPE dEy_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEy_dz = DIFFZH1(EY);
              if (pml_z_h) {
                M_EY_Z(0, 0, 0) = bzh[z] * M_EY_Z(0, 0, 0) + azh[z] * dEy_dz;
                dEy_dz = dEy_dz / kzh[z] + M_EY_Z(0, 0, 0);
              }
              dEy_dz_pml = dEy_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEz_dy = DIFFYH1(EZ);
              if (pml_y_h) {
                M_EZ_Y(0, 0, 0) = byh[y] * M_EZ_Y(0, 0, 0) + ayh[y] * dEz_dy;
                dEz_dy = dEz_dy / kyh[y] + M_EZ_Y(0, 0, 0);
              }
              dEz_dy_pml = dEz_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEz_dx = DIFFXH1(EZ);
              if (pml_x_h) {
                M_EZ_X(0, 0, 0) = bxh[x] * M_EZ_X(0, 0, 0) + axh[x] * dEz_dx;
                dEz_dx = dEz_dx / kxh[x] + M_EZ_X(0, 0, 0);
              }
              dEz_dx_pml = dEz_dx;
            }
            if (z < nz - FD_PAD) {
              TIDE_DTYPE dEx_dz = DIFFZH1(EX);
              if (pml_z_h) {
                M_EX_Z(0, 0, 0) = bzh[z] * M_EX_Z(0, 0, 0) + azh[z] * dEx_dz;
                dEx_dz = dEx_dz / kzh[z] + M_EX_Z(0, 0, 0);
              }
              dEx_dz_pml = dEx_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dEx_dy = DIFFYH1(EX);
              if (pml_y_h) {
                M_EX_Y(0, 0, 0) = byh[y] * M_EX_Y(0, 0, 0) + ayh[y] * dEx_dy;
                dEx_dy = dEx_dy / kyh[y] + M_EX_Y(0, 0, 0);
              }
              dEx_dy_pml = dEx_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dEy_dx = DIFFXH1(EY);
              if (pml_x_h) {
                M_EY_X(0, 0, 0) = bxh[x] * M_EY_X(0, 0, 0) + axh[x] * dEy_dx;
                dEy_dx = dEy_dx / kxh[x] + M_EY_X(0, 0, 0);
              }
              dEy_dx_pml = dEy_dx;
            }

            HX(0, 0, 0) -= cq_val * (dEy_dz_pml - dEz_dy_pml);
            HY(0, 0, 0) -= cq_val * (dEz_dx_pml - dEx_dz_pml);
            HZ(0, 0, 0) -= cq_val * (dEx_dy_pml - dEy_dx_pml);
          }
        }
      }
    }

TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            TIDE_DTYPE const cq_val = CQ_AT(0, 0, 0);
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dDEy_dz_pml = 0;
            TIDE_DTYPE dDEz_dy_pml = 0;
            TIDE_DTYPE dDEz_dx_pml = 0;
            TIDE_DTYPE dDEx_dz_pml = 0;
            TIDE_DTYPE dDEx_dy_pml = 0;
            TIDE_DTYPE dDEy_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dDEy_dz = DIFFZH1(DEY);
              if (pml_z_h) {
                DM_EY_Z(0, 0, 0) =
                    bzh[z] * DM_EY_Z(0, 0, 0) + azh[z] * dDEy_dz;
                dDEy_dz = dDEy_dz / kzh[z] + DM_EY_Z(0, 0, 0);
              }
              dDEy_dz_pml = dDEy_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dDEz_dy = DIFFYH1(DEZ);
              if (pml_y_h) {
                DM_EZ_Y(0, 0, 0) =
                    byh[y] * DM_EZ_Y(0, 0, 0) + ayh[y] * dDEz_dy;
                dDEz_dy = dDEz_dy / kyh[y] + DM_EZ_Y(0, 0, 0);
              }
              dDEz_dy_pml = dDEz_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dDEz_dx = DIFFXH1(DEZ);
              if (pml_x_h) {
                DM_EZ_X(0, 0, 0) =
                    bxh[x] * DM_EZ_X(0, 0, 0) + axh[x] * dDEz_dx;
                dDEz_dx = dDEz_dx / kxh[x] + DM_EZ_X(0, 0, 0);
              }
              dDEz_dx_pml = dDEz_dx;
            }
            if (z < nz - FD_PAD) {
              TIDE_DTYPE dDEx_dz = DIFFZH1(DEX);
              if (pml_z_h) {
                DM_EX_Z(0, 0, 0) =
                    bzh[z] * DM_EX_Z(0, 0, 0) + azh[z] * dDEx_dz;
                dDEx_dz = dDEx_dz / kzh[z] + DM_EX_Z(0, 0, 0);
              }
              dDEx_dz_pml = dDEx_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dDEx_dy = DIFFYH1(DEX);
              if (pml_y_h) {
                DM_EX_Y(0, 0, 0) =
                    byh[y] * DM_EX_Y(0, 0, 0) + ayh[y] * dDEx_dy;
                dDEx_dy = dDEx_dy / kyh[y] + DM_EX_Y(0, 0, 0);
              }
              dDEx_dy_pml = dDEx_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dDEy_dx = DIFFXH1(DEY);
              if (pml_x_h) {
                DM_EY_X(0, 0, 0) =
                    bxh[x] * DM_EY_X(0, 0, 0) + axh[x] * dDEy_dx;
                dDEy_dx = dDEy_dx / kxh[x] + DM_EY_X(0, 0, 0);
              }
              dDEy_dx_pml = dDEy_dx;
            }

            DHX(0, 0, 0) -= cq_val * (dDEy_dz_pml - dDEz_dy_pml);
            DHY(0, 0, 0) -= cq_val * (dDEz_dx_pml - dDEx_dz_pml);
            DHZ(0, 0, 0) -= cq_val * (dDEx_dy_pml - dDEy_dx_pml);
          }
        }
      }
    }

TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            int64_t const idx = IDX(z, y, x);
            int64_t const store_linear = shot_idx * shot_numel + idx;
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            TIDE_DTYPE const cb_val = CB_AT(0, 0, 0);
            TIDE_DTYPE const dca_val = DCA_AT(0, 0, 0);
            TIDE_DTYPE const dcb_val = DCB_AT(0, 0, 0);
            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dHy_dz = DIFFZ1(HY);
            TIDE_DTYPE dHz_dy = DIFFY1(HZ);
            TIDE_DTYPE dHz_dx = DIFFX1(HZ);
            TIDE_DTYPE dHx_dz = DIFFZ1(HX);
            TIDE_DTYPE dHx_dy = DIFFY1(HX);
            TIDE_DTYPE dHy_dx = DIFFX1(HY);

            if (pml_z) {
              M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
              dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
              M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
              dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
              dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
              M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
              dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
              dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
              M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
              dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
            TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
            TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

            if (ex_store != NULL) {
              ex_store[store_linear] = EX(0, 0, 0);
              ey_store[store_linear] = EY(0, 0, 0);
              ez_store[store_linear] = EZ(0, 0, 0);
            }
            if (curl_x_store != NULL) {
              curl_x_store[store_linear] = curl_x;
              curl_y_store[store_linear] = curl_y;
              curl_z_store[store_linear] = curl_z;
            }

            TIDE_DTYPE ddHy_dz = DIFFZ1(DHY);
            TIDE_DTYPE ddHz_dy = DIFFY1(DHZ);
            TIDE_DTYPE ddHz_dx = DIFFX1(DHZ);
            TIDE_DTYPE ddHx_dz = DIFFZ1(DHX);
            TIDE_DTYPE ddHx_dy = DIFFY1(DHX);
            TIDE_DTYPE ddHy_dx = DIFFX1(DHY);

            if (pml_z) {
              DM_HY_Z(0, 0, 0) =
                  bz[z] * DM_HY_Z(0, 0, 0) + az[z] * ddHy_dz;
              ddHy_dz = ddHy_dz / kz[z] + DM_HY_Z(0, 0, 0);
              DM_HX_Z(0, 0, 0) =
                  bz[z] * DM_HX_Z(0, 0, 0) + az[z] * ddHx_dz;
              ddHx_dz = ddHx_dz / kz[z] + DM_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              DM_HZ_Y(0, 0, 0) =
                  by[y] * DM_HZ_Y(0, 0, 0) + ay[y] * ddHz_dy;
              ddHz_dy = ddHz_dy / ky[y] + DM_HZ_Y(0, 0, 0);
              DM_HX_Y(0, 0, 0) =
                  by[y] * DM_HX_Y(0, 0, 0) + ay[y] * ddHx_dy;
              ddHx_dy = ddHx_dy / ky[y] + DM_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              DM_HZ_X(0, 0, 0) =
                  bx[x] * DM_HZ_X(0, 0, 0) + ax[x] * ddHz_dx;
              ddHz_dx = ddHz_dx / kx[x] + DM_HZ_X(0, 0, 0);
              DM_HY_X(0, 0, 0) =
                  bx[x] * DM_HY_X(0, 0, 0) + ax[x] * ddHy_dx;
              ddHy_dx = ddHy_dx / kx[x] + DM_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const dcurl_x = ddHy_dz - ddHz_dy;
            TIDE_DTYPE const dcurl_y = ddHz_dx - ddHx_dz;
            TIDE_DTYPE const dcurl_z = ddHx_dy - ddHy_dx;

            if (dex_store != NULL) {
              dex_store[store_linear] = DEX(0, 0, 0);
              dey_store[store_linear] = DEY(0, 0, 0);
              dez_store[store_linear] = DEZ(0, 0, 0);
            }
            if (dcurl_x_store != NULL) {
              dcurl_x_store[store_linear] = dcurl_x;
              dcurl_y_store[store_linear] = dcurl_y;
              dcurl_z_store[store_linear] = dcurl_z;
            }

            TIDE_DTYPE const ex_old = EX(0, 0, 0);
            TIDE_DTYPE const ey_old = EY(0, 0, 0);
            TIDE_DTYPE const ez_old = EZ(0, 0, 0);

            DEX(0, 0, 0) =
                ca_val * DEX(0, 0, 0) + cb_val * dcurl_x + dca_val * ex_old +
                dcb_val * curl_x;
            DEY(0, 0, 0) =
                ca_val * DEY(0, 0, 0) + cb_val * dcurl_y + dca_val * ey_old +
                dcb_val * curl_y;
            DEZ(0, 0, 0) =
                ca_val * DEZ(0, 0, 0) + cb_val * dcurl_z + dca_val * ez_old +
                dcb_val * curl_z;

            EX(0, 0, 0) = ca_val * EX(0, 0, 0) + cb_val * curl_x;
            EY(0, 0, 0) = ca_val * EY(0, 0, 0) + cb_val * curl_y;
            EZ(0, 0, 0) = ca_val * EZ(0, 0, 0) + cb_val * curl_z;
          }
        }
      }
    }

    int64_t const source_time_offset = t * n_shots * n_sources_per_shot;
    if (n_sources_per_shot > 0 && sources_i != NULL) {
      if (source_component == 0) {
        add_sources_component(
            ex, f0, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
        add_sources_component(
            dex, df, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      } else if (source_component == 2) {
        add_sources_component(
            ez, f0, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
        add_sources_component(
            dez, df, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      } else {
        add_sources_component(
            ey, f0, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
        add_sources_component(
            dey, df, sources_i, source_time_offset, n_shots, shot_numel, n_sources_per_shot);
      }
    }

    int64_t const recv_time_offset = t * n_shots * n_receivers_per_shot;
    if (n_receivers_per_shot > 0 && r != NULL && receivers_i != NULL) {
      if (receiver_component == 0) {
        record_receivers_component(
            r, dex, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      } else if (receiver_component == 2) {
        record_receivers_component(
            r, dez, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      } else {
        record_receivers_component(
            r, dey, receivers_i, recv_time_offset, n_shots, shot_numel, n_receivers_per_shot);
      }
    }
  }

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

TIDE_EXTERN_C TIDE_EXPORT void FUNC(backward)(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const grad_r,
    TIDE_DTYPE *__restrict const lambda_ex,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const lambda_ez,
    TIDE_DTYPE *__restrict const lambda_hx,
    TIDE_DTYPE *__restrict const lambda_hy,
    TIDE_DTYPE *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
    TIDE_DTYPE *__restrict const m_lambda_ez_y,
    TIDE_DTYPE *__restrict const m_lambda_ez_x,
    TIDE_DTYPE *__restrict const m_lambda_ex_z,
    TIDE_DTYPE *__restrict const m_lambda_ex_y,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE *__restrict const m_lambda_hz_y,
    TIDE_DTYPE *__restrict const m_lambda_hy_z,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const m_lambda_hy_x,
    TIDE_DTYPE *__restrict const m_lambda_hx_y,
    TIDE_DTYPE *__restrict const store_1,
    TIDE_DTYPE *__restrict const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *__restrict const store_3,
    TIDE_DTYPE *__restrict const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *__restrict const store_5,
    TIDE_DTYPE *__restrict const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *__restrict const store_7,
    TIDE_DTYPE *__restrict const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *__restrict const store_9,
    TIDE_DTYPE *__restrict const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *__restrict const store_11,
    TIDE_DTYPE *__restrict const store_12,
    char **store_filenames_6,
    TIDE_DTYPE *__restrict const grad_f,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    bool const zero_grad_on_entry,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    int64_t const storage_mode,
    int64_t const storage_format,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_z0,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_z1,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)device;
  (void)execution_backend;
  (void)compute_stream_handle;
  (void)storage_stream_handle;
  (void)storage_format;
  (void)store_2;
  (void)store_filenames_1;
  (void)store_4;
  (void)store_filenames_2;
  (void)store_6;
  (void)store_filenames_3;
  (void)store_8;
  (void)store_filenames_4;
  (void)store_10;
  (void)store_filenames_5;
  (void)store_12;
  (void)store_filenames_6;
  (void)az;

#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = nz * ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  int64_t const step_ratio_eff = step_ratio > 0 ? step_ratio : 1;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      (storage_format == STORAGE_FORMAT_FULL) &&
      (shot_bytes_uncomp == (int64_t)(shot_numel * (int64_t)sizeof(TIDE_DTYPE)));
  bool const reduce_grad_ca =
      ca_requires_grad && !ca_batched && grad_ca != NULL && grad_ca_shot != NULL;
  bool const reduce_grad_cb =
      cb_requires_grad && !cb_batched && grad_cb != NULL && grad_cb_shot != NULL;
  TIDE_DTYPE *__restrict grad_ca_accum = reduce_grad_ca ? grad_ca_shot : grad_ca;
  TIDE_DTYPE *__restrict grad_cb_accum = reduce_grad_cb ? grad_cb_shot : grad_cb;

  if (zero_grad_on_entry) {
    if (grad_f != NULL && nt > 0 && n_shots > 0 && n_sources_per_shot > 0) {
      size_t numel = (size_t)nt * (size_t)n_shots * (size_t)n_sources_per_shot;
      tide_zero_if_not_null(grad_f, numel * sizeof(TIDE_DTYPE));
    }
    if (ca_requires_grad && grad_ca != NULL) {
      size_t n = (size_t)(ca_batched ? n_shots : 1) * (size_t)shot_numel;
      tide_zero_if_not_null(grad_ca, n * sizeof(TIDE_DTYPE));
    }
    if (cb_requires_grad && grad_cb != NULL) {
      size_t n = (size_t)(cb_batched ? n_shots : 1) * (size_t)shot_numel;
      tide_zero_if_not_null(grad_cb, n * sizeof(TIDE_DTYPE));
    }
    if ((ca_requires_grad || cb_requires_grad) && grad_eps != NULL) {
      size_t n = (size_t)(ca_batched ? n_shots : 1) * (size_t)shot_numel;
      tide_zero_if_not_null(grad_eps, n * sizeof(TIDE_DTYPE));
    }
    if ((ca_requires_grad || cb_requires_grad) && grad_sigma != NULL) {
      size_t n = (size_t)(ca_batched ? n_shots : 1) * (size_t)shot_numel;
      tide_zero_if_not_null(grad_sigma, n * sizeof(TIDE_DTYPE));
    }
  }
  if (reduce_grad_ca) {
    tide_zero_if_not_null(grad_ca_shot, (size_t)store_size * sizeof(TIDE_DTYPE));
  }
  if (reduce_grad_cb) {
    tide_zero_if_not_null(grad_cb_shot, (size_t)store_size * sizeof(TIDE_DTYPE));
  }

  TIDE_DTYPE *__restrict lambda_src_field = lambda_ey;
  if (source_component == 0) {
    lambda_src_field = lambda_ex;
  } else if (source_component == 2) {
    lambda_src_field = lambda_ez;
  }
  TIDE_DTYPE *__restrict lambda_recv_field = lambda_ey;
  if (receiver_component == 0) {
    lambda_recv_field = lambda_ex;
  } else if (receiver_component == 2) {
    lambda_recv_field = lambda_ez;
  }

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    bool const do_grad = (t % step_ratio_eff) == 0;
    bool const grad_ca_step = do_grad && ca_requires_grad && storage_direct &&
                              store_1 != NULL && store_3 != NULL &&
                              store_5 != NULL;
    bool const grad_cb_step = do_grad && cb_requires_grad && storage_direct &&
                              store_7 != NULL && store_9 != NULL &&
                              store_11 != NULL;

    int64_t const store_idx = t / step_ratio_eff;
    int64_t const store_offset = store_idx * store_size;

    TIDE_DTYPE const *__restrict const ex_store =
        grad_ca_step ? (store_1 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const ey_store =
        grad_ca_step ? (store_3 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const ez_store =
        grad_ca_step ? (store_5 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_x_store =
        grad_cb_step ? (store_7 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_y_store =
        grad_cb_step ? (store_9 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_z_store =
        grad_cb_step ? (store_11 + store_offset) : NULL;

    if (n_receivers_per_shot > 0 && grad_r != NULL && receivers_i != NULL) {
      add_sources_component(
          lambda_recv_field, grad_r, receivers_i,
          t * n_shots * n_receivers_per_shot, n_shots, shot_numel,
          n_receivers_per_shot);
    }

    if (n_sources_per_shot > 0 && grad_f != NULL && sources_i != NULL) {
      record_receivers_component(
          grad_f, lambda_src_field, sources_i,
          t * n_shots * n_sources_per_shot, n_shots, shot_numel,
          n_sources_per_shot);
    }

    TIDE_OMP_INDEX shot_idx;
    /* λ_H update: λ_H^{n+1} <- λ_H^{n+1} + (dE^{n+1}/dH^{n+1})^T λ_E^{n+1}. */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dLey_dz_pml = 0;
            TIDE_DTYPE dLez_dy_pml = 0;
            TIDE_DTYPE dLez_dx_pml = 0;
            TIDE_DTYPE dLex_dz_pml = 0;
            TIDE_DTYPE dLex_dy_pml = 0;
            TIDE_DTYPE dLey_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dLey_dz = -DIFFZ1_ADJ(CB_AT, LAMBDA_EY);
              if (pml_z_h) {
                M_LAMBDA_EY_Z(0, 0, 0) =
                    bzh[z] * M_LAMBDA_EY_Z(0, 0, 0) + azh[z] * dLey_dz;
                dLey_dz = dLey_dz / kzh[z] + M_LAMBDA_EY_Z(0, 0, 0);
              }
              dLey_dz_pml = dLey_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dLez_dy = -DIFFY1_ADJ(CB_AT, LAMBDA_EZ);
              if (pml_y_h) {
                M_LAMBDA_EZ_Y(0, 0, 0) =
                    byh[y] * M_LAMBDA_EZ_Y(0, 0, 0) + ayh[y] * dLez_dy;
                dLez_dy = dLez_dy / kyh[y] + M_LAMBDA_EZ_Y(0, 0, 0);
              }
              dLez_dy_pml = dLez_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dLez_dx = -DIFFX1_ADJ(CB_AT, LAMBDA_EZ);
              if (pml_x_h) {
                M_LAMBDA_EZ_X(0, 0, 0) =
                    bxh[x] * M_LAMBDA_EZ_X(0, 0, 0) + axh[x] * dLez_dx;
                dLez_dx = dLez_dx / kxh[x] + M_LAMBDA_EZ_X(0, 0, 0);
              }
              dLez_dx_pml = dLez_dx;
            }
            if (z < nz - FD_PAD) {
              TIDE_DTYPE dLex_dz = -DIFFZ1_ADJ(CB_AT, LAMBDA_EX);
              if (pml_z_h) {
                M_LAMBDA_EX_Z(0, 0, 0) =
                    bzh[z] * M_LAMBDA_EX_Z(0, 0, 0) + azh[z] * dLex_dz;
                dLex_dz = dLex_dz / kzh[z] + M_LAMBDA_EX_Z(0, 0, 0);
              }
              dLex_dz_pml = dLex_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dLex_dy = -DIFFY1_ADJ(CB_AT, LAMBDA_EX);
              if (pml_y_h) {
                M_LAMBDA_EX_Y(0, 0, 0) =
                    byh[y] * M_LAMBDA_EX_Y(0, 0, 0) + ayh[y] * dLex_dy;
                dLex_dy = dLex_dy / kyh[y] + M_LAMBDA_EX_Y(0, 0, 0);
              }
              dLex_dy_pml = dLex_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dLey_dx = -DIFFX1_ADJ(CB_AT, LAMBDA_EY);
              if (pml_x_h) {
                M_LAMBDA_EY_X(0, 0, 0) =
                    bxh[x] * M_LAMBDA_EY_X(0, 0, 0) + axh[x] * dLey_dx;
                dLey_dx = dLey_dx / kxh[x] + M_LAMBDA_EY_X(0, 0, 0);
              }
              dLey_dx_pml = dLey_dx;
            }

            LAMBDA_HX(0, 0, 0) += dLey_dz_pml - dLez_dy_pml;
            LAMBDA_HY(0, 0, 0) += dLez_dx_pml - dLex_dz_pml;
            LAMBDA_HZ(0, 0, 0) += dLex_dy_pml - dLey_dx_pml;
          }
        }
      }
    }

    /* λ_E update and gradient accumulation. */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            int64_t const idx = IDX(z, y, x);
            int64_t const idx_shot = shot_idx * shot_numel + idx;
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dLhy_dz = DIFFZH1_ADJ(CQ_AT, LAMBDA_HY);
            TIDE_DTYPE dLhz_dy = DIFFYH1_ADJ(CQ_AT, LAMBDA_HZ);
            TIDE_DTYPE dLhz_dx = DIFFXH1_ADJ(CQ_AT, LAMBDA_HZ);
            TIDE_DTYPE dLhx_dz = DIFFZH1_ADJ(CQ_AT, LAMBDA_HX);
            TIDE_DTYPE dLhx_dy = DIFFYH1_ADJ(CQ_AT, LAMBDA_HX);
            TIDE_DTYPE dLhy_dx = DIFFXH1_ADJ(CQ_AT, LAMBDA_HY);

            if (pml_z) {
              M_LAMBDA_HY_Z(0, 0, 0) =
                  bz[z] * M_LAMBDA_HY_Z(0, 0, 0) + az[z] * dLhy_dz;
              dLhy_dz = dLhy_dz / kz[z] + M_LAMBDA_HY_Z(0, 0, 0);
              M_LAMBDA_HX_Z(0, 0, 0) =
                  bz[z] * M_LAMBDA_HX_Z(0, 0, 0) + az[z] * dLhx_dz;
              dLhx_dz = dLhx_dz / kz[z] + M_LAMBDA_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              M_LAMBDA_HZ_Y(0, 0, 0) =
                  by[y] * M_LAMBDA_HZ_Y(0, 0, 0) + ay[y] * dLhz_dy;
              dLhz_dy = dLhz_dy / ky[y] + M_LAMBDA_HZ_Y(0, 0, 0);
              M_LAMBDA_HX_Y(0, 0, 0) =
                  by[y] * M_LAMBDA_HX_Y(0, 0, 0) + ay[y] * dLhx_dy;
              dLhx_dy = dLhx_dy / ky[y] + M_LAMBDA_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              M_LAMBDA_HZ_X(0, 0, 0) =
                  bx[x] * M_LAMBDA_HZ_X(0, 0, 0) + ax[x] * dLhz_dx;
              dLhz_dx = dLhz_dx / kx[x] + M_LAMBDA_HZ_X(0, 0, 0);
              M_LAMBDA_HY_X(0, 0, 0) =
                  bx[x] * M_LAMBDA_HY_X(0, 0, 0) + ax[x] * dLhy_dx;
              dLhy_dx = dLhy_dx / kx[x] + M_LAMBDA_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const curl_lambda_x = dLhy_dz - dLhz_dy;
            TIDE_DTYPE const curl_lambda_y = dLhz_dx - dLhx_dz;
            TIDE_DTYPE const curl_lambda_z = dLhx_dy - dLhy_dx;

            TIDE_DTYPE const lex_curr = LAMBDA_EX(0, 0, 0);
            TIDE_DTYPE const ley_curr = LAMBDA_EY(0, 0, 0);
            TIDE_DTYPE const lez_curr = LAMBDA_EZ(0, 0, 0);

            LAMBDA_EX(0, 0, 0) = ca_val * lex_curr + curl_lambda_x;
            LAMBDA_EY(0, 0, 0) = ca_val * ley_curr + curl_lambda_y;
            LAMBDA_EZ(0, 0, 0) = ca_val * lez_curr + curl_lambda_z;

            if (grad_ca_step) {
              TIDE_DTYPE const acc_ca =
                  lex_curr * ex_store[idx_shot] +
                  ley_curr * ey_store[idx_shot] +
                  lez_curr * ez_store[idx_shot];
              if (ca_batched || reduce_grad_ca) {
                grad_ca_accum[idx_shot] += acc_ca * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_ca[idx] += acc_ca * (TIDE_DTYPE)step_ratio_eff;
              }
            }
            if (grad_cb_step) {
              TIDE_DTYPE const acc_cb =
                  lex_curr * curl_x_store[idx_shot] +
                  ley_curr * curl_y_store[idx_shot] +
                  lez_curr * curl_z_store[idx_shot];
              if (cb_batched || reduce_grad_cb) {
                grad_cb_accum[idx_shot] += acc_cb * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_cb[idx] += acc_cb * (TIDE_DTYPE)step_ratio_eff;
              }
            }
          }
        }
      }
    }
  }

  if (reduce_grad_ca) {
    combine_grad_shot_3d(grad_ca, grad_ca_shot, n_shots, shot_numel);
  }
  if (reduce_grad_cb) {
    combine_grad_shot_3d(grad_cb, grad_cb_shot, n_shots, shot_numel);
  }

  convert_grad_ca_cb_to_eps_sigma_3d(
      ca, cb, grad_ca, grad_cb, grad_eps, grad_sigma, dt, n_shots, nz, ny, nx,
      ca_batched, cb_batched, ca_requires_grad, cb_requires_grad);

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

TIDE_EXTERN_C TIDE_EXPORT void FUNC(born_backward_bggrad)(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const f0,
    TIDE_DTYPE const *__restrict const df,
    TIDE_DTYPE const *__restrict const grad_r,
    TIDE_DTYPE *__restrict const lambda_ex,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const lambda_ez,
    TIDE_DTYPE *__restrict const lambda_hx,
    TIDE_DTYPE *__restrict const lambda_hy,
    TIDE_DTYPE *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
    TIDE_DTYPE *__restrict const m_lambda_ez_y,
    TIDE_DTYPE *__restrict const m_lambda_ez_x,
    TIDE_DTYPE *__restrict const m_lambda_ex_z,
    TIDE_DTYPE *__restrict const m_lambda_ex_y,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE *__restrict const m_lambda_hz_y,
    TIDE_DTYPE *__restrict const m_lambda_hy_z,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const m_lambda_hy_x,
    TIDE_DTYPE *__restrict const m_lambda_hx_y,
    TIDE_DTYPE *__restrict const eta_ex,
    TIDE_DTYPE *__restrict const eta_ey,
    TIDE_DTYPE *__restrict const eta_ez,
    TIDE_DTYPE *__restrict const eta_hx,
    TIDE_DTYPE *__restrict const eta_hy,
    TIDE_DTYPE *__restrict const eta_hz,
    TIDE_DTYPE *__restrict const m_eta_ey_z,
    TIDE_DTYPE *__restrict const m_eta_ez_y,
    TIDE_DTYPE *__restrict const m_eta_ez_x,
    TIDE_DTYPE *__restrict const m_eta_ex_z,
    TIDE_DTYPE *__restrict const m_eta_ex_y,
    TIDE_DTYPE *__restrict const m_eta_ey_x,
    TIDE_DTYPE *__restrict const m_eta_hz_y,
    TIDE_DTYPE *__restrict const m_eta_hy_z,
    TIDE_DTYPE *__restrict const m_eta_hx_z,
    TIDE_DTYPE *__restrict const m_eta_hz_x,
    TIDE_DTYPE *__restrict const m_eta_hy_x,
    TIDE_DTYPE *__restrict const m_eta_hx_y,
    TIDE_DTYPE *__restrict const store_1,
    TIDE_DTYPE *__restrict const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *__restrict const store_3,
    TIDE_DTYPE *__restrict const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *__restrict const store_5,
    TIDE_DTYPE *__restrict const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *__restrict const store_7,
    TIDE_DTYPE *__restrict const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *__restrict const store_9,
    TIDE_DTYPE *__restrict const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *__restrict const store_11,
    TIDE_DTYPE *__restrict const store_12,
    char **store_filenames_6,
    TIDE_DTYPE const *__restrict const dstore_ex,
    TIDE_DTYPE const *__restrict const dstore_ey,
    TIDE_DTYPE const *__restrict const dstore_ez,
    TIDE_DTYPE const *__restrict const dstore_curl_x,
    TIDE_DTYPE const *__restrict const dstore_curl_y,
    TIDE_DTYPE const *__restrict const dstore_curl_z,
    TIDE_DTYPE *__restrict const grad_f0,
    TIDE_DTYPE *__restrict const grad_df,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_dca,
    TIDE_DTYPE *__restrict const grad_dcb,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    TIDE_DTYPE *__restrict const grad_dca_shot,
    TIDE_DTYPE *__restrict const grad_dcb_shot,
    TIDE_DTYPE *__restrict const eta_source_ex,
    TIDE_DTYPE *__restrict const eta_source_ey,
    TIDE_DTYPE *__restrict const eta_source_ez,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    int64_t const storage_mode,
    int64_t const storage_format,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const dca_requires_grad,
    bool const dcb_requires_grad,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_z0,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_z1,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)device;
  (void)execution_backend;
  (void)compute_stream_handle;
  (void)storage_stream_handle;
  (void)dt;
  (void)f0;
  (void)df;
  (void)store_2;
  (void)store_filenames_1;
  (void)store_4;
  (void)store_filenames_2;
  (void)store_6;
  (void)store_filenames_3;
  (void)store_8;
  (void)store_filenames_4;
  (void)store_10;
  (void)store_filenames_5;
  (void)store_12;
  (void)store_filenames_6;

#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = nz * ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  int64_t const step_ratio_eff = step_ratio > 0 ? step_ratio : 1;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      (storage_format == STORAGE_FORMAT_FULL) &&
      (shot_bytes_uncomp == (int64_t)(shot_numel * (int64_t)sizeof(TIDE_DTYPE)));
  if (!storage_direct) {
    fprintf(stderr,
            "born_backward_bggrad currently supports full-precision device "
            "storage only for 3D CPU.\n");
    abort();
  }

  bool const reduce_grad_ca =
      ca_requires_grad && !ca_batched && grad_ca != NULL && grad_ca_shot != NULL;
  bool const reduce_grad_cb =
      cb_requires_grad && !cb_batched && grad_cb != NULL && grad_cb_shot != NULL;
  bool const reduce_grad_dca =
      dca_requires_grad && !ca_batched && grad_dca != NULL && grad_dca_shot != NULL;
  bool const reduce_grad_dcb =
      dcb_requires_grad && !cb_batched && grad_dcb != NULL && grad_dcb_shot != NULL;
  TIDE_DTYPE *__restrict grad_ca_accum = reduce_grad_ca ? grad_ca_shot : grad_ca;
  TIDE_DTYPE *__restrict grad_cb_accum = reduce_grad_cb ? grad_cb_shot : grad_cb;
  TIDE_DTYPE *__restrict grad_dca_accum = reduce_grad_dca ? grad_dca_shot : grad_dca;
  TIDE_DTYPE *__restrict grad_dcb_accum = reduce_grad_dcb ? grad_dcb_shot : grad_dcb;

  size_t const state_bytes = (size_t)store_size * sizeof(TIDE_DTYPE);
  tide_zero_if_not_null(lambda_ex, state_bytes);
  tide_zero_if_not_null(lambda_ey, state_bytes);
  tide_zero_if_not_null(lambda_ez, state_bytes);
  tide_zero_if_not_null(lambda_hx, state_bytes);
  tide_zero_if_not_null(lambda_hy, state_bytes);
  tide_zero_if_not_null(lambda_hz, state_bytes);
  tide_zero_if_not_null(eta_ex, state_bytes);
  tide_zero_if_not_null(eta_ey, state_bytes);
  tide_zero_if_not_null(eta_ez, state_bytes);
  tide_zero_if_not_null(eta_hx, state_bytes);
  tide_zero_if_not_null(eta_hy, state_bytes);
  tide_zero_if_not_null(eta_hz, state_bytes);
  tide_zero_if_not_null(m_lambda_ey_z, state_bytes);
  tide_zero_if_not_null(m_lambda_ez_y, state_bytes);
  tide_zero_if_not_null(m_lambda_ez_x, state_bytes);
  tide_zero_if_not_null(m_lambda_ex_z, state_bytes);
  tide_zero_if_not_null(m_lambda_ex_y, state_bytes);
  tide_zero_if_not_null(m_lambda_ey_x, state_bytes);
  tide_zero_if_not_null(m_lambda_hz_y, state_bytes);
  tide_zero_if_not_null(m_lambda_hy_z, state_bytes);
  tide_zero_if_not_null(m_lambda_hx_z, state_bytes);
  tide_zero_if_not_null(m_lambda_hz_x, state_bytes);
  tide_zero_if_not_null(m_lambda_hy_x, state_bytes);
  tide_zero_if_not_null(m_lambda_hx_y, state_bytes);
  tide_zero_if_not_null(m_eta_ey_z, state_bytes);
  tide_zero_if_not_null(m_eta_ez_y, state_bytes);
  tide_zero_if_not_null(m_eta_ez_x, state_bytes);
  tide_zero_if_not_null(m_eta_ex_z, state_bytes);
  tide_zero_if_not_null(m_eta_ex_y, state_bytes);
  tide_zero_if_not_null(m_eta_ey_x, state_bytes);
  tide_zero_if_not_null(m_eta_hz_y, state_bytes);
  tide_zero_if_not_null(m_eta_hy_z, state_bytes);
  tide_zero_if_not_null(m_eta_hx_z, state_bytes);
  tide_zero_if_not_null(m_eta_hz_x, state_bytes);
  tide_zero_if_not_null(m_eta_hy_x, state_bytes);
  tide_zero_if_not_null(m_eta_hx_y, state_bytes);
  tide_zero_if_not_null(eta_source_ex, state_bytes);
  tide_zero_if_not_null(eta_source_ey, state_bytes);
  tide_zero_if_not_null(eta_source_ez, state_bytes);

  if (grad_f0 != NULL && nt > 0 && n_shots > 0 && n_sources_per_shot > 0) {
    size_t const numel =
        (size_t)nt * (size_t)n_shots * (size_t)n_sources_per_shot;
    tide_zero_if_not_null(grad_f0, numel * sizeof(TIDE_DTYPE));
  }
  if (grad_df != NULL && nt > 0 && n_shots > 0 && n_sources_per_shot > 0) {
    size_t const numel =
        (size_t)nt * (size_t)n_shots * (size_t)n_sources_per_shot;
    tide_zero_if_not_null(grad_df, numel * sizeof(TIDE_DTYPE));
  }
  if (ca_requires_grad && grad_ca != NULL) {
    size_t const n = (size_t)(ca_batched ? n_shots : 1) * (size_t)shot_numel;
    tide_zero_if_not_null(grad_ca, n * sizeof(TIDE_DTYPE));
  }
  if (cb_requires_grad && grad_cb != NULL) {
    size_t const n = (size_t)(cb_batched ? n_shots : 1) * (size_t)shot_numel;
    tide_zero_if_not_null(grad_cb, n * sizeof(TIDE_DTYPE));
  }
  if (dca_requires_grad && grad_dca != NULL) {
    size_t const n = (size_t)(ca_batched ? n_shots : 1) * (size_t)shot_numel;
    tide_zero_if_not_null(grad_dca, n * sizeof(TIDE_DTYPE));
  }
  if (dcb_requires_grad && grad_dcb != NULL) {
    size_t const n = (size_t)(cb_batched ? n_shots : 1) * (size_t)shot_numel;
    tide_zero_if_not_null(grad_dcb, n * sizeof(TIDE_DTYPE));
  }
  if (reduce_grad_ca) {
    tide_zero_if_not_null(grad_ca_shot, state_bytes);
  }
  if (reduce_grad_cb) {
    tide_zero_if_not_null(grad_cb_shot, state_bytes);
  }
  if (reduce_grad_dca) {
    tide_zero_if_not_null(grad_dca_shot, state_bytes);
  }
  if (reduce_grad_dcb) {
    tide_zero_if_not_null(grad_dcb_shot, state_bytes);
  }

  TIDE_DTYPE *__restrict lambda_src_field = lambda_ey;
  TIDE_DTYPE *__restrict eta_src_field = eta_ey;
  if (source_component == 0) {
    lambda_src_field = lambda_ex;
    eta_src_field = eta_ex;
  } else if (source_component == 2) {
    lambda_src_field = lambda_ez;
    eta_src_field = eta_ez;
  }
  TIDE_DTYPE *__restrict lambda_recv_field = lambda_ey;
  if (receiver_component == 0) {
    lambda_recv_field = lambda_ex;
  } else if (receiver_component == 2) {
    lambda_recv_field = lambda_ez;
  }

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    bool const do_grad = (t % step_ratio_eff) == 0;
    bool const grad_ca_step =
        do_grad && ca_requires_grad && store_1 != NULL && store_3 != NULL &&
        store_5 != NULL;
    bool const grad_cb_step =
        do_grad && cb_requires_grad && store_7 != NULL && store_9 != NULL &&
        store_11 != NULL;
    bool const grad_dca_step =
        do_grad && dca_requires_grad && store_1 != NULL && store_3 != NULL &&
        store_5 != NULL;
    bool const grad_dcb_step =
        do_grad && dcb_requires_grad && store_7 != NULL && store_9 != NULL &&
        store_11 != NULL;
    bool const direct_ca_step =
        do_grad && ca_requires_grad && dstore_ex != NULL && dstore_ey != NULL &&
        dstore_ez != NULL;
    bool const direct_cb_step =
        do_grad && cb_requires_grad && dstore_curl_x != NULL &&
        dstore_curl_y != NULL && dstore_curl_z != NULL;

    int64_t const store_idx = t / step_ratio_eff;
    int64_t const store_offset = store_idx * store_size;

    TIDE_DTYPE const *__restrict const ex_store =
        (grad_ca_step || grad_dca_step) ? (store_1 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const ey_store =
        (grad_ca_step || grad_dca_step) ? (store_3 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const ez_store =
        (grad_ca_step || grad_dca_step) ? (store_5 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_x_store =
        (grad_cb_step || grad_dcb_step) ? (store_7 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_y_store =
        (grad_cb_step || grad_dcb_step) ? (store_9 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_z_store =
        (grad_cb_step || grad_dcb_step) ? (store_11 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const dex_store =
        direct_ca_step ? (dstore_ex + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const dey_store =
        direct_ca_step ? (dstore_ey + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const dez_store =
        direct_ca_step ? (dstore_ez + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const dcurl_x_store =
        direct_cb_step ? (dstore_curl_x + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const dcurl_y_store =
        direct_cb_step ? (dstore_curl_y + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const dcurl_z_store =
        direct_cb_step ? (dstore_curl_z + store_offset) : NULL;

    if (n_receivers_per_shot > 0 && grad_r != NULL && receivers_i != NULL) {
      add_sources_component(
          lambda_recv_field, grad_r, receivers_i,
          t * n_shots * n_receivers_per_shot, n_shots, shot_numel,
          n_receivers_per_shot);
    }
    if (n_sources_per_shot > 0 && sources_i != NULL) {
      if (grad_df != NULL) {
        record_receivers_component(
            grad_df, lambda_src_field, sources_i,
            t * n_shots * n_sources_per_shot, n_shots, shot_numel,
            n_sources_per_shot);
      }
      if (grad_f0 != NULL) {
        record_receivers_component(
            grad_f0, eta_src_field, sources_i,
            t * n_shots * n_sources_per_shot, n_shots, shot_numel,
            n_sources_per_shot);
      }
    }

    TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            int64_t const idx = IDX(z, y, x);
            int64_t const idx_shot = shot_idx * shot_numel + idx;
            TIDE_DTYPE const lex_curr = LAMBDA_EX(0, 0, 0);
            TIDE_DTYPE const ley_curr = LAMBDA_EY(0, 0, 0);
            TIDE_DTYPE const lez_curr = LAMBDA_EZ(0, 0, 0);

            eta_source_ex[idx_shot] = DCA_AT(0, 0, 0) * lex_curr;
            eta_source_ey[idx_shot] = DCA_AT(0, 0, 0) * ley_curr;
            eta_source_ez[idx_shot] = DCA_AT(0, 0, 0) * lez_curr;

            if (direct_ca_step) {
              TIDE_DTYPE const acc_ca =
                  lex_curr * dex_store[idx_shot] +
                  ley_curr * dey_store[idx_shot] +
                  lez_curr * dez_store[idx_shot];
              if (ca_batched || reduce_grad_ca) {
                grad_ca_accum[idx_shot] +=
                    acc_ca * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_ca[idx] += acc_ca * (TIDE_DTYPE)step_ratio_eff;
              }
            }
            if (direct_cb_step) {
              TIDE_DTYPE const acc_cb =
                  lex_curr * dcurl_x_store[idx_shot] +
                  ley_curr * dcurl_y_store[idx_shot] +
                  lez_curr * dcurl_z_store[idx_shot];
              if (cb_batched || reduce_grad_cb) {
                grad_cb_accum[idx_shot] +=
                    acc_cb * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_cb[idx] += acc_cb * (TIDE_DTYPE)step_ratio_eff;
              }
            }
          }
        }
      }
    }

    /* Direct dcb * lambda contribution into the background H adjoint. */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dLey_dz = 0;
            TIDE_DTYPE dLez_dy = 0;
            TIDE_DTYPE dLez_dx = 0;
            TIDE_DTYPE dLex_dz = 0;
            TIDE_DTYPE dLex_dy = 0;
            TIDE_DTYPE dLey_dx = 0;

            if (z < nz - FD_PAD) {
              dLey_dz = -DIFFZ1_ADJ(DCB_AT, LAMBDA_EY);
              if (pml_z_h) {
                dLey_dz = dLey_dz / kzh[z] + azh[z] * dLey_dz;
              }
            }
            if (y < ny - FD_PAD) {
              dLez_dy = -DIFFY1_ADJ(DCB_AT, LAMBDA_EZ);
              if (pml_y_h) {
                dLez_dy = dLez_dy / kyh[y] + ayh[y] * dLez_dy;
              }
            }
            if (x < nx - FD_PAD) {
              dLez_dx = -DIFFX1_ADJ(DCB_AT, LAMBDA_EZ);
              if (pml_x_h) {
                dLez_dx = dLez_dx / kxh[x] + axh[x] * dLez_dx;
              }
            }
            if (z < nz - FD_PAD) {
              dLex_dz = -DIFFZ1_ADJ(DCB_AT, LAMBDA_EX);
              if (pml_z_h) {
                dLex_dz = dLex_dz / kzh[z] + azh[z] * dLex_dz;
              }
            }
            if (y < ny - FD_PAD) {
              dLex_dy = -DIFFY1_ADJ(DCB_AT, LAMBDA_EX);
              if (pml_y_h) {
                dLex_dy = dLex_dy / kyh[y] + ayh[y] * dLex_dy;
              }
            }
            if (x < nx - FD_PAD) {
              dLey_dx = -DIFFX1_ADJ(DCB_AT, LAMBDA_EY);
              if (pml_x_h) {
                dLey_dx = dLey_dx / kxh[x] + axh[x] * dLey_dx;
              }
            }

            ETA_HX(0, 0, 0) += dLey_dz - dLez_dy;
            ETA_HY(0, 0, 0) += dLez_dx - dLex_dz;
            ETA_HZ(0, 0, 0) += dLex_dy - dLey_dx;
          }
        }
      }
    }

    /* Background E update transpose for eta. */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dEey_dz = 0;
            TIDE_DTYPE dEez_dy = 0;
            TIDE_DTYPE dEez_dx = 0;
            TIDE_DTYPE dEex_dz = 0;
            TIDE_DTYPE dEex_dy = 0;
            TIDE_DTYPE dEey_dx = 0;

            if (z < nz - FD_PAD) {
              dEey_dz = -DIFFZ1_ADJ(CB_AT, ETA_EY);
              if (pml_z_h) {
                M_ETA_EY_Z(0, 0, 0) =
                    bzh[z] * M_ETA_EY_Z(0, 0, 0) + azh[z] * dEey_dz;
                dEey_dz = dEey_dz / kzh[z] + M_ETA_EY_Z(0, 0, 0);
              }
            }
            if (y < ny - FD_PAD) {
              dEez_dy = -DIFFY1_ADJ(CB_AT, ETA_EZ);
              if (pml_y_h) {
                M_ETA_EZ_Y(0, 0, 0) =
                    byh[y] * M_ETA_EZ_Y(0, 0, 0) + ayh[y] * dEez_dy;
                dEez_dy = dEez_dy / kyh[y] + M_ETA_EZ_Y(0, 0, 0);
              }
            }
            if (x < nx - FD_PAD) {
              dEez_dx = -DIFFX1_ADJ(CB_AT, ETA_EZ);
              if (pml_x_h) {
                M_ETA_EZ_X(0, 0, 0) =
                    bxh[x] * M_ETA_EZ_X(0, 0, 0) + axh[x] * dEez_dx;
                dEez_dx = dEez_dx / kxh[x] + M_ETA_EZ_X(0, 0, 0);
              }
            }
            if (z < nz - FD_PAD) {
              dEex_dz = -DIFFZ1_ADJ(CB_AT, ETA_EX);
              if (pml_z_h) {
                M_ETA_EX_Z(0, 0, 0) =
                    bzh[z] * M_ETA_EX_Z(0, 0, 0) + azh[z] * dEex_dz;
                dEex_dz = dEex_dz / kzh[z] + M_ETA_EX_Z(0, 0, 0);
              }
            }
            if (y < ny - FD_PAD) {
              dEex_dy = -DIFFY1_ADJ(CB_AT, ETA_EX);
              if (pml_y_h) {
                M_ETA_EX_Y(0, 0, 0) =
                    byh[y] * M_ETA_EX_Y(0, 0, 0) + ayh[y] * dEex_dy;
                dEex_dy = dEex_dy / kyh[y] + M_ETA_EX_Y(0, 0, 0);
              }
            }
            if (x < nx - FD_PAD) {
              dEey_dx = -DIFFX1_ADJ(CB_AT, ETA_EY);
              if (pml_x_h) {
                M_ETA_EY_X(0, 0, 0) =
                    bxh[x] * M_ETA_EY_X(0, 0, 0) + axh[x] * dEey_dx;
                dEey_dx = dEey_dx / kxh[x] + M_ETA_EY_X(0, 0, 0);
              }
            }

            ETA_HX(0, 0, 0) += dEey_dz - dEez_dy;
            ETA_HY(0, 0, 0) += dEez_dx - dEex_dz;
            ETA_HZ(0, 0, 0) += dEex_dy - dEey_dx;
          }
        }
      }
    }

    /* Scattered E update transpose for lambda. */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dLey_dz = 0;
            TIDE_DTYPE dLez_dy = 0;
            TIDE_DTYPE dLez_dx = 0;
            TIDE_DTYPE dLex_dz = 0;
            TIDE_DTYPE dLex_dy = 0;
            TIDE_DTYPE dLey_dx = 0;

            if (z < nz - FD_PAD) {
              dLey_dz = -DIFFZ1_ADJ(CB_AT, LAMBDA_EY);
              if (pml_z_h) {
                M_LAMBDA_EY_Z(0, 0, 0) =
                    bzh[z] * M_LAMBDA_EY_Z(0, 0, 0) + azh[z] * dLey_dz;
                dLey_dz = dLey_dz / kzh[z] + M_LAMBDA_EY_Z(0, 0, 0);
              }
            }
            if (y < ny - FD_PAD) {
              dLez_dy = -DIFFY1_ADJ(CB_AT, LAMBDA_EZ);
              if (pml_y_h) {
                M_LAMBDA_EZ_Y(0, 0, 0) =
                    byh[y] * M_LAMBDA_EZ_Y(0, 0, 0) + ayh[y] * dLez_dy;
                dLez_dy = dLez_dy / kyh[y] + M_LAMBDA_EZ_Y(0, 0, 0);
              }
            }
            if (x < nx - FD_PAD) {
              dLez_dx = -DIFFX1_ADJ(CB_AT, LAMBDA_EZ);
              if (pml_x_h) {
                M_LAMBDA_EZ_X(0, 0, 0) =
                    bxh[x] * M_LAMBDA_EZ_X(0, 0, 0) + axh[x] * dLez_dx;
                dLez_dx = dLez_dx / kxh[x] + M_LAMBDA_EZ_X(0, 0, 0);
              }
            }
            if (z < nz - FD_PAD) {
              dLex_dz = -DIFFZ1_ADJ(CB_AT, LAMBDA_EX);
              if (pml_z_h) {
                M_LAMBDA_EX_Z(0, 0, 0) =
                    bzh[z] * M_LAMBDA_EX_Z(0, 0, 0) + azh[z] * dLex_dz;
                dLex_dz = dLex_dz / kzh[z] + M_LAMBDA_EX_Z(0, 0, 0);
              }
            }
            if (y < ny - FD_PAD) {
              dLex_dy = -DIFFY1_ADJ(CB_AT, LAMBDA_EX);
              if (pml_y_h) {
                M_LAMBDA_EX_Y(0, 0, 0) =
                    byh[y] * M_LAMBDA_EX_Y(0, 0, 0) + ayh[y] * dLex_dy;
                dLex_dy = dLex_dy / kyh[y] + M_LAMBDA_EX_Y(0, 0, 0);
              }
            }
            if (x < nx - FD_PAD) {
              dLey_dx = -DIFFX1_ADJ(CB_AT, LAMBDA_EY);
              if (pml_x_h) {
                M_LAMBDA_EY_X(0, 0, 0) =
                    bxh[x] * M_LAMBDA_EY_X(0, 0, 0) + axh[x] * dLey_dx;
                dLey_dx = dLey_dx / kxh[x] + M_LAMBDA_EY_X(0, 0, 0);
              }
            }

            LAMBDA_HX(0, 0, 0) += dLey_dz - dLez_dy;
            LAMBDA_HY(0, 0, 0) += dLez_dx - dLex_dz;
            LAMBDA_HZ(0, 0, 0) += dLex_dy - dLey_dx;
          }
        }
      }
    }

    /* Background H update transpose for eta. */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            int64_t const idx = IDX(z, y, x);
            int64_t const idx_shot = shot_idx * shot_numel + idx;
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dEhy_dz = DIFFZH1_ADJ(CQ_AT, ETA_HY);
            TIDE_DTYPE dEhz_dy = DIFFYH1_ADJ(CQ_AT, ETA_HZ);
            TIDE_DTYPE dEhz_dx = DIFFXH1_ADJ(CQ_AT, ETA_HZ);
            TIDE_DTYPE dEhx_dz = DIFFZH1_ADJ(CQ_AT, ETA_HX);
            TIDE_DTYPE dEhx_dy = DIFFYH1_ADJ(CQ_AT, ETA_HX);
            TIDE_DTYPE dEhy_dx = DIFFXH1_ADJ(CQ_AT, ETA_HY);

            if (pml_z) {
              M_ETA_HY_Z(0, 0, 0) =
                  bz[z] * M_ETA_HY_Z(0, 0, 0) + az[z] * dEhy_dz;
              dEhy_dz = dEhy_dz / kz[z] + M_ETA_HY_Z(0, 0, 0);
              M_ETA_HX_Z(0, 0, 0) =
                  bz[z] * M_ETA_HX_Z(0, 0, 0) + az[z] * dEhx_dz;
              dEhx_dz = dEhx_dz / kz[z] + M_ETA_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              M_ETA_HZ_Y(0, 0, 0) =
                  by[y] * M_ETA_HZ_Y(0, 0, 0) + ay[y] * dEhz_dy;
              dEhz_dy = dEhz_dy / ky[y] + M_ETA_HZ_Y(0, 0, 0);
              M_ETA_HX_Y(0, 0, 0) =
                  by[y] * M_ETA_HX_Y(0, 0, 0) + ay[y] * dEhx_dy;
              dEhx_dy = dEhx_dy / ky[y] + M_ETA_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              M_ETA_HZ_X(0, 0, 0) =
                  bx[x] * M_ETA_HZ_X(0, 0, 0) + ax[x] * dEhz_dx;
              dEhz_dx = dEhz_dx / kx[x] + M_ETA_HZ_X(0, 0, 0);
              M_ETA_HY_X(0, 0, 0) =
                  bx[x] * M_ETA_HY_X(0, 0, 0) + ax[x] * dEhy_dx;
              dEhy_dx = dEhy_dx / kx[x] + M_ETA_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const curl_eta_x = dEhy_dz - dEhz_dy;
            TIDE_DTYPE const curl_eta_y = dEhz_dx - dEhx_dz;
            TIDE_DTYPE const curl_eta_z = dEhx_dy - dEhy_dx;

            TIDE_DTYPE const eex_curr = ETA_EX(0, 0, 0);
            TIDE_DTYPE const eey_curr = ETA_EY(0, 0, 0);
            TIDE_DTYPE const eez_curr = ETA_EZ(0, 0, 0);

            ETA_EX(0, 0, 0) =
                ca_val * eex_curr + curl_eta_x + eta_source_ex[idx_shot];
            ETA_EY(0, 0, 0) =
                ca_val * eey_curr + curl_eta_y + eta_source_ey[idx_shot];
            ETA_EZ(0, 0, 0) =
                ca_val * eez_curr + curl_eta_z + eta_source_ez[idx_shot];

            if (grad_ca_step) {
              TIDE_DTYPE const acc_ca =
                  eex_curr * ex_store[idx_shot] +
                  eey_curr * ey_store[idx_shot] +
                  eez_curr * ez_store[idx_shot];
              if (ca_batched || reduce_grad_ca) {
                grad_ca_accum[idx_shot] +=
                    acc_ca * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_ca[idx] += acc_ca * (TIDE_DTYPE)step_ratio_eff;
              }
            }
            if (grad_cb_step) {
              TIDE_DTYPE const acc_cb =
                  eex_curr * curl_x_store[idx_shot] +
                  eey_curr * curl_y_store[idx_shot] +
                  eez_curr * curl_z_store[idx_shot];
              if (cb_batched || reduce_grad_cb) {
                grad_cb_accum[idx_shot] +=
                    acc_cb * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_cb[idx] += acc_cb * (TIDE_DTYPE)step_ratio_eff;
              }
            }
          }
        }
      }
    }

    /* Scattered H update transpose for lambda. */
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            int64_t const idx = IDX(z, y, x);
            int64_t const idx_shot = shot_idx * shot_numel + idx;
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dLhy_dz = DIFFZH1_ADJ(CQ_AT, LAMBDA_HY);
            TIDE_DTYPE dLhz_dy = DIFFYH1_ADJ(CQ_AT, LAMBDA_HZ);
            TIDE_DTYPE dLhz_dx = DIFFXH1_ADJ(CQ_AT, LAMBDA_HZ);
            TIDE_DTYPE dLhx_dz = DIFFZH1_ADJ(CQ_AT, LAMBDA_HX);
            TIDE_DTYPE dLhx_dy = DIFFYH1_ADJ(CQ_AT, LAMBDA_HX);
            TIDE_DTYPE dLhy_dx = DIFFXH1_ADJ(CQ_AT, LAMBDA_HY);

            if (pml_z) {
              M_LAMBDA_HY_Z(0, 0, 0) =
                  bz[z] * M_LAMBDA_HY_Z(0, 0, 0) + az[z] * dLhy_dz;
              dLhy_dz = dLhy_dz / kz[z] + M_LAMBDA_HY_Z(0, 0, 0);
              M_LAMBDA_HX_Z(0, 0, 0) =
                  bz[z] * M_LAMBDA_HX_Z(0, 0, 0) + az[z] * dLhx_dz;
              dLhx_dz = dLhx_dz / kz[z] + M_LAMBDA_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              M_LAMBDA_HZ_Y(0, 0, 0) =
                  by[y] * M_LAMBDA_HZ_Y(0, 0, 0) + ay[y] * dLhz_dy;
              dLhz_dy = dLhz_dy / ky[y] + M_LAMBDA_HZ_Y(0, 0, 0);
              M_LAMBDA_HX_Y(0, 0, 0) =
                  by[y] * M_LAMBDA_HX_Y(0, 0, 0) + ay[y] * dLhx_dy;
              dLhx_dy = dLhx_dy / ky[y] + M_LAMBDA_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              M_LAMBDA_HZ_X(0, 0, 0) =
                  bx[x] * M_LAMBDA_HZ_X(0, 0, 0) + ax[x] * dLhz_dx;
              dLhz_dx = dLhz_dx / kx[x] + M_LAMBDA_HZ_X(0, 0, 0);
              M_LAMBDA_HY_X(0, 0, 0) =
                  bx[x] * M_LAMBDA_HY_X(0, 0, 0) + ax[x] * dLhy_dx;
              dLhy_dx = dLhy_dx / kx[x] + M_LAMBDA_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const curl_lambda_x = dLhy_dz - dLhz_dy;
            TIDE_DTYPE const curl_lambda_y = dLhz_dx - dLhx_dz;
            TIDE_DTYPE const curl_lambda_z = dLhx_dy - dLhy_dx;

            TIDE_DTYPE const lex_curr = LAMBDA_EX(0, 0, 0);
            TIDE_DTYPE const ley_curr = LAMBDA_EY(0, 0, 0);
            TIDE_DTYPE const lez_curr = LAMBDA_EZ(0, 0, 0);

            LAMBDA_EX(0, 0, 0) = ca_val * lex_curr + curl_lambda_x;
            LAMBDA_EY(0, 0, 0) = ca_val * ley_curr + curl_lambda_y;
            LAMBDA_EZ(0, 0, 0) = ca_val * lez_curr + curl_lambda_z;

            if (grad_dca_step) {
              TIDE_DTYPE const acc_dca =
                  lex_curr * ex_store[idx_shot] +
                  ley_curr * ey_store[idx_shot] +
                  lez_curr * ez_store[idx_shot];
              if (ca_batched || reduce_grad_dca) {
                grad_dca_accum[idx_shot] +=
                    acc_dca * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_dca[idx] += acc_dca * (TIDE_DTYPE)step_ratio_eff;
              }
            }
            if (grad_dcb_step) {
              TIDE_DTYPE const acc_dcb =
                  lex_curr * curl_x_store[idx_shot] +
                  ley_curr * curl_y_store[idx_shot] +
                  lez_curr * curl_z_store[idx_shot];
              if (cb_batched || reduce_grad_dcb) {
                grad_dcb_accum[idx_shot] +=
                    acc_dcb * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_dcb[idx] += acc_dcb * (TIDE_DTYPE)step_ratio_eff;
              }
            }
          }
        }
      }
    }
  }

  if (reduce_grad_ca) {
    combine_grad_shot_3d(grad_ca, grad_ca_shot, n_shots, shot_numel);
  }
  if (reduce_grad_cb) {
    combine_grad_shot_3d(grad_cb, grad_cb_shot, n_shots, shot_numel);
  }
  if (reduce_grad_dca) {
    combine_grad_shot_3d(grad_dca, grad_dca_shot, n_shots, shot_numel);
  }
  if (reduce_grad_dcb) {
    combine_grad_shot_3d(grad_dcb, grad_dcb_shot, n_shots, shot_numel);
  }

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

TIDE_EXTERN_C TIDE_EXPORT void FUNC(born_backward)(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const grad_r,
    TIDE_DTYPE *__restrict const lambda_ex,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const lambda_ez,
    TIDE_DTYPE *__restrict const lambda_hx,
    TIDE_DTYPE *__restrict const lambda_hy,
    TIDE_DTYPE *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
    TIDE_DTYPE *__restrict const m_lambda_ez_y,
    TIDE_DTYPE *__restrict const m_lambda_ez_x,
    TIDE_DTYPE *__restrict const m_lambda_ex_z,
    TIDE_DTYPE *__restrict const m_lambda_ex_y,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE *__restrict const m_lambda_hz_y,
    TIDE_DTYPE *__restrict const m_lambda_hy_z,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const m_lambda_hy_x,
    TIDE_DTYPE *__restrict const m_lambda_hx_y,
    TIDE_DTYPE *__restrict const store_1,
    TIDE_DTYPE *__restrict const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *__restrict const store_3,
    TIDE_DTYPE *__restrict const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *__restrict const store_5,
    TIDE_DTYPE *__restrict const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *__restrict const store_7,
    TIDE_DTYPE *__restrict const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *__restrict const store_9,
    TIDE_DTYPE *__restrict const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *__restrict const store_11,
    TIDE_DTYPE *__restrict const store_12,
    char **store_filenames_6,
    TIDE_DTYPE *__restrict const grad_f,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    int64_t const storage_mode,
    int64_t const storage_format,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_z0,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_z1,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)device;
  (void)execution_backend;
  (void)compute_stream_handle;
  (void)storage_stream_handle;
  (void)dt;
  (void)store_2;
  (void)store_filenames_1;
  (void)store_4;
  (void)store_filenames_2;
  (void)store_6;
  (void)store_filenames_3;
  (void)store_8;
  (void)store_filenames_4;
  (void)store_10;
  (void)store_filenames_5;
  (void)store_12;
  (void)store_filenames_6;
  (void)az;

#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = nz * ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  int64_t const step_ratio_eff = step_ratio > 0 ? step_ratio : 1;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      (storage_format == STORAGE_FORMAT_FULL) &&
      (shot_bytes_uncomp == (int64_t)(shot_numel * (int64_t)sizeof(TIDE_DTYPE)));
  bool const reduce_grad_ca =
      ca_requires_grad && !ca_batched && grad_ca != NULL && grad_ca_shot != NULL;
  bool const reduce_grad_cb =
      cb_requires_grad && !cb_batched && grad_cb != NULL && grad_cb_shot != NULL;
  TIDE_DTYPE *__restrict grad_ca_accum = reduce_grad_ca ? grad_ca_shot : grad_ca;
  TIDE_DTYPE *__restrict grad_cb_accum = reduce_grad_cb ? grad_cb_shot : grad_cb;

  if (grad_f != NULL && nt > 0 && n_shots > 0 && n_sources_per_shot > 0) {
    size_t const numel =
        (size_t)nt * (size_t)n_shots * (size_t)n_sources_per_shot;
    tide_zero_if_not_null(grad_f, numel * sizeof(TIDE_DTYPE));
  }
  if (ca_requires_grad && grad_ca != NULL) {
    size_t const n = (size_t)(ca_batched ? n_shots : 1) * (size_t)shot_numel;
    tide_zero_if_not_null(grad_ca, n * sizeof(TIDE_DTYPE));
  }
  if (cb_requires_grad && grad_cb != NULL) {
    size_t const n = (size_t)(cb_batched ? n_shots : 1) * (size_t)shot_numel;
    tide_zero_if_not_null(grad_cb, n * sizeof(TIDE_DTYPE));
  }
  if (reduce_grad_ca) {
    tide_zero_if_not_null(grad_ca_shot, (size_t)store_size * sizeof(TIDE_DTYPE));
  }
  if (reduce_grad_cb) {
    tide_zero_if_not_null(grad_cb_shot, (size_t)store_size * sizeof(TIDE_DTYPE));
  }

  TIDE_DTYPE *__restrict lambda_src_field = lambda_ey;
  if (source_component == 0) {
    lambda_src_field = lambda_ex;
  } else if (source_component == 2) {
    lambda_src_field = lambda_ez;
  }
  TIDE_DTYPE *__restrict lambda_recv_field = lambda_ey;
  if (receiver_component == 0) {
    lambda_recv_field = lambda_ex;
  } else if (receiver_component == 2) {
    lambda_recv_field = lambda_ez;
  }

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    bool const do_grad = (t % step_ratio_eff) == 0;
    bool const grad_ca_step =
        do_grad && ca_requires_grad && storage_direct && store_1 != NULL &&
        store_3 != NULL && store_5 != NULL;
    bool const grad_cb_step =
        do_grad && cb_requires_grad && storage_direct && store_7 != NULL &&
        store_9 != NULL && store_11 != NULL;

    int64_t const store_idx = t / step_ratio_eff;
    int64_t const store_offset = store_idx * store_size;

    TIDE_DTYPE const *__restrict const ex_store =
        grad_ca_step ? (store_1 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const ey_store =
        grad_ca_step ? (store_3 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const ez_store =
        grad_ca_step ? (store_5 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_x_store =
        grad_cb_step ? (store_7 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_y_store =
        grad_cb_step ? (store_9 + store_offset) : NULL;
    TIDE_DTYPE const *__restrict const curl_z_store =
        grad_cb_step ? (store_11 + store_offset) : NULL;

    if (n_receivers_per_shot > 0 && grad_r != NULL && receivers_i != NULL) {
      add_sources_component(
          lambda_recv_field, grad_r, receivers_i,
          t * n_shots * n_receivers_per_shot, n_shots, shot_numel,
          n_receivers_per_shot);
    }

    if (n_sources_per_shot > 0 && grad_f != NULL && sources_i != NULL) {
      record_receivers_component(
          grad_f, lambda_src_field, sources_i,
          t * n_shots * n_sources_per_shot, n_shots, shot_numel,
          n_sources_per_shot);
    }

    TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
            bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
            bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

            TIDE_DTYPE dLey_dz_pml = 0;
            TIDE_DTYPE dLez_dy_pml = 0;
            TIDE_DTYPE dLez_dx_pml = 0;
            TIDE_DTYPE dLex_dz_pml = 0;
            TIDE_DTYPE dLex_dy_pml = 0;
            TIDE_DTYPE dLey_dx_pml = 0;

            if (z < nz - FD_PAD) {
              TIDE_DTYPE dLey_dz = -DIFFZ1_ADJ(CB_AT, LAMBDA_EY);
              if (pml_z_h) {
                M_LAMBDA_EY_Z(0, 0, 0) =
                    bzh[z] * M_LAMBDA_EY_Z(0, 0, 0) + azh[z] * dLey_dz;
                dLey_dz = dLey_dz / kzh[z] + M_LAMBDA_EY_Z(0, 0, 0);
              }
              dLey_dz_pml = dLey_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dLez_dy = -DIFFY1_ADJ(CB_AT, LAMBDA_EZ);
              if (pml_y_h) {
                M_LAMBDA_EZ_Y(0, 0, 0) =
                    byh[y] * M_LAMBDA_EZ_Y(0, 0, 0) + ayh[y] * dLez_dy;
                dLez_dy = dLez_dy / kyh[y] + M_LAMBDA_EZ_Y(0, 0, 0);
              }
              dLez_dy_pml = dLez_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dLez_dx = -DIFFX1_ADJ(CB_AT, LAMBDA_EZ);
              if (pml_x_h) {
                M_LAMBDA_EZ_X(0, 0, 0) =
                    bxh[x] * M_LAMBDA_EZ_X(0, 0, 0) + axh[x] * dLez_dx;
                dLez_dx = dLez_dx / kxh[x] + M_LAMBDA_EZ_X(0, 0, 0);
              }
              dLez_dx_pml = dLez_dx;
            }
            if (z < nz - FD_PAD) {
              TIDE_DTYPE dLex_dz = -DIFFZ1_ADJ(CB_AT, LAMBDA_EX);
              if (pml_z_h) {
                M_LAMBDA_EX_Z(0, 0, 0) =
                    bzh[z] * M_LAMBDA_EX_Z(0, 0, 0) + azh[z] * dLex_dz;
                dLex_dz = dLex_dz / kzh[z] + M_LAMBDA_EX_Z(0, 0, 0);
              }
              dLex_dz_pml = dLex_dz;
            }
            if (y < ny - FD_PAD) {
              TIDE_DTYPE dLex_dy = -DIFFY1_ADJ(CB_AT, LAMBDA_EX);
              if (pml_y_h) {
                M_LAMBDA_EX_Y(0, 0, 0) =
                    byh[y] * M_LAMBDA_EX_Y(0, 0, 0) + ayh[y] * dLex_dy;
                dLex_dy = dLex_dy / kyh[y] + M_LAMBDA_EX_Y(0, 0, 0);
              }
              dLex_dy_pml = dLex_dy;
            }
            if (x < nx - FD_PAD) {
              TIDE_DTYPE dLey_dx = -DIFFX1_ADJ(CB_AT, LAMBDA_EY);
              if (pml_x_h) {
                M_LAMBDA_EY_X(0, 0, 0) =
                    bxh[x] * M_LAMBDA_EY_X(0, 0, 0) + axh[x] * dLey_dx;
                dLey_dx = dLey_dx / kxh[x] + M_LAMBDA_EY_X(0, 0, 0);
              }
              dLey_dx_pml = dLey_dx;
            }

            LAMBDA_HX(0, 0, 0) += dLey_dz_pml - dLez_dy_pml;
            LAMBDA_HY(0, 0, 0) += dLez_dx_pml - dLex_dz_pml;
            LAMBDA_HZ(0, 0, 0) += dLex_dy_pml - dLey_dx_pml;
          }
        }
      }
    }

TIDE_OMP_PARALLEL_FOR
    for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
        for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
          for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
            int64_t const idx = IDX(z, y, x);
            int64_t const idx_shot = shot_idx * shot_numel + idx;
            TIDE_DTYPE const ca_val = CA_AT(0, 0, 0);
            bool const pml_z = (z < pml_z0) || (z >= pml_z1);
            bool const pml_y = (y < pml_y0) || (y >= pml_y1);
            bool const pml_x = (x < pml_x0) || (x >= pml_x1);

            TIDE_DTYPE dLhy_dz = DIFFZH1_ADJ(CQ_AT, LAMBDA_HY);
            TIDE_DTYPE dLhz_dy = DIFFYH1_ADJ(CQ_AT, LAMBDA_HZ);
            TIDE_DTYPE dLhz_dx = DIFFXH1_ADJ(CQ_AT, LAMBDA_HZ);
            TIDE_DTYPE dLhx_dz = DIFFZH1_ADJ(CQ_AT, LAMBDA_HX);
            TIDE_DTYPE dLhx_dy = DIFFYH1_ADJ(CQ_AT, LAMBDA_HX);
            TIDE_DTYPE dLhy_dx = DIFFXH1_ADJ(CQ_AT, LAMBDA_HY);

            if (pml_z) {
              M_LAMBDA_HY_Z(0, 0, 0) =
                  bz[z] * M_LAMBDA_HY_Z(0, 0, 0) + az[z] * dLhy_dz;
              dLhy_dz = dLhy_dz / kz[z] + M_LAMBDA_HY_Z(0, 0, 0);
              M_LAMBDA_HX_Z(0, 0, 0) =
                  bz[z] * M_LAMBDA_HX_Z(0, 0, 0) + az[z] * dLhx_dz;
              dLhx_dz = dLhx_dz / kz[z] + M_LAMBDA_HX_Z(0, 0, 0);
            }
            if (pml_y) {
              M_LAMBDA_HZ_Y(0, 0, 0) =
                  by[y] * M_LAMBDA_HZ_Y(0, 0, 0) + ay[y] * dLhz_dy;
              dLhz_dy = dLhz_dy / ky[y] + M_LAMBDA_HZ_Y(0, 0, 0);
              M_LAMBDA_HX_Y(0, 0, 0) =
                  by[y] * M_LAMBDA_HX_Y(0, 0, 0) + ay[y] * dLhx_dy;
              dLhx_dy = dLhx_dy / ky[y] + M_LAMBDA_HX_Y(0, 0, 0);
            }
            if (pml_x) {
              M_LAMBDA_HZ_X(0, 0, 0) =
                  bx[x] * M_LAMBDA_HZ_X(0, 0, 0) + ax[x] * dLhz_dx;
              dLhz_dx = dLhz_dx / kx[x] + M_LAMBDA_HZ_X(0, 0, 0);
              M_LAMBDA_HY_X(0, 0, 0) =
                  bx[x] * M_LAMBDA_HY_X(0, 0, 0) + ax[x] * dLhy_dx;
              dLhy_dx = dLhy_dx / kx[x] + M_LAMBDA_HY_X(0, 0, 0);
            }

            TIDE_DTYPE const curl_lambda_x = dLhy_dz - dLhz_dy;
            TIDE_DTYPE const curl_lambda_y = dLhz_dx - dLhx_dz;
            TIDE_DTYPE const curl_lambda_z = dLhx_dy - dLhy_dx;

            TIDE_DTYPE const lex_curr = LAMBDA_EX(0, 0, 0);
            TIDE_DTYPE const ley_curr = LAMBDA_EY(0, 0, 0);
            TIDE_DTYPE const lez_curr = LAMBDA_EZ(0, 0, 0);

            LAMBDA_EX(0, 0, 0) = ca_val * lex_curr + curl_lambda_x;
            LAMBDA_EY(0, 0, 0) = ca_val * ley_curr + curl_lambda_y;
            LAMBDA_EZ(0, 0, 0) = ca_val * lez_curr + curl_lambda_z;

            if (grad_ca_step) {
              TIDE_DTYPE const acc_ca =
                  lex_curr * ex_store[idx_shot] +
                  ley_curr * ey_store[idx_shot] +
                  lez_curr * ez_store[idx_shot];
              if (ca_batched || reduce_grad_ca) {
                grad_ca_accum[idx_shot] += acc_ca * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_ca[idx] += acc_ca * (TIDE_DTYPE)step_ratio_eff;
              }
            }
            if (grad_cb_step) {
              TIDE_DTYPE const acc_cb =
                  lex_curr * curl_x_store[idx_shot] +
                  ley_curr * curl_y_store[idx_shot] +
                  lez_curr * curl_z_store[idx_shot];
              if (cb_batched || reduce_grad_cb) {
                grad_cb_accum[idx_shot] += acc_cb * (TIDE_DTYPE)step_ratio_eff;
              } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                grad_cb[idx] += acc_cb * (TIDE_DTYPE)step_ratio_eff;
              }
            }
          }
        }
      }
    }
  }

  if (reduce_grad_ca) {
    combine_grad_shot_3d(grad_ca, grad_ca_shot, n_shots, shot_numel);
  }
  if (reduce_grad_cb) {
    combine_grad_shot_3d(grad_cb, grad_cb_shot, n_shots, shot_numel);
  }

#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

#undef IDX
#undef IDX_SHOT
#undef EX
#undef EY
#undef EZ
#undef HX
#undef HY
#undef HZ
#undef CA_AT
#undef CB_AT
#undef CQ_AT
#undef M_HZ_Y
#undef M_HY_Z
#undef M_HX_Z
#undef M_HZ_X
#undef M_HY_X
#undef M_HX_Y
#undef M_EY_Z
#undef M_EZ_Y
#undef M_EZ_X
#undef M_EX_Z
#undef M_EX_Y
#undef M_EY_X
#undef LAMBDA_EX
#undef LAMBDA_EY
#undef LAMBDA_EZ
#undef LAMBDA_HX
#undef LAMBDA_HY
#undef LAMBDA_HZ
#undef M_LAMBDA_EY_Z
#undef M_LAMBDA_EZ_Y
#undef M_LAMBDA_EZ_X
#undef M_LAMBDA_EX_Z
#undef M_LAMBDA_EX_Y
#undef M_LAMBDA_EY_X
#undef M_LAMBDA_HZ_Y
#undef M_LAMBDA_HY_Z
#undef M_LAMBDA_HX_Z
#undef M_LAMBDA_HZ_X
#undef M_LAMBDA_HY_X
#undef M_LAMBDA_HX_Y
#undef ETA_EX
#undef ETA_EY
#undef ETA_EZ
#undef ETA_HX
#undef ETA_HY
#undef ETA_HZ
#undef M_ETA_EY_Z
#undef M_ETA_EZ_Y
#undef M_ETA_EZ_X
#undef M_ETA_EX_Z
#undef M_ETA_EX_Y
#undef M_ETA_EY_X
#undef M_ETA_HZ_Y
#undef M_ETA_HY_Z
#undef M_ETA_HX_Z
#undef M_ETA_HZ_X
#undef M_ETA_HY_X
#undef M_ETA_HX_Y

} // namespace FUNC(Inst)
