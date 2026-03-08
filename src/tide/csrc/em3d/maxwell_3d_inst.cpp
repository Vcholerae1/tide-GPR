/*
 * Maxwell 3D CPU backend implementation (forward path).
 *
 * This implementation mirrors the 3D Python backend update equations
 * (staggered-grid FDTD + CPML) for inference use.
 *
 * Notes:
 * - forward_with_storage currently reuses forward behavior.
 * - backward remains a zero-gradient scaffold and is not used when
 *   Python dispatch selects gradient-capable fallback.
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

static inline void tide_zero_if_not_null(void *ptr, size_t bytes) {
  if (ptr != NULL && bytes > 0) {
    memset(ptr, 0, bytes);
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
    int64_t const device) {
  (void)dt;
  (void)step_ratio;
  (void)device;

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
    int64_t const device) {
  (void)dt;
  (void)device;
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
    int64_t const device) {
  (void)device;
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
      (shot_bytes_uncomp == (int64_t)(shot_numel * (int64_t)sizeof(TIDE_DTYPE)));

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
  if (grad_ca_shot != NULL) {
    size_t n = (size_t)n_shots * (size_t)shot_numel;
    tide_zero_if_not_null(grad_ca_shot, n * sizeof(TIDE_DTYPE));
  }
  if (grad_cb_shot != NULL) {
    size_t n = (size_t)n_shots * (size_t)shot_numel;
    tide_zero_if_not_null(grad_cb_shot, n * sizeof(TIDE_DTYPE));
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
              if (ca_batched) {
                grad_ca[idx_shot] += acc_ca * (TIDE_DTYPE)step_ratio_eff;
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
              if (cb_batched) {
                grad_cb[idx_shot] += acc_cb * (TIDE_DTYPE)step_ratio_eff;
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

  convert_grad_ca_cb_to_eps_sigma_3d(
      ca, cb, grad_ca, grad_cb, grad_eps, grad_sigma, dt, n_shots, nz, ny, nx,
      ca_batched, cb_batched, ca_requires_grad, cb_requires_grad);

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

} // namespace FUNC(Inst)
