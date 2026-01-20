/*
 * 3D Maxwell FDTD propagator (CPU implementation)
 *
 * Forward-only implementation for 3D full Maxwell with CPML.
 *
 * Field components: Ex, Ey, Ez, Hx, Hy, Hz
 * Grid ordering: [nz, ny, nx] with z as slowest dimension.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common_cpu.h"
#include "staggered_grid_3d.h"

#define CAT_I(name, accuracy, dtype, device) \
  maxwell_3d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) \
  CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, cpu)

// 3D indexing macros
#define IDX(z, y, x) ((z) * ny * nx + (y) * nx + (x))
#define IDX_SHOT(shot, z, y, x) ((shot) * shot_numel + (z) * ny * nx + (y) * nx + (x))

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Field access macros
#define EX(dz, dy, dx) ex[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define EY(dz, dy, dx) ey[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define EZ(dz, dy, dx) ez[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define HX(dz, dy, dx) hx[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define HY(dz, dy, dx) hy[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define HZ(dz, dy, dx) hz[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

// Material parameter access macros
#define CA(dz, dy, dx) (ca_batched ? ca[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))] : ca[IDX(z + (dz), y + (dy), x + (dx))])
#define CB(dz, dy, dx) (cb_batched ? cb[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))] : cb[IDX(z + (dz), y + (dy), x + (dx))])
#define CQ(dz, dy, dx) (cq_batched ? cq[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))] : cq[IDX(z + (dz), y + (dy), x + (dx))])

// PML memory variable macros (H update uses E-derived memories)
#define M_EY_Z(dz, dy, dx) m_ey_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EZ_Y(dz, dy, dx) m_ez_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EZ_X(dz, dy, dx) m_ez_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EX_Z(dz, dy, dx) m_ex_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EX_Y(dz, dy, dx) m_ex_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_EY_X(dz, dy, dx) m_ey_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

// PML memory variable macros (E update uses H-derived memories)
#define M_HZ_Y(dz, dy, dx) m_hz_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HY_Z(dz, dy, dx) m_hy_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HX_Z(dz, dy, dx) m_hx_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HZ_X(dz, dy, dx) m_hz_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HY_X(dz, dy, dx) m_hy_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_HX_Y(dz, dy, dx) m_hx_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

static void add_sources_field(
    TIDE_DTYPE *const field,
    TIDE_DTYPE const *const f,
    int64_t const *const sources_i,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_sources_per_shot) {
  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR_COLLAPSE2
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (int64_t source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
      int64_t k = shot_idx * n_sources_per_shot + source_idx;
      if (sources_i[k] >= 0) {
        field[shot_idx * shot_numel + sources_i[k]] += f[k];
      }
    }
  }
}


static void record_receivers_field(
    TIDE_DTYPE *const r,
    TIDE_DTYPE const *const field,
    int64_t const *const receivers_i,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_receivers_per_shot) {
  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR_COLLAPSE2
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (int64_t receiver_idx = 0; receiver_idx < n_receivers_per_shot; ++receiver_idx) {
      int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
      if (receivers_i[k] >= 0) {
        r[k] = field[shot_idx * shot_numel + receivers_i[k]];
      }
    }
  }
}


static void forward_kernel_h_3d(
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const ex,
    TIDE_DTYPE const *const ey,
    TIDE_DTYPE const *const ez,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hy,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_ez_y,
    TIDE_DTYPE *const m_ez_x,
    TIDE_DTYPE *const m_ex_z,
    TIDE_DTYPE *const m_ex_y,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE const *const az,
    TIDE_DTYPE const *const bz,
    TIDE_DTYPE const *const azh,
    TIDE_DTYPE const *const bzh,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const kz,
    TIDE_DTYPE const *const kzh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_z0,
    int64_t const pml_z1,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const cq_batched) {

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR_COLLAPSE4
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
      for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
        for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
          bool pml_z = z < pml_z0h || z >= pml_z1h;
          bool pml_y = y < pml_y0h || y >= pml_y1h;
          bool pml_x = x < pml_x0h || x >= pml_x1h;
          TIDE_DTYPE const cq_val = CQ(0, 0, 0);

          if (z < nz - FD_PAD && y < ny - FD_PAD) {
            TIDE_DTYPE dEy_dz = DIFFZH1(EY);
            if (pml_z) {
              M_EY_Z(0, 0, 0) = bzh[z] * M_EY_Z(0, 0, 0) + azh[z] * dEy_dz;
              dEy_dz = dEy_dz / kzh[z] + M_EY_Z(0, 0, 0);
            }
            TIDE_DTYPE dEz_dy = DIFFYH1(EZ);
            if (pml_y) {
              M_EZ_Y(0, 0, 0) = byh[y] * M_EZ_Y(0, 0, 0) + ayh[y] * dEz_dy;
              dEz_dy = dEz_dy / kyh[y] + M_EZ_Y(0, 0, 0);
            }
            HX(0, 0, 0) -= cq_val * (dEy_dz - dEz_dy);
          }

          if (z < nz - FD_PAD && x < nx - FD_PAD) {
            TIDE_DTYPE dEz_dx = DIFFXH1(EZ);
            if (pml_x) {
              M_EZ_X(0, 0, 0) = bxh[x] * M_EZ_X(0, 0, 0) + axh[x] * dEz_dx;
              dEz_dx = dEz_dx / kxh[x] + M_EZ_X(0, 0, 0);
            }
            TIDE_DTYPE dEx_dz = DIFFZH1(EX);
            if (pml_z) {
              M_EX_Z(0, 0, 0) = bzh[z] * M_EX_Z(0, 0, 0) + azh[z] * dEx_dz;
              dEx_dz = dEx_dz / kzh[z] + M_EX_Z(0, 0, 0);
            }
            HY(0, 0, 0) -= cq_val * (dEz_dx - dEx_dz);
          }

          if (y < ny - FD_PAD && x < nx - FD_PAD) {
            TIDE_DTYPE dEx_dy = DIFFYH1(EX);
            if (pml_y) {
              M_EX_Y(0, 0, 0) = byh[y] * M_EX_Y(0, 0, 0) + ayh[y] * dEx_dy;
              dEx_dy = dEx_dy / kyh[y] + M_EX_Y(0, 0, 0);
            }
            TIDE_DTYPE dEy_dx = DIFFXH1(EY);
            if (pml_x) {
              M_EY_X(0, 0, 0) = bxh[x] * M_EY_X(0, 0, 0) + axh[x] * dEy_dx;
              dEy_dx = dEy_dx / kxh[x] + M_EY_X(0, 0, 0);
            }
            HZ(0, 0, 0) -= cq_val * (dEx_dy - dEy_dx);
          }
        }
      }
    }
  }
}


static void forward_kernel_e_3d(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const hx,
    TIDE_DTYPE const *const hy,
    TIDE_DTYPE const *const hz,
    TIDE_DTYPE *const ex,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const ez,
    TIDE_DTYPE *const m_hz_y,
    TIDE_DTYPE *const m_hy_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const m_hy_x,
    TIDE_DTYPE *const m_hx_y,
    TIDE_DTYPE const *const az,
    TIDE_DTYPE const *const bz,
    TIDE_DTYPE const *const azh,
    TIDE_DTYPE const *const bzh,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const kz,
    TIDE_DTYPE const *const kzh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_z0,
    int64_t const pml_z1,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cb_batched) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR_COLLAPSE4
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
      for (int64_t y = FD_PAD; y < ny - FD_PAD + 1; ++y) {
        for (int64_t x = FD_PAD; x < nx - FD_PAD + 1; ++x) {
          bool pml_z = z < pml_z0 || z >= pml_z1;
          bool pml_y = y < pml_y0 || y >= pml_y1;
          bool pml_x = x < pml_x0 || x >= pml_x1;
          TIDE_DTYPE const ca_val = CA(0, 0, 0);
          TIDE_DTYPE const cb_val = CB(0, 0, 0);

          TIDE_DTYPE dHz_dy = DIFFY1(HZ);
          if (pml_y) {
            M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
            dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
          }
          TIDE_DTYPE dHy_dz = DIFFZ1(HY);
          if (pml_z) {
            M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
            dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
          }
          EX(0, 0, 0) = ca_val * EX(0, 0, 0) + cb_val * (dHz_dy - dHy_dz);

          TIDE_DTYPE dHx_dz = DIFFZ1(HX);
          if (pml_z) {
            M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
            dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
          }
          TIDE_DTYPE dHz_dx = DIFFX1(HZ);
          if (pml_x) {
            M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
            dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
          }
          EY(0, 0, 0) = ca_val * EY(0, 0, 0) + cb_val * (dHx_dz - dHz_dx);

          TIDE_DTYPE dHy_dx = DIFFX1(HY);
          if (pml_x) {
            M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
            dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
          }
          TIDE_DTYPE dHx_dy = DIFFY1(HX);
          if (pml_y) {
            M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
            dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
          }
          EZ(0, 0, 0) = ca_val * EZ(0, 0, 0) + cb_val * (dHy_dx - dHx_dy);
        }
      }
    }
  }
}


/*
 * Forward propagation entry point (CPU)
 */
#ifdef __cplusplus
extern "C"
#endif
#ifdef _WIN32
__declspec(dllexport)
#endif
void FUNC(forward)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE *const ex,
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const ez,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hy,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_hz_y,
    TIDE_DTYPE *const m_hy_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const m_hy_x,
    TIDE_DTYPE *const m_hx_y,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_ez_y,
    TIDE_DTYPE *const m_ez_x,
    TIDE_DTYPE *const m_ex_z,
    TIDE_DTYPE *const m_ex_y,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const r,
    TIDE_DTYPE const *const az,
    TIDE_DTYPE const *const bz,
    TIDE_DTYPE const *const azh,
    TIDE_DTYPE const *const bzh,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const kz,
    TIDE_DTYPE const *const kzh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
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

  TIDE_DTYPE *source_field = NULL;
  TIDE_DTYPE *receiver_field = NULL;
  switch (source_component) {
    case 0: source_field = ex; break;
    case 1: source_field = ey; break;
    case 2: source_field = ez; break;
    default: source_field = ey; break;
  }
  switch (receiver_component) {
    case 0: receiver_field = ex; break;
    case 1: receiver_field = ey; break;
    case 2: receiver_field = ez; break;
    case 3: receiver_field = hx; break;
    case 4: receiver_field = hy; break;
    case 5: receiver_field = hz; break;
    default: receiver_field = ey; break;
  }

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h_3d(
        cq, ex, ey, ez, hx, hy, hz,
        m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
        az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
        kz, kzh, ky, kyh, kx, kxh,
        rdz, rdy, rdx,
        n_shots, nz, ny, nx, shot_numel,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        cq_batched);

    forward_kernel_e_3d(
        ca, cb, hx, hy, hz, ex, ey, ez,
        m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
        az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
        kz, kzh, ky, kyh, kx, kxh,
        rdz, rdy, rdx,
        n_shots, nz, ny, nx, shot_numel,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched);

    if (n_sources_per_shot > 0) {
      add_sources_field(
          source_field,
          f + t * n_shots * n_sources_per_shot,
          sources_i,
          n_shots,
          shot_numel,
          n_sources_per_shot);
    }

    if (n_receivers_per_shot > 0) {
      record_receivers_field(
          r + t * n_shots * n_receivers_per_shot,
          receiver_field,
          receivers_i,
          n_shots,
          shot_numel,
          n_receivers_per_shot);
    }
  }
#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}
