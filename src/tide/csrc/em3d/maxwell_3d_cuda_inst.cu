/*
 * Maxwell 3D CUDA backend implementation.
 *
 * This file provides CUDA forward and backward propagation kernels for the
 * 3D Maxwell staggered-grid + CPML scheme.
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

#define ND_INDEX(i, dz, dy, dx) ((i) + (dz) * ny * nx + (dy) * nx + (dx))
#define ND_INDEX_J(j, dz, dy, dx) ((j) + (dz) * ny * nx + (dy) * nx + (dx))

#define EX(dz, dy, dx) ex[ND_INDEX(i, dz, dy, dx)]
#define EY(dz, dy, dx) ey[ND_INDEX(i, dz, dy, dx)]
#define EZ(dz, dy, dx) ez[ND_INDEX(i, dz, dy, dx)]
#define HX(dz, dy, dx) hx[ND_INDEX(i, dz, dy, dx)]
#define HY(dz, dy, dx) hy[ND_INDEX(i, dz, dy, dx)]
#define HZ(dz, dy, dx) hz[ND_INDEX(i, dz, dy, dx)]
#define CB_I(dz, dy, dx) \
  (cb_batched ? cb[ND_INDEX(i, dz, dy, dx)] : cb[ND_INDEX_J(j, dz, dy, dx)])
#define CQ_I(dz, dy, dx) \
  (cq_batched ? cq[ND_INDEX(i, dz, dy, dx)] : cq[ND_INDEX_J(j, dz, dy, dx)])

namespace FUNC(Inst) {

namespace {

template <typename T>
__host__ __device__ __forceinline__ T tide_max(T a, T b) {
  return a > b ? a : b;
}

constexpr TIDE_DTYPE kEp0 = (TIDE_DTYPE)8.8541878128e-12;

__constant__ TIDE_DTYPE rdz;
__constant__ TIDE_DTYPE rdy;
__constant__ TIDE_DTYPE rdx;

__constant__ int64_t n_shots;
__constant__ int64_t nz;
__constant__ int64_t ny;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_sources_per_shot;
__constant__ int64_t n_receivers_per_shot;

__constant__ int64_t pml_z0;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_z1;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x1;

__constant__ bool ca_batched;
__constant__ bool cb_batched;
__constant__ bool cq_batched;

template <typename T>
__device__ __forceinline__ void atomic_add_tide(T *addr, T val) {
  if constexpr (std::is_same<T, double>::value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    atomicAdd(addr, val);
#else
    unsigned long long int *address_as_ull = (unsigned long long int *)addr;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
      assumed = old;
      old = atomicCAS(
          address_as_ull,
          assumed,
          __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
  } else {
    static_assert(std::is_same<T, float>::value,
                  "atomic_add_tide only supports float/double");
    atomicAdd(addr, val);
  }
}

__global__ void add_sources_component(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_sources_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const src = sources_i[idx];
  if (src >= 0) {
    int64_t const shot_idx = idx / n_sources_per_shot;
    field[shot_idx * shot_numel + src] += f[idx];
  }
}

__global__ void record_receivers_component(
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE const *__restrict const field,
    int64_t const *__restrict const receivers_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_receivers_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const rec = receivers_i[idx];
  if (rec >= 0) {
    int64_t const shot_idx = idx / n_receivers_per_shot;
    r[idx] = field[shot_idx * shot_numel + rec];
  } else {
    r[idx] = (TIDE_DTYPE)0;
  }
}

__global__ void forward_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ex,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const ez,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hy,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE *__restrict const m_ez_y,
    TIDE_DTYPE *__restrict const m_ez_x,
    TIDE_DTYPE *__restrict const m_ex_z,
    TIDE_DTYPE *__restrict const m_ex_y,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  int64_t const shot_idx = i / shot_numel;
  int64_t const j = i - shot_idx * shot_numel;

  int64_t const yz_stride = ny * nx;
  int64_t const z = j / yz_stride;
  int64_t const rem = j - z * yz_stride;
  int64_t const y = rem / nx;
  int64_t const x = rem - y * nx;

  if (z < FD_PAD || z >= nz - FD_PAD + 1 || y < FD_PAD || y >= ny - FD_PAD + 1 ||
      x < FD_PAD || x >= nx - FD_PAD + 1) {
    return;
  }

#define EX_L(dz, dy, dx) EX(dz, dy, dx)
#define EY_L(dz, dy, dx) EY(dz, dy, dx)
#define EZ_L(dz, dy, dx) EZ(dz, dy, dx)

  TIDE_DTYPE const cq_val = cq_batched ? cq[i] : cq[j];

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

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
    TIDE_DTYPE dEy_dz = DIFFZH1(EY_L);
    if (pml_z_h) {
      m_ey_z[i] = bzh[z] * m_ey_z[i] + azh[z] * dEy_dz;
      dEy_dz = dEy_dz / kzh[z] + m_ey_z[i];
    }
    dEy_dz_pml = dEy_dz;
  }

  if (y < ny - FD_PAD) {
    TIDE_DTYPE dEz_dy = DIFFYH1(EZ_L);
    if (pml_y_h) {
      m_ez_y[i] = byh[y] * m_ez_y[i] + ayh[y] * dEz_dy;
      dEz_dy = dEz_dy / kyh[y] + m_ez_y[i];
    }
    dEz_dy_pml = dEz_dy;
  }

  if (x < nx - FD_PAD) {
    TIDE_DTYPE dEz_dx = DIFFXH1(EZ_L);
    if (pml_x_h) {
      m_ez_x[i] = bxh[x] * m_ez_x[i] + axh[x] * dEz_dx;
      dEz_dx = dEz_dx / kxh[x] + m_ez_x[i];
    }
    dEz_dx_pml = dEz_dx;
  }

  if (z < nz - FD_PAD) {
    TIDE_DTYPE dEx_dz = DIFFZH1(EX_L);
    if (pml_z_h) {
      m_ex_z[i] = bzh[z] * m_ex_z[i] + azh[z] * dEx_dz;
      dEx_dz = dEx_dz / kzh[z] + m_ex_z[i];
    }
    dEx_dz_pml = dEx_dz;
  }

  if (y < ny - FD_PAD) {
    TIDE_DTYPE dEx_dy = DIFFYH1(EX_L);
    if (pml_y_h) {
      m_ex_y[i] = byh[y] * m_ex_y[i] + ayh[y] * dEx_dy;
      dEx_dy = dEx_dy / kyh[y] + m_ex_y[i];
    }
    dEx_dy_pml = dEx_dy;
  }

  if (x < nx - FD_PAD) {
    TIDE_DTYPE dEy_dx = DIFFXH1(EY_L);
    if (pml_x_h) {
      m_ey_x[i] = bxh[x] * m_ey_x[i] + axh[x] * dEy_dx;
      dEy_dx = dEy_dx / kxh[x] + m_ey_x[i];
    }
    dEy_dx_pml = dEy_dx;
  }

  hx[i] -= cq_val * (dEy_dz_pml - dEz_dy_pml);
  hy[i] -= cq_val * (dEz_dx_pml - dEx_dz_pml);
  hz[i] -= cq_val * (dEx_dy_pml - dEy_dx_pml);

#undef EX_L
#undef EY_L
#undef EZ_L
}

__global__ void forward_kernel_e(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  int64_t const shot_idx = i / shot_numel;
  int64_t const j = i - shot_idx * shot_numel;

  int64_t const yz_stride = ny * nx;
  int64_t const z = j / yz_stride;
  int64_t const rem = j - z * yz_stride;
  int64_t const y = rem / nx;
  int64_t const x = rem - y * nx;

  if (z < FD_PAD || z >= nz - FD_PAD + 1 || y < FD_PAD || y >= ny - FD_PAD + 1 ||
      x < FD_PAD || x >= nx - FD_PAD + 1) {
    return;
  }

#define HX_L(dz, dy, dx) HX(dz, dy, dx)
#define HY_L(dz, dy, dx) HY(dz, dy, dx)
#define HZ_L(dz, dy, dx) HZ(dz, dy, dx)

  TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
  TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];

  bool const pml_z_v = (z < pml_z0) || (z >= pml_z1);
  bool const pml_y_v = (y < pml_y0) || (y >= pml_y1);
  bool const pml_x_v = (x < pml_x0) || (x >= pml_x1);

  TIDE_DTYPE dHy_dz = DIFFZ1(HY_L);
  TIDE_DTYPE dHz_dy = DIFFY1(HZ_L);
  TIDE_DTYPE dHz_dx = DIFFX1(HZ_L);
  TIDE_DTYPE dHx_dz = DIFFZ1(HX_L);
  TIDE_DTYPE dHx_dy = DIFFY1(HX_L);
  TIDE_DTYPE dHy_dx = DIFFX1(HY_L);

  if (pml_z_v) {
    m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
    dHy_dz = dHy_dz / kz[z] + m_hy_z[i];

    m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
    dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
  }

  if (pml_y_v) {
    m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
    dHz_dy = dHz_dy / ky[y] + m_hz_y[i];

    m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
    dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
  }

  if (pml_x_v) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
    dHz_dx = dHz_dx / kx[x] + m_hz_x[i];

    m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
    dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
  }

  ex[i] = ca_val * ex[i] + cb_val * (dHy_dz - dHz_dy);
  ey[i] = ca_val * ey[i] + cb_val * (dHz_dx - dHx_dz);
  ez[i] = ca_val * ez[i] + cb_val * (dHx_dy - dHy_dx);

#undef HX_L
#undef HY_L
#undef HZ_L
}

__global__ void forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE *__restrict const ex_store,
    TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const ez_store,
    TIDE_DTYPE *__restrict const curl_x_store,
    TIDE_DTYPE *__restrict const curl_y_store,
    TIDE_DTYPE *__restrict const curl_z_store,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  int64_t const shot_idx = i / shot_numel;
  int64_t const j = i - shot_idx * shot_numel;

  int64_t const yz_stride = ny * nx;
  int64_t const z = j / yz_stride;
  int64_t const rem = j - z * yz_stride;
  int64_t const y = rem / nx;
  int64_t const x = rem - y * nx;

  if (z < FD_PAD || z >= nz - FD_PAD + 1 || y < FD_PAD || y >= ny - FD_PAD + 1 ||
      x < FD_PAD || x >= nx - FD_PAD + 1) {
    return;
  }

#define HX_S(dz, dy, dx) HX(dz, dy, dx)
#define HY_S(dz, dy, dx) HY(dz, dy, dx)
#define HZ_S(dz, dy, dx) HZ(dz, dy, dx)

  TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
  TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];

  bool const pml_z_v = (z < pml_z0) || (z >= pml_z1);
  bool const pml_y_v = (y < pml_y0) || (y >= pml_y1);
  bool const pml_x_v = (x < pml_x0) || (x >= pml_x1);

  TIDE_DTYPE dHy_dz = DIFFZ1(HY_S);
  TIDE_DTYPE dHz_dy = DIFFY1(HZ_S);
  TIDE_DTYPE dHz_dx = DIFFX1(HZ_S);
  TIDE_DTYPE dHx_dz = DIFFZ1(HX_S);
  TIDE_DTYPE dHx_dy = DIFFY1(HX_S);
  TIDE_DTYPE dHy_dx = DIFFX1(HY_S);

  if (pml_z_v) {
    m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
    dHy_dz = dHy_dz / kz[z] + m_hy_z[i];
    m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
    dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
  }
  if (pml_y_v) {
    m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
    dHz_dy = dHz_dy / ky[y] + m_hz_y[i];
    m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
    dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
  }
  if (pml_x_v) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
    dHz_dx = dHz_dx / kx[x] + m_hz_x[i];
    m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
    dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
  }

  TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
  TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
  TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

  if (ca_requires_grad && ex_store != nullptr) {
    ex_store[i] = ex[i];
    ey_store[i] = ey[i];
    ez_store[i] = ez[i];
  }
  if (cb_requires_grad && curl_x_store != nullptr) {
    curl_x_store[i] = curl_x;
    curl_y_store[i] = curl_y;
    curl_z_store[i] = curl_z;
  }

  ex[i] = ca_val * ex[i] + cb_val * curl_x;
  ey[i] = ca_val * ey[i] + cb_val * curl_y;
  ez[i] = ca_val * ez[i] + cb_val * curl_z;

#undef HX_S
#undef HY_S
#undef HZ_S
}

__global__ void add_adjoint_receivers_component(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const grad_r,
    int64_t const *__restrict const receivers_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_receivers_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const rec = receivers_i[idx];
  if (rec >= 0) {
    int64_t const shot_idx = idx / n_receivers_per_shot;
    field[shot_idx * shot_numel + rec] += grad_r[idx];
  }
}

__global__ void record_adjoint_at_sources_component(
    TIDE_DTYPE *__restrict const grad_f,
    TIDE_DTYPE const *__restrict const field,
    int64_t const *__restrict const sources_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_sources_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const src = sources_i[idx];
  if (src >= 0) {
    int64_t const shot_idx = idx / n_sources_per_shot;
    grad_f[idx] = field[shot_idx * shot_numel + src];
  } else {
    grad_f[idx] = (TIDE_DTYPE)0;
  }
}

__global__ void backward_kernel_lambda_h(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ex,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const lambda_ez,
    TIDE_DTYPE *__restrict const lambda_hx,
    TIDE_DTYPE *__restrict const lambda_hy,
    TIDE_DTYPE *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
    TIDE_DTYPE *__restrict const m_lambda_ez_y,
    TIDE_DTYPE *__restrict const m_lambda_ez_x,
    TIDE_DTYPE *__restrict const m_lambda_ex_z,
    TIDE_DTYPE *__restrict const m_lambda_ex_y,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  int64_t const shot_idx = i / shot_numel;
  int64_t const j = i - shot_idx * shot_numel;

  int64_t const yz_stride = ny * nx;
  int64_t const z = j / yz_stride;
  int64_t const rem = j - z * yz_stride;
  int64_t const y = rem / nx;
  int64_t const x = rem - y * nx;

  if (z < FD_PAD || z >= nz - FD_PAD + 1 || y < FD_PAD || y >= ny - FD_PAD + 1 ||
      x < FD_PAD || x >= nx - FD_PAD + 1) {
    return;
  }

#define LEX_H(dz, dy, dx) lambda_ex[ND_INDEX(i, dz, dy, dx)]
#define LEY_H(dz, dy, dx) lambda_ey[ND_INDEX(i, dz, dy, dx)]
#define LEZ_H(dz, dy, dx) lambda_ez[ND_INDEX(i, dz, dy, dx)]

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = tide_max(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);

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
    TIDE_DTYPE dLey_dz = -DIFFZ1_ADJ(CB_I, LEY_H);
    if (pml_z_h) {
      m_lambda_ey_z[i] = bzh[z] * m_lambda_ey_z[i] + azh[z] * dLey_dz;
      dLey_dz = dLey_dz / kzh[z] + m_lambda_ey_z[i];
    }
    dLey_dz_pml = dLey_dz;
  }
  if (y < ny - FD_PAD) {
    TIDE_DTYPE dLez_dy = -DIFFY1_ADJ(CB_I, LEZ_H);
    if (pml_y_h) {
      m_lambda_ez_y[i] = byh[y] * m_lambda_ez_y[i] + ayh[y] * dLez_dy;
      dLez_dy = dLez_dy / kyh[y] + m_lambda_ez_y[i];
    }
    dLez_dy_pml = dLez_dy;
  }
  if (x < nx - FD_PAD) {
    TIDE_DTYPE dLez_dx = -DIFFX1_ADJ(CB_I, LEZ_H);
    if (pml_x_h) {
      m_lambda_ez_x[i] = bxh[x] * m_lambda_ez_x[i] + axh[x] * dLez_dx;
      dLez_dx = dLez_dx / kxh[x] + m_lambda_ez_x[i];
    }
    dLez_dx_pml = dLez_dx;
  }
  if (z < nz - FD_PAD) {
    TIDE_DTYPE dLex_dz = -DIFFZ1_ADJ(CB_I, LEX_H);
    if (pml_z_h) {
      m_lambda_ex_z[i] = bzh[z] * m_lambda_ex_z[i] + azh[z] * dLex_dz;
      dLex_dz = dLex_dz / kzh[z] + m_lambda_ex_z[i];
    }
    dLex_dz_pml = dLex_dz;
  }
  if (y < ny - FD_PAD) {
    TIDE_DTYPE dLex_dy = -DIFFY1_ADJ(CB_I, LEX_H);
    if (pml_y_h) {
      m_lambda_ex_y[i] = byh[y] * m_lambda_ex_y[i] + ayh[y] * dLex_dy;
      dLex_dy = dLex_dy / kyh[y] + m_lambda_ex_y[i];
    }
    dLex_dy_pml = dLex_dy;
  }
  if (x < nx - FD_PAD) {
    TIDE_DTYPE dLey_dx = -DIFFX1_ADJ(CB_I, LEY_H);
    if (pml_x_h) {
      m_lambda_ey_x[i] = bxh[x] * m_lambda_ey_x[i] + axh[x] * dLey_dx;
      dLey_dx = dLey_dx / kxh[x] + m_lambda_ey_x[i];
    }
    dLey_dx_pml = dLey_dx;
  }

  lambda_hx[i] += dLey_dz_pml - dLez_dy_pml;
  lambda_hy[i] += dLez_dx_pml - dLex_dz_pml;
  lambda_hz[i] += dLex_dy_pml - dLey_dx_pml;

#undef LEX_H
#undef LEY_H
#undef LEZ_H
}

__global__ void backward_kernel_lambda_e_with_grad(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hy,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ex,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const lambda_ez,
    TIDE_DTYPE *__restrict const m_lambda_hz_y,
    TIDE_DTYPE *__restrict const m_lambda_hy_z,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const m_lambda_hy_x,
    TIDE_DTYPE *__restrict const m_lambda_hx_y,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const ex_store,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const ez_store,
    TIDE_DTYPE const *__restrict const curl_x_store,
    TIDE_DTYPE const *__restrict const curl_y_store,
    TIDE_DTYPE const *__restrict const curl_z_store,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
    bool const grad_ca_step,
    bool const grad_cb_step,
    int64_t const step_ratio_eff) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  int64_t const shot_idx = i / shot_numel;
  int64_t const j = i - shot_idx * shot_numel;

  int64_t const yz_stride = ny * nx;
  int64_t const z = j / yz_stride;
  int64_t const rem = j - z * yz_stride;
  int64_t const y = rem / nx;
  int64_t const x = rem - y * nx;

  if (z < FD_PAD || z >= nz - FD_PAD + 1 || y < FD_PAD || y >= ny - FD_PAD + 1 ||
      x < FD_PAD || x >= nx - FD_PAD + 1) {
    return;
  }

#define LHX_E(dz, dy, dx) lambda_hx[ND_INDEX(i, dz, dy, dx)]
#define LHY_E(dz, dy, dx) lambda_hy[ND_INDEX(i, dz, dy, dx)]
#define LHZ_E(dz, dy, dx) lambda_hz[ND_INDEX(i, dz, dy, dx)]

  TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];

  bool const pml_z_v = (z < pml_z0) || (z >= pml_z1);
  bool const pml_y_v = (y < pml_y0) || (y >= pml_y1);
  bool const pml_x_v = (x < pml_x0) || (x >= pml_x1);

  TIDE_DTYPE dLhy_dz = DIFFZH1_ADJ(CQ_I, LHY_E);
  TIDE_DTYPE dLhz_dy = DIFFYH1_ADJ(CQ_I, LHZ_E);
  TIDE_DTYPE dLhz_dx = DIFFXH1_ADJ(CQ_I, LHZ_E);
  TIDE_DTYPE dLhx_dz = DIFFZH1_ADJ(CQ_I, LHX_E);
  TIDE_DTYPE dLhx_dy = DIFFYH1_ADJ(CQ_I, LHX_E);
  TIDE_DTYPE dLhy_dx = DIFFXH1_ADJ(CQ_I, LHY_E);

  if (pml_z_v) {
    m_lambda_hy_z[i] = bz[z] * m_lambda_hy_z[i] + az[z] * dLhy_dz;
    dLhy_dz = dLhy_dz / kz[z] + m_lambda_hy_z[i];
    m_lambda_hx_z[i] = bz[z] * m_lambda_hx_z[i] + az[z] * dLhx_dz;
    dLhx_dz = dLhx_dz / kz[z] + m_lambda_hx_z[i];
  }
  if (pml_y_v) {
    m_lambda_hz_y[i] = by[y] * m_lambda_hz_y[i] + ay[y] * dLhz_dy;
    dLhz_dy = dLhz_dy / ky[y] + m_lambda_hz_y[i];
    m_lambda_hx_y[i] = by[y] * m_lambda_hx_y[i] + ay[y] * dLhx_dy;
    dLhx_dy = dLhx_dy / ky[y] + m_lambda_hx_y[i];
  }
  if (pml_x_v) {
    m_lambda_hz_x[i] = bx[x] * m_lambda_hz_x[i] + ax[x] * dLhz_dx;
    dLhz_dx = dLhz_dx / kx[x] + m_lambda_hz_x[i];
    m_lambda_hy_x[i] = bx[x] * m_lambda_hy_x[i] + ax[x] * dLhy_dx;
    dLhy_dx = dLhy_dx / kx[x] + m_lambda_hy_x[i];
  }

  TIDE_DTYPE const curl_lambda_x = dLhy_dz - dLhz_dy;
  TIDE_DTYPE const curl_lambda_y = dLhz_dx - dLhx_dz;
  TIDE_DTYPE const curl_lambda_z = dLhx_dy - dLhy_dx;

  TIDE_DTYPE const lex_curr = lambda_ex[i];
  TIDE_DTYPE const ley_curr = lambda_ey[i];
  TIDE_DTYPE const lez_curr = lambda_ez[i];

  lambda_ex[i] = ca_val * lex_curr + curl_lambda_x;
  lambda_ey[i] = ca_val * ley_curr + curl_lambda_y;
  lambda_ez[i] = ca_val * lez_curr + curl_lambda_z;

  if (grad_ca_step && grad_ca != nullptr && ex_store != nullptr && ey_store != nullptr &&
      ez_store != nullptr) {
    TIDE_DTYPE const acc_ca =
        lex_curr * ex_store[i] + ley_curr * ey_store[i] + lez_curr * ez_store[i];
    TIDE_DTYPE const scaled = acc_ca * (TIDE_DTYPE)step_ratio_eff;
    if (ca_batched) {
      grad_ca[i] += scaled;
    } else {
      atomic_add_tide(&grad_ca[j], scaled);
    }
  }

  if (grad_cb_step && grad_cb != nullptr && curl_x_store != nullptr &&
      curl_y_store != nullptr && curl_z_store != nullptr) {
    TIDE_DTYPE const acc_cb = lex_curr * curl_x_store[i] + ley_curr * curl_y_store[i] +
                              lez_curr * curl_z_store[i];
    TIDE_DTYPE const scaled = acc_cb * (TIDE_DTYPE)step_ratio_eff;
    if (cb_batched) {
      grad_cb[i] += scaled;
    } else {
      atomic_add_tide(&grad_cb[j], scaled);
    }
  }

#undef LHX_E
#undef LHY_E
#undef LHZ_E
}

__global__ void convert_grad_ca_cb_to_eps_sigma_3d(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const grad_ca,
    TIDE_DTYPE const *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE const dt_h,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = ca_batched ? (n_shots * shot_numel) : shot_numel;
  if (idx >= total) {
    return;
  }

  int64_t shot_idx = ca_batched ? (idx / shot_numel) : 0;
  int64_t j = ca_batched ? (idx - shot_idx * shot_numel) : idx;
  int64_t out_idx = ca_batched ? idx : j;
  int64_t ca_idx = ca_batched ? idx : j;
  int64_t cb_idx = cb_batched ? (shot_idx * shot_numel + j) : j;

  TIDE_DTYPE const ca_val = ca[ca_idx];
  TIDE_DTYPE const cb_val = cb[cb_idx];
  TIDE_DTYPE const cb_sq = cb_val * cb_val;
  TIDE_DTYPE const inv_dt = (TIDE_DTYPE)1 / dt_h;

  TIDE_DTYPE grad_ca_val = 0;
  TIDE_DTYPE grad_cb_val = 0;
  if (ca_requires_grad && grad_ca != nullptr) {
    grad_ca_val = grad_ca[out_idx];
  }
  if (cb_requires_grad && grad_cb != nullptr) {
    grad_cb_val = grad_cb[out_idx];
  }

  TIDE_DTYPE const dca_de = ((TIDE_DTYPE)1 - ca_val) * cb_val * inv_dt;
  TIDE_DTYPE const dcb_de = -cb_sq * inv_dt;
  TIDE_DTYPE const dca_ds = -((TIDE_DTYPE)0.5) * ((TIDE_DTYPE)1 + ca_val) * cb_val;
  TIDE_DTYPE const dcb_ds = -((TIDE_DTYPE)0.5) * cb_sq;

  if (grad_eps != nullptr) {
    TIDE_DTYPE const grad_e = grad_ca_val * dca_de + grad_cb_val * dcb_de;
    grad_eps[out_idx] = grad_e * kEp0;
  }
  if (grad_sigma != nullptr) {
    grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
  }
}

static void set_constants(
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const shot_numel_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h) {
  struct ConstantCache3D {
    bool initialized = false;
    TIDE_DTYPE rdz_h = (TIDE_DTYPE)0;
    TIDE_DTYPE rdy_h = (TIDE_DTYPE)0;
    TIDE_DTYPE rdx_h = (TIDE_DTYPE)0;
    int64_t n_shots_h = -1;
    int64_t nz_h = -1;
    int64_t ny_h = -1;
    int64_t nx_h = -1;
    int64_t shot_numel_h = -1;
    int64_t n_sources_per_shot_h = -1;
    int64_t n_receivers_per_shot_h = -1;
    int64_t pml_z0_h = -1;
    int64_t pml_y0_h = -1;
    int64_t pml_x0_h = -1;
    int64_t pml_z1_h = -1;
    int64_t pml_y1_h = -1;
    int64_t pml_x1_h = -1;
    bool ca_batched_h = false;
    bool cb_batched_h = false;
    bool cq_batched_h = false;
  };
  static ConstantCache3D cache{};
  if (cache.initialized && cache.rdz_h == rdz_h && cache.rdy_h == rdy_h &&
      cache.rdx_h == rdx_h && cache.n_shots_h == n_shots_h &&
      cache.nz_h == nz_h && cache.ny_h == ny_h && cache.nx_h == nx_h &&
      cache.shot_numel_h == shot_numel_h &&
      cache.n_sources_per_shot_h == n_sources_per_shot_h &&
      cache.n_receivers_per_shot_h == n_receivers_per_shot_h &&
      cache.pml_z0_h == pml_z0_h && cache.pml_y0_h == pml_y0_h &&
      cache.pml_x0_h == pml_x0_h && cache.pml_z1_h == pml_z1_h &&
      cache.pml_y1_h == pml_y1_h && cache.pml_x1_h == pml_x1_h &&
      cache.ca_batched_h == ca_batched_h &&
      cache.cb_batched_h == cb_batched_h &&
      cache.cq_batched_h == cq_batched_h) {
    return;
  }

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(TIDE_DTYPE)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE)),
                            __FILE__, __LINE__);

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(
      cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t)),
      __FILE__, __LINE__);
  tide::cuda_check_or_abort(
      cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t)),
      __FILE__, __LINE__);

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)),
                            __FILE__, __LINE__);

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool)),
                            __FILE__, __LINE__);

  cache.initialized = true;
  cache.rdz_h = rdz_h;
  cache.rdy_h = rdy_h;
  cache.rdx_h = rdx_h;
  cache.n_shots_h = n_shots_h;
  cache.nz_h = nz_h;
  cache.ny_h = ny_h;
  cache.nx_h = nx_h;
  cache.shot_numel_h = shot_numel_h;
  cache.n_sources_per_shot_h = n_sources_per_shot_h;
  cache.n_receivers_per_shot_h = n_receivers_per_shot_h;
  cache.pml_z0_h = pml_z0_h;
  cache.pml_y0_h = pml_y0_h;
  cache.pml_x0_h = pml_x0_h;
  cache.pml_z1_h = pml_z1_h;
  cache.pml_y1_h = pml_y1_h;
  cache.pml_x1_h = pml_x1_h;
  cache.ca_batched_h = ca_batched_h;
  cache.cb_batched_h = cb_batched_h;
  cache.cq_batched_h = cq_batched_h;
}

struct ScalarLaunchConfig3D {
  int threads_cells;
  int64_t blocks_cells;
  int threads_sr;
  int64_t blocks_sources;
  int64_t blocks_receivers;
};

static inline ScalarLaunchConfig3D make_scalar_launch_config_3d(
    int64_t const n_shots_h, int64_t const shot_numel_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h) {
  ScalarLaunchConfig3D cfg{};
  cfg.threads_cells = 256;
  int64_t const total_cells = n_shots_h * shot_numel_h;
  cfg.blocks_cells = (total_cells + cfg.threads_cells - 1) / cfg.threads_cells;

  cfg.threads_sr = 256;
  int64_t const total_sources = n_shots_h * n_sources_per_shot_h;
  int64_t const total_receivers = n_shots_h * n_receivers_per_shot_h;
  cfg.blocks_sources =
      total_sources > 0 ? (total_sources + cfg.threads_sr - 1) / cfg.threads_sr
                        : 1;
  cfg.blocks_receivers =
      total_receivers > 0 ? (total_receivers + cfg.threads_sr - 1) / cfg.threads_sr
                          : 1;
  return cfg;
}

}  // namespace

extern "C" void FUNC(forward)(
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device) {
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;

  cudaSetDevice((int)device);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *source_field = ey;
  if (source_component == 0) {
    source_field = ex;
  } else if (source_component == 2) {
    source_field = ez;
  }

  TIDE_DTYPE const *receiver_field = ey;
  if (receiver_component == 0) {
    receiver_field = ex;
  } else if (receiver_component == 2) {
    receiver_field = ez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, shot_numel_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h<<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells>>>(
        cq,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        m_ey_z,
        m_ez_y,
        m_ez_x,
        m_ex_z,
        m_ex_y,
        m_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    forward_kernel_e<<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells>>>(
        ca,
        cb,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        m_hy_z,
        m_hz_y,
        m_hz_x,
        m_hx_z,
        m_hx_y,
        m_hy_x,
        az,
        bz,
        ay,
        by,
        ax,
        bx,
        kz,
        ky,
        kx);

    if (n_sources_per_shot_h > 0 && f != nullptr && sources_i != nullptr) {
      add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                              launch_cfg.threads_sr>>>(
          source_field,
          f + t * n_shots_h * n_sources_per_shot_h,
          sources_i);
    }

    if (n_receivers_per_shot_h > 0 && r != nullptr && receivers_i != nullptr) {
      record_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                   launch_cfg.threads_sr>>>(
          r + t * n_shots_h * n_receivers_per_shot_h,
          receiver_field,
          receivers_i);
    }
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(forward_with_storage)(
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
    TIDE_DTYPE *const store_1,
    TIDE_DTYPE *const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *const store_3,
    TIDE_DTYPE *const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *const store_5,
    TIDE_DTYPE *const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *const store_7,
    TIDE_DTYPE *const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *const store_9,
    TIDE_DTYPE *const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *const store_11,
    TIDE_DTYPE *const store_12,
    char **store_filenames_6,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device) {
  (void)dt_h;
  (void)n_threads;
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

  cudaSetDevice((int)device);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const step_ratio_eff = step_ratio_h > 0 ? step_ratio_h : 1;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      (shot_bytes_uncomp == (int64_t)(shot_numel_h * (int64_t)sizeof(TIDE_DTYPE)));

  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *source_field = ey;
  if (source_component == 0) {
    source_field = ex;
  } else if (source_component == 2) {
    source_field = ez;
  }

  TIDE_DTYPE const *receiver_field = ey;
  if (receiver_component == 0) {
    receiver_field = ex;
  } else if (receiver_component == 2) {
    receiver_field = ez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, shot_numel_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h<<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells>>>(
        cq,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        m_ey_z,
        m_ez_y,
        m_ez_x,
        m_ex_z,
        m_ex_y,
        m_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    bool const do_store = storage_direct && ((t % step_ratio_eff) == 0);
    if (do_store) {
      int64_t const store_idx = t / step_ratio_eff;
      int64_t const step_offset = store_idx * n_shots_h * shot_numel_h;
      TIDE_DTYPE *const ex_store_t =
          (ca_requires_grad && store_1 != nullptr) ? (store_1 + step_offset) : nullptr;
      TIDE_DTYPE *const ey_store_t =
          (ca_requires_grad && store_3 != nullptr) ? (store_3 + step_offset) : nullptr;
      TIDE_DTYPE *const ez_store_t =
          (ca_requires_grad && store_5 != nullptr) ? (store_5 + step_offset) : nullptr;
      TIDE_DTYPE *const curl_x_store_t =
          (cb_requires_grad && store_7 != nullptr) ? (store_7 + step_offset) : nullptr;
      TIDE_DTYPE *const curl_y_store_t =
          (cb_requires_grad && store_9 != nullptr) ? (store_9 + step_offset) : nullptr;
      TIDE_DTYPE *const curl_z_store_t =
          (cb_requires_grad && store_11 != nullptr) ? (store_11 + step_offset) : nullptr;

      forward_kernel_e_with_storage<<<(unsigned)launch_cfg.blocks_cells,
                                      launch_cfg.threads_cells>>>(
          ca,
          cb,
          ex,
          ey,
          ez,
          hx,
          hy,
          hz,
          m_hy_z,
          m_hz_y,
          m_hz_x,
          m_hx_z,
          m_hx_y,
          m_hy_x,
          az,
          bz,
          ay,
          by,
          ax,
          bx,
          kz,
          ky,
          kx,
          ex_store_t,
          ey_store_t,
          ez_store_t,
          curl_x_store_t,
          curl_y_store_t,
          curl_z_store_t,
          ca_requires_grad,
          cb_requires_grad);
    } else {
      forward_kernel_e<<<(unsigned)launch_cfg.blocks_cells,
                         launch_cfg.threads_cells>>>(
          ca,
          cb,
          ex,
          ey,
          ez,
          hx,
          hy,
          hz,
          m_hy_z,
          m_hz_y,
          m_hz_x,
          m_hx_z,
          m_hx_y,
          m_hy_x,
          az,
          bz,
          ay,
          by,
          ax,
          bx,
          kz,
          ky,
          kx);
    }

    if (n_sources_per_shot_h > 0 && f != nullptr && sources_i != nullptr) {
      add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                              launch_cfg.threads_sr>>>(
          source_field,
          f + t * n_shots_h * n_sources_per_shot_h,
          sources_i);
    }
    if (n_receivers_per_shot_h > 0 && r != nullptr && receivers_i != nullptr) {
      record_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                   launch_cfg.threads_sr>>>(
          r + t * n_shots_h * n_receivers_per_shot_h,
          receiver_field,
          receivers_i);
    }
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(backward)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const lambda_ex,
    TIDE_DTYPE *const lambda_ey,
    TIDE_DTYPE *const lambda_ez,
    TIDE_DTYPE *const lambda_hx,
    TIDE_DTYPE *const lambda_hy,
    TIDE_DTYPE *const lambda_hz,
    TIDE_DTYPE *const m_lambda_ey_z,
    TIDE_DTYPE *const m_lambda_ez_y,
    TIDE_DTYPE *const m_lambda_ez_x,
    TIDE_DTYPE *const m_lambda_ex_z,
    TIDE_DTYPE *const m_lambda_ex_y,
    TIDE_DTYPE *const m_lambda_ey_x,
    TIDE_DTYPE *const m_lambda_hz_y,
    TIDE_DTYPE *const m_lambda_hy_z,
    TIDE_DTYPE *const m_lambda_hx_z,
    TIDE_DTYPE *const m_lambda_hz_x,
    TIDE_DTYPE *const m_lambda_hy_x,
    TIDE_DTYPE *const m_lambda_hx_y,
    TIDE_DTYPE *const store_1,
    TIDE_DTYPE *const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *const store_3,
    TIDE_DTYPE *const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *const store_5,
    TIDE_DTYPE *const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *const store_7,
    TIDE_DTYPE *const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *const store_9,
    TIDE_DTYPE *const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *const store_11,
    TIDE_DTYPE *const store_12,
    char **store_filenames_6,
    TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_eps,
    TIDE_DTYPE *const grad_sigma,
    TIDE_DTYPE *const grad_ca_shot,
    TIDE_DTYPE *const grad_cb_shot,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device) {
  cudaSetDevice((int)device);
  (void)n_threads;
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

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const step_ratio_eff = step_ratio_h > 0 ? step_ratio_h : 1;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      (shot_bytes_uncomp == (int64_t)(shot_numel_h * (int64_t)sizeof(TIDE_DTYPE)));

  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *lambda_src_field = lambda_ey;
  if (source_component == 0) {
    lambda_src_field = lambda_ex;
  } else if (source_component == 2) {
    lambda_src_field = lambda_ez;
  }

  TIDE_DTYPE *lambda_recv_field = lambda_ey;
  if (receiver_component == 0) {
    lambda_recv_field = lambda_ex;
  } else if (receiver_component == 2) {
    lambda_recv_field = lambda_ez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, shot_numel_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  if (grad_f != nullptr && nt > 0 && n_shots_h > 0 && n_sources_per_shot_h > 0) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_f, 0,
                   (size_t)nt * (size_t)n_shots_h *
                       (size_t)n_sources_per_shot_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (ca_requires_grad && grad_ca != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_ca, 0,
                   (size_t)(ca_batched_h ? n_shots_h : 1) *
                       (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (cb_requires_grad && grad_cb != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_cb, 0,
                   (size_t)(cb_batched_h ? n_shots_h : 1) *
                       (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if ((ca_requires_grad || cb_requires_grad) && grad_eps != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_eps, 0,
                   (size_t)(ca_batched_h ? n_shots_h : 1) *
                       (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if ((ca_requires_grad || cb_requires_grad) && grad_sigma != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_sigma, 0,
                   (size_t)(ca_batched_h ? n_shots_h : 1) *
                       (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (grad_ca_shot != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_ca_shot, 0,
                   (size_t)n_shots_h * (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (grad_cb_shot != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_cb_shot, 0,
                   (size_t)n_shots_h * (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    bool const do_grad = ((t % step_ratio_eff) == 0);
    bool const grad_ca_step = do_grad && ca_requires_grad && storage_direct &&
                              store_1 != nullptr && store_3 != nullptr &&
                              store_5 != nullptr;
    bool const grad_cb_step = do_grad && cb_requires_grad && storage_direct &&
                              store_7 != nullptr && store_9 != nullptr &&
                              store_11 != nullptr;

    int64_t const store_idx = t / step_ratio_eff;
    int64_t const store_offset = store_idx * n_shots_h * shot_numel_h;

    TIDE_DTYPE const *const ex_store = grad_ca_step ? (store_1 + store_offset) : nullptr;
    TIDE_DTYPE const *const ey_store = grad_ca_step ? (store_3 + store_offset) : nullptr;
    TIDE_DTYPE const *const ez_store = grad_ca_step ? (store_5 + store_offset) : nullptr;
    TIDE_DTYPE const *const curl_x_store =
        grad_cb_step ? (store_7 + store_offset) : nullptr;
    TIDE_DTYPE const *const curl_y_store =
        grad_cb_step ? (store_9 + store_offset) : nullptr;
    TIDE_DTYPE const *const curl_z_store =
        grad_cb_step ? (store_11 + store_offset) : nullptr;

    if (n_receivers_per_shot_h > 0 && grad_r != nullptr && receivers_i != nullptr) {
      add_adjoint_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                        launch_cfg.threads_sr>>>(
          lambda_recv_field,
          grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

    if (n_sources_per_shot_h > 0 && grad_f != nullptr && sources_i != nullptr) {
      record_adjoint_at_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                                            launch_cfg.threads_sr>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h,
          lambda_src_field,
          sources_i);
    }

    backward_kernel_lambda_h<<<(unsigned)launch_cfg.blocks_cells,
                               launch_cfg.threads_cells>>>(
        cb,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_ey_z,
        m_lambda_ez_y,
        m_lambda_ez_x,
        m_lambda_ex_z,
        m_lambda_ex_y,
        m_lambda_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    backward_kernel_lambda_e_with_grad<<<(unsigned)launch_cfg.blocks_cells,
                                         launch_cfg.threads_cells>>>(
        ca,
        cq,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        m_lambda_hz_y,
        m_lambda_hy_z,
        m_lambda_hx_z,
        m_lambda_hz_x,
        m_lambda_hy_x,
        m_lambda_hx_y,
        az,
        bz,
        ay,
        by,
        ax,
        bx,
        kz,
        ky,
        kx,
        ex_store,
        ey_store,
        ez_store,
        curl_x_store,
        curl_y_store,
        curl_z_store,
        grad_ca,
        grad_cb,
        grad_ca_step,
        grad_cb_step,
        step_ratio_eff);
  }

  if ((grad_eps != nullptr || grad_sigma != nullptr) &&
      (ca_requires_grad || cb_requires_grad)) {
    int64_t const total_conv = (ca_batched_h ? n_shots_h : 1) * shot_numel_h;
    int const threads_conv = 256;
    int64_t const blocks_conv = (total_conv + threads_conv - 1) / threads_conv;
    convert_grad_ca_cb_to_eps_sigma_3d<<<(unsigned)blocks_conv, threads_conv>>>(
        ca,
        cb,
        grad_ca,
        grad_cb,
        grad_eps,
        grad_sigma,
        dt_h,
        ca_requires_grad,
        cb_requires_grad);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

#undef ND_INDEX
#undef ND_INDEX_J
#undef EX
#undef EY
#undef EZ
#undef HX
#undef HY
#undef HZ
#undef CB_I
#undef CQ_I
#undef gpuErrchk

} // namespace FUNC(Inst)
