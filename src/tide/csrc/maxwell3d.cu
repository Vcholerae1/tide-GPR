/*
 * 3D Maxwell FDTD propagator (CUDA implementation)
 *
 * Forward-only implementation for 3D full Maxwell with CPML.
 * Grid ordering: [nz, ny, nx] with z as slowest dimension.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

#include "common_gpu.h"
#include "storage_utils.h"
#include "staggered_grid_3d.h"

namespace cg = cooperative_groups;

// CPU storage pipelining: Number of ping-pong buffers for async D2H/H2D copies
#ifndef NUM_BUFFERS
#define NUM_BUFFERS 3
#endif

// Profiling support: enable with -DTIDE_PROFILING during compilation
#ifdef TIDE_PROFILING
#define PROF_EVENT_CREATE(e) cudaEventCreate(&(e))
#define PROF_RECORD(e, s) cudaEventRecord((e), (s))
#define PROF_ELAPSED(start, end, ms) cudaEventElapsedTime(&(ms), (start), (end))
#define PROF_PRINT(name, ms) fprintf(stderr, "[TIDE PROF] %s: %.3f ms\n", (name), (ms))
#else
#define PROF_EVENT_CREATE(e) ((void)0)
#define PROF_RECORD(e, s) ((void)0)
#define PROF_ELAPSED(start, end, ms) ((void)0)
#define PROF_PRINT(name, ms) ((void)0)
#endif

#define CAT_I(name, accuracy, dtype, device) \
  maxwell_3d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) \
  CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, cuda)

// 3D indexing macros
#define IDX(z, y, x) ((z) * ny * nx + (y) * nx + (x))
#define IDX_SHOT(shot, z, y, x) ((shot) * shot_numel + (z) * ny * nx + (y) * nx + (x))

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define EP0 ((TIDE_DTYPE)8.8541878128e-12)

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

// Adjoint field access macros
#define LEX(dz, dy, dx) lambda_ex[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LEY(dz, dy, dx) lambda_ey[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LEZ(dz, dy, dx) lambda_ez[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LHX(dz, dy, dx) lambda_hx[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LHY(dz, dy, dx) lambda_hy[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define LHZ(dz, dy, dx) lambda_hz[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

// Adjoint PML memory variables (E-derived)
#define M_L_EY_Z(dz, dy, dx) m_lambda_ey_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_EZ_Y(dz, dy, dx) m_lambda_ez_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_EZ_X(dz, dy, dx) m_lambda_ez_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_EX_Z(dz, dy, dx) m_lambda_ex_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_EX_Y(dz, dy, dx) m_lambda_ex_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_EY_X(dz, dy, dx) m_lambda_ey_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]

// Adjoint PML memory variables (H-derived)
#define M_L_HZ_Y(dz, dy, dx) m_lambda_hz_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_HY_Z(dz, dy, dx) m_lambda_hy_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_HX_Z(dz, dy, dx) m_lambda_hx_z[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_HZ_X(dz, dy, dx) m_lambda_hz_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_HY_X(dz, dy, dx) m_lambda_hy_x[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]
#define M_L_HX_Y(dz, dy, dx) m_lambda_hx_y[IDX_SHOT(shot_idx, z + (dz), y + (dy), x + (dx))]


namespace {

__device__ __forceinline__ TIDE_DTYPE diffx1_sh(
    TIDE_DTYPE const *const s,
    int const lidx,
    int const pitch,
    TIDE_DTYPE const rdx) {
#if TIDE_STENCIL == 2
  return (s[lidx] - s[lidx - 1]) * rdx;
#elif TIDE_STENCIL == 4
  return ((TIDE_DTYPE)(9.0 / 8.0) * (s[lidx] - s[lidx - 1]) +
          (TIDE_DTYPE)(-1.0 / 24.0) * (s[lidx + 1] - s[lidx - 2])) *
         rdx;
#elif TIDE_STENCIL == 6
  return ((TIDE_DTYPE)(75.0 / 64.0) * (s[lidx] - s[lidx - 1]) +
          (TIDE_DTYPE)(-25.0 / 384.0) * (s[lidx + 1] - s[lidx - 2]) +
          (TIDE_DTYPE)(3.0 / 640.0) * (s[lidx + 2] - s[lidx - 3])) *
         rdx;
#elif TIDE_STENCIL == 8
  return ((TIDE_DTYPE)(1225.0 / 1024.0) * (s[lidx] - s[lidx - 1]) +
          (TIDE_DTYPE)(-245.0 / 3072.0) * (s[lidx + 1] - s[lidx - 2]) +
          (TIDE_DTYPE)(49.0 / 5120.0) * (s[lidx + 2] - s[lidx - 3]) +
          (TIDE_DTYPE)(-5.0 / 7168.0) * (s[lidx + 3] - s[lidx - 4])) *
         rdx;
#endif
}

__device__ __forceinline__ TIDE_DTYPE diffxh1_sh(
    TIDE_DTYPE const *const s,
    int const lidx,
    int const pitch,
    TIDE_DTYPE const rdx) {
#if TIDE_STENCIL == 2
  return (s[lidx + 1] - s[lidx]) * rdx;
#elif TIDE_STENCIL == 4
  return ((TIDE_DTYPE)(9.0 / 8.0) * (s[lidx + 1] - s[lidx]) +
          (TIDE_DTYPE)(-1.0 / 24.0) * (s[lidx + 2] - s[lidx - 1])) *
         rdx;
#elif TIDE_STENCIL == 6
  return ((TIDE_DTYPE)(75.0 / 64.0) * (s[lidx + 1] - s[lidx]) +
          (TIDE_DTYPE)(-25.0 / 384.0) * (s[lidx + 2] - s[lidx - 1]) +
          (TIDE_DTYPE)(3.0 / 640.0) * (s[lidx + 3] - s[lidx - 2])) *
         rdx;
#elif TIDE_STENCIL == 8
  return ((TIDE_DTYPE)(1225.0 / 1024.0) * (s[lidx + 1] - s[lidx]) +
          (TIDE_DTYPE)(-245.0 / 3072.0) * (s[lidx + 2] - s[lidx - 1]) +
          (TIDE_DTYPE)(49.0 / 5120.0) * (s[lidx + 3] - s[lidx - 2]) +
          (TIDE_DTYPE)(-5.0 / 7168.0) * (s[lidx + 4] - s[lidx - 3])) *
         rdx;
#endif
}

__device__ __forceinline__ TIDE_DTYPE diffy1_sh(
    TIDE_DTYPE const *const s,
    int const lidx,
    int const pitch,
    TIDE_DTYPE const rdy) {
#if TIDE_STENCIL == 2
  return (s[lidx] - s[lidx - pitch]) * rdy;
#elif TIDE_STENCIL == 4
  return ((TIDE_DTYPE)(9.0 / 8.0) * (s[lidx] - s[lidx - pitch]) +
          (TIDE_DTYPE)(-1.0 / 24.0) * (s[lidx + pitch] - s[lidx - 2 * pitch])) *
         rdy;
#elif TIDE_STENCIL == 6
  return ((TIDE_DTYPE)(75.0 / 64.0) * (s[lidx] - s[lidx - pitch]) +
          (TIDE_DTYPE)(-25.0 / 384.0) * (s[lidx + pitch] - s[lidx - 2 * pitch]) +
          (TIDE_DTYPE)(3.0 / 640.0) * (s[lidx + 2 * pitch] -
                                       s[lidx - 3 * pitch])) *
         rdy;
#elif TIDE_STENCIL == 8
  return ((TIDE_DTYPE)(1225.0 / 1024.0) * (s[lidx] - s[lidx - pitch]) +
          (TIDE_DTYPE)(-245.0 / 3072.0) *
              (s[lidx + pitch] - s[lidx - 2 * pitch]) +
          (TIDE_DTYPE)(49.0 / 5120.0) *
              (s[lidx + 2 * pitch] - s[lidx - 3 * pitch]) +
          (TIDE_DTYPE)(-5.0 / 7168.0) *
              (s[lidx + 3 * pitch] - s[lidx - 4 * pitch])) *
         rdy;
#endif
}

__device__ __forceinline__ TIDE_DTYPE diffyh1_sh(
    TIDE_DTYPE const *const s,
    int const lidx,
    int const pitch,
    TIDE_DTYPE const rdy) {
#if TIDE_STENCIL == 2
  return (s[lidx + pitch] - s[lidx]) * rdy;
#elif TIDE_STENCIL == 4
  return ((TIDE_DTYPE)(9.0 / 8.0) * (s[lidx + pitch] - s[lidx]) +
          (TIDE_DTYPE)(-1.0 / 24.0) * (s[lidx + 2 * pitch] -
                                       s[lidx - pitch])) *
         rdy;
#elif TIDE_STENCIL == 6
  return ((TIDE_DTYPE)(75.0 / 64.0) * (s[lidx + pitch] - s[lidx]) +
          (TIDE_DTYPE)(-25.0 / 384.0) * (s[lidx + 2 * pitch] -
                                         s[lidx - pitch]) +
          (TIDE_DTYPE)(3.0 / 640.0) * (s[lidx + 3 * pitch] -
                                       s[lidx - 2 * pitch])) *
         rdy;
#elif TIDE_STENCIL == 8
  return ((TIDE_DTYPE)(1225.0 / 1024.0) * (s[lidx + pitch] - s[lidx]) +
          (TIDE_DTYPE)(-245.0 / 3072.0) * (s[lidx + 2 * pitch] -
                                           s[lidx - pitch]) +
          (TIDE_DTYPE)(49.0 / 5120.0) * (s[lidx + 3 * pitch] -
                                         s[lidx - 2 * pitch]) +
          (TIDE_DTYPE)(-5.0 / 7168.0) * (s[lidx + 4 * pitch] -
                                         s[lidx - 3 * pitch])) *
         rdy;
#endif
}

__device__ __forceinline__ TIDE_DTYPE diffz1_reg(
    TIDE_DTYPE const *const q,
    TIDE_DTYPE const rdz) {
#if TIDE_STENCIL == 2
  return (q[FD_PAD] - q[FD_PAD - 1]) * rdz;
#elif TIDE_STENCIL == 4
  return ((TIDE_DTYPE)(9.0 / 8.0) * (q[FD_PAD] - q[FD_PAD - 1]) +
          (TIDE_DTYPE)(-1.0 / 24.0) * (q[FD_PAD + 1] - q[FD_PAD - 2])) *
         rdz;
#elif TIDE_STENCIL == 6
  return ((TIDE_DTYPE)(75.0 / 64.0) * (q[FD_PAD] - q[FD_PAD - 1]) +
          (TIDE_DTYPE)(-25.0 / 384.0) * (q[FD_PAD + 1] - q[FD_PAD - 2]) +
          (TIDE_DTYPE)(3.0 / 640.0) * (q[FD_PAD + 2] - q[FD_PAD - 3])) *
         rdz;
#elif TIDE_STENCIL == 8
  return ((TIDE_DTYPE)(1225.0 / 1024.0) * (q[FD_PAD] - q[FD_PAD - 1]) +
          (TIDE_DTYPE)(-245.0 / 3072.0) * (q[FD_PAD + 1] - q[FD_PAD - 2]) +
          (TIDE_DTYPE)(49.0 / 5120.0) * (q[FD_PAD + 2] - q[FD_PAD - 3]) +
          (TIDE_DTYPE)(-5.0 / 7168.0) * (q[FD_PAD + 3] - q[FD_PAD - 4])) *
         rdz;
#endif
}

__device__ __forceinline__ TIDE_DTYPE diffzh1_reg(
    TIDE_DTYPE const *const q,
    TIDE_DTYPE const rdz) {
#if TIDE_STENCIL == 2
  return (q[FD_PAD + 1] - q[FD_PAD]) * rdz;
#elif TIDE_STENCIL == 4
  return ((TIDE_DTYPE)(9.0 / 8.0) * (q[FD_PAD + 1] - q[FD_PAD]) +
          (TIDE_DTYPE)(-1.0 / 24.0) * (q[FD_PAD + 2] - q[FD_PAD - 1])) *
         rdz;
#elif TIDE_STENCIL == 6
  return ((TIDE_DTYPE)(75.0 / 64.0) * (q[FD_PAD + 1] - q[FD_PAD]) +
          (TIDE_DTYPE)(-25.0 / 384.0) * (q[FD_PAD + 2] - q[FD_PAD - 1]) +
          (TIDE_DTYPE)(3.0 / 640.0) * (q[FD_PAD + 3] - q[FD_PAD - 2])) *
         rdz;
#elif TIDE_STENCIL == 8
  return ((TIDE_DTYPE)(1225.0 / 1024.0) * (q[FD_PAD + 1] - q[FD_PAD]) +
          (TIDE_DTYPE)(-245.0 / 3072.0) * (q[FD_PAD + 2] - q[FD_PAD - 1]) +
          (TIDE_DTYPE)(49.0 / 5120.0) * (q[FD_PAD + 3] - q[FD_PAD - 2]) +
          (TIDE_DTYPE)(-5.0 / 7168.0) * (q[FD_PAD + 4] - q[FD_PAD - 3])) *
         rdz;
#endif
}

__device__ __forceinline__ void shift_queue(TIDE_DTYPE *const q) {
#pragma unroll
  for (int i = 0; i < 2 * FD_PAD; ++i) {
    q[i] = q[i + 1];
  }
}

__global__ void add_sources_field(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_sources_per_shot) {
  int64_t shot_idx = (int64_t)blockIdx.y;
  int64_t source_idx = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                       (int64_t)threadIdx.x;
  if (shot_idx < n_shots && source_idx < n_sources_per_shot) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (src >= 0) {
      field[shot_idx * shot_numel + src] += f[k];
    }
  }
}


__global__ void record_receivers_field(
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE const *__restrict const field,
    int64_t const *__restrict const receivers_i,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_receivers_per_shot) {
  int64_t shot_idx = (int64_t)blockIdx.y;
  int64_t receiver_idx = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                         (int64_t)threadIdx.x;
  if (shot_idx < n_shots && receiver_idx < n_receivers_per_shot) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (rec >= 0) {
      r[k] = field[shot_idx * shot_numel + rec];
    }
  }
}


__global__ __launch_bounds__(256) void forward_kernel_h_3d(
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
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const base_x = (int64_t)blockIdx.x * (int64_t)blockDim.x;
  int64_t const base_y = (int64_t)blockIdx.y * (int64_t)blockDim.y;
  int64_t const x = base_x + (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = base_y + (int64_t)threadIdx.y + FD_PAD;

  bool const in_domain = (x < nx - FD_PAD + 1) && (y < ny - FD_PAD + 1);
  bool const in_bounds_xy = (x < nx) && (y < ny);

  int64_t const plane_stride = ny * nx;
  int64_t const shot_offset = shot_idx * shot_numel;

  int const tile_w = (int)blockDim.x + 2 * FD_PAD;
  int const tile_h = (int)blockDim.y + 2 * FD_PAD;
  int const tile_size = tile_w * tile_h;
  bool const full_tile_xy =
      (base_x + tile_w <= nx) && (base_y + tile_h <= ny);

  extern __shared__ TIDE_DTYPE s_mem[];
  TIDE_DTYPE *const s_ex0 = s_mem;
  TIDE_DTYPE *const s_ey0 = s_ex0 + tile_size;
  TIDE_DTYPE *const s_ez0 = s_ey0 + tile_size;
  TIDE_DTYPE *const s_ex1 = s_ez0 + tile_size;
  TIDE_DTYPE *const s_ey1 = s_ex1 + tile_size;
  TIDE_DTYPE *const s_ez1 = s_ey1 + tile_size;

  TIDE_DTYPE *s_ex_cur = s_ex0;
  TIDE_DTYPE *s_ey_cur = s_ey0;
  TIDE_DTYPE *s_ez_cur = s_ez0;
  TIDE_DTYPE *s_ex_next = s_ex1;
  TIDE_DTYPE *s_ey_next = s_ey1;
  TIDE_DTYPE *s_ez_next = s_ez1;

  int const lx = (int)threadIdx.x + FD_PAD;
  int const ly = (int)threadIdx.y + FD_PAD;
  int const lidx = ly * tile_w + lx;

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

  bool const pml_y = y < pml_y0h || y >= pml_y1h;
  bool const pml_x = x < pml_x0h || x >= pml_x1h;

  TIDE_DTYPE ex_z[2 * FD_PAD + 1];
  TIDE_DTYPE ey_z[2 * FD_PAD + 1];

#pragma unroll
  for (int dz = -FD_PAD; dz <= FD_PAD; ++dz) {
    int64_t const z_idx = (int64_t)FD_PAD + (int64_t)dz;
    TIDE_DTYPE ex_val = 0;
    TIDE_DTYPE ey_val = 0;
    if (in_bounds_xy && z_idx >= 0 && z_idx < nz) {
      int64_t const idx = shot_offset + z_idx * plane_stride + y * nx + x;
      ex_val = ex[idx];
      ey_val = ey[idx];
    }
    ex_z[dz + FD_PAD] = ex_val;
    ey_z[dz + FD_PAD] = ey_val;
  }

  int64_t const z_end = nz - FD_PAD + 1;
  if (z_end > FD_PAD) {
    int64_t const z_offset = shot_offset + (int64_t)FD_PAD * plane_stride;
    if (full_tile_xy) {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int const s_row = sy * tile_w;
        int64_t const row_offset = (base_y + (int64_t)sy) * nx;
        int64_t const idx_base = z_offset + row_offset + base_x;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int const s_idx = s_row + sx;
          int64_t const idx = idx_base + sx;
          s_ex_cur[s_idx] = ex[idx];
          s_ey_cur[s_idx] = ey[idx];
          s_ez_cur[s_idx] = ez[idx];
        }
      }
    } else {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int64_t const gy = base_y + (int64_t)sy;
        bool const y_ok = gy < ny;
        int const s_row = sy * tile_w;
        int64_t const row_offset = gy * nx;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int64_t const gx = base_x + (int64_t)sx;
          int const s_idx = s_row + sx;
          if (y_ok && gx < nx) {
            int64_t const idx = z_offset + row_offset + gx;
            s_ex_cur[s_idx] = ex[idx];
            s_ey_cur[s_idx] = ey[idx];
            s_ez_cur[s_idx] = ez[idx];
          } else {
            s_ex_cur[s_idx] = 0;
            s_ey_cur[s_idx] = 0;
            s_ez_cur[s_idx] = 0;
          }
        }
      }
    }
  }

  __syncthreads();

  for (int64_t z = FD_PAD; z < z_end; ++z) {
    if (in_domain) {
      bool const pml_z = z < pml_z0h || z >= pml_z1h;
      TIDE_DTYPE const cq_val = CQ(0, 0, 0);

      if (z < nz - FD_PAD && y < ny - FD_PAD) {
        TIDE_DTYPE dEy_dz = diffzh1_reg(ey_z, rdz);
        if (pml_z) {
          M_EY_Z(0, 0, 0) = bzh[z] * M_EY_Z(0, 0, 0) + azh[z] * dEy_dz;
          dEy_dz = dEy_dz / kzh[z] + M_EY_Z(0, 0, 0);
        }
        TIDE_DTYPE dEz_dy = diffyh1_sh(s_ez_cur, lidx, tile_w, rdy);
        if (pml_y) {
          M_EZ_Y(0, 0, 0) = byh[y] * M_EZ_Y(0, 0, 0) + ayh[y] * dEz_dy;
          dEz_dy = dEz_dy / kyh[y] + M_EZ_Y(0, 0, 0);
        }
        HX(0, 0, 0) -= cq_val * (dEy_dz - dEz_dy);
      }

      if (z < nz - FD_PAD && x < nx - FD_PAD) {
        TIDE_DTYPE dEz_dx = diffxh1_sh(s_ez_cur, lidx, tile_w, rdx);
        if (pml_x) {
          M_EZ_X(0, 0, 0) = bxh[x] * M_EZ_X(0, 0, 0) + axh[x] * dEz_dx;
          dEz_dx = dEz_dx / kxh[x] + M_EZ_X(0, 0, 0);
        }
        TIDE_DTYPE dEx_dz = diffzh1_reg(ex_z, rdz);
        if (pml_z) {
          M_EX_Z(0, 0, 0) = bzh[z] * M_EX_Z(0, 0, 0) + azh[z] * dEx_dz;
          dEx_dz = dEx_dz / kzh[z] + M_EX_Z(0, 0, 0);
        }
        HY(0, 0, 0) -= cq_val * (dEz_dx - dEx_dz);
      }

      if (y < ny - FD_PAD && x < nx - FD_PAD) {
        TIDE_DTYPE dEx_dy = diffyh1_sh(s_ex_cur, lidx, tile_w, rdy);
        if (pml_y) {
          M_EX_Y(0, 0, 0) = byh[y] * M_EX_Y(0, 0, 0) + ayh[y] * dEx_dy;
          dEx_dy = dEx_dy / kyh[y] + M_EX_Y(0, 0, 0);
        }
        TIDE_DTYPE dEy_dx = diffxh1_sh(s_ey_cur, lidx, tile_w, rdx);
        if (pml_x) {
          M_EY_X(0, 0, 0) = bxh[x] * M_EY_X(0, 0, 0) + axh[x] * dEy_dx;
          dEy_dx = dEy_dx / kxh[x] + M_EY_X(0, 0, 0);
        }
        HZ(0, 0, 0) -= cq_val * (dEx_dy - dEy_dx);
      }
    }

    if (z + 1 < z_end) {
      int64_t const z_offset = shot_offset + (z + 1) * plane_stride;
      if (full_tile_xy) {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int const s_row = sy * tile_w;
          int64_t const row_offset = (base_y + (int64_t)sy) * nx;
          int64_t const idx_base = z_offset + row_offset + base_x;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int const s_idx = s_row + sx;
            int64_t const idx = idx_base + sx;
            s_ex_next[s_idx] = ex[idx];
            s_ey_next[s_idx] = ey[idx];
            s_ez_next[s_idx] = ez[idx];
          }
        }
      } else {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int64_t const gy = base_y + (int64_t)sy;
          bool const y_ok = gy < ny;
          int const s_row = sy * tile_w;
          int64_t const row_offset = gy * nx;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int64_t const gx = base_x + (int64_t)sx;
            int const s_idx = s_row + sx;
            if (y_ok && gx < nx) {
              int64_t const idx = z_offset + row_offset + gx;
              s_ex_next[s_idx] = ex[idx];
              s_ey_next[s_idx] = ey[idx];
              s_ez_next[s_idx] = ez[idx];
            } else {
              s_ex_next[s_idx] = 0;
              s_ey_next[s_idx] = 0;
              s_ez_next[s_idx] = 0;
            }
          }
        }
      }

      shift_queue(ex_z);
      shift_queue(ey_z);
      int64_t const z_new = z + FD_PAD + 1;
      TIDE_DTYPE ex_val = 0;
      TIDE_DTYPE ey_val = 0;
      if (in_bounds_xy && z_new < nz) {
        int64_t const idx = shot_offset + z_new * plane_stride + y * nx + x;
        ex_val = ex[idx];
        ey_val = ey[idx];
      }
      ex_z[2 * FD_PAD] = ex_val;
      ey_z[2 * FD_PAD] = ey_val;
    }

    TIDE_DTYPE *tmp = s_ex_cur;
    s_ex_cur = s_ex_next;
    s_ex_next = tmp;
    tmp = s_ey_cur;
    s_ey_cur = s_ey_next;
    s_ey_next = tmp;
    tmp = s_ez_cur;
    s_ez_cur = s_ez_next;
    s_ez_next = tmp;

    __syncthreads();
  }
}

__global__ __launch_bounds__(256) void forward_kernel_h_3d_naive(
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
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

  bool const pml_y = y < pml_y0h || y >= pml_y1h;
  bool const pml_x = x < pml_x0h || x >= pml_x1h;

  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z = z < pml_z0h || z >= pml_z1h;
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


__global__ __launch_bounds__(256) void forward_kernel_e_3d(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
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
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const base_x = (int64_t)blockIdx.x * (int64_t)blockDim.x;
  int64_t const base_y = (int64_t)blockIdx.y * (int64_t)blockDim.y;
  int64_t const x = base_x + (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = base_y + (int64_t)threadIdx.y + FD_PAD;

  bool const in_domain = (x < nx - FD_PAD + 1) && (y < ny - FD_PAD + 1);
  bool const in_bounds_xy = (x < nx) && (y < ny);

  int64_t const plane_stride = ny * nx;
  int64_t const shot_offset = shot_idx * shot_numel;

  int const tile_w = (int)blockDim.x + 2 * FD_PAD;
  int const tile_h = (int)blockDim.y + 2 * FD_PAD;
  int const tile_size = tile_w * tile_h;
  bool const full_tile_xy =
      (base_x + tile_w <= nx) && (base_y + tile_h <= ny);

  extern __shared__ TIDE_DTYPE s_mem[];
  TIDE_DTYPE *const s_hx0 = s_mem;
  TIDE_DTYPE *const s_hy0 = s_hx0 + tile_size;
  TIDE_DTYPE *const s_hz0 = s_hy0 + tile_size;
  TIDE_DTYPE *const s_hx1 = s_hz0 + tile_size;
  TIDE_DTYPE *const s_hy1 = s_hx1 + tile_size;
  TIDE_DTYPE *const s_hz1 = s_hy1 + tile_size;

  TIDE_DTYPE *s_hx_cur = s_hx0;
  TIDE_DTYPE *s_hy_cur = s_hy0;
  TIDE_DTYPE *s_hz_cur = s_hz0;
  TIDE_DTYPE *s_hx_next = s_hx1;
  TIDE_DTYPE *s_hy_next = s_hy1;
  TIDE_DTYPE *s_hz_next = s_hz1;

  int const lx = (int)threadIdx.x + FD_PAD;
  int const ly = (int)threadIdx.y + FD_PAD;
  int const lidx = ly * tile_w + lx;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  TIDE_DTYPE hx_z[2 * FD_PAD + 1];
  TIDE_DTYPE hy_z[2 * FD_PAD + 1];

#pragma unroll
  for (int dz = -FD_PAD; dz <= FD_PAD; ++dz) {
    int64_t const z_idx = (int64_t)FD_PAD + (int64_t)dz;
    TIDE_DTYPE hx_val = 0;
    TIDE_DTYPE hy_val = 0;
    if (in_bounds_xy && z_idx >= 0 && z_idx < nz) {
      int64_t const idx = shot_offset + z_idx * plane_stride + y * nx + x;
      hx_val = hx[idx];
      hy_val = hy[idx];
    }
    hx_z[dz + FD_PAD] = hx_val;
    hy_z[dz + FD_PAD] = hy_val;
  }

  int64_t const z_end = nz - FD_PAD + 1;
  if (z_end > FD_PAD) {
    int64_t const z_offset = shot_offset + (int64_t)FD_PAD * plane_stride;
    if (full_tile_xy) {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int const s_row = sy * tile_w;
        int64_t const row_offset = (base_y + (int64_t)sy) * nx;
        int64_t const idx_base = z_offset + row_offset + base_x;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int const s_idx = s_row + sx;
          int64_t const idx = idx_base + sx;
          s_hx_cur[s_idx] = hx[idx];
          s_hy_cur[s_idx] = hy[idx];
          s_hz_cur[s_idx] = hz[idx];
        }
      }
    } else {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int64_t const gy = base_y + (int64_t)sy;
        bool const y_ok = gy < ny;
        int const s_row = sy * tile_w;
        int64_t const row_offset = gy * nx;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int64_t const gx = base_x + (int64_t)sx;
          int const s_idx = s_row + sx;
          if (y_ok && gx < nx) {
            int64_t const idx = z_offset + row_offset + gx;
            s_hx_cur[s_idx] = hx[idx];
            s_hy_cur[s_idx] = hy[idx];
            s_hz_cur[s_idx] = hz[idx];
          } else {
            s_hx_cur[s_idx] = 0;
            s_hy_cur[s_idx] = 0;
            s_hz_cur[s_idx] = 0;
          }
        }
      }
    }
  }

  __syncthreads();

  for (int64_t z = FD_PAD; z < z_end; ++z) {
    if (in_domain) {
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      TIDE_DTYPE const ca_val = CA(0, 0, 0);
      TIDE_DTYPE const cb_val = CB(0, 0, 0);

      TIDE_DTYPE dHz_dy = diffy1_sh(s_hz_cur, lidx, tile_w, rdy);
      if (pml_y) {
        M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
        dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
      }
      TIDE_DTYPE dHy_dz = diffz1_reg(hy_z, rdz);
      if (pml_z) {
        M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
        dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
      }
      EX(0, 0, 0) = ca_val * EX(0, 0, 0) + cb_val * (dHz_dy - dHy_dz);

      TIDE_DTYPE dHx_dz = diffz1_reg(hx_z, rdz);
      if (pml_z) {
        M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
        dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
      }
      TIDE_DTYPE dHz_dx = diffx1_sh(s_hz_cur, lidx, tile_w, rdx);
      if (pml_x) {
        M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
        dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
      }
      EY(0, 0, 0) = ca_val * EY(0, 0, 0) + cb_val * (dHx_dz - dHz_dx);

      TIDE_DTYPE dHy_dx = diffx1_sh(s_hy_cur, lidx, tile_w, rdx);
      if (pml_x) {
        M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
        dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
      }
      TIDE_DTYPE dHx_dy = diffy1_sh(s_hx_cur, lidx, tile_w, rdy);
      if (pml_y) {
        M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
        dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
      }
      EZ(0, 0, 0) = ca_val * EZ(0, 0, 0) + cb_val * (dHy_dx - dHx_dy);
    }

    if (z + 1 < z_end) {
      int64_t const z_offset = shot_offset + (z + 1) * plane_stride;
      if (full_tile_xy) {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int const s_row = sy * tile_w;
          int64_t const row_offset = (base_y + (int64_t)sy) * nx;
          int64_t const idx_base = z_offset + row_offset + base_x;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int const s_idx = s_row + sx;
            int64_t const idx = idx_base + sx;
            s_hx_next[s_idx] = hx[idx];
            s_hy_next[s_idx] = hy[idx];
            s_hz_next[s_idx] = hz[idx];
          }
        }
      } else {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int64_t const gy = base_y + (int64_t)sy;
          bool const y_ok = gy < ny;
          int const s_row = sy * tile_w;
          int64_t const row_offset = gy * nx;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int64_t const gx = base_x + (int64_t)sx;
            int const s_idx = s_row + sx;
            if (y_ok && gx < nx) {
              int64_t const idx = z_offset + row_offset + gx;
              s_hx_next[s_idx] = hx[idx];
              s_hy_next[s_idx] = hy[idx];
              s_hz_next[s_idx] = hz[idx];
            } else {
              s_hx_next[s_idx] = 0;
              s_hy_next[s_idx] = 0;
              s_hz_next[s_idx] = 0;
            }
          }
        }
      }

      shift_queue(hx_z);
      shift_queue(hy_z);
      int64_t const z_new = z + FD_PAD + 1;
      TIDE_DTYPE hx_val = 0;
      TIDE_DTYPE hy_val = 0;
      if (in_bounds_xy && z_new < nz) {
        int64_t const idx = shot_offset + z_new * plane_stride + y * nx + x;
        hx_val = hx[idx];
        hy_val = hy[idx];
      }
      hx_z[2 * FD_PAD] = hx_val;
      hy_z[2 * FD_PAD] = hy_val;
    }

    TIDE_DTYPE *tmp = s_hx_cur;
    s_hx_cur = s_hx_next;
    s_hx_next = tmp;
    tmp = s_hy_cur;
    s_hy_cur = s_hy_next;
    s_hy_next = tmp;
    tmp = s_hz_cur;
    s_hz_cur = s_hz_next;
    s_hz_next = tmp;

    __syncthreads();
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_3d_with_storage_tile(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const ex_store,
    TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const ez_store,
    TIDE_DTYPE *__restrict const curlx_store,
    TIDE_DTYPE *__restrict const curly_store,
    TIDE_DTYPE *__restrict const curlz_store,
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
    bool const cb_batched,
    bool const store_e,
    bool const store_curl) {
  (void)azh;
  (void)bzh;
  (void)ayh;
  (void)byh;
  (void)axh;
  (void)bxh;
  (void)kzh;
  (void)kyh;
  (void)kxh;

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const base_x = (int64_t)blockIdx.x * (int64_t)blockDim.x;
  int64_t const base_y = (int64_t)blockIdx.y * (int64_t)blockDim.y;
  int64_t const x = base_x + (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = base_y + (int64_t)threadIdx.y + FD_PAD;

  bool const in_domain = (x < nx - FD_PAD + 1) && (y < ny - FD_PAD + 1);
  bool const in_bounds_xy = (x < nx) && (y < ny);

  int64_t const plane_stride = ny * nx;
  int64_t const shot_offset = shot_idx * shot_numel;

  int const tile_w = (int)blockDim.x + 2 * FD_PAD;
  int const tile_h = (int)blockDim.y + 2 * FD_PAD;
  int const tile_size = tile_w * tile_h;
  bool const full_tile_xy =
      (base_x + tile_w <= nx) && (base_y + tile_h <= ny);

  extern __shared__ TIDE_DTYPE s_mem[];
  TIDE_DTYPE *const s_hx0 = s_mem;
  TIDE_DTYPE *const s_hy0 = s_hx0 + tile_size;
  TIDE_DTYPE *const s_hz0 = s_hy0 + tile_size;
  TIDE_DTYPE *const s_hx1 = s_hz0 + tile_size;
  TIDE_DTYPE *const s_hy1 = s_hx1 + tile_size;
  TIDE_DTYPE *const s_hz1 = s_hy1 + tile_size;

  TIDE_DTYPE *s_hx_cur = s_hx0;
  TIDE_DTYPE *s_hy_cur = s_hy0;
  TIDE_DTYPE *s_hz_cur = s_hz0;
  TIDE_DTYPE *s_hx_next = s_hx1;
  TIDE_DTYPE *s_hy_next = s_hy1;
  TIDE_DTYPE *s_hz_next = s_hz1;

  int const lx = (int)threadIdx.x + FD_PAD;
  int const ly = (int)threadIdx.y + FD_PAD;
  int const lidx = ly * tile_w + lx;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  TIDE_DTYPE hx_z[2 * FD_PAD + 1];
  TIDE_DTYPE hy_z[2 * FD_PAD + 1];

#pragma unroll
  for (int dz = -FD_PAD; dz <= FD_PAD; ++dz) {
    int64_t const z_idx = (int64_t)FD_PAD + (int64_t)dz;
    TIDE_DTYPE hx_val = 0;
    TIDE_DTYPE hy_val = 0;
    if (in_bounds_xy && z_idx >= 0 && z_idx < nz) {
      int64_t const idx = shot_offset + z_idx * plane_stride + y * nx + x;
      hx_val = hx[idx];
      hy_val = hy[idx];
    }
    hx_z[dz + FD_PAD] = hx_val;
    hy_z[dz + FD_PAD] = hy_val;
  }

  int64_t const z_end = nz - FD_PAD + 1;
  if (z_end > FD_PAD) {
    int64_t const z_offset = shot_offset + (int64_t)FD_PAD * plane_stride;
    if (full_tile_xy) {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int const s_row = sy * tile_w;
        int64_t const row_offset = (base_y + (int64_t)sy) * nx;
        int64_t const idx_base = z_offset + row_offset + base_x;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int const s_idx = s_row + sx;
          int64_t const idx = idx_base + sx;
          s_hx_cur[s_idx] = hx[idx];
          s_hy_cur[s_idx] = hy[idx];
          s_hz_cur[s_idx] = hz[idx];
        }
      }
    } else {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int64_t const gy = base_y + (int64_t)sy;
        bool const y_ok = gy < ny;
        int const s_row = sy * tile_w;
        int64_t const row_offset = gy * nx;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int64_t const gx = base_x + (int64_t)sx;
          int const s_idx = s_row + sx;
          if (y_ok && gx < nx) {
            int64_t const idx = z_offset + row_offset + gx;
            s_hx_cur[s_idx] = hx[idx];
            s_hy_cur[s_idx] = hy[idx];
            s_hz_cur[s_idx] = hz[idx];
          } else {
            s_hx_cur[s_idx] = 0;
            s_hy_cur[s_idx] = 0;
            s_hz_cur[s_idx] = 0;
          }
        }
      }
    }
  }

  __syncthreads();

  for (int64_t z = FD_PAD; z < z_end; ++z) {
    if (in_domain) {
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      TIDE_DTYPE const ca_val = CA(0, 0, 0);
      TIDE_DTYPE const cb_val = CB(0, 0, 0);

      TIDE_DTYPE dHz_dy = diffy1_sh(s_hz_cur, lidx, tile_w, rdy);
      if (pml_y) {
        M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
        dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
      }
      TIDE_DTYPE dHy_dz = diffz1_reg(hy_z, rdz);
      if (pml_z) {
        M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
        dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
      }
      TIDE_DTYPE curl_x = dHz_dy - dHy_dz;

      TIDE_DTYPE dHx_dz = diffz1_reg(hx_z, rdz);
      if (pml_z) {
        M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
        dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
      }
      TIDE_DTYPE dHz_dx = diffx1_sh(s_hz_cur, lidx, tile_w, rdx);
      if (pml_x) {
        M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
        dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
      }
      TIDE_DTYPE curl_y = dHx_dz - dHz_dx;

      TIDE_DTYPE dHy_dx = diffx1_sh(s_hy_cur, lidx, tile_w, rdx);
      if (pml_x) {
        M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
        dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
      }
      TIDE_DTYPE dHx_dy = diffy1_sh(s_hx_cur, lidx, tile_w, rdy);
      if (pml_y) {
        M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
        dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
      }
      TIDE_DTYPE curl_z = dHy_dx - dHx_dy;

      int64_t const j = z * ny * nx + y * nx + x;
      int64_t const i = shot_offset + j;

      TIDE_DTYPE const ex_old = ex[i];
      TIDE_DTYPE const ey_old = ey[i];
      TIDE_DTYPE const ez_old = ez[i];

      if (store_e && ex_store != nullptr) ex_store[i] = ex_old;
      if (store_e && ey_store != nullptr) ey_store[i] = ey_old;
      if (store_e && ez_store != nullptr) ez_store[i] = ez_old;
      if (store_curl && curlx_store != nullptr) curlx_store[i] = curl_x;
      if (store_curl && curly_store != nullptr) curly_store[i] = curl_y;
      if (store_curl && curlz_store != nullptr) curlz_store[i] = curl_z;

      ex[i] = ca_val * ex_old + cb_val * curl_x;
      ey[i] = ca_val * ey_old + cb_val * curl_y;
      ez[i] = ca_val * ez_old + cb_val * curl_z;
    }

    if (z + 1 < z_end) {
      int64_t const z_offset = shot_offset + (z + 1) * plane_stride;
      if (full_tile_xy) {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int const s_row = sy * tile_w;
          int64_t const row_offset = (base_y + (int64_t)sy) * nx;
          int64_t const idx_base = z_offset + row_offset + base_x;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int const s_idx = s_row + sx;
            int64_t const idx = idx_base + sx;
            s_hx_next[s_idx] = hx[idx];
            s_hy_next[s_idx] = hy[idx];
            s_hz_next[s_idx] = hz[idx];
          }
        }
      } else {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int64_t const gy = base_y + (int64_t)sy;
          bool const y_ok = gy < ny;
          int const s_row = sy * tile_w;
          int64_t const row_offset = gy * nx;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int64_t const gx = base_x + (int64_t)sx;
            int const s_idx = s_row + sx;
            if (y_ok && gx < nx) {
              int64_t const idx = z_offset + row_offset + gx;
              s_hx_next[s_idx] = hx[idx];
              s_hy_next[s_idx] = hy[idx];
              s_hz_next[s_idx] = hz[idx];
            } else {
              s_hx_next[s_idx] = 0;
              s_hy_next[s_idx] = 0;
              s_hz_next[s_idx] = 0;
            }
          }
        }
      }

      shift_queue(hx_z);
      shift_queue(hy_z);
      int64_t const z_new = z + FD_PAD + 1;
      TIDE_DTYPE hx_val = 0;
      TIDE_DTYPE hy_val = 0;
      if (in_bounds_xy && z_new < nz) {
        int64_t const idx = shot_offset + z_new * plane_stride + y * nx + x;
        hx_val = hx[idx];
        hy_val = hy[idx];
      }
      hx_z[2 * FD_PAD] = hx_val;
      hy_z[2 * FD_PAD] = hy_val;
    }

    TIDE_DTYPE *tmp = s_hx_cur;
    s_hx_cur = s_hx_next;
    s_hx_next = tmp;
    tmp = s_hy_cur;
    s_hy_cur = s_hy_next;
    s_hy_next = tmp;
    tmp = s_hz_cur;
    s_hz_cur = s_hz_next;
    s_hz_next = tmp;

    __syncthreads();
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_3d_with_storage_tile_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    __nv_bfloat16 *__restrict const ex_store,
    __nv_bfloat16 *__restrict const ey_store,
    __nv_bfloat16 *__restrict const ez_store,
    __nv_bfloat16 *__restrict const curlx_store,
    __nv_bfloat16 *__restrict const curly_store,
    __nv_bfloat16 *__restrict const curlz_store,
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
    bool const cb_batched,
    bool const store_e,
    bool const store_curl) {
  (void)azh;
  (void)bzh;
  (void)ayh;
  (void)byh;
  (void)axh;
  (void)bxh;
  (void)kzh;
  (void)kyh;
  (void)kxh;

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const base_x = (int64_t)blockIdx.x * (int64_t)blockDim.x;
  int64_t const base_y = (int64_t)blockIdx.y * (int64_t)blockDim.y;
  int64_t const x = base_x + (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = base_y + (int64_t)threadIdx.y + FD_PAD;

  bool const in_domain = (x < nx - FD_PAD + 1) && (y < ny - FD_PAD + 1);
  bool const in_bounds_xy = (x < nx) && (y < ny);

  int64_t const plane_stride = ny * nx;
  int64_t const shot_offset = shot_idx * shot_numel;

  int const tile_w = (int)blockDim.x + 2 * FD_PAD;
  int const tile_h = (int)blockDim.y + 2 * FD_PAD;
  int const tile_size = tile_w * tile_h;
  bool const full_tile_xy =
      (base_x + tile_w <= nx) && (base_y + tile_h <= ny);

  extern __shared__ TIDE_DTYPE s_mem[];
  TIDE_DTYPE *const s_hx0 = s_mem;
  TIDE_DTYPE *const s_hy0 = s_hx0 + tile_size;
  TIDE_DTYPE *const s_hz0 = s_hy0 + tile_size;
  TIDE_DTYPE *const s_hx1 = s_hz0 + tile_size;
  TIDE_DTYPE *const s_hy1 = s_hx1 + tile_size;
  TIDE_DTYPE *const s_hz1 = s_hy1 + tile_size;

  TIDE_DTYPE *s_hx_cur = s_hx0;
  TIDE_DTYPE *s_hy_cur = s_hy0;
  TIDE_DTYPE *s_hz_cur = s_hz0;
  TIDE_DTYPE *s_hx_next = s_hx1;
  TIDE_DTYPE *s_hy_next = s_hy1;
  TIDE_DTYPE *s_hz_next = s_hz1;

  int const lx = (int)threadIdx.x + FD_PAD;
  int const ly = (int)threadIdx.y + FD_PAD;
  int const lidx = ly * tile_w + lx;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  TIDE_DTYPE hx_z[2 * FD_PAD + 1];
  TIDE_DTYPE hy_z[2 * FD_PAD + 1];

#pragma unroll
  for (int dz = -FD_PAD; dz <= FD_PAD; ++dz) {
    int64_t const z_idx = (int64_t)FD_PAD + (int64_t)dz;
    TIDE_DTYPE hx_val = 0;
    TIDE_DTYPE hy_val = 0;
    if (in_bounds_xy && z_idx >= 0 && z_idx < nz) {
      int64_t const idx = shot_offset + z_idx * plane_stride + y * nx + x;
      hx_val = hx[idx];
      hy_val = hy[idx];
    }
    hx_z[dz + FD_PAD] = hx_val;
    hy_z[dz + FD_PAD] = hy_val;
  }

  int64_t const z_end = nz - FD_PAD + 1;
  if (z_end > FD_PAD) {
    int64_t const z_offset = shot_offset + (int64_t)FD_PAD * plane_stride;
    if (full_tile_xy) {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int const s_row = sy * tile_w;
        int64_t const row_offset = (base_y + (int64_t)sy) * nx;
        int64_t const idx_base = z_offset + row_offset + base_x;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int const s_idx = s_row + sx;
          int64_t const idx = idx_base + sx;
          s_hx_cur[s_idx] = hx[idx];
          s_hy_cur[s_idx] = hy[idx];
          s_hz_cur[s_idx] = hz[idx];
        }
      }
    } else {
      for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
        int64_t const gy = base_y + (int64_t)sy;
        bool const y_ok = gy < ny;
        int const s_row = sy * tile_w;
        int64_t const row_offset = gy * nx;
        for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
          int64_t const gx = base_x + (int64_t)sx;
          int const s_idx = s_row + sx;
          if (y_ok && gx < nx) {
            int64_t const idx = z_offset + row_offset + gx;
            s_hx_cur[s_idx] = hx[idx];
            s_hy_cur[s_idx] = hy[idx];
            s_hz_cur[s_idx] = hz[idx];
          } else {
            s_hx_cur[s_idx] = 0;
            s_hy_cur[s_idx] = 0;
            s_hz_cur[s_idx] = 0;
          }
        }
      }
    }
  }

  __syncthreads();

  for (int64_t z = FD_PAD; z < z_end; ++z) {
    if (in_domain) {
      bool const pml_z = z < pml_z0 || z >= pml_z1;
      TIDE_DTYPE const ca_val = CA(0, 0, 0);
      TIDE_DTYPE const cb_val = CB(0, 0, 0);

      TIDE_DTYPE dHz_dy = diffy1_sh(s_hz_cur, lidx, tile_w, rdy);
      if (pml_y) {
        M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
        dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
      }
      TIDE_DTYPE dHy_dz = diffz1_reg(hy_z, rdz);
      if (pml_z) {
        M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
        dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
      }
      TIDE_DTYPE curl_x = dHz_dy - dHy_dz;

      TIDE_DTYPE dHx_dz = diffz1_reg(hx_z, rdz);
      if (pml_z) {
        M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
        dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
      }
      TIDE_DTYPE dHz_dx = diffx1_sh(s_hz_cur, lidx, tile_w, rdx);
      if (pml_x) {
        M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
        dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
      }
      TIDE_DTYPE curl_y = dHx_dz - dHz_dx;

      TIDE_DTYPE dHy_dx = diffx1_sh(s_hy_cur, lidx, tile_w, rdx);
      if (pml_x) {
        M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
        dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
      }
      TIDE_DTYPE dHx_dy = diffy1_sh(s_hx_cur, lidx, tile_w, rdy);
      if (pml_y) {
        M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
        dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
      }
      TIDE_DTYPE curl_z = dHy_dx - dHx_dy;

      int64_t const j = z * ny * nx + y * nx + x;
      int64_t const i = shot_offset + j;

      TIDE_DTYPE const ex_old = ex[i];
      TIDE_DTYPE const ey_old = ey[i];
      TIDE_DTYPE const ez_old = ez[i];

      if (store_e && ex_store != nullptr) {
        ex_store[i] = __float2bfloat16((float)ex_old);
      }
      if (store_e && ey_store != nullptr) {
        ey_store[i] = __float2bfloat16((float)ey_old);
      }
      if (store_e && ez_store != nullptr) {
        ez_store[i] = __float2bfloat16((float)ez_old);
      }
      if (store_curl && curlx_store != nullptr) {
        curlx_store[i] = __float2bfloat16((float)curl_x);
      }
      if (store_curl && curly_store != nullptr) {
        curly_store[i] = __float2bfloat16((float)curl_y);
      }
      if (store_curl && curlz_store != nullptr) {
        curlz_store[i] = __float2bfloat16((float)curl_z);
      }

      ex[i] = ca_val * ex_old + cb_val * curl_x;
      ey[i] = ca_val * ey_old + cb_val * curl_y;
      ez[i] = ca_val * ez_old + cb_val * curl_z;
    }

    if (z + 1 < z_end) {
      int64_t const z_offset = shot_offset + (z + 1) * plane_stride;
      if (full_tile_xy) {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int const s_row = sy * tile_w;
          int64_t const row_offset = (base_y + (int64_t)sy) * nx;
          int64_t const idx_base = z_offset + row_offset + base_x;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int const s_idx = s_row + sx;
            int64_t const idx = idx_base + sx;
            s_hx_next[s_idx] = hx[idx];
            s_hy_next[s_idx] = hy[idx];
            s_hz_next[s_idx] = hz[idx];
          }
        }
      } else {
        for (int sy = (int)threadIdx.y; sy < tile_h; sy += (int)blockDim.y) {
          int64_t const gy = base_y + (int64_t)sy;
          bool const y_ok = gy < ny;
          int const s_row = sy * tile_w;
          int64_t const row_offset = gy * nx;
          for (int sx = (int)threadIdx.x; sx < tile_w; sx += (int)blockDim.x) {
            int64_t const gx = base_x + (int64_t)sx;
            int const s_idx = s_row + sx;
            if (y_ok && gx < nx) {
              int64_t const idx = z_offset + row_offset + gx;
              s_hx_next[s_idx] = hx[idx];
              s_hy_next[s_idx] = hy[idx];
              s_hz_next[s_idx] = hz[idx];
            } else {
              s_hx_next[s_idx] = 0;
              s_hy_next[s_idx] = 0;
              s_hz_next[s_idx] = 0;
            }
          }
        }
      }

      shift_queue(hx_z);
      shift_queue(hy_z);
      int64_t const z_new = z + FD_PAD + 1;
      TIDE_DTYPE hx_val = 0;
      TIDE_DTYPE hy_val = 0;
      if (in_bounds_xy && z_new < nz) {
        int64_t const idx = shot_offset + z_new * plane_stride + y * nx + x;
        hx_val = hx[idx];
        hy_val = hy[idx];
      }
      hx_z[2 * FD_PAD] = hx_val;
      hy_z[2 * FD_PAD] = hy_val;
    }

    TIDE_DTYPE *tmp = s_hx_cur;
    s_hx_cur = s_hx_next;
    s_hx_next = tmp;
    tmp = s_hy_cur;
    s_hy_cur = s_hy_next;
    s_hy_next = tmp;
    tmp = s_hz_cur;
    s_hz_cur = s_hz_next;
    s_hz_next = tmp;

    __syncthreads();
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_3d_naive(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
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
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z = z < pml_z0 || z >= pml_z1;
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

__global__ void forward_kernel_e_3d_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const ex_store,
    TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const ez_store,
    TIDE_DTYPE *__restrict const curlx_store,
    TIDE_DTYPE *__restrict const curly_store,
    TIDE_DTYPE *__restrict const curlz_store,
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
    bool const cb_batched,
    bool const store_e,
    bool const store_curl) {
  (void)azh;
  (void)bzh;
  (void)ayh;
  (void)byh;
  (void)axh;
  (void)bxh;
  (void)kzh;
  (void)kyh;
  (void)kxh;

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z = z < pml_z0 || z >= pml_z1;
    int64_t const j = z * ny * nx + y * nx + x;
    int64_t const i = shot_idx * shot_numel + j;

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
    TIDE_DTYPE curl_x = dHz_dy - dHy_dz;

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
    TIDE_DTYPE curl_y = dHx_dz - dHz_dx;

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
    TIDE_DTYPE curl_z = dHy_dx - dHx_dy;

    TIDE_DTYPE ex_old = ex[i];
    TIDE_DTYPE ey_old = ey[i];
    TIDE_DTYPE ez_old = ez[i];

    if (store_e && ex_store != nullptr) ex_store[i] = ex_old;
    if (store_e && ey_store != nullptr) ey_store[i] = ey_old;
    if (store_e && ez_store != nullptr) ez_store[i] = ez_old;
    if (store_curl && curlx_store != nullptr) curlx_store[i] = curl_x;
    if (store_curl && curly_store != nullptr) curly_store[i] = curl_y;
    if (store_curl && curlz_store != nullptr) curlz_store[i] = curl_z;

    ex[i] = ca_val * ex_old + cb_val * curl_x;
    ey[i] = ca_val * ey_old + cb_val * curl_y;
    ez[i] = ca_val * ez_old + cb_val * curl_z;
  }
}

__global__ void forward_kernel_e_3d_with_storage_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    __nv_bfloat16 *__restrict const ex_store,
    __nv_bfloat16 *__restrict const ey_store,
    __nv_bfloat16 *__restrict const ez_store,
    __nv_bfloat16 *__restrict const curlx_store,
    __nv_bfloat16 *__restrict const curly_store,
    __nv_bfloat16 *__restrict const curlz_store,
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
    bool const cb_batched,
    bool const store_e,
    bool const store_curl) {
  (void)azh;
  (void)bzh;
  (void)ayh;
  (void)byh;
  (void)axh;
  (void)bxh;
  (void)kzh;
  (void)kyh;
  (void)kxh;

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z = z < pml_z0 || z >= pml_z1;
    int64_t const j = z * ny * nx + y * nx + x;
    int64_t const i = shot_idx * shot_numel + j;

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
    TIDE_DTYPE curl_x = dHz_dy - dHy_dz;

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
    TIDE_DTYPE curl_y = dHx_dz - dHz_dx;

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
    TIDE_DTYPE curl_z = dHy_dx - dHx_dy;

    TIDE_DTYPE ex_old = ex[i];
    TIDE_DTYPE ey_old = ey[i];
    TIDE_DTYPE ez_old = ez[i];

    if (store_e && ex_store != nullptr) {
      ex_store[i] = __float2bfloat16((float)ex_old);
    }
    if (store_e && ey_store != nullptr) {
      ey_store[i] = __float2bfloat16((float)ey_old);
    }
    if (store_e && ez_store != nullptr) {
      ez_store[i] = __float2bfloat16((float)ez_old);
    }
    if (store_curl && curlx_store != nullptr) {
      curlx_store[i] = __float2bfloat16((float)curl_x);
    }
    if (store_curl && curly_store != nullptr) {
      curly_store[i] = __float2bfloat16((float)curl_y);
    }
    if (store_curl && curlz_store != nullptr) {
      curlz_store[i] = __float2bfloat16((float)curl_z);
    }

    ex[i] = ca_val * ex_old + cb_val * curl_x;
    ey[i] = ca_val * ey_old + cb_val * curl_y;
    ez[i] = ca_val * ez_old + cb_val * curl_z;
  }
}

__global__ void backward_kernel_lambda_h_3d(
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
    bool const cb_batched) {
  (void)az;
  (void)bz;
  (void)ay;
  (void)by;
  (void)ax;
  (void)bx;
  (void)kz;
  (void)ky;
  (void)kx;

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

  bool const pml_y = y < pml_y0h || y >= pml_y1h;
  bool const pml_x = x < pml_x0h || x >= pml_x1h;

  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z = z < pml_z0h || z >= pml_z1h;
    TIDE_DTYPE const cb_val = CB(0, 0, 0);

    if (z < nz - FD_PAD && y < ny - FD_PAD) {
      TIDE_DTYPE d_lambda_ey_dz = DIFFZH1(LEY);
      if (pml_z) {
        M_L_EY_Z(0, 0, 0) = bzh[z] * M_L_EY_Z(0, 0, 0) + azh[z] * d_lambda_ey_dz;
        d_lambda_ey_dz = d_lambda_ey_dz / kzh[z] + M_L_EY_Z(0, 0, 0);
      }
      TIDE_DTYPE d_lambda_ez_dy = DIFFYH1(LEZ);
      if (pml_y) {
        M_L_EZ_Y(0, 0, 0) = byh[y] * M_L_EZ_Y(0, 0, 0) + ayh[y] * d_lambda_ez_dy;
        d_lambda_ez_dy = d_lambda_ez_dy / kyh[y] + M_L_EZ_Y(0, 0, 0);
      }
      LHX(0, 0, 0) += cb_val * (d_lambda_ey_dz - d_lambda_ez_dy);
    }

    if (z < nz - FD_PAD && x < nx - FD_PAD) {
      TIDE_DTYPE d_lambda_ez_dx = DIFFXH1(LEZ);
      if (pml_x) {
        M_L_EZ_X(0, 0, 0) = bxh[x] * M_L_EZ_X(0, 0, 0) + axh[x] * d_lambda_ez_dx;
        d_lambda_ez_dx = d_lambda_ez_dx / kxh[x] + M_L_EZ_X(0, 0, 0);
      }
      TIDE_DTYPE d_lambda_ex_dz = DIFFZH1(LEX);
      if (pml_z) {
        M_L_EX_Z(0, 0, 0) = bzh[z] * M_L_EX_Z(0, 0, 0) + azh[z] * d_lambda_ex_dz;
        d_lambda_ex_dz = d_lambda_ex_dz / kzh[z] + M_L_EX_Z(0, 0, 0);
      }
      LHY(0, 0, 0) += cb_val * (d_lambda_ez_dx - d_lambda_ex_dz);
    }

    if (y < ny - FD_PAD && x < nx - FD_PAD) {
      TIDE_DTYPE d_lambda_ex_dy = DIFFYH1(LEX);
      if (pml_y) {
        M_L_EX_Y(0, 0, 0) = byh[y] * M_L_EX_Y(0, 0, 0) + ayh[y] * d_lambda_ex_dy;
        d_lambda_ex_dy = d_lambda_ex_dy / kyh[y] + M_L_EX_Y(0, 0, 0);
      }
      TIDE_DTYPE d_lambda_ey_dx = DIFFXH1(LEY);
      if (pml_x) {
        M_L_EY_X(0, 0, 0) = bxh[x] * M_L_EY_X(0, 0, 0) + axh[x] * d_lambda_ey_dx;
        d_lambda_ey_dx = d_lambda_ey_dx / kxh[x] + M_L_EY_X(0, 0, 0);
      }
      LHZ(0, 0, 0) += cb_val * (d_lambda_ex_dy - d_lambda_ey_dx);
    }
  }
}

__global__ void backward_kernel_lambda_e_3d_with_grad(
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
    TIDE_DTYPE const *__restrict const ex_store,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const ez_store,
    TIDE_DTYPE const *__restrict const curlx_store,
    TIDE_DTYPE const *__restrict const curly_store,
    TIDE_DTYPE const *__restrict const curlz_store,
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
    bool const cq_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    int64_t const step_ratio_val) {
  (void)azh;
  (void)bzh;
  (void)ayh;
  (void)byh;
  (void)axh;
  (void)bxh;
  (void)kzh;
  (void)kyh;
  (void)kxh;

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z = z < pml_z0 || z >= pml_z1;
    int64_t const j = z * ny * nx + y * nx + x;
    int64_t const i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_val = CA(0, 0, 0);
    TIDE_DTYPE const cq_val = CQ(0, 0, 0);

    TIDE_DTYPE d_lambda_hz_dy = DIFFY1(LHZ);
    if (pml_y) {
      M_L_HZ_Y(0, 0, 0) = by[y] * M_L_HZ_Y(0, 0, 0) + ay[y] * d_lambda_hz_dy;
      d_lambda_hz_dy = d_lambda_hz_dy / ky[y] + M_L_HZ_Y(0, 0, 0);
    }
    TIDE_DTYPE d_lambda_hy_dz = DIFFZ1(LHY);
    if (pml_z) {
      M_L_HY_Z(0, 0, 0) = bz[z] * M_L_HY_Z(0, 0, 0) + az[z] * d_lambda_hy_dz;
      d_lambda_hy_dz = d_lambda_hy_dz / kz[z] + M_L_HY_Z(0, 0, 0);
    }
    TIDE_DTYPE curl_x = d_lambda_hz_dy - d_lambda_hy_dz;

    TIDE_DTYPE d_lambda_hx_dz = DIFFZ1(LHX);
    if (pml_z) {
      M_L_HX_Z(0, 0, 0) = bz[z] * M_L_HX_Z(0, 0, 0) + az[z] * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / kz[z] + M_L_HX_Z(0, 0, 0);
    }
    TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LHZ);
    if (pml_x) {
      M_L_HZ_X(0, 0, 0) = bx[x] * M_L_HZ_X(0, 0, 0) + ax[x] * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / kx[x] + M_L_HZ_X(0, 0, 0);
    }
    TIDE_DTYPE curl_y = d_lambda_hx_dz - d_lambda_hz_dx;

    TIDE_DTYPE d_lambda_hy_dx = DIFFX1(LHY);
    if (pml_x) {
      M_L_HY_X(0, 0, 0) = bx[x] * M_L_HY_X(0, 0, 0) + ax[x] * d_lambda_hy_dx;
      d_lambda_hy_dx = d_lambda_hy_dx / kx[x] + M_L_HY_X(0, 0, 0);
    }
    TIDE_DTYPE d_lambda_hx_dy = DIFFY1(LHX);
    if (pml_y) {
      M_L_HX_Y(0, 0, 0) = by[y] * M_L_HX_Y(0, 0, 0) + ay[y] * d_lambda_hx_dy;
      d_lambda_hx_dy = d_lambda_hx_dy / ky[y] + M_L_HX_Y(0, 0, 0);
    }
    TIDE_DTYPE curl_z = d_lambda_hy_dx - d_lambda_hx_dy;

    TIDE_DTYPE lambda_ex_curr = lambda_ex[i];
    TIDE_DTYPE lambda_ey_curr = lambda_ey[i];
    TIDE_DTYPE lambda_ez_curr = lambda_ez[i];

    lambda_ex[i] = ca_val * lambda_ex_curr + cq_val * curl_x;
    lambda_ey[i] = ca_val * lambda_ey_curr + cq_val * curl_y;
    lambda_ez[i] = ca_val * lambda_ez_curr + cq_val * curl_z;

    if (!pml_z && !pml_y && !pml_x) {
      if (ca_requires_grad && grad_ca_shot != nullptr) {
        if (ex_store != nullptr) grad_ca_shot[i] += lambda_ex_curr * ex_store[i] * (TIDE_DTYPE)step_ratio_val;
        if (ey_store != nullptr) grad_ca_shot[i] += lambda_ey_curr * ey_store[i] * (TIDE_DTYPE)step_ratio_val;
        if (ez_store != nullptr) grad_ca_shot[i] += lambda_ez_curr * ez_store[i] * (TIDE_DTYPE)step_ratio_val;
      }
      if (cb_requires_grad && grad_cb_shot != nullptr) {
        if (curlx_store != nullptr) grad_cb_shot[i] += lambda_ex_curr * curlx_store[i] * (TIDE_DTYPE)step_ratio_val;
        if (curly_store != nullptr) grad_cb_shot[i] += lambda_ey_curr * curly_store[i] * (TIDE_DTYPE)step_ratio_val;
        if (curlz_store != nullptr) grad_cb_shot[i] += lambda_ez_curr * curlz_store[i] * (TIDE_DTYPE)step_ratio_val;
      }
    }
  }
}

__global__ void backward_kernel_lambda_e_3d_with_grad_bf16(
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
    __nv_bfloat16 const *__restrict const ex_store,
    __nv_bfloat16 const *__restrict const ey_store,
    __nv_bfloat16 const *__restrict const ez_store,
    __nv_bfloat16 const *__restrict const curlx_store,
    __nv_bfloat16 const *__restrict const curly_store,
    __nv_bfloat16 const *__restrict const curlz_store,
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
    bool const cq_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    int64_t const step_ratio_val) {
  (void)azh;
  (void)bzh;
  (void)ayh;
  (void)byh;
  (void)axh;
  (void)bxh;
  (void)kzh;
  (void)kyh;
  (void)kxh;

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;

  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z = z < pml_z0 || z >= pml_z1;
    int64_t const j = z * ny * nx + y * nx + x;
    int64_t const i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_val = CA(0, 0, 0);
    TIDE_DTYPE const cq_val = CQ(0, 0, 0);

    TIDE_DTYPE d_lambda_hz_dy = DIFFY1(LHZ);
    if (pml_y) {
      M_L_HZ_Y(0, 0, 0) = by[y] * M_L_HZ_Y(0, 0, 0) + ay[y] * d_lambda_hz_dy;
      d_lambda_hz_dy = d_lambda_hz_dy / ky[y] + M_L_HZ_Y(0, 0, 0);
    }
    TIDE_DTYPE d_lambda_hy_dz = DIFFZ1(LHY);
    if (pml_z) {
      M_L_HY_Z(0, 0, 0) = bz[z] * M_L_HY_Z(0, 0, 0) + az[z] * d_lambda_hy_dz;
      d_lambda_hy_dz = d_lambda_hy_dz / kz[z] + M_L_HY_Z(0, 0, 0);
    }
    TIDE_DTYPE curl_x = d_lambda_hz_dy - d_lambda_hy_dz;

    TIDE_DTYPE d_lambda_hx_dz = DIFFZ1(LHX);
    if (pml_z) {
      M_L_HX_Z(0, 0, 0) = bz[z] * M_L_HX_Z(0, 0, 0) + az[z] * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / kz[z] + M_L_HX_Z(0, 0, 0);
    }
    TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LHZ);
    if (pml_x) {
      M_L_HZ_X(0, 0, 0) = bx[x] * M_L_HZ_X(0, 0, 0) + ax[x] * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / kx[x] + M_L_HZ_X(0, 0, 0);
    }
    TIDE_DTYPE curl_y = d_lambda_hx_dz - d_lambda_hz_dx;

    TIDE_DTYPE d_lambda_hy_dx = DIFFX1(LHY);
    if (pml_x) {
      M_L_HY_X(0, 0, 0) = bx[x] * M_L_HY_X(0, 0, 0) + ax[x] * d_lambda_hy_dx;
      d_lambda_hy_dx = d_lambda_hy_dx / kx[x] + M_L_HY_X(0, 0, 0);
    }
    TIDE_DTYPE d_lambda_hx_dy = DIFFY1(LHX);
    if (pml_y) {
      M_L_HX_Y(0, 0, 0) = by[y] * M_L_HX_Y(0, 0, 0) + ay[y] * d_lambda_hx_dy;
      d_lambda_hx_dy = d_lambda_hx_dy / ky[y] + M_L_HX_Y(0, 0, 0);
    }
    TIDE_DTYPE curl_z = d_lambda_hy_dx - d_lambda_hx_dy;

    TIDE_DTYPE lambda_ex_curr = lambda_ex[i];
    TIDE_DTYPE lambda_ey_curr = lambda_ey[i];
    TIDE_DTYPE lambda_ez_curr = lambda_ez[i];

    lambda_ex[i] = ca_val * lambda_ex_curr + cq_val * curl_x;
    lambda_ey[i] = ca_val * lambda_ey_curr + cq_val * curl_y;
    lambda_ez[i] = ca_val * lambda_ez_curr + cq_val * curl_z;

    if (!pml_z && !pml_y && !pml_x) {
      if (ca_requires_grad && grad_ca_shot != nullptr) {
        if (ex_store != nullptr) grad_ca_shot[i] += lambda_ex_curr * (TIDE_DTYPE)__bfloat162float(ex_store[i]) * (TIDE_DTYPE)step_ratio_val;
        if (ey_store != nullptr) grad_ca_shot[i] += lambda_ey_curr * (TIDE_DTYPE)__bfloat162float(ey_store[i]) * (TIDE_DTYPE)step_ratio_val;
        if (ez_store != nullptr) grad_ca_shot[i] += lambda_ez_curr * (TIDE_DTYPE)__bfloat162float(ez_store[i]) * (TIDE_DTYPE)step_ratio_val;
      }
      if (cb_requires_grad && grad_cb_shot != nullptr) {
        if (curlx_store != nullptr) grad_cb_shot[i] += lambda_ex_curr * (TIDE_DTYPE)__bfloat162float(curlx_store[i]) * (TIDE_DTYPE)step_ratio_val;
        if (curly_store != nullptr) grad_cb_shot[i] += lambda_ey_curr * (TIDE_DTYPE)__bfloat162float(curly_store[i]) * (TIDE_DTYPE)step_ratio_val;
        if (curlz_store != nullptr) grad_cb_shot[i] += lambda_ez_curr * (TIDE_DTYPE)__bfloat162float(curlz_store[i]) * (TIDE_DTYPE)step_ratio_val;
      }
    }
  }
}

__global__ void combine_grad_3d(
    TIDE_DTYPE *__restrict const grad,
    TIDE_DTYPE const *__restrict const grad_shot,
    int64_t const shot_numel,
    int64_t const n_shots) {
  int64_t const idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (idx >= shot_numel) return;
  TIDE_DTYPE sum = 0;
#pragma unroll 4
  for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    sum += grad_shot[shot_idx * shot_numel + idx];
  }
  grad[idx] += sum;
}

__global__ void convert_grad_ca_cb_to_eps_sigma_3d(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const grad_ca,
    TIDE_DTYPE const *__restrict const grad_cb,
    TIDE_DTYPE const *__restrict const grad_ca_shot,
    TIDE_DTYPE const *__restrict const grad_cb_shot,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE const dt,
    int64_t const shot_numel,
    int64_t const n_shots,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h) {
  int64_t const idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  bool const has_shot = ca_batched_h || cb_batched_h;
  int64_t const total = has_shot ? (n_shots * shot_numel) : shot_numel;
  if (idx >= total) return;

  int64_t const shot_idx = has_shot ? (idx / shot_numel) : 0;
  int64_t const j = has_shot ? (idx - shot_idx * shot_numel) : idx;
  int64_t const out_idx = has_shot ? idx : j;
  int64_t const ca_idx = ca_batched_h ? idx : j;
  int64_t const cb_idx = cb_batched_h ? idx : j;

  TIDE_DTYPE const ca_val = ca[ca_idx];
  TIDE_DTYPE const cb_val = cb[cb_idx];
  TIDE_DTYPE const cb_sq = cb_val * cb_val;
  TIDE_DTYPE const inv_dt = (TIDE_DTYPE)1 / dt;

  TIDE_DTYPE grad_ca_val = 0;
  if (ca_requires_grad) {
    grad_ca_val = ca_batched_h ? grad_ca_shot[idx] : grad_ca[j];
  }
  TIDE_DTYPE grad_cb_val = 0;
  if (cb_requires_grad) {
    grad_cb_val = cb_batched_h ? grad_cb_shot[idx] : grad_cb[j];
  }

  TIDE_DTYPE const dca_de = ((TIDE_DTYPE)1 - ca_val) * cb_val * inv_dt;
  TIDE_DTYPE const dcb_de = -cb_sq * inv_dt;
  TIDE_DTYPE const dca_ds = -((TIDE_DTYPE)0.5) * ((TIDE_DTYPE)1 + ca_val) * cb_val;
  TIDE_DTYPE const dcb_ds = -((TIDE_DTYPE)0.5) * cb_sq;

  if (grad_eps != nullptr) {
    TIDE_DTYPE const grad_e = grad_ca_val * dca_de + grad_cb_val * dcb_de;
    grad_eps[out_idx] = grad_e * EP0;
  }
  if (grad_sigma != nullptr) {
    grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
  }
}

/*
 * Fused H+E forward kernel using cooperative groups for grid-level synchronization.
 * This reduces kernel launch overhead by combining H and E updates in a single kernel.
 *
 * Benefits:
 * - Eliminates kernel launch overhead between H and E updates
 * - Better GPU utilization for small grids
 * - Maintains correct gradient storage for backward pass
 *
 * Requirements:
 * - GPU must support cooperative launch (compute capability >= 6.0)
 * - Grid size must fit in available SMs for cooperative launch
 */
__global__ void __launch_bounds__(256) forward_kernel_fused_3d(
    // Material parameters
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
    // Field arrays (read/write)
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hy,
    TIDE_DTYPE *__restrict const hz,
    // PML memory variables for H update (E-derived)
    TIDE_DTYPE *__restrict const m_ey_z,
    TIDE_DTYPE *__restrict const m_ez_y,
    TIDE_DTYPE *__restrict const m_ez_x,
    TIDE_DTYPE *__restrict const m_ex_z,
    TIDE_DTYPE *__restrict const m_ex_y,
    TIDE_DTYPE *__restrict const m_ey_x,
    // PML memory variables for E update (H-derived)
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const m_hx_y,
    // Gradient storage arrays (optional)
    TIDE_DTYPE *__restrict const ex_store,
    TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const ez_store,
    TIDE_DTYPE *__restrict const curlx_store,
    TIDE_DTYPE *__restrict const curly_store,
    TIDE_DTYPE *__restrict const curlz_store,
    // PML coefficients
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
    // Grid parameters
    TIDE_DTYPE const rdz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const nz,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    // PML boundaries
    int64_t const pml_z0,
    int64_t const pml_z1,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    // Flags
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    bool const store_e,
    bool const store_curl) {

  cg::grid_group grid = cg::this_grid();

  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;

  if (shot_idx >= n_shots) return;
  if (x >= nx - FD_PAD + 1 || y >= ny - FD_PAD + 1) return;

  int64_t const pml_z0h = pml_z0;
  int64_t const pml_z1h = MAX(pml_z0, pml_z1 - 1);
  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

  bool const pml_y_h = y < pml_y0h || y >= pml_y1h;
  bool const pml_x_h = x < pml_x0h || x >= pml_x1h;
  bool const pml_y_e = y < pml_y0 || y >= pml_y1;
  bool const pml_x_e = x < pml_x0 || x >= pml_x1;

  // ========== Phase 1: Update H field ==========
  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z_h = z < pml_z0h || z >= pml_z1h;
    TIDE_DTYPE const cq_val = CQ(0, 0, 0);

    // Update Hx: dEy/dz - dEz/dy
    if (z < nz - FD_PAD && y < ny - FD_PAD) {
      TIDE_DTYPE dEy_dz = DIFFZH1(EY);
      if (pml_z_h) {
        M_EY_Z(0, 0, 0) = bzh[z] * M_EY_Z(0, 0, 0) + azh[z] * dEy_dz;
        dEy_dz = dEy_dz / kzh[z] + M_EY_Z(0, 0, 0);
      }
      TIDE_DTYPE dEz_dy = DIFFYH1(EZ);
      if (pml_y_h) {
        M_EZ_Y(0, 0, 0) = byh[y] * M_EZ_Y(0, 0, 0) + ayh[y] * dEz_dy;
        dEz_dy = dEz_dy / kyh[y] + M_EZ_Y(0, 0, 0);
      }
      HX(0, 0, 0) -= cq_val * (dEy_dz - dEz_dy);
    }

    // Update Hy: dEz/dx - dEx/dz
    if (z < nz - FD_PAD && x < nx - FD_PAD) {
      TIDE_DTYPE dEz_dx = DIFFXH1(EZ);
      if (pml_x_h) {
        M_EZ_X(0, 0, 0) = bxh[x] * M_EZ_X(0, 0, 0) + axh[x] * dEz_dx;
        dEz_dx = dEz_dx / kxh[x] + M_EZ_X(0, 0, 0);
      }
      TIDE_DTYPE dEx_dz = DIFFZH1(EX);
      if (pml_z_h) {
        M_EX_Z(0, 0, 0) = bzh[z] * M_EX_Z(0, 0, 0) + azh[z] * dEx_dz;
        dEx_dz = dEx_dz / kzh[z] + M_EX_Z(0, 0, 0);
      }
      HY(0, 0, 0) -= cq_val * (dEz_dx - dEx_dz);
    }

    // Update Hz: dEx/dy - dEy/dx
    if (y < ny - FD_PAD && x < nx - FD_PAD) {
      TIDE_DTYPE dEx_dy = DIFFYH1(EX);
      if (pml_y_h) {
        M_EX_Y(0, 0, 0) = byh[y] * M_EX_Y(0, 0, 0) + ayh[y] * dEx_dy;
        dEx_dy = dEx_dy / kyh[y] + M_EX_Y(0, 0, 0);
      }
      TIDE_DTYPE dEy_dx = DIFFXH1(EY);
      if (pml_x_h) {
        M_EY_X(0, 0, 0) = bxh[x] * M_EY_X(0, 0, 0) + axh[x] * dEy_dx;
        dEy_dx = dEy_dx / kxh[x] + M_EY_X(0, 0, 0);
      }
      HZ(0, 0, 0) -= cq_val * (dEx_dy - dEy_dx);
    }
  }

  // ========== Global Synchronization ==========
  // Wait for all blocks to complete H update before starting E update
  grid.sync();

  // ========== Phase 2: Update E field (with optional gradient storage) ==========
  for (int64_t z = FD_PAD; z < nz - FD_PAD + 1; ++z) {
    bool const pml_z_e = z < pml_z0 || z >= pml_z1;
    int64_t const j = z * ny * nx + y * nx + x;
    int64_t const i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca_val = CA(0, 0, 0);
    TIDE_DTYPE const cb_val = CB(0, 0, 0);

    // Compute curl(H)
    TIDE_DTYPE dHz_dy = DIFFY1(HZ);
    if (pml_y_e) {
      M_HZ_Y(0, 0, 0) = by[y] * M_HZ_Y(0, 0, 0) + ay[y] * dHz_dy;
      dHz_dy = dHz_dy / ky[y] + M_HZ_Y(0, 0, 0);
    }
    TIDE_DTYPE dHy_dz = DIFFZ1(HY);
    if (pml_z_e) {
      M_HY_Z(0, 0, 0) = bz[z] * M_HY_Z(0, 0, 0) + az[z] * dHy_dz;
      dHy_dz = dHy_dz / kz[z] + M_HY_Z(0, 0, 0);
    }
    TIDE_DTYPE curl_x = dHz_dy - dHy_dz;

    TIDE_DTYPE dHx_dz = DIFFZ1(HX);
    if (pml_z_e) {
      M_HX_Z(0, 0, 0) = bz[z] * M_HX_Z(0, 0, 0) + az[z] * dHx_dz;
      dHx_dz = dHx_dz / kz[z] + M_HX_Z(0, 0, 0);
    }
    TIDE_DTYPE dHz_dx = DIFFX1(HZ);
    if (pml_x_e) {
      M_HZ_X(0, 0, 0) = bx[x] * M_HZ_X(0, 0, 0) + ax[x] * dHz_dx;
      dHz_dx = dHz_dx / kx[x] + M_HZ_X(0, 0, 0);
    }
    TIDE_DTYPE curl_y = dHx_dz - dHz_dx;

    TIDE_DTYPE dHy_dx = DIFFX1(HY);
    if (pml_x_e) {
      M_HY_X(0, 0, 0) = bx[x] * M_HY_X(0, 0, 0) + ax[x] * dHy_dx;
      dHy_dx = dHy_dx / kx[x] + M_HY_X(0, 0, 0);
    }
    TIDE_DTYPE dHx_dy = DIFFY1(HX);
    if (pml_y_e) {
      M_HX_Y(0, 0, 0) = by[y] * M_HX_Y(0, 0, 0) + ay[y] * dHx_dy;
      dHx_dy = dHx_dy / ky[y] + M_HX_Y(0, 0, 0);
    }
    TIDE_DTYPE curl_z = dHy_dx - dHx_dy;

    // Load old E values for gradient storage
    TIDE_DTYPE ex_old = ex[i];
    TIDE_DTYPE ey_old = ey[i];
    TIDE_DTYPE ez_old = ez[i];

    // Store snapshots for gradient computation (before E update)
    if (store_e && ex_store != nullptr) ex_store[i] = ex_old;
    if (store_e && ey_store != nullptr) ey_store[i] = ey_old;
    if (store_e && ez_store != nullptr) ez_store[i] = ez_old;
    if (store_curl && curlx_store != nullptr) curlx_store[i] = curl_x;
    if (store_curl && curly_store != nullptr) curly_store[i] = curl_y;
    if (store_curl && curlz_store != nullptr) curlz_store[i] = curl_z;

    // Update E field: E_new = ca * E_old + cb * curl(H)
    ex[i] = ca_val * ex_old + cb_val * curl_x;
    ey[i] = ca_val * ey_old + cb_val * curl_y;
    ez[i] = ca_val * ez_old + cb_val * curl_z;
  }
}

}  // namespace

// Check if cooperative launch is supported (must be outside anonymous namespace)
static bool check_cooperative_launch_support(int device) {
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
  return supportsCoopLaunch != 0;
}

// Block configuration for optimal performance
// Default: 32x8=256 threads - optimal for most grid sizes
// Environment variable TIDE_BLOCK_X and TIDE_BLOCK_Y can override for tuning
static void get_block_xy(int *bx, int *by) {
  // Default block size: 32x8 = 256 threads
  // This provides good balance for typical grid sizes (128³ - 256³)
  // For very large grids (512³+), consider TIDE_BLOCK_X=32 TIDE_BLOCK_Y=16
  *bx = 32;
  *by = 8;

  // Allow runtime override via environment variables for tuning
  const char* env_bx = getenv("TIDE_BLOCK_X");
  const char* env_by = getenv("TIDE_BLOCK_Y");
  if (env_bx) *bx = atoi(env_bx);
  if (env_by) *by = atoi(env_by);
}

/*
 * Forward propagation entry point (CUDA)
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
  (void)n_threads;
  if (device >= 0) {
    cudaSetDevice((int)device);
  }

  int64_t const shot_numel = nz * ny * nx;
  int block_x = 32;
  int block_y = 4;
  get_block_xy(&block_x, &block_y);
  dim3 dimBlock(block_x, block_y, 1);
  dim3 dimGrid(
      (int)((nx - 2 * FD_PAD + dimBlock.x - 1) / dimBlock.x),
      (int)((ny - 2 * FD_PAD + dimBlock.y - 1) / dimBlock.y),
      (int)n_shots);
  size_t const shared_bytes =
      (size_t)(dimBlock.x + 2 * FD_PAD) *
      (size_t)(dimBlock.y + 2 * FD_PAD) * 6 * sizeof(TIDE_DTYPE);

  dim3 dimBlock_sources(256, 1, 1);
  dim3 dimGrid_sources(
      (int)((n_sources_per_shot + dimBlock_sources.x - 1) / dimBlock_sources.x),
      (int)n_shots,
      1);

  dim3 dimBlock_receivers(256, 1, 1);
  dim3 dimGrid_receivers(
      (int)((n_receivers_per_shot + dimBlock_receivers.x - 1) / dimBlock_receivers.x),
      (int)n_shots,
      1);

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

  bool const use_naive = false;

  // Check if fused kernel should be used (controlled by environment variable)
  // TIDE_FUSED_KERNEL=1 enables the fused cooperative kernel
  bool use_fused = false;
  const char* env_fused = getenv("TIDE_FUSED_KERNEL");
  if (env_fused && atoi(env_fused) == 1) {
    int current_device;
    cudaGetDevice(&current_device);
    if (check_cooperative_launch_support(current_device)) {
      // Check if grid size is compatible with cooperative launch
      int numBlocksPerSm = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &numBlocksPerSm, forward_kernel_fused_3d, block_x * block_y, 0);
      int numSMs;
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, current_device);
      int maxBlocks = numBlocksPerSm * numSMs;
      int totalBlocks = dimGrid.x * dimGrid.y * dimGrid.z;
      use_fused = (totalBlocks <= maxBlocks);
      // Debug output (uncomment if needed)
      // if (use_fused) {
      //   fprintf(stderr, "TIDE: Using fused kernel (%d blocks, max %d)\n", totalBlocks, maxBlocks);
      // } else {
      //   fprintf(stderr, "TIDE: Fused kernel disabled - grid too large (%d blocks > %d max)\n",
      //           totalBlocks, maxBlocks);
      // }
    }
  }

  // Prepare null pointers for store arrays (needed for cooperative kernel args)
  TIDE_DTYPE *null_store = nullptr;
  bool store_e_val = false;
  bool store_curl_val = false;

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    if (use_fused) {
      // Use fused cooperative kernel (H + E in single launch)
      // Total: 64 parameters
      void *kernelArgs[] = {
        // 0-2: Material parameters
        (void*)&ca, (void*)&cb, (void*)&cq,
        // 3-8: Field arrays
        (void*)&ex, (void*)&ey, (void*)&ez,
        (void*)&hx, (void*)&hy, (void*)&hz,
        // 9-14: PML memory for H update (E-derived)
        (void*)&m_ey_z, (void*)&m_ez_y, (void*)&m_ez_x,
        (void*)&m_ex_z, (void*)&m_ex_y, (void*)&m_ey_x,
        // 15-20: PML memory for E update (H-derived)
        (void*)&m_hz_y, (void*)&m_hy_z, (void*)&m_hx_z,
        (void*)&m_hz_x, (void*)&m_hy_x, (void*)&m_hx_y,
        // 21-26: Gradient storage (nullptr for forward-only)
        (void*)&null_store, (void*)&null_store, (void*)&null_store,
        (void*)&null_store, (void*)&null_store, (void*)&null_store,
        // 27-44: PML coefficients (18 total)
        (void*)&az, (void*)&bz, (void*)&azh, (void*)&bzh,
        (void*)&ay, (void*)&by, (void*)&ayh, (void*)&byh,
        (void*)&ax, (void*)&bx, (void*)&axh, (void*)&bxh,
        (void*)&kz, (void*)&kzh, (void*)&ky, (void*)&kyh,
        (void*)&kx, (void*)&kxh,
        // 45-47: Grid spacing reciprocals
        (void*)&rdz, (void*)&rdy, (void*)&rdx,
        // 48-52: Grid dimensions
        (void*)&n_shots, (void*)&nz, (void*)&ny, (void*)&nx, (void*)&shot_numel,
        // 53-58: PML boundaries
        (void*)&pml_z0, (void*)&pml_z1, (void*)&pml_y0, (void*)&pml_y1,
        (void*)&pml_x0, (void*)&pml_x1,
        // 59-61: Batched flags
        (void*)&ca_batched, (void*)&cb_batched, (void*)&cq_batched,
        // 62-63: Store flags
        (void*)&store_e_val, (void*)&store_curl_val
      };

      cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)forward_kernel_fused_3d,
        dimGrid, dimBlock,
        kernelArgs,
        0,  // shared memory
        0   // stream
      );
      if (err != cudaSuccess) {
        fprintf(stderr, "CUDA cooperative launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
    } else if (use_naive) {
      forward_kernel_h_3d_naive<<<dimGrid, dimBlock>>>(
          cq, ex, ey, ez, hx, hy, hz,
          m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          cq_batched);
      CHECK_KERNEL_ERROR;

      forward_kernel_e_3d_naive<<<dimGrid, dimBlock>>>(
          ca, cb, hx, hy, hz, ex, ey, ez,
          m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cb_batched);
      CHECK_KERNEL_ERROR;
    } else {
      forward_kernel_h_3d<<<dimGrid, dimBlock, shared_bytes>>>(
          cq, ex, ey, ez, hx, hy, hz,
          m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          cq_batched);
      CHECK_KERNEL_ERROR;

      forward_kernel_e_3d<<<dimGrid, dimBlock, shared_bytes>>>(
          ca, cb, hx, hy, hz, ex, ey, ez,
          m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cb_batched);
      CHECK_KERNEL_ERROR;
    }

    if (n_sources_per_shot > 0) {
      add_sources_field<<<dimGrid_sources, dimBlock_sources>>>(
          source_field,
          f + t * n_shots * n_sources_per_shot,
          sources_i,
          n_shots,
          shot_numel,
          n_sources_per_shot);
      CHECK_KERNEL_ERROR;
    }

    if (n_receivers_per_shot > 0) {
      record_receivers_field<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots * n_receivers_per_shot,
          receiver_field,
          receivers_i,
          n_shots,
          shot_numel,
          n_receivers_per_shot);
      CHECK_KERNEL_ERROR;
    }
  }
}

/*
 * Forward propagation with snapshot storage (CUDA)
 */
#ifdef __cplusplus
extern "C"
#endif
#ifdef _WIN32
__declspec(dllexport)
#endif
void FUNC(forward_with_storage)(
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
    void *const ex_store_1,
    void *const ex_store_3,
    char const *const *const ex_filenames,
    void *const ey_store_1,
    void *const ey_store_3,
    char const *const *const ey_filenames,
    void *const ez_store_1,
    void *const ez_store_3,
    char const *const *const ez_filenames,
    void *const curlx_store_1,
    void *const curlx_store_3,
    char const *const *const curlx_filenames,
    void *const curly_store_1,
    void *const curly_store_3,
    char const *const *const curly_filenames,
    void *const curlz_store_1,
    void *const curlz_store_3,
    char const *const *const curlz_filenames,
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
  (void)n_threads;

  if (device >= 0) {
    cudaSetDevice((int)device);
  }

  int64_t const shot_numel = nz * ny * nx;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp * (size_t)n_shots;
  bool const storage_bf16 = (shot_bytes_uncomp == shot_numel * 2);
  bool const can_store = (storage_mode != STORAGE_NONE);

  cudaStream_t copy_stream = nullptr;
  cudaEvent_t store_ready;
  cudaEvent_t copy_done[NUM_BUFFERS];
  bool copy_in_flight[NUM_BUFFERS];
  for (int i = 0; i < NUM_BUFFERS; i++) copy_in_flight[i] = false;

#ifdef TIDE_PROFILING
  cudaEvent_t prof_wait_start, prof_wait_end, prof_copy_start, prof_copy_end;
  float total_wait_ms = 0.0f, total_copy_ms = 0.0f;
  int n_waits = 0, n_copies = 0;
#endif

  if (storage_mode == STORAGE_CPU) {
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
#ifdef TIDE_PROFILING
    PROF_EVENT_CREATE(store_ready);
    PROF_EVENT_CREATE(prof_wait_start);
    PROF_EVENT_CREATE(prof_wait_end);
    PROF_EVENT_CREATE(prof_copy_start);
    PROF_EVENT_CREATE(prof_copy_end);
    for (int i = 0; i < NUM_BUFFERS; i++) {
      PROF_EVENT_CREATE(copy_done[i]);
    }
#else
    cudaEventCreateWithFlags(&store_ready, cudaEventDisableTiming);
    for (int i = 0; i < NUM_BUFFERS; i++) {
      cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming);
    }
#endif
  }

  int block_x = 32;
  int block_y = 4;
  get_block_xy(&block_x, &block_y);
  dim3 dimBlock(block_x, block_y, 1);
  dim3 dimGrid(
      (int)((nx - 2 * FD_PAD + dimBlock.x - 1) / dimBlock.x),
      (int)((ny - 2 * FD_PAD + dimBlock.y - 1) / dimBlock.y),
      (int)n_shots);
  size_t const shared_bytes =
      (size_t)(dimBlock.x + 2 * FD_PAD) *
      (size_t)(dimBlock.y + 2 * FD_PAD) * 6 * sizeof(TIDE_DTYPE);

  dim3 dimBlock_sources(256, 1, 1);
  dim3 dimGrid_sources(
      (int)((n_sources_per_shot + dimBlock_sources.x - 1) / dimBlock_sources.x),
      (int)n_shots,
      1);

  dim3 dimBlock_receivers(256, 1, 1);
  dim3 dimGrid_receivers(
      (int)((n_receivers_per_shot + dimBlock_receivers.x - 1) / dimBlock_receivers.x),
      (int)n_shots,
      1);

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

  FILE *fp_ex = nullptr;
  FILE *fp_ey = nullptr;
  FILE *fp_ez = nullptr;
  FILE *fp_curlx = nullptr;
  FILE *fp_curly = nullptr;
  FILE *fp_curlz = nullptr;
  if (storage_mode == STORAGE_DISK) {
    if (ca_requires_grad) {
      if (ex_filenames != nullptr) fp_ex = fopen(ex_filenames[0], "wb");
      if (ey_filenames != nullptr) fp_ey = fopen(ey_filenames[0], "wb");
      if (ez_filenames != nullptr) fp_ez = fopen(ez_filenames[0], "wb");
    }
    if (cb_requires_grad) {
      if (curlx_filenames != nullptr) fp_curlx = fopen(curlx_filenames[0], "wb");
      if (curly_filenames != nullptr) fp_curly = fopen(curly_filenames[0], "wb");
      if (curlz_filenames != nullptr) fp_curlz = fopen(curlz_filenames[0], "wb");
    }
  }

  auto store1_offset_bytes = [&](int64_t step_idx) -> size_t {
    if (storage_mode == STORAGE_DEVICE) {
      return (size_t)step_idx * bytes_per_step_store;
    }
    if (storage_mode == STORAGE_CPU) {
      return (size_t)(step_idx & 1) * bytes_per_step_store;
    }
    return 0;
  };
  auto store3_offset_bytes = [&](int64_t step_idx) -> size_t {
    return (storage_mode == STORAGE_CPU)
               ? (size_t)step_idx * bytes_per_step_store
               : 0;
  };

  bool const use_naive = false;

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    if (use_naive) {
      forward_kernel_h_3d_naive<<<dimGrid, dimBlock>>>(
          cq, ex, ey, ez, hx, hy, hz,
          m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          cq_batched);
    } else {
      forward_kernel_h_3d<<<dimGrid, dimBlock, shared_bytes>>>(
          cq, ex, ey, ez, hx, hy, hz,
          m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          cq_batched);
    }
    CHECK_KERNEL_ERROR;

    bool const store_step = (can_store && (t % step_ratio) == 0);
    bool const store_e = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;

    if (store_e || store_curl) {
      int64_t const step_idx = t / step_ratio;
      int const store_buf = (storage_mode == STORAGE_CPU) ? (int)(step_idx % NUM_BUFFERS) : 0;
      if (storage_mode == STORAGE_CPU && copy_in_flight[store_buf]) {
#ifdef TIDE_PROFILING
        PROF_RECORD(prof_wait_start, 0);
#endif
        cudaStreamWaitEvent(0, copy_done[store_buf], 0);
#ifdef TIDE_PROFILING
        PROF_RECORD(prof_wait_end, 0);
        cudaDeviceSynchronize();
        float wait_ms;
        PROF_ELAPSED(prof_wait_start, prof_wait_end, wait_ms);
        total_wait_ms += wait_ms;
        n_waits++;
#endif
      }
      size_t const store_offset = store1_offset_bytes(step_idx);
      size_t const store3_offset = store3_offset_bytes(step_idx);

      void *__restrict const ex_store_1_t =
          (uint8_t *)ex_store_1 + store_offset;
      void *__restrict const ex_store_3_t =
          (uint8_t *)ex_store_3 + store3_offset;
      void *__restrict const ey_store_1_t =
          (uint8_t *)ey_store_1 + store_offset;
      void *__restrict const ey_store_3_t =
          (uint8_t *)ey_store_3 + store3_offset;
      void *__restrict const ez_store_1_t =
          (uint8_t *)ez_store_1 + store_offset;
      void *__restrict const ez_store_3_t =
          (uint8_t *)ez_store_3 + store3_offset;
      void *__restrict const curlx_store_1_t =
          (uint8_t *)curlx_store_1 + store_offset;
      void *__restrict const curlx_store_3_t =
          (uint8_t *)curlx_store_3 + store3_offset;
      void *__restrict const curly_store_1_t =
          (uint8_t *)curly_store_1 + store_offset;
      void *__restrict const curly_store_3_t =
          (uint8_t *)curly_store_3 + store3_offset;
      void *__restrict const curlz_store_1_t =
          (uint8_t *)curlz_store_1 + store_offset;
      void *__restrict const curlz_store_3_t =
          (uint8_t *)curlz_store_3 + store3_offset;

      if (storage_bf16) {
        if (use_naive) {
          forward_kernel_e_3d_with_storage_bf16<<<dimGrid, dimBlock>>>(
              ca, cb, hx, hy, hz, ex, ey, ez,
              m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
              store_e ? (__nv_bfloat16 *)ex_store_1_t : nullptr,
              store_e ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
              store_e ? (__nv_bfloat16 *)ez_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curlx_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curly_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curlz_store_1_t : nullptr,
              az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
              kz, kzh, ky, kyh, kx, kxh,
              rdz, rdy, rdx,
              n_shots, nz, ny, nx, shot_numel,
              pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
              ca_batched, cb_batched, store_e, store_curl);
        } else {
          forward_kernel_e_3d_with_storage_tile_bf16<<<dimGrid, dimBlock, shared_bytes>>>(
              ca, cb, hx, hy, hz, ex, ey, ez,
              m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
              store_e ? (__nv_bfloat16 *)ex_store_1_t : nullptr,
              store_e ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
              store_e ? (__nv_bfloat16 *)ez_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curlx_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curly_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curlz_store_1_t : nullptr,
              az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
              kz, kzh, ky, kyh, kx, kxh,
              rdz, rdy, rdx,
              n_shots, nz, ny, nx, shot_numel,
              pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
              ca_batched, cb_batched, store_e, store_curl);
        }
      } else {
        if (use_naive) {
          forward_kernel_e_3d_with_storage<<<dimGrid, dimBlock>>>(
              ca, cb, hx, hy, hz, ex, ey, ez,
              m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
              store_e ? (TIDE_DTYPE *)ex_store_1_t : nullptr,
              store_e ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
              store_e ? (TIDE_DTYPE *)ez_store_1_t : nullptr,
              store_curl ? (TIDE_DTYPE *)curlx_store_1_t : nullptr,
              store_curl ? (TIDE_DTYPE *)curly_store_1_t : nullptr,
              store_curl ? (TIDE_DTYPE *)curlz_store_1_t : nullptr,
              az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
              kz, kzh, ky, kyh, kx, kxh,
              rdz, rdy, rdx,
              n_shots, nz, ny, nx, shot_numel,
              pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
              ca_batched, cb_batched, store_e, store_curl);
        } else {
          forward_kernel_e_3d_with_storage_tile<<<dimGrid, dimBlock, shared_bytes>>>(
              ca, cb, hx, hy, hz, ex, ey, ez,
              m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
              store_e ? (TIDE_DTYPE *)ex_store_1_t : nullptr,
              store_e ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
              store_e ? (TIDE_DTYPE *)ez_store_1_t : nullptr,
              store_curl ? (TIDE_DTYPE *)curlx_store_1_t : nullptr,
              store_curl ? (TIDE_DTYPE *)curly_store_1_t : nullptr,
              store_curl ? (TIDE_DTYPE *)curlz_store_1_t : nullptr,
              az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
              kz, kzh, ky, kyh, kx, kxh,
              rdz, rdy, rdx,
              n_shots, nz, ny, nx, shot_numel,
              pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
              ca_batched, cb_batched, store_e, store_curl);
        }
      }
      CHECK_KERNEL_ERROR;

      if (storage_mode == STORAGE_CPU) {
        cudaEventRecord(store_ready, 0);
        cudaStreamWaitEvent(copy_stream, store_ready, 0);
#ifdef TIDE_PROFILING
        PROF_RECORD(prof_copy_start, copy_stream);
#endif
        if (store_e) {
          cudaMemcpyAsync(
              ex_store_3_t, ex_store_1_t, bytes_per_step_store,
              cudaMemcpyDeviceToHost, copy_stream);
          cudaMemcpyAsync(
              ey_store_3_t, ey_store_1_t, bytes_per_step_store,
              cudaMemcpyDeviceToHost, copy_stream);
          cudaMemcpyAsync(
              ez_store_3_t, ez_store_1_t, bytes_per_step_store,
              cudaMemcpyDeviceToHost, copy_stream);
        }
        if (store_curl) {
          cudaMemcpyAsync(
              curlx_store_3_t, curlx_store_1_t, bytes_per_step_store,
              cudaMemcpyDeviceToHost, copy_stream);
          cudaMemcpyAsync(
              curly_store_3_t, curly_store_1_t, bytes_per_step_store,
              cudaMemcpyDeviceToHost, copy_stream);
          cudaMemcpyAsync(
              curlz_store_3_t, curlz_store_1_t, bytes_per_step_store,
              cudaMemcpyDeviceToHost, copy_stream);
        }
#ifdef TIDE_PROFILING
        PROF_RECORD(prof_copy_end, copy_stream);
#endif
        cudaEventRecord(copy_done[store_buf], copy_stream);
        copy_in_flight[store_buf] = true;
#ifdef TIDE_PROFILING
        n_copies++;
#endif
      } else if (storage_mode == STORAGE_DISK) {
        if (store_e) {
          storage_save_snapshot_gpu(
              ex_store_1_t, ex_store_3_t, fp_ex, storage_mode, step_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_save_snapshot_gpu(
              ey_store_1_t, ey_store_3_t, fp_ey, storage_mode, step_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_save_snapshot_gpu(
              ez_store_1_t, ez_store_3_t, fp_ez, storage_mode, step_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
        }
        if (store_curl) {
          storage_save_snapshot_gpu(
              curlx_store_1_t, curlx_store_3_t, fp_curlx, storage_mode, step_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_save_snapshot_gpu(
              curly_store_1_t, curly_store_3_t, fp_curly, storage_mode, step_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_save_snapshot_gpu(
              curlz_store_1_t, curlz_store_3_t, fp_curlz, storage_mode, step_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
        }
      }
    } else {
      if (use_naive) {
        forward_kernel_e_3d_naive<<<dimGrid, dimBlock>>>(
            ca, cb, hx, hy, hz, ex, ey, ez,
            m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
            az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
            kz, kzh, ky, kyh, kx, kxh,
            rdz, rdy, rdx,
            n_shots, nz, ny, nx, shot_numel,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            ca_batched, cb_batched);
      } else {
        forward_kernel_e_3d<<<dimGrid, dimBlock, shared_bytes>>>(
            ca, cb, hx, hy, hz, ex, ey, ez,
            m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y,
            az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
            kz, kzh, ky, kyh, kx, kxh,
            rdz, rdy, rdx,
            n_shots, nz, ny, nx, shot_numel,
            pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
            ca_batched, cb_batched);
      }
      CHECK_KERNEL_ERROR;
    }

    if (n_sources_per_shot > 0) {
      add_sources_field<<<dimGrid_sources, dimBlock_sources>>>(
          source_field,
          f + t * n_shots * n_sources_per_shot,
          sources_i,
          n_shots,
          shot_numel,
          n_sources_per_shot);
      CHECK_KERNEL_ERROR;
    }

    if (n_receivers_per_shot > 0) {
      record_receivers_field<<<dimGrid_receivers, dimBlock_receivers>>>(
          r + t * n_shots * n_receivers_per_shot,
          receiver_field,
          receivers_i,
          n_shots,
          shot_numel,
          n_receivers_per_shot);
      CHECK_KERNEL_ERROR;
    }
  }

  if (storage_mode == STORAGE_CPU) {
    cudaStreamSynchronize(copy_stream);
    cudaEventDestroy(copy_done[0]);
    cudaEventDestroy(copy_done[1]);
    cudaEventDestroy(store_ready);
    cudaStreamDestroy(copy_stream);
  }

  if (fp_ex != nullptr) fclose(fp_ex);
  if (fp_ey != nullptr) fclose(fp_ey);
  if (fp_ez != nullptr) fclose(fp_ez);
  if (fp_curlx != nullptr) fclose(fp_curlx);
  if (fp_curly != nullptr) fclose(fp_curly);
  if (fp_curlz != nullptr) fclose(fp_curlz);
}

/*
 * Backward propagation (ASM) entry point (CUDA)
 */
#ifdef __cplusplus
extern "C"
#endif
#ifdef _WIN32
__declspec(dllexport)
#endif
void FUNC(backward)(
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
    void *const ex_store_1,
    void *const ex_store_3,
    char const *const *const ex_filenames,
    void *const ey_store_1,
    void *const ey_store_3,
    char const *const *const ey_filenames,
    void *const ez_store_1,
    void *const ez_store_3,
    char const *const *const ez_filenames,
    void *const curlx_store_1,
    void *const curlx_store_3,
    char const *const *const curlx_filenames,
    void *const curly_store_1,
    void *const curly_store_3,
    char const *const *const curly_filenames,
    void *const curlz_store_1,
    void *const curlz_store_3,
    char const *const *const curlz_filenames,
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
  (void)n_threads;

  if (device >= 0) {
    cudaSetDevice((int)device);
  }

  int64_t const shot_numel = nz * ny * nx;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp * (size_t)n_shots;
  bool const storage_bf16 = (shot_bytes_uncomp == shot_numel * 2);
  bool const can_store = (storage_mode != STORAGE_NONE);

  cudaStream_t copy_stream = nullptr;
  cudaEvent_t copy_done[NUM_BUFFERS];
  bool copy_in_flight[NUM_BUFFERS];
  for (int i = 0; i < NUM_BUFFERS; i++) copy_in_flight[i] = false;

#ifdef TIDE_PROFILING
  cudaEvent_t prof_prefetch_start, prof_prefetch_end, prof_wait_start, prof_wait_end;
  float total_prefetch_ms = 0.0f, total_wait_ms = 0.0f;
  int n_prefetches = 0, n_waits = 0;
#endif

  if (storage_mode == STORAGE_CPU) {
    cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
#ifdef TIDE_PROFILING
    PROF_EVENT_CREATE(prof_prefetch_start);
    PROF_EVENT_CREATE(prof_prefetch_end);
    PROF_EVENT_CREATE(prof_wait_start);
    PROF_EVENT_CREATE(prof_wait_end);
    for (int i = 0; i < NUM_BUFFERS; i++) {
      PROF_EVENT_CREATE(copy_done[i]);
    }
#else
    for (int i = 0; i < NUM_BUFFERS; i++) {
      cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming);
    }
#endif
  }


  int block_x = 32;
  int block_y = 4;
  get_block_xy(&block_x, &block_y);
  dim3 dimBlock(block_x, block_y, 1);
  dim3 dimGrid(
      (int)((nx - 2 * FD_PAD + dimBlock.x - 1) / dimBlock.x),
      (int)((ny - 2 * FD_PAD + dimBlock.y - 1) / dimBlock.y),
      (int)n_shots);

  dim3 dimBlock_sources(256, 1, 1);
  dim3 dimGrid_sources(
      (int)((n_sources_per_shot + dimBlock_sources.x - 1) / dimBlock_sources.x),
      (int)n_shots,
      1);

  dim3 dimBlock_receivers(256, 1, 1);
  dim3 dimGrid_receivers(
      (int)((n_receivers_per_shot + dimBlock_receivers.x - 1) / dimBlock_receivers.x),
      (int)n_shots,
      1);

  TIDE_DTYPE *lambda_source_field = NULL;
  TIDE_DTYPE *lambda_receiver_field = NULL;
  switch (source_component) {
    case 0: lambda_source_field = lambda_ex; break;
    case 1: lambda_source_field = lambda_ey; break;
    case 2: lambda_source_field = lambda_ez; break;
    default: lambda_source_field = lambda_ey; break;
  }
  switch (receiver_component) {
    case 0: lambda_receiver_field = lambda_ex; break;
    case 1: lambda_receiver_field = lambda_ey; break;
    case 2: lambda_receiver_field = lambda_ez; break;
    case 3: lambda_receiver_field = lambda_hx; break;
    case 4: lambda_receiver_field = lambda_hy; break;
    case 5: lambda_receiver_field = lambda_hz; break;
    default: lambda_receiver_field = lambda_ey; break;
  }

  FILE *fp_ex = nullptr;
  FILE *fp_ey = nullptr;
  FILE *fp_ez = nullptr;
  FILE *fp_curlx = nullptr;
  FILE *fp_curly = nullptr;
  FILE *fp_curlz = nullptr;
  if (storage_mode == STORAGE_DISK) {
    if (ca_requires_grad) {
      if (ex_filenames != nullptr) fp_ex = fopen(ex_filenames[0], "rb");
      if (ey_filenames != nullptr) fp_ey = fopen(ey_filenames[0], "rb");
      if (ez_filenames != nullptr) fp_ez = fopen(ez_filenames[0], "rb");
    }
    if (cb_requires_grad) {
      if (curlx_filenames != nullptr) fp_curlx = fopen(curlx_filenames[0], "rb");
      if (curly_filenames != nullptr) fp_curly = fopen(curly_filenames[0], "rb");
      if (curlz_filenames != nullptr) fp_curlz = fopen(curlz_filenames[0], "rb");
    }
  }

  auto store1_offset_bytes = [&](int64_t step_idx) -> size_t {
    if (storage_mode == STORAGE_DEVICE) {
      return (size_t)step_idx * bytes_per_step_store;
    }
    if (storage_mode == STORAGE_CPU) {
      return (size_t)(step_idx & 1) * bytes_per_step_store;
    }
    return 0;
  };
  auto store3_offset_bytes = [&](int64_t step_idx) -> size_t {
    return (storage_mode == STORAGE_CPU)
               ? (size_t)step_idx * bytes_per_step_store
               : 0;
  };

  auto prefetch_snapshots = [&](int64_t store_idx, bool want_e, bool want_curl) {
    if (storage_mode != STORAGE_CPU || (!want_e && !want_curl)) {
      return;
    }
    int const store_buf = (int)(store_idx % NUM_BUFFERS);
    if (copy_in_flight[store_buf]) {
      cudaStreamWaitEvent(copy_stream, copy_done[store_buf], 0);
    }
#ifdef TIDE_PROFILING
    PROF_RECORD(prof_prefetch_start, copy_stream);
#endif
    size_t const store_offset = store1_offset_bytes(store_idx);
    size_t const store3_offset = store3_offset_bytes(store_idx);
    void *ex_store_1_t = (uint8_t *)ex_store_1 + store_offset;
    void *ey_store_1_t = (uint8_t *)ey_store_1 + store_offset;
    void *ez_store_1_t = (uint8_t *)ez_store_1 + store_offset;
    void *curlx_store_1_t = (uint8_t *)curlx_store_1 + store_offset;
    void *curly_store_1_t = (uint8_t *)curly_store_1 + store_offset;
    void *curlz_store_1_t = (uint8_t *)curlz_store_1 + store_offset;
    void *ex_store_3_t = (uint8_t *)ex_store_3 + store3_offset;
    void *ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;
    void *ez_store_3_t = (uint8_t *)ez_store_3 + store3_offset;
    void *curlx_store_3_t = (uint8_t *)curlx_store_3 + store3_offset;
    void *curly_store_3_t = (uint8_t *)curly_store_3 + store3_offset;
    void *curlz_store_3_t = (uint8_t *)curlz_store_3 + store3_offset;
    if (want_e) {
      cudaMemcpyAsync(
          ex_store_1_t, ex_store_3_t, bytes_per_step_store,
          cudaMemcpyHostToDevice, copy_stream);
      cudaMemcpyAsync(
          ey_store_1_t, ey_store_3_t, bytes_per_step_store,
          cudaMemcpyHostToDevice, copy_stream);
      cudaMemcpyAsync(
          ez_store_1_t, ez_store_3_t, bytes_per_step_store,
          cudaMemcpyHostToDevice, copy_stream);
    }
    if (want_curl) {
      cudaMemcpyAsync(
          curlx_store_1_t, curlx_store_3_t, bytes_per_step_store,
          cudaMemcpyHostToDevice, copy_stream);
      cudaMemcpyAsync(
          curly_store_1_t, curly_store_3_t, bytes_per_step_store,
          cudaMemcpyHostToDevice, copy_stream);
      cudaMemcpyAsync(
          curlz_store_1_t, curlz_store_3_t, bytes_per_step_store,
          cudaMemcpyHostToDevice, copy_stream);
    }
#ifdef TIDE_PROFILING
    PROF_RECORD(prof_prefetch_end, copy_stream);
#endif
    cudaEventRecord(copy_done[store_buf], copy_stream);
    copy_in_flight[store_buf] = true;
#ifdef TIDE_PROFILING
    n_prefetches++;
#endif
  };

  int64_t const t_min = start_t - nt;
  if (storage_mode == STORAGE_CPU && (ca_requires_grad || cb_requires_grad)) {
    int64_t t_prefetch = start_t - 1;
    int64_t const mod = t_prefetch % step_ratio;
    if (mod != 0) t_prefetch -= mod;
    if (t_prefetch >= t_min) {
      prefetch_snapshots(
          t_prefetch / step_ratio, ca_requires_grad, cb_requires_grad);
    }
  }

  for (int64_t t = start_t - 1; t >= t_min; --t) {
    if (n_receivers_per_shot > 0) {
      add_sources_field<<<dimGrid_receivers, dimBlock_receivers>>>(
          lambda_receiver_field,
          grad_r + t * n_shots * n_receivers_per_shot,
          receivers_i,
          n_shots,
          shot_numel,
          n_receivers_per_shot);
      CHECK_KERNEL_ERROR;
    }

    if (n_sources_per_shot > 0) {
      record_receivers_field<<<dimGrid_sources, dimBlock_sources>>>(
          grad_f + t * n_shots * n_sources_per_shot,
          lambda_source_field,
          sources_i,
          n_shots,
          shot_numel,
          n_sources_per_shot);
      CHECK_KERNEL_ERROR;
    }

    bool const do_grad = can_store && ((t % step_ratio) == 0);
    bool const grad_e = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;

    int64_t store_idx = -1;
    void *ex_store_1_t = nullptr;
    void *ey_store_1_t = nullptr;
    void *ez_store_1_t = nullptr;
    void *curlx_store_1_t = nullptr;
    void *curly_store_1_t = nullptr;
    void *curlz_store_1_t = nullptr;
    if (do_grad) {
      store_idx = t / step_ratio;
      size_t const store_offset = store1_offset_bytes(store_idx);
      size_t const store3_offset = store3_offset_bytes(store_idx);
      ex_store_1_t = (uint8_t *)ex_store_1 + store_offset;
      ey_store_1_t = (uint8_t *)ey_store_1 + store_offset;
      ez_store_1_t = (uint8_t *)ez_store_1 + store_offset;
      curlx_store_1_t = (uint8_t *)curlx_store_1 + store_offset;
      curly_store_1_t = (uint8_t *)curly_store_1 + store_offset;
      curlz_store_1_t = (uint8_t *)curlz_store_1 + store_offset;

      if (storage_mode == STORAGE_CPU && (grad_e || grad_curl)) {
        int const store_buf = (int)(store_idx % NUM_BUFFERS);
        if (!copy_in_flight[store_buf]) {
          prefetch_snapshots(store_idx, grad_e, grad_curl);
        }
#ifdef TIDE_PROFILING
        PROF_RECORD(prof_wait_start, 0);
#endif
        cudaStreamWaitEvent(0, copy_done[store_buf], 0);
#ifdef TIDE_PROFILING
        PROF_RECORD(prof_wait_end, 0);
        cudaDeviceSynchronize();
        float wait_ms;
        PROF_ELAPSED(prof_wait_start, prof_wait_end, wait_ms);
        total_wait_ms += wait_ms;
        n_waits++;
#endif
      } else if (storage_mode == STORAGE_DISK && (grad_e || grad_curl)) {
        void *ex_store_3_t = (uint8_t *)ex_store_3 + store3_offset;
        void *ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;
        void *ez_store_3_t = (uint8_t *)ez_store_3 + store3_offset;
        void *curlx_store_3_t = (uint8_t *)curlx_store_3 + store3_offset;
        void *curly_store_3_t = (uint8_t *)curly_store_3 + store3_offset;
        void *curlz_store_3_t = (uint8_t *)curlz_store_3 + store3_offset;
        if (grad_e) {
          storage_load_snapshot_gpu(
              ex_store_1_t, ex_store_3_t, fp_ex, storage_mode, store_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_load_snapshot_gpu(
              ey_store_1_t, ey_store_3_t, fp_ey, storage_mode, store_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_load_snapshot_gpu(
              ez_store_1_t, ez_store_3_t, fp_ez, storage_mode, store_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
        }
        if (grad_curl) {
          storage_load_snapshot_gpu(
              curlx_store_1_t, curlx_store_3_t, fp_curlx, storage_mode, store_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_load_snapshot_gpu(
              curly_store_1_t, curly_store_3_t, fp_curly, storage_mode, store_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
          storage_load_snapshot_gpu(
              curlz_store_1_t, curlz_store_3_t, fp_curlz, storage_mode, store_idx,
              (size_t)shot_bytes_uncomp, (size_t)n_shots);
        }
      }
    }

    backward_kernel_lambda_h_3d<<<dimGrid, dimBlock>>>(
        cb, lambda_ex, lambda_ey, lambda_ez,
        lambda_hx, lambda_hy, lambda_hz,
        m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x,
        m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x,
        az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
      
        kz, kzh, ky, kyh, kx, kxh,
        rdz, rdy, rdx,
        n_shots, nz, ny, nx, shot_numel,
        pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
        cb_batched);
    CHECK_KERNEL_ERROR;

    if (storage_bf16) {
      backward_kernel_lambda_e_3d_with_grad_bf16<<<dimGrid, dimBlock>>>(
          ca, cq, lambda_hx, lambda_hy, lambda_hz,
          lambda_ex, lambda_ey, lambda_ez,
          m_lambda_hz_y, m_lambda_hy_z, m_lambda_hx_z,
          m_lambda_hz_x, m_lambda_hy_x, m_lambda_hx_y,
          grad_e ? (__nv_bfloat16 const *)ex_store_1_t : nullptr,
          grad_e ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
          grad_e ? (__nv_bfloat16 const *)ez_store_1_t : nullptr,
          grad_curl ? (__nv_bfloat16 const *)curlx_store_1_t : nullptr,
          grad_curl ? (__nv_bfloat16 const *)curly_store_1_t : nullptr,
          grad_curl ? (__nv_bfloat16 const *)curlz_store_1_t : nullptr,
          grad_ca_shot, grad_cb_shot,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cq_batched, grad_e, grad_curl, step_ratio);
    } else {
      backward_kernel_lambda_e_3d_with_grad<<<dimGrid, dimBlock>>>(
          ca, cq, lambda_hx, lambda_hy, lambda_hz,
          lambda_ex, lambda_ey, lambda_ez,
          m_lambda_hz_y, m_lambda_hy_z, m_lambda_hx_z,
          m_lambda_hz_x, m_lambda_hy_x, m_lambda_hx_y,
          grad_e ? (TIDE_DTYPE const *)ex_store_1_t : nullptr,
          grad_e ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
          grad_e ? (TIDE_DTYPE const *)ez_store_1_t : nullptr,
          grad_curl ? (TIDE_DTYPE const *)curlx_store_1_t : nullptr,
          grad_curl ? (TIDE_DTYPE const *)curly_store_1_t : nullptr,
          grad_curl ? (TIDE_DTYPE const *)curlz_store_1_t : nullptr,
          grad_ca_shot, grad_cb_shot,
          az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh,
          kz, kzh, ky, kyh, kx, kxh,
          rdz, rdy, rdx,
          n_shots, nz, ny, nx, shot_numel,
          pml_z0, pml_z1, pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cq_batched, grad_e, grad_curl, step_ratio);
    }
    CHECK_KERNEL_ERROR;

    if (storage_mode == STORAGE_CPU && do_grad &&
        (ca_requires_grad || cb_requires_grad)) {
      int64_t const next_t = t - step_ratio;
      if (next_t >= t_min) {
        prefetch_snapshots(store_idx - 1, ca_requires_grad, cb_requires_grad);
      }
    }
  }

  if (ca_requires_grad && !ca_batched) {
    dim3 dimBlock_combine(256, 1, 1);
    dim3 dimGrid_combine(
        (int)((shot_numel + dimBlock_combine.x - 1) / dimBlock_combine.x),
        1, 1);
    combine_grad_3d<<<dimGrid_combine, dimBlock_combine>>>(
        grad_ca, grad_ca_shot, shot_numel, n_shots);
    CHECK_KERNEL_ERROR;
  }
  if (cb_requires_grad && !cb_batched) {
    dim3 dimBlock_combine(256, 1, 1);
    dim3 dimGrid_combine(
        (int)((shot_numel + dimBlock_combine.x - 1) / dimBlock_combine.x),
        1, 1);
    combine_grad_3d<<<dimGrid_combine, dimBlock_combine>>>(
        grad_cb, grad_cb_shot, shot_numel, n_shots);
    CHECK_KERNEL_ERROR;
  }

  if ((grad_eps != nullptr || grad_sigma != nullptr) &&
      (ca_requires_grad || cb_requires_grad)) {
    dim3 dimBlock_conv(256, 1, 1);
    int64_t const total = (ca_batched || cb_batched)
                              ? (n_shots * shot_numel)
                              : shot_numel;
    dim3 dimGrid_conv(
        (int)((total + dimBlock_conv.x - 1) / dimBlock_conv.x), 1, 1);
    convert_grad_ca_cb_to_eps_sigma_3d<<<dimGrid_conv, dimBlock_conv>>>(
        ca, cb, grad_ca, grad_cb, grad_ca_shot, grad_cb_shot,
        grad_eps, grad_sigma, dt, shot_numel, n_shots,
        ca_requires_grad, cb_requires_grad,
        ca_batched, cb_batched);
    CHECK_KERNEL_ERROR;
  }

  if (storage_mode == STORAGE_CPU) {
    cudaStreamSynchronize(copy_stream);
#ifdef TIDE_PROFILING
    if (n_prefetches > 0) {
      PROF_PRINT("Backward 3D H2D prefetch count", (float)n_prefetches);
    }
    if (n_waits > 0) {
      float avg_wait_ms = total_wait_ms / n_waits;
      PROF_PRINT("Backward 3D avg wait time", avg_wait_ms);
      PROF_PRINT("Backward 3D total wait time", total_wait_ms);
    }
    cudaEventDestroy(prof_prefetch_start);
    cudaEventDestroy(prof_prefetch_end);
    cudaEventDestroy(prof_wait_start);
    cudaEventDestroy(prof_wait_end);
#endif
    for (int i = 0; i < NUM_BUFFERS; i++) {
      cudaEventDestroy(copy_done[i]);
    }
    cudaStreamDestroy(copy_stream);
  }

  if (fp_ex != nullptr) fclose(fp_ex);
  if (fp_ey != nullptr) fclose(fp_ey);
  if (fp_ez != nullptr) fclose(fp_ez);
  if (fp_curlx != nullptr) fclose(fp_curlx);
  if (fp_curly != nullptr) fclose(fp_curly);
  if (fp_curlz != nullptr) fclose(fp_curlz);
}
