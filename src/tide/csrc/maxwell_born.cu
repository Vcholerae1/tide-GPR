/*
 * Maxwell Born wave equation propagator (CUDA implementation)
 * 
 * This file implements the Born approximation for 2D TM Maxwell equations.
 * It propagates both background and scattered wavefields simultaneously.
 *
 * Background fields: Ey_bg, Hx_bg, Hz_bg (satisfy full Maxwell equations)
 * Scattered fields: Ey_sc, Hx_sc, Hz_sc (satisfy linearized equations with scattering source)
 *
 * Born approximation linearization:
 *   Ey_sc^{n+1} = Ca0 * Ey_sc^n + Cb0 * (curl H_sc)^n
 *               + δCa * Ey_bg^n + δCb * (curl H_bg)^n
 *
 *   Hx_sc^{n+1/2} = Hx_sc^{n-1/2} - Cq0 * D_z[Ey_sc] - δCq * D_z[Ey_bg]
 *   Hz_sc^{n+1/2} = Hz_sc^{n-1/2} + Cq0 * D_x[Ey_sc] + δCq * D_x[Ey_bg]
 *
 * where:
 *   δCa, δCb, δCq are the perturbations in material coefficients
 *   caused by scattering perturbations δε, δσ, δμ
 */

#include <stdio.h>
#include <cstdint>
#include <cuda_bf16.h>
#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#ifndef TIDE_DEVICE
#define TIDE_DEVICE cuda
#endif

#ifndef NUM_BUFFERS
#define NUM_BUFFERS 3
#endif

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
  maxwell_born_tm_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) \
  CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, TIDE_DEVICE)

// 2D indexing macros
#define ND_INDEX(i, dy, dx) (i + (dy)*nx + (dx))
#define ND_INDEX_J(j, dy, dx) (j + (dy)*nx + (dx))

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

// Background field access macros
#define EY_BG(dy, dx) ey_bg[ND_INDEX(i, dy, dx)]
#define HX_BG(dy, dx) hx_bg[ND_INDEX(i, dy, dx)]
#define HZ_BG(dy, dx) hz_bg[ND_INDEX(i, dy, dx)]

// Scattered field access macros
#define EY_SC(dy, dx) ey_sc[ND_INDEX(i, dy, dx)]
#define HX_SC(dy, dx) hx_sc[ND_INDEX(i, dy, dx)]
#define HZ_SC(dy, dx) hz_sc[ND_INDEX(i, dy, dx)]

// Material parameter access macros (background)
#define CA0(dy, dx) ca0_shot[ND_INDEX_J(j, dy, dx)]
#define CB0(dy, dx) cb0_shot[ND_INDEX_J(j, dy, dx)]
#define CQ0(dy, dx) cq0_shot[ND_INDEX_J(j, dy, dx)]

// Material parameter perturbation access macros
#define DCA(dy, dx) dca_shot[ND_INDEX_J(j, dy, dx)]
#define DCB(dy, dx) dcb_shot[ND_INDEX_J(j, dy, dx)]
#define DCQ(dy, dx) dcq_shot[ND_INDEX_J(j, dy, dx)]

// PML memory variable macros for background fields
#define M_HX_Z_BG(dy, dx) m_hx_z_bg[ND_INDEX(i, dy, dx)]
#define M_HZ_X_BG(dy, dx) m_hz_x_bg[ND_INDEX(i, dy, dx)]
#define M_EY_X_BG(dy, dx) m_ey_x_bg[ND_INDEX(i, dy, dx)]
#define M_EY_Z_BG(dy, dx) m_ey_z_bg[ND_INDEX(i, dy, dx)]

// PML memory variable macros for scattered fields
#define M_HX_Z_SC(dy, dx) m_hx_z_sc[ND_INDEX(i, dy, dx)]
#define M_HZ_X_SC(dy, dx) m_hz_x_sc[ND_INDEX(i, dy, dx)]
#define M_EY_X_SC(dy, dx) m_ey_x_sc[ND_INDEX(i, dy, dx)]
#define M_EY_Z_SC(dy, dx) m_ey_z_sc[ND_INDEX(i, dy, dx)]

#define MAX(a, b) (a > b ? a : b)

#define EP0 ((TIDE_DTYPE)8.8541878128e-12)

namespace {

// Device constants
__constant__ TIDE_DTYPE rdy;
__constant__ TIDE_DTYPE rdx;
__constant__ int64_t n_shots;
__constant__ int64_t ny;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_sources_per_shot;
__constant__ int64_t n_receivers_per_shot;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_x1;
__constant__ bool ca0_batched;
__constant__ bool cb0_batched;
__constant__ bool cq0_batched;
__constant__ bool dca_batched;
__constant__ bool dcb_batched;
__constant__ bool dcq_batched;

// Add source to Ey fields (both background and scattered)
__global__ void add_sources_ey_both(
    TIDE_DTYPE *__restrict const ey_bg,
    TIDE_DTYPE *__restrict const ey_sc,
    TIDE_DTYPE const *__restrict const f,
    TIDE_DTYPE const *__restrict const f_sc,
    int64_t const *__restrict const sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      ey_bg[shot_idx * shot_numel + src] += f[k];
      ey_sc[shot_idx * shot_numel + src] += f_sc[k];
    }
  }
}

// Add source to Ey field (single field)
__global__ void add_sources_ey(
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      ey[shot_idx * shot_numel + src] += f[k];
    }
  }
}

// Record field at receiver locations
__global__ void record_receivers_ey(
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE const *__restrict const ey,
    int64_t const *__restrict receivers_i) {
  int64_t receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (0 <= rec) {
      r[k] = ey[shot_idx * shot_numel + rec];
    }
  }
}

// Forward kernel: Update H fields for background wavefield
__global__ __launch_bounds__(256) void forward_kernel_h_bg(
    TIDE_DTYPE const *__restrict const cq0,
    TIDE_DTYPE const *__restrict const ey_bg,
    TIDE_DTYPE *__restrict const hx_bg,
    TIDE_DTYPE *__restrict const hz_bg,
    TIDE_DTYPE *__restrict const m_ey_x_bg,
    TIDE_DTYPE *__restrict const m_ey_z_bg,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {
  
#if FD_PAD > 1
  extern __shared__ TIDE_DTYPE shmem[];
  TIDE_DTYPE *__restrict const tile_ey = shmem;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;

  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  int64_t const tile_numel = tile_w * tile_h;
  
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      tile_ey[ly * tile_pitch + lx] = __ldg(&ey_bg[base + gy * nx + gx]);
    } else {
      tile_ey[ly * tile_pitch + lx] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define EY_BG_L(dy, dx) tile_ey[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define EY_BG_L(dy, dx) EY_BG(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const pml_y0h = pml_y0;
    int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
    int64_t const pml_x0h = pml_x0;
    int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const cq0_shot_i = cq0_batched ? cq0[i] : cq0[j];

    TIDE_DTYPE byh_val = __ldg(&byh[y]);
    TIDE_DTYPE ayh_val = __ldg(&ayh[y]);
    TIDE_DTYPE kyh_val = __ldg(&kyh[y]);
    TIDE_DTYPE bxh_val = __ldg(&bxh[x]);
    TIDE_DTYPE axh_val = __ldg(&axh[x]);
    TIDE_DTYPE kxh_val = __ldg(&kxh[x]);

    // Update Hx_bg: Hx = Hx - cq0 * dEy/dz
    if (y < ny - FD_PAD) {
      bool pml_y = y < pml_y0h || y >= pml_y1h;
      TIDE_DTYPE dey_dz = DIFFYH1(EY_BG_L);

      if (pml_y) {
        m_ey_z_bg[i] = byh_val * m_ey_z_bg[i] + ayh_val * dey_dz;
        dey_dz = dey_dz / kyh_val + m_ey_z_bg[i];
      }

      hx_bg[i] -= cq0_shot_i * dey_dz;
    }

    // Update Hz_bg: Hz = Hz + cq0 * dEy/dx
    if (x < nx - FD_PAD) {
      bool pml_x = x < pml_x0h || x >= pml_x1h;
      TIDE_DTYPE dey_dx = DIFFXH1(EY_BG_L);

      if (pml_x) {
        m_ey_x_bg[i] = bxh_val * m_ey_x_bg[i] + axh_val * dey_dx;
        dey_dx = dey_dx / kxh_val + m_ey_x_bg[i];
      }

      hz_bg[i] += cq0_shot_i * dey_dx;
    }
  }

#undef EY_BG_L
}

// Forward kernel: Update H fields for scattered wavefield
// Includes Born scattering source terms: -δCq * D[Ey_bg]
__global__ __launch_bounds__(256) void forward_kernel_h_sc(
    TIDE_DTYPE const *__restrict const cq0,
    TIDE_DTYPE const *__restrict const dcq,
    TIDE_DTYPE const *__restrict const ey_bg,
    TIDE_DTYPE const *__restrict const ey_sc,
    TIDE_DTYPE *__restrict const hx_sc,
    TIDE_DTYPE *__restrict const hz_sc,
    TIDE_DTYPE *__restrict const m_ey_x_bg,
    TIDE_DTYPE *__restrict const m_ey_z_bg,
    TIDE_DTYPE *__restrict const m_ey_x_sc,
    TIDE_DTYPE *__restrict const m_ey_z_sc,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {
  
#if FD_PAD > 1
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_ey_bg = shmem;
  TIDE_DTYPE *__restrict const tile_ey_sc = shmem + tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const tile_pitch = tile_w;
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;

  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const g = base + gy * nx + gx;
      int64_t const offset = ly * tile_pitch + lx;
      tile_ey_bg[offset] = __ldg(&ey_bg[g]);
      tile_ey_sc[offset] = __ldg(&ey_sc[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_ey_bg[offset] = (TIDE_DTYPE)0;
      tile_ey_sc[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define EY_BG_L(dy, dx) tile_ey_bg[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define EY_SC_L(dy, dx) tile_ey_sc[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define EY_BG_L(dy, dx) EY_BG(dy, dx)
#define EY_SC_L(dy, dx) EY_SC(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t const pml_y0h = pml_y0;
    int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
    int64_t const pml_x0h = pml_x0;
    int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);

    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const cq0_shot_i = cq0_batched ? cq0[i] : cq0[j];
    TIDE_DTYPE const dcq_shot_i = dcq_batched ? dcq[i] : dcq[j];

    TIDE_DTYPE byh_val = __ldg(&byh[y]);
    TIDE_DTYPE ayh_val = __ldg(&ayh[y]);
    TIDE_DTYPE kyh_val = __ldg(&kyh[y]);
    TIDE_DTYPE bxh_val = __ldg(&bxh[x]);
    TIDE_DTYPE axh_val = __ldg(&axh[x]);
    TIDE_DTYPE kxh_val = __ldg(&kxh[x]);

    // Update Hx_sc: Hx_sc = Hx_sc - cq0 * dEy_sc/dz - dcq * dEy_bg/dz
    if (y < ny - FD_PAD) {
      bool pml_y = y < pml_y0h || y >= pml_y1h;
      
      TIDE_DTYPE dey_bg_dz = DIFFYH1(EY_BG_L);
      TIDE_DTYPE dey_sc_dz = DIFFYH1(EY_SC_L);

      if (pml_y) {
        // Note: m_ey_z_bg is already updated in forward_kernel_h_bg
        // We need to read the CPML-corrected derivative
        TIDE_DTYPE dey_bg_dz_cpml = dey_bg_dz / kyh_val + m_ey_z_bg[i];
        
        m_ey_z_sc[i] = byh_val * m_ey_z_sc[i] + ayh_val * dey_sc_dz;
        dey_sc_dz = dey_sc_dz / kyh_val + m_ey_z_sc[i];
        
        hx_sc[i] -= cq0_shot_i * dey_sc_dz + dcq_shot_i * dey_bg_dz_cpml;
      } else {
        hx_sc[i] -= cq0_shot_i * dey_sc_dz + dcq_shot_i * dey_bg_dz;
      }
    }

    // Update Hz_sc: Hz_sc = Hz_sc + cq0 * dEy_sc/dx + dcq * dEy_bg/dx
    if (x < nx - FD_PAD) {
      bool pml_x = x < pml_x0h || x >= pml_x1h;
      
      TIDE_DTYPE dey_bg_dx = DIFFXH1(EY_BG_L);
      TIDE_DTYPE dey_sc_dx = DIFFXH1(EY_SC_L);

      if (pml_x) {
        TIDE_DTYPE dey_bg_dx_cpml = dey_bg_dx / kxh_val + m_ey_x_bg[i];
        
        m_ey_x_sc[i] = bxh_val * m_ey_x_sc[i] + axh_val * dey_sc_dx;
        dey_sc_dx = dey_sc_dx / kxh_val + m_ey_x_sc[i];
        
        hz_sc[i] += cq0_shot_i * dey_sc_dx + dcq_shot_i * dey_bg_dx_cpml;
      } else {
        hz_sc[i] += cq0_shot_i * dey_sc_dx + dcq_shot_i * dey_bg_dx;
      }
    }
  }

#undef EY_BG_L
#undef EY_SC_L
}

// Forward kernel: Update E field for background wavefield
__global__ __launch_bounds__(256) void forward_kernel_e_bg(
    TIDE_DTYPE const *__restrict const ca0,
    TIDE_DTYPE const *__restrict const cb0,
    TIDE_DTYPE const *__restrict const hx_bg,
    TIDE_DTYPE const *__restrict const hz_bg,
    TIDE_DTYPE *__restrict const ey_bg,
    TIDE_DTYPE *__restrict const m_hx_z_bg,
    TIDE_DTYPE *__restrict const m_hz_x_bg,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

#if FD_PAD > 1
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_hx = shmem;
  TIDE_DTYPE *__restrict const tile_hz = shmem + tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;
  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const g = base + gy * nx + gx;
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = __ldg(&hx_bg[g]);
      tile_hz[offset] = __ldg(&hz_bg[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx[offset] = (TIDE_DTYPE)0;
      tile_hz[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define HX_BG_L(dy, dx) tile_hx[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_BG_L(dy, dx) tile_hz[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define HX_BG_L(dy, dx) HX_BG(dy, dx)
#define HZ_BG_L(dy, dx) HZ_BG(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca0_shot_i = ca0_batched ? ca0[i] : ca0[j];
    TIDE_DTYPE const cb0_shot_i = cb0_batched ? cb0[i] : cb0[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE dhz_dx = DIFFX1(HZ_BG_L);
    TIDE_DTYPE dhx_dz = DIFFY1(HX_BG_L);

    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      m_hz_x_bg[i] = bx_val * m_hz_x_bg[i] + ax_val * dhz_dx;
      dhz_dx = dhz_dx / kx_val + m_hz_x_bg[i];
    }

    if (pml_y) {
      m_hx_z_bg[i] = by_val * m_hx_z_bg[i] + ay_val * dhx_dz;
      dhx_dz = dhx_dz / ky_val + m_hx_z_bg[i];
    }

    ey_bg[i] = ca0_shot_i * ey_bg[i] + cb0_shot_i * (dhz_dx - dhx_dz);
  }

#undef HX_BG_L
#undef HZ_BG_L
}

// Forward kernel: Update E field for scattered wavefield
// Includes Born scattering source terms: δCa * Ey_bg + δCb * curl(H_bg)
__global__ __launch_bounds__(256) void forward_kernel_e_sc(
    TIDE_DTYPE const *__restrict const ca0,
    TIDE_DTYPE const *__restrict const cb0,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const hx_bg,
    TIDE_DTYPE const *__restrict const hz_bg,
    TIDE_DTYPE const *__restrict const hx_sc,
    TIDE_DTYPE const *__restrict const hz_sc,
    TIDE_DTYPE const *__restrict const ey_bg,
    TIDE_DTYPE *__restrict const ey_sc,
    TIDE_DTYPE *__restrict const m_hx_z_bg,
    TIDE_DTYPE *__restrict const m_hz_x_bg,
    TIDE_DTYPE *__restrict const m_hx_z_sc,
    TIDE_DTYPE *__restrict const m_hz_x_sc,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh) {

#if FD_PAD > 1
  extern __shared__ TIDE_DTYPE shmem[];
  int64_t const tile_w = (int64_t)blockDim.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)blockDim.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_pitch = tile_w;
  int64_t const tile_numel = tile_w * tile_h;
  TIDE_DTYPE *__restrict const tile_hx_bg = shmem;
  TIDE_DTYPE *__restrict const tile_hz_bg = shmem + tile_numel;
  TIDE_DTYPE *__restrict const tile_hx_sc = shmem + 2 * tile_numel;
  TIDE_DTYPE *__restrict const tile_hz_sc = shmem + 3 * tile_numel;
#endif

  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + FD_PAD;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + FD_PAD;
  int64_t shot_idx = (int64_t)blockIdx.z * (int64_t)blockDim.z +
                     (int64_t)threadIdx.z;

  if (shot_idx >= n_shots) return;

#if FD_PAD > 1
  int64_t const x0 = (int64_t)blockIdx.x * (int64_t)blockDim.x + FD_PAD;
  int64_t const y0 = (int64_t)blockIdx.y * (int64_t)blockDim.y + FD_PAD;
  int64_t const base = shot_idx * shot_numel;
  int64_t const t = (int64_t)threadIdx.y * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x;
  int64_t const nthreads = (int64_t)blockDim.x * (int64_t)blockDim.y;
  
  for (int64_t idx = t; idx < tile_numel; idx += nthreads) {
    int64_t const ly = idx / tile_w;
    int64_t const lx = idx - ly * tile_w;
    int64_t const gx = x0 - FD_PAD + lx;
    int64_t const gy = y0 - FD_PAD + ly;
    if (0 <= gx && gx < nx && 0 <= gy && gy < ny) {
      int64_t const g = base + gy * nx + gx;
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx_bg[offset] = __ldg(&hx_bg[g]);
      tile_hz_bg[offset] = __ldg(&hz_bg[g]);
      tile_hx_sc[offset] = __ldg(&hx_sc[g]);
      tile_hz_sc[offset] = __ldg(&hz_sc[g]);
    } else {
      int64_t const offset = ly * tile_pitch + lx;
      tile_hx_bg[offset] = (TIDE_DTYPE)0;
      tile_hz_bg[offset] = (TIDE_DTYPE)0;
      tile_hx_sc[offset] = (TIDE_DTYPE)0;
      tile_hz_sc[offset] = (TIDE_DTYPE)0;
    }
  }
  __syncthreads();

#define HX_BG_L(dy, dx) tile_hx_bg[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_BG_L(dy, dx) tile_hz_bg[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HX_SC_L(dy, dx) tile_hx_sc[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#define HZ_SC_L(dy, dx) tile_hz_sc[((int64_t)threadIdx.y + (int64_t)FD_PAD + (dy)) * tile_pitch + ((int64_t)threadIdx.x + (int64_t)FD_PAD + (dx))]
#else
#define HX_BG_L(dy, dx) HX_BG(dy, dx)
#define HZ_BG_L(dy, dx) HZ_BG(dy, dx)
#define HX_SC_L(dy, dx) HX_SC(dy, dx)
#define HZ_SC_L(dy, dx) HZ_SC(dy, dx)
#endif

  if (y < ny - FD_PAD + 1 && x < nx - FD_PAD + 1 && shot_idx < n_shots) {
    int64_t j = y * nx + x;
    int64_t i = shot_idx * shot_numel + j;

    TIDE_DTYPE const ca0_shot_i = ca0_batched ? ca0[i] : ca0[j];
    TIDE_DTYPE const cb0_shot_i = cb0_batched ? cb0[i] : cb0[j];
    TIDE_DTYPE const dca_shot_i = dca_batched ? dca[i] : dca[j];
    TIDE_DTYPE const dcb_shot_i = dcb_batched ? dcb[i] : dcb[j];

    bool pml_y = y < pml_y0 || y >= pml_y1;
    bool pml_x = x < pml_x0 || x >= pml_x1;

    TIDE_DTYPE dhz_bg_dx = DIFFX1(HZ_BG_L);
    TIDE_DTYPE dhx_bg_dz = DIFFY1(HX_BG_L);
    TIDE_DTYPE dhz_sc_dx = DIFFX1(HZ_SC_L);
    TIDE_DTYPE dhx_sc_dz = DIFFY1(HX_SC_L);

    TIDE_DTYPE bx_val = __ldg(&bx[x]);
    TIDE_DTYPE ax_val = __ldg(&ax[x]);
    TIDE_DTYPE kx_val = __ldg(&kx[x]);
    TIDE_DTYPE by_val = __ldg(&by[y]);
    TIDE_DTYPE ay_val = __ldg(&ay[y]);
    TIDE_DTYPE ky_val = __ldg(&ky[y]);

    if (pml_x) {
      // Note: m_hz_x_bg is already updated in forward_kernel_e_bg
      TIDE_DTYPE dhz_bg_dx_cpml = dhz_bg_dx / kx_val + m_hz_x_bg[i];
      
      m_hz_x_sc[i] = bx_val * m_hz_x_sc[i] + ax_val * dhz_sc_dx;
      dhz_sc_dx = dhz_sc_dx / kx_val + m_hz_x_sc[i];
      
      // For scattering source: use CPML-corrected background curl
      dhz_bg_dx = dhz_bg_dx_cpml;
    }

    if (pml_y) {
      TIDE_DTYPE dhx_bg_dz_cpml = dhx_bg_dz / ky_val + m_hx_z_bg[i];
      
      m_hx_z_sc[i] = by_val * m_hx_z_sc[i] + ay_val * dhx_sc_dz;
      dhx_sc_dz = dhx_sc_dz / ky_val + m_hx_z_sc[i];
      
      dhx_bg_dz = dhx_bg_dz_cpml;
    }

    TIDE_DTYPE curl_h_bg = dhz_bg_dx - dhx_bg_dz;
    TIDE_DTYPE curl_h_sc = dhz_sc_dx - dhx_sc_dz;

    // Born scattering update:
    // Ey_sc^{n+1} = Ca0 * Ey_sc^n + Cb0 * curl_H_sc
    //             + δCa * Ey_bg^n + δCb * curl_H_bg
    ey_sc[i] = ca0_shot_i * ey_sc[i] + cb0_shot_i * curl_h_sc
             + dca_shot_i * ey_bg[i] + dcb_shot_i * curl_h_bg;
  }

#undef HX_BG_L
#undef HZ_BG_L
#undef HX_SC_L
#undef HZ_SC_L
}

// Combine per-shot gradients
__global__ void combine_grad(
    TIDE_DTYPE *__restrict const grad,
    TIDE_DTYPE const *__restrict const grad_shot) {
  int64_t const x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
                    (int64_t)threadIdx.x + FD_PAD;
  int64_t const y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
                    (int64_t)threadIdx.y + FD_PAD;
  if (y < ny - FD_PAD && x < nx - FD_PAD) {
    int64_t const i = y * nx + x;
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      grad[i] += grad_shot[shot_idx * shot_numel + i];
    }
  }
}

} // namespace

extern "C" {

// Forward propagation function
void FUNC(forward)(
    TIDE_DTYPE const *__restrict const ca0,
    TIDE_DTYPE const *__restrict const cb0,
    TIDE_DTYPE const *__restrict const cq0,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const dcq,
    TIDE_DTYPE *__restrict const ey_bg,
    TIDE_DTYPE *__restrict const hx_bg,
    TIDE_DTYPE *__restrict const hz_bg,
    TIDE_DTYPE *__restrict const ey_sc,
    TIDE_DTYPE *__restrict const hx_sc,
    TIDE_DTYPE *__restrict const hz_sc,
    TIDE_DTYPE *__restrict const m_ey_x_bg,
    TIDE_DTYPE *__restrict const m_ey_z_bg,
    TIDE_DTYPE *__restrict const m_hx_z_bg,
    TIDE_DTYPE *__restrict const m_hz_x_bg,
    TIDE_DTYPE *__restrict const m_ey_x_sc,
    TIDE_DTYPE *__restrict const m_ey_z_sc,
    TIDE_DTYPE *__restrict const m_hx_z_sc,
    TIDE_DTYPE *__restrict const m_hz_x_sc,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kx,
    TIDE_DTYPE const *__restrict const kxh,
    TIDE_DTYPE *__restrict const r_bg,
    TIDE_DTYPE *__restrict const r_sc,
    TIDE_DTYPE const *__restrict const f,
    TIDE_DTYPE const *__restrict const f_sc,
    int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i,
    TIDE_DTYPE const rdy_val,
    TIDE_DTYPE const rdx_val,
    int64_t const ny_val,
    int64_t const nx_val,
    int64_t const n_shots_val,
    int64_t const n_sources_per_shot_val,
    int64_t const n_receivers_per_shot_val,
    int64_t const pml_y0_val,
    int64_t const pml_y1_val,
    int64_t const pml_x0_val,
    int64_t const pml_x1_val,
    int64_t const nt,
    bool const ca0_batched_val,
    bool const cb0_batched_val,
    bool const cq0_batched_val,
    bool const dca_batched_val,
    bool const dcb_batched_val,
    bool const dcq_batched_val) {
  
  // Copy constants to device
  gpuErrchk(cudaMemcpyToSymbol(rdy, &rdy_val, sizeof(TIDE_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(rdx, &rdx_val, sizeof(TIDE_DTYPE)));
  gpuErrchk(cudaMemcpyToSymbol(ny, &ny_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(nx, &nx_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_shots, &n_shots_val, sizeof(int64_t)));
  int64_t const shot_numel_val = ny_val * nx_val;
  gpuErrchk(cudaMemcpyToSymbol(shot_numel, &shot_numel_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y0, &pml_y0_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_y1, &pml_y1_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x0, &pml_x0_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(pml_x1, &pml_x1_val, sizeof(int64_t)));
  gpuErrchk(cudaMemcpyToSymbol(ca0_batched, &ca0_batched_val, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(cb0_batched, &cb0_batched_val, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(cq0_batched, &cq0_batched_val, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(dca_batched, &dca_batched_val, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(dcb_batched, &dcb_batched_val, sizeof(bool)));
  gpuErrchk(cudaMemcpyToSymbol(dcq_batched, &dcq_batched_val, sizeof(bool)));

  // Setup grid and block dimensions
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(
      (nx_val - 2 * FD_PAD + dimBlock.x - 1) / dimBlock.x,
      (ny_val - 2 * FD_PAD + dimBlock.y - 1) / dimBlock.y,
      n_shots_val);

  dim3 dimBlock_sources(32, 1, 1);
  dim3 dimGrid_sources(
      (n_sources_per_shot_val + dimBlock_sources.x - 1) / dimBlock_sources.x,
      n_shots_val, 1);

  dim3 dimBlock_receivers(32, 1, 1);
  dim3 dimGrid_receivers(
      (n_receivers_per_shot_val + dimBlock_receivers.x - 1) / dimBlock_receivers.x,
      n_shots_val, 1);

  // Calculate shared memory size
  size_t shmem_size_h = 0;
  size_t shmem_size_e = 0;
#if FD_PAD > 1
  int64_t const tile_w = (int64_t)dimBlock.x + 2 * (int64_t)FD_PAD;
  int64_t const tile_h = (int64_t)dimBlock.y + 2 * (int64_t)FD_PAD;
  int64_t const tile_numel = tile_w * tile_h;
  
  // H kernels need 2 tiles (Ey_bg + Ey_sc)
  shmem_size_h = 2 * tile_numel * sizeof(TIDE_DTYPE);
  
  // E kernel needs 4 tiles (Hx_bg + Hz_bg + Hx_sc + Hz_sc)
  shmem_size_e = 4 * tile_numel * sizeof(TIDE_DTYPE);
#endif

  // Time stepping loop
  for (int64_t t = 0; t < nt; ++t) {
    // Add sources (background gets f, scattered gets f_sc)
    if (n_sources_per_shot_val > 0) {
      add_sources_ey_both<<<dimGrid_sources, dimBlock_sources>>>(
          ey_bg, ey_sc,
          f + t * n_shots_val * n_sources_per_shot_val,
          f_sc + t * n_shots_val * n_sources_per_shot_val,
          sources_i);
    }

    // Update H fields
    // First update background H (computes CPML-corrected derivatives)
    forward_kernel_h_bg<<<dimGrid, dimBlock, shmem_size_h>>>(
        cq0, ey_bg, hx_bg, hz_bg,
        m_ey_x_bg, m_ey_z_bg,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    // Then update scattered H (uses CPML-corrected background derivatives)
    forward_kernel_h_sc<<<dimGrid, dimBlock, shmem_size_h>>>(
        cq0, dcq, ey_bg, ey_sc, hx_sc, hz_sc,
        m_ey_x_bg, m_ey_z_bg, m_ey_x_sc, m_ey_z_sc,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    // Update E fields
    // First update background E (computes CPML-corrected curl)
    forward_kernel_e_bg<<<dimGrid, dimBlock, shmem_size_e>>>(
        ca0, cb0, hx_bg, hz_bg, ey_bg,
        m_hx_z_bg, m_hz_x_bg,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    // Then update scattered E (uses CPML-corrected background curl)
    forward_kernel_e_sc<<<dimGrid, dimBlock, shmem_size_e>>>(
        ca0, cb0, dca, dcb,
        hx_bg, hz_bg, hx_sc, hz_sc, ey_bg, ey_sc,
        m_hx_z_bg, m_hz_x_bg, m_hx_z_sc, m_hz_x_sc,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    // Record receivers
    if (n_receivers_per_shot_val > 0) {
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r_bg + t * n_shots_val * n_receivers_per_shot_val,
          ey_bg, receivers_i);
      
      record_receivers_ey<<<dimGrid_receivers, dimBlock_receivers>>>(
          r_sc + t * n_shots_val * n_receivers_per_shot_val,
          ey_sc, receivers_i);
    }
  }

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

} // extern "C"
