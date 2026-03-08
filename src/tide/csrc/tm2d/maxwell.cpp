/*
 * Maxwell wave equation propagator (CPU implementation)
 *
 * This file contains the CPU implementation of the 2D TM Maxwell equations
 * propagator with complete Adjoint State Method (ASM) support for gradient
 * computation.
 *
 * TM mode fields: Ey (electric), Hx, Hz (magnetic)
 *
 * Adjoint State Method for Maxwell TM equations:
 * ================================================
 * Forward equations (discrete):
 *   E_y^{n+1} = C_a * E_y^n + C_b * (∂H_z/∂x - ∂H_x/∂z)
 *   H_x^{n+1/2} = H_x^{n-1/2} - C_q * ∂E_y/∂z
 *   H_z^{n+1/2} = H_z^{n-1/2} + C_q * ∂E_y/∂x
 *
 * Adjoint equations (time-reversed):
 *   λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * (∂λ_Hz/∂x - ∂λ_Hx/∂z) +
 * residual_injection λ_Hx^{n-1/2} = λ_Hx^{n+1/2} - C_b * ∂λ_Ey/∂z λ_Hz^{n-1/2}
 * = λ_Hz^{n+1/2} + C_b * ∂λ_Ey/∂x
 *
 * Model gradients:
 *   ∂J/∂C_a = Σ_n E_y^n * λ_Ey^{n+1}
 *   ∂J/∂C_b = Σ_n curl_H^n * λ_Ey^{n+1}
 *
 * Storage requirements:
 *   - ey_store: E_y field at each step_ratio time step [nt/step_ratio, n_shots,
 * ny, nx]
 *   - curl_h_store: (∂H_z/∂x - ∂H_x/∂z) at each step_ratio time step
 * [nt/step_ratio, n_shots, ny, nx]
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common_cpu.h"
#include "storage_utils.h"
#define CAT_I(name, accuracy, dtype, device)                                   \
  maxwell_tm_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, cpu)

static inline int64_t tide_idx_2d(int64_t y, int64_t x, int64_t nx) {
  return y * nx + x;
}

static inline int64_t tide_idx_2d_shot(int64_t shot, int64_t y, int64_t x,
                                       int64_t shot_numel, int64_t nx) {
  return shot * shot_numel + tide_idx_2d(y, x, nx);
}

template <typename T> static inline T tide_max(T a, T b) {
  return a > b ? a : b;
}

template <typename T> static inline T tide_min(T a, T b) {
  return a < b ? a : b;
}

static constexpr TIDE_DTYPE kEp0 = (TIDE_DTYPE)8.8541878128e-12;

typedef uint16_t tide_bfloat16;

static inline tide_bfloat16 tide_float_to_bf16(float value) {
  union {
    float f;
    uint32_t u;
  } tmp;
  tmp.f = value;
  uint32_t lsb = (tmp.u >> 16) & 1u;
  tmp.u += 0x7FFFu + lsb;
  return (tide_bfloat16)(tmp.u >> 16);
}

static inline float tide_bf16_to_float(tide_bfloat16 value) {
  union {
    uint32_t u;
    float f;
  } tmp;
  tmp.u = ((uint32_t)value) << 16;
  return tmp.f;
}

template <typename T>
static void add_sources_ey(T *__restrict const ey,
                           T const *__restrict const f,
                           int64_t const *__restrict const sources_i,
                           int64_t const n_shots, int64_t const shot_numel,
                           int64_t const n_sources_per_shot) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t source_idx = 0; source_idx < n_sources_per_shot;
         ++source_idx) {
      int64_t k = shot_idx * n_sources_per_shot + source_idx;
      if (sources_i[k] >= 0) {
        ey[shot_idx * shot_numel + sources_i[k]] += f[k];
      }
    }
  }
}

template <typename T>
static void subtract_sources_ey(T *__restrict const ey,
                                T const *__restrict const f,
                                int64_t const *__restrict const sources_i,
                                int64_t const n_shots, int64_t const shot_numel,
                                int64_t const n_sources_per_shot) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t source_idx = 0; source_idx < n_sources_per_shot;
         ++source_idx) {
      int64_t k = shot_idx * n_sources_per_shot + source_idx;
      if (sources_i[k] >= 0) {
        ey[shot_idx * shot_numel + sources_i[k]] -= f[k];
      }
    }
  }
}

template <typename T>
static void record_receivers_ey(T *__restrict const r,
                                T const *__restrict const ey,
                                int64_t const *__restrict const receivers_i,
                                int64_t const n_shots, int64_t const shot_numel,
                                int64_t const n_receivers_per_shot) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t receiver_idx = 0; receiver_idx < n_receivers_per_shot;
         ++receiver_idx) {
      int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
      if (receivers_i[k] >= 0) {
        r[k] = ey[shot_idx * shot_numel + receivers_i[k]];
      }
    }
  }
}

template <typename T>
static void gather_boundary_3_cpu(
    T const *__restrict const ey,
    T const *__restrict const hx,
    T const *__restrict const hz, T *__restrict const bey,
    T *__restrict const bhx, T *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel, int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      bey[store_offset] = ey[field_offset];
      bhx[store_offset] = hx[field_offset];
      bhz[store_offset] = hz[field_offset];
    }
  }
}

template <typename T>
static void
scatter_boundary_cpu(T *__restrict const field,
                     T const *__restrict const store,
                     int64_t const *__restrict const boundary_indices,
                     int64_t const boundary_numel, int64_t const n_shots,
                     int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      field[field_offset] = store[store_offset];
    }
  }
}

static void
scatter_boundary_2_cpu(TIDE_DTYPE *__restrict const hx,
                       TIDE_DTYPE *__restrict const hz,
                       TIDE_DTYPE const *__restrict const bhx,
                       TIDE_DTYPE const *__restrict const bhz,
                       int64_t const *__restrict const boundary_indices,
                       int64_t const boundary_numel, int64_t const n_shots,
                       int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      hx[field_offset] = bhx[store_offset];
      hz[field_offset] = bhz[store_offset];
    }
  }
}

static void gather_boundary_3_cpu_bf16(
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, tide_bfloat16 *__restrict const bey,
    tide_bfloat16 *__restrict const bhx, tide_bfloat16 *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel, int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      bey[store_offset] = tide_float_to_bf16((float)ey[field_offset]);
      bhx[store_offset] = tide_float_to_bf16((float)hx[field_offset]);
      bhz[store_offset] = tide_float_to_bf16((float)hz[field_offset]);
    }
  }
}

static void
scatter_boundary_cpu_bf16(TIDE_DTYPE *__restrict const field,
                          tide_bfloat16 const *__restrict const store,
                          int64_t const *__restrict const boundary_indices,
                          int64_t const boundary_numel, int64_t const n_shots,
                          int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      field[field_offset] = (TIDE_DTYPE)tide_bf16_to_float(store[store_offset]);
    }
  }
}

static void
scatter_boundary_2_cpu_bf16(TIDE_DTYPE *__restrict const hx,
                            TIDE_DTYPE *__restrict const hz,
                            tide_bfloat16 const *__restrict const bhx,
                            tide_bfloat16 const *__restrict const bhz,
                            int64_t const *__restrict const boundary_indices,
                            int64_t const boundary_numel, int64_t const n_shots,
                            int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      hx[field_offset] = (TIDE_DTYPE)tide_bf16_to_float(bhx[store_offset]);
      hz[field_offset] = (TIDE_DTYPE)tide_bf16_to_float(bhz[store_offset]);
    }
  }
}

static inline void *boundary_store_ptr(void *store_1, void *store_3,
                                       int64_t storage_mode, int64_t step_idx,
                                       int64_t step_elems, size_t elem_size) {
  size_t const offset_bytes = (size_t)step_idx * (size_t)step_elems * elem_size;
  if (storage_mode == STORAGE_DEVICE) {
    return (uint8_t *)store_1 + offset_bytes;
  }
  if (storage_mode == STORAGE_CPU && store_3 != NULL) {
    return (uint8_t *)store_3 + offset_bytes;
  }
  return (uint8_t *)store_1;
}

/*
 * work_* buffers are fully overwritten in [FD_PAD, ny-FD_PAD] x [FD_PAD,
 * nx-FD_PAD]. Only derivative halos need explicit zeroing each backward step.
 */

#include "maxwell_tm_core.cuh"
using namespace tide;


#include "maxwell_tm_cpu_instantiations.inc"
