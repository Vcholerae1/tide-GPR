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
 *   λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * (∂λ_Hz/∂x - ∂λ_Hx/∂z) + residual_injection
 *   λ_Hx^{n-1/2} = λ_Hx^{n+1/2} - C_b * ∂λ_Ey/∂z
 *   λ_Hz^{n-1/2} = λ_Hz^{n+1/2} + C_b * ∂λ_Ey/∂x
 *
 * Model gradients:
 *   ∂J/∂C_a = Σ_n E_y^n * λ_Ey^{n+1}
 *   ∂J/∂C_b = Σ_n curl_H^n * λ_Ey^{n+1}
 *
 * Storage requirements:
 *   - ey_store: E_y field at each step_ratio time step [nt/step_ratio, n_shots, ny, nx]
 *   - curl_h_store: (∂H_z/∂x - ∂H_x/∂z) at each step_ratio time step [nt/step_ratio, n_shots, ny, nx]
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common_cpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#define CAT_I(name, accuracy, dtype, device) \
  maxwell_tm_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) \
  CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, cpu)

// 2D indexing macros
#define IDX(y, x) ((y) * nx + (x))
#define IDX_SHOT(shot, y, x) ((shot) * shot_numel + (y) * nx + (x))

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Vacuum permittivity (F/m) to convert dL/d(epsilon_abs) -> dL/d(epsilon_r)
#define EP0 ((TIDE_DTYPE)8.8541878128e-12)

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

// FP8 E4M3 format (1 sign bit, 4 exponent bits, 3 mantissa bits)
// Dynamic range: ~10^-5 to ~448
typedef uint8_t tide_fp8_e4m3;

static inline tide_fp8_e4m3 tide_float_to_fp8_e4m3(float value) {
  if (value == 0.0f) {
    return 0;
  }

  union {
    float f;
    uint32_t u;
  } tmp;
  tmp.f = value;

  uint8_t sign = (tmp.u >> 31) ? 0x80 : 0;
  float ax = fabsf(value);

  // Handle infinity/NaN - saturate to max representable
  if (!isfinite(ax)) {
    return (uint8_t)(sign | 0x7F);
  }

  // Use frexpf to extract mantissa and exponent
  int exp;
  float m = frexpf(ax, &exp);  // ax = m * 2^exp, m in [0.5, 1)
  int e = exp - 1;
  int exp_field = e + 7;  // Bias of 7 for E4M3
  int mant = 0;

  if (exp_field <= 0) {
    // Subnormal - compute mantissa directly
    // Smallest normal: 2^-6, subnormal range: 2^-9 to 2^-6
    mant = (int)roundf(ax * 512.0f);
    if (mant <= 0) {
      return sign;  // Underflow to zero
    }
    if (mant > 7) {
      mant = 7;  // Clamp to max subnormal
    }
    exp_field = 0;
  } else if (exp_field >= 0xF) {
    // Overflow - saturate to max normal
    exp_field = 0xE;
    mant = 7;
  } else {
    // Normal number
    float frac = m * 2.0f - 1.0f;  // Extract fractional part
    mant = (int)roundf(frac * 8.0f);
    if (mant == 8) {
      // Rounding overflow - increment exponent
      mant = 0;
      exp_field += 1;
      if (exp_field >= 0xF) {
        exp_field = 0xE;
        mant = 7;
      }
    }
  }

  return (uint8_t)(sign | ((uint8_t)exp_field << 3) | (uint8_t)(mant & 0x7));
}

static inline float tide_fp8_e4m3_to_float(tide_fp8_e4m3 value) {
  if (value == 0) {
    return 0.0f;
  }

  int sign = value & 0x80;
  int exp_field = (value >> 3) & 0xF;
  int mant = value & 0x7;
  float val;

  if (exp_field == 0) {
    // Subnormal
    float frac = (float)mant / 8.0f;
    val = ldexpf(frac, -6);  // 2^-6 * (mant/8)
  } else {
    // Normal
    float frac = 1.0f + (float)mant / 8.0f;
    val = ldexpf(frac, exp_field - 7);
  }

  return sign ? -val : val;
}

// Field access macros for stencil operations
#define EY(dy, dx) ey[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define HX(dy, dx) hx[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define HZ(dy, dx) hz[IDX_SHOT(shot_idx, y + (dy), x + (dx))]

// Adjoint field access macros
#define LAMBDA_EY(dy, dx) lambda_ey[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define LAMBDA_HX(dy, dx) lambda_hx[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define LAMBDA_HZ(dy, dx) lambda_hz[IDX_SHOT(shot_idx, y + (dy), x + (dx))]

// Material parameter access macros
#define CA(dy, dx) (ca_batched ? ca[IDX_SHOT(shot_idx, y + (dy), x + (dx))] : ca[IDX(y + (dy), x + (dx))])
#define CB(dy, dx) (cb_batched ? cb[IDX_SHOT(shot_idx, y + (dy), x + (dx))] : cb[IDX(y + (dy), x + (dx))])
#define CQ(dy, dx) (cq_batched ? cq[IDX_SHOT(shot_idx, y + (dy), x + (dx))] : cq[IDX(y + (dy), x + (dx))])

// PML memory variable macros
#define M_EY_X(dy, dx) m_ey_x[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define M_EY_Z(dy, dx) m_ey_z[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define M_HX_Z(dy, dx) m_hx_z[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define M_HZ_X(dy, dx) m_hz_x[IDX_SHOT(shot_idx, y + (dy), x + (dx))]

// Adjoint PML memory variable macros
#define M_LAMBDA_EY_X(dy, dx) m_lambda_ey_x[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define M_LAMBDA_EY_Z(dy, dx) m_lambda_ey_z[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define M_LAMBDA_HX_Z(dy, dx) m_lambda_hx_z[IDX_SHOT(shot_idx, y + (dy), x + (dx))]
#define M_LAMBDA_HZ_X(dy, dx) m_lambda_hz_x[IDX_SHOT(shot_idx, y + (dy), x + (dx))]

static void convert_grad_ca_cb_to_eps_sigma(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const grad_ca,
    TIDE_DTYPE const *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE const dt,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    bool const ca_batched,
    bool const cb_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {
  if ((grad_eps == NULL && grad_sigma == NULL) || (!ca_requires_grad && !cb_requires_grad)) {
    return;
  }

  int64_t const shot_numel = ny * nx;
  int64_t const out_shots = ca_batched ? n_shots : 1;
  TIDE_DTYPE const inv_dt = (TIDE_DTYPE)1 / dt;

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < out_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    int64_t const out_offset = ca_batched ? shot_offset : 0;
    int64_t const ca_offset = ca_batched ? shot_offset : 0;
    int64_t const cb_offset = cb_batched ? shot_offset : 0;
    TIDE_DTYPE const *__restrict const ca_ptr = ca + ca_offset;
    TIDE_DTYPE const *__restrict const cb_ptr = cb + cb_offset;
TIDE_OMP_SIMD_COLLAPSE2
    for (int64_t y = 0; y < ny; ++y) {
      for (int64_t x = 0; x < nx; ++x) {
        int64_t const idx = IDX(y, x);
        int64_t const out_idx = out_offset + idx;

        TIDE_DTYPE const ca_val = ca_ptr[idx];
        TIDE_DTYPE const cb_val = cb_ptr[idx];

        TIDE_DTYPE const grad_ca_val =
            (ca_requires_grad && grad_ca != NULL) ? grad_ca[out_idx] : (TIDE_DTYPE)0;
        TIDE_DTYPE const grad_cb_val =
            (cb_requires_grad && grad_cb != NULL) ? grad_cb[out_idx] : (TIDE_DTYPE)0;

        TIDE_DTYPE const cb_sq = cb_val * cb_val;
        TIDE_DTYPE const dca_de = ((TIDE_DTYPE)1 - ca_val) * cb_val * inv_dt;
        TIDE_DTYPE const dcb_de = -cb_sq * inv_dt;
        TIDE_DTYPE const dca_ds = -((TIDE_DTYPE)0.5) * ((TIDE_DTYPE)1 + ca_val) * cb_val;
        TIDE_DTYPE const dcb_ds = -((TIDE_DTYPE)0.5) * cb_sq;

        if (grad_eps != NULL) {
          TIDE_DTYPE const grad_e = grad_ca_val * dca_de + grad_cb_val * dcb_de;
          grad_eps[out_idx] = grad_e * EP0;
        }
        if (grad_sigma != NULL) {
          grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
        }
      }
    }
  }
}


static void add_sources_ey(
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_sources_per_shot) {
  
  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
      int64_t k = shot_idx * n_sources_per_shot + source_idx;
      if (sources_i[k] >= 0) {
        ey[shot_idx * shot_numel + sources_i[k]] += f[k];
      }
    }
  }
}

static void subtract_sources_ey(
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_sources_per_shot) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t source_idx = 0; source_idx < n_sources_per_shot; ++source_idx) {
      int64_t k = shot_idx * n_sources_per_shot + source_idx;
      if (sources_i[k] >= 0) {
        ey[shot_idx * shot_numel + sources_i[k]] -= f[k];
      }
    }
  }
}


static void record_receivers_ey(
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE const *__restrict const ey,
    int64_t const *__restrict const receivers_i,
    int64_t const n_shots,
    int64_t const shot_numel,
    int64_t const n_receivers_per_shot) {
  
  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t receiver_idx = 0; receiver_idx < n_receivers_per_shot; ++receiver_idx) {
      int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
      if (receivers_i[k] >= 0) {
        r[k] = ey[shot_idx * shot_numel + receivers_i[k]];
      }
    }
  }
}

static void gather_boundary_3_cpu(
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const bey,
    TIDE_DTYPE *__restrict const bhx,
    TIDE_DTYPE *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
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

static void scatter_boundary_cpu(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const store,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
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

static void scatter_boundary_2_cpu(
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE const *__restrict const bhx,
    TIDE_DTYPE const *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
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
    TIDE_DTYPE const *__restrict const hz,
    tide_bfloat16 *__restrict const bey,
    tide_bfloat16 *__restrict const bhx,
    tide_bfloat16 *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
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

static void scatter_boundary_cpu_bf16(
    TIDE_DTYPE *__restrict const field,
    tide_bfloat16 const *__restrict const store,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
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

static void scatter_boundary_2_cpu_bf16(
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    tide_bfloat16 const *__restrict const bhx,
    tide_bfloat16 const *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
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

// FP8 boundary storage functions
static void gather_boundary_3_cpu_fp8(
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    tide_fp8_e4m3 *__restrict const bey,
    tide_fp8_e4m3 *__restrict const bhx,
    tide_fp8_e4m3 *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      bey[store_offset] = tide_float_to_fp8_e4m3((float)ey[field_offset]);
      bhx[store_offset] = tide_float_to_fp8_e4m3((float)hx[field_offset]);
      bhz[store_offset] = tide_float_to_fp8_e4m3((float)hz[field_offset]);
    }
  }
}

static void scatter_boundary_cpu_fp8(
    TIDE_DTYPE *__restrict const field,
    tide_fp8_e4m3 const *__restrict const store,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      field[field_offset] = (TIDE_DTYPE)tide_fp8_e4m3_to_float(store[store_offset]);
    }
  }
}

static void scatter_boundary_2_cpu_fp8(
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    tide_fp8_e4m3 const *__restrict const bhx,
    tide_fp8_e4m3 const *__restrict const bhz,
    int64_t const *__restrict const boundary_indices,
    int64_t const boundary_numel,
    int64_t const n_shots,
    int64_t const shot_numel) {

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
TIDE_OMP_SIMD
    for (int64_t bi = 0; bi < boundary_numel; ++bi) {
      int64_t const grid_idx = boundary_indices[bi];
      int64_t const field_offset = shot_idx * shot_numel + grid_idx;
      int64_t const store_offset = shot_idx * boundary_numel + bi;
      hx[field_offset] = (TIDE_DTYPE)tide_fp8_e4m3_to_float(bhx[store_offset]);
      hz[field_offset] = (TIDE_DTYPE)tide_fp8_e4m3_to_float(bhz[store_offset]);
    }
  }
}

static inline void *boundary_store_ptr(
    void *store_1,
    void *store_3,
    int64_t storage_mode,
    int64_t step_idx,
    int64_t step_elems,
    size_t elem_size) {
  size_t const offset_bytes =
      (size_t)step_idx * (size_t)step_elems * elem_size;
  if (storage_mode == STORAGE_DEVICE) {
    return (uint8_t *)store_1 + offset_bytes;
  }
  if (storage_mode == STORAGE_CPU && store_3 != NULL) {
    return (uint8_t *)store_3 + offset_bytes;
  }
  return (uint8_t *)store_1;
}


static void forward_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const m_ey_z,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const cq_batched) {

  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0h, pml_y1h, ny - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0h, pml_x1h, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const cq_ptr =
        cq_batched ? (cq + shot_offset) : cq;
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            TIDE_DTYPE const cq_val = cq_ptr[idx];

            if (y < ny - FD_PAD) {
              TIDE_DTYPE dey_dz = DIFFYH1(EY);

              if (pml_y != 1) {
                M_EY_Z(0, 0) = byh[y] * M_EY_Z(0, 0) + ayh[y] * dey_dz;
                dey_dz = dey_dz / kyh[y] + M_EY_Z(0, 0);
              }

              HX(0, 0) -= cq_val * dey_dz;
            }

            if (x < nx - FD_PAD) {
              TIDE_DTYPE dey_dx = DIFFXH1(EY);

              if (pml_x != 1) {
                M_EY_X(0, 0) = bxh[x] * M_EY_X(0, 0) + axh[x] * dey_dx;
                dey_dx = dey_dx / kxh[x] + M_EY_X(0, 0);
              }

              HZ(0, 0) += cq_val * dey_dx;
            }
          }
        }
      }
    }
  }
}


/*
 * Forward E kernel with optional storage for gradient computation
 *
 * When ca_requires_grad or cb_requires_grad is true, stores:
 *   - ey_store: E_y field before update (needed for grad_ca)
 *   - curl_h_store: (dHz/dx - dHx/dz) (needed for grad_cb)
 */
static void forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cb_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const curl_h_store) {

  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const ca_ptr =
        ca_batched ? (ca + shot_offset) : ca;
    TIDE_DTYPE const *__restrict const cb_ptr =
        cb_batched ? (cb + shot_offset) : cb;
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            int64_t const store_idx = shot_offset + idx;
            TIDE_DTYPE const ca_val = ca_ptr[idx];
            TIDE_DTYPE const cb_val = cb_ptr[idx];

            TIDE_DTYPE dhz_dx = DIFFX1(HZ);
            TIDE_DTYPE dhx_dz = DIFFY1(HX);

            if (pml_x != 1) {
              M_HZ_X(0, 0) = bx[x] * M_HZ_X(0, 0) + ax[x] * dhz_dx;
              dhz_dx = dhz_dx / kx[x] + M_HZ_X(0, 0);
            }

            if (pml_y != 1) {
              M_HX_Z(0, 0) = by[y] * M_HX_Z(0, 0) + ay[y] * dhx_dz;
              dhx_dz = dhx_dz / ky[y] + M_HX_Z(0, 0);
            }

            TIDE_DTYPE curl_h = dhz_dx - dhx_dz;

            if (ca_requires_grad && ey_store != NULL) {
              ey_store[store_idx] = EY(0, 0);
            }
            if (cb_requires_grad && curl_h_store != NULL) {
              curl_h_store[store_idx] = curl_h;
            }

            EY(0, 0) = ca_val * EY(0, 0) + cb_val * curl_h;
          }
        }
      }
    }
  }
}

static void forward_kernel_e_with_storage_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cb_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    tide_bfloat16 *__restrict const ey_store,
    tide_bfloat16 *__restrict const curl_h_store) {

  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const ca_ptr =
        ca_batched ? (ca + shot_offset) : ca;
    TIDE_DTYPE const *__restrict const cb_ptr =
        cb_batched ? (cb + shot_offset) : cb;
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            int64_t const store_idx = shot_offset + idx;
            TIDE_DTYPE const ca_val = ca_ptr[idx];
            TIDE_DTYPE const cb_val = cb_ptr[idx];

            TIDE_DTYPE dhz_dx = DIFFX1(HZ);
            TIDE_DTYPE dhx_dz = DIFFY1(HX);

            if (pml_x != 1) {
              M_HZ_X(0, 0) = bx[x] * M_HZ_X(0, 0) + ax[x] * dhz_dx;
              dhz_dx = dhz_dx / kx[x] + M_HZ_X(0, 0);
            }

            if (pml_y != 1) {
              M_HX_Z(0, 0) = by[y] * M_HX_Z(0, 0) + ay[y] * dhx_dz;
              dhx_dz = dhx_dz / ky[y] + M_HX_Z(0, 0);
            }

            TIDE_DTYPE curl_h = dhz_dx - dhx_dz;
            if (ca_requires_grad && ey_store != NULL) {
              ey_store[store_idx] = tide_float_to_bf16((float)EY(0, 0));
            }
            if (cb_requires_grad && curl_h_store != NULL) {
              curl_h_store[store_idx] = tide_float_to_bf16((float)curl_h);
            }

            EY(0, 0) = ca_val * EY(0, 0) + cb_val * curl_h;
          }
        }
      }
    }
  }
}

static void forward_kernel_e_with_storage_fp8(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hz_x,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cb_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    tide_fp8_e4m3 *__restrict const ey_store,
    tide_fp8_e4m3 *__restrict const curl_h_store) {

  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const ca_ptr =
        ca_batched ? (ca + shot_offset) : ca;
    TIDE_DTYPE const *__restrict const cb_ptr =
        cb_batched ? (cb + shot_offset) : cb;
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            int64_t const store_idx = shot_offset + idx;
            TIDE_DTYPE const ca_val = ca_ptr[idx];
            TIDE_DTYPE const cb_val = cb_ptr[idx];

            TIDE_DTYPE dhz_dx = DIFFX1(HZ);
            TIDE_DTYPE dhx_dz = DIFFY1(HX);

            if (pml_x != 1) {
              M_HZ_X(0, 0) = bx[x] * M_HZ_X(0, 0) + ax[x] * dhz_dx;
              dhz_dx = dhz_dx / kx[x] + M_HZ_X(0, 0);
            }

            if (pml_y != 1) {
              M_HX_Z(0, 0) = by[y] * M_HX_Z(0, 0) + ay[y] * dhx_dz;
              dhx_dz = dhx_dz / ky[y] + M_HX_Z(0, 0);
            }

            TIDE_DTYPE curl_h = dhz_dx - dhx_dz;
            if (ca_requires_grad && ey_store != NULL) {
              ey_store[store_idx] = tide_float_to_fp8_e4m3((float)EY(0, 0));
            }
            if (cb_requires_grad && curl_h_store != NULL) {
              curl_h_store[store_idx] = tide_float_to_fp8_e4m3((float)curl_h);
            }

            EY(0, 0) = ca_val * EY(0, 0) + cb_val * curl_h;
          }
        }
      }
    }
  }
}


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
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const r,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const n_sources_per_shot,
    int64_t const n_receivers_per_shot,
    int64_t const step_ratio,
    bool const ca_batched,
    bool const cb_batched,
    bool const cq_batched,
    int64_t const start_t,
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const n_threads,
    int64_t const device /* unused for CPU */) {
  
  (void)device;
  (void)dt;
  (void)step_ratio;
#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif
  
  int64_t const shot_numel = ny * nx;

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h(
        cq, ey, hx, hz, m_ey_x, m_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh,
        rdy, rdx,
        n_shots, ny, nx, shot_numel,
        pml_y0, pml_y1, pml_x0, pml_x1,
        cq_batched);

    forward_kernel_e_with_storage(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh,
        rdy, rdx,
        n_shots, ny, nx, shot_numel,
        pml_y0, pml_y1, pml_x0, pml_x1,
        ca_batched, cb_batched,
        false, false,  // No storage for standard forward
        NULL, NULL);

    if (n_sources_per_shot > 0) {
      add_sources_ey(
          ey, f + t * n_shots * n_sources_per_shot, sources_i,
          n_shots, shot_numel, n_sources_per_shot);
    }

    if (n_receivers_per_shot > 0) {
      record_receivers_ey(
          r + t * n_shots * n_receivers_per_shot,
          ey, receivers_i,
          n_shots, shot_numel, n_receivers_per_shot);
    }
  }
#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}


/*
 * Forward with storage for backward pass
 * 
 * This function performs forward propagation while storing the values
 * needed for gradient computation in the backward pass.
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
    TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE *const r,
    TIDE_DTYPE *const ey_store_1,
    void *const ey_store_3,
    char const *const *const ey_filenames,
    TIDE_DTYPE *const curl_store_1,
    void *const curl_store_3,
    char const *const *const curl_filenames,
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
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
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const n_threads,
    int64_t const device /* unused for CPU */) {
  
  (void)device;
  (void)dt;
#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  bool const storage_fp8 = (shot_bytes_uncomp == shot_numel * 1);
  bool const storage_bf16 = (shot_bytes_uncomp == shot_numel * 2);

  FILE **fp_ey = NULL;
  FILE **fp_curl = NULL;
  if (storage_mode == STORAGE_DISK) {
    if (ca_requires_grad) {
      fp_ey = (FILE **)malloc((size_t)n_shots * sizeof(FILE *));
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        fp_ey[shot] = fopen(ey_filenames[shot], "wb");
      }
    }
    if (cb_requires_grad) {
      fp_curl = (FILE **)malloc((size_t)n_shots * sizeof(FILE *));
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        fp_curl[shot] = fopen(curl_filenames[shot], "wb");
      }
    }
  }

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h(
        cq, ey, hx, hz, m_ey_x, m_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh,
        rdy, rdx,
        n_shots, ny, nx, shot_numel,
        pml_y0, pml_y1, pml_x0, pml_x1,
        cq_batched);

    bool const store_step = ((t % step_ratio) == 0);
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    int64_t const step_idx = t / step_ratio;

    int64_t const store_offset =
        (storage_mode == STORAGE_DEVICE ? step_idx * store_size : 0);

    if (storage_fp8) {
      tide_fp8_e4m3 *const ey_store_1_t =
          (tide_fp8_e4m3 *)ey_store_1 + store_offset;
      tide_fp8_e4m3 *const curl_store_1_t =
          (tide_fp8_e4m3 *)curl_store_1 + store_offset;

      forward_kernel_e_with_storage_fp8(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
          ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh,
          rdy, rdx,
          n_shots, ny, nx, shot_numel,
          pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cb_batched,
          store_ey,
          store_curl,
          store_ey ? ey_store_1_t : NULL,
          store_curl ? curl_store_1_t : NULL);

      if (store_ey && storage_mode == STORAGE_DISK) {
        for (int64_t shot = 0; shot < n_shots; ++shot) {
          storage_save_snapshot_cpu(
              (void *)(ey_store_1_t + shot * shot_numel), fp_ey[shot],
              storage_mode, step_idx, (size_t)shot_bytes_uncomp);
        }
      }
      if (store_curl && storage_mode == STORAGE_DISK) {
        for (int64_t shot = 0; shot < n_shots; ++shot) {
          storage_save_snapshot_cpu(
              (void *)(curl_store_1_t + shot * shot_numel), fp_curl[shot],
              storage_mode, step_idx, (size_t)shot_bytes_uncomp);
        }
      }
    } else if (storage_bf16) {
      tide_bfloat16 *const ey_store_1_t =
          (tide_bfloat16 *)ey_store_1 + store_offset;
      tide_bfloat16 *const curl_store_1_t =
          (tide_bfloat16 *)curl_store_1 + store_offset;

      forward_kernel_e_with_storage_bf16(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
          ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh,
          rdy, rdx,
          n_shots, ny, nx, shot_numel,
          pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cb_batched,
          store_ey,
          store_curl,
          store_ey ? ey_store_1_t : NULL,
          store_curl ? curl_store_1_t : NULL);

      if (store_ey && storage_mode == STORAGE_DISK) {
        for (int64_t shot = 0; shot < n_shots; ++shot) {
          storage_save_snapshot_cpu(
              (void *)(ey_store_1_t + shot * shot_numel), fp_ey[shot],
              storage_mode, step_idx, (size_t)shot_bytes_uncomp);
        }
      }
      if (store_curl && storage_mode == STORAGE_DISK) {
        for (int64_t shot = 0; shot < n_shots; ++shot) {
          storage_save_snapshot_cpu(
              (void *)(curl_store_1_t + shot * shot_numel), fp_curl[shot],
              storage_mode, step_idx, (size_t)shot_bytes_uncomp);
        }
      }
    } else {
      TIDE_DTYPE *const ey_store_1_t =
          ey_store_1 + store_offset;
      TIDE_DTYPE *const curl_store_1_t =
          curl_store_1 + store_offset;

      forward_kernel_e_with_storage(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
          ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh,
          rdy, rdx,
          n_shots, ny, nx, shot_numel,
          pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cb_batched,
          store_ey,
          store_curl,
          store_ey ? ey_store_1_t : NULL,
          store_curl ? curl_store_1_t : NULL);

      if (store_ey && storage_mode == STORAGE_DISK) {
        for (int64_t shot = 0; shot < n_shots; ++shot) {
          storage_save_snapshot_cpu(
              (void *)(ey_store_1_t + shot * shot_numel), fp_ey[shot],
              storage_mode, step_idx, (size_t)shot_bytes_uncomp);
        }
      }
      if (store_curl && storage_mode == STORAGE_DISK) {
        for (int64_t shot = 0; shot < n_shots; ++shot) {
          storage_save_snapshot_cpu(
              (void *)(curl_store_1_t + shot * shot_numel), fp_curl[shot],
              storage_mode, step_idx, (size_t)shot_bytes_uncomp);
        }
      }
    }

    if (n_sources_per_shot > 0) {
      add_sources_ey(
          ey, f + t * n_shots * n_sources_per_shot, sources_i,
          n_shots, shot_numel, n_sources_per_shot);
    }

    if (n_receivers_per_shot > 0) {
      record_receivers_ey(
          r + t * n_shots * n_receivers_per_shot,
          ey, receivers_i,
          n_shots, shot_numel, n_receivers_per_shot);
    }
  }

  if (fp_ey != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot) fclose(fp_ey[shot]);
    free(fp_ey);
  }
  if (fp_curl != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot) fclose(fp_curl[shot]);
    free(fp_curl);
  }
#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

/*
 * Backward kernel for adjoint λ_H fields update
 * 
 * Adjoint equations for H fields (time reversed, swap Cb and Cq roles):
 *   λ_Hx^{n-1/2} = λ_Hx^{n+1/2} - C_b * ∂λ_Ey/∂z
 *   λ_Hz^{n-1/2} = λ_Hz^{n+1/2} + C_b * ∂λ_Ey/∂x
 */
static void backward_kernel_lambda_h(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const lambda_hx,
    TIDE_DTYPE *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const cb_batched) {

  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = MAX(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = MAX(pml_x0, pml_x1 - 1);
  int64_t const pml_bounds_yh[] = {FD_PAD, pml_y0h, pml_y1h, ny - FD_PAD + 1};
  int64_t const pml_bounds_xh[] = {FD_PAD, pml_x0h, pml_x1h, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const cb_ptr =
        cb_batched ? (cb + shot_offset) : cb;
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_yh[pml_y]; y < pml_bounds_yh[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_xh[pml_x]; x < pml_bounds_xh[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            TIDE_DTYPE const cb_val = cb_ptr[idx];

            if (y < ny - FD_PAD) {
              TIDE_DTYPE d_lambda_ey_dz = DIFFYH1(LAMBDA_EY);

              if (pml_y != 1) {
                M_LAMBDA_EY_Z(0, 0) = byh[y] * M_LAMBDA_EY_Z(0, 0) + ayh[y] * d_lambda_ey_dz;
                d_lambda_ey_dz = d_lambda_ey_dz / kyh[y] + M_LAMBDA_EY_Z(0, 0);
              }

              LAMBDA_HX(0, 0) -= cb_val * d_lambda_ey_dz;
            }

            if (x < nx - FD_PAD) {
              TIDE_DTYPE d_lambda_ey_dx = DIFFXH1(LAMBDA_EY);

              if (pml_x != 1) {
                M_LAMBDA_EY_X(0, 0) = bxh[x] * M_LAMBDA_EY_X(0, 0) + axh[x] * d_lambda_ey_dx;
                d_lambda_ey_dx = d_lambda_ey_dx / kxh[x] + M_LAMBDA_EY_X(0, 0);
              }

              LAMBDA_HZ(0, 0) += cb_val * d_lambda_ey_dx;
            }
          }
        }
      }
    }
  }
}


/*
 * Backward kernel for adjoint λ_Ey field update with gradient accumulation
 * 
 * Adjoint equation for E field (time reversed, swap Cb and Cq roles):
 *   λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * (∂λ_Hz/∂x - ∂λ_Hx/∂z)
 *
 * Gradient accumulation:
 *   grad_ca += λ_Ey^{n+1} * E_y^n
 *   grad_cb += λ_Ey^{n+1} * curl_H^n
 *
 * Uses pml_bounds arrays to divide domain into 9 regions (3x3 grid):
 *   pml_y/pml_x == 0: Left/Top PML region
 *   pml_y/pml_x == 1: Interior region (where gradients are accumulated)
 *   pml_y/pml_x == 2: Right/Bottom PML region
 */
static void backward_kernel_lambda_e_with_grad(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cq_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    int64_t const step_ratio) {

  // PML region bounds arrays
  // pml_bounds[0] = FD_PAD (start of computational domain)
  // pml_bounds[1] = pml_y0 (start of interior region)
  // pml_bounds[2] = pml_y1 (end of interior region)
  // pml_bounds[3] = ny - FD_PAD + 1 (end of computational domain)
  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const ca_ptr =
        ca_batched ? (ca + shot_offset) : ca;
    TIDE_DTYPE const *__restrict const cq_ptr =
        cq_batched ? (cq + shot_offset) : cq;
    // Loop over 3x3 grid of regions
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            int64_t const store_idx = shot_offset + idx;
            TIDE_DTYPE const ca_val = ca_ptr[idx];
            TIDE_DTYPE const cq_val = cq_ptr[idx];

            // Compute d(λ_Hz)/dx at integer grid points
            TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LAMBDA_HZ);
            // Compute d(λ_Hx)/dz at integer grid points
            TIDE_DTYPE d_lambda_hx_dz = DIFFY1(LAMBDA_HX);

            // Apply adjoint CPML for d(λ_Hz)/dx (only in PML regions)
            if (pml_x != 1) {
              M_LAMBDA_HZ_X(0, 0) = bx[x] * M_LAMBDA_HZ_X(0, 0) + ax[x] * d_lambda_hz_dx;
              d_lambda_hz_dx = d_lambda_hz_dx / kx[x] + M_LAMBDA_HZ_X(0, 0);
            }

            // Apply adjoint CPML for d(λ_Hx)/dz (only in PML regions)
            if (pml_y != 1) {
              M_LAMBDA_HX_Z(0, 0) = by[y] * M_LAMBDA_HX_Z(0, 0) + ay[y] * d_lambda_hx_dz;
              d_lambda_hx_dz = d_lambda_hx_dz / ky[y] + M_LAMBDA_HX_Z(0, 0);
            }

            // curl_λH = d(λ_Hz)/dx - d(λ_Hx)/dz
            TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;

            // Store current λ_Ey before update (this is λ_Ey^{n+1})
            TIDE_DTYPE lambda_ey_curr = LAMBDA_EY(0, 0);

            // Update λ_Ey: λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * curl_λH
            LAMBDA_EY(0, 0) = ca_val * lambda_ey_curr + cq_val * curl_lambda_h;

            // Accumulate gradients only in interior region (pml_y == 1 && pml_x == 1)
            if (pml_y == 1 && pml_x == 1) {
              // grad_ca += λ_Ey^{n+1} * E_y^n
              if (ca_requires_grad && ey_store != NULL) {
                TIDE_DTYPE ey_n = ey_store[store_idx];
                if (ca_batched) {
                  grad_ca[store_idx] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio;
                } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                  grad_ca[idx] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio;
                }
              }

              // grad_cb += λ_Ey^{n+1} * curl_H^n
              if (cb_requires_grad && curl_h_store != NULL) {
                TIDE_DTYPE curl_h_n = curl_h_store[store_idx];
                if (ca_batched) {
                  grad_cb[store_idx] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio;
                } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                  grad_cb[idx] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio;
                }
              }
            }
          }
        }
      }
    }
  }
}

static void backward_kernel_lambda_e_with_grad_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    tide_bfloat16 const *__restrict const ey_store,
    tide_bfloat16 const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cq_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    int64_t const step_ratio) {

  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const ca_ptr =
        ca_batched ? (ca + shot_offset) : ca;
    TIDE_DTYPE const *__restrict const cq_ptr =
        cq_batched ? (cq + shot_offset) : cq;
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            int64_t const store_idx = shot_offset + idx;
            TIDE_DTYPE const ca_val = ca_ptr[idx];
            TIDE_DTYPE const cq_val = cq_ptr[idx];

            TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LAMBDA_HZ);
            TIDE_DTYPE d_lambda_hx_dz = DIFFY1(LAMBDA_HX);

            if (pml_x != 1) {
              M_LAMBDA_HZ_X(0, 0) = bx[x] * M_LAMBDA_HZ_X(0, 0) + ax[x] * d_lambda_hz_dx;
              d_lambda_hz_dx = d_lambda_hz_dx / kx[x] + M_LAMBDA_HZ_X(0, 0);
            }
            if (pml_y != 1) {
              M_LAMBDA_HX_Z(0, 0) = by[y] * M_LAMBDA_HX_Z(0, 0) + ay[y] * d_lambda_hx_dz;
              d_lambda_hx_dz = d_lambda_hx_dz / ky[y] + M_LAMBDA_HX_Z(0, 0);
            }

            TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;
            TIDE_DTYPE lambda_ey_curr = LAMBDA_EY(0, 0);
            LAMBDA_EY(0, 0) = ca_val * lambda_ey_curr + cq_val * curl_lambda_h;

            if (pml_y == 1 && pml_x == 1) {
              if (ca_requires_grad && ey_store != NULL) {
                TIDE_DTYPE ey_n =
                    (TIDE_DTYPE)tide_bf16_to_float(ey_store[store_idx]);
                if (ca_batched) {
                  grad_ca[store_idx] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio;
                } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                  grad_ca[idx] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio;
                }
              }

              if (cb_requires_grad && curl_h_store != NULL) {
                TIDE_DTYPE curl_h_n =
                    (TIDE_DTYPE)tide_bf16_to_float(curl_h_store[store_idx]);
                if (ca_batched) {
                  grad_cb[store_idx] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio;
                } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                  grad_cb[idx] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio;
                }
              }
            }
          }
        }
      }
    }
  }
}

static void backward_kernel_lambda_e_with_grad_fp8(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    tide_fp8_e4m3 const *__restrict const ey_store,
    tide_fp8_e4m3 const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
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
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cq_batched,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    int64_t const step_ratio) {

  int64_t const pml_bounds_y[] = {FD_PAD, pml_y0, pml_y1, ny - FD_PAD + 1};
  int64_t const pml_bounds_x[] = {FD_PAD, pml_x0, pml_x1, nx - FD_PAD + 1};

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const ca_ptr =
        ca_batched ? (ca + shot_offset) : ca;
    TIDE_DTYPE const *__restrict const cq_ptr =
        cq_batched ? (cq + shot_offset) : cq;
    for (int pml_y = 0; pml_y < 3; ++pml_y) {
      for (int pml_x = 0; pml_x < 3; ++pml_x) {
TIDE_OMP_SIMD_COLLAPSE2
        for (int64_t y = pml_bounds_y[pml_y]; y < pml_bounds_y[pml_y + 1]; ++y) {
          for (int64_t x = pml_bounds_x[pml_x]; x < pml_bounds_x[pml_x + 1]; ++x) {
            int64_t const idx = IDX(y, x);
            int64_t const store_idx = shot_offset + idx;
            TIDE_DTYPE const ca_val = ca_ptr[idx];
            TIDE_DTYPE const cq_val = cq_ptr[idx];

            TIDE_DTYPE d_lambda_hz_dx = DIFFX1(LAMBDA_HZ);
            TIDE_DTYPE d_lambda_hx_dz = DIFFY1(LAMBDA_HX);

            if (pml_x != 1) {
              M_LAMBDA_HZ_X(0, 0) = bx[x] * M_LAMBDA_HZ_X(0, 0) + ax[x] * d_lambda_hz_dx;
              d_lambda_hz_dx = d_lambda_hz_dx / kx[x] + M_LAMBDA_HZ_X(0, 0);
            }
            if (pml_y != 1) {
              M_LAMBDA_HX_Z(0, 0) = by[y] * M_LAMBDA_HX_Z(0, 0) + ay[y] * d_lambda_hx_dz;
              d_lambda_hx_dz = d_lambda_hx_dz / ky[y] + M_LAMBDA_HX_Z(0, 0);
            }

            TIDE_DTYPE curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;
            TIDE_DTYPE lambda_ey_curr = LAMBDA_EY(0, 0);
            LAMBDA_EY(0, 0) = ca_val * lambda_ey_curr + cq_val * curl_lambda_h;

            if (pml_y == 1 && pml_x == 1) {
              if (ca_requires_grad && ey_store != NULL) {
                TIDE_DTYPE ey_n =
                    (TIDE_DTYPE)tide_fp8_e4m3_to_float(ey_store[store_idx]);
                if (ca_batched) {
                  grad_ca[store_idx] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio;
                } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                  grad_ca[idx] += lambda_ey_curr * ey_n * (TIDE_DTYPE)step_ratio;
                }
              }

              if (cb_requires_grad && curl_h_store != NULL) {
                TIDE_DTYPE curl_h_n =
                    (TIDE_DTYPE)tide_fp8_e4m3_to_float(curl_h_store[store_idx]);
                if (ca_batched) {
                  grad_cb[store_idx] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio;
                } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                  grad_cb[idx] += lambda_ey_curr * curl_h_n * (TIDE_DTYPE)step_ratio;
                }
              }
            }
          }
        }
      }
    }
  }
}

static void inverse_kernel_e_and_curl(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const curl_h_out,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const ca_batched,
    bool const cb_batched) {

  int64_t const y0 = MAX(FD_PAD, pml_y0);
  int64_t const y1 = MIN(ny - FD_PAD + 1, pml_y1);
  int64_t const x0 = MAX(FD_PAD, pml_x0);
  int64_t const x1 = MIN(nx - FD_PAD + 1, pml_x1);

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const ca_ptr =
        ca_batched ? (ca + shot_offset) : ca;
    TIDE_DTYPE const *__restrict const cb_ptr =
        cb_batched ? (cb + shot_offset) : cb;
TIDE_OMP_SIMD_COLLAPSE2
    for (int64_t y = y0; y < y1; ++y) {
      for (int64_t x = x0; x < x1; ++x) {
        int64_t const idx = IDX(y, x);
        int64_t const store_idx = shot_offset + idx;
        TIDE_DTYPE const ca_val = ca_ptr[idx];
        TIDE_DTYPE const cb_val = cb_ptr[idx];

        TIDE_DTYPE const dhz_dx = DIFFX1(HZ);
        TIDE_DTYPE const dhx_dz = DIFFY1(HX);
        TIDE_DTYPE const curl_h = dhz_dx - dhx_dz;

        curl_h_out[store_idx] = curl_h;
        ey[store_idx] = (ey[store_idx] - cb_val * curl_h) / ca_val;
      }
    }
  }
}

static void inverse_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    int64_t const n_shots,
    int64_t const ny,
    int64_t const nx,
    int64_t const shot_numel,
    int64_t const pml_y0,
    int64_t const pml_y1,
    int64_t const pml_x0,
    int64_t const pml_x1,
    bool const cq_batched) {

  int64_t const y0 = MAX(FD_PAD, pml_y0);
  int64_t const y1 = MIN(ny - FD_PAD + 1, pml_y1);
  int64_t const x0 = MAX(FD_PAD, pml_x0);
  int64_t const x1 = MIN(nx - FD_PAD + 1, pml_x1);

  TIDE_OMP_INDEX shot_idx;
TIDE_OMP_PARALLEL_FOR
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const cq_ptr =
        cq_batched ? (cq + shot_offset) : cq;
TIDE_OMP_SIMD_COLLAPSE2
    for (int64_t y = y0; y < y1; ++y) {
      for (int64_t x = x0; x < x1; ++x) {
        int64_t const idx = IDX(y, x);
        TIDE_DTYPE const cq_val = cq_ptr[idx];

        if (y < ny - FD_PAD) {
          TIDE_DTYPE const dey_dz = DIFFYH1(EY);
          HX(0, 0) += cq_val * dey_dz;
        }
        if (x < nx - FD_PAD) {
          TIDE_DTYPE const dey_dx = DIFFXH1(EY);
          HZ(0, 0) -= cq_val * dey_dx;
        }
      }
    }
  }
}

/*
 * Full backward pass for Maxwell TM equations
 *
 * Implements the Adjoint State Method to compute:
 *   - grad_ca: gradient w.r.t. C_a coefficient
 *   - grad_cb: gradient w.r.t. C_b coefficient
 *   - grad_eps: gradient w.r.t. epsilon_r
 *   - grad_sigma: gradient w.r.t. conductivity
 *   - grad_f: gradient w.r.t. source amplitudes
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
    TIDE_DTYPE *const lambda_ey,
    TIDE_DTYPE *const lambda_hx,
    TIDE_DTYPE *const lambda_hz,
    TIDE_DTYPE *const m_lambda_ey_x,
    TIDE_DTYPE *const m_lambda_ey_z,
    TIDE_DTYPE *const m_lambda_hx_z,
    TIDE_DTYPE *const m_lambda_hz_x,
    TIDE_DTYPE *const ey_store_1,
    void *const ey_store_3,
    char const *const *const ey_filenames,
    TIDE_DTYPE *const curl_store_1,
    void *const curl_store_3,
    char const *const *const curl_filenames,
    TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_eps,
    TIDE_DTYPE *const grad_sigma,
    TIDE_DTYPE *const grad_ca_shot,  /* unused in CPU - for API compatibility */
    TIDE_DTYPE *const grad_cb_shot,  /* unused in CPU - for API compatibility */
    TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i,
    int64_t const *const receivers_i,
    TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx,
    TIDE_DTYPE const dt,
    int64_t const nt,
    int64_t const n_shots,
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
    int64_t const pml_y0,
    int64_t const pml_x0,
    int64_t const pml_y1,
    int64_t const pml_x1,
    int64_t const n_threads,
    int64_t const device /* unused for CPU */) {
  
  (void)device;
  (void)grad_ca_shot;  // Not needed in CPU version
  (void)grad_cb_shot;  // Not needed in CPU version
  (void)ey_store_3;
  (void)curl_store_3;
#ifdef _OPENMP
  int const prev_threads = omp_get_max_threads();
  if (n_threads > 0) {
    omp_set_num_threads((int)n_threads);
  }
#else
  (void)n_threads;
#endif

  int64_t const shot_numel = ny * nx;
  int64_t const store_size = n_shots * shot_numel;
  bool const storage_fp8 = (shot_bytes_uncomp == shot_numel * 1);
  bool const storage_bf16 = (shot_bytes_uncomp == shot_numel * 2);

  FILE **fp_ey = NULL;
  FILE **fp_curl = NULL;
  if (storage_mode == STORAGE_DISK) {
    if (ca_requires_grad) {
      fp_ey = (FILE **)malloc((size_t)n_shots * sizeof(FILE *));
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        fp_ey[shot] = fopen(ey_filenames[shot], "rb");
      }
    }
    if (cb_requires_grad) {
      fp_curl = (FILE **)malloc((size_t)n_shots * sizeof(FILE *));
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        fp_curl[shot] = fopen(curl_filenames[shot], "rb");
      }
    }
  }

  // Time reversed loop: from t = start_t - 1 down to start_t - nt
  // 
  // Forward order was: H_update -> E_update(store) -> source_inject -> record
  // Backward order is: record(adjoint) -> source_inject(adjoint) -> E_update(adjoint) -> H_update(adjoint)
  // Which translates to: grad_r_inject -> grad_f_record -> λ_E_update(grad_accum) -> λ_H_update
  
  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    // Determine storage index for this time step
    int64_t const store_idx = t / step_ratio;
    bool const do_grad = (t % step_ratio) == 0;
    bool const grad_ey = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;

    int64_t const store_offset =
        (storage_mode == STORAGE_DEVICE ? store_idx * store_size : 0);

    if (storage_fp8) {
      tide_fp8_e4m3 *const ey_store_1_t =
          (tide_fp8_e4m3 *)ey_store_1 + store_offset;
      tide_fp8_e4m3 *const curl_store_1_t =
          (tide_fp8_e4m3 *)curl_store_1 + store_offset;

      if (storage_mode == STORAGE_DISK) {
        if (grad_ey) {
          for (int64_t shot = 0; shot < n_shots; ++shot) {
            storage_load_snapshot_cpu(
                (void *)(ey_store_1_t + shot * shot_numel), fp_ey[shot],
                storage_mode, store_idx, (size_t)shot_bytes_uncomp);
          }
        }
        if (grad_curl) {
          for (int64_t shot = 0; shot < n_shots; ++shot) {
            storage_load_snapshot_cpu(
                (void *)(curl_store_1_t + shot * shot_numel), fp_curl[shot],
                storage_mode, store_idx, (size_t)shot_bytes_uncomp);
          }
        }
      }
    } else if (storage_bf16) {
      tide_bfloat16 *const ey_store_1_t =
          (tide_bfloat16 *)ey_store_1 + store_offset;
      tide_bfloat16 *const curl_store_1_t =
          (tide_bfloat16 *)curl_store_1 + store_offset;

      if (storage_mode == STORAGE_DISK) {
        if (grad_ey) {
          for (int64_t shot = 0; shot < n_shots; ++shot) {
            storage_load_snapshot_cpu(
                (void *)(ey_store_1_t + shot * shot_numel), fp_ey[shot],
                storage_mode, store_idx, (size_t)shot_bytes_uncomp);
          }
        }
        if (grad_curl) {
          for (int64_t shot = 0; shot < n_shots; ++shot) {
            storage_load_snapshot_cpu(
                (void *)(curl_store_1_t + shot * shot_numel), fp_curl[shot],
                storage_mode, store_idx, (size_t)shot_bytes_uncomp);
          }
        }
      }
    } else {
      TIDE_DTYPE *const ey_store_1_t =
          ey_store_1 + store_offset;
      TIDE_DTYPE *const curl_store_1_t =
          curl_store_1 + store_offset;

      if (storage_mode == STORAGE_DISK) {
        if (grad_ey) {
          for (int64_t shot = 0; shot < n_shots; ++shot) {
            storage_load_snapshot_cpu(
                (void *)(ey_store_1_t + shot * shot_numel), fp_ey[shot],
                storage_mode, store_idx, (size_t)shot_bytes_uncomp);
          }
        }
        if (grad_curl) {
          for (int64_t shot = 0; shot < n_shots; ++shot) {
            storage_load_snapshot_cpu(
                (void *)(curl_store_1_t + shot * shot_numel), fp_curl[shot],
                storage_mode, store_idx, (size_t)shot_bytes_uncomp);
          }
        }
      }
    }

    // Inject adjoint residuals into λ_Ey^{t+1} (adjoint of receiver recording)
    if (n_receivers_per_shot > 0) {
      add_sources_ey(
          lambda_ey, grad_r + t * n_shots * n_receivers_per_shot, receivers_i,
          n_shots, shot_numel, n_receivers_per_shot);
    }

    // Record adjoint source gradient using λ_Ey^{t+1} (adjoint of source injection)
    if (n_sources_per_shot > 0) {
      record_receivers_ey(
          grad_f + t * n_shots * n_sources_per_shot,
          lambda_ey, sources_i,
          n_shots, shot_numel, n_sources_per_shot);
    }

    // Backward λ_Ey update with gradient accumulation
    // This computes: λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * curl_λH
    // And accumulates: grad_ca += λ_Ey^{n+1} * E_y^n, grad_cb += λ_Ey^{n+1} * curl_H^n
    if (storage_fp8 && (grad_ey || grad_curl)) {
      tide_fp8_e4m3 *const ey_store_1_t =
          (tide_fp8_e4m3 *)ey_store_1 + store_offset;
      tide_fp8_e4m3 *const curl_store_1_t =
          (tide_fp8_e4m3 *)curl_store_1 + store_offset;
      backward_kernel_lambda_e_with_grad_fp8(
          ca, cq, lambda_hx, lambda_hz, lambda_ey,
          m_lambda_hx_z, m_lambda_hz_x,
          grad_ey ? ey_store_1_t : NULL,
          grad_curl ? curl_store_1_t : NULL,
          grad_ca, grad_cb,
          ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh,
          rdy, rdx,
          n_shots, ny, nx, shot_numel,
          pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cq_batched,
          grad_ey, grad_curl,
          step_ratio);
    } else if (storage_bf16 && (grad_ey || grad_curl)) {
      tide_bfloat16 *const ey_store_1_t =
          (tide_bfloat16 *)ey_store_1 + store_offset;
      tide_bfloat16 *const curl_store_1_t =
          (tide_bfloat16 *)curl_store_1 + store_offset;
      backward_kernel_lambda_e_with_grad_bf16(
          ca, cq, lambda_hx, lambda_hz, lambda_ey,
          m_lambda_hx_z, m_lambda_hz_x,
          grad_ey ? ey_store_1_t : NULL,
          grad_curl ? curl_store_1_t : NULL,
          grad_ca, grad_cb,
          ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh,
          rdy, rdx,
          n_shots, ny, nx, shot_numel,
          pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cq_batched,
          grad_ey, grad_curl,
          step_ratio);
    } else {
      TIDE_DTYPE *const ey_store_1_t =
          (storage_fp8 || storage_bf16) ? NULL : (ey_store_1 + store_offset);
      TIDE_DTYPE *const curl_store_1_t =
          (storage_fp8 || storage_bf16) ? NULL : (curl_store_1 + store_offset);
      backward_kernel_lambda_e_with_grad(
          ca, cq, lambda_hx, lambda_hz, lambda_ey,
          m_lambda_hx_z, m_lambda_hz_x,
          grad_ey ? ey_store_1_t : NULL,
          grad_curl ? curl_store_1_t : NULL,
          grad_ca, grad_cb,
          ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh,
          rdy, rdx,
          n_shots, ny, nx, shot_numel,
          pml_y0, pml_y1, pml_x0, pml_x1,
          ca_batched, cq_batched,
          grad_ey, grad_curl,
          step_ratio);
    }

    // Backward λ_H fields update
    backward_kernel_lambda_h(
        cb, lambda_ey, lambda_hx, lambda_hz,
        m_lambda_ey_x, m_lambda_ey_z,
        ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh,
        rdy, rdx,
        n_shots, ny, nx, shot_numel,
        pml_y0, pml_y1, pml_x0, pml_x1,
        cb_batched);
  }

  if (fp_ey != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot) fclose(fp_ey[shot]);
    free(fp_ey);
  }
  if (fp_curl != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot) fclose(fp_curl[shot]);
    free(fp_curl);
  }

  convert_grad_ca_cb_to_eps_sigma(
      ca, cb, grad_ca, grad_cb, grad_eps, grad_sigma,
      dt, n_shots, ny, nx, ca_batched, cb_batched,
      ca_requires_grad, cb_requires_grad);
#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}
