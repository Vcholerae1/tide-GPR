/*
 * Maxwell wave equation propagator (CUDA implementation)
 *
 * This file contains the CUDA implementation of the 2D TM Maxwell equations
 * propagator with complete Adjoint State Method (ASM) support for gradient
 * computation.
 *
 * TM mode fields: Ey (electric), Hx, Hz (magnetic)
 *
 * EXACT DISCRETE Adjoint State Method for Maxwell TM equations:
 * =============================================================
 * Forward equations (discrete):
 *   E_y^{n+1} = C_a * E_y^n + C_b * (D_x[H_z] - D_z[H_x])
 *   H_x^{n+1/2} = H_x^{n-1/2} - C_q * D_z^h[E_y]
 *   H_z^{n+1/2} = H_z^{n-1/2} + C_q * D_x^h[E_y]
 *
 * Exact discrete adjoint equations (time-reversed with transposed operators):
 *   λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * (D_x^{hT}[λ_Hz] - D_z^{hT}[λ_Hx])
 *   λ_Hx^{n-1/2} = λ_Hx^{n+1/2} - C_b * D_z^T[λ_Ey]
 *   λ_Hz^{n-1/2} = λ_Hz^{n+1/2} + C_b * D_x^T[λ_Ey]
 *
 * Model gradients:
 *   ∂J/∂C_a = Σ_n E_y^n * λ_Ey^{n+1}
 *   ∂J/∂C_b = Σ_n curl_H^n * λ_Ey^{n+1}
 *
 * Gradient accumulation strategy:
 *   - Use per-shot gradient arrays (grad_ca_shot, grad_cb_shot)
 *   - Each shot writes to its own memory region (no race condition)
 *   - Use combine_grad kernel to sum across shots at the end
 */

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <type_traits>
#include "common_gpu.h"
#include "staggered_grid.h"
#include "storage_utils.h"

#ifndef TIDE_DEVICE
#define TIDE_DEVICE cuda
#endif

// CPU storage pipelining: Number of ping-pong buffers for async D2H/H2D copies
// Increasing this reduces synchronization stalls between compute and copy
#ifndef NUM_BUFFERS
#define NUM_BUFFERS 3
#endif

#define CAT_I(name, accuracy, dtype, device)                                   \
  maxwell_tm_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, TIDE_DEVICE)

// 2D indexing macros
#define ND_INDEX(i, dy, dx) (i + (dy) * nx + (dx))
#define ND_INDEX_J(j, dy, dx) (j + (dy) * nx + (dx))

#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
// Field access macros
#define EY(dy, dx) ey[ND_INDEX(i, dy, dx)]
#define HX(dy, dx) hx[ND_INDEX(i, dy, dx)]
#define HZ(dy, dx) hz[ND_INDEX(i, dy, dx)]

// Adjoint field access macros
// Removed old Adjoint PML memory variable macros

#define MAX(a, b) (a > b ? a : b)

using field_t = TIDE_DTYPE;
using accum_t = typename std::conditional<std::is_same<field_t, half>::value,
                                          float, field_t>::type;
using scalar_t = typename std::conditional<std::is_same<field_t, half>::value,
                                           float, field_t>::type;
constexpr bool kFieldIsHalf = std::is_same<field_t, half>::value;
constexpr accum_t kEp0 = (accum_t)8.8541878128e-12;

namespace {

// Device constants
__constant__ scalar_t rdy;
__constant__ scalar_t rdx;
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
__constant__ bool ca_batched;
__constant__ bool cb_batched;
__constant__ bool cq_batched;

// Add source to field
template <typename T>
__global__ void add_sources_ey(T *__restrict const ey,
                               T const *__restrict const f,
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

// Add adjoint source at receiver locations (for backward pass)
template <typename T>
__global__ void
add_adjoint_sources_ey(T *__restrict const ey,
                       T const *__restrict const f,
                       int64_t const *__restrict const receivers_i) {
  int64_t receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (0 <= rec) {
      ey[shot_idx * shot_numel + rec] += f[k];
    }
  }
}

// Record field at receiver locations
template <typename T>
__global__ void record_receivers_ey(T *__restrict const r,
                                    T const *__restrict const ey,
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

// Record adjoint field at source locations (for backward pass - source
// gradient)
template <typename T>
__global__ void
record_adjoint_at_sources(T *__restrict const grad_f,
                          T const *__restrict const lambda_ey,
                          int64_t const *__restrict sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (0 <= src) {
      grad_f[k] = lambda_ey[shot_idx * shot_numel + src];
    }
  }
}

#include "maxwell_tm_core.cuh"

using namespace tide;


#undef FUNC
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, cuda)

#define TIDE_STENCIL 2
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 2
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 2
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE
