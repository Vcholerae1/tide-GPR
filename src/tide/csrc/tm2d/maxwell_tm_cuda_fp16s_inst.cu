#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

#include "common_gpu.h"
#include "storage_utils.h"

#define CAT_I(name, accuracy, dtype, variant, device)                          \
  maxwell_tm_##accuracy##_##dtype##_##name##_##variant##_##device
#define CAT(name, accuracy, dtype, variant, device)                            \
  CAT_I(name, accuracy, dtype, variant, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, fp16s, cuda)

#include "maxwell_tm_core.cuh"

#undef DIFFY1
#undef DIFFX1
#undef DIFFYH1
#undef DIFFXH1
#undef DIFFY1_ADJ
#undef DIFFX1_ADJ
#undef DIFFYH1_ADJ
#undef DIFFXH1_ADJ

#ifdef STAGGERED_GRID_H
#undef STAGGERED_GRID_H
#endif
#include "staggered_grid.h"

namespace FUNC(Inst) {

using coeff_t = TIDE_DTYPE;
constexpr int kFdPad = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;

#ifndef TIDE_TM_BLOCK_X
#define TIDE_TM_BLOCK_X 32
#endif
#ifndef TIDE_TM_BLOCK_Y
#define TIDE_TM_BLOCK_Y 8
#endif

namespace {

__constant__ double rdy;
__constant__ double rdx;
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

struct DeviceConstantCache2D {
  bool initialized = false;
  coeff_t rdy_h = 0;
  coeff_t rdx_h = 0;
  int64_t n_shots_h = -1;
  int64_t ny_h = -1;
  int64_t nx_h = -1;
  int64_t shot_numel_h = -1;
  int64_t n_sources_per_shot_h = -1;
  int64_t n_receivers_per_shot_h = -1;
  int64_t pml_y0_h = -1;
  int64_t pml_y1_h = -1;
  int64_t pml_x0_h = -1;
  int64_t pml_x1_h = -1;
  bool ca_batched_h = false;
  bool cb_batched_h = false;
  bool cq_batched_h = false;
  int64_t device = -1;
};

template <typename T>
__device__ __forceinline__ float load_field(T const *ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_field<__half>(__half const *ptr,
                                                    int64_t idx) {
  return __half2float(ptr[idx]);
}

template <typename T>
__device__ __forceinline__ void store_field(T *ptr, int64_t idx, float value) {
  ptr[idx] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_field<__half>(__half *ptr, int64_t idx,
                                                    float value) {
  ptr[idx] = __float2half_rn(value);
}

template <typename StorageT> struct DecodingGlobalAccessor {
  StorageT const *ptr;
  int nx;
  TIDE_HOST_DEVICE DecodingGlobalAccessor(StorageT const *p, int w)
      : ptr(p), nx(w) {}
  TIDE_HOST_DEVICE float operator()(int64_t base, int y, int x) const {
    return load_field(ptr, base + static_cast<int64_t>(y) * nx + x);
  }
};

TIDE_HOST_DEVICE int64_t tide_nd_index(int64_t base, int64_t dy, int64_t dx,
                                       int64_t nx_val) {
  return base + dy * nx_val + dx;
}

TIDE_HOST_DEVICE int64_t tide_nd_index_j(int64_t base, int64_t dy, int64_t dx,
                                         int64_t nx_val) {
  return base + dy * nx_val + dx;
}

template <typename T>
TIDE_HOST_DEVICE T tide_max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
TIDE_HOST_DEVICE T tide_min(T a, T b) {
  return a < b ? a : b;
}

__device__ __forceinline__ coeff_t ldg_coeff(
    coeff_t const *__restrict const coeff, bool const coeff_is_batched,
    int64_t const shot_index, int64_t const grid_index) {
  return __ldg(coeff + (coeff_is_batched ? shot_index : grid_index));
}

__device__ __forceinline__ coeff_t step_ratio_to_field(
    int64_t const step_ratio_val) {
  return static_cast<coeff_t>(step_ratio_val);
}

struct BoundaryLaunchLayout {
  int64_t domain_x_begin;
  int64_t domain_x_end;
  int64_t domain_y_begin;
  int64_t domain_y_end;
  int64_t interior_x_begin;
  int64_t interior_x_end;
  int64_t interior_y_begin;
  int64_t interior_y_end;
  int64_t domain_width;
  int64_t top_count;
  int64_t bottom_count;
  int64_t left_count;
  int64_t right_count;
  int64_t left_width;
  int64_t right_width;
  int64_t total_count;
};

__host__ __forceinline__ BoundaryLaunchLayout make_boundary_launch_layout(
    int64_t const domain_y_begin, int64_t const domain_y_end,
    int64_t const domain_x_begin, int64_t const domain_x_end,
    int64_t interior_y_begin, int64_t interior_y_end,
    int64_t interior_x_begin, int64_t interior_x_end,
    bool const has_interior) {
  BoundaryLaunchLayout layout{};
  layout.domain_x_begin = domain_x_begin;
  layout.domain_x_end = domain_x_end;
  layout.domain_y_begin = domain_y_begin;
  layout.domain_y_end = domain_y_end;
  layout.domain_width = domain_x_end - domain_x_begin;
  layout.left_width = 0;
  layout.right_width = 0;
  layout.top_count = 0;
  layout.bottom_count = 0;
  layout.left_count = 0;
  layout.right_count = 0;
  layout.total_count = 0;

  if (layout.domain_width <= 0 || domain_y_end <= domain_y_begin) {
    return layout;
  }

  int64_t const domain_height = domain_y_end - domain_y_begin;
  if (!has_interior) {
    layout.interior_x_begin = domain_x_begin;
    layout.interior_x_end = domain_x_begin;
    layout.interior_y_begin = domain_y_end;
    layout.interior_y_end = domain_y_end;
    layout.top_count = domain_height * layout.domain_width;
    layout.total_count = layout.top_count;
    return layout;
  }

  if (interior_x_begin < domain_x_begin)
    interior_x_begin = domain_x_begin;
  if (interior_x_end > domain_x_end)
    interior_x_end = domain_x_end;
  if (interior_y_begin < domain_y_begin)
    interior_y_begin = domain_y_begin;
  if (interior_y_end > domain_y_end)
    interior_y_end = domain_y_end;
  if (interior_x_end < interior_x_begin)
    interior_x_end = interior_x_begin;
  if (interior_y_end < interior_y_begin)
    interior_y_end = interior_y_begin;

  layout.interior_x_begin = interior_x_begin;
  layout.interior_x_end = interior_x_end;
  layout.interior_y_begin = interior_y_begin;
  layout.interior_y_end = interior_y_end;

  int64_t const top_rows = interior_y_begin - domain_y_begin;
  int64_t const bottom_rows = domain_y_end - interior_y_end;
  int64_t const mid_rows = interior_y_end - interior_y_begin;
  layout.left_width = interior_x_begin - domain_x_begin;
  layout.right_width = domain_x_end - interior_x_end;

  layout.top_count = top_rows * layout.domain_width;
  layout.bottom_count = bottom_rows * layout.domain_width;
  layout.left_count = mid_rows * layout.left_width;
  layout.right_count = mid_rows * layout.right_width;
  layout.total_count =
      layout.top_count + layout.bottom_count + layout.left_count +
      layout.right_count;
  return layout;
}

__device__ __forceinline__ bool decode_boundary_point(
    BoundaryLaunchLayout const &layout, int64_t boundary_index, int64_t &y,
    int64_t &x) {
  if (boundary_index < 0 || boundary_index >= layout.total_count ||
      layout.domain_width <= 0) {
    return false;
  }

  int64_t idx = boundary_index;
  if (idx < layout.top_count) {
    y = layout.domain_y_begin + idx / layout.domain_width;
    x = layout.domain_x_begin + idx % layout.domain_width;
    return true;
  }
  idx -= layout.top_count;

  if (idx < layout.bottom_count) {
    y = layout.interior_y_end + idx / layout.domain_width;
    x = layout.domain_x_begin + idx % layout.domain_width;
    return true;
  }
  idx -= layout.bottom_count;

  if (layout.left_width > 0 && idx < layout.left_count) {
    y = layout.interior_y_begin + idx / layout.left_width;
    x = layout.domain_x_begin + idx % layout.left_width;
    return true;
  }
  idx -= layout.left_count;

  if (layout.right_width > 0 && idx < layout.right_count) {
    y = layout.interior_y_begin + idx / layout.right_width;
    x = layout.interior_x_end + idx % layout.right_width;
    return true;
  }
  return false;
}

static inline unsigned int to_dim_u32(int64_t value) {
  return static_cast<unsigned int>(value > 0 ? value : 1);
}

static inline void sync_device_constants_if_needed(
    DeviceConstantCache2D &cache, coeff_t const rdy_h, coeff_t const rdx_h,
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const shot_numel_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const pml_y0_h,
    int64_t const pml_x0_h, int64_t const pml_y1_h, int64_t const pml_x1_h,
    bool const ca_batched_h, bool const cb_batched_h, bool const cq_batched_h,
    int64_t const device) {
  if (cache.initialized && cache.device == device && cache.rdy_h == rdy_h &&
      cache.rdx_h == rdx_h && cache.n_shots_h == n_shots_h &&
      cache.ny_h == ny_h && cache.nx_h == nx_h &&
      cache.shot_numel_h == shot_numel_h &&
      cache.n_sources_per_shot_h == n_sources_per_shot_h &&
      cache.n_receivers_per_shot_h == n_receivers_per_shot_h &&
      cache.pml_y0_h == pml_y0_h && cache.pml_y1_h == pml_y1_h &&
      cache.pml_x0_h == pml_x0_h && cache.pml_x1_h == pml_x1_h &&
      cache.ca_batched_h == ca_batched_h &&
      cache.cb_batched_h == cb_batched_h &&
      cache.cq_batched_h == cq_batched_h) {
    return;
  }

  double const rdy_const = static_cast<double>(rdy_h);
  double const rdx_const = static_cast<double>(rdx_h);
  cudaMemcpyToSymbol(rdy, &rdy_const, sizeof(double));
  cudaMemcpyToSymbol(rdx, &rdx_const, sizeof(double));
  cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t));
  cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t));
  cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t));
  cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t));
  cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h,
                     sizeof(int64_t));
  cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h,
                     sizeof(int64_t));
  cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t));
  cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t));
  cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t));
  cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t));
  cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool));
  cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool));
  cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool));

  cache.initialized = true;
  cache.rdy_h = rdy_h;
  cache.rdx_h = rdx_h;
  cache.n_shots_h = n_shots_h;
  cache.ny_h = ny_h;
  cache.nx_h = nx_h;
  cache.shot_numel_h = shot_numel_h;
  cache.n_sources_per_shot_h = n_sources_per_shot_h;
  cache.n_receivers_per_shot_h = n_receivers_per_shot_h;
  cache.pml_y0_h = pml_y0_h;
  cache.pml_y1_h = pml_y1_h;
  cache.pml_x0_h = pml_x0_h;
  cache.pml_x1_h = pml_x1_h;
  cache.ca_batched_h = ca_batched_h;
  cache.cb_batched_h = cb_batched_h;
  cache.cq_batched_h = cq_batched_h;
  cache.device = device;
}

struct TMForwardLaunchConfig {
  dim3 dimBlock;
  dim3 dimGrid;
  dim3 dimBlockSources;
  dim3 dimGridSources;
  dim3 dimBlockReceivers;
  dim3 dimGridReceivers;
};

static inline TMForwardLaunchConfig make_tm_forward_launch_config(
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h) {
  TMForwardLaunchConfig cfg{};
  cfg.dimBlock = dim3(TIDE_TM_BLOCK_X, TIDE_TM_BLOCK_Y, 1);

  int64_t const gridx =
      (nx_h - 2 * kFdPad + 2 + cfg.dimBlock.x - 1) / cfg.dimBlock.x;
  int64_t const gridy =
      (ny_h - 2 * kFdPad + 2 + cfg.dimBlock.y - 1) / cfg.dimBlock.y;
  cfg.dimGrid = dim3(to_dim_u32(gridx), to_dim_u32(gridy), to_dim_u32(n_shots_h));

  cfg.dimBlockSources = dim3(32, 1, 1);
  cfg.dimGridSources =
      dim3(to_dim_u32((n_sources_per_shot_h + cfg.dimBlockSources.x - 1) /
                      cfg.dimBlockSources.x),
           to_dim_u32(n_shots_h), 1);

  cfg.dimBlockReceivers = dim3(32, 1, 1);
  cfg.dimGridReceivers =
      dim3(to_dim_u32((n_receivers_per_shot_h + cfg.dimBlockReceivers.x - 1) /
                      cfg.dimBlockReceivers.x),
           to_dim_u32(n_shots_h), 1);
  return cfg;
}

struct TMEbisuRuntimeConfig {
  bool enabled = false;
  int64_t steps = 0;
  int64_t tile_x = 64;
  int64_t tile_y = 16;
  int64_t ilp = 1;
};

static inline int64_t read_env_i64(char const *name, int64_t fallback) {
  char const *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char *end = nullptr;
  long long parsed = std::strtoll(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

static inline bool read_env_flag(char const *name) {
  char const *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static inline TMEbisuRuntimeConfig read_tm_ebisu_runtime_config() {
  TMEbisuRuntimeConfig cfg{};
  cfg.steps = read_env_i64("TIDE_TM_EBISU_STEPS", 0);
  cfg.enabled = cfg.steps > 0;
  cfg.tile_x = read_env_i64("TIDE_TM_EBISU_TILE_X", 64);
  cfg.tile_y = read_env_i64("TIDE_TM_EBISU_TILE_Y", 16);
  cfg.ilp = read_env_i64("TIDE_TM_EBISU_ILP", 1);
  if (cfg.tile_x <= 0) {
    cfg.tile_x = 64;
  }
  if (cfg.tile_y <= 0) {
    cfg.tile_y = 16;
  }
  if (cfg.ilp != 1 && cfg.ilp != 2 && cfg.ilp != 4) {
    cfg.ilp = 1;
  }
  return cfg;
}

static inline size_t ring_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode_h,
    size_t const bytes_per_step_store) {
  if (storage_mode_h == STORAGE_DEVICE) {
    return static_cast<size_t>(step_idx) * bytes_per_step_store;
  }
  if (storage_mode_h == STORAGE_CPU) {
    return static_cast<size_t>(step_idx % NUM_BUFFERS) * bytes_per_step_store;
  }
  return 0;
}

static inline size_t cpu_linear_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode_h,
    size_t const bytes_per_step_store) {
  if (storage_mode_h == STORAGE_CPU) {
    return static_cast<size_t>(step_idx) * bytes_per_step_store;
  }
  return 0;
}

static inline void require_fp16_capability(int64_t device) {
  cudaDeviceProp prop{};
  tide::cuda_check_or_abort(cudaGetDeviceProperties(&prop, static_cast<int>(device)),
                            __FILE__, __LINE__);
  if (prop.major * 10 + prop.minor < 53) {
    fprintf(stderr,
            "TM2D fp16_scaled requires sm_53+ but device %d reports sm_%d%d.\n",
            static_cast<int>(device), prop.major, prop.minor);
    abort();
  }
}

__global__ void add_sources_ey_fp16s(__half *__restrict const ey,
                                     coeff_t const *__restrict const f,
                                     int64_t const *__restrict const sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (src >= 0) {
      int64_t const idx = shot_idx * shot_numel + src;
      float const value = load_field(ey, idx) + f[k];
      store_field(ey, idx, value);
    }
  }
}

__global__ void add_adjoint_sources_ey_fp16s(
    coeff_t *__restrict const ey, coeff_t const *__restrict const f,
    int64_t const *__restrict const receivers_i) {
  int64_t receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
    if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
      int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
      int64_t const rec = receivers_i[k];
      if (rec >= 0) {
        int64_t const idx = shot_idx * shot_numel + rec;
      ey[idx] += f[k];
      }
    }
}

__global__ void record_receivers_ey_fp16s(coeff_t *__restrict const r,
                                          __half const *__restrict const ey,
                                          int64_t const *__restrict receivers_i) {
  int64_t receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (rec >= 0) {
      r[k] = load_field(ey, shot_idx * shot_numel + rec);
    }
  }
}

__global__ void record_adjoint_at_sources_fp16s(
    coeff_t *__restrict const grad_f, coeff_t const *__restrict const lambda_ey,
    int64_t const *__restrict sources_i) {
  int64_t source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t shot_idx =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
    if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
      int64_t const k = shot_idx * n_sources_per_shot + source_idx;
      int64_t const src = sources_i[k];
      if (src >= 0) {
      grad_f[k] = lambda_ey[shot_idx * shot_numel + src];
      }
    }
}

template <int ILP>
__global__ __launch_bounds__(256, 1) void forward_kernel_ebisu_tb_fp16s(
    coeff_t const *__restrict const ca, coeff_t const *__restrict const cb,
    coeff_t const *__restrict const cq, coeff_t const *__restrict const f,
    __half *__restrict const ey, __half *__restrict const hx,
    __half *__restrict const hz, coeff_t *__restrict const m_ey_x,
    coeff_t *__restrict const m_ey_z, coeff_t *__restrict const m_hx_z,
    coeff_t *__restrict const m_hz_x, coeff_t const *__restrict const ay,
    coeff_t const *__restrict const ayh, coeff_t const *__restrict const ax,
    coeff_t const *__restrict const axh, coeff_t const *__restrict const by,
    coeff_t const *__restrict const byh, coeff_t const *__restrict const bx,
    coeff_t const *__restrict const bxh, coeff_t const *__restrict const ky,
    coeff_t const *__restrict const kyh, coeff_t const *__restrict const kx,
    coeff_t const *__restrict const kxh, int64_t const *__restrict const sources_i,
    int64_t const *__restrict const receivers_i, coeff_t *__restrict const r,
    int64_t const core_y, int64_t const core_x, int64_t const halo,
    int64_t const k_steps) {
  extern __shared__ unsigned char smem_raw[];
  coeff_t *ey_a = reinterpret_cast<coeff_t *>(smem_raw);
  int const core_y_i = static_cast<int>(core_y);
  int const core_x_i = static_cast<int>(core_x);
  int const halo_i = static_cast<int>(halo);
  int const k_steps_i = static_cast<int>(k_steps);
  int const ny_i = static_cast<int>(ny);
  int const nx_i = static_cast<int>(nx);
  int const shot_numel_i = static_cast<int>(shot_numel);
  int const tile_y = core_y_i + 2 * halo_i;
  int const tile_x = core_x_i + 2 * halo_i;
  int const tile_elems = tile_y * tile_x;
  coeff_t *ey_b = ey_a + tile_elems;
  coeff_t *hx_s = ey_b + tile_elems;
  coeff_t *hz_s = hx_s + tile_elems;
  coeff_t *m_ey_x_s = hz_s + tile_elems;
  coeff_t *m_ey_z_s = m_ey_x_s + tile_elems;
  coeff_t *m_hx_z_s = m_ey_z_s + tile_elems;
  coeff_t *m_hz_x_s = m_hx_z_s + tile_elems;
  coeff_t const rdy_t = static_cast<coeff_t>(rdy);
  coeff_t const rdx_t = static_cast<coeff_t>(rdx);
  int const domain_y_begin = kFdPad;
  int const domain_x_begin = kFdPad;
  int const domain_y_end = ny_i - kFdPad + 1;
  int const domain_x_end = nx_i - kFdPad + 1;

  int64_t const shot_idx = static_cast<int64_t>(blockIdx.z);
  if (shot_idx >= n_shots) {
    return;
  }

  int64_t const shot_offset = shot_idx * shot_numel;
  int const out_y0 = domain_y_begin + static_cast<int>(blockIdx.y) * core_y_i;
  int const out_x0 = domain_x_begin + static_cast<int>(blockIdx.x) * core_x_i;
  int const load_y0 = out_y0 - halo_i;
  int const load_x0 = out_x0 - halo_i;
  int const tid = static_cast<int>(threadIdx.x);
  int const block_threads = static_cast<int>(blockDim.x);

  for (int base = tid; base < tile_elems; base += block_threads * ILP) {
#pragma unroll
    for (int lane = 0; lane < ILP; ++lane) {
      int const li = base + lane * block_threads;
      if (li >= tile_elems) {
        continue;
      }
      int const ly = li / tile_x;
      int const lx = li - ly * tile_x;
      int gy = load_y0 + ly;
      int gx = load_x0 + lx;
      gy = tide_max<int>(0, tide_min<int>(gy, ny_i - 1));
      gx = tide_max<int>(0, tide_min<int>(gx, nx_i - 1));
      int64_t const gj = static_cast<int64_t>(gy) * nx_i + gx;
      int64_t const gi = shot_offset + gj;
      ey_a[li] = load_field(ey, gi);
      hx_s[li] = load_field(hx, gi);
      hz_s[li] = load_field(hz, gi);
      m_ey_x_s[li] = m_ey_x[gi];
      m_ey_z_s[li] = m_ey_z[gi];
      m_hx_z_s[li] = m_hx_z[gi];
      m_hz_x_s[li] = m_hz_x[gi];
    }
  }
  __syncthreads();

  coeff_t *ey_src = ey_a;
  coeff_t *ey_dst = ey_b;
  for (int step = 0; step < k_steps_i; ++step) {
    int const lo = step;
    int const hi_y = tile_y - step;
    int const hi_x = tile_x - step;

    for (int base = tid; base < tile_elems; base += block_threads * ILP) {
#pragma unroll
      for (int lane = 0; lane < ILP; ++lane) {
        int const li = base + lane * block_threads;
        if (li >= tile_elems) {
          continue;
        }
        int const ly = li / tile_x;
        int const lx = li - ly * tile_x;
        if (ly >= lo && ly + 1 < hi_y && lx >= lo && lx + 1 < hi_x) {
          int const gy = load_y0 + ly;
          int const gx = load_x0 + lx;
          if (gy >= kFdPad && gx >= kFdPad && gy < ny_i - kFdPad + 1 &&
              gx < nx_i - kFdPad + 1) {
            int64_t const gj = static_cast<int64_t>(gy) * nx_i + gx;
            int64_t const gi = shot_offset + gj;
            coeff_t const cq_val = cq_batched ? cq[gi] : cq[gj];

            if (gy < ny_i - kFdPad) {
              bool const in_pml_y = gy < pml_y0 ||
                                    gy >= tide_max<int64_t>(pml_y0, pml_y1 - 1);
              coeff_t dey_dz = (ey_src[li + tile_x] - ey_src[li]) * rdy_t;
              if (in_pml_y) {
                coeff_t const m_new =
                    __ldg(&byh[gy]) * m_ey_z_s[li] + __ldg(&ayh[gy]) * dey_dz;
                m_ey_z_s[li] = m_new;
                dey_dz = dey_dz / __ldg(&kyh[gy]) + m_new;
              }
              hx_s[li] -= cq_val * dey_dz;
            }

            if (gx < nx_i - kFdPad) {
              bool const in_pml_x = gx < pml_x0 ||
                                    gx >= tide_max<int64_t>(pml_x0, pml_x1 - 1);
              coeff_t dey_dx = (ey_src[li + 1] - ey_src[li]) * rdx_t;
              if (in_pml_x) {
                coeff_t const m_new =
                    __ldg(&bxh[gx]) * m_ey_x_s[li] + __ldg(&axh[gx]) * dey_dx;
                m_ey_x_s[li] = m_new;
                dey_dx = dey_dx / __ldg(&kxh[gx]) + m_new;
              }
              hz_s[li] += cq_val * dey_dx;
            }
          }
        }
      }
    }
    __syncthreads();

    for (int base = tid; base < tile_elems; base += block_threads * ILP) {
#pragma unroll
      for (int lane = 0; lane < ILP; ++lane) {
        int const li = base + lane * block_threads;
        if (li >= tile_elems) {
          continue;
        }
        int const ly = li / tile_x;
        int const lx = li - ly * tile_x;
        if (ly > lo && ly < hi_y - 1 && lx > lo && lx < hi_x - 1) {
          int const gy = load_y0 + ly;
          int const gx = load_x0 + lx;
          if (gy >= kFdPad && gx >= kFdPad && gy < ny_i - kFdPad + 1 &&
              gx < nx_i - kFdPad + 1) {
            int64_t const gj = static_cast<int64_t>(gy) * nx_i + gx;
            int64_t const gi = shot_offset + gj;
            coeff_t const ca_val = ca_batched ? ca[gi] : ca[gj];
            coeff_t const cb_val = cb_batched ? cb[gi] : cb[gj];
            bool const in_pml_y = gy < pml_y0 || gy >= pml_y1;
            bool const in_pml_x = gx < pml_x0 || gx >= pml_x1;

            coeff_t dhz_dx = (hz_s[li] - hz_s[li - 1]) * rdx_t;
            coeff_t dhx_dz = (hx_s[li] - hx_s[li - tile_x]) * rdy_t;

            if (in_pml_x) {
              coeff_t const m_new =
                  __ldg(&bx[gx]) * m_hz_x_s[li] + __ldg(&ax[gx]) * dhz_dx;
              m_hz_x_s[li] = m_new;
              dhz_dx = dhz_dx / __ldg(&kx[gx]) + m_new;
            }
            if (in_pml_y) {
              coeff_t const m_new =
                  __ldg(&by[gy]) * m_hx_z_s[li] + __ldg(&ay[gy]) * dhx_dz;
              m_hx_z_s[li] = m_new;
              dhx_dz = dhx_dz / __ldg(&ky[gy]) + m_new;
            }

            ey_dst[li] = ca_val * ey_src[li] + cb_val * (dhz_dx - dhx_dz);
          }
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      if (n_sources_per_shot > 0 && sources_i != nullptr && f != nullptr) {
        coeff_t const *const f_step =
            f + static_cast<int64_t>(step) * n_shots * n_sources_per_shot;
        for (int64_t source_idx = 0; source_idx < n_sources_per_shot;
             ++source_idx) {
          int64_t const src =
              sources_i[shot_idx * n_sources_per_shot + source_idx];
          if (src < 0) {
            continue;
          }
          int const sy = static_cast<int>(src / nx_i);
          int const sx = static_cast<int>(src - static_cast<int64_t>(sy) * nx_i);
          int const ly = sy - load_y0;
          int const lx = sx - load_x0;
          if (ly > lo && ly < hi_y - 1 && lx > lo && lx < hi_x - 1) {
            int const li = ly * tile_x + lx;
            ey_dst[li] += f_step[shot_idx * n_sources_per_shot + source_idx];
          }
        }
      }
      if (n_receivers_per_shot > 0 && receivers_i != nullptr && r != nullptr) {
        coeff_t *const r_step =
            r + static_cast<int64_t>(step) * n_shots * n_receivers_per_shot;
        for (int64_t receiver_idx = 0; receiver_idx < n_receivers_per_shot;
             ++receiver_idx) {
          int64_t const rec =
              receivers_i[shot_idx * n_receivers_per_shot + receiver_idx];
          if (rec < 0) {
            continue;
          }
          int const ry = static_cast<int>(rec / nx_i);
          int const rx = static_cast<int>(rec - static_cast<int64_t>(ry) * nx_i);
          if (ry >= out_y0 && ry < out_y0 + core_y_i && rx >= out_x0 &&
              rx < out_x0 + core_x_i) {
            int const li = (ry - load_y0) * tile_x + (rx - load_x0);
            r_step[shot_idx * n_receivers_per_shot + receiver_idx] = ey_dst[li];
          }
        }
      }
    }
    __syncthreads();

    coeff_t *tmp = ey_src;
    ey_src = ey_dst;
    ey_dst = tmp;
  }

  int const valid_out_y =
      tide_max<int>(0, tide_min<int>(core_y_i, domain_y_end - out_y0));
  int const valid_out_x =
      tide_max<int>(0, tide_min<int>(core_x_i, domain_x_end - out_x0));
  int const core_elems = valid_out_y * valid_out_x;
  int const core_offset = halo_i * tile_x + halo_i;

  for (int base = tid; base < core_elems; base += block_threads * ILP) {
#pragma unroll
    for (int lane = 0; lane < ILP; ++lane) {
      int const ci = base + lane * block_threads;
      if (ci >= core_elems) {
        continue;
      }
      int const cy = ci / valid_out_x;
      int const cx = ci - cy * valid_out_x;
      int const gy = out_y0 + cy;
      int const gx = out_x0 + cx;
      int64_t const gj = static_cast<int64_t>(gy) * nx_i + gx;
      int64_t const gi = shot_offset + gj;
      int const li = core_offset + cy * tile_x + cx;
      store_field(ey, gi, ey_src[li]);
      store_field(hx, gi, hx_s[li]);
      store_field(hz, gi, hz_s[li]);
      m_ey_x[gi] = m_ey_x_s[li];
      m_ey_z[gi] = m_ey_z_s[li];
      m_hx_z[gi] = m_hx_z_s[li];
      m_hz_x[gi] = m_hz_x_s[li];
    }
  }
}

template <int ILP>
static inline cudaError_t launch_tm_ebisu_fp16s_kernel(
    dim3 const dim_grid, dim3 const dim_block, size_t const shared_bytes,
    coeff_t const *const ca, coeff_t const *const cb, coeff_t const *const cq,
    coeff_t const *const f, __half *const ey, __half *const hx,
    __half *const hz, coeff_t *const m_ey_x, coeff_t *const m_ey_z,
    coeff_t *const m_hx_z, coeff_t *const m_hz_x, coeff_t const *const ay,
    coeff_t const *const ayh, coeff_t const *const ax, coeff_t const *const axh,
    coeff_t const *const by, coeff_t const *const byh, coeff_t const *const bx,
    coeff_t const *const bxh, coeff_t const *const ky, coeff_t const *const kyh,
    coeff_t const *const kx, coeff_t const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    coeff_t *const r, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const k_steps) {
  forward_kernel_ebisu_tb_fp16s<ILP>
      <<<dim_grid, dim_block, shared_bytes>>>(
          ca, cb, cq, f, ey, hx, hz, m_ey_x, m_ey_z, m_hx_z, m_hz_x, ay, ayh,
          ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, sources_i, receivers_i,
          r, core_y, core_x, halo, k_steps);
  return cudaGetLastError();
}

__global__ __launch_bounds__(256) void forward_kernel_h_fp16s(
    coeff_t const *__restrict const cq, __half const *__restrict const ey,
    __half *__restrict const hx, __half *__restrict const hz,
    coeff_t *__restrict const m_ey_x, coeff_t *__restrict const m_ey_z,
    coeff_t const *__restrict const ay, coeff_t const *__restrict const ayh,
    coeff_t const *__restrict const ax, coeff_t const *__restrict const axh,
    coeff_t const *__restrict const by, coeff_t const *__restrict const byh,
    coeff_t const *__restrict const bx, coeff_t const *__restrict const bxh,
    coeff_t const *__restrict const ky, coeff_t const *__restrict const kyh,
    coeff_t const *__restrict const kx, coeff_t const *__restrict const kxh) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (y < kFdPad || x < kFdPad || y >= ny - kFdPad + 1 ||
      x >= nx - kFdPad + 1 || shot_idx >= n_shots) {
    return;
  }

  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = pml_y1 > pml_y0 ? pml_y1 - 1 : pml_y0;
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = pml_x1 > pml_x0 ? pml_x1 - 1 : pml_x0;
  int64_t const shot_offset = shot_idx * shot_numel;
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  float const cq_val = cq_batched ? cq[i] : cq[j];
  DecodingGlobalAccessor<__half> ey_acc(ey, (int)nx);

  if (y < ny - kFdPad) {
    bool const in_pml_y = y < pml_y0h || y >= pml_y1h;
    float dey_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
        ey_acc, shot_offset, (int)y, (int)x, (int)nx, static_cast<float>(rdy));
    if (in_pml_y) {
      m_ey_z[i] = byh[y] * m_ey_z[i] + ayh[y] * dey_dz;
      dey_dz = dey_dz / kyh[y] + m_ey_z[i];
    }
    store_field(hx, i, load_field(hx, i) - cq_val * dey_dz);
  }

  if (x < nx - kFdPad) {
    bool const in_pml_x = x < pml_x0h || x >= pml_x1h;
    float dey_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
        ey_acc, shot_offset, (int)y, (int)x, (int)nx, static_cast<float>(rdx));
    if (in_pml_x) {
      m_ey_x[i] = bxh[x] * m_ey_x[i] + axh[x] * dey_dx;
      dey_dx = dey_dx / kxh[x] + m_ey_x[i];
    }
    store_field(hz, i, load_field(hz, i) + cq_val * dey_dx);
  }
}

template <typename StoreT>
__global__ __launch_bounds__(256) void forward_kernel_e_fp16s(
    coeff_t const *__restrict const ca, coeff_t const *__restrict const cb,
    __half const *__restrict const hx, __half const *__restrict const hz,
    __half *__restrict const ey, coeff_t *__restrict const m_hx_z,
    coeff_t *__restrict const m_hz_x, StoreT *__restrict const ey_store,
    StoreT *__restrict const curl_h_store, coeff_t const *__restrict const ay,
    coeff_t const *__restrict const ayh, coeff_t const *__restrict const ax,
    coeff_t const *__restrict const axh, coeff_t const *__restrict const by,
    coeff_t const *__restrict const byh, coeff_t const *__restrict const bx,
    coeff_t const *__restrict const bxh, coeff_t const *__restrict const ky,
    coeff_t const *__restrict const kyh, coeff_t const *__restrict const kx,
    coeff_t const *__restrict const kxh, bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;

  if (y < kFdPad || x < kFdPad || y >= ny - kFdPad + 1 ||
      x >= nx - kFdPad + 1 || shot_idx >= n_shots) {
    return;
  }

  int64_t const shot_offset = shot_idx * shot_numel;
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  float const ca_val = ca_batched ? ca[i] : ca[j];
  float const cb_val = cb_batched ? cb[i] : cb[j];
  bool const in_pml_y = y < pml_y0 || y >= pml_y1;
  bool const in_pml_x = x < pml_x0 || x >= pml_x1;

  DecodingGlobalAccessor<__half> hz_acc(hz, (int)nx);
  DecodingGlobalAccessor<__half> hx_acc(hx, (int)nx);

  float dhz_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
      hz_acc, shot_offset, (int)y, (int)x, (int)nx, static_cast<float>(rdx));
  float dhx_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
      hx_acc, shot_offset, (int)y, (int)x, (int)nx, static_cast<float>(rdy));

  if (in_pml_x) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
    dhz_dx = dhz_dx / kx[x] + m_hz_x[i];
  }
  if (in_pml_y) {
    m_hx_z[i] = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
    dhx_dz = dhx_dz / ky[y] + m_hx_z[i];
  }

  float const curl_h = dhz_dx - dhx_dz;
  float const ey_prev = load_field(ey, i);

  if (ca_requires_grad && ey_store != nullptr) {
    ey_store[i] = ::tide::encode_snapshot<StoreT, float>(ey_prev);
  }
  if (cb_requires_grad && curl_h_store != nullptr) {
    curl_h_store[i] = ::tide::encode_snapshot<StoreT, float>(curl_h);
  }

  store_field(ey, i, ca_val * ey_prev + cb_val * curl_h);
}

__device__ __forceinline__ coeff_t cpml_tmp_from_m_new(
    coeff_t const m_new, coeff_t const b_val) {
  return b_val != static_cast<coeff_t>(0) ? (m_new / b_val)
                                          : static_cast<coeff_t>(0);
}

__device__ __forceinline__ coeff_t transformed_lambda_h_work_x_exact(
    coeff_t const *__restrict const cb,
    coeff_t const *__restrict const lambda_ey,
    coeff_t const *__restrict const m_lambda_ey_x,
    coeff_t const *__restrict const ax,
    coeff_t const *__restrict const bx,
    coeff_t const *__restrict const kx, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_x0h,
    int64_t const pml_x1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  coeff_t const cb_val = ldg_coeff(cb, cb_batched, i, j);
  coeff_t const g = cb_val * lambda_ey[i];
  if (x < pml_x0h || x >= pml_x1h) {
    coeff_t const bx_val = __ldg(&bx[x]);
    coeff_t const tmp_x = cpml_tmp_from_m_new(m_lambda_ey_x[i], bx_val);
    return g / __ldg(&kx[x]) + __ldg(&ax[x]) * tmp_x;
  }
  return g;
}

__device__ __forceinline__ coeff_t transformed_lambda_h_work_z_exact(
    coeff_t const *__restrict const cb,
    coeff_t const *__restrict const lambda_ey,
    coeff_t const *__restrict const m_lambda_ey_z,
    coeff_t const *__restrict const ay,
    coeff_t const *__restrict const by,
    coeff_t const *__restrict const ky, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_y0h,
    int64_t const pml_y1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  coeff_t const cb_val = ldg_coeff(cb, cb_batched, i, j);
  coeff_t const g = cb_val * lambda_ey[i];
  if (y < pml_y0h || y >= pml_y1h) {
    coeff_t const by_val = __ldg(&by[y]);
    coeff_t const tmp_z = cpml_tmp_from_m_new(m_lambda_ey_z[i], by_val);
    return g / __ldg(&ky[y]) + __ldg(&ay[y]) * tmp_z;
  }
  return g;
}

__device__ __forceinline__ coeff_t transformed_lambda_e_work_x_exact(
    coeff_t const *__restrict const cq,
    coeff_t const *__restrict const lambda_hz,
    coeff_t const *__restrict const m_lambda_hz_x,
    coeff_t const *__restrict const axh,
    coeff_t const *__restrict const bxh,
    coeff_t const *__restrict const kxh, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_x0h,
    int64_t const pml_x1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  coeff_t const cq_val = ldg_coeff(cq, cq_batched, i, j);
  coeff_t const g = cq_val * lambda_hz[i];
  if (x < pml_x0h || x >= pml_x1h) {
    coeff_t const bx_val = __ldg(&bxh[x]);
    coeff_t const tmp_x = cpml_tmp_from_m_new(m_lambda_hz_x[i], bx_val);
    return g / __ldg(&kxh[x]) + __ldg(&axh[x]) * tmp_x;
  }
  return g;
}

__device__ __forceinline__ coeff_t transformed_lambda_e_work_z_exact(
    coeff_t const *__restrict const cq,
    coeff_t const *__restrict const lambda_hx,
    coeff_t const *__restrict const m_lambda_hx_z,
    coeff_t const *__restrict const ayh,
    coeff_t const *__restrict const byh,
    coeff_t const *__restrict const kyh, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_y0h,
    int64_t const pml_y1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  coeff_t const cq_val = ldg_coeff(cq, cq_batched, i, j);
  coeff_t const g = -cq_val * lambda_hx[i];
  if (y < pml_y0h || y >= pml_y1h) {
    coeff_t const by_val = __ldg(&byh[y]);
    coeff_t const tmp_z = cpml_tmp_from_m_new(m_lambda_hx_z[i], by_val);
    return g / __ldg(&kyh[y]) + __ldg(&ayh[y]) * tmp_z;
  }
  return g;
}

template <typename T> struct ConstantOneAccessor2D {
  TIDE_HOST_DEVICE T operator()(int64_t, int64_t) const {
    return static_cast<T>(1);
  }
};

template <typename T> struct CbLambdaEyAccessor2D {
  T const *cb;
  T const *lambda_ey;
  bool cb_is_batched;
  int64_t i;
  int64_t j;
  int64_t nx_val;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    int64_t const shot_index = tide_nd_index(i, dy, dx, nx_val);
    int64_t const grid_index = tide_nd_index_j(j, dy, dx, nx_val);
    return ldg_coeff(cb, cb_is_batched, shot_index, grid_index) *
           lambda_ey[shot_index];
  }
};

template <typename T> struct CqLambdaHzAccessor2D {
  T const *cq;
  T const *lambda_hz;
  bool cq_is_batched;
  int64_t i;
  int64_t j;
  int64_t nx_val;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    int64_t const shot_index = tide_nd_index(i, dy, dx, nx_val);
    int64_t const grid_index = tide_nd_index_j(j, dy, dx, nx_val);
    return ldg_coeff(cq, cq_is_batched, shot_index, grid_index) *
           lambda_hz[shot_index];
  }
};

template <typename T> struct CqLambdaHxAccessor2D {
  T const *cq;
  T const *lambda_hx;
  bool cq_is_batched;
  int64_t i;
  int64_t j;
  int64_t nx_val;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    int64_t const shot_index = tide_nd_index(i, dy, dx, nx_val);
    int64_t const grid_index = tide_nd_index_j(j, dy, dx, nx_val);
    return -ldg_coeff(cq, cq_is_batched, shot_index, grid_index) *
           lambda_hx[shot_index];
  }
};

template <typename T> struct LambdaHWorkXExactAccessor2D {
  T const *cb;
  T const *lambda_ey;
  T const *m_lambda_ey_x;
  T const *ax;
  T const *bx;
  T const *kx;
  int64_t shot_offset;
  int64_t y;
  int64_t x;
  int64_t pml_x0h;
  int64_t pml_x1h;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    return transformed_lambda_h_work_x_exact(cb, lambda_ey, m_lambda_ey_x, ax,
                                             bx, kx, shot_offset, y + dy,
                                             x + dx, pml_x0h, pml_x1h);
  }
};

template <typename T> struct LambdaHWorkZExactAccessor2D {
  T const *cb;
  T const *lambda_ey;
  T const *m_lambda_ey_z;
  T const *ay;
  T const *by;
  T const *ky;
  int64_t shot_offset;
  int64_t y;
  int64_t x;
  int64_t pml_y0h;
  int64_t pml_y1h;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    return transformed_lambda_h_work_z_exact(cb, lambda_ey, m_lambda_ey_z, ay,
                                             by, ky, shot_offset, y + dy,
                                             x + dx, pml_y0h, pml_y1h);
  }
};

template <typename T> struct LambdaEWorkXExactAccessor2D {
  T const *cq;
  T const *lambda_hz;
  T const *m_lambda_hz_x;
  T const *axh;
  T const *bxh;
  T const *kxh;
  int64_t shot_offset;
  int64_t y;
  int64_t x;
  int64_t pml_x0h;
  int64_t pml_x1h;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    return transformed_lambda_e_work_x_exact(cq, lambda_hz, m_lambda_hz_x, axh,
                                             bxh, kxh, shot_offset, y + dy,
                                             x + dx, pml_x0h, pml_x1h);
  }
};

template <typename T> struct LambdaEWorkZExactAccessor2D {
  T const *cq;
  T const *lambda_hx;
  T const *m_lambda_hx_z;
  T const *ayh;
  T const *byh;
  T const *kyh;
  int64_t shot_offset;
  int64_t y;
  int64_t x;
  int64_t pml_y0h;
  int64_t pml_y1h;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    return transformed_lambda_e_work_z_exact(cq, lambda_hx, m_lambda_hx_z, ayh,
                                             byh, kyh, shot_offset, y + dy,
                                             x + dx, pml_y0h, pml_y1h);
  }
};

__global__ void backward_kernel_lambda_h_update_m_exact_fp16s(
    coeff_t const *__restrict const cb,
    coeff_t const *__restrict const lambda_ey,
    coeff_t *__restrict const m_lambda_ey_x,
    coeff_t *__restrict const m_lambda_ey_z,
    coeff_t const *__restrict const by,
    coeff_t const *__restrict const bx,
    BoundaryLaunchLayout const layout) {
  int64_t const boundary_index =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (shot_idx >= n_shots) {
    return;
  }

  int64_t x = 0;
  int64_t y = 0;
  if (!decode_boundary_point(layout, boundary_index, y, x)) {
    return;
  }

  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);
  bool const pml_y = y < pml_y0h || y >= pml_y1h;
  bool const pml_x = x < pml_x0h || x >= pml_x1h;

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  coeff_t const cb_val = ldg_coeff(cb, cb_batched, i, j);
  coeff_t const g = cb_val * lambda_ey[i];
  if (pml_x) {
    coeff_t const tmp_x = m_lambda_ey_x[i] + g;
    m_lambda_ey_x[i] = __ldg(&bx[x]) * tmp_x;
  }
  if (pml_y) {
    coeff_t const tmp_z = m_lambda_ey_z[i] + g;
    m_lambda_ey_z[i] = __ldg(&by[y]) * tmp_z;
  }
}

__global__ void backward_kernel_lambda_h_apply_exact_interior_fp16s(
    coeff_t const *__restrict const cb,
    coeff_t const *__restrict const lambda_ey,
    coeff_t *__restrict const lambda_hx, coeff_t *__restrict const lambda_hz,
    int64_t const y_begin, int64_t const y_end, int64_t const x_begin,
    int64_t const x_end) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + x_begin;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + y_begin;
  int64_t const shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y >= y_end || x >= x_end) {
    return;
  }
  if (y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  ConstantOneAccessor2D<coeff_t> constant_one{};
  CbLambdaEyAccessor2D<coeff_t> g_cb_l{cb, lambda_ey, cb_batched, i, j, nx};
  if (y < ny - kFdPad) {
    lambda_hx[i] -= DIFFY1_ADJ(constant_one, g_cb_l);
  }
  if (x < nx - kFdPad) {
    lambda_hz[i] += DIFFX1_ADJ(constant_one, g_cb_l);
  }
}

__global__ void backward_kernel_lambda_h_apply_exact_boundary_fp16s(
    coeff_t const *__restrict const cb,
    coeff_t const *__restrict const lambda_ey,
    coeff_t const *__restrict const m_lambda_ey_x,
    coeff_t const *__restrict const m_lambda_ey_z,
    coeff_t *__restrict const lambda_hx, coeff_t *__restrict const lambda_hz,
    coeff_t const *__restrict const ay,
    coeff_t const *__restrict const ax,
    coeff_t const *__restrict const by,
    coeff_t const *__restrict const bx,
    coeff_t const *__restrict const ky,
    coeff_t const *__restrict const kx,
    BoundaryLaunchLayout const layout) {
  int64_t const boundary_index =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (shot_idx >= n_shots) {
    return;
  }

  int64_t x = 0;
  int64_t y = 0;
  if (!decode_boundary_point(layout, boundary_index, y, x)) {
    return;
  }

  int64_t const pml_y0h = pml_y0;
  int64_t const pml_y1h = tide_max(pml_y0, pml_y1 - 1);
  int64_t const pml_x0h = pml_x0;
  int64_t const pml_x1h = tide_max(pml_x0, pml_x1 - 1);
  int64_t const shot_offset = shot_idx * shot_numel;
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  ConstantOneAccessor2D<coeff_t> constant_one{};
  LambdaHWorkXExactAccessor2D<coeff_t> work_x_l{
      cb, lambda_ey, m_lambda_ey_x, ax, bx, kx, shot_offset,
      y,  x,         pml_x0h,       pml_x1h};
  LambdaHWorkZExactAccessor2D<coeff_t> work_z_l{
      cb, lambda_ey, m_lambda_ey_z, ay, by, ky, shot_offset,
      y,  x,         pml_y0h,       pml_y1h};
  if (y < ny - kFdPad) {
    lambda_hx[i] -= DIFFY1_ADJ(constant_one, work_z_l);
  }
  if (x < nx - kFdPad) {
    lambda_hz[i] += DIFFX1_ADJ(constant_one, work_x_l);
  }
}

__global__ void backward_kernel_lambda_e_update_m_exact_fp16s(
    coeff_t const *__restrict const cq,
    coeff_t const *__restrict const lambda_hx,
    coeff_t const *__restrict const lambda_hz,
    coeff_t *__restrict const m_lambda_hx_z,
    coeff_t *__restrict const m_lambda_hz_x,
    coeff_t const *__restrict const byh,
    coeff_t const *__restrict const bxh,
    BoundaryLaunchLayout const layout) {
  int64_t const boundary_index =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (shot_idx >= n_shots) {
    return;
  }

  int64_t x = 0;
  int64_t y = 0;
  if (!decode_boundary_point(layout, boundary_index, y, x)) {
    return;
  }

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;
  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  coeff_t const cq_val = ldg_coeff(cq, cq_batched, i, j);
  if (pml_x) {
    coeff_t const g_x = cq_val * lambda_hz[i];
    coeff_t const tmp_x = m_lambda_hz_x[i] + g_x;
    m_lambda_hz_x[i] = __ldg(&bxh[x]) * tmp_x;
  }
  if (pml_y) {
    coeff_t const g_z = -cq_val * lambda_hx[i];
    coeff_t const tmp_z = m_lambda_hx_z[i] + g_z;
    m_lambda_hx_z[i] = __ldg(&byh[y]) * tmp_z;
  }
}

__global__ void backward_kernel_lambda_e_apply_exact_interior_fp16s(
    coeff_t const *__restrict const ca,
    coeff_t const *__restrict const cq,
    coeff_t const *__restrict const lambda_hx,
    coeff_t const *__restrict const lambda_hz,
    coeff_t *__restrict const lambda_ey,
    __half const *__restrict const ey_store,
    __half const *__restrict const curl_h_store,
    coeff_t *__restrict const grad_ca_shot,
    coeff_t *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val,
    int64_t const y_begin, int64_t const y_end, int64_t const x_begin,
    int64_t const x_end) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + x_begin;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + y_begin;
  int64_t const shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y >= y_end || x >= x_end) {
    return;
  }
  if (y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  coeff_t const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<coeff_t> constant_one{};
  CqLambdaHzAccessor2D<coeff_t> g_x_l{cq, lambda_hz, cq_batched, i, j, nx};
  CqLambdaHxAccessor2D<coeff_t> g_z_l{cq, lambda_hx, cq_batched, i, j, nx};
  coeff_t const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, g_x_l) + DIFFYH1_ADJ(constant_one, g_z_l);
  coeff_t const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
  if (ca_requires_grad && ey_store != nullptr) {
    coeff_t const ey_n = ::tide::decode_snapshot<__half, coeff_t>(ey_store[i]);
    grad_ca_shot[i] +=
        lambda_ey_curr * ey_n * step_ratio_to_field(step_ratio_val);
  }
  if (cb_requires_grad && curl_h_store != nullptr) {
    coeff_t const curl_h_n =
        ::tide::decode_snapshot<__half, coeff_t>(curl_h_store[i]);
    grad_cb_shot[i] +=
        lambda_ey_curr * curl_h_n * step_ratio_to_field(step_ratio_val);
  }
}

__global__ void backward_kernel_lambda_e_apply_exact_boundary_fp16s(
    coeff_t const *__restrict const ca,
    coeff_t const *__restrict const cq,
    coeff_t const *__restrict const lambda_hx,
    coeff_t const *__restrict const lambda_hz,
    coeff_t const *__restrict const m_lambda_hx_z,
    coeff_t const *__restrict const m_lambda_hz_x,
    coeff_t *__restrict const lambda_ey,
    __half const *__restrict const ey_store,
    __half const *__restrict const curl_h_store,
    coeff_t *__restrict const grad_ca_shot,
    coeff_t *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val,
    coeff_t const *__restrict const ayh,
    coeff_t const *__restrict const axh,
    coeff_t const *__restrict const byh,
    coeff_t const *__restrict const bxh,
    coeff_t const *__restrict const kyh,
    coeff_t const *__restrict const kxh,
    BoundaryLaunchLayout const layout) {
  int64_t const boundary_index =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (shot_idx >= n_shots) {
    return;
  }

  int64_t x = 0;
  int64_t y = 0;
  if (!decode_boundary_point(layout, boundary_index, y, x)) {
    return;
  }

  bool const pml_y = y < pml_y0 || y >= pml_y1;
  bool const pml_x = x < pml_x0 || x >= pml_x1;
  int64_t const shot_offset = shot_idx * shot_numel;
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  coeff_t const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<coeff_t> constant_one{};
  LambdaEWorkXExactAccessor2D<coeff_t> work_x_l{
      cq, lambda_hz, m_lambda_hz_x, axh, bxh, kxh, shot_offset,
      y,  x,         pml_x0,        pml_x1};
  LambdaEWorkZExactAccessor2D<coeff_t> work_z_l{
      cq, lambda_hx, m_lambda_hx_z, ayh, byh, kyh, shot_offset,
      y,  x,         pml_y0,        pml_y1};
  coeff_t const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, work_x_l) +
      DIFFYH1_ADJ(constant_one, work_z_l);
  coeff_t const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
  if (!pml_y && !pml_x && ca_requires_grad && ey_store != nullptr) {
    coeff_t const ey_n = ::tide::decode_snapshot<__half, coeff_t>(ey_store[i]);
    grad_ca_shot[i] +=
        lambda_ey_curr * ey_n * step_ratio_to_field(step_ratio_val);
  }
  if (!pml_y && !pml_x && cb_requires_grad && curl_h_store != nullptr) {
    coeff_t const curl_h_n =
        ::tide::decode_snapshot<__half, coeff_t>(curl_h_store[i]);
    grad_cb_shot[i] +=
        lambda_ey_curr * curl_h_n * step_ratio_to_field(step_ratio_val);
  }
}

__global__ void combine_grad(coeff_t *__restrict const grad,
                             coeff_t const *__restrict const grad_shot) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + kFdPad;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + kFdPad;
  if (x >= nx - kFdPad + 1 || y >= ny - kFdPad + 1) {
    return;
  }
  int64_t const j = y * nx + x;
  float sum = 0.0f;
  for (int64_t shot = 0; shot < n_shots; ++shot) {
    sum += grad_shot[shot * shot_numel + j];
  }
  grad[j] = sum;
}

} // namespace

extern "C" void FUNC(forward)(
    coeff_t const *const ca, coeff_t const *const cb, coeff_t const *const cq,
    coeff_t const *const f, __half *const ey, __half *const hx,
    __half *const hz, coeff_t *const m_ey_x, coeff_t *const m_ey_z,
    coeff_t *const m_hx_z, coeff_t *const m_hz_x,
    coeff_t const *const debye_a, coeff_t const *const debye_b,
    coeff_t const *const debye_cp, coeff_t *const polarization,
    coeff_t *const ey_prev, coeff_t *const r, int64_t const n_poles_h,
    coeff_t const *const ay, coeff_t const *const by,
    coeff_t const *const ayh, coeff_t const *const byh,
    coeff_t const *const ax, coeff_t const *const bx,
    coeff_t const *const axh, coeff_t const *const bxh,
    coeff_t const *const ky, coeff_t const *const kyh,
    coeff_t const *const kx, coeff_t const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    coeff_t const rdy_h, coeff_t const rdx_h, coeff_t const dt_h,
    int64_t const nt, int64_t const n_shots_h, int64_t const ny_h,
    int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    bool const has_dispersion,
    bool const ca_batched_h, bool const cb_batched_h, bool const cq_batched_h,
    int64_t const start_t, int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h, int64_t const n_threads,
    int64_t const device) {
  cudaSetDevice(static_cast<int>(device));
  require_fp16_capability(device);
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;
  (void)debye_a;
  (void)debye_b;
  (void)debye_cp;
  (void)polarization;
  (void)ey_prev;
  (void)n_poles_h;
  (void)has_dispersion;

  int64_t const shot_numel_h = ny_h * nx_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);

  TMEbisuRuntimeConfig const ebisu_cfg = read_tm_ebisu_runtime_config();
  bool const debug_path = read_env_flag("TIDE_TM_DEBUG_PATH");
  if (debug_path && ebisu_cfg.enabled) {
    fprintf(stderr,
            "TIDE TM fp16s ebisu fallback: disabled for this precision path disp=%d\n",
            has_dispersion ? 1 : 0);
  }

  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h_fp16s<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);
    forward_kernel_e_fp16s<__half><<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x, nullptr, nullptr, ay, ayh, ax, axh,
        by, byh, bx, bxh, ky, kyh, kx, kxh, false, false);
    if (n_sources_per_shot_h > 0) {
      add_sources_ey_fp16s<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey_fp16s<<<launch_cfg.dimGridReceivers,
                                  launch_cfg.dimBlockReceivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  }

  if (debug_path) {
    fprintf(stderr, "TIDE TM fp16s path: baseline\n");
  }
  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(forward_with_storage)(
    coeff_t const *const ca, coeff_t const *const cb, coeff_t const *const cq,
    coeff_t const *const f, __half *const ey, __half *const hx,
    __half *const hz, coeff_t *const m_ey_x, coeff_t *const m_ey_z,
    coeff_t *const m_hx_z, coeff_t *const m_hz_x, coeff_t *const r,
    void *const ey_store_1, void *const ey_store_3,
    char const *const *const ey_filenames, void *const curl_store_1,
    void *const curl_store_3, char const *const *const curl_filenames,
    coeff_t const *const ay, coeff_t const *const by,
    coeff_t const *const ayh, coeff_t const *const byh,
    coeff_t const *const ax, coeff_t const *const bx,
    coeff_t const *const axh, coeff_t const *const bxh,
    coeff_t const *const ky, coeff_t const *const kyh,
    coeff_t const *const kx, coeff_t const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    coeff_t const rdy_h, coeff_t const rdx_h, coeff_t const dt_h,
    int64_t const nt, int64_t const n_shots_h, int64_t const ny_h,
    int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    int64_t const storage_mode_h, int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp_h, bool const ca_requires_grad,
    bool const cb_requires_grad, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device) {
  cudaSetDevice(static_cast<int>(device));
  require_fp16_capability(device);
  (void)n_threads;

  if (storage_format_h != STORAGE_FORMAT_FP16) {
    fprintf(stderr,
            "TM2D fp16_scaled expected STORAGE_FORMAT_FP16, got %lld.\n",
            static_cast<long long>(storage_format_h));
    abort();
  }

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      static_cast<size_t>(shot_bytes_uncomp_h) * static_cast<size_t>(n_shots_h);
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);

  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  FILE *fp_ey = nullptr;
  FILE *fp_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad)
      fp_ey = fopen(ey_filenames[0], "wb");
    if (cb_requires_grad)
      fp_curl = fopen(curl_filenames[0], "wb");
  }

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h_fp16s<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);

    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    if (store_ey || store_curl) {
      int64_t const step_idx = t / step_ratio_h;
      size_t const store1_offset =
          ring_storage_offset_bytes(step_idx, storage_mode_h, bytes_per_step_store);
      size_t const store3_offset = cpu_linear_storage_offset_bytes(
          step_idx, storage_mode_h, bytes_per_step_store);

      void *const ey_store_1_t = (uint8_t *)ey_store_1 + store1_offset;
      void *const ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;
      void *const curl_store_1_t = (uint8_t *)curl_store_1 + store1_offset;
      void *const curl_store_3_t = (uint8_t *)curl_store_3 + store3_offset;

      forward_kernel_e_fp16s<__half><<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
          store_ey ? (__half *)ey_store_1_t : nullptr,
          store_curl ? (__half *)curl_store_1_t : nullptr, ay, ayh, ax, axh, by,
          byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);

      if (storage_mode_h == STORAGE_CPU) {
        if (store_ey) {
          tide::cuda_check_or_abort(cudaMemcpy(ey_store_3_t, ey_store_1_t,
                                               bytes_per_step_store,
                                               cudaMemcpyDeviceToHost),
                                    __FILE__, __LINE__);
        }
        if (store_curl) {
          tide::cuda_check_or_abort(cudaMemcpy(curl_store_3_t, curl_store_1_t,
                                               bytes_per_step_store,
                                               cudaMemcpyDeviceToHost),
                                    __FILE__, __LINE__);
        }
      } else {
        if (store_ey) {
          storage_save_snapshot_gpu(ey_store_1_t, ey_store_3_t, fp_ey,
                                    storage_mode_h, step_idx,
                                    (size_t)shot_bytes_uncomp_h,
                                    (size_t)n_shots_h);
        }
        if (store_curl) {
          storage_save_snapshot_gpu(curl_store_1_t, curl_store_3_t, fp_curl,
                                    storage_mode_h, step_idx,
                                    (size_t)shot_bytes_uncomp_h,
                                    (size_t)n_shots_h);
        }
      }
    } else {
      forward_kernel_e_fp16s<__half><<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, nullptr, nullptr, ay, ayh, ax,
          axh, by, byh, bx, bxh, ky, kyh, kx, kxh, false, false);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey_fp16s<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey_fp16s<<<launch_cfg.dimGridReceivers,
                                  launch_cfg.dimBlockReceivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  }

  if (fp_ey != nullptr)
    fclose(fp_ey);
  if (fp_curl != nullptr)
    fclose(fp_curl);

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(backward)(
    coeff_t const *const ca, coeff_t const *const cb, coeff_t const *const cq,
    coeff_t const *const grad_r, coeff_t *const lambda_ey,
    coeff_t *const lambda_hx, coeff_t *const lambda_hz,
    coeff_t *const m_lambda_ey_x,
    coeff_t *const m_lambda_ey_z, coeff_t *const m_lambda_hx_z,
    coeff_t *const m_lambda_hz_x, void *const ey_store_1, void *const ey_store_3,
    char const *const *const ey_filenames, void *const curl_store_1,
    void *const curl_store_3, char const *const *const curl_filenames,
    coeff_t *const grad_f, coeff_t *const grad_ca, coeff_t *const grad_cb,
    coeff_t *const grad_ca_shot, coeff_t *const grad_cb_shot,
    coeff_t const *const ay, coeff_t const *const by,
    coeff_t const *const ayh, coeff_t const *const byh,
    coeff_t const *const ax, coeff_t const *const bx,
    coeff_t const *const axh, coeff_t const *const bxh,
    coeff_t const *const ky, coeff_t const *const kyh,
    coeff_t const *const kx, coeff_t const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    coeff_t const rdy_h, coeff_t const rdx_h, coeff_t const dt_h,
    int64_t const nt, int64_t const n_shots_h, int64_t const ny_h,
    int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    int64_t const storage_mode_h, int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp_h, bool const ca_requires_grad,
    bool const cb_requires_grad, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device) {
  cudaSetDevice(static_cast<int>(device));
  require_fp16_capability(device);
  (void)dt_h;
  (void)n_threads;

  if (storage_format_h != STORAGE_FORMAT_FP16) {
    fprintf(stderr,
            "TM2D fp16_scaled expected STORAGE_FORMAT_FP16, got %lld.\n",
            static_cast<long long>(storage_format_h));
    abort();
  }

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      static_cast<size_t>(shot_bytes_uncomp_h) * static_cast<size_t>(n_shots_h);
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);

  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);
  dim3 const dimBlock = launch_cfg.dimBlock;
  int64_t interior_x_begin = pml_x0_h + kFdPad;
  int64_t interior_x_end = pml_x1_h - kFdPad;
  int64_t interior_y_begin = pml_y0_h + kFdPad;
  int64_t interior_y_end = pml_y1_h - kFdPad;
  if (interior_x_begin < kFdPad)
    interior_x_begin = kFdPad;
  if (interior_y_begin < kFdPad)
    interior_y_begin = kFdPad;
  int64_t const domain_x_end = nx_h - kFdPad + 1;
  int64_t const domain_y_end = ny_h - kFdPad + 1;
  if (interior_x_end > domain_x_end)
    interior_x_end = domain_x_end;
  if (interior_y_end > domain_y_end)
    interior_y_end = domain_y_end;
  int64_t const domain_x_begin = kFdPad;
  int64_t const domain_y_begin = kFdPad;
  bool const has_interior =
      interior_x_begin < interior_x_end && interior_y_begin < interior_y_end;
  dim3 dimGridInterior(1, 1, to_dim_u32(n_shots_h));
  if (has_interior) {
    int64_t const interior_gridx =
        (interior_x_end - interior_x_begin + dimBlock.x - 1) / dimBlock.x;
    int64_t const interior_gridy =
        (interior_y_end - interior_y_begin + dimBlock.y - 1) / dimBlock.y;
    dimGridInterior = dim3(to_dim_u32(interior_gridx),
                           to_dim_u32(interior_gridy),
                           to_dim_u32(n_shots_h));
  }
  BoundaryLaunchLayout const boundary_layout = make_boundary_launch_layout(
      domain_y_begin, domain_y_end, domain_x_begin, domain_x_end,
      interior_y_begin, interior_y_end, interior_x_begin, interior_x_end,
      has_interior);
  dim3 dimBlockBoundary(256, 1, 1);
  dim3 dimGridBoundary(1, to_dim_u32(n_shots_h), 1);
  if (boundary_layout.total_count > 0) {
    dimGridBoundary.x =
        (boundary_layout.total_count + dimBlockBoundary.x - 1) /
        dimBlockBoundary.x;
  }

  FILE *fp_ey = nullptr;
  FILE *fp_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad)
      fp_ey = fopen(ey_filenames[0], "rb");
    if (cb_requires_grad)
      fp_curl = fopen(curl_filenames[0], "rb");
  }

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey_fp16s<<<launch_cfg.dimGridReceivers,
                                     launch_cfg.dimBlockReceivers>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }
    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources_fp16s<<<launch_cfg.dimGridSources,
                                        launch_cfg.dimBlockSources>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, lambda_ey, sources_i);
    }

    int64_t const store_idx = t / step_ratio_h;
    bool const do_grad = (t % step_ratio_h) == 0;
    bool const grad_ey = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;

    size_t const store1_offset =
        ring_storage_offset_bytes(store_idx, storage_mode_h, bytes_per_step_store);
    size_t const store3_offset = cpu_linear_storage_offset_bytes(
        store_idx, storage_mode_h, bytes_per_step_store);
    void *const ey_store_1_t = (uint8_t *)ey_store_1 + store1_offset;
    void *const ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;
    void *const curl_store_1_t = (uint8_t *)curl_store_1 + store1_offset;
    void *const curl_store_3_t = (uint8_t *)curl_store_3 + store3_offset;

    if (storage_mode_h == STORAGE_CPU && (grad_ey || grad_curl)) {
      if (grad_ey) {
        tide::cuda_check_or_abort(cudaMemcpy(ey_store_1_t, ey_store_3_t,
                                             bytes_per_step_store,
                                             cudaMemcpyHostToDevice),
                                  __FILE__, __LINE__);
      }
      if (grad_curl) {
        tide::cuda_check_or_abort(cudaMemcpy(curl_store_1_t, curl_store_3_t,
                                             bytes_per_step_store,
                                             cudaMemcpyHostToDevice),
                                  __FILE__, __LINE__);
      }
    } else if (storage_mode_h == STORAGE_DISK) {
      if (grad_ey) {
        storage_load_snapshot_gpu(ey_store_1_t, ey_store_3_t, fp_ey,
                                  storage_mode_h, store_idx,
                                  (size_t)shot_bytes_uncomp_h,
                                  (size_t)n_shots_h);
      }
      if (grad_curl) {
        storage_load_snapshot_gpu(curl_store_1_t, curl_store_3_t, fp_curl,
                                  storage_mode_h, store_idx,
                                  (size_t)shot_bytes_uncomp_h,
                                  (size_t)n_shots_h);
      }
    }

    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_h_update_m_exact_fp16s<<<dimGridBoundary,
                                                       dimBlockBoundary>>>(
          cb, lambda_ey, m_lambda_ey_x, m_lambda_ey_z, by, bx,
          boundary_layout);
    }
    if (has_interior) {
      backward_kernel_lambda_h_apply_exact_interior_fp16s<<<dimGridInterior,
                                                             dimBlock>>>(
          cb, lambda_ey, lambda_hx, lambda_hz, interior_y_begin,
          interior_y_end, interior_x_begin, interior_x_end);
    }
    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_h_apply_exact_boundary_fp16s<<<dimGridBoundary,
                                                             dimBlockBoundary>>>(
          cb, lambda_ey, m_lambda_ey_x, m_lambda_ey_z, lambda_hx, lambda_hz, ay,
          ax, by, bx, ky, kx, boundary_layout);
    }

    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_e_update_m_exact_fp16s<<<dimGridBoundary,
                                                       dimBlockBoundary>>>(
          cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x, byh, bxh,
          boundary_layout);
    }

    if (grad_ey || grad_curl) {
      if (has_interior) {
        backward_kernel_lambda_e_apply_exact_interior_fp16s<<<dimGridInterior,
                                                               dimBlock>>>(
            ca, cq, lambda_hx, lambda_hz, lambda_ey,
            grad_ey ? (__half const *)ey_store_1_t : nullptr,
            grad_curl ? (__half const *)curl_store_1_t : nullptr, grad_ca_shot,
            grad_cb_shot, grad_ey, grad_curl, step_ratio_h, interior_y_begin,
            interior_y_end, interior_x_begin, interior_x_end);
      }
      if (boundary_layout.total_count > 0) {
        backward_kernel_lambda_e_apply_exact_boundary_fp16s<<<dimGridBoundary,
                                                               dimBlockBoundary>>>(
            ca, cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x,
            lambda_ey, grad_ey ? (__half const *)ey_store_1_t : nullptr,
            grad_curl ? (__half const *)curl_store_1_t : nullptr, grad_ca_shot,
            grad_cb_shot, grad_ey, grad_curl, step_ratio_h, ayh, axh, byh, bxh,
            kyh, kxh, boundary_layout);
      }
    } else {
      if (has_interior) {
        backward_kernel_lambda_e_apply_exact_interior_fp16s<<<dimGridInterior,
                                                               dimBlock>>>(
            ca, cq, lambda_hx, lambda_hz, lambda_ey, nullptr, nullptr,
            grad_ca_shot, grad_cb_shot, false, false, 1, interior_y_begin,
            interior_y_end, interior_x_begin, interior_x_end);
      }
      if (boundary_layout.total_count > 0) {
        backward_kernel_lambda_e_apply_exact_boundary_fp16s<<<dimGridBoundary,
                                                               dimBlockBoundary>>>(
            ca, cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x,
            lambda_ey, nullptr, nullptr, grad_ca_shot, grad_cb_shot, false,
            false, 1, ayh, axh, byh, bxh, kyh, kxh, boundary_layout);
      }
    }
  }

  if (fp_ey != nullptr)
    fclose(fp_ey);
  if (fp_curl != nullptr)
    fclose(fp_curl);

  dim3 const dimBlockCombine(32, 32, 1);
  dim3 const dimGridCombine(
      (nx_h - 2 * kFdPad + dimBlockCombine.x - 1) / dimBlockCombine.x,
      (ny_h - 2 * kFdPad + dimBlockCombine.y - 1) / dimBlockCombine.y, 1);
  if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGridCombine, dimBlockCombine>>>(grad_ca, grad_ca_shot);
  }
  if (cb_requires_grad && !cb_batched_h) {
    combine_grad<<<dimGridCombine, dimBlockCombine>>>(grad_cb, grad_cb_shot);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

} // namespace FUNC(Inst)
