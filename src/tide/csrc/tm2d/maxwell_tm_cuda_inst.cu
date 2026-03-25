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
#include <vector>
#include "staggered_grid.h"

namespace FUNC(Inst) {
using tide_field_t = TIDE_DTYPE;
using tide_scalar_t = tide_field_t;
constexpr bool kFieldIsHalf = false;
constexpr int kFdPad = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;
#ifndef TIDE_TM_BLOCK_X
#define TIDE_TM_BLOCK_X 32
#endif
#ifndef TIDE_TM_BLOCK_Y
#define TIDE_TM_BLOCK_Y 8
#endif

namespace {
struct ScopedEventArray {
  cudaEvent_t events[NUM_BUFFERS]{};

  ~ScopedEventArray() {
    for (int i = 0; i < NUM_BUFFERS; ++i) {
      if (events[i] != nullptr) {
        cudaEventDestroy(events[i]);
      }
    }
  }
};

static inline cudaStream_t resolve_cuda_stream(void *const stream_handle) {
  return stream_handle != nullptr ? reinterpret_cast<cudaStream_t>(stream_handle)
                                  : static_cast<cudaStream_t>(0);
}

__device__ __forceinline__ TIDE_DTYPE step_ratio_to_field(
    int64_t const step_ratio_val) {
  return static_cast<TIDE_DTYPE>(static_cast<tide_scalar_t>(step_ratio_val));
}

__device__ __forceinline__ TIDE_DTYPE ldg_coeff(
    TIDE_DTYPE const *__restrict const coeff, bool const coeff_batched,
    int64_t const shot_index, int64_t const grid_index) {
  return __ldg(coeff + (coeff_batched ? shot_index : grid_index));
}

template <typename T>
__host__ __device__ __forceinline__ T tide_min(T a, T b) {
  return a < b ? a : b;
}

static inline bool read_env_flag(char const *name) {
  char const *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
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

  if (layout.domain_width <= 0 || domain_y_end <= domain_y_begin) {
    return layout;
  }

  if (!has_interior) {
    layout.interior_x_begin = domain_x_begin;
    layout.interior_x_end = domain_x_begin;
    layout.interior_y_begin = domain_y_end;
    layout.interior_y_end = domain_y_end;
    layout.top_count = (domain_y_end - domain_y_begin) * layout.domain_width;
    layout.total_count = layout.top_count;
    return layout;
  }

  interior_x_begin = tide_max<int64_t>(interior_x_begin, domain_x_begin);
  interior_x_end = tide_min<int64_t>(interior_x_end, domain_x_end);
  interior_y_begin = tide_max<int64_t>(interior_y_begin, domain_y_begin);
  interior_y_end = tide_min<int64_t>(interior_y_end, domain_y_end);
  interior_x_end = tide_max<int64_t>(interior_x_end, interior_x_begin);
  interior_y_end = tide_max<int64_t>(interior_y_end, interior_y_begin);

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

struct DeviceConstantCache2D {
  bool initialized = false;
  tide_scalar_t rdy_h = 0;
  tide_scalar_t rdx_h = 0;
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

static inline void sync_device_constants_if_needed(
    DeviceConstantCache2D &cache, tide_scalar_t const rdy_h,
    tide_scalar_t const rdx_h, int64_t const n_shots_h, int64_t const ny_h,
    int64_t const nx_h, int64_t const shot_numel_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, bool const ca_batched_h, bool const cb_batched_h,
    bool const cq_batched_h, int64_t const device) {
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

static inline unsigned int to_dim_u32(int64_t value) {
  return static_cast<unsigned int>(value > 0 ? value : 1);
}

static inline TMForwardLaunchConfig make_tm_forward_launch_config(
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h) {
  TMForwardLaunchConfig cfg{};
  cfg.dimBlock = dim3(TIDE_TM_BLOCK_X, TIDE_TM_BLOCK_Y, 1);

  int64_t const gridx =
      (nx_h - 2 * kFdPad + 2 + cfg.dimBlock.x - 1) / cfg.dimBlock.x;
  int64_t const gridy =
      (ny_h - 2 * kFdPad + 2 + cfg.dimBlock.y - 1) / cfg.dimBlock.y;
  cfg.dimGrid =
      dim3(to_dim_u32(gridx), to_dim_u32(gridy), to_dim_u32(n_shots_h));

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

static inline size_t ring_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode_h,
    size_t const bytes_per_step_store) {
  if (storage_mode_h == STORAGE_DEVICE) {
    return (size_t)step_idx * bytes_per_step_store;
  }
  if (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) {
    return (size_t)(step_idx % NUM_BUFFERS) * bytes_per_step_store;
  }
  return 0;
}

static inline size_t cpu_linear_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode_h,
    size_t const bytes_per_step_store) {
  if (storage_mode_h == STORAGE_CPU) {
    return (size_t)step_idx * bytes_per_step_store;
  }
  return 0;
}

static inline size_t host_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode_h,
    size_t const bytes_per_step_store) {
  if (storage_mode_h == STORAGE_DISK) {
    return (size_t)(step_idx % NUM_BUFFERS) * bytes_per_step_store;
  }
  return cpu_linear_storage_offset_bytes(step_idx, storage_mode_h,
                                         bytes_per_step_store);
}

// Forward kernel: Update H fields (Hx and Hz) using unsplit path.
__global__ __launch_bounds__(256) void forward_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const ey, TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz, TIDE_DTYPE *__restrict const m_ey_x,
    TIDE_DTYPE *__restrict const m_ey_z, TIDE_DTYPE const *__restrict const ay,
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
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  ::tide::GridParams<TIDE_DTYPE> params = {
      ay,      ayh,   ax,    axh,        by,     byh,    bx,
      bxh,     ky,    kyh,   kx,         kxh,    static_cast<TIDE_DTYPE>(rdy),
      static_cast<TIDE_DTYPE>(rdx),
      n_shots, ny,    nx,    shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  false, false, cq_batched};
  ::tide::forward_kernel_h_core<TIDE_DTYPE, TIDE_STENCIL>(
      params, cq, ey, hx, hz, m_ey_x, m_ey_z, y, x, shot_idx);
}

// Forward kernel: Update E field (Ey) using unsplit path.
__global__ __launch_bounds__(256) void forward_kernel_e(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
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
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  ::tide::GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,
      static_cast<TIDE_DTYPE>(rdy), static_cast<TIDE_DTYPE>(rdx),
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false};
  ::tide::forward_kernel_e_core<TIDE_DTYPE, TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, y, x, shot_idx);
}

__global__ __launch_bounds__(256) void forward_kernel_e_debye(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const ey_prev,
    TIDE_DTYPE const *__restrict const debye_cp,
    TIDE_DTYPE *__restrict const polarization, int64_t const n_poles_h,
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
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  int const FD_PAD = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;
  if (y < FD_PAD || x < FD_PAD || y >= ny - FD_PAD + 1 || x >= nx - FD_PAD + 1 ||
      shot_idx >= n_shots) {
    return;
  }
  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  ey_prev[i] = ey[i];

  ::tide::GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,
      static_cast<TIDE_DTYPE>(rdy), static_cast<TIDE_DTYPE>(rdx),
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false};
  ::tide::forward_kernel_e_core<TIDE_DTYPE, TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, y, x, shot_idx);

  TIDE_DTYPE pol_term = 0;
  for (int64_t pole = 0; pole < n_poles_h; ++pole) {
    int64_t const coeff_idx = pole * shot_numel + j;
    int64_t const pol_idx = (shot_idx * n_poles_h + pole) * shot_numel + j;
    pol_term += debye_cp[coeff_idx] * polarization[pol_idx];
  }
  ey[i] += pol_term;
}

__global__ __launch_bounds__(256) void update_polarization_debye(
    TIDE_DTYPE const *__restrict const ey_prev,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const debye_a,
    TIDE_DTYPE const *__restrict const debye_b,
    TIDE_DTYPE *__restrict const polarization, int64_t const n_poles_h) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (x >= nx || y >= ny || shot_idx >= n_shots) {
    return;
  }
  int64_t const j = y * nx + x;
  int64_t const field_idx = shot_idx * shot_numel + j;
  TIDE_DTYPE const e_sum = ey[field_idx] + ey_prev[field_idx];
  for (int64_t pole = 0; pole < n_poles_h; ++pole) {
    int64_t const coeff_idx = pole * shot_numel + j;
    int64_t const pol_idx = (shot_idx * n_poles_h + pole) * shot_numel + j;
    polarization[pol_idx] =
        debye_a[coeff_idx] * polarization[pol_idx] +
        debye_b[coeff_idx] * e_sum;
  }
}

// Forward kernel with snapshot storage (fp32/fp64 store).
__global__ __launch_bounds__(256) void forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const ey_store, TIDE_DTYPE *__restrict const curl_h_store,
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
    TIDE_DTYPE const *__restrict const kxh, bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  ::tide::GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,
      static_cast<TIDE_DTYPE>(rdy), static_cast<TIDE_DTYPE>(rdx),
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false};
  ::tide::forward_kernel_e_with_storage_core<TIDE_DTYPE, TIDE_DTYPE, TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_store, curl_h_store,
      ca_requires_grad, cb_requires_grad, y, x, shot_idx);
}

// Forward kernel with snapshot storage (bf16 store).
__global__ __launch_bounds__(256) void forward_kernel_e_with_storage_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    __nv_bfloat16 *__restrict const ey_store,
    __nv_bfloat16 *__restrict const curl_h_store,
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
    TIDE_DTYPE const *__restrict const kxh, bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  ::tide::GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,
      static_cast<TIDE_DTYPE>(rdy), static_cast<TIDE_DTYPE>(rdx),
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false};
  ::tide::forward_kernel_e_with_storage_core<TIDE_DTYPE, __nv_bfloat16,
                                           TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_store, curl_h_store,
      ca_requires_grad, cb_requires_grad, y, x, shot_idx);
}

// Exact CPML adjoint helpers: update memory terms first, then apply transposed
// operators while reconstructing transformed fluxes directly from m_new + field.
__device__ __forceinline__ TIDE_DTYPE cpml_tmp_from_m_new(
    TIDE_DTYPE const m_new, TIDE_DTYPE const b_val) {
  return b_val != (TIDE_DTYPE)0 ? (m_new / b_val) : (TIDE_DTYPE)0;
}

__device__ __forceinline__ TIDE_DTYPE transformed_lambda_h_work_x_exact(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const m_lambda_ey_x,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kx, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_x0h,
    int64_t const pml_x1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  TIDE_DTYPE const cb_val = ldg_coeff(cb, cb_batched, i, j);
  TIDE_DTYPE const g = cb_val * lambda_ey[i];
  if (x < pml_x0h || x >= pml_x1h) {
    TIDE_DTYPE const bx_val = __ldg(&bx[x]);
    TIDE_DTYPE const tmp_x = cpml_tmp_from_m_new(m_lambda_ey_x[i], bx_val);
    return g / __ldg(&kx[x]) + __ldg(&ax[x]) * tmp_x;
  }
  return g;
}

__device__ __forceinline__ TIDE_DTYPE transformed_lambda_h_work_z_exact(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const m_lambda_ey_z,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ky, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_y0h,
    int64_t const pml_y1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  TIDE_DTYPE const cb_val = ldg_coeff(cb, cb_batched, i, j);
  TIDE_DTYPE const g = cb_val * lambda_ey[i];
  if (y < pml_y0h || y >= pml_y1h) {
    TIDE_DTYPE const by_val = __ldg(&by[y]);
    TIDE_DTYPE const tmp_z = cpml_tmp_from_m_new(m_lambda_ey_z[i], by_val);
    return g / __ldg(&ky[y]) + __ldg(&ay[y]) * tmp_z;
  }
  return g;
}

__device__ __forceinline__ TIDE_DTYPE transformed_lambda_e_work_x_exact(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE const *__restrict const m_lambda_hz_x,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kxh, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_x0h,
    int64_t const pml_x1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  TIDE_DTYPE const cq_val = ldg_coeff(cq, cq_batched, i, j);
  TIDE_DTYPE const g = cq_val * lambda_hz[i];
  if (x < pml_x0h || x >= pml_x1h) {
    TIDE_DTYPE const bx_val = __ldg(&bxh[x]);
    TIDE_DTYPE const tmp_x = cpml_tmp_from_m_new(m_lambda_hz_x[i], bx_val);
    return g / __ldg(&kxh[x]) + __ldg(&axh[x]) * tmp_x;
  }
  return g;
}

__device__ __forceinline__ TIDE_DTYPE transformed_lambda_e_work_z_exact(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const m_lambda_hx_z,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const kyh, int64_t const shot_offset,
    int64_t const y, int64_t const x, int64_t const pml_y0h,
    int64_t const pml_y1h) {
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  TIDE_DTYPE const cq_val = ldg_coeff(cq, cq_batched, i, j);
  TIDE_DTYPE const g = -cq_val * lambda_hx[i];
  if (y < pml_y0h || y >= pml_y1h) {
    TIDE_DTYPE const by_val = __ldg(&byh[y]);
    TIDE_DTYPE const tmp_z = cpml_tmp_from_m_new(m_lambda_hx_z[i], by_val);
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
  bool cb_batched;
  int64_t i;
  int64_t j;
  int64_t nx;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    int64_t const shot_index = tide_nd_index(i, dy, dx, nx);
    int64_t const grid_index = tide_nd_index_j(j, dy, dx, nx);
    return ldg_coeff(cb, cb_batched, shot_index, grid_index) *
           lambda_ey[shot_index];
  }
};

template <typename T> struct CqLambdaHzAccessor2D {
  T const *cq;
  T const *lambda_hz;
  bool cq_batched;
  int64_t i;
  int64_t j;
  int64_t nx;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    int64_t const shot_index = tide_nd_index(i, dy, dx, nx);
    int64_t const grid_index = tide_nd_index_j(j, dy, dx, nx);
    return ldg_coeff(cq, cq_batched, shot_index, grid_index) *
           lambda_hz[shot_index];
  }
};

template <typename T> struct CqLambdaHxAccessor2D {
  T const *cq;
  T const *lambda_hx;
  bool cq_batched;
  int64_t i;
  int64_t j;
  int64_t nx;

  TIDE_HOST_DEVICE T operator()(int64_t dy, int64_t dx) const {
    int64_t const shot_index = tide_nd_index(i, dy, dx, nx);
    int64_t const grid_index = tide_nd_index_j(j, dy, dx, nx);
    return -ldg_coeff(cq, cq_batched, shot_index, grid_index) *
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

__global__ void backward_kernel_lambda_h_update_m_exact(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const m_lambda_ey_x,
    TIDE_DTYPE *__restrict const m_lambda_ey_z,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const bx,
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
  TIDE_DTYPE const cb_val = ldg_coeff(cb, cb_batched, i, j);
  TIDE_DTYPE const g = cb_val * lambda_ey[i];
  if (pml_x) {
    TIDE_DTYPE const tmp_x = m_lambda_ey_x[i] + g;
    m_lambda_ey_x[i] = __ldg(&bx[x]) * tmp_x;
  }
  if (pml_y) {
    TIDE_DTYPE const tmp_z = m_lambda_ey_z[i] + g;
    m_lambda_ey_z[i] = __ldg(&by[y]) * tmp_z;
  }
}

__global__ void backward_kernel_lambda_h_apply_exact_interior(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE *__restrict const lambda_hx, TIDE_DTYPE *__restrict const lambda_hz,
    int64_t const y_begin, int64_t const y_end, int64_t const x_begin,
    int64_t const x_end) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + x_begin;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + y_begin;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y >= y_end || x >= x_end)
    return;
  if (y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1)
    return;

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  CbLambdaEyAccessor2D<TIDE_DTYPE> g_cb_l{cb, lambda_ey, cb_batched, i, j, nx};
  if (y < ny - kFdPad) {
    lambda_hx[i] -= DIFFY1_ADJ(constant_one, g_cb_l);
  }
  if (x < nx - kFdPad) {
    lambda_hz[i] += DIFFX1_ADJ(constant_one, g_cb_l);
  }
}

__global__ void backward_kernel_lambda_h_apply_exact_boundary(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const m_lambda_ey_x,
    TIDE_DTYPE const *__restrict const m_lambda_ey_z,
    TIDE_DTYPE *__restrict const lambda_hx, TIDE_DTYPE *__restrict const lambda_hz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx,
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
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  LambdaHWorkXExactAccessor2D<TIDE_DTYPE> work_x_l{
      cb, lambda_ey, m_lambda_ey_x, ax, bx, kx, shot_offset,
      y,  x,         pml_x0h,       pml_x1h};
  LambdaHWorkZExactAccessor2D<TIDE_DTYPE> work_z_l{
      cb, lambda_ey, m_lambda_ey_z, ay, by, ky, shot_offset,
      y,  x,         pml_y0h,       pml_y1h};
  if (y < ny - kFdPad) {
    lambda_hx[i] -= DIFFY1_ADJ(constant_one, work_z_l);
  }
  if (x < nx - kFdPad) {
    lambda_hz[i] += DIFFX1_ADJ(constant_one, work_x_l);
  }
}

__global__ void backward_kernel_lambda_e_update_m_exact(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const m_lambda_hx_z,
    TIDE_DTYPE *__restrict const m_lambda_hz_x,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bxh,
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
  TIDE_DTYPE const cq_val = ldg_coeff(cq, cq_batched, i, j);
  if (pml_x) {
    TIDE_DTYPE const g_x = cq_val * lambda_hz[i];
    TIDE_DTYPE const tmp_x = m_lambda_hz_x[i] + g_x;
    m_lambda_hz_x[i] = __ldg(&bxh[x]) * tmp_x;
  }
  if (pml_y) {
    TIDE_DTYPE const g_z = -cq_val * lambda_hx[i];
    TIDE_DTYPE const tmp_z = m_lambda_hx_z[i] + g_z;
    m_lambda_hx_z[i] = __ldg(&byh[y]) * tmp_z;
  }
}

__global__ void backward_kernel_lambda_e_apply_exact_interior(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val,
    int64_t const y_begin, int64_t const y_end, int64_t const x_begin,
    int64_t const x_end) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + x_begin;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + y_begin;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y >= y_end || x >= x_end)
    return;
  if (y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1)
    return;

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  TIDE_DTYPE const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  CqLambdaHzAccessor2D<TIDE_DTYPE> g_x_l{cq, lambda_hz, cq_batched, i, j, nx};
  CqLambdaHxAccessor2D<TIDE_DTYPE> g_z_l{cq, lambda_hx, cq_batched, i, j, nx};
  TIDE_DTYPE const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, g_x_l) + DIFFYH1_ADJ(constant_one, g_z_l);
  TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
  if (ca_requires_grad && ey_store != nullptr) {
    grad_ca_shot[i] +=
        lambda_ey_curr * ey_store[i] * step_ratio_to_field(step_ratio_val);
  }
  if (cb_requires_grad && curl_h_store != nullptr) {
    grad_cb_shot[i] +=
        lambda_ey_curr * curl_h_store[i] * step_ratio_to_field(step_ratio_val);
  }
}

__global__ void backward_kernel_lambda_e_apply_exact_interior_nograd(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey, int64_t const y_begin,
    int64_t const y_end, int64_t const x_begin, int64_t const x_end) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + x_begin;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + y_begin;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y >= y_end || x >= x_end)
    return;
  if (y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1)
    return;

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  TIDE_DTYPE const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  CqLambdaHzAccessor2D<TIDE_DTYPE> g_x_l{cq, lambda_hz, cq_batched, i, j, nx};
  CqLambdaHxAccessor2D<TIDE_DTYPE> g_z_l{cq, lambda_hx, cq_batched, i, j, nx};
  TIDE_DTYPE const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, g_x_l) + DIFFYH1_ADJ(constant_one, g_z_l);
  TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
}

__global__ void backward_kernel_lambda_e_apply_exact_boundary(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE const *__restrict const m_lambda_hx_z,
    TIDE_DTYPE const *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const ey_store,
    TIDE_DTYPE const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh,
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
  TIDE_DTYPE const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  LambdaEWorkXExactAccessor2D<TIDE_DTYPE> work_x_l{
      cq, lambda_hz, m_lambda_hz_x, axh, bxh, kxh, shot_offset,
      y,  x,         pml_x0,        pml_x1};
  LambdaEWorkZExactAccessor2D<TIDE_DTYPE> work_z_l{
      cq, lambda_hx, m_lambda_hx_z, ayh, byh, kyh, shot_offset,
      y,  x,         pml_y0,        pml_y1};
  TIDE_DTYPE const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, work_x_l) +
      DIFFYH1_ADJ(constant_one, work_z_l);
  TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
  if (!pml_y && !pml_x && ca_requires_grad && ey_store != nullptr) {
    grad_ca_shot[i] +=
        lambda_ey_curr * ey_store[i] * step_ratio_to_field(step_ratio_val);
  }
  if (!pml_y && !pml_x && cb_requires_grad && curl_h_store != nullptr) {
    grad_cb_shot[i] +=
        lambda_ey_curr * curl_h_store[i] * step_ratio_to_field(step_ratio_val);
  }
}

__global__ void backward_kernel_lambda_e_apply_exact_boundary_nograd(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE const *__restrict const m_lambda_hx_z,
    TIDE_DTYPE const *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh,
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

  int64_t const shot_offset = shot_idx * shot_numel;
  int64_t const j = y * nx + x;
  int64_t const i = shot_offset + j;
  TIDE_DTYPE const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  LambdaEWorkXExactAccessor2D<TIDE_DTYPE> work_x_l{
      cq, lambda_hz, m_lambda_hz_x, axh, bxh, kxh, shot_offset,
      y,  x,         pml_x0,        pml_x1};
  LambdaEWorkZExactAccessor2D<TIDE_DTYPE> work_z_l{
      cq, lambda_hx, m_lambda_hx_z, ayh, byh, kyh, shot_offset,
      y,  x,         pml_y0,        pml_y1};
  TIDE_DTYPE const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, work_x_l) +
      DIFFYH1_ADJ(constant_one, work_z_l);
  TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
}

__global__ void backward_kernel_lambda_e_apply_exact_bf16_interior(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE *__restrict const lambda_ey,
    __nv_bfloat16 const *__restrict const ey_store,
    __nv_bfloat16 const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val,
    int64_t const y_begin, int64_t const y_end, int64_t const x_begin,
    int64_t const x_end) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x +
              (int64_t)threadIdx.x + x_begin;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y +
              (int64_t)threadIdx.y + y_begin;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y >= y_end || x >= x_end)
    return;
  if (y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1)
    return;

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  TIDE_DTYPE const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  CqLambdaHzAccessor2D<TIDE_DTYPE> g_x_l{cq, lambda_hz, cq_batched, i, j, nx};
  CqLambdaHxAccessor2D<TIDE_DTYPE> g_z_l{cq, lambda_hx, cq_batched, i, j, nx};
  TIDE_DTYPE const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, g_x_l) + DIFFYH1_ADJ(constant_one, g_z_l);
  TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
  if (ca_requires_grad && ey_store != nullptr) {
    TIDE_DTYPE const ey_n = (TIDE_DTYPE)__bfloat162float(ey_store[i]);
    grad_ca_shot[i] +=
        lambda_ey_curr * ey_n * step_ratio_to_field(step_ratio_val);
  }
  if (cb_requires_grad && curl_h_store != nullptr) {
    TIDE_DTYPE const curl_h_n = (TIDE_DTYPE)__bfloat162float(curl_h_store[i]);
    grad_cb_shot[i] +=
        lambda_ey_curr * curl_h_n * step_ratio_to_field(step_ratio_val);
  }
}

__global__ void backward_kernel_lambda_e_apply_exact_bf16_boundary(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE const *__restrict const m_lambda_hx_z,
    TIDE_DTYPE const *__restrict const m_lambda_hz_x,
    TIDE_DTYPE *__restrict const lambda_ey,
    __nv_bfloat16 const *__restrict const ey_store,
    __nv_bfloat16 const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh,
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
  TIDE_DTYPE const ca_val = ldg_coeff(ca, ca_batched, i, j);
  ConstantOneAccessor2D<TIDE_DTYPE> constant_one{};
  LambdaEWorkXExactAccessor2D<TIDE_DTYPE> work_x_l{
      cq, lambda_hz, m_lambda_hz_x, axh, bxh, kxh, shot_offset,
      y,  x,         pml_x0,        pml_x1};
  LambdaEWorkZExactAccessor2D<TIDE_DTYPE> work_z_l{
      cq, lambda_hx, m_lambda_hx_z, ayh, byh, kyh, shot_offset,
      y,  x,         pml_y0,        pml_y1};
  TIDE_DTYPE const curl_lambda_h =
      DIFFXH1_ADJ(constant_one, work_x_l) +
      DIFFYH1_ADJ(constant_one, work_z_l);
  TIDE_DTYPE const lambda_ey_curr = lambda_ey[i];
  lambda_ey[i] = ca_val * lambda_ey_curr + curl_lambda_h;
  if (!pml_y && !pml_x && ca_requires_grad && ey_store != nullptr) {
    TIDE_DTYPE const ey_n = (TIDE_DTYPE)__bfloat162float(ey_store[i]);
    grad_ca_shot[i] +=
        lambda_ey_curr * ey_n * step_ratio_to_field(step_ratio_val);
  }
  if (!pml_y && !pml_x && cb_requires_grad && curl_h_store != nullptr) {
    TIDE_DTYPE const curl_h_n = (TIDE_DTYPE)__bfloat162float(curl_h_store[i]);
    grad_cb_shot[i] +=
        lambda_ey_curr * curl_h_n * step_ratio_to_field(step_ratio_val);
  }
}

// Combine per-shot gradients into final gradient (sum across shots)
__global__ void combine_grad(TIDE_DTYPE *__restrict const grad,
                             TIDE_DTYPE const *__restrict const grad_shot) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x + kFdPad;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y + kFdPad;
  if (y < ny - kFdPad && x < nx - kFdPad) {
    int64_t j = y * nx + x;
    int64_t const stride = shot_numel;
    TIDE_DTYPE sum = 0;
#pragma unroll 4
    for (int64_t shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
      sum += grad_shot[shot_idx * stride + j];
    }
    grad[j] += sum;
  }
}

} // namespace

// Forward propagation function
extern "C" void FUNC(forward)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f, TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx, TIDE_DTYPE *const hz, TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z, TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x, TIDE_DTYPE const *const debye_a,
    TIDE_DTYPE const *const debye_b, TIDE_DTYPE const *const debye_cp,
    TIDE_DTYPE *const polarization, TIDE_DTYPE *const ey_prev,
    TIDE_DTYPE *const r, int64_t const n_poles_h, TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by, TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh, TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh, TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh, TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, tide_scalar_t const rdy_h,
    tide_scalar_t const rdx_h, tide_scalar_t const dt_h, int64_t const nt,
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h, bool const has_dispersion, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);

  int64_t const shot_numel_h = ny_h * nx_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);

  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  bool const debug_path = read_env_flag("TIDE_TM_DEBUG_PATH");

  if (debug_path) {
    std::fprintf(stderr, "TIDE TM path: baseline\n");
  }

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);
    if (has_dispersion) {
      forward_kernel_e_debye<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                               stream_compute>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_prev, debye_cp, polarization,
          n_poles_h, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    } else {
      forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                         stream_compute>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources, 0,
                       stream_compute>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (has_dispersion) {
      update_polarization_debye<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                                  stream_compute>>>(
          ey_prev, ey, debye_a, debye_b, polarization, n_poles_h);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<launch_cfg.dimGridReceivers,
                            launch_cfg.dimBlockReceivers, 0,
                            stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(forward_with_storage)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f, TIDE_DTYPE *const ey,
    TIDE_DTYPE *const hx, TIDE_DTYPE *const hz, TIDE_DTYPE *const m_ey_x,
    TIDE_DTYPE *const m_ey_z, TIDE_DTYPE *const m_hx_z,
    TIDE_DTYPE *const m_hz_x, TIDE_DTYPE *const r, void *const ey_store_1,
    void *const ey_store_3, char const *const *const ey_filenames,
    void *const curl_store_1, void *const curl_store_3,
    char const *const *const curl_filenames, TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const by, TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const byh, TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bxh, TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh, TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, tide_scalar_t const rdy_h,
    tide_scalar_t const rdx_h, tide_scalar_t const dt_h, int64_t const nt,
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h, int64_t const storage_mode_h,
    int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp_h, bool const ca_requires_grad,
    bool const cb_requires_grad, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle, void *const storage_stream_handle) {

  cudaSetDevice(device);
  (void)n_threads;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (storage_format_h == STORAGE_FORMAT_BF16);
  bool const use_storage_pipeline =
      (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) &&
      (ca_requires_grad || cb_requires_grad) && stream_storage != nullptr &&
      stream_storage != stream_compute;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);

  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  void *async_disk_ey = nullptr;
  void *async_disk_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad)
      async_disk_ey =
          storage_async_disk_open(ey_filenames[0], true, NUM_BUFFERS);
    if (cb_requires_grad)
      async_disk_curl =
          storage_async_disk_open(curl_filenames[0], true, NUM_BUFFERS);
  }

  ScopedEventArray storage_done_events;
  ScopedEventArray compute_done_events;
  if (use_storage_pipeline) {
    for (int i = 0; i < NUM_BUFFERS; ++i) {
      tide::cuda_check_or_abort(cudaEventCreate(&storage_done_events.events[i]),
                                __FILE__, __LINE__);
      tide::cuda_check_or_abort(cudaEventCreate(&compute_done_events.events[i]),
                                __FILE__, __LINE__);
      tide::cuda_check_or_abort(
          cudaEventRecord(storage_done_events.events[i], stream_storage),
          __FILE__, __LINE__);
    }
  }

  auto run_step = [&](int64_t t) {
    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    bool const want_store = store_ey || store_curl;
    int slot = 0;
    cudaEvent_t slot_storage_done = nullptr;
    cudaEvent_t slot_compute_done = nullptr;
    if (want_store) {
      int64_t const step_idx = t / step_ratio_h;
      slot = (int)(step_idx % NUM_BUFFERS);
      if (use_storage_pipeline) {
        slot_storage_done = storage_done_events.events[slot];
        slot_compute_done = compute_done_events.events[slot];
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_compute, slot_storage_done, 0),
            __FILE__, __LINE__);
      }
    }

    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);
    if (want_store) {
      int64_t const step_idx = t / step_ratio_h;
      size_t const store1_offset = ring_storage_offset_bytes(
          step_idx, storage_mode_h, bytes_per_step_store);
      size_t const store3_offset = host_storage_offset_bytes(
          step_idx, storage_mode_h, bytes_per_step_store);

      void *__restrict const ey_store_1_t =
          (uint8_t *)ey_store_1 + store1_offset;
      void *__restrict const ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;

      void *__restrict const curl_store_1_t =
          (uint8_t *)curl_store_1 + store1_offset;
      void *__restrict const curl_store_3_t =
          (uint8_t *)curl_store_3 + store3_offset;

      if (storage_bf16_h) {
        forward_kernel_e_with_storage_bf16<<<launch_cfg.dimGrid,
                                             launch_cfg.dimBlock, 0,
                                             stream_compute>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
            store_curl ? (__nv_bfloat16 *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      } else {
        forward_kernel_e_with_storage<<<launch_cfg.dimGrid,
                                        launch_cfg.dimBlock, 0,
                                        stream_compute>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
            store_curl ? (TIDE_DTYPE *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      }

      if (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) {
        cudaStream_t save_stream = stream_compute;
        if (use_storage_pipeline) {
          tide::cuda_check_or_abort(
              cudaEventRecord(slot_compute_done, stream_compute), __FILE__,
              __LINE__);
          tide::cuda_check_or_abort(
              cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
              __FILE__, __LINE__);
          save_stream = stream_storage;
        }
        if (storage_mode_h == STORAGE_DISK) {
          int64_t const file_offset =
              step_idx * (int64_t)bytes_per_step_store;
          if (store_ey) {
            storage_async_disk_wait_slot(async_disk_ey, slot);
            tide::cuda_check_or_abort(
                cudaMemcpyAsync(ey_store_3_t, ey_store_1_t,
                                bytes_per_step_store, cudaMemcpyDeviceToHost,
                                save_stream),
                __FILE__, __LINE__);
            cudaEvent_t ready_event = nullptr;
            tide::cuda_check_or_abort(
                cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaEventRecord(ready_event, save_stream), __FILE__, __LINE__);
            storage_async_disk_enqueue_write(
                async_disk_ey, slot, ey_store_3_t, bytes_per_step_store,
                file_offset, ready_event);
          }
          if (store_curl) {
            storage_async_disk_wait_slot(async_disk_curl, slot);
            tide::cuda_check_or_abort(
                cudaMemcpyAsync(curl_store_3_t, curl_store_1_t,
                                bytes_per_step_store, cudaMemcpyDeviceToHost,
                                save_stream),
                __FILE__, __LINE__);
            cudaEvent_t ready_event = nullptr;
            tide::cuda_check_or_abort(
                cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaEventRecord(ready_event, save_stream), __FILE__, __LINE__);
            storage_async_disk_enqueue_write(
                async_disk_curl, slot, curl_store_3_t, bytes_per_step_store,
                file_offset, ready_event);
          }
        } else {
          if (store_ey) {
            storage_copy_snapshot_d2h(ey_store_1_t, ey_store_3_t,
                                      (size_t)shot_bytes_uncomp_h,
                                      (size_t)n_shots_h, save_stream);
          }
          if (store_curl) {
            storage_copy_snapshot_d2h(curl_store_1_t, curl_store_3_t,
                                      (size_t)shot_bytes_uncomp_h,
                                      (size_t)n_shots_h, save_stream);
          }
        }
        if (use_storage_pipeline) {
          tide::cuda_check_or_abort(
              cudaEventRecord(slot_storage_done, save_stream), __FILE__,
              __LINE__);
        }
      } else {
        (void)slot_storage_done;
        (void)slot_compute_done;
      }
    } else {
      forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                         stream_compute>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources, 0,
                       stream_compute>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<launch_cfg.dimGridReceivers,
                            launch_cfg.dimBlockReceivers, 0,
                            stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  if (use_storage_pipeline) {
    tide::cuda_check_or_abort(cudaStreamSynchronize(stream_storage), __FILE__,
                              __LINE__);
  }
  storage_async_disk_close(async_disk_ey);
  storage_async_disk_close(async_disk_curl);

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(backward)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const lambda_ey, TIDE_DTYPE *const lambda_hx,
    TIDE_DTYPE *const lambda_hz, TIDE_DTYPE *const m_lambda_ey_x,
    TIDE_DTYPE *const m_lambda_ey_z, TIDE_DTYPE *const m_lambda_hx_z,
    TIDE_DTYPE *const m_lambda_hz_x, void *const ey_store_1,
    void *const ey_store_3, char const *const *const ey_filenames,
    void *const curl_store_1, void *const curl_store_3,
    char const *const *const curl_filenames, TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca, TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE
        *const grad_ca_shot, // [n_shots, ny, nx] - per-shot gradient workspace
    TIDE_DTYPE
        *const grad_cb_shot, // [n_shots, ny, nx] - per-shot gradient workspace
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    tide_scalar_t const rdy_h, tide_scalar_t const rdx_h,
    tide_scalar_t const dt_h,
    int64_t const nt, int64_t const n_shots_h, int64_t const ny_h,
    int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    int64_t const storage_mode_h, int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp_h,
    bool const ca_requires_grad, bool const cb_requires_grad,
    bool const ca_batched_h, bool const cb_batched_h, bool const cq_batched_h,
    int64_t const start_t, int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h, int64_t const n_threads,
    int64_t const device, void *const compute_stream_handle,
    void *const storage_stream_handle) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)n_threads;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (storage_format_h == STORAGE_FORMAT_BF16);
  bool const use_storage_pipeline =
      (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) &&
      (ca_requires_grad || cb_requires_grad) && stream_storage != nullptr &&
      stream_storage != stream_compute;
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
    dimGridInterior =
        dim3(to_dim_u32(interior_gridx), to_dim_u32(interior_gridy),
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

  void *async_disk_ey = nullptr;
  void *async_disk_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad)
      async_disk_ey =
          storage_async_disk_open(ey_filenames[0], false, NUM_BUFFERS);
    if (cb_requires_grad)
      async_disk_curl =
          storage_async_disk_open(curl_filenames[0], false, NUM_BUFFERS);
  }

  ScopedEventArray storage_done_events;
  ScopedEventArray compute_done_events;
  if (use_storage_pipeline) {
    for (int i = 0; i < NUM_BUFFERS; ++i) {
      tide::cuda_check_or_abort(cudaEventCreate(&storage_done_events.events[i]),
                                __FILE__, __LINE__);
      tide::cuda_check_or_abort(cudaEventCreate(&compute_done_events.events[i]),
                                __FILE__, __LINE__);
      tide::cuda_check_or_abort(
          cudaEventRecord(compute_done_events.events[i], stream_compute),
          __FILE__, __LINE__);
    }
  }

  int64_t const first_store_idx = start_t / step_ratio_h - 1;
  int64_t const last_store_idx = (start_t - nt) / step_ratio_h;
  if (storage_mode_h == STORAGE_DISK) {
    int64_t const prefetch_count = tide_min<int64_t>(NUM_BUFFERS, nt / step_ratio_h);
    for (int64_t i = 0; i < prefetch_count; ++i) {
      int64_t const store_idx = first_store_idx - i;
      int const slot = (int)(store_idx % NUM_BUFFERS);
      size_t const store3_offset = host_storage_offset_bytes(
          store_idx, storage_mode_h, bytes_per_step_store);
      if (ca_requires_grad) {
        storage_async_disk_enqueue_read(
            async_disk_ey, slot, (uint8_t *)ey_store_3 + store3_offset,
            bytes_per_step_store,
            store_idx * (int64_t)bytes_per_step_store, nullptr);
      }
      if (cb_requires_grad) {
        storage_async_disk_enqueue_read(
            async_disk_curl, slot, (uint8_t *)curl_store_3 + store3_offset,
            bytes_per_step_store,
            store_idx * (int64_t)bytes_per_step_store, nullptr);
      }
    }
  }

  // Time reversed loop
  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    int slot = 0;
    cudaEvent_t slot_storage_done = nullptr;
    cudaEvent_t slot_compute_done = nullptr;
    cudaStream_t load_stream = stream_compute;

    int64_t const store_idx = t / step_ratio_h;
    bool const do_grad = (t % step_ratio_h) == 0;
    bool const grad_ey = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;
    bool const want_load = grad_ey || grad_curl;
    if (want_load) {
      slot = (int)(store_idx % NUM_BUFFERS);
      if (use_storage_pipeline) {
        slot_storage_done = storage_done_events.events[slot];
        slot_compute_done = compute_done_events.events[slot];
      }
    }

    size_t const store1_offset = ring_storage_offset_bytes(
        store_idx, storage_mode_h, bytes_per_step_store);
    size_t const store3_offset = host_storage_offset_bytes(
        store_idx, storage_mode_h, bytes_per_step_store);

    void *__restrict const ey_store_1_t = (uint8_t *)ey_store_1 + store1_offset;
    void *__restrict const ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;

    void *__restrict const curl_store_1_t =
        (uint8_t *)curl_store_1 + store1_offset;
    void *__restrict const curl_store_3_t =
        (uint8_t *)curl_store_3 + store3_offset;

    if (storage_mode_h == STORAGE_CPU && want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      if (grad_ey) {
        storage_copy_snapshot_h2d((void *)ey_store_1_t, (void *)ey_store_3_t,
                                  (size_t)shot_bytes_uncomp_h,
                                  (size_t)n_shots_h, load_stream);
      }
      if (grad_curl) {
        storage_copy_snapshot_h2d((void *)curl_store_1_t,
                                  (void *)curl_store_3_t,
                                  (size_t)shot_bytes_uncomp_h,
                                  (size_t)n_shots_h, load_stream);
      }
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, stream_storage), __FILE__,
            __LINE__);
      }
    } else if (storage_mode_h == STORAGE_DISK) {
      if (want_load && use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      if (grad_ey) {
        storage_async_disk_wait_slot(async_disk_ey, slot);
        tide::cuda_check_or_abort(
            cudaMemcpyAsync((void *)ey_store_1_t, (void *)ey_store_3_t,
                            bytes_per_step_store, cudaMemcpyHostToDevice,
                            load_stream),
            __FILE__, __LINE__);
      }
      if (grad_curl) {
        storage_async_disk_wait_slot(async_disk_curl, slot);
        tide::cuda_check_or_abort(
            cudaMemcpyAsync((void *)curl_store_1_t, (void *)curl_store_3_t,
                            bytes_per_step_store, cudaMemcpyHostToDevice,
                            load_stream),
            __FILE__, __LINE__);
      }
      if (want_load && use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, load_stream), __FILE__,
            __LINE__);
      }
    }

    // Inject adjoint source (receiver residual) at receiver locations
    // Use add_adjoint_sources_ey which checks n_receivers_per_shot
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<launch_cfg.dimGridReceivers,
                               launch_cfg.dimBlockReceivers, 0,
                               stream_compute>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

    // Record adjoint field at source locations for source gradient
    // Use record_adjoint_at_sources which checks n_sources_per_shot
    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources<<<launch_cfg.dimGridSources,
                                  launch_cfg.dimBlockSources, 0,
                                  stream_compute>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, lambda_ey, sources_i);
    }
    if (want_load && use_storage_pipeline) {
      tide::cuda_check_or_abort(
          cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
          __LINE__);
    }

    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_h_update_m_exact<<<dimGridBoundary,
                                                dimBlockBoundary, 0,
                                                stream_compute>>>(
          cb, lambda_ey, m_lambda_ey_x, m_lambda_ey_z, by, bx,
          boundary_layout);
    }
    if (has_interior) {
      backward_kernel_lambda_h_apply_exact_interior<<<dimGridInterior,
                                                      dimBlock, 0,
                                                      stream_compute>>>(
          cb, lambda_ey, lambda_hx, lambda_hz, interior_y_begin, interior_y_end,
          interior_x_begin, interior_x_end);
    }
    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_h_apply_exact_boundary<<<dimGridBoundary,
                                                      dimBlockBoundary, 0,
                                                      stream_compute>>>(
          cb, lambda_ey, m_lambda_ey_x, m_lambda_ey_z, lambda_hx, lambda_hz,
          ay, ax, by, bx, ky, kx, boundary_layout);
    }

    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_e_update_m_exact<<<dimGridBoundary,
                                                dimBlockBoundary, 0,
                                                stream_compute>>>(
          cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x, byh, bxh,
          boundary_layout);
    }

    if (grad_ey || grad_curl) {
      if (storage_bf16_h) {
        if (has_interior) {
          backward_kernel_lambda_e_apply_exact_bf16_interior<<<dimGridInterior,
                                                                dimBlock, 0,
                                                                stream_compute>>>(
              ca, cq, lambda_hx, lambda_hz, lambda_ey,
              grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
              grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h,
              interior_y_begin, interior_y_end, interior_x_begin,
              interior_x_end);
        }
        if (boundary_layout.total_count > 0) {
          backward_kernel_lambda_e_apply_exact_bf16_boundary
              <<<dimGridBoundary, dimBlockBoundary, 0, stream_compute>>>(
                  ca, cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x,
                  lambda_ey,
                  grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
                  grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
                  grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h,
                  ayh, axh, byh, bxh, kyh, kxh, boundary_layout);
        }
      } else {
        if (has_interior) {
          backward_kernel_lambda_e_apply_exact_interior<<<dimGridInterior,
                                                           dimBlock, 0,
                                                           stream_compute>>>(
              ca, cq, lambda_hx, lambda_hz, lambda_ey,
              grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
              grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h,
              interior_y_begin, interior_y_end, interior_x_begin,
              interior_x_end);
        }
        if (boundary_layout.total_count > 0) {
          backward_kernel_lambda_e_apply_exact_boundary<<<dimGridBoundary,
                                                          dimBlockBoundary, 0,
                                                          stream_compute>>>(
              ca, cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x,
              lambda_ey,
              grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
              grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h, ayh,
              axh, byh, bxh, kyh, kxh, boundary_layout);
        }
      }
    } else {
      if (has_interior) {
        backward_kernel_lambda_e_apply_exact_interior_nograd<<<dimGridInterior,
                                                                dimBlock, 0,
                                                                stream_compute>>>(
            ca, cq, lambda_hx, lambda_hz, lambda_ey, interior_y_begin,
            interior_y_end, interior_x_begin, interior_x_end);
      }
      if (boundary_layout.total_count > 0) {
        backward_kernel_lambda_e_apply_exact_boundary_nograd<<<dimGridBoundary,
                                                                dimBlockBoundary,
                                                                0,
                                                                stream_compute>>>(
            ca, cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x,
            lambda_ey, ayh, axh, byh, bxh, kyh, kxh, boundary_layout);
      }
    }

    if (want_load && use_storage_pipeline) {
      tide::cuda_check_or_abort(
          cudaEventRecord(slot_compute_done, stream_compute), __FILE__,
          __LINE__);
    }

    if (want_load && storage_mode_h == STORAGE_DISK) {
      int64_t const future_store_idx = store_idx - NUM_BUFFERS;
      if (future_store_idx >= last_store_idx) {
        size_t const future_store3_offset = host_storage_offset_bytes(
            future_store_idx, storage_mode_h, bytes_per_step_store);
        int64_t const future_file_offset =
            future_store_idx * (int64_t)bytes_per_step_store;
        if (grad_ey) {
          cudaEvent_t ready_event = nullptr;
          tide::cuda_check_or_abort(
              cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
              __FILE__, __LINE__);
          tide::cuda_check_or_abort(
              cudaEventRecord(ready_event, load_stream), __FILE__, __LINE__);
          storage_async_disk_enqueue_read(
              async_disk_ey, slot, (uint8_t *)ey_store_3 + future_store3_offset,
              bytes_per_step_store, future_file_offset, ready_event);
        }
        if (grad_curl) {
          cudaEvent_t ready_event = nullptr;
          tide::cuda_check_or_abort(
              cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
              __FILE__, __LINE__);
          tide::cuda_check_or_abort(
              cudaEventRecord(ready_event, load_stream), __FILE__, __LINE__);
          storage_async_disk_enqueue_read(
              async_disk_curl, slot,
              (uint8_t *)curl_store_3 + future_store3_offset,
              bytes_per_step_store, future_file_offset, ready_event);
        }
      }
    }

  }
  storage_async_disk_close(async_disk_ey);
  storage_async_disk_close(async_disk_curl);

  // Combine per-shot gradients (only if not batched - batched case keeps
  // per-shot grads)
  dim3 dimBlock_combine(32, 32, 1);
  dim3 dimGrid_combine(
      (nx_h - 2 * kFdPad + dimBlock_combine.x - 1) / dimBlock_combine.x,
      (ny_h - 2 * kFdPad + dimBlock_combine.y - 1) / dimBlock_combine.y, 1);

  if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_ca, grad_ca_shot);
  }
  if (cb_requires_grad && !cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_cb, grad_cb_shot);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

} // namespace FUNC(Inst)
