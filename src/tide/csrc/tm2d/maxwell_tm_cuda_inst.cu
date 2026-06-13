#undef DIFFY1
#undef DIFFX1
#undef DIFFYH1
#undef DIFFXH1

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

constexpr TIDE_DTYPE kEp0 = (TIDE_DTYPE)8.8541878128e-12;

__device__ __forceinline__ TIDE_DTYPE ldg_coeff(
    TIDE_DTYPE const *__restrict const coeff, bool const coeff_batched,
    int64_t const shot_index, int64_t const grid_index) {
  return __ldg(coeff + (coeff_batched ? shot_index : grid_index));
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

static inline dim3 make_tm_cell_grid(
    dim3 const block, int64_t const n_shots_h, int64_t const ny_h,
    int64_t const nx_h) {
  int64_t const gridx = (nx_h + block.x - 1) / block.x;
  int64_t const gridy = (ny_h + block.y - 1) / block.y;
  return dim3(to_dim_u32(gridx), to_dim_u32(gridy), to_dim_u32(n_shots_h));
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

__global__ __launch_bounds__(256) void forward_kernel_e_with_physical_storage(
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
  ::tide::forward_kernel_e_with_storage_core<TIDE_DTYPE, TIDE_DTYPE,
                                             TIDE_STENCIL, true>(
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

__global__ __launch_bounds__(256) void forward_kernel_e_with_physical_storage_bf16(
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
                                             TIDE_STENCIL, true>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_store, curl_h_store,
      ca_requires_grad, cb_requires_grad, y, x, shot_idx);
}

template <typename StoreT>
__global__ __launch_bounds__(256) void forward_kernel_e_with_delta_storage_t(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    StoreT *__restrict const delta_ey_store,
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
  ::tide::forward_kernel_e_with_delta_storage_core<TIDE_DTYPE, StoreT,
                                                   TIDE_STENCIL>(
      params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, delta_ey_store, y, x,
      shot_idx);
}

__global__ __launch_bounds__(256) void born_forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE const *__restrict const dhx,
    TIDE_DTYPE const *__restrict const dhz, TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const dm_hx_z, TIDE_DTYPE *__restrict const dm_hz_x,
    TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const dey_store,
    TIDE_DTYPE *__restrict const dcurl_h_store,
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
  ::tide::forward_kernel_e_born_with_storage_core<TIDE_DTYPE, TIDE_DTYPE,
                                                  TIDE_STENCIL>(
      params, ca, cb, dca, dcb, hx, hz, ey, m_hx_z, m_hz_x, dhx, dhz, dey,
      dm_hx_z, dm_hz_x, ey_store, curl_h_store, dey_store, dcurl_h_store,
      ca_requires_grad,
      cb_requires_grad, y, x, shot_idx);
}

__global__ __launch_bounds__(256) void born_forward_kernel_e_with_storage_bf16(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const m_hx_z, TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE const *__restrict const dhx,
    TIDE_DTYPE const *__restrict const dhz, TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const dm_hx_z, TIDE_DTYPE *__restrict const dm_hz_x,
    __nv_bfloat16 *__restrict const ey_store,
    __nv_bfloat16 *__restrict const curl_h_store,
    __nv_bfloat16 *__restrict const dey_store,
    __nv_bfloat16 *__restrict const dcurl_h_store,
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
  ::tide::forward_kernel_e_born_with_storage_core<TIDE_DTYPE, __nv_bfloat16,
                                                  TIDE_STENCIL>(
      params, ca, cb, dca, dcb, hx, hz, ey, m_hx_z, m_hz_x, dhx, dhz, dey,
      dm_hx_z, dm_hz_x, ey_store, curl_h_store, dey_store, dcurl_h_store,
      ca_requires_grad,
      cb_requires_grad, y, x, shot_idx);
}

template <typename StoreT>
__global__ __launch_bounds__(256) void born_background_prepare_direct_kernel(
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const lambda_sc_ey,
    StoreT const *__restrict const dey_store,
    StoreT const *__restrict const dcurl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    TIDE_DTYPE *__restrict const eta_source_old,
    TIDE_DTYPE *__restrict const work_x, TIDE_DTYPE *__restrict const work_z,
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
    TIDE_DTYPE const *__restrict const kxh, int64_t const step_ratio_val) {
  int64_t x = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y = (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  ::tide::GridParams<TIDE_DTYPE> params = {
      ay,      ayh,         ax,         axh,        by,      byh,
      bx,      bxh,         ky,         kyh,        kx,      kxh,
      static_cast<TIDE_DTYPE>(rdy), static_cast<TIDE_DTYPE>(rdx),
      n_shots, ny,          nx,         shot_numel, pml_y0,  pml_y1,
      pml_x0,  pml_x1,      ca_batched, cb_batched, false};
  ::tide::born_background_prepare_direct_core<TIDE_DTYPE, StoreT, TIDE_STENCIL>(
      params, dca, dcb, lambda_sc_ey, dey_store, dcurl_h_store, grad_ca_shot,
      grad_cb_shot, eta_source_old, work_x, work_z, step_ratio_val, y, x,
      shot_idx);
}

__global__ __launch_bounds__(256) void born_backward_apply_e_to_h_kernel(
    TIDE_DTYPE const *__restrict const work_x,
    TIDE_DTYPE const *__restrict const work_z,
    TIDE_DTYPE *__restrict const lambda_hx,
    TIDE_DTYPE *__restrict const lambda_hz,
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
      pml_x1,  false,      false,      false};
  ::tide::born_backward_apply_e_to_h_core<TIDE_DTYPE, TIDE_STENCIL>(
      params, work_x, work_z, lambda_hx, lambda_hz, y, x, shot_idx);
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

__global__ void add_inplace(TIDE_DTYPE *__restrict const dest,
                            TIDE_DTYPE const *__restrict const src) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx < n_shots && y < ny && x < nx) {
    int64_t const i = shot_idx * shot_numel + y * nx + x;
    dest[i] += src[i];
  }
}

template <typename StoreT, bool GradCa, bool GradCb,
          bool PHYSICAL_STORAGE = false>
__global__ void coeff_grad_kernel(
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, int64_t const step_ratio_val) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  int64_t store_i = 0;
  if constexpr (PHYSICAL_STORAGE) {
    int64_t const phys_y = pml_y1 - pml_y0;
    int64_t const phys_x = pml_x1 - pml_x0;
    if (shot_idx >= n_shots || phys_y <= 0 || phys_x <= 0 || y >= phys_y ||
        x >= phys_x) {
      return;
    }
    store_i = (shot_idx * phys_y + y) * phys_x + x;
    y += pml_y0;
    x += pml_x0;
  }
  if (shot_idx >= n_shots || y < kFdPad || x < kFdPad ||
      y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  if constexpr (!PHYSICAL_STORAGE) {
    store_i = i;
  }
  TIDE_DTYPE const lambda_val = lambda_ey[i];
  TIDE_DTYPE const step_scale = step_ratio_to_field(step_ratio_val);
  if constexpr (GradCa) {
    TIDE_DTYPE const ey_n =
        tide::decode_snapshot<StoreT, TIDE_DTYPE>(ey_store[store_i]);
    grad_ca_shot[i] += lambda_val * ey_n * step_scale;
  }
  if constexpr (GradCb) {
    TIDE_DTYPE const curl_h_n =
        tide::decode_snapshot<StoreT, TIDE_DTYPE>(curl_h_store[store_i]);
    grad_cb_shot[i] += lambda_val * curl_h_n * step_scale;
  }
}

template <typename StoreT>
__global__ void direct_epsilon_grad_delta_kernel(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const delta_ey_store,
    TIDE_DTYPE *__restrict const grad_eps_shot,
    int64_t const step_ratio_val, tide_scalar_t const dt_val) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y < kFdPad || x < kFdPad ||
      y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  TIDE_DTYPE const cb_val = ldg_coeff(cb, cb_batched, shot_idx, j);
  TIDE_DTYPE const delta_ey =
      tide::decode_snapshot<StoreT, TIDE_DTYPE>(delta_ey_store[i]);
  TIDE_DTYPE const scale =
      -kEp0 * cb_val *
      (static_cast<TIDE_DTYPE>(1) / static_cast<TIDE_DTYPE>(dt_val)) *
      step_ratio_to_field(step_ratio_val);
  grad_eps_shot[i] += lambda_ey[i] * delta_ey * scale;
}

template <typename StoreT>
__global__ void store_ey_snapshot_kernel(
    TIDE_DTYPE const *__restrict const ey,
    StoreT *__restrict const ey_store) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y >= ny || x >= nx) {
    return;
  }

  int64_t const i = shot_idx * shot_numel + y * nx + x;
  ey_store[i] = tide::encode_snapshot<StoreT, TIDE_DTYPE>(ey[i]);
}

template <typename StoreT>
__global__ void material_grad_direct_eonly_kernel(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const ey_next_store,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE *__restrict const grad_eps_shot,
    TIDE_DTYPE *__restrict const grad_sigma_shot,
    bool const grad_eps_step, bool const grad_sigma_step,
    bool const endpoint_interval,
    int64_t const step_ratio_val, tide_scalar_t const dt_val) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y < kFdPad || x < kFdPad ||
      y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  TIDE_DTYPE const cb_val = ldg_coeff(cb, cb_batched, shot_idx, j);
  TIDE_DTYPE const ey_curr =
      tide::decode_snapshot<StoreT, TIDE_DTYPE>(ey_store[i]);
  TIDE_DTYPE const ey_next =
      tide::decode_snapshot<StoreT, TIDE_DTYPE>(ey_next_store[i]);
  TIDE_DTYPE const lambda_val = lambda_ey[i];
  TIDE_DTYPE const step_scale = step_ratio_to_field(step_ratio_val);

  if (grad_eps_step && grad_eps != nullptr) {
    TIDE_DTYPE const eps_step_scale =
        endpoint_interval ? static_cast<TIDE_DTYPE>(1) : step_scale;
    TIDE_DTYPE const scaled =
        lambda_val * (ey_curr - ey_next) * cb_val *
        (static_cast<TIDE_DTYPE>(1) / static_cast<TIDE_DTYPE>(dt_val)) * kEp0 *
        eps_step_scale;
    if (ca_batched) {
      grad_eps[i] += scaled;
    } else if (grad_eps_shot != nullptr) {
      grad_eps_shot[i] += scaled;
    } else {
      atomicAdd(&grad_eps[j], scaled);
    }
  }

  if (grad_sigma_step && grad_sigma != nullptr) {
    TIDE_DTYPE const scaled = -static_cast<TIDE_DTYPE>(0.5) * lambda_val *
                              (ey_curr + ey_next) * cb_val * step_scale;
    if (ca_batched) {
      grad_sigma[i] += scaled;
    } else if (grad_sigma_shot != nullptr) {
      grad_sigma_shot[i] += scaled;
    } else {
      atomicAdd(&grad_sigma[j], scaled);
    }
  }
}

template <typename StoreT>
__global__ void material_grad_direct_ecurl_kernel(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    bool const grad_eps_step, bool const grad_sigma_step,
    int64_t const step_ratio_val, tide_scalar_t const dt_val) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx >= n_shots || y < kFdPad || x < kFdPad ||
      y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  int64_t const coeff_idx = ca_batched ? i : j;

  TIDE_DTYPE const ca_val = __ldg(ca + coeff_idx);
  TIDE_DTYPE const cb_val = __ldg(cb + coeff_idx);
  TIDE_DTYPE const dt = static_cast<TIDE_DTYPE>(dt_val);
  TIDE_DTYPE const half = static_cast<TIDE_DTYPE>(0.5);
  TIDE_DTYPE const ey_n =
      tide::decode_snapshot<StoreT, TIDE_DTYPE>(ey_store[i]);
  TIDE_DTYPE const curl_h_n =
      tide::decode_snapshot<StoreT, TIDE_DTYPE>(curl_h_store[i]);
  TIDE_DTYPE const lambda_val = lambda_ey[i];
  TIDE_DTYPE const step_scale = step_ratio_to_field(step_ratio_val);

  if (grad_eps_step && grad_eps != nullptr) {
    TIDE_DTYPE const dca_deps =
        kEp0 * (static_cast<TIDE_DTYPE>(1) - ca_val) * cb_val / dt;
    TIDE_DTYPE const dcb_deps = -kEp0 * cb_val * cb_val / dt;
    TIDE_DTYPE const scaled =
        lambda_val * (ey_n * dca_deps + curl_h_n * dcb_deps) * step_scale;
    if (ca_batched) {
      grad_eps[i] += scaled;
    } else if (n_shots == 1) {
      grad_eps[j] += scaled;
    } else {
      atomicAdd(&grad_eps[j], scaled);
    }
  }

  if (grad_sigma_step && grad_sigma != nullptr) {
    TIDE_DTYPE const dca_dsigma =
        -half * (static_cast<TIDE_DTYPE>(1) + ca_val) * cb_val;
    TIDE_DTYPE const dcb_dsigma = -half * cb_val * cb_val;
    TIDE_DTYPE const scaled =
        lambda_val * (ey_n * dca_dsigma + curl_h_n * dcb_dsigma) * step_scale;
    if (ca_batched) {
      grad_sigma[i] += scaled;
    } else if (n_shots == 1) {
      grad_sigma[j] += scaled;
    } else {
      atomicAdd(&grad_sigma[j], scaled);
    }
  }
}

template <typename StoreT>
static inline void launch_coeff_grad_kernel(
    TMForwardLaunchConfig const &launch_cfg, cudaStream_t const stream,
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val) {
  if (ca_requires_grad && cb_requires_grad) {
    coeff_grad_kernel<StoreT, true, true>
        <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, ey_store, curl_h_store, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  } else if (ca_requires_grad) {
    coeff_grad_kernel<StoreT, true, false>
        <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, ey_store, nullptr, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  } else if (cb_requires_grad) {
    coeff_grad_kernel<StoreT, false, true>
        <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, nullptr, curl_h_store, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  }
}

template <typename StoreT>
static inline void launch_coeff_grad_physical_kernel(
    TMForwardLaunchConfig const &launch_cfg, dim3 const physical_grid,
    cudaStream_t const stream, TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio_val) {
  if (ca_requires_grad && cb_requires_grad) {
    coeff_grad_kernel<StoreT, true, true, true>
        <<<physical_grid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, ey_store, curl_h_store, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  } else if (ca_requires_grad) {
    coeff_grad_kernel<StoreT, true, false, true>
        <<<physical_grid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, ey_store, nullptr, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  } else if (cb_requires_grad) {
    coeff_grad_kernel<StoreT, false, true, true>
        <<<physical_grid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, nullptr, curl_h_store, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  }
}

template <typename StoreT>
static inline void launch_direct_epsilon_grad_delta_kernel(
    TMForwardLaunchConfig const &launch_cfg, cudaStream_t const stream,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const delta_ey_store,
    TIDE_DTYPE *__restrict const grad_eps_shot, int64_t const step_ratio_val,
    tide_scalar_t const dt_val) {
  direct_epsilon_grad_delta_kernel<StoreT>
      <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
          cb, lambda_ey, delta_ey_store, grad_eps_shot, step_ratio_val, dt_val);
}

template <typename StoreT>
static inline void launch_store_ey_snapshot_kernel(
    TMForwardLaunchConfig const &launch_cfg, cudaStream_t const stream,
    TIDE_DTYPE const *__restrict const ey, StoreT *__restrict const ey_store) {
  store_ey_snapshot_kernel<StoreT>
      <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(ey, ey_store);
}

template <typename StoreT>
static inline void launch_material_grad_direct_eonly_kernel(
    TMForwardLaunchConfig const &launch_cfg, cudaStream_t const stream,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const ey_next_store,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE *__restrict const grad_eps_shot,
    TIDE_DTYPE *__restrict const grad_sigma_shot, bool const grad_eps_step,
    bool const grad_sigma_step, bool const endpoint_interval,
    int64_t const step_ratio_val, tide_scalar_t const dt_val) {
  material_grad_direct_eonly_kernel<StoreT>
      <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
          cb, lambda_ey, ey_store, ey_next_store, grad_eps, grad_sigma,
          grad_eps_shot, grad_sigma_shot, grad_eps_step, grad_sigma_step,
          endpoint_interval, step_ratio_val, dt_val);
}

template <typename StoreT>
static inline void launch_material_grad_direct_ecurl_kernel(
    TMForwardLaunchConfig const &launch_cfg, cudaStream_t const stream,
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma, bool const grad_eps_step,
    bool const grad_sigma_step, int64_t const step_ratio_val,
    tide_scalar_t const dt_val) {
  material_grad_direct_ecurl_kernel<StoreT>
      <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
          ca, cb, lambda_ey, ey_store, curl_h_store, grad_eps, grad_sigma,
          grad_eps_step, grad_sigma_step, step_ratio_val, dt_val);
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
    int64_t const execution_backend_h,
    void *const compute_stream_handle, void *const storage_stream_handle) {

  cudaSetDevice(device);
  (void)n_threads;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = ny_h * nx_h;
  int64_t const physical_ny_h = pml_y1_h - pml_y0_h;
  int64_t const physical_nx_h = pml_x1_h - pml_x0_h;
  int64_t const physical_numel_h =
      physical_ny_h > 0 && physical_nx_h > 0 ? physical_ny_h * physical_nx_h
                                              : 0;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (storage_format_h == STORAGE_FORMAT_BF16);
  bool const direct_epsilon_grad = execution_backend_h == 1;
  bool const direct_material_grad =
      execution_backend_h == 2 || execution_backend_h == 3;
  bool const direct_material_ecurl_grad = execution_backend_h == 4;
  bool const direct_material_any =
      direct_material_grad || direct_material_ecurl_grad;
  size_t const full_bytes_per_shot = (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  size_t const bf16_bytes_per_shot = (size_t)shot_numel_h * sizeof(__nv_bfloat16);
  size_t const physical_full_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(TIDE_DTYPE);
  size_t const physical_bf16_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(__nv_bfloat16);
  bool const storage_full_h = storage_format_h == STORAGE_FORMAT_FULL;
  bool const storage_full_domain =
      ((storage_full_h && shot_bytes_uncomp_h == (int64_t)full_bytes_per_shot) ||
       (storage_bf16_h && shot_bytes_uncomp_h == (int64_t)bf16_bytes_per_shot));
  bool const storage_physical =
      storage_mode_h == STORAGE_DEVICE && !direct_epsilon_grad &&
      !direct_material_any &&
      physical_numel_h > 0 && physical_numel_h != shot_numel_h &&
      ((storage_full_h &&
        shot_bytes_uncomp_h == (int64_t)physical_full_bytes_per_shot) ||
       (storage_bf16_h &&
        shot_bytes_uncomp_h == (int64_t)physical_bf16_bytes_per_shot));
  bool const use_storage_pipeline =
      (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) &&
      storage_full_domain &&
      (ca_requires_grad || cb_requires_grad || direct_material_any) &&
      stream_storage != nullptr &&
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
    if (ca_requires_grad || direct_material_any)
      async_disk_ey =
          storage_async_disk_open(ey_filenames[0], true, NUM_BUFFERS);
    if (cb_requires_grad || direct_epsilon_grad || direct_material_ecurl_grad)
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
    bool const store_ey =
        store_step &&
        ((ca_requires_grad && !direct_epsilon_grad) || direct_material_any);
    bool const store_curl =
        store_step &&
        (((cb_requires_grad || direct_epsilon_grad) && !direct_material_grad) ||
         direct_material_ecurl_grad);
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
        if (direct_epsilon_grad) {
          forward_kernel_e_with_delta_storage_t<__nv_bfloat16>
              <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
                  ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
                  (__nv_bfloat16 *)curl_store_1_t, ay, ayh, ax, axh, by, byh,
                  bx, bxh, ky, kyh, kx, kxh);
        } else if (storage_physical) {
          forward_kernel_e_with_physical_storage_bf16<<<launch_cfg.dimGrid,
                                                        launch_cfg.dimBlock, 0,
                                                        stream_compute>>>(
              ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
              store_ey ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curl_store_1_t : nullptr, ay, ayh,
              ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey,
              store_curl);
        } else {
          forward_kernel_e_with_storage_bf16<<<launch_cfg.dimGrid,
                                               launch_cfg.dimBlock, 0,
                                               stream_compute>>>(
              ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
              store_ey ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
              store_curl ? (__nv_bfloat16 *)curl_store_1_t : nullptr, ay, ayh,
              ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey,
              store_curl);
        }
      } else {
        if (direct_epsilon_grad) {
          forward_kernel_e_with_delta_storage_t<TIDE_DTYPE>
              <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
                  ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
                  (TIDE_DTYPE *)curl_store_1_t, ay, ayh, ax, axh, by, byh, bx,
                  bxh, ky, kyh, kx, kxh);
        } else if (storage_physical) {
          forward_kernel_e_with_physical_storage<<<launch_cfg.dimGrid,
                                                   launch_cfg.dimBlock, 0,
                                                   stream_compute>>>(
              ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
              store_ey ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
              store_curl ? (TIDE_DTYPE *)curl_store_1_t : nullptr, ay, ayh, ax,
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

  if (direct_material_grad && storage_mode_h == STORAGE_DEVICE &&
      ey_store_1 != nullptr) {
    int64_t const final_store_idx = (start_t + nt) / step_ratio_h;
    size_t const final_store_offset = ring_storage_offset_bytes(
        final_store_idx, storage_mode_h, bytes_per_step_store);
    void *__restrict const ey_store_final =
        (uint8_t *)ey_store_1 + final_store_offset;
    if (storage_bf16_h) {
      launch_store_ey_snapshot_kernel<__nv_bfloat16>(
          launch_cfg, stream_compute, ey, (__nv_bfloat16 *)ey_store_final);
    } else {
      launch_store_ey_snapshot_kernel<TIDE_DTYPE>(
          launch_cfg, stream_compute, ey, (TIDE_DTYPE *)ey_store_final);
    }
  }

  if (use_storage_pipeline) {
    tide::cuda_check_or_abort(cudaStreamSynchronize(stream_storage), __FILE__,
                              __LINE__);
  }
  storage_async_disk_close(async_disk_ey);
  storage_async_disk_close(async_disk_curl);

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(born_forward)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb, TIDE_DTYPE const *const f0,
    TIDE_DTYPE const *const df, TIDE_DTYPE *const ey, TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz, TIDE_DTYPE *const m_ey_x, TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z, TIDE_DTYPE *const m_hz_x, TIDE_DTYPE *const dey,
    TIDE_DTYPE *const dhx, TIDE_DTYPE *const dhz, TIDE_DTYPE *const dm_ey_x,
    TIDE_DTYPE *const dm_ey_z, TIDE_DTYPE *const dm_hx_z,
    TIDE_DTYPE *const dm_hz_x, TIDE_DTYPE *const r,
    TIDE_DTYPE *const background_r,
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    tide_scalar_t const rdy_h, tide_scalar_t const rdx_h,
    tide_scalar_t const dt_h, int64_t const nt, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    bool const ca_batched_h, bool const cb_batched_h, bool const cq_batched_h,
    int64_t const start_t, int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h, int64_t const n_threads,
    int64_t const device, void *const compute_stream_handle) {

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

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, dey, dhx, dhz, dm_ey_x, dm_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);
    born_forward_kernel_e_with_storage<<<launch_cfg.dimGrid, launch_cfg.dimBlock,
                                         0, stream_compute>>>(
        ca, cb, dca, dcb, hx, hz, ey, m_hx_z, m_hz_x, dhx, dhz, dey, dm_hx_z,
        dm_hz_x, nullptr, nullptr, nullptr, nullptr, ay, ayh, ax, axh, by, byh,
        bx, bxh, ky, kyh, kx, kxh, false, false);

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources, 0,
                       stream_compute>>>(
          ey, f0 + t * n_shots_h * n_sources_per_shot_h, sources_i);
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources, 0,
                       stream_compute>>>(
          dey, df + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<launch_cfg.dimGridReceivers,
                            launch_cfg.dimBlockReceivers, 0,
                            stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, dey, receivers_i);
      if (background_r != nullptr) {
        record_receivers_ey<<<launch_cfg.dimGridReceivers,
                              launch_cfg.dimBlockReceivers, 0,
                              stream_compute>>>(
            background_r + t * n_shots_h * n_receivers_per_shot_h, ey,
            receivers_i);
      }
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(born_forward_with_storage)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb, TIDE_DTYPE const *const f0,
    TIDE_DTYPE const *const df, TIDE_DTYPE *const ey, TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz, TIDE_DTYPE *const m_ey_x, TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z, TIDE_DTYPE *const m_hz_x, TIDE_DTYPE *const dey,
    TIDE_DTYPE *const dhx, TIDE_DTYPE *const dhz, TIDE_DTYPE *const dm_ey_x,
    TIDE_DTYPE *const dm_ey_z, TIDE_DTYPE *const dm_hx_z,
    TIDE_DTYPE *const dm_hz_x, TIDE_DTYPE *const r,
    TIDE_DTYPE *const background_r, void *const ey_store_1,
    void *const ey_store_3, char const *const *const ey_filenames,
    void *const curl_store_1, void *const curl_store_3,
    char const *const *const curl_filenames, void *const dey_store,
    void *const dcurl_store, TIDE_DTYPE const *const ay,
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
  (void)dt_h;
  (void)n_threads;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = ny_h * nx_h;
  int64_t const physical_ny_h = pml_y1_h - pml_y0_h;
  int64_t const physical_nx_h = pml_x1_h - pml_x0_h;
  int64_t const physical_numel_h =
      physical_ny_h > 0 && physical_nx_h > 0 ? physical_ny_h * physical_nx_h
                                              : 0;
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
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, dey, dhx, dhz, dm_ey_x, dm_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh);

    if (want_store) {
      int64_t const step_idx = t / step_ratio_h;
      size_t const store1_offset = ring_storage_offset_bytes(
          step_idx, storage_mode_h, bytes_per_step_store);
      size_t const store3_offset = host_storage_offset_bytes(
          step_idx, storage_mode_h, bytes_per_step_store);
      size_t const direct_store_offset =
          (size_t)step_idx * bytes_per_step_store;

      void *__restrict const ey_store_1_t =
          (uint8_t *)ey_store_1 + store1_offset;
      void *__restrict const ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;
      void *__restrict const curl_store_1_t =
          (uint8_t *)curl_store_1 + store1_offset;
      void *__restrict const curl_store_3_t =
          (uint8_t *)curl_store_3 + store3_offset;
      void *__restrict const dey_store_t =
          dey_store != nullptr
              ? (uint8_t *)dey_store + direct_store_offset
              : nullptr;
      void *__restrict const dcurl_store_t =
          dcurl_store != nullptr
              ? (uint8_t *)dcurl_store + direct_store_offset
              : nullptr;

      if (storage_bf16_h) {
        born_forward_kernel_e_with_storage_bf16<<<launch_cfg.dimGrid,
                                                  launch_cfg.dimBlock, 0,
                                                  stream_compute>>>(
            ca, cb, dca, dcb, hx, hz, ey, m_hx_z, m_hz_x, dhx, dhz, dey,
            dm_hx_z, dm_hz_x, store_ey ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
            store_curl ? (__nv_bfloat16 *)curl_store_1_t : nullptr,
            (__nv_bfloat16 *)dey_store_t, (__nv_bfloat16 *)dcurl_store_t, ay,
            ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey,
            store_curl);
      } else {
        born_forward_kernel_e_with_storage<<<launch_cfg.dimGrid,
                                             launch_cfg.dimBlock, 0,
                                             stream_compute>>>(
            ca, cb, dca, dcb, hx, hz, ey, m_hx_z, m_hz_x, dhx, dhz, dey,
            dm_hx_z, dm_hz_x, store_ey ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
            store_curl ? (TIDE_DTYPE *)curl_store_1_t : nullptr,
            (TIDE_DTYPE *)dey_store_t, (TIDE_DTYPE *)dcurl_store_t, ay, ayh, ax,
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
      }
    } else {
      born_forward_kernel_e_with_storage<<<launch_cfg.dimGrid,
                                           launch_cfg.dimBlock, 0,
                                           stream_compute>>>(
          ca, cb, dca, dcb, hx, hz, ey, m_hx_z, m_hz_x, dhx, dhz, dey,
          dm_hx_z, dm_hz_x, nullptr, nullptr, nullptr, nullptr, ay, ayh, ax,
          axh, by, byh, bx, bxh, ky, kyh, kx, kxh, false, false);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources, 0,
                       stream_compute>>>(
          ey, f0 + t * n_shots_h * n_sources_per_shot_h, sources_i);
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources, 0,
                       stream_compute>>>(
          dey, df + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<launch_cfg.dimGridReceivers,
                            launch_cfg.dimBlockReceivers, 0,
                            stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, dey, receivers_i);
      if (background_r != nullptr) {
        record_receivers_ey<<<launch_cfg.dimGridReceivers,
                              launch_cfg.dimBlockReceivers, 0,
                              stream_compute>>>(
            background_r + t * n_shots_h * n_receivers_per_shot_h, ey,
            receivers_i);
      }
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

extern "C" void FUNC(born_backward)(
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
    TIDE_DTYPE *const grad_ca_shot, TIDE_DTYPE *const grad_cb_shot,
    TIDE_DTYPE *const work_x, TIDE_DTYPE *const work_z,
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    tide_scalar_t const rdy_h, tide_scalar_t const rdx_h,
    tide_scalar_t const dt_h, int64_t const nt, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h, int64_t const n_sources_per_shot_h,
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
  (void)work_x;
  (void)work_z;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = ny_h * nx_h;
  int64_t const physical_ny_h = pml_y1_h - pml_y0_h;
  int64_t const physical_nx_h = pml_x1_h - pml_x0_h;
  int64_t const physical_numel_h =
      physical_ny_h > 0 && physical_nx_h > 0 ? physical_ny_h * physical_nx_h
                                              : 0;
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

  if (ca_requires_grad && !ca_batched_h) {
    tide::cuda_check_or_abort(
        cudaMemsetAsync(grad_ca_shot, 0,
                        (size_t)n_shots_h * (size_t)shot_numel_h *
                            sizeof(TIDE_DTYPE),
                        stream_compute),
        __FILE__, __LINE__);
  }
  if (cb_requires_grad && !cb_batched_h) {
    tide::cuda_check_or_abort(
        cudaMemsetAsync(grad_cb_shot, 0,
                        (size_t)n_shots_h * (size_t)shot_numel_h *
                            sizeof(TIDE_DTYPE),
                        stream_compute),
        __FILE__, __LINE__);
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

    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x, m_lambda_ey_z, ay,
        ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        ca, cb, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
        m_lambda_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);

    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<launch_cfg.dimGridReceivers,
                               launch_cfg.dimBlockReceivers, 0,
                               stream_compute>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }
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

    if (grad_ey || grad_curl) {
      if (storage_bf16_h) {
        launch_coeff_grad_kernel<__nv_bfloat16>(
            launch_cfg, stream_compute, lambda_ey,
            grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
            grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
            grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
      } else {
        launch_coeff_grad_kernel<TIDE_DTYPE>(
            launch_cfg, stream_compute, lambda_ey,
            grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
            grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
            grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
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
    TIDE_DTYPE *const grad_eps,
    TIDE_DTYPE *const grad_sigma,
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
    int64_t const device, int64_t const execution_backend_h,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {

  cudaSetDevice(device);
  (void)n_threads;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = ny_h * nx_h;
  int64_t const physical_ny_h = pml_y1_h - pml_y0_h;
  int64_t const physical_nx_h = pml_x1_h - pml_x0_h;
  int64_t const physical_numel_h =
      physical_ny_h > 0 && physical_nx_h > 0 ? physical_ny_h * physical_nx_h
                                              : 0;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (storage_format_h == STORAGE_FORMAT_BF16);
  bool const direct_epsilon_grad = execution_backend_h == 1;
  bool const direct_material_endpoint_grad = execution_backend_h == 3;
  bool const direct_material_grad =
      execution_backend_h == 2 || direct_material_endpoint_grad;
  bool const direct_material_ecurl_grad = execution_backend_h == 4;
  bool const direct_material_any =
      direct_material_grad || direct_material_ecurl_grad;
  size_t const full_bytes_per_shot = (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  size_t const bf16_bytes_per_shot = (size_t)shot_numel_h * sizeof(__nv_bfloat16);
  size_t const physical_full_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(TIDE_DTYPE);
  size_t const physical_bf16_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(__nv_bfloat16);
  bool const storage_full_h = storage_format_h == STORAGE_FORMAT_FULL;
  bool const storage_full_domain =
      ((storage_full_h && shot_bytes_uncomp_h == (int64_t)full_bytes_per_shot) ||
       (storage_bf16_h && shot_bytes_uncomp_h == (int64_t)bf16_bytes_per_shot));
  bool const storage_physical =
      storage_mode_h == STORAGE_DEVICE && !direct_epsilon_grad &&
      !direct_material_any &&
      physical_numel_h > 0 && physical_numel_h != shot_numel_h &&
      ((storage_full_h &&
        shot_bytes_uncomp_h == (int64_t)physical_full_bytes_per_shot) ||
       (storage_bf16_h &&
        shot_bytes_uncomp_h == (int64_t)physical_bf16_bytes_per_shot));
  bool const use_storage_pipeline =
      (storage_mode_h == STORAGE_CPU || storage_mode_h == STORAGE_DISK) &&
      storage_full_domain &&
      (ca_requires_grad || cb_requires_grad || direct_material_any) &&
      stream_storage != nullptr &&
      stream_storage != stream_compute;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);

  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);
  dim3 const physical_grid = make_tm_cell_grid(
      launch_cfg.dimBlock, n_shots_h, physical_ny_h, physical_nx_h);

  void *async_disk_ey = nullptr;
  void *async_disk_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad || direct_material_any)
      async_disk_ey =
          storage_async_disk_open(ey_filenames[0], false, NUM_BUFFERS);
    if (cb_requires_grad || direct_epsilon_grad || direct_material_ecurl_grad)
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
    bool const grad_eps_step =
        do_grad && direct_epsilon_grad && grad_eps != nullptr;
    bool const direct_material_step =
        do_grad && direct_material_grad &&
        (grad_eps != nullptr || grad_sigma != nullptr);
    bool const direct_material_ecurl_step =
        do_grad && direct_material_ecurl_grad &&
        (grad_eps != nullptr || grad_sigma != nullptr);
    bool const grad_ey =
        do_grad && ca_requires_grad && !direct_epsilon_grad &&
        !direct_material_any;
    bool const grad_curl =
        do_grad && cb_requires_grad && !direct_epsilon_grad &&
        !direct_material_any;
    bool const want_load =
        grad_eps_step || direct_material_step || direct_material_ecurl_step ||
        grad_ey || grad_curl;
    if (want_load) {
      slot = (int)(store_idx % NUM_BUFFERS);
      if (use_storage_pipeline) {
        slot_storage_done = storage_done_events.events[slot];
        slot_compute_done = compute_done_events.events[slot];
      }
    }

    size_t const store1_offset = ring_storage_offset_bytes(
        store_idx, storage_mode_h, bytes_per_step_store);
    size_t const next_store1_offset = ring_storage_offset_bytes(
        store_idx + 1, storage_mode_h, bytes_per_step_store);
    size_t const store3_offset = host_storage_offset_bytes(
        store_idx, storage_mode_h, bytes_per_step_store);

    void *__restrict const ey_store_1_t = (uint8_t *)ey_store_1 + store1_offset;
    void *__restrict const ey_next_store_1_t =
        (uint8_t *)ey_store_1 + next_store1_offset;
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
      if (grad_ey || direct_material_ecurl_step) {
        storage_copy_snapshot_h2d((void *)ey_store_1_t, (void *)ey_store_3_t,
                                  (size_t)shot_bytes_uncomp_h,
                                  (size_t)n_shots_h, load_stream);
      }
      if (grad_curl || grad_eps_step || direct_material_ecurl_step) {
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
      if (grad_ey || direct_material_ecurl_step) {
        storage_async_disk_wait_slot(async_disk_ey, slot);
        tide::cuda_check_or_abort(
            cudaMemcpyAsync((void *)ey_store_1_t, (void *)ey_store_3_t,
                            bytes_per_step_store, cudaMemcpyHostToDevice,
                            load_stream),
            __FILE__, __LINE__);
      }
      if (grad_curl || grad_eps_step || direct_material_ecurl_step) {
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

    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x, m_lambda_ey_z, ay,
        ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        ca, cb, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
        m_lambda_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);

    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<launch_cfg.dimGridReceivers,
                               launch_cfg.dimBlockReceivers, 0,
                               stream_compute>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

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

    if (direct_material_step) {
      if (storage_bf16_h) {
        launch_material_grad_direct_eonly_kernel<__nv_bfloat16>(
            launch_cfg, stream_compute, cb, lambda_ey,
            (__nv_bfloat16 const *)ey_store_1_t,
            (__nv_bfloat16 const *)ey_next_store_1_t, grad_eps, grad_sigma,
            grad_ca_shot, grad_cb_shot, grad_eps != nullptr,
            grad_sigma != nullptr, direct_material_endpoint_grad, step_ratio_h,
            dt_h);
      } else {
        launch_material_grad_direct_eonly_kernel<TIDE_DTYPE>(
            launch_cfg, stream_compute, cb, lambda_ey,
            (TIDE_DTYPE const *)ey_store_1_t,
            (TIDE_DTYPE const *)ey_next_store_1_t, grad_eps, grad_sigma,
            grad_ca_shot, grad_cb_shot, grad_eps != nullptr,
            grad_sigma != nullptr, direct_material_endpoint_grad, step_ratio_h,
            dt_h);
      }
    } else if (direct_material_ecurl_step) {
      if (storage_bf16_h) {
        launch_material_grad_direct_ecurl_kernel<__nv_bfloat16>(
            launch_cfg, stream_compute, ca, cb, lambda_ey,
            (__nv_bfloat16 const *)ey_store_1_t,
            (__nv_bfloat16 const *)curl_store_1_t, grad_eps, grad_sigma,
            grad_eps != nullptr, grad_sigma != nullptr, step_ratio_h, dt_h);
      } else {
        launch_material_grad_direct_ecurl_kernel<TIDE_DTYPE>(
            launch_cfg, stream_compute, ca, cb, lambda_ey,
            (TIDE_DTYPE const *)ey_store_1_t,
            (TIDE_DTYPE const *)curl_store_1_t, grad_eps, grad_sigma,
            grad_eps != nullptr, grad_sigma != nullptr, step_ratio_h, dt_h);
      }
    } else if (grad_eps_step) {
      if (storage_bf16_h) {
        launch_direct_epsilon_grad_delta_kernel<__nv_bfloat16>(
            launch_cfg, stream_compute, cb, lambda_ey,
            (__nv_bfloat16 const *)curl_store_1_t, grad_ca_shot,
            step_ratio_h, dt_h);
      } else {
        launch_direct_epsilon_grad_delta_kernel<TIDE_DTYPE>(
            launch_cfg, stream_compute, cb, lambda_ey,
            (TIDE_DTYPE const *)curl_store_1_t, grad_ca_shot, step_ratio_h,
            dt_h);
      }
    } else if (grad_ey || grad_curl) {
      if (storage_bf16_h) {
        if (storage_physical) {
          launch_coeff_grad_physical_kernel<__nv_bfloat16>(
              launch_cfg, physical_grid, stream_compute, lambda_ey,
              grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
              grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
        } else {
          launch_coeff_grad_kernel<__nv_bfloat16>(
              launch_cfg, stream_compute, lambda_ey,
              grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
              grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
        }
      } else {
        if (storage_physical) {
          launch_coeff_grad_physical_kernel<TIDE_DTYPE>(
              launch_cfg, physical_grid, stream_compute, lambda_ey,
              grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
              grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
        } else {
          launch_coeff_grad_kernel<TIDE_DTYPE>(
              launch_cfg, stream_compute, lambda_ey,
              grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
              grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h);
        }
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
        if (grad_ey || direct_material_ecurl_step) {
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
        if (grad_curl || grad_eps_step || direct_material_ecurl_step) {
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

  if ((direct_epsilon_grad || direct_material_grad) && grad_eps != nullptr &&
      !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_eps, grad_ca_shot);
  } else if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_ca, grad_ca_shot);
  }
  if (direct_material_grad && grad_sigma != nullptr && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_sigma, grad_cb_shot);
  } else if (!direct_epsilon_grad && !direct_material_grad && cb_requires_grad &&
             !cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_cb, grad_cb_shot);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(born_backward_bggrad)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb, TIDE_DTYPE const *const f0,
    TIDE_DTYPE const *const df, TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const ey_store_1, void *const ey_store_3,
    char const *const *const ey_filenames, TIDE_DTYPE *const curl_store_1,
    void *const curl_store_3, char const *const *const curl_filenames,
    void const *const dey_store,
    void const *const dcurl_store,
    TIDE_DTYPE *const ey, TIDE_DTYPE *const hx, TIDE_DTYPE *const hz,
    TIDE_DTYPE *const dey, TIDE_DTYPE *const dhx, TIDE_DTYPE *const dhz,
    TIDE_DTYPE *const grad_f0, TIDE_DTYPE *const grad_df,
    TIDE_DTYPE *const grad_ca, TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_dca, TIDE_DTYPE *const grad_dcb,
    TIDE_DTYPE *const m_lambda_ey_x, TIDE_DTYPE *const m_lambda_ey_z,
    TIDE_DTYPE *const m_lambda_hx_z, TIDE_DTYPE *const m_lambda_hz_x,
    TIDE_DTYPE *const m_eta_ey_x, TIDE_DTYPE *const m_eta_ey_z,
    TIDE_DTYPE *const m_eta_hx_z, TIDE_DTYPE *const m_eta_hz_x,
    TIDE_DTYPE *const eta_source_old, TIDE_DTYPE *const work_eta_x,
    TIDE_DTYPE *const work_eta_z, TIDE_DTYPE *const grad_ca_shot,
    TIDE_DTYPE *const grad_cb_shot, TIDE_DTYPE *const grad_dca_shot,
    TIDE_DTYPE *const grad_dcb_shot,
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    tide_scalar_t const rdy_h, tide_scalar_t const rdx_h,
    tide_scalar_t const dt_h, int64_t const nt, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    int64_t const storage_mode_h, int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp_h, bool const ca_requires_grad,
    bool const cb_requires_grad, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle, void *const storage_stream_handle) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)n_threads;
  (void)f0;
  (void)df;
  (void)ey_store_3;
  (void)curl_store_3;
  (void)ey_filenames;
  (void)curl_filenames;
  (void)storage_stream_handle;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);

  if (storage_mode_h != STORAGE_DEVICE) {
    std::fprintf(stderr,
                 "born_backward_bggrad currently supports storage_mode='device' "
                 "only in the TM2D CUDA prototype.\n");
    std::abort();
  }
  if (!ca_requires_grad || !cb_requires_grad) {
    std::fprintf(stderr,
                 "born_backward_bggrad requires both Ey and curl snapshots in "
                 "the current TM2D CUDA prototype.\n");
    std::abort();
  }
  if (dey_store == nullptr || dcurl_store == nullptr) {
    std::fprintf(stderr,
                 "born_backward_bggrad requires explicit scattered snapshots.\n");
    std::abort();
  }

  int64_t const shot_numel_h = ny_h * nx_h;
  int64_t const store_size = n_shots_h * shot_numel_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (storage_format_h == STORAGE_FORMAT_BF16);

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
  dim3 dimBlock_combine(32, 32, 1);
  dim3 dimGrid_combine(
      (nx_h - 2 * kFdPad + dimBlock_combine.x - 1) / dimBlock_combine.x,
      (ny_h - 2 * kFdPad + dimBlock_combine.y - 1) / dimBlock_combine.y, 1);

  auto zero_tensor = [&](TIDE_DTYPE *ptr, size_t count) {
    tide::cuda_check_or_abort(
        cudaMemsetAsync(ptr, 0, count * sizeof(TIDE_DTYPE), stream_compute),
        __FILE__, __LINE__);
  };

  TIDE_DTYPE *lambda_ey = ey;
  TIDE_DTYPE *lambda_hx = hx;
  TIDE_DTYPE *lambda_hz = hz;
  TIDE_DTYPE *eta_ey = dey;
  TIDE_DTYPE *eta_hx = dhx;
  TIDE_DTYPE *eta_hz = dhz;
  zero_tensor(lambda_ey, (size_t)store_size);
  zero_tensor(lambda_hx, (size_t)store_size);
  zero_tensor(lambda_hz, (size_t)store_size);
  zero_tensor(eta_ey, (size_t)store_size);
  zero_tensor(eta_hx, (size_t)store_size);
  zero_tensor(eta_hz, (size_t)store_size);
  zero_tensor(m_lambda_ey_x, (size_t)store_size);
  zero_tensor(m_lambda_ey_z, (size_t)store_size);
  zero_tensor(m_lambda_hx_z, (size_t)store_size);
  zero_tensor(m_lambda_hz_x, (size_t)store_size);
  zero_tensor(m_eta_ey_x, (size_t)store_size);
  zero_tensor(m_eta_ey_z, (size_t)store_size);
  zero_tensor(m_eta_hx_z, (size_t)store_size);
  zero_tensor(m_eta_hz_x, (size_t)store_size);
  zero_tensor(eta_source_old, (size_t)store_size);
  zero_tensor(work_eta_x, (size_t)store_size);
  zero_tensor(work_eta_z, (size_t)store_size);

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    int64_t const store_idx = t / step_ratio_h;
    bool const do_grad = (t % step_ratio_h) == 0;

    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x, m_lambda_ey_z, ay,
        ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        ca, cb, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
        m_lambda_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, eta_ey, eta_hx, eta_hz, m_eta_ey_x, m_eta_ey_z, ay, ayh, ax, axh,
        by, byh, bx, bxh, ky, kyh, kx, kxh);
    forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        ca, cb, eta_hx, eta_hz, eta_ey, m_eta_hx_z, m_eta_hz_x, ay, ayh, ax,
        axh, by, byh, bx, bxh, ky, kyh, kx, kxh);

    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<launch_cfg.dimGridReceivers,
                               launch_cfg.dimBlockReceivers, 0,
                               stream_compute>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }
    if (do_grad) {
      size_t const direct_store_offset =
          (size_t)store_idx * (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
      void const *const dey_store_t =
          (uint8_t const *)dey_store + direct_store_offset;
      void const *const dcurl_store_t =
          (uint8_t const *)dcurl_store + direct_store_offset;

      if (storage_bf16_h) {
        born_background_prepare_direct_kernel<__nv_bfloat16>
            <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
                dca, dcb, lambda_ey, (__nv_bfloat16 const *)dey_store_t,
                (__nv_bfloat16 const *)dcurl_store_t, grad_ca_shot,
                grad_cb_shot, eta_source_old, work_eta_x, work_eta_z, ay, ayh,
                ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, step_ratio_h);
      } else {
        born_background_prepare_direct_kernel<TIDE_DTYPE>
            <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
                dca, dcb, lambda_ey, (TIDE_DTYPE const *)dey_store_t,
                (TIDE_DTYPE const *)dcurl_store_t, grad_ca_shot, grad_cb_shot,
                eta_source_old, work_eta_x, work_eta_z, ay, ayh, ax, axh, by,
                byh, bx, bxh, ky, kyh, kx, kxh, step_ratio_h);
      }
      born_backward_apply_e_to_h_kernel<<<launch_cfg.dimGrid,
                                          launch_cfg.dimBlock, 0,
                                          stream_compute>>>(
          work_eta_x, work_eta_z, eta_hx, eta_hz, ay, ayh, ax, axh, by, byh,
          bx, bxh, ky, kyh, kx, kxh);
      if (storage_bf16_h) {
        __nv_bfloat16 const *const ey_store_t =
            (__nv_bfloat16 const *)ey_store_1 + store_idx * store_size;
        __nv_bfloat16 const *const curl_store_t =
            (__nv_bfloat16 const *)curl_store_1 + store_idx * store_size;
        launch_coeff_grad_kernel<__nv_bfloat16>(
            launch_cfg, stream_compute, lambda_ey, ey_store_t, curl_store_t,
            grad_dca_shot, grad_dcb_shot, true, true, step_ratio_h);
      } else {
        TIDE_DTYPE const *const ey_store_t =
            (TIDE_DTYPE const *)ey_store_1 + store_idx * store_size;
        TIDE_DTYPE const *const curl_store_t =
            (TIDE_DTYPE const *)curl_store_1 + store_idx * store_size;
        launch_coeff_grad_kernel<TIDE_DTYPE>(
            launch_cfg, stream_compute, lambda_ey, ey_store_t, curl_store_t,
            grad_dca_shot, grad_dcb_shot, true, true, step_ratio_h);
      }
      add_inplace<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                    stream_compute>>>(eta_ey, eta_source_old);
    }

    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources<<<launch_cfg.dimGridSources,
                                  launch_cfg.dimBlockSources, 0,
                                  stream_compute>>>(
          grad_df + t * n_shots_h * n_sources_per_shot_h, lambda_ey, sources_i);
      record_adjoint_at_sources<<<launch_cfg.dimGridSources,
                                  launch_cfg.dimBlockSources, 0,
                                  stream_compute>>>(
          grad_f0 + t * n_shots_h * n_sources_per_shot_h, eta_ey, sources_i);
    }
  }

  if (!ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_ca, grad_ca_shot);
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dca, grad_dca_shot);
  } else {
    add_inplace<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
        grad_ca, grad_ca_shot);
    add_inplace<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
        grad_dca, grad_dca_shot);
  }
  if (!cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_cb, grad_cb_shot);
    combine_grad<<<dimGrid_combine, dimBlock_combine, 0, stream_compute>>>(
        grad_dcb, grad_dcb_shot);
  } else {
    add_inplace<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
        grad_cb, grad_cb_shot);
    add_inplace<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
        grad_dcb, grad_dcb_shot);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaStreamSynchronize(stream_compute), __FILE__,
                            __LINE__);
}

} // namespace FUNC(Inst)
