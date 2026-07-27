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
  // These symbols are shared by every pass and precision variant in this
  // stencil instantiation, while each host entry point previously owned an
  // independent cache. Alternating native and reduced-I/O calls could therefore
  // leave a stale symbol value behind. The copy cost is paid once per solver
  // call and is negligible relative to time stepping, so always synchronize.

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

#if defined(TIDE_BUILD_FP16_IO) && TIDE_BUILD_FP16_IO

// Experimental native-FP32 path inspired by SeisCL's vectorized kernels. Each
// thread owns two adjacent x cells, while a whole block cooperatively stages a
// scalar field tile (including the finite-difference halo) in shared memory.
// Keeping the tile scalar makes the cross-pair x stencil explicit and avoids
// imposing an aligned/padded public tensor layout.
struct SharedFloatTileAccessor {
  float const *tile;
  int width;
  int origin_y;
  int origin_x;

  __device__ __forceinline__ float operator()(int64_t, int const y,
                                               int const x) const {
    return tile[(y - origin_y) * width + (x - origin_x)];
  }
};

__device__ __forceinline__ float2 load_float2_safe(float const *ptr,
                                                    int64_t const i) {
  if ((reinterpret_cast<uintptr_t>(ptr + i) & 0x7U) == 0U) {
    return *reinterpret_cast<float2 const *>(ptr + i);
  }
  return make_float2(ptr[i], ptr[i + 1]);
}

__device__ __forceinline__ void store_float2_safe(float *ptr,
                                                   int64_t const i,
                                                   float2 const value) {
  if ((reinterpret_cast<uintptr_t>(ptr + i) & 0x7U) == 0U) {
    *reinterpret_cast<float2 *>(ptr + i) = value;
  } else {
    ptr[i] = value.x;
    ptr[i + 1] = value.y;
  }
}

__device__ __forceinline__ void load_fp32_pair_tile(
    float *__restrict const tile, float const *__restrict const field,
    int64_t const base, int const tile_width, int const tile_height,
    int const origin_y, int const origin_x) {
  int const local_linear =
      (int)threadIdx.y * (int)blockDim.x + (int)threadIdx.x;
  int const block_threads = (int)blockDim.x * (int)blockDim.y;
  int const tile_count = tile_width * tile_height;
  for (int linear = local_linear; linear < tile_count;
       linear += block_threads) {
    int const local_y = linear / tile_width;
    int const local_x = linear - local_y * tile_width;
    int const global_y = origin_y + local_y;
    int const global_x = origin_x + local_x;
    tile[linear] =
        global_y >= 0 && global_y < ny && global_x >= 0 && global_x < nx
            ? field[base + (int64_t)global_y * nx + global_x]
            : 0.0f;
  }
}

static inline dim3 make_fp32_pair_grid(TMForwardLaunchConfig const &cfg,
                                       int64_t const nx_h) {
  int64_t const active_x = nx_h - 2 * kFdPad + 1;
  int64_t const pairs = (active_x + 1) / 2;
  return dim3(to_dim_u32((pairs + cfg.dimBlock.x - 1) / cfg.dimBlock.x),
              cfg.dimGrid.y, cfg.dimGrid.z);
}

static inline size_t fp32_pair_tile_bytes(TMForwardLaunchConfig const &cfg) {
  size_t const tile_width = 2 * (size_t)cfg.dimBlock.x + 2 * (size_t)kFdPad;
  size_t const tile_height = (size_t)cfg.dimBlock.y + 2 * (size_t)kFdPad;
  return tile_width * tile_height * sizeof(float);
}

__global__ __launch_bounds__(256) void forward_kernel_h_fp32_pair_tile(
    float const *__restrict const cq, float const *__restrict const ey,
    float *__restrict const hx, float *__restrict const hz,
    float *__restrict const m_ey_x, float *__restrict const m_ey_z,
    float const *__restrict const ayh, float const *__restrict const axh,
    float const *__restrict const byh, float const *__restrict const bxh,
    float const *__restrict const kyh, float const *__restrict const kxh) {
  extern __shared__ float tile[];
  int const tile_width = 2 * (int)blockDim.x + 2 * kFdPad;
  int const tile_height = (int)blockDim.y + 2 * kFdPad;
  int const origin_y = (int)blockIdx.y * (int)blockDim.y;
  int const origin_x = 2 * (int)blockIdx.x * (int)blockDim.x;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const base = shot_idx * shot_numel;
  load_fp32_pair_tile(tile, ey, base, tile_width, tile_height, origin_y,
                      origin_x);
  __syncthreads();

  int const y = kFdPad + origin_y + (int)threadIdx.y;
  int const x0 = kFdPad + origin_x + 2 * (int)threadIdx.x;
  if (shot_idx >= n_shots || y >= ny - kFdPad + 1 ||
      x0 >= nx - kFdPad + 1) {
    return;
  }

  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const i0 = base + (int64_t)y * nx + x0;
  float2 hx_new = two ? load_float2_safe(hx, i0)
                      : make_float2(hx[i0], 0.0f);
  float2 hz_new = two ? load_float2_safe(hz, i0)
                      : make_float2(hz[i0], 0.0f);
  SharedFloatTileAccessor const field{tile, tile_width, origin_y, origin_x};
  int64_t const pml_y1h = pml_y1 > pml_y0 ? pml_y1 - 1 : pml_y0;
  int64_t const pml_x1h = pml_x1 > pml_x0 ? pml_x1 - 1 : pml_x0;

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int const x = x0 + lane;
    int64_t const j = (int64_t)y * nx + x;
    int64_t const i = base + j;
    float const cq_val = cq[cq_batched ? i : j];
    if (y < ny - kFdPad) {
      float dey_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
          field, 0, y, x, (int)nx, (float)rdy);
      if (y < pml_y0 || y >= pml_y1h) {
        float const memory = byh[y] * m_ey_z[i] + ayh[y] * dey_dz;
        m_ey_z[i] = memory;
        dey_dz = dey_dz / kyh[y] + memory;
      }
      (&hx_new.x)[lane] -= cq_val * dey_dz;
    }
    if (x < nx - kFdPad) {
      float dey_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
          field, 0, y, x, (int)nx, (float)rdx);
      if (x < pml_x0 || x >= pml_x1h) {
        float const memory = bxh[x] * m_ey_x[i] + axh[x] * dey_dx;
        m_ey_x[i] = memory;
        dey_dx = dey_dx / kxh[x] + memory;
      }
      (&hz_new.x)[lane] += cq_val * dey_dx;
    }
  }

  if (two) {
    store_float2_safe(hx, i0, hx_new);
    store_float2_safe(hz, i0, hz_new);
  } else {
    hx[i0] = hx_new.x;
    hz[i0] = hz_new.x;
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_fp32_pair_tile(
    float const *__restrict const ca, float const *__restrict const cb,
    float const *__restrict const hx, float const *__restrict const hz,
    float *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx) {
  extern __shared__ float shared_tiles[];
  int const tile_width = 2 * (int)blockDim.x + 2 * kFdPad;
  int const tile_height = (int)blockDim.y + 2 * kFdPad;
  int const tile_count = tile_width * tile_height;
  float *const hz_tile = shared_tiles;
  float *const hx_tile = shared_tiles + tile_count;
  int const origin_y = (int)blockIdx.y * (int)blockDim.y;
  int const origin_x = 2 * (int)blockIdx.x * (int)blockDim.x;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int64_t const base = shot_idx * shot_numel;
  int const y = kFdPad + origin_y + (int)threadIdx.y;
  int const x0 = kFdPad + origin_x + 2 * (int)threadIdx.x;
  bool const active = shot_idx < n_shots && y < ny - kFdPad + 1 &&
                      x0 < nx - kFdPad + 1;
  bool const two = active && x0 + 1 < nx - kFdPad + 1;
  load_fp32_pair_tile(hz_tile, hz, base, tile_width, tile_height, origin_y,
                      origin_x);
  load_fp32_pair_tile(hx_tile, hx, base, tile_width, tile_height, origin_y,
                      origin_x);
  __syncthreads();
  SharedFloatTileAccessor const hz_field{hz_tile, tile_width, origin_y,
                                         origin_x};
  SharedFloatTileAccessor const hx_field{hx_tile, tile_width, origin_y,
                                         origin_x};
  float2 dhz_dx = make_float2(0.0f, 0.0f);
  if (active) {
    dhz_dx.x = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
        hz_field, 0, y, x0, (int)nx, (float)rdx);
    if (two) {
      dhz_dx.y = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
          hz_field, 0, y, x0 + 1, (int)nx, (float)rdx);
    }
  }
  if (!active) return;

  float2 dhx_dz = make_float2(
      ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
          hx_field, 0, y, x0, (int)nx, (float)rdy),
      two ? ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
                hx_field, 0, y, x0 + 1, (int)nx, (float)rdy)
          : 0.0f);
  int64_t const i0 = base + (int64_t)y * nx + x0;
  float2 ey_new = two ? load_float2_safe(ey, i0)
                      : make_float2(ey[i0], 0.0f);

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int const x = x0 + lane;
    int64_t const j = (int64_t)y * nx + x;
    int64_t const i = base + j;
    float dx = (&dhz_dx.x)[lane];
    float dz = (&dhx_dz.x)[lane];
    if (x < pml_x0 || x >= pml_x1) {
      float const memory = bx[x] * m_hz_x[i] + ax[x] * dx;
      m_hz_x[i] = memory;
      dx = dx / kx[x] + memory;
    }
    if (y < pml_y0 || y >= pml_y1) {
      float const memory = by[y] * m_hx_z[i] + ay[y] * dz;
      m_hx_z[i] = memory;
      dz = dz / ky[y] + memory;
    }
    float const ca_val = ca[ca_batched ? i : j];
    float const cb_val = cb[cb_batched ? i : j];
    (&ey_new.x)[lane] =
        ca_val * (&ey_new.x)[lane] + cb_val * (dx - dz);
  }

  if (two) {
    store_float2_safe(ey, i0, ey_new);
  } else {
    ey[i0] = ey_new.x;
  }
}

#endif

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

#if defined(TIDE_BUILD_FP16_IO) && TIDE_BUILD_FP16_IO

// Stage-one compact CPML path. Primary fields and material coefficients keep
// the native scalar FP32 layout. Only directional CPML memories are compact:
// x memories are [shot, y, compact_x], y memories are
// [shot, compact_y, x]. The extra compact cell covers the inclusive electric
// field endpoint; the staggered H right boundary starts one cell earlier.
__device__ __forceinline__ int64_t compact_pml_index(
    int64_t const coordinate, int64_t const left_begin,
    int64_t const right_begin, int64_t const left_width) {
  return coordinate < left_begin ? coordinate - kFdPad
                                 : left_width + coordinate - right_begin;
}

__global__ __launch_bounds__(256) void forward_kernel_h_compact(
    float const *__restrict const cq, float const *__restrict const ey,
    float *__restrict const hx, float *__restrict const hz,
    float *__restrict const m_ey_x, float *__restrict const m_ey_z,
    float const *__restrict const ayh, float const *__restrict const axh,
    float const *__restrict const byh, float const *__restrict const bxh,
    float const *__restrict const kyh, float const *__restrict const kxh) {
  int64_t const x = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const shot = (int64_t)blockIdx.z;
  if (shot >= n_shots || y < kFdPad || x < kFdPad ||
      y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot * shot_numel + j;
  float const cq_val = cq[cq_batched ? i : j];
  ::tide::GlobalFieldAccessor<float> const field(ey, nx);
  int64_t const pml_y1h = pml_y1 > pml_y0 ? pml_y1 - 1 : pml_y0;
  int64_t const pml_x1h = pml_x1 > pml_x0 ? pml_x1 - 1 : pml_x0;
  int64_t const compact_y = (pml_y0 - kFdPad) + (ny - kFdPad + 1 - pml_y1);
  int64_t const compact_x = (pml_x0 - kFdPad) + (nx - kFdPad + 1 - pml_x1);

  if (y < ny - kFdPad) {
    float derivative = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
        field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdy);
    if (y < pml_y0 || y >= pml_y1h) {
      int64_t const cy = compact_pml_index(y, pml_y0, pml_y1h,
                                            pml_y0 - kFdPad);
      int64_t const mi = (shot * compact_y + cy) * nx + x;
      float const memory = byh[y] * m_ey_z[mi] + ayh[y] * derivative;
      m_ey_z[mi] = memory;
      derivative = derivative / kyh[y] + memory;
    }
    hx[i] -= cq_val * derivative;
  }

  if (x < nx - kFdPad) {
    float derivative = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
        field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdx);
    if (x < pml_x0 || x >= pml_x1h) {
      int64_t const cx = compact_pml_index(x, pml_x0, pml_x1h,
                                            pml_x0 - kFdPad);
      int64_t const mi = (shot * ny + y) * compact_x + cx;
      float const memory = bxh[x] * m_ey_x[mi] + axh[x] * derivative;
      m_ey_x[mi] = memory;
      derivative = derivative / kxh[x] + memory;
    }
    hz[i] += cq_val * derivative;
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_compact(
    float const *__restrict const ca, float const *__restrict const cb,
    float const *__restrict const hx, float const *__restrict const hz,
    float *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx) {
  int64_t const x = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const shot = (int64_t)blockIdx.z;
  if (shot >= n_shots || y < kFdPad || x < kFdPad ||
      y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot * shot_numel + j;
  ::tide::GlobalFieldAccessor<float> const hz_field(hz, nx);
  ::tide::GlobalFieldAccessor<float> const hx_field(hx, nx);
  float dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
      hz_field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdx);
  float dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
      hx_field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdy);
  int64_t const compact_y = (pml_y0 - kFdPad) + (ny - kFdPad + 1 - pml_y1);
  int64_t const compact_x = (pml_x0 - kFdPad) + (nx - kFdPad + 1 - pml_x1);

  if (x < pml_x0 || x >= pml_x1) {
    int64_t const cx = compact_pml_index(x, pml_x0, pml_x1,
                                          pml_x0 - kFdPad);
    int64_t const mi = (shot * ny + y) * compact_x + cx;
    float const memory = bx[x] * m_hz_x[mi] + ax[x] * dx;
    m_hz_x[mi] = memory;
    dx = dx / kx[x] + memory;
  }
  if (y < pml_y0 || y >= pml_y1) {
    int64_t const cy = compact_pml_index(y, pml_y0, pml_y1,
                                          pml_y0 - kFdPad);
    int64_t const mi = (shot * compact_y + cy) * nx + x;
    float const memory = by[y] * m_hx_z[mi] + ay[y] * dz;
    m_hx_z[mi] = memory;
    dz = dz / ky[y] + memory;
  }

  float const ca_val = ca[ca_batched ? i : j];
  float const cb_val = cb[cb_batched ? i : j];
  ey[i] = ca_val * ey[i] + cb_val * (dx - dz);
}

__global__ __launch_bounds__(256) void pack_material_pairs(
    float const *__restrict const ca, float const *__restrict const cb,
    float const *__restrict const cq, float4 *__restrict const ca_cb_pairs,
    float2 *__restrict const cq_pairs, int64_t const pair_count) {
  int64_t const pair = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = kFdPad + (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  if (pair >= pair_count || y >= ny - kFdPad + 1) return;
  int64_t const x0 = kFdPad + 2 * pair;
  int64_t const j0 = y * nx + x0;
  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const packed = (y - kFdPad) * pair_count + pair;
  ca_cb_pairs[packed] =
      make_float4(ca[j0], two ? ca[j0 + 1] : 0.0f, cb[j0],
                  two ? cb[j0 + 1] : 0.0f);
  cq_pairs[packed] =
      make_float2(cq[j0], two ? cq[j0 + 1] : 0.0f);
}

__global__ __launch_bounds__(256) void forward_kernel_h_compact_vec2(
    float2 const *__restrict const cq_pairs,
    float const *__restrict const ey, float *__restrict const hx,
    float *__restrict const hz, float *__restrict const m_ey_x,
    float *__restrict const m_ey_z, float const *__restrict const ayh,
    float const *__restrict const axh, float const *__restrict const byh,
    float const *__restrict const bxh, float const *__restrict const kyh,
    float const *__restrict const kxh, int64_t const pair_count) {
  int64_t const pair = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = kFdPad + (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const shot = (int64_t)blockIdx.z;
  if (shot >= n_shots || pair >= pair_count || y >= ny - kFdPad + 1) return;
  int64_t const x0 = kFdPad + 2 * pair;
  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const i0 = shot * shot_numel + y * nx + x0;
  float2 hx_new = two ? load_float2_safe(hx, i0) : make_float2(hx[i0], 0.0f);
  float2 hz_new = two ? load_float2_safe(hz, i0) : make_float2(hz[i0], 0.0f);
  float2 const cq_pair = cq_pairs[(y - kFdPad) * pair_count + pair];
  ::tide::GlobalFieldAccessor<float> const field(ey, nx);
  int64_t const pml_y1h = pml_y1 > pml_y0 ? pml_y1 - 1 : pml_y0;
  int64_t const pml_x1h = pml_x1 > pml_x0 ? pml_x1 - 1 : pml_x0;
  int64_t const compact_y = (pml_y0 - kFdPad) + (ny - kFdPad + 1 - pml_y1);
  int64_t const compact_x = (pml_x0 - kFdPad) + (nx - kFdPad + 1 - pml_x1);

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int64_t const x = x0 + lane;
    float const cq_val = (&cq_pair.x)[lane];
    if (y < ny - kFdPad) {
      float derivative = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
          field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdy);
      if (y < pml_y0 || y >= pml_y1h) {
        int64_t const cy = compact_pml_index(y, pml_y0, pml_y1h,
                                              pml_y0 - kFdPad);
        int64_t const mi = (shot * compact_y + cy) * nx + x;
        float const memory = byh[y] * m_ey_z[mi] + ayh[y] * derivative;
        m_ey_z[mi] = memory;
        derivative = derivative / kyh[y] + memory;
      }
      (&hx_new.x)[lane] -= cq_val * derivative;
    }
    if (x < nx - kFdPad) {
      float derivative = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
          field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdx);
      if (x < pml_x0 || x >= pml_x1h) {
        int64_t const cx = compact_pml_index(x, pml_x0, pml_x1h,
                                              pml_x0 - kFdPad);
        int64_t const mi = (shot * ny + y) * compact_x + cx;
        float const memory = bxh[x] * m_ey_x[mi] + axh[x] * derivative;
        m_ey_x[mi] = memory;
        derivative = derivative / kxh[x] + memory;
      }
      (&hz_new.x)[lane] += cq_val * derivative;
    }
  }
  if (two) {
    store_float2_safe(hx, i0, hx_new);
    store_float2_safe(hz, i0, hz_new);
  } else {
    hx[i0] = hx_new.x;
    hz[i0] = hz_new.x;
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_compact_vec2(
    float4 const *__restrict const ca_cb_pairs,
    float const *__restrict const hx, float const *__restrict const hz,
    float *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx, int64_t const pair_count) {
  int64_t const pair = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = kFdPad + (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  int64_t const shot = (int64_t)blockIdx.z;
  if (shot >= n_shots || pair >= pair_count || y >= ny - kFdPad + 1) return;
  int64_t const x0 = kFdPad + 2 * pair;
  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const i0 = shot * shot_numel + y * nx + x0;
  float2 ey_new = two ? load_float2_safe(ey, i0) : make_float2(ey[i0], 0.0f);
  float4 const coeff = ca_cb_pairs[(y - kFdPad) * pair_count + pair];
  ::tide::GlobalFieldAccessor<float> const hz_field(hz, nx);
  ::tide::GlobalFieldAccessor<float> const hx_field(hx, nx);
  int64_t const compact_y = (pml_y0 - kFdPad) + (ny - kFdPad + 1 - pml_y1);
  int64_t const compact_x = (pml_x0 - kFdPad) + (nx - kFdPad + 1 - pml_x1);

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int64_t const x = x0 + lane;
    float dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
        hz_field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdx);
    float dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
        hx_field, shot * shot_numel, (int)y, (int)x, (int)nx, (float)rdy);
    if (x < pml_x0 || x >= pml_x1) {
      int64_t const cx = compact_pml_index(x, pml_x0, pml_x1,
                                            pml_x0 - kFdPad);
      int64_t const mi = (shot * ny + y) * compact_x + cx;
      float const memory = bx[x] * m_hz_x[mi] + ax[x] * dx;
      m_hz_x[mi] = memory;
      dx = dx / kx[x] + memory;
    }
    if (y < pml_y0 || y >= pml_y1) {
      int64_t const cy = compact_pml_index(y, pml_y0, pml_y1,
                                            pml_y0 - kFdPad);
      int64_t const mi = (shot * compact_y + cy) * nx + x;
      float const memory = by[y] * m_hx_z[mi] + ay[y] * dz;
      m_hx_z[mi] = memory;
      dz = dz / ky[y] + memory;
    }
    float const ca_val = lane == 0 ? coeff.x : coeff.y;
    float const cb_val = lane == 0 ? coeff.z : coeff.w;
    (&ey_new.x)[lane] = ca_val * (&ey_new.x)[lane] + cb_val * (dx - dz);
  }
  if (two) {
    store_float2_safe(ey, i0, ey_new);
  } else {
    ey[i0] = ey_new.x;
  }
}

__global__ __launch_bounds__(256) void forward_kernel_h_compact_vec2_shared(
    float2 const *__restrict const cq_pairs,
    float const *__restrict const ey, float *__restrict const hx,
    float *__restrict const hz, float *__restrict const m_ey_x,
    float *__restrict const m_ey_z, float const *__restrict const ayh,
    float const *__restrict const axh, float const *__restrict const byh,
    float const *__restrict const bxh, float const *__restrict const kyh,
    float const *__restrict const kxh, int64_t const pair_count) {
  extern __shared__ float tile[];
  int const tile_width = 2 * (int)blockDim.x + 2 * kFdPad;
  int const tile_height = (int)blockDim.y + 2 * kFdPad;
  int const origin_y = (int)blockIdx.y * (int)blockDim.y;
  int const origin_x = 2 * (int)blockIdx.x * (int)blockDim.x;
  int64_t const shot = (int64_t)blockIdx.z;
  load_fp32_pair_tile(tile, ey, shot * shot_numel, tile_width, tile_height,
                      origin_y, origin_x);
  __syncthreads();

  int64_t const pair = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = kFdPad + origin_y + threadIdx.y;
  if (shot >= n_shots || pair >= pair_count || y >= ny - kFdPad + 1) return;
  int64_t const x0 = kFdPad + origin_x + 2 * threadIdx.x;
  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const i0 = shot * shot_numel + y * nx + x0;
  float2 hx_new = two ? load_float2_safe(hx, i0) : make_float2(hx[i0], 0.0f);
  float2 hz_new = two ? load_float2_safe(hz, i0) : make_float2(hz[i0], 0.0f);
  float2 const cq_pair = cq_pairs[(y - kFdPad) * pair_count + pair];
  SharedFloatTileAccessor const field{tile, tile_width, origin_y, origin_x};
  int64_t const pml_y1h = pml_y1 > pml_y0 ? pml_y1 - 1 : pml_y0;
  int64_t const pml_x1h = pml_x1 > pml_x0 ? pml_x1 - 1 : pml_x0;
  int64_t const compact_y = (pml_y0 - kFdPad) + (ny - kFdPad + 1 - pml_y1);
  int64_t const compact_x = (pml_x0 - kFdPad) + (nx - kFdPad + 1 - pml_x1);

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int64_t const x = x0 + lane;
    float const cq_val = (&cq_pair.x)[lane];
    if (y < ny - kFdPad) {
      float derivative = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
          field, 0, (int)y, (int)x, (int)nx, (float)rdy);
      if (y < pml_y0 || y >= pml_y1h) {
        int64_t const cy = compact_pml_index(y, pml_y0, pml_y1h,
                                              pml_y0 - kFdPad);
        int64_t const mi = (shot * compact_y + cy) * nx + x;
        float const memory = byh[y] * m_ey_z[mi] + ayh[y] * derivative;
        m_ey_z[mi] = memory;
        derivative = derivative / kyh[y] + memory;
      }
      (&hx_new.x)[lane] -= cq_val * derivative;
    }
    if (x < nx - kFdPad) {
      float derivative = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
          field, 0, (int)y, (int)x, (int)nx, (float)rdx);
      if (x < pml_x0 || x >= pml_x1h) {
        int64_t const cx = compact_pml_index(x, pml_x0, pml_x1h,
                                              pml_x0 - kFdPad);
        int64_t const mi = (shot * ny + y) * compact_x + cx;
        float const memory = bxh[x] * m_ey_x[mi] + axh[x] * derivative;
        m_ey_x[mi] = memory;
        derivative = derivative / kxh[x] + memory;
      }
      (&hz_new.x)[lane] += cq_val * derivative;
    }
  }
  if (two) {
    store_float2_safe(hx, i0, hx_new);
    store_float2_safe(hz, i0, hz_new);
  } else {
    hx[i0] = hx_new.x;
    hz[i0] = hz_new.x;
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_compact_vec2_shared(
    float4 const *__restrict const ca_cb_pairs,
    float const *__restrict const hx, float const *__restrict const hz,
    float *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx, int64_t const pair_count) {
  extern __shared__ float tiles[];
  int const tile_width = 2 * (int)blockDim.x + 2 * kFdPad;
  int const tile_height = (int)blockDim.y + 2 * kFdPad;
  int const tile_count = tile_width * tile_height;
  float *const hz_tile = tiles;
  float *const hx_tile = tiles + tile_count;
  int const origin_y = (int)blockIdx.y * (int)blockDim.y;
  int const origin_x = 2 * (int)blockIdx.x * (int)blockDim.x;
  int64_t const shot = (int64_t)blockIdx.z;
  load_fp32_pair_tile(hz_tile, hz, shot * shot_numel, tile_width, tile_height,
                      origin_y, origin_x);
  load_fp32_pair_tile(hx_tile, hx, shot * shot_numel, tile_width, tile_height,
                      origin_y, origin_x);
  __syncthreads();

  int64_t const pair = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const y = kFdPad + origin_y + threadIdx.y;
  if (shot >= n_shots || pair >= pair_count || y >= ny - kFdPad + 1) return;
  int64_t const x0 = kFdPad + origin_x + 2 * threadIdx.x;
  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const i0 = shot * shot_numel + y * nx + x0;
  float2 ey_new = two ? load_float2_safe(ey, i0) : make_float2(ey[i0], 0.0f);
  float4 const coeff = ca_cb_pairs[(y - kFdPad) * pair_count + pair];
  SharedFloatTileAccessor const hz_field{hz_tile, tile_width, origin_y,
                                          origin_x};
  SharedFloatTileAccessor const hx_field{hx_tile, tile_width, origin_y,
                                          origin_x};
  int64_t const compact_y = (pml_y0 - kFdPad) + (ny - kFdPad + 1 - pml_y1);
  int64_t const compact_x = (pml_x0 - kFdPad) + (nx - kFdPad + 1 - pml_x1);

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int64_t const x = x0 + lane;
    float dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
        hz_field, 0, (int)y, (int)x, (int)nx, (float)rdx);
    float dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
        hx_field, 0, (int)y, (int)x, (int)nx, (float)rdy);
    if (x < pml_x0 || x >= pml_x1) {
      int64_t const cx = compact_pml_index(x, pml_x0, pml_x1,
                                            pml_x0 - kFdPad);
      int64_t const mi = (shot * ny + y) * compact_x + cx;
      float const memory = bx[x] * m_hz_x[mi] + ax[x] * dx;
      m_hz_x[mi] = memory;
      dx = dx / kx[x] + memory;
    }
    if (y < pml_y0 || y >= pml_y1) {
      int64_t const cy = compact_pml_index(y, pml_y0, pml_y1,
                                            pml_y0 - kFdPad);
      int64_t const mi = (shot * compact_y + cy) * nx + x;
      float const memory = by[y] * m_hx_z[mi] + ay[y] * dz;
      m_hx_z[mi] = memory;
      dz = dz / ky[y] + memory;
    }
    float const ca_val = lane == 0 ? coeff.x : coeff.y;
    float const cb_val = lane == 0 ? coeff.z : coeff.w;
    (&ey_new.x)[lane] = ca_val * (&ey_new.x)[lane] + cb_val * (dx - dz);
  }
  if (two) {
    store_float2_safe(ey, i0, ey_new);
  } else {
    ey[i0] = ey_new.x;
  }
}

#endif

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
__global__ __launch_bounds__(256)
void born_tangent_kernel_e_from_snapshots(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const dhx,
    TIDE_DTYPE const *__restrict const dhz, TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const dm_hx_z, TIDE_DTYPE *__restrict const dm_hz_x,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    StoreT *__restrict const dey_store,
    StoreT *__restrict const dcurl_h_store,
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
  ::tide::forward_kernel_e_born_from_snapshots_core<
      TIDE_DTYPE, StoreT, TIDE_STENCIL>(
      params, ca, cb, dca, dcb, dhx, dhz, dey, dm_hx_z, dm_hz_x, ey_store,
      curl_h_store, dey_store, dcurl_h_store, y, x, shot_idx);
}

template <typename StoreT>
__global__ __launch_bounds__(256) void born_background_prepare_direct_kernel(
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const cq,
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
      pml_x0,  pml_x1,      ca_batched, cb_batched, cq_batched};
  ::tide::born_background_prepare_direct_core<TIDE_DTYPE, StoreT, TIDE_STENCIL>(
      params, cb, cq, dca, dcb, lambda_sc_ey, dey_store, dcurl_h_store,
      grad_ca_shot, grad_cb_shot, eta_source_old, work_x, work_z,
      step_ratio_val, y, x, shot_idx);
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

__global__ void add_inplace_and_zero(TIDE_DTYPE *__restrict const dest,
                                     TIDE_DTYPE *__restrict const src) {
  int64_t x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t shot_idx =
      (int64_t)blockIdx.z * (int64_t)blockDim.z + (int64_t)threadIdx.z;
  if (shot_idx < n_shots && y < ny && x < nx) {
    int64_t const i = shot_idx * shot_numel + y * nx + x;
    dest[i] += src[i];
    src[i] = static_cast<TIDE_DTYPE>(0);
  }
}

template <typename StoreT, bool GradCa, bool GradCb>
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
  if (shot_idx >= n_shots || y < kFdPad || x < kFdPad ||
      y >= ny - kFdPad + 1 || x >= nx - kFdPad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  TIDE_DTYPE const lambda_val = lambda_ey[i];
  TIDE_DTYPE const step_scale = step_ratio_to_field(step_ratio_val);
  if constexpr (GradCa) {
    TIDE_DTYPE const ey_n =
        tide::decode_snapshot<StoreT, TIDE_DTYPE>(ey_store[i]);
    grad_ca_shot[i] += lambda_val * ey_n * step_scale;
  }
  if constexpr (GradCb) {
    TIDE_DTYPE const curl_h_n =
        tide::decode_snapshot<StoreT, TIDE_DTYPE>(curl_h_store[i]);
    grad_cb_shot[i] += lambda_val * curl_h_n * step_scale;
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
#if defined(TIDE_BUILD_FP16_IO) && TIDE_BUILD_FP16_IO
  bool const fp32_pair_tile =
      read_env_flag("TIDE_TM_FP32_FLOAT2_TILE") && !has_dispersion;
  dim3 const pair_grid = make_fp32_pair_grid(launch_cfg, nx_h);
  size_t const pair_tile_bytes = fp32_pair_tile_bytes(launch_cfg);
#else
  bool const fp32_pair_tile = false;
#endif

  if (debug_path) {
    std::fprintf(stderr, "TIDE TM path: %s\n",
                 fp32_pair_tile ? "fp32_float2_tile" : "baseline");
  }

  auto run_step = [&](int64_t t) {
#if defined(TIDE_BUILD_FP16_IO) && TIDE_BUILD_FP16_IO
    if (fp32_pair_tile) {
      forward_kernel_h_fp32_pair_tile<<<pair_grid, launch_cfg.dimBlock,
                                        pair_tile_bytes, stream_compute>>>(
          cq, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh, kxh);
    } else
#endif
    {
      forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                         stream_compute>>>(
          cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh,
          ky, kyh, kx, kxh);
    }
    if (has_dispersion) {
      forward_kernel_e_debye<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                               stream_compute>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_prev, debye_cp, polarization,
          n_poles_h, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    }
#if defined(TIDE_BUILD_FP16_IO) && TIDE_BUILD_FP16_IO
    else if (fp32_pair_tile) {
      forward_kernel_e_fp32_pair_tile<<<pair_grid, launch_cfg.dimBlock,
                                        2 * pair_tile_bytes, stream_compute>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx);
    }
#endif
    else {
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

#if defined(TIDE_BUILD_FP16_IO) && TIDE_BUILD_FP16_IO

// Forward-only scalar FP32 entry point with compact directional CPML state.
// Its ABI intentionally matches forward(), allowing the Python resolver to
// select it as a backend variant without adding public stride parameters.
extern "C" void FUNC(forward_compact)(
    float const *const ca, float const *const cb, float const *const cq,
    float const *const f, float *const ey, float *const hx, float *const hz,
    float *const m_ey_x, float *const m_ey_z, float *const m_hx_z,
    float *const m_hz_x, float const *const debye_a,
    float const *const debye_b, float const *const debye_cp,
    float *const polarization, float *const ey_prev, float *const r,
    int64_t const n_poles_h, float const *const ay, float const *const by,
    float const *const ayh, float const *const byh, float const *const ax,
    float const *const bx, float const *const axh, float const *const bxh,
    float const *const ky, float const *const kyh, float const *const kx,
    float const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, float const rdy_h, float const rdx_h,
    float const dt_h, int64_t const nt, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    bool const has_dispersion, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h,
    int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle) {
  (void)debye_a;
  (void)debye_b;
  (void)debye_cp;
  (void)polarization;
  (void)ey_prev;
  (void)n_poles_h;
  (void)ay;
  (void)ax;
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;
  if (has_dispersion) {
    std::fprintf(stderr, "Tide compact TM path does not support dispersion.\n");
    std::abort();
  }

  cudaSetDevice(device);
  cudaStream_t const stream_compute = resolve_cuda_stream(compute_stream_handle);
  int64_t const shot_numel_h = ny_h * nx_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);
  TMForwardLaunchConfig const cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h_compact<<<cfg.dimGrid, cfg.dimBlock, 0, stream_compute>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh, kxh);
    forward_kernel_e_compact<<<cfg.dimGrid, cfg.dimBlock, 0, stream_compute>>>(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx);
    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<cfg.dimGridSources, cfg.dimBlockSources, 0,
                       stream_compute>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<cfg.dimGridReceivers, cfg.dimBlockReceivers, 0,
                            stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  }
  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(forward_compact_vec2)(
    float const *const ca, float const *const cb, float const *const cq,
    float const *const f, float *const ey, float *const hx, float *const hz,
    float *const m_ey_x, float *const m_ey_z, float *const m_hx_z,
    float *const m_hz_x, float const *const debye_a,
    float const *const debye_b, float const *const debye_cp,
    float *const polarization, float *const ey_prev, float *const r,
    int64_t const n_poles_h, float const *const ay, float const *const by,
    float const *const ayh, float const *const byh, float const *const ax,
    float const *const bx, float const *const axh, float const *const bxh,
    float const *const ky, float const *const kyh, float const *const kx,
    float const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, float const rdy_h, float const rdx_h,
    float const dt_h, int64_t const nt, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    bool const has_dispersion, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h,
    int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle) {
  (void)debye_a;
  (void)debye_b;
  (void)debye_cp;
  (void)polarization;
  (void)ey_prev;
  (void)n_poles_h;
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;
  if (has_dispersion || ca_batched_h || cb_batched_h || cq_batched_h) {
    std::fprintf(stderr,
                 "Tide compact vec2 TM path requires unbatched nondispersive coefficients.\n");
    std::abort();
  }

  cudaSetDevice(device);
  cudaStream_t const stream_compute = resolve_cuda_stream(compute_stream_handle);
  int64_t const shot_numel_h = ny_h * nx_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, false, false, false, device);
  TMForwardLaunchConfig const cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);
  int64_t const active_x = nx_h - 2 * kFdPad + 1;
  int64_t const active_y = ny_h - 2 * kFdPad + 1;
  int64_t const pair_count = (active_x + 1) / 2;
  TMForwardLaunchConfig vec_cfg = cfg;
  // Diagnostic-only sweep controls. The public mode needs no environment
  // variable; stage-three measurements use these to select the final default.
  if (char const *value = std::getenv("TIDE_TM_VEC2_BLOCK_X")) {
    int const parsed = std::atoi(value);
    if (parsed == 16 || parsed == 32) vec_cfg.dimBlock.x = parsed;
  }
  if (char const *value = std::getenv("TIDE_TM_VEC2_BLOCK_Y")) {
    int const parsed = std::atoi(value);
    if (parsed == 4 || parsed == 8 || parsed == 16) vec_cfg.dimBlock.y = parsed;
  }
  if (vec_cfg.dimBlock.x * vec_cfg.dimBlock.y > 256) {
    vec_cfg.dimBlock = cfg.dimBlock;
  }
  vec_cfg.dimGrid =
      dim3(to_dim_u32((pair_count + vec_cfg.dimBlock.x - 1) /
                      vec_cfg.dimBlock.x),
           to_dim_u32((active_y + vec_cfg.dimBlock.y - 1) /
                      vec_cfg.dimBlock.y),
           to_dim_u32(n_shots_h));
  size_t const packed_count = (size_t)active_y * (size_t)pair_count;
  float4 *ca_cb_pairs = nullptr;
  float2 *cq_pairs = nullptr;
  tide::cuda_check_or_abort(
      cudaMallocAsync(&ca_cb_pairs, packed_count * sizeof(float4), stream_compute),
      __FILE__, __LINE__);
  tide::cuda_check_or_abort(
      cudaMallocAsync(&cq_pairs, packed_count * sizeof(float2), stream_compute),
      __FILE__, __LINE__);

  dim3 const pair_grid = vec_cfg.dimGrid;
  dim3 const pack_grid(pair_grid.x, pair_grid.y, 1);
  bool const shared_tile = read_env_flag("TIDE_TM_VEC2_SHARED_TILE");
  size_t const tile_bytes = fp32_pair_tile_bytes(vec_cfg);
  pack_material_pairs<<<pack_grid, vec_cfg.dimBlock, 0, stream_compute>>>(
      ca, cb, cq, ca_cb_pairs, cq_pairs, pair_count);
  for (int64_t t = start_t; t < start_t + nt; ++t) {
    if (shared_tile) {
      forward_kernel_h_compact_vec2_shared<<<pair_grid, vec_cfg.dimBlock,
                                             tile_bytes, stream_compute>>>(
          cq_pairs, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh,
          kxh, pair_count);
      forward_kernel_e_compact_vec2_shared<<<pair_grid, vec_cfg.dimBlock,
                                             2 * tile_bytes,
                                             stream_compute>>>(
          ca_cb_pairs, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx,
          pair_count);
    } else {
      forward_kernel_h_compact_vec2<<<pair_grid, vec_cfg.dimBlock, 0,
                                      stream_compute>>>(
          cq_pairs, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh,
          kxh, pair_count);
      forward_kernel_e_compact_vec2<<<pair_grid, vec_cfg.dimBlock, 0,
                                      stream_compute>>>(
          ca_cb_pairs, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx,
          pair_count);
    }
    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<cfg.dimGridSources, cfg.dimBlockSources, 0,
                       stream_compute>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<cfg.dimGridReceivers, cfg.dimBlockReceivers, 0,
                            stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  }
  tide::cuda_check_or_abort(cudaFreeAsync(cq_pairs, stream_compute), __FILE__,
                            __LINE__);
  tide::cuda_check_or_abort(cudaFreeAsync(ca_cb_pairs, stream_compute), __FILE__,
                            __LINE__);
  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

namespace {

struct HalfGlobalAccessor {
  half const *ptr;
  int64_t width;

  __device__ __forceinline__ float operator()(int64_t base, int y, int x) const {
    return __half2float(ptr[base + (int64_t)y * width + x]);
  }
};

__device__ __forceinline__ half2 load_half2_safe(half const *ptr,
                                                  int64_t const i) {
  if ((reinterpret_cast<uintptr_t>(ptr + i) & 0x3U) == 0U) {
    return *reinterpret_cast<half2 const *>(ptr + i);
  }
  return __halves2half2(ptr[i], ptr[i + 1]);
}

__device__ __forceinline__ void store_half2_safe(half *ptr, int64_t const i,
                                                  half2 const value) {
  if ((reinterpret_cast<uintptr_t>(ptr + i) & 0x3U) == 0U) {
    *reinterpret_cast<half2 *>(ptr + i) = value;
  } else {
    ptr[i] = __low2half(value);
    ptr[i + 1] = __high2half(value);
  }
}

__device__ __forceinline__ half2 fp16_pair_delta(
    half const *ptr, int64_t const base, int const y, int const x,
    int const y_plus, int const x_plus, int const y_minus,
    int const x_minus) {
  half2 const plus =
      load_half2_safe(ptr, base + (int64_t)(y + y_plus) * nx + x + x_plus);
  half2 const minus = load_half2_safe(
      ptr, base + (int64_t)(y + y_minus) * nx + x + x_minus);
  return __hsub2(plus, minus);
}

template <bool AlongX, bool ForwardHalf>
__device__ __forceinline__ half2 fp16_pair_diff(half const *ptr,
                                                int64_t const base,
                                                int const y, int const x,
                                                float const reciprocal_d) {
  constexpr int yp1 = AlongX ? 0 : (ForwardHalf ? 1 : 0);
  constexpr int xp1 = AlongX ? (ForwardHalf ? 1 : 0) : 0;
  constexpr int ym1 = AlongX ? 0 : (ForwardHalf ? 0 : -1);
  constexpr int xm1 = AlongX ? (ForwardHalf ? 0 : -1) : 0;
  half2 value = fp16_pair_delta(ptr, base, y, x, yp1, xp1, ym1, xm1);
  if constexpr (TIDE_STENCIL >= 4) {
    value = __hmul2(__float2half2_rn(9.0f / 8.0f), value);
    constexpr int yp2 = AlongX ? 0 : (ForwardHalf ? 2 : 1);
    constexpr int xp2 = AlongX ? (ForwardHalf ? 2 : 1) : 0;
    constexpr int ym2 = AlongX ? 0 : (ForwardHalf ? -1 : -2);
    constexpr int xm2 = AlongX ? (ForwardHalf ? -1 : -2) : 0;
    value = __hfma2(__float2half2_rn(-1.0f / 24.0f),
                    fp16_pair_delta(ptr, base, y, x, yp2, xp2, ym2, xm2),
                    value);
  }
  if constexpr (TIDE_STENCIL >= 6) {
    // Replace the fourth-order coefficients with their sixth-order values.
    half2 const d1 = fp16_pair_delta(ptr, base, y, x, yp1, xp1, ym1, xm1);
    constexpr int yp2 = AlongX ? 0 : (ForwardHalf ? 2 : 1);
    constexpr int xp2 = AlongX ? (ForwardHalf ? 2 : 1) : 0;
    constexpr int ym2 = AlongX ? 0 : (ForwardHalf ? -1 : -2);
    constexpr int xm2 = AlongX ? (ForwardHalf ? -1 : -2) : 0;
    half2 const d2 = fp16_pair_delta(ptr, base, y, x, yp2, xp2, ym2, xm2);
    constexpr int yp3 = AlongX ? 0 : (ForwardHalf ? 3 : 2);
    constexpr int xp3 = AlongX ? (ForwardHalf ? 3 : 2) : 0;
    constexpr int ym3 = AlongX ? 0 : (ForwardHalf ? -2 : -3);
    constexpr int xm3 = AlongX ? (ForwardHalf ? -2 : -3) : 0;
    value = __hmul2(__float2half2_rn(75.0f / 64.0f), d1);
    value = __hfma2(__float2half2_rn(-25.0f / 384.0f), d2, value);
    value = __hfma2(
        __float2half2_rn(3.0f / 640.0f),
        fp16_pair_delta(ptr, base, y, x, yp3, xp3, ym3, xm3), value);
  }
  if constexpr (TIDE_STENCIL >= 8) {
    half2 const d1 = fp16_pair_delta(ptr, base, y, x, yp1, xp1, ym1, xm1);
    constexpr int yp2 = AlongX ? 0 : (ForwardHalf ? 2 : 1);
    constexpr int xp2 = AlongX ? (ForwardHalf ? 2 : 1) : 0;
    constexpr int ym2 = AlongX ? 0 : (ForwardHalf ? -1 : -2);
    constexpr int xm2 = AlongX ? (ForwardHalf ? -1 : -2) : 0;
    constexpr int yp3 = AlongX ? 0 : (ForwardHalf ? 3 : 2);
    constexpr int xp3 = AlongX ? (ForwardHalf ? 3 : 2) : 0;
    constexpr int ym3 = AlongX ? 0 : (ForwardHalf ? -2 : -3);
    constexpr int xm3 = AlongX ? (ForwardHalf ? -2 : -3) : 0;
    constexpr int yp4 = AlongX ? 0 : (ForwardHalf ? 4 : 3);
    constexpr int xp4 = AlongX ? (ForwardHalf ? 4 : 3) : 0;
    constexpr int ym4 = AlongX ? 0 : (ForwardHalf ? -3 : -4);
    constexpr int xm4 = AlongX ? (ForwardHalf ? -3 : -4) : 0;
    value = __hmul2(__float2half2_rn(1225.0f / 1024.0f), d1);
    value = __hfma2(
        __float2half2_rn(-245.0f / 3072.0f),
        fp16_pair_delta(ptr, base, y, x, yp2, xp2, ym2, xm2), value);
    value = __hfma2(
        __float2half2_rn(49.0f / 5120.0f),
        fp16_pair_delta(ptr, base, y, x, yp3, xp3, ym3, xm3), value);
    value = __hfma2(
        __float2half2_rn(-5.0f / 7168.0f),
        fp16_pair_delta(ptr, base, y, x, yp4, xp4, ym4, xm4), value);
  }
  if constexpr (TIDE_STENCIL == 2) {
    // value already contains the second-order difference.
  }
  return __hmul2(value, __float2half2_rn(reciprocal_d));
}

static inline dim3 make_fp16_half2_grid(TMForwardLaunchConfig const &cfg,
                                        int64_t const nx_h) {
  int64_t const active_x = nx_h - 2 * kFdPad + 1;
  int64_t const pairs = (active_x + 1) / 2;
  return dim3(to_dim_u32((pairs + cfg.dimBlock.x - 1) / cfg.dimBlock.x),
              cfg.dimGrid.y, cfg.dimGrid.z);
}

static inline bool fp16_half2_enabled() {
  char const *value = std::getenv("TIDE_TM_FP16_HALF2");
  return value == nullptr || value[0] == '\0' || value[0] != '0';
}

static inline bool fp16_half2_arithmetic_enabled() {
  return read_env_flag("TIDE_TM_FP16_HALF2_ARITH");
}

__global__ __launch_bounds__(256) void forward_kernel_h_fp16_io(
    float const *__restrict const cq, half const *__restrict const ey,
    half *__restrict const hx, half *__restrict const hz,
    float *__restrict const m_ey_x, float *__restrict const m_ey_z,
    float const *__restrict const ayh, float const *__restrict const axh,
    float const *__restrict const byh, float const *__restrict const bxh,
    float const *__restrict const kyh, float const *__restrict const kxh) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int const fd_pad = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;
  if (shot_idx >= n_shots || y < fd_pad || x < fd_pad ||
      y >= ny - fd_pad + 1 || x >= nx - fd_pad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  int64_t const base = shot_idx * shot_numel;
  float const cq_val = cq[cq_batched ? i : j];
  HalfGlobalAccessor const ey_acc{ey, nx};
  int64_t const pml_y1h = pml_y1 > pml_y0 ? pml_y1 - 1 : pml_y0;
  int64_t const pml_x1h = pml_x1 > pml_x0 ? pml_x1 - 1 : pml_x0;

  if (y < ny - fd_pad) {
    float dey_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
        ey_acc, base, (int)y, (int)x, (int)nx, (float)rdy);
    if (y < pml_y0 || y >= pml_y1h) {
      float const memory = byh[y] * m_ey_z[i] + ayh[y] * dey_dz;
      m_ey_z[i] = memory;
      dey_dz = dey_dz / kyh[y] + memory;
    }
    float const value = __half2float(hx[i]) - cq_val * dey_dz;
    hx[i] = __float2half_rn(value);
  }

  if (x < nx - fd_pad) {
    float dey_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
        ey_acc, base, (int)y, (int)x, (int)nx, (float)rdx);
    if (x < pml_x0 || x >= pml_x1h) {
      float const memory = bxh[x] * m_ey_x[i] + axh[x] * dey_dx;
      m_ey_x[i] = memory;
      dey_dx = dey_dx / kxh[x] + memory;
    }
    float const value = __half2float(hz[i]) + cq_val * dey_dx;
    hz[i] = __float2half_rn(value);
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_fp16_io(
    float const *__restrict const ca, float const *__restrict const cb,
    half const *__restrict const hx, half const *__restrict const hz,
    half *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int const fd_pad = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;
  if (shot_idx >= n_shots || y < fd_pad || x < fd_pad ||
      y >= ny - fd_pad + 1 || x >= nx - fd_pad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  int64_t const base = shot_idx * shot_numel;
  HalfGlobalAccessor const hx_acc{hx, nx};
  HalfGlobalAccessor const hz_acc{hz, nx};
  float dhz_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
      hz_acc, base, (int)y, (int)x, (int)nx, (float)rdx);
  float dhx_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
      hx_acc, base, (int)y, (int)x, (int)nx, (float)rdy);

  if (x < pml_x0 || x >= pml_x1) {
    float const memory = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
    m_hz_x[i] = memory;
    dhz_dx = dhz_dx / kx[x] + memory;
  }
  if (y < pml_y0 || y >= pml_y1) {
    float const memory = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
    m_hx_z[i] = memory;
    dhx_dz = dhx_dz / ky[y] + memory;
  }

  float const ca_val = ca[ca_batched ? i : j];
  float const cb_val = cb[cb_batched ? i : j];
  float const value =
      ca_val * __half2float(ey[i]) + cb_val * (dhz_dx - dhx_dz);
  ey[i] = __float2half_rn(value);
}

// SeisCL-style packed path: two adjacent half cells are loaded/stored as a
// half2 whenever alignment permits, while the stencil and material update use
// float lanes.  Odd row pitches are supported without changing the public
// tensor layout.
__global__ __launch_bounds__(256) void forward_kernel_h_fp16_half2(
    float const *__restrict const cq, half const *__restrict const ey,
    half *__restrict const hx, half *__restrict const hz,
    float *__restrict const m_ey_x, float *__restrict const m_ey_z,
    float const *__restrict const ayh, float const *__restrict const axh,
    float const *__restrict const byh, float const *__restrict const bxh,
    float const *__restrict const kyh, float const *__restrict const kxh,
    bool const aggressive_arithmetic) {
  int64_t const pair_x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const x0 = (int64_t)kFdPad + 2 * pair_x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  if (shot_idx >= n_shots || y < kFdPad || y >= ny - kFdPad + 1 ||
      x0 >= nx - kFdPad + 1) {
    return;
  }

  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const base = shot_idx * shot_numel;
  int64_t const i0 = base + y * nx + x0;
  HalfGlobalAccessor const ey_acc{ey, nx};
  int64_t const pml_y1h = pml_y1 > pml_y0 ? pml_y1 - 1 : pml_y0;
  int64_t const pml_x1h = pml_x1 > pml_x0 ? pml_x1 - 1 : pml_x0;
  float2 hx_old = two ? __half22float2(load_half2_safe(hx, i0))
                      : make_float2(__half2float(hx[i0]), 0.0f);
  float2 hz_old = two ? __half22float2(load_half2_safe(hz, i0))
                      : make_float2(__half2float(hz[i0]), 0.0f);
  float2 hx_new = hx_old;
  float2 hz_new = hz_old;

  if (aggressive_arithmetic && two && y < ny - kFdPad &&
      x0 + 1 < nx - kFdPad && y >= pml_y0 && y < pml_y1h &&
      x0 >= pml_x0 && x0 + 1 < pml_x1h) {
    int64_t const j0 = y * nx + x0;
    half2 const cq2 = __halves2half2(
        __float2half_rn(cq[cq_batched ? i0 : j0]),
        __float2half_rn(cq[cq_batched ? i0 + 1 : j0 + 1]));
    half2 const dz =
        fp16_pair_diff<false, true>(ey, base, (int)y, (int)x0, (float)rdy);
    half2 const dx =
        fp16_pair_diff<true, true>(ey, base, (int)y, (int)x0, (float)rdx);
    store_half2_safe(hx, i0,
                     __hsub2(load_half2_safe(hx, i0), __hmul2(cq2, dz)));
    store_half2_safe(hz, i0,
                     __hadd2(load_half2_safe(hz, i0), __hmul2(cq2, dx)));
    return;
  }

#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int const x = (int)x0 + lane;
    int64_t const j = y * nx + x;
    int64_t const i = base + j;
    float const cq_val = cq[cq_batched ? i : j];
    if (y < ny - kFdPad) {
      float dey_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
          ey_acc, base, (int)y, x, (int)nx, (float)rdy);
      if (y < pml_y0 || y >= pml_y1h) {
        float const memory = byh[y] * m_ey_z[i] + ayh[y] * dey_dz;
        m_ey_z[i] = memory;
        dey_dz = dey_dz / kyh[y] + memory;
      }
      (&hx_new.x)[lane] = (&hx_old.x)[lane] - cq_val * dey_dz;
    }
    if (x < nx - kFdPad) {
      float dey_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
          ey_acc, base, (int)y, x, (int)nx, (float)rdx);
      if (x < pml_x0 || x >= pml_x1h) {
        float const memory = bxh[x] * m_ey_x[i] + axh[x] * dey_dx;
        m_ey_x[i] = memory;
        dey_dx = dey_dx / kxh[x] + memory;
      }
      (&hz_new.x)[lane] = (&hz_old.x)[lane] + cq_val * dey_dx;
    }
  }
  if (two) {
    store_half2_safe(hx, i0, __float22half2_rn(hx_new));
    store_half2_safe(hz, i0, __float22half2_rn(hz_new));
  } else {
    hx[i0] = __float2half_rn(hx_new.x);
    hz[i0] = __float2half_rn(hz_new.x);
  }
}

__global__ __launch_bounds__(256) void forward_kernel_e_fp16_half2(
    float const *__restrict const ca, float const *__restrict const cb,
    half const *__restrict const hx, half const *__restrict const hz,
    half *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx, bool const aggressive_arithmetic) {
  int64_t const pair_x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const x0 = (int64_t)kFdPad + 2 * pair_x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  if (shot_idx >= n_shots || y < kFdPad || y >= ny - kFdPad + 1 ||
      x0 >= nx - kFdPad + 1) {
    return;
  }
  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const base = shot_idx * shot_numel;
  int64_t const i0 = base + y * nx + x0;
  HalfGlobalAccessor const hx_acc{hx, nx};
  HalfGlobalAccessor const hz_acc{hz, nx};
  float2 ey_old = two ? __half22float2(load_half2_safe(ey, i0))
                      : make_float2(__half2float(ey[i0]), 0.0f);
  float2 ey_new = ey_old;
  if (aggressive_arithmetic && two && y >= pml_y0 && y < pml_y1 &&
      x0 >= pml_x0 && x0 + 1 < pml_x1) {
    int64_t const j0 = y * nx + x0;
    half2 const ca2 = __halves2half2(
        __float2half_rn(ca[ca_batched ? i0 : j0]),
        __float2half_rn(ca[ca_batched ? i0 + 1 : j0 + 1]));
    half2 const cb2 = __halves2half2(
        __float2half_rn(cb[cb_batched ? i0 : j0]),
        __float2half_rn(cb[cb_batched ? i0 + 1 : j0 + 1]));
    half2 const dhz_dx =
        fp16_pair_diff<true, false>(hz, base, (int)y, (int)x0, (float)rdx);
    half2 const dhx_dz =
        fp16_pair_diff<false, false>(hx, base, (int)y, (int)x0, (float)rdy);
    half2 const updated =
        __hfma2(cb2, __hsub2(dhz_dx, dhx_dz),
                __hmul2(ca2, load_half2_safe(ey, i0)));
    store_half2_safe(ey, i0, updated);
    return;
  }
#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int const x = (int)x0 + lane;
    int64_t const j = y * nx + x;
    int64_t const i = base + j;
    float dhz_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
        hz_acc, base, (int)y, x, (int)nx, (float)rdx);
    float dhx_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
        hx_acc, base, (int)y, x, (int)nx, (float)rdy);
    if (x < pml_x0 || x >= pml_x1) {
      float const memory = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
      m_hz_x[i] = memory;
      dhz_dx = dhz_dx / kx[x] + memory;
    }
    if (y < pml_y0 || y >= pml_y1) {
      float const memory = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
      m_hx_z[i] = memory;
      dhx_dz = dhx_dz / ky[y] + memory;
    }
    float const ca_val = ca[ca_batched ? i : j];
    float const cb_val = cb[cb_batched ? i : j];
    (&ey_new.x)[lane] =
        ca_val * (&ey_old.x)[lane] + cb_val * (dhz_dx - dhx_dz);
  }
  if (two) {
    store_half2_safe(ey, i0, __float22half2_rn(ey_new));
  } else {
    ey[i0] = __float2half_rn(ey_new.x);
  }
}

__global__ void add_sources_ey_fp16_io(
    half *__restrict const ey, float const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (src >= 0) {
      int64_t const i = shot_idx * shot_numel + src;
      ey[i] = __float2half_rn(__half2float(ey[i]) + f[k]);
    }
  }
}

__global__ void record_receivers_ey_fp16_io(
    float *__restrict const r, half const *__restrict const ey,
    int64_t const *__restrict const receivers_i) {
  int64_t const receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (rec >= 0) {
      r[k] = __half2float(ey[shot_idx * shot_numel + rec]);
    }
  }
}

template <typename StoreT>
__global__ __launch_bounds__(256) void forward_kernel_e_with_storage_fp16_io(
    float const *__restrict const ca, float const *__restrict const cb,
    half const *__restrict const hx, half const *__restrict const hz,
    half *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, StoreT *__restrict const ey_store,
    StoreT *__restrict const curl_h_store, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx, bool const store_ey,
    bool const store_curl) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int const fd_pad = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;
  if (shot_idx >= n_shots || y < fd_pad || x < fd_pad ||
      y >= ny - fd_pad + 1 || x >= nx - fd_pad + 1) {
    return;
  }

  int64_t const j = y * nx + x;
  int64_t const i = shot_idx * shot_numel + j;
  int64_t const base = shot_idx * shot_numel;
  HalfGlobalAccessor const hx_acc{hx, nx};
  HalfGlobalAccessor const hz_acc{hz, nx};
  float dhz_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
      hz_acc, base, (int)y, (int)x, (int)nx, (float)rdx);
  float dhx_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
      hx_acc, base, (int)y, (int)x, (int)nx, (float)rdy);

  if (x < pml_x0 || x >= pml_x1) {
    float const memory = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
    m_hz_x[i] = memory;
    dhz_dx = dhz_dx / kx[x] + memory;
  }
  if (y < pml_y0 || y >= pml_y1) {
    float const memory = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
    m_hx_z[i] = memory;
    dhx_dz = dhx_dz / ky[y] + memory;
  }

  float const curl_h = dhz_dx - dhx_dz;
  float const ey_old = __half2float(ey[i]);
  if (store_ey) {
    ey_store[i] = ::tide::encode_snapshot<StoreT, float>(ey_old);
  }
  if (store_curl) {
    curl_h_store[i] = ::tide::encode_snapshot<StoreT, float>(curl_h);
  }
  float const ca_val = ca[ca_batched ? i : j];
  float const cb_val = cb[cb_batched ? i : j];
  ey[i] = __float2half_rn(ca_val * ey_old + cb_val * curl_h);
}

template <typename StoreT>
__global__ __launch_bounds__(256)
void forward_kernel_e_with_storage_fp16_half2(
    float const *__restrict const ca, float const *__restrict const cb,
    half const *__restrict const hx, half const *__restrict const hz,
    half *__restrict const ey, float *__restrict const m_hx_z,
    float *__restrict const m_hz_x, StoreT *__restrict const ey_store,
    StoreT *__restrict const curl_h_store, float const *__restrict const ay,
    float const *__restrict const ax, float const *__restrict const by,
    float const *__restrict const bx, float const *__restrict const ky,
    float const *__restrict const kx, bool const store_ey,
    bool const store_curl, bool const aggressive_arithmetic) {
  int64_t const pair_x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const x0 = (int64_t)kFdPad + 2 * pair_x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  if (shot_idx >= n_shots || y < kFdPad || y >= ny - kFdPad + 1 ||
      x0 >= nx - kFdPad + 1) {
    return;
  }
  bool const two = x0 + 1 < nx - kFdPad + 1;
  int64_t const base = shot_idx * shot_numel;
  int64_t const i0 = base + y * nx + x0;
  HalfGlobalAccessor const hx_acc{hx, nx};
  HalfGlobalAccessor const hz_acc{hz, nx};
  float2 ey_old = two ? __half22float2(load_half2_safe(ey, i0))
                      : make_float2(__half2float(ey[i0]), 0.0f);
  float2 ey_new = ey_old;
  if (aggressive_arithmetic && two && y >= pml_y0 && y < pml_y1 &&
      x0 >= pml_x0 && x0 + 1 < pml_x1) {
    int64_t const j0 = y * nx + x0;
    half2 const ey2 = load_half2_safe(ey, i0);
    half2 const ca2 = __halves2half2(
        __float2half_rn(ca[ca_batched ? i0 : j0]),
        __float2half_rn(ca[ca_batched ? i0 + 1 : j0 + 1]));
    half2 const cb2 = __halves2half2(
        __float2half_rn(cb[cb_batched ? i0 : j0]),
        __float2half_rn(cb[cb_batched ? i0 + 1 : j0 + 1]));
    half2 const curl2 = __hsub2(
        fp16_pair_diff<true, false>(hz, base, (int)y, (int)x0, (float)rdx),
        fp16_pair_diff<false, false>(hx, base, (int)y, (int)x0, (float)rdy));
    float2 const ey_values = __half22float2(ey2);
    float2 const curl_values = __half22float2(curl2);
    if (store_ey) {
      ey_store[i0] = ::tide::encode_snapshot<StoreT, float>(ey_values.x);
      ey_store[i0 + 1] = ::tide::encode_snapshot<StoreT, float>(ey_values.y);
    }
    if (store_curl) {
      curl_h_store[i0] =
          ::tide::encode_snapshot<StoreT, float>(curl_values.x);
      curl_h_store[i0 + 1] =
          ::tide::encode_snapshot<StoreT, float>(curl_values.y);
    }
    store_half2_safe(ey, i0, __hfma2(cb2, curl2, __hmul2(ca2, ey2)));
    return;
  }
#pragma unroll
  for (int lane = 0; lane < 2; ++lane) {
    if (lane == 1 && !two) break;
    int const x = (int)x0 + lane;
    int64_t const j = y * nx + x;
    int64_t const i = base + j;
    float dhz_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
        hz_acc, base, (int)y, x, (int)nx, (float)rdx);
    float dhx_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
        hx_acc, base, (int)y, x, (int)nx, (float)rdy);
    if (x < pml_x0 || x >= pml_x1) {
      float const memory = bx[x] * m_hz_x[i] + ax[x] * dhz_dx;
      m_hz_x[i] = memory;
      dhz_dx = dhz_dx / kx[x] + memory;
    }
    if (y < pml_y0 || y >= pml_y1) {
      float const memory = by[y] * m_hx_z[i] + ay[y] * dhx_dz;
      m_hx_z[i] = memory;
      dhx_dz = dhx_dz / ky[y] + memory;
    }
    float const curl_h = dhz_dx - dhx_dz;
    if (store_ey) {
      ey_store[i] =
          ::tide::encode_snapshot<StoreT, float>((&ey_old.x)[lane]);
    }
    if (store_curl) {
      curl_h_store[i] = ::tide::encode_snapshot<StoreT, float>(curl_h);
    }
    float const ca_val = ca[ca_batched ? i : j];
    float const cb_val = cb[cb_batched ? i : j];
    (&ey_new.x)[lane] = ca_val * (&ey_old.x)[lane] + cb_val * curl_h;
  }
  if (two) {
    store_half2_safe(ey, i0, __float22half2_rn(ey_new));
  } else {
    ey[i0] = __float2half_rn(ey_new.x);
  }
}

__global__ void add_adjoint_sources_ey_fp16_io(
    half *__restrict const lambda_ey, float const *__restrict const grad_r,
    int64_t const *__restrict const receivers_i) {
  int64_t const receiver_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (receiver_idx < n_receivers_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_receivers_per_shot + receiver_idx;
    int64_t const rec = receivers_i[k];
    if (rec >= 0) {
      int64_t const i = shot_idx * shot_numel + rec;
      lambda_ey[i] =
          __float2half_rn(__half2float(lambda_ey[i]) + grad_r[k]);
    }
  }
}

__global__ void record_adjoint_at_sources_fp16_io(
    float *__restrict const grad_f, half const *__restrict const lambda_ey,
    int64_t const *__restrict const sources_i) {
  int64_t const source_idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const shot_idx = (int64_t)blockIdx.y;
  if (source_idx < n_sources_per_shot && shot_idx < n_shots) {
    int64_t const k = shot_idx * n_sources_per_shot + source_idx;
    int64_t const src = sources_i[k];
    if (src >= 0) {
      grad_f[k] = __half2float(lambda_ey[shot_idx * shot_numel + src]);
    }
  }
}

template <typename StoreT, bool GradCa, bool GradCb>
__global__ void coeff_grad_kernel_fp16_io(
    half const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    float *__restrict const grad_ca_shot,
    float *__restrict const grad_cb_shot, int64_t const step_ratio_val) {
  int64_t const x =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const y =
      (int64_t)blockIdx.y * (int64_t)blockDim.y + (int64_t)threadIdx.y;
  int64_t const shot_idx = (int64_t)blockIdx.z;
  int const fd_pad = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;
  if (shot_idx >= n_shots || y < fd_pad || x < fd_pad ||
      y >= ny - fd_pad + 1 || x >= nx - fd_pad + 1) {
    return;
  }
  int64_t const i = shot_idx * shot_numel + y * nx + x;
  float const lambda_val = __half2float(lambda_ey[i]);
  float const step_scale = (float)step_ratio_val;
  if constexpr (GradCa) {
    float const ey_n = ::tide::decode_snapshot<StoreT, float>(ey_store[i]);
    grad_ca_shot[i] += lambda_val * ey_n * step_scale;
  }
  if constexpr (GradCb) {
    float const curl_h_n =
        ::tide::decode_snapshot<StoreT, float>(curl_h_store[i]);
    grad_cb_shot[i] += lambda_val * curl_h_n * step_scale;
  }
}

template <typename StoreT>
static inline void launch_coeff_grad_kernel_fp16_io(
    TMForwardLaunchConfig const &launch_cfg, cudaStream_t const stream,
    half const *__restrict const lambda_ey,
    StoreT const *__restrict const ey_store,
    StoreT const *__restrict const curl_h_store,
    float *__restrict const grad_ca_shot,
    float *__restrict const grad_cb_shot, bool const grad_ca,
    bool const grad_cb, int64_t const step_ratio_val) {
  if (grad_ca && grad_cb) {
    coeff_grad_kernel_fp16_io<StoreT, true, true>
        <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, ey_store, curl_h_store, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  } else if (grad_ca) {
    coeff_grad_kernel_fp16_io<StoreT, true, false>
        <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, ey_store, nullptr, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  } else if (grad_cb) {
    coeff_grad_kernel_fp16_io<StoreT, false, true>
        <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream>>>(
            lambda_ey, nullptr, curl_h_store, grad_ca_shot, grad_cb_shot,
            step_ratio_val);
  }
}

} // namespace

extern "C" void FUNC(forward_fp16_io)(
    float const *const ca, float const *const cb, float const *const cq,
    float const *const f, half *const ey, half *const hx, half *const hz,
    float *const m_ey_x, float *const m_ey_z, float *const m_hx_z,
    float *const m_hz_x, float const *const debye_a,
    float const *const debye_b, float const *const debye_cp,
    float *const polarization, float *const ey_prev, float *const r,
    int64_t const n_poles_h, float const *const ay, float const *const by,
    float const *const ayh, float const *const byh, float const *const ax,
    float const *const bx, float const *const axh, float const *const bxh,
    float const *const ky, float const *const kyh, float const *const kx,
    float const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, float const rdy_h, float const rdx_h,
    float const dt_h, int64_t const nt, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    bool const has_dispersion, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h,
    int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle) {
  (void)debye_a;
  (void)debye_b;
  (void)debye_cp;
  (void)polarization;
  (void)ey_prev;
  (void)n_poles_h;
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;
  if (has_dispersion) {
    std::fprintf(stderr, "fp16_io does not support dispersion.\n");
    std::abort();
  }

  cudaSetDevice(device);
  cudaStream_t const stream_compute = resolve_cuda_stream(compute_stream_handle);
  int64_t const shot_numel_h = ny_h * nx_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);
  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h);
  bool const use_half2 = fp16_half2_enabled();
  bool const aggressive_arithmetic = fp16_half2_arithmetic_enabled();
  dim3 const half2_grid = make_fp16_half2_grid(launch_cfg, nx_h);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    if (use_half2) {
      forward_kernel_h_fp16_half2<<<half2_grid, launch_cfg.dimBlock, 0,
                                    stream_compute>>>(
          cq, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh, kxh,
          aggressive_arithmetic);
      forward_kernel_e_fp16_half2<<<half2_grid, launch_cfg.dimBlock, 0,
                                    stream_compute>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx,
          aggressive_arithmetic);
    } else {
      forward_kernel_h_fp16_io<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                                 stream_compute>>>(
          cq, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh, kxh);
      forward_kernel_e_fp16_io<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                                 stream_compute>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx);
    }
    if (n_sources_per_shot_h > 0) {
      add_sources_ey_fp16_io<<<launch_cfg.dimGridSources,
                               launch_cfg.dimBlockSources, 0,
                               stream_compute>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey_fp16_io<<<launch_cfg.dimGridReceivers,
                                    launch_cfg.dimBlockReceivers, 0,
                                    stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  }
  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(forward_with_storage_fp16_io)(
    float const *const ca, float const *const cb, float const *const cq,
    float const *const f, half *const ey, half *const hx, half *const hz,
    float *const m_ey_x, float *const m_ey_z, float *const m_hx_z,
    float *const m_hz_x, float *const r, void *const ey_store_1,
    void *const ey_store_3, char const *const *const ey_filenames,
    void *const curl_store_1, void *const curl_store_3,
    char const *const *const curl_filenames, float const *const ay,
    float const *const by, float const *const ayh, float const *const byh,
    float const *const ax, float const *const bx, float const *const axh,
    float const *const bxh, float const *const ky, float const *const kyh,
    float const *const kx, float const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    float const rdy_h, float const rdx_h, float const dt_h, int64_t const nt,
    int64_t const n_shots_h, int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    int64_t const storage_mode_h, int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp_h, bool const ca_requires_grad,
    bool const cb_requires_grad, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h,
    int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle, void *const storage_stream_handle) {
  (void)dt_h;
  (void)n_threads;
  (void)ey_store_3;
  (void)ey_filenames;
  (void)curl_store_3;
  (void)curl_filenames;
  (void)storage_stream_handle;
  if (storage_mode_h != STORAGE_DEVICE) {
    std::fprintf(stderr,
                 "fp16_io gradient mode currently requires device storage.\n");
    std::abort();
  }
  cudaSetDevice(device);
  cudaStream_t const stream_compute = resolve_cuda_stream(compute_stream_handle);
  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);
  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h);
  bool const use_half2 = fp16_half2_enabled();
  bool const aggressive_arithmetic = fp16_half2_arithmetic_enabled();
  dim3 const half2_grid = make_fp16_half2_grid(launch_cfg, nx_h);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    bool const store_step = (t % step_ratio_h) == 0;
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    if (use_half2) {
      forward_kernel_h_fp16_half2<<<half2_grid, launch_cfg.dimBlock, 0,
                                    stream_compute>>>(
          cq, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh, kxh,
          aggressive_arithmetic);
    } else {
      forward_kernel_h_fp16_io<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                                 stream_compute>>>(
          cq, ey, hx, hz, m_ey_x, m_ey_z, ayh, axh, byh, bxh, kyh, kxh);
    }
    if (store_ey || store_curl) {
      size_t const offset = (size_t)(t / step_ratio_h) * bytes_per_step;
      if (storage_format_h == STORAGE_FORMAT_BF16) {
        if (use_half2) {
          forward_kernel_e_with_storage_fp16_half2<__nv_bfloat16>
              <<<half2_grid, launch_cfg.dimBlock, 0, stream_compute>>>(
                  ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
                  store_ey ? (__nv_bfloat16 *)((uint8_t *)ey_store_1 + offset)
                           : nullptr,
                  store_curl
                      ? (__nv_bfloat16 *)((uint8_t *)curl_store_1 + offset)
                      : nullptr,
                  ay, ax, by, bx, ky, kx, store_ey, store_curl,
                  aggressive_arithmetic);
        } else {
          forward_kernel_e_with_storage_fp16_io<__nv_bfloat16>
              <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
                ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
                store_ey ? (__nv_bfloat16 *)((uint8_t *)ey_store_1 + offset)
                         : nullptr,
                store_curl
                    ? (__nv_bfloat16 *)((uint8_t *)curl_store_1 + offset)
                    : nullptr,
                ay, ax, by, bx, ky, kx, store_ey, store_curl);
        }
      } else {
        if (use_half2) {
          forward_kernel_e_with_storage_fp16_half2<float>
              <<<half2_grid, launch_cfg.dimBlock, 0, stream_compute>>>(
                  ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
                  store_ey ? (float *)((uint8_t *)ey_store_1 + offset) : nullptr,
                  store_curl ? (float *)((uint8_t *)curl_store_1 + offset)
                             : nullptr,
                  ay, ax, by, bx, ky, kx, store_ey, store_curl,
                  aggressive_arithmetic);
        } else {
          forward_kernel_e_with_storage_fp16_io<float>
              <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
                ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
                store_ey ? (float *)((uint8_t *)ey_store_1 + offset) : nullptr,
                store_curl ? (float *)((uint8_t *)curl_store_1 + offset)
                           : nullptr,
                ay, ax, by, bx, ky, kx, store_ey, store_curl);
        }
      }
    } else {
      if (use_half2) {
        forward_kernel_e_fp16_half2<<<half2_grid, launch_cfg.dimBlock, 0,
                                      stream_compute>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx,
            aggressive_arithmetic);
      } else {
        forward_kernel_e_fp16_io<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                                   stream_compute>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ax, by, bx, ky, kx);
      }
    }
    if (n_sources_per_shot_h > 0) {
      add_sources_ey_fp16_io<<<launch_cfg.dimGridSources,
                               launch_cfg.dimBlockSources, 0,
                               stream_compute>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey_fp16_io<<<launch_cfg.dimGridReceivers,
                                    launch_cfg.dimBlockReceivers, 0,
                                    stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  }
  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(backward_fp16_io)(
    float const *const ca, float const *const cb, float const *const cq,
    float const *const grad_r, half *const lambda_ey, half *const lambda_hx,
    half *const lambda_hz, float *const m_lambda_ey_x,
    float *const m_lambda_ey_z, float *const m_lambda_hx_z,
    float *const m_lambda_hz_x, void *const ey_store_1,
    void *const ey_store_3, char const *const *const ey_filenames,
    void *const curl_store_1, void *const curl_store_3,
    char const *const *const curl_filenames, float *const grad_f,
    float *const grad_ca, float *const grad_cb, float *const grad_ca_shot,
    float *const grad_cb_shot, float const *const ay, float const *const by,
    float const *const ayh, float const *const byh, float const *const ax,
    float const *const bx, float const *const axh, float const *const bxh,
    float const *const ky, float const *const kyh, float const *const kx,
    float const *const kxh, int64_t const *const sources_i,
    int64_t const *const receivers_i, float const rdy_h, float const rdx_h,
    float const dt_h, int64_t const nt, int64_t const n_shots_h,
    int64_t const ny_h, int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const step_ratio_h,
    int64_t const storage_mode_h, int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp_h, bool const ca_requires_grad,
    bool const cb_requires_grad, bool const ca_batched_h,
    bool const cb_batched_h, bool const cq_batched_h, int64_t const start_t,
    int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h,
    int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle, void *const storage_stream_handle) {
  (void)dt_h;
  (void)n_threads;
  (void)ey_store_3;
  (void)ey_filenames;
  (void)curl_store_3;
  (void)curl_filenames;
  (void)storage_stream_handle;
  if (storage_mode_h != STORAGE_DEVICE) {
    std::fprintf(stderr,
                 "fp16_io gradient mode currently requires device storage.\n");
    std::abort();
  }
  cudaSetDevice(device);
  cudaStream_t const stream_compute = resolve_cuda_stream(compute_stream_handle);
  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);
  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h);
  bool const use_half2 = fp16_half2_enabled();
  bool const aggressive_arithmetic = fp16_half2_arithmetic_enabled();
  dim3 const half2_grid = make_fp16_half2_grid(launch_cfg, nx_h);

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    if (use_half2) {
      forward_kernel_h_fp16_half2<<<half2_grid, launch_cfg.dimBlock, 0,
                                    stream_compute>>>(
          cq, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x,
          m_lambda_ey_z, ayh, axh, byh, bxh, kyh, kxh,
          aggressive_arithmetic);
      forward_kernel_e_fp16_half2<<<half2_grid, launch_cfg.dimBlock, 0,
                                    stream_compute>>>(
          ca, cb, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
          m_lambda_hz_x, ay, ax, by, bx, ky, kx, aggressive_arithmetic);
    } else {
      forward_kernel_h_fp16_io<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                                 stream_compute>>>(
          cq, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x,
          m_lambda_ey_z, ayh, axh, byh, bxh, kyh, kxh);
      forward_kernel_e_fp16_io<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                                 stream_compute>>>(
          ca, cb, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
          m_lambda_hz_x, ay, ax, by, bx, ky, kx);
    }
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey_fp16_io<<<launch_cfg.dimGridReceivers,
                                       launch_cfg.dimBlockReceivers, 0,
                                       stream_compute>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }
    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources_fp16_io<<<launch_cfg.dimGridSources,
                                          launch_cfg.dimBlockSources, 0,
                                          stream_compute>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, lambda_ey,
          sources_i);
    }
    if ((t % step_ratio_h) == 0) {
      size_t const offset = (size_t)(t / step_ratio_h) * bytes_per_step;
      if (storage_format_h == STORAGE_FORMAT_BF16) {
        launch_coeff_grad_kernel_fp16_io<__nv_bfloat16>(
            launch_cfg, stream_compute, lambda_ey,
            ca_requires_grad
                ? (__nv_bfloat16 const *)((uint8_t *)ey_store_1 + offset)
                : nullptr,
            cb_requires_grad
                ? (__nv_bfloat16 const *)((uint8_t *)curl_store_1 + offset)
                : nullptr,
            grad_ca_shot, grad_cb_shot, ca_requires_grad, cb_requires_grad,
            step_ratio_h);
      } else {
        launch_coeff_grad_kernel_fp16_io<float>(
            launch_cfg, stream_compute, lambda_ey,
            ca_requires_grad
                ? (float const *)((uint8_t *)ey_store_1 + offset)
                : nullptr,
            cb_requires_grad
                ? (float const *)((uint8_t *)curl_store_1 + offset)
                : nullptr,
            grad_ca_shot, grad_cb_shot, ca_requires_grad, cb_requires_grad,
            step_ratio_h);
      }
    }
  }

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

#endif

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

extern "C" void FUNC(born_tangent_forward_with_storage)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb, TIDE_DTYPE const *const df,
    TIDE_DTYPE *const dey, TIDE_DTYPE *const dhx, TIDE_DTYPE *const dhz,
    TIDE_DTYPE *const dm_ey_x, TIDE_DTYPE *const dm_ey_z,
    TIDE_DTYPE *const dm_hx_z, TIDE_DTYPE *const dm_hz_x,
    TIDE_DTYPE *const r, void const *const ey_store,
    void const *const curl_store, void *const dey_store,
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
    int64_t const step_ratio_h, int64_t const storage_format_h,
    bool const ca_batched_h, bool const cb_batched_h,
    bool const cq_batched_h, int64_t const start_t, int64_t const pml_y0_h,
    int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device,
    void *const compute_stream_handle) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)n_threads;
  if (step_ratio_h != 1) {
    std::fprintf(
        stderr,
        "born_tangent_forward_with_storage requires step_ratio=1.\n");
    std::abort();
  }
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
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

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                       stream_compute>>>(
        cq, dey, dhx, dhz, dm_ey_x, dm_ey_z, ay, ayh, ax, axh, by, byh, bx,
        bxh, ky, kyh, kx, kxh);
    int64_t const store_idx = t;
    if (storage_bf16_h) {
      born_tangent_kernel_e_from_snapshots<__nv_bfloat16>
          <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
              ca, cb, dca, dcb, dhx, dhz, dey, dm_hx_z, dm_hz_x,
              (const __nv_bfloat16 *)ey_store + store_idx * store_size,
              (const __nv_bfloat16 *)curl_store + store_idx * store_size,
              (__nv_bfloat16 *)dey_store + store_idx * store_size,
              (__nv_bfloat16 *)dcurl_store + store_idx * store_size, ay, ayh,
              ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    } else {
      born_tangent_kernel_e_from_snapshots<TIDE_DTYPE>
          <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
              ca, cb, dca, dcb, dhx, dhz, dey, dm_hx_z, dm_hz_x,
              (const TIDE_DTYPE *)ey_store + store_idx * store_size,
              (const TIDE_DTYPE *)curl_store + store_idx * store_size,
              (TIDE_DTYPE *)dey_store + store_idx * store_size,
              (TIDE_DTYPE *)dcurl_store + store_idx * store_size, ay, ayh, ax,
              axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    }
    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources, 0,
                       stream_compute>>>(
          dey, df + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }
    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<launch_cfg.dimGridReceivers,
                            launch_cfg.dimBlockReceivers, 0,
                            stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, dey, receivers_i);
    }
  }
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

extern "C" void FUNC(born_backward_bggrad)(
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb, TIDE_DTYPE const *const f0,
    TIDE_DTYPE const *const df, TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE const *const grad_background_r,
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
    add_inplace_and_zero<<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0,
                           stream_compute>>>(eta_ey, eta_source_old);

    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<launch_cfg.dimGridReceivers,
                               launch_cfg.dimBlockReceivers, 0,
                               stream_compute>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
      add_adjoint_sources_ey<<<launch_cfg.dimGridReceivers,
                               launch_cfg.dimBlockReceivers, 0,
                               stream_compute>>>(
          eta_ey,
          grad_background_r +
              t * n_shots_h * n_receivers_per_shot_h,
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
                cb, cq, dca, dcb, lambda_ey,
                (__nv_bfloat16 const *)dey_store_t,
                (__nv_bfloat16 const *)dcurl_store_t, grad_ca_shot,
                grad_cb_shot, eta_source_old, work_eta_x, work_eta_z, ay, ayh,
                ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh, step_ratio_h);
      } else {
        born_background_prepare_direct_kernel<TIDE_DTYPE>
            <<<launch_cfg.dimGrid, launch_cfg.dimBlock, 0, stream_compute>>>(
                cb, cq, dca, dcb, lambda_ey,
                (TIDE_DTYPE const *)dey_store_t,
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
      if (storage_bf16_h) {
        __nv_bfloat16 const *const ey_store_t =
            (__nv_bfloat16 const *)ey_store_1 + store_idx * store_size;
        __nv_bfloat16 const *const curl_store_t =
            (__nv_bfloat16 const *)curl_store_1 + store_idx * store_size;
        launch_coeff_grad_kernel<__nv_bfloat16>(
            launch_cfg, stream_compute, eta_ey, ey_store_t, curl_store_t,
            grad_ca_shot, grad_cb_shot, ca_requires_grad, cb_requires_grad,
            step_ratio_h);
      } else {
        TIDE_DTYPE const *const ey_store_t =
            (TIDE_DTYPE const *)ey_store_1 + store_idx * store_size;
        TIDE_DTYPE const *const curl_store_t =
            (TIDE_DTYPE const *)curl_store_1 + store_idx * store_size;
        launch_coeff_grad_kernel<TIDE_DTYPE>(
            launch_cfg, stream_compute, eta_ey, ey_store_t, curl_store_t,
            grad_ca_shot, grad_cb_shot, ca_requires_grad, cb_requires_grad,
            step_ratio_h);
      }
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
