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
#include <cooperative_groups.h>
#include <cuda_pipeline_primitives.h>
#include "staggered_grid.h"

namespace FUNC(Inst) {
using tide_field_t = TIDE_DTYPE;
using tide_scalar_t = tide_field_t;
constexpr bool kFieldIsHalf = false;
constexpr int kFdPad = ::tide::StencilTraits<TIDE_STENCIL>::FD_PAD;
namespace cg = cooperative_groups;
#ifndef TIDE_TM_BLOCK_X
#define TIDE_TM_BLOCK_X 32
#endif
#ifndef TIDE_TM_BLOCK_Y
#define TIDE_TM_BLOCK_Y 8
#endif

namespace {
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
__device__ __forceinline__ void tm_async_copy_global_to_shared(
    T *__restrict__ dst_shared, T const *__restrict__ src_global) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __pipeline_memcpy_async((void *)dst_shared, (void const *)src_global,
                          sizeof(T));
#else
  *dst_shared = *src_global;
#endif
}

__device__ __forceinline__ void tm_async_copy_commit_and_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __pipeline_commit();
  __pipeline_wait_prior(0);
#endif
}

template <typename T>
__host__ __device__ __forceinline__ T tide_min(T a, T b) {
  return a < b ? a : b;
}

enum : int {
  kTMEbisuPathNone = 0,
  kTMEbisuPathNoPml = 1,
  kTMEbisuPathFace = 2,
  kTMEbisuPathFull = 3,
};

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

struct TMEbisuWorkspace {
  int64_t device = -1;
  size_t bytes = 0;
  TIDE_DTYPE *ey = nullptr;
  TIDE_DTYPE *hx = nullptr;
  TIDE_DTYPE *hz = nullptr;
  TIDE_DTYPE *m_ey_x = nullptr;
  TIDE_DTYPE *m_ey_z = nullptr;
  TIDE_DTYPE *m_hx_z = nullptr;
  TIDE_DTYPE *m_hz_x = nullptr;
  TIDE_DTYPE *ey_alt = nullptr;
  TIDE_DTYPE *hx_alt = nullptr;
  TIDE_DTYPE *hz_alt = nullptr;
  TIDE_DTYPE *m_ey_x_alt = nullptr;
  TIDE_DTYPE *m_ey_z_alt = nullptr;
  TIDE_DTYPE *m_hx_z_alt = nullptr;
  TIDE_DTYPE *m_hz_x_alt = nullptr;
  int2 *no_pml_tiles = nullptr;
  int2 *y_pml_tiles = nullptr;
  int2 *x_pml_tiles = nullptr;
  int2 *full_tiles = nullptr;
  int *no_pml_tile_lookup = nullptr;
  int *face_tile_lookup = nullptr;
  int *full_tile_lookup = nullptr;
  int *single_source_path = nullptr;
  int *single_source_block = nullptr;
  int *single_source_li = nullptr;
  int *single_receiver_path = nullptr;
  int *single_receiver_block = nullptr;
  int *single_receiver_li = nullptr;
  size_t no_pml_tile_capacity = 0;
  size_t y_pml_tile_capacity = 0;
  size_t x_pml_tile_capacity = 0;
  size_t full_tile_capacity = 0;
  size_t no_pml_tile_lookup_capacity = 0;
  size_t face_tile_lookup_capacity = 0;
  size_t full_tile_lookup_capacity = 0;
  size_t shot_capacity = 0;
};

static inline void release_tm_ebisu_workspace(TMEbisuWorkspace &ws) {
  auto free_if_needed = [&](TIDE_DTYPE *ptr) {
    if (ptr != nullptr) {
      tide::cuda_check_or_abort(cudaFree(ptr), __FILE__, __LINE__);
    }
  };
  free_if_needed(ws.ey);
  free_if_needed(ws.hx);
  free_if_needed(ws.hz);
  free_if_needed(ws.m_ey_x);
  free_if_needed(ws.m_ey_z);
  free_if_needed(ws.m_hx_z);
  free_if_needed(ws.m_hz_x);
  free_if_needed(ws.ey_alt);
  free_if_needed(ws.hx_alt);
  free_if_needed(ws.hz_alt);
  free_if_needed(ws.m_ey_x_alt);
  free_if_needed(ws.m_ey_z_alt);
  free_if_needed(ws.m_hx_z_alt);
  free_if_needed(ws.m_hz_x_alt);
  if (ws.no_pml_tiles != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.no_pml_tiles), __FILE__, __LINE__);
  }
  if (ws.y_pml_tiles != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.y_pml_tiles), __FILE__, __LINE__);
  }
  if (ws.x_pml_tiles != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.x_pml_tiles), __FILE__, __LINE__);
  }
  if (ws.full_tiles != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.full_tiles), __FILE__, __LINE__);
  }
  if (ws.no_pml_tile_lookup != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.no_pml_tile_lookup), __FILE__,
                              __LINE__);
  }
  if (ws.face_tile_lookup != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.face_tile_lookup), __FILE__,
                              __LINE__);
  }
  if (ws.full_tile_lookup != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.full_tile_lookup), __FILE__,
                              __LINE__);
  }
  if (ws.single_source_path != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.single_source_path), __FILE__,
                              __LINE__);
  }
  if (ws.single_source_block != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.single_source_block), __FILE__,
                              __LINE__);
  }
  if (ws.single_source_li != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.single_source_li), __FILE__,
                              __LINE__);
  }
  if (ws.single_receiver_path != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.single_receiver_path), __FILE__,
                              __LINE__);
  }
  if (ws.single_receiver_block != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.single_receiver_block), __FILE__,
                              __LINE__);
  }
  if (ws.single_receiver_li != nullptr) {
    tide::cuda_check_or_abort(cudaFree(ws.single_receiver_li), __FILE__,
                              __LINE__);
  }
  ws = {};
}

static inline void ensure_tm_ebisu_workspace(TMEbisuWorkspace &ws,
                                             int64_t const device,
                                             size_t const bytes) {
  if (ws.device == device && ws.bytes == bytes && ws.ey != nullptr &&
      ws.hx != nullptr && ws.hz != nullptr && ws.m_ey_x != nullptr &&
      ws.m_ey_z != nullptr && ws.m_hx_z != nullptr && ws.m_hz_x != nullptr &&
      ws.ey_alt != nullptr && ws.hx_alt != nullptr && ws.hz_alt != nullptr &&
      ws.m_ey_x_alt != nullptr && ws.m_ey_z_alt != nullptr &&
      ws.m_hx_z_alt != nullptr && ws.m_hz_x_alt != nullptr) {
    return;
  }
  release_tm_ebisu_workspace(ws);
  ws.device = device;
  ws.bytes = bytes;
  auto alloc = [&](TIDE_DTYPE **ptr) {
    tide::cuda_check_or_abort(cudaMalloc((void **)ptr, bytes), __FILE__,
                              __LINE__);
  };
  alloc(&ws.ey);
  alloc(&ws.hx);
  alloc(&ws.hz);
  alloc(&ws.m_ey_x);
  alloc(&ws.m_ey_z);
  alloc(&ws.m_hx_z);
  alloc(&ws.m_hz_x);
  alloc(&ws.ey_alt);
  alloc(&ws.hx_alt);
  alloc(&ws.hz_alt);
  alloc(&ws.m_ey_x_alt);
  alloc(&ws.m_ey_z_alt);
  alloc(&ws.m_hx_z_alt);
  alloc(&ws.m_hz_x_alt);
}

static inline void ensure_tm_ebisu_tile_capacity(TMEbisuWorkspace &ws,
                                                 size_t const no_pml_count,
                                                 size_t const y_pml_count,
                                                 size_t const x_pml_count,
                                                 size_t const full_count) {
  auto ensure = [&](int2 **ptr, size_t &capacity, size_t const count) {
    if (count == 0) {
      return;
    }
    if (*ptr != nullptr && capacity >= count) {
      return;
    }
    if (*ptr != nullptr) {
      tide::cuda_check_or_abort(cudaFree(*ptr), __FILE__, __LINE__);
    }
    tide::cuda_check_or_abort(cudaMalloc((void **)ptr, count * sizeof(int2)),
                              __FILE__, __LINE__);
    capacity = count;
  };
  ensure(&ws.no_pml_tiles, ws.no_pml_tile_capacity, no_pml_count);
  ensure(&ws.y_pml_tiles, ws.y_pml_tile_capacity, y_pml_count);
  ensure(&ws.x_pml_tiles, ws.x_pml_tile_capacity, x_pml_count);
  ensure(&ws.full_tiles, ws.full_tile_capacity, full_count);
}

static inline void ensure_tm_ebisu_lookup_capacity(int **ptr, size_t &capacity,
                                                   size_t const count) {
  if (count == 0) {
    return;
  }
  if (*ptr != nullptr && capacity >= count) {
    return;
  }
  if (*ptr != nullptr) {
    tide::cuda_check_or_abort(cudaFree(*ptr), __FILE__, __LINE__);
  }
  tide::cuda_check_or_abort(cudaMalloc((void **)ptr, count * sizeof(int)),
                            __FILE__, __LINE__);
  capacity = count;
}

static inline void ensure_tm_ebisu_shot_capacity(TMEbisuWorkspace &ws,
                                                 size_t const count) {
  if (count == 0) {
    return;
  }
  if (ws.single_source_path != nullptr && ws.single_source_block != nullptr &&
      ws.single_source_li != nullptr && ws.single_receiver_path != nullptr &&
      ws.single_receiver_block != nullptr &&
      ws.single_receiver_li != nullptr &&
      ws.shot_capacity >= count) {
    return;
  }
  auto ensure = [&](int **ptr) {
    if (*ptr != nullptr) {
      tide::cuda_check_or_abort(cudaFree(*ptr), __FILE__, __LINE__);
    }
    tide::cuda_check_or_abort(cudaMalloc((void **)ptr, count * sizeof(int)),
                              __FILE__, __LINE__);
  };
  ensure(&ws.single_source_path);
  ensure(&ws.single_source_block);
  ensure(&ws.single_source_li);
  ensure(&ws.single_receiver_path);
  ensure(&ws.single_receiver_block);
  ensure(&ws.single_receiver_li);
  ws.shot_capacity = count;
}

static inline bool tm_ebisu_tile_fully_inside_no_pml_host(
    int64_t const out_y0, int64_t const out_x0, int64_t const core_y,
    int64_t const core_x, int64_t const halo, int64_t const pml_y0_h,
    int64_t const pml_x0_h, int64_t const pml_y1_h, int64_t const pml_x1_h) {
  int64_t const load_y0 = out_y0 - halo;
  int64_t const load_x0 = out_x0 - halo;
  int64_t const tile_y = core_y + 2 * halo;
  int64_t const tile_x = core_x + 2 * halo;
  return load_y0 >= pml_y0_h && load_x0 >= pml_x0_h &&
         load_y0 + tile_y <= pml_y1_h && load_x0 + tile_x <= pml_x1_h;
}

static inline bool tm_ebisu_tile_fully_inside_x_no_pml_host(
    int64_t const out_x0, int64_t const core_x, int64_t const halo,
    int64_t const pml_x0_h, int64_t const pml_x1_h) {
  int64_t const load_x0 = out_x0 - halo;
  int64_t const tile_x = core_x + 2 * halo;
  return load_x0 >= pml_x0_h && load_x0 + tile_x <= pml_x1_h;
}

static inline bool tm_ebisu_tile_fully_inside_y_no_pml_host(
    int64_t const out_y0, int64_t const core_y, int64_t const halo,
    int64_t const pml_y0_h, int64_t const pml_y1_h) {
  int64_t const load_y0 = out_y0 - halo;
  int64_t const tile_y = core_y + 2 * halo;
  return load_y0 >= pml_y0_h && load_y0 + tile_y <= pml_y1_h;
}

static inline void build_tm_ebisu_tile_lists(
    std::vector<int2> &no_pml_tiles, std::vector<int2> &y_pml_tiles,
    std::vector<int2> &x_pml_tiles, std::vector<int2> &full_tiles,
    int64_t const domain_y, int64_t const domain_x, int64_t const core_y,
    int64_t const core_x, int64_t const halo, int64_t const pml_y0_h,
    int64_t const pml_x0_h, int64_t const pml_y1_h, int64_t const pml_x1_h,
    bool const split_face_pml) {
  no_pml_tiles.clear();
  y_pml_tiles.clear();
  x_pml_tiles.clear();
  full_tiles.clear();
  int64_t const tiles_y = (domain_y + core_y - 1) / core_y;
  int64_t const tiles_x = (domain_x + core_x - 1) / core_x;
  size_t const reserve_tiles = static_cast<size_t>(tiles_y * tiles_x);
  no_pml_tiles.reserve(reserve_tiles);
  full_tiles.reserve(reserve_tiles);
  for (int64_t tile_y_idx = 0; tile_y_idx < tiles_y; ++tile_y_idx) {
    int64_t const out_y0 = kFdPad + tile_y_idx * core_y;
    for (int64_t tile_x_idx = 0; tile_x_idx < tiles_x; ++tile_x_idx) {
      int64_t const out_x0 = kFdPad + tile_x_idx * core_x;
      int2 const tile = make_int2(static_cast<int>(tile_x_idx),
                                  static_cast<int>(tile_y_idx));
      if (tm_ebisu_tile_fully_inside_no_pml_host(
              out_y0, out_x0, core_y, core_x, halo, pml_y0_h, pml_x0_h,
              pml_y1_h, pml_x1_h)) {
        no_pml_tiles.push_back(tile);
      } else if (
          split_face_pml &&
          tm_ebisu_tile_fully_inside_x_no_pml_host(out_x0, core_x, halo,
                                                   pml_x0_h, pml_x1_h) &&
          !tm_ebisu_tile_fully_inside_y_no_pml_host(out_y0, core_y, halo,
                                                    pml_y0_h, pml_y1_h)) {
        y_pml_tiles.push_back(tile);
      } else if (
          split_face_pml &&
          tm_ebisu_tile_fully_inside_y_no_pml_host(out_y0, core_y, halo,
                                                   pml_y0_h, pml_y1_h) &&
          !tm_ebisu_tile_fully_inside_x_no_pml_host(out_x0, core_x, halo,
                                                    pml_x0_h, pml_x1_h)) {
        x_pml_tiles.push_back(tile);
      } else {
        full_tiles.push_back(tile);
      }
    }
  }
}

static inline void build_tm_ebisu_decoupled_tile_lists(
    std::vector<int2> &deep_no_pml_tiles,
    std::vector<int2> &band_no_pml_tiles, std::vector<int2> &face_tiles,
    std::vector<int2> &full_tiles, size_t &y_face_count,
    size_t &x_face_count, int64_t const domain_y, int64_t const domain_x,
    int64_t const core_y, int64_t const core_x, int64_t const deep_halo,
    int64_t const face_halo, int64_t const pml_y0_h, int64_t const pml_x0_h,
    int64_t const pml_y1_h, int64_t const pml_x1_h,
    bool const split_face_pml) {
  deep_no_pml_tiles.clear();
  band_no_pml_tiles.clear();
  face_tiles.clear();
  full_tiles.clear();
  y_face_count = 0;
  x_face_count = 0;
  int64_t const tiles_y = (domain_y + core_y - 1) / core_y;
  int64_t const tiles_x = (domain_x + core_x - 1) / core_x;
  size_t const reserve_tiles = static_cast<size_t>(tiles_y * tiles_x);
  deep_no_pml_tiles.reserve(reserve_tiles);
  band_no_pml_tiles.reserve(reserve_tiles);
  face_tiles.reserve(reserve_tiles);
  full_tiles.reserve(reserve_tiles);
  for (int64_t tile_y_idx = 0; tile_y_idx < tiles_y; ++tile_y_idx) {
    int64_t const out_y0 = kFdPad + tile_y_idx * core_y;
    for (int64_t tile_x_idx = 0; tile_x_idx < tiles_x; ++tile_x_idx) {
      int64_t const out_x0 = kFdPad + tile_x_idx * core_x;
      int2 const tile = make_int2(static_cast<int>(tile_x_idx),
                                  static_cast<int>(tile_y_idx));
      bool const deep_no_pml = tm_ebisu_tile_fully_inside_no_pml_host(
          out_y0, out_x0, core_y, core_x, deep_halo, pml_y0_h, pml_x0_h,
          pml_y1_h, pml_x1_h);
      bool const face_no_pml = tm_ebisu_tile_fully_inside_no_pml_host(
          out_y0, out_x0, core_y, core_x, face_halo, pml_y0_h, pml_x0_h,
          pml_y1_h, pml_x1_h);
      if (deep_no_pml) {
        deep_no_pml_tiles.push_back(tile);
      } else if (face_no_pml) {
        band_no_pml_tiles.push_back(tile);
      } else if (
          split_face_pml &&
          tm_ebisu_tile_fully_inside_x_no_pml_host(out_x0, core_x, face_halo,
                                                   pml_x0_h, pml_x1_h) &&
          !tm_ebisu_tile_fully_inside_y_no_pml_host(out_y0, core_y, face_halo,
                                                    pml_y0_h, pml_y1_h)) {
        face_tiles.push_back(tile);
        ++y_face_count;
      } else if (
          split_face_pml &&
          tm_ebisu_tile_fully_inside_y_no_pml_host(out_y0, core_y, face_halo,
                                                   pml_y0_h, pml_y1_h) &&
          !tm_ebisu_tile_fully_inside_x_no_pml_host(out_x0, core_x, face_halo,
                                                    pml_x0_h, pml_x1_h)) {
        face_tiles.push_back(
            make_int2(-static_cast<int>(tile_x_idx) - 1,
                      static_cast<int>(tile_y_idx)));
        ++x_face_count;
      } else {
        full_tiles.push_back(tile);
      }
    }
  }
}

static inline void build_tm_ebisu_face_tile_lookup(
    std::vector<int> &lookup, std::vector<int2> const &face_tiles,
    int64_t const domain_y, int64_t const domain_x, int64_t const core_y,
    int64_t const core_x) {
  int64_t const tiles_y = (domain_y + core_y - 1) / core_y;
  int64_t const tiles_x = (domain_x + core_x - 1) / core_x;
  size_t const count = static_cast<size_t>(tiles_y * tiles_x);
  lookup.assign(count, -1);
  for (size_t block_idx = 0; block_idx < face_tiles.size(); ++block_idx) {
    int2 const tile = face_tiles[block_idx];
    int const tile_x_idx = tile.x < 0 ? -tile.x - 1 : tile.x;
    int const tile_y_idx = tile.y;
    size_t const dense_idx =
        static_cast<size_t>(tile_y_idx) * static_cast<size_t>(tiles_x) +
        static_cast<size_t>(tile_x_idx);
    if (dense_idx < lookup.size()) {
      lookup[dense_idx] = static_cast<int>(block_idx);
    }
  }
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

struct TMFusedRuntimeConfig {
  bool enabled = false;
  int64_t steps = 0;
  int64_t threads = 256;
  int64_t ilp = 4;
  int64_t blocks_per_sm = 1;
};

struct TMEbisuRuntimeConfig {
  bool enabled = false;
  bool face_pml = false;
  bool face_async_copy = true;
  int64_t steps = 0;
  int64_t face_steps = 0;
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

static inline TMFusedRuntimeConfig read_tm_fused_runtime_config() {
  TMFusedRuntimeConfig cfg{};
  cfg.steps = read_env_i64("TIDE_TM_FUSED_STEPS", 0);
  cfg.enabled = cfg.steps > 0;
  cfg.threads = read_env_i64("TIDE_TM_FUSED_THREADS", 256);
  cfg.ilp = read_env_i64("TIDE_TM_FUSED_ILP", 4);
  cfg.blocks_per_sm = read_env_i64("TIDE_TM_FUSED_BLOCKS_PER_SM", 1);
  if (cfg.threads <= 0) {
    cfg.threads = 256;
  }
  if (cfg.ilp != 1 && cfg.ilp != 2 && cfg.ilp != 4) {
    cfg.ilp = 4;
  }
  if (cfg.blocks_per_sm <= 0) {
    cfg.blocks_per_sm = 1;
  }
  return cfg;
}

static inline TMEbisuRuntimeConfig read_tm_ebisu_runtime_config() {
  TMEbisuRuntimeConfig cfg{};
  cfg.steps = read_env_i64("TIDE_TM_EBISU_STEPS", 0);
  cfg.enabled = cfg.steps > 0;
  cfg.face_pml = read_env_flag("TIDE_TM_EBISU_FACE_PML") ||
                 read_env_flag("TIDE_TM_EBISU_TOP_PML_Y");
  cfg.face_async_copy = !read_env_flag("TIDE_TM_EBISU_FACE_ASYNC_COPY_OFF");
  cfg.face_steps = read_env_i64("TIDE_TM_EBISU_FACE_STEPS", cfg.steps);
  cfg.tile_x = read_env_i64("TIDE_TM_EBISU_TILE_X", 64);
  cfg.tile_y = read_env_i64("TIDE_TM_EBISU_TILE_Y", 16);
  cfg.ilp = read_env_i64("TIDE_TM_EBISU_ILP", 1);
  if (cfg.face_steps <= 0) {
    cfg.face_steps = cfg.steps;
  }
  if (cfg.face_steps > cfg.steps) {
    cfg.face_steps = cfg.steps;
  }
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

__device__ __forceinline__ bool ebisu_tile_fully_inside_no_pml(
    int const out_y0, int const out_x0, int const core_y, int const core_x,
    int const halo) {
  int const load_y0 = out_y0 - halo;
  int const load_x0 = out_x0 - halo;
  int const tile_y = core_y + 2 * halo;
  int const tile_x = core_x + 2 * halo;
  return load_y0 >= static_cast<int>(pml_y0) &&
         load_x0 >= static_cast<int>(pml_x0) &&
         load_y0 + tile_y <= static_cast<int>(pml_y1) &&
         load_x0 + tile_x <= static_cast<int>(pml_x1);
}

template <int ILP>
__global__ __launch_bounds__(256, 1) void forward_kernel_ebisu_tb_no_pml(
    TIDE_DTYPE const *__restrict__ ca, TIDE_DTYPE const *__restrict__ cb,
    TIDE_DTYPE const *__restrict__ cq, TIDE_DTYPE *__restrict__ ey,
    TIDE_DTYPE *__restrict__ hx, TIDE_DTYPE *__restrict__ hz,
    int64_t const core_y, int64_t const core_x, int64_t const halo,
    int64_t const k_steps) {
  extern __shared__ unsigned char smem_raw[];
  TIDE_DTYPE *ey_a = reinterpret_cast<TIDE_DTYPE *>(smem_raw);
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
  TIDE_DTYPE *ey_b = ey_a + tile_elems;
  TIDE_DTYPE *hx_s = ey_b + tile_elems;
  TIDE_DTYPE *hz_s = hx_s + tile_elems;
  TIDE_DTYPE const rdy_t = static_cast<TIDE_DTYPE>(rdy);
  TIDE_DTYPE const rdx_t = static_cast<TIDE_DTYPE>(rdx);

  int64_t const shot_idx = (int64_t)blockIdx.z;
  if (shot_idx >= n_shots) {
    return;
  }

  int64_t const shot_offset = shot_idx * shot_numel;
  int const out_y0 = kFdPad + (int)blockIdx.y * core_y_i;
  int const out_x0 = kFdPad + (int)blockIdx.x * core_x_i;
  int const load_y0 = out_y0 - halo_i;
  int const load_x0 = out_x0 - halo_i;
  int const tid = (int)threadIdx.x;
  int const block_threads = (int)blockDim.x;

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
      int64_t const gj = (int64_t)gy * nx_i + gx;
      int64_t const gi = shot_offset + gj;
      ey_a[li] = ey[gi];
      hx_s[li] = hx[gi];
      hz_s[li] = hz[gi];
    }
  }
  __syncthreads();

  TIDE_DTYPE *ey_src = ey_a;
  TIDE_DTYPE *ey_dst = ey_b;
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
            int64_t const gj = (int64_t)gy * nx_i + gx;
            int64_t const gi = shot_offset + gj;
            TIDE_DTYPE const cq_val = cq_batched ? cq[gi] : cq[gj];
            hx_s[li] -= cq_val * (ey_src[li + tile_x] - ey_src[li]) * rdy_t;
            hz_s[li] += cq_val * (ey_src[li + 1] - ey_src[li]) * rdx_t;
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
            int64_t const gj = (int64_t)gy * nx_i + gx;
            int64_t const gi = shot_offset + gj;
            TIDE_DTYPE const ca_val = ca_batched ? ca[gi] : ca[gj];
            TIDE_DTYPE const cb_val = cb_batched ? cb[gi] : cb[gj];
            TIDE_DTYPE const curl_h =
                (hz_s[li] - hz_s[li - 1]) * rdx_t -
                (hx_s[li] - hx_s[li - tile_x]) * rdy_t;
            ey_dst[li] = ca_val * ey_src[li] + cb_val * curl_h;
          }
        }
      }
    }
    __syncthreads();

    TIDE_DTYPE *tmp = ey_src;
    ey_src = ey_dst;
    ey_dst = tmp;
  }

  int const interior_y = ny_i - 2 * kFdPad;
  int const interior_x = nx_i - 2 * kFdPad;
  int const valid_out_y =
      tide_max<int>(0, tide_min<int>(core_y_i, interior_y - (int)blockIdx.y * core_y_i));
  int const valid_out_x =
      tide_max<int>(0, tide_min<int>(core_x_i, interior_x - (int)blockIdx.x * core_x_i));
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
      int64_t const gj = (int64_t)gy * nx_i + gx;
      int64_t const gi = shot_offset + gj;
      int const li = core_offset + cy * tile_x + cx;
      ey[gi] = ey_src[li];
      hx[gi] = hx_s[li];
      hz[gi] = hz_s[li];
    }
  }
}

template <int ILP>
__global__ __launch_bounds__(256, 1) void forward_kernel_ebisu_tb_no_pml_oop(
    TIDE_DTYPE const *__restrict__ ca, TIDE_DTYPE const *__restrict__ cb,
    TIDE_DTYPE const *__restrict__ cq, TIDE_DTYPE const *__restrict__ f,
    TIDE_DTYPE const *__restrict__ ey_in, TIDE_DTYPE const *__restrict__ hx_in,
    TIDE_DTYPE const *__restrict__ hz_in, TIDE_DTYPE *__restrict__ ey_out,
    TIDE_DTYPE *__restrict__ hx_out, TIDE_DTYPE *__restrict__ hz_out,
    int const *__restrict__ single_source_path,
    int const *__restrict__ single_source_block,
    int const *__restrict__ single_source_li,
    int const *__restrict__ single_receiver_path,
    int const *__restrict__ single_receiver_block,
    int const *__restrict__ single_receiver_li,
    int64_t const *__restrict__ sources_i,
    int64_t const *__restrict__ receivers_i, TIDE_DTYPE *__restrict__ r,
    int64_t const core_y, int64_t const core_x, int64_t const halo,
    int64_t const k_steps, int2 const *__restrict__ tile_coords) {
  extern __shared__ unsigned char smem_raw[];
  TIDE_DTYPE *ey_a = reinterpret_cast<TIDE_DTYPE *>(smem_raw);
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
  TIDE_DTYPE *ey_b = ey_a + tile_elems;
  TIDE_DTYPE *hx_s = ey_b + tile_elems;
  TIDE_DTYPE *hz_s = hx_s + tile_elems;
  TIDE_DTYPE const rdy_t = static_cast<TIDE_DTYPE>(rdy);
  TIDE_DTYPE const rdx_t = static_cast<TIDE_DTYPE>(rdx);

  int64_t const shot_idx = static_cast<int64_t>(blockIdx.z);
  if (shot_idx >= n_shots) {
    return;
  }

  int tile_y_idx = static_cast<int>(blockIdx.y);
  int tile_x_idx = static_cast<int>(blockIdx.x);
  if (tile_coords != nullptr) {
    int2 const tile = tile_coords[blockIdx.x];
    tile_x_idx = tile.x;
    tile_y_idx = tile.y;
  }
  int const out_y0 = kFdPad + tile_y_idx * core_y_i;
  int const out_x0 = kFdPad + tile_x_idx * core_x_i;
  if (tile_coords == nullptr &&
      !ebisu_tile_fully_inside_no_pml(out_y0, out_x0, core_y_i, core_x_i,
                                      halo_i)) {
    return;
  }

  int64_t const shot_offset = shot_idx * shot_numel;
  int const load_y0 = out_y0 - halo_i;
  int const load_x0 = out_x0 - halo_i;
  int const tid = static_cast<int>(threadIdx.x);
  int const block_threads = static_cast<int>(blockDim.x);
  int const compact_block_idx = tile_coords != nullptr ? static_cast<int>(blockIdx.x)
                                                       : -1;

  for (int base = tid; base < tile_elems; base += block_threads * ILP) {
#pragma unroll
    for (int lane = 0; lane < ILP; ++lane) {
      int const li = base + lane * block_threads;
      if (li >= tile_elems) {
        continue;
      }
      int const ly = li / tile_x;
      int const lx = li - ly * tile_x;
      int const gy = load_y0 + ly;
      int const gx = load_x0 + lx;
      int64_t const gj = static_cast<int64_t>(gy) * nx_i + gx;
      int64_t const gi = shot_offset + gj;
      ey_a[li] = ey_in[gi];
      hx_s[li] = hx_in[gi];
      hz_s[li] = hz_in[gi];
    }
  }
  __syncthreads();

  TIDE_DTYPE *ey_src = ey_a;
  TIDE_DTYPE *ey_dst = ey_b;
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
            TIDE_DTYPE const cq_val = cq_batched ? cq[gi] : cq[gj];
            hx_s[li] -= cq_val * (ey_src[li + tile_x] - ey_src[li]) * rdy_t;
            hz_s[li] += cq_val * (ey_src[li + 1] - ey_src[li]) * rdx_t;
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
            TIDE_DTYPE const ca_val = ca_batched ? ca[gi] : ca[gj];
            TIDE_DTYPE const cb_val = cb_batched ? cb[gi] : cb[gj];
            TIDE_DTYPE const curl_h =
                (hz_s[li] - hz_s[li - 1]) * rdx_t -
                (hx_s[li] - hx_s[li - tile_x]) * rdy_t;
            ey_dst[li] = ca_val * ey_src[li] + cb_val * curl_h;
          }
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      if (n_sources_per_shot == 1 && single_source_path != nullptr &&
          single_source_block != nullptr && single_source_li != nullptr &&
          f != nullptr) {
        if (single_source_path[shot_idx] == kTMEbisuPathNoPml &&
            single_source_block[shot_idx] == compact_block_idx) {
          int const src_li = single_source_li[shot_idx];
          if (src_li >= 0) {
            int const ly = src_li / tile_x;
            int const lx = src_li - ly * tile_x;
            if (ly > lo && ly < hi_y - 1 && lx > lo && lx < hi_x - 1) {
              TIDE_DTYPE const *const f_step =
                  f + static_cast<int64_t>(step) * n_shots * n_sources_per_shot;
              ey_dst[src_li] += f_step[shot_idx * n_sources_per_shot];
            }
          }
        }
      } else if (n_sources_per_shot > 0 && sources_i != nullptr && f != nullptr) {
        TIDE_DTYPE const *const f_step =
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
      if (n_receivers_per_shot == 1 && single_receiver_path != nullptr &&
          single_receiver_block != nullptr && single_receiver_li != nullptr &&
          r != nullptr) {
        if (single_receiver_path[shot_idx] == kTMEbisuPathNoPml &&
            single_receiver_block[shot_idx] == compact_block_idx) {
          int const rec_li = single_receiver_li[shot_idx];
          if (rec_li >= 0) {
            TIDE_DTYPE *const r_step =
                r + static_cast<int64_t>(step) * n_shots * n_receivers_per_shot;
            r_step[shot_idx * n_receivers_per_shot] = ey_dst[rec_li];
          }
        }
      } else if (n_receivers_per_shot > 0 && receivers_i != nullptr &&
                 r != nullptr) {
        TIDE_DTYPE *const r_step =
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

    TIDE_DTYPE *tmp = ey_src;
    ey_src = ey_dst;
    ey_dst = tmp;
  }

  int const interior_y = ny_i - 2 * kFdPad;
  int const interior_x = nx_i - 2 * kFdPad;
  int const valid_out_y = tide_max<int>(
      0, tide_min<int>(core_y_i, interior_y - static_cast<int>(blockIdx.y) * core_y_i));
  int const valid_out_x = tide_max<int>(
      0, tide_min<int>(core_x_i, interior_x - static_cast<int>(blockIdx.x) * core_x_i));
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
      ey_out[gi] = ey_src[li];
      hx_out[gi] = hx_s[li];
      hz_out[gi] = hz_s[li];
    }
  }
}

template <int ILP>
__global__ __launch_bounds__(256, 1) void forward_kernel_ebisu_tb_full(
    TIDE_DTYPE const *__restrict__ ca, TIDE_DTYPE const *__restrict__ cb,
    TIDE_DTYPE const *__restrict__ cq, TIDE_DTYPE const *__restrict__ f,
    TIDE_DTYPE *__restrict__ ey, TIDE_DTYPE *__restrict__ hx,
    TIDE_DTYPE *__restrict__ hz, TIDE_DTYPE *__restrict__ m_ey_x,
    TIDE_DTYPE *__restrict__ m_ey_z, TIDE_DTYPE *__restrict__ m_hx_z,
    TIDE_DTYPE *__restrict__ m_hz_x, TIDE_DTYPE const *__restrict__ ay,
    TIDE_DTYPE const *__restrict__ ayh, TIDE_DTYPE const *__restrict__ ax,
    TIDE_DTYPE const *__restrict__ axh, TIDE_DTYPE const *__restrict__ by,
    TIDE_DTYPE const *__restrict__ byh, TIDE_DTYPE const *__restrict__ bx,
    TIDE_DTYPE const *__restrict__ bxh, TIDE_DTYPE const *__restrict__ ky,
    TIDE_DTYPE const *__restrict__ kyh, TIDE_DTYPE const *__restrict__ kx,
    TIDE_DTYPE const *__restrict__ kxh,
    int64_t const *__restrict__ sources_i,
    int64_t const *__restrict__ receivers_i, TIDE_DTYPE *__restrict__ r,
    int64_t const core_y, int64_t const core_x, int64_t const halo,
    int64_t const k_steps) {
  extern __shared__ unsigned char smem_raw[];
  TIDE_DTYPE *ey_a = reinterpret_cast<TIDE_DTYPE *>(smem_raw);
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
  TIDE_DTYPE *ey_b = ey_a + tile_elems;
  TIDE_DTYPE *hx_s = ey_b + tile_elems;
  TIDE_DTYPE *hz_s = hx_s + tile_elems;
  TIDE_DTYPE *m_ey_x_s = hz_s + tile_elems;
  TIDE_DTYPE *m_ey_z_s = m_ey_x_s + tile_elems;
  TIDE_DTYPE *m_hx_z_s = m_ey_z_s + tile_elems;
  TIDE_DTYPE *m_hz_x_s = m_hx_z_s + tile_elems;
  TIDE_DTYPE const rdy_t = static_cast<TIDE_DTYPE>(rdy);
  TIDE_DTYPE const rdx_t = static_cast<TIDE_DTYPE>(rdx);
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
      ey_a[li] = ey[gi];
      hx_s[li] = hx[gi];
      hz_s[li] = hz[gi];
      m_ey_x_s[li] = m_ey_x[gi];
      m_ey_z_s[li] = m_ey_z[gi];
      m_hx_z_s[li] = m_hx_z[gi];
      m_hz_x_s[li] = m_hz_x[gi];
    }
  }
  __syncthreads();

  TIDE_DTYPE *ey_src = ey_a;
  TIDE_DTYPE *ey_dst = ey_b;
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
            TIDE_DTYPE const cq_val = cq_batched ? cq[gi] : cq[gj];

            if (gy < ny_i - kFdPad) {
              bool const in_pml_y =
                  gy < pml_y0 || gy >= tide_max<int64_t>(pml_y0, pml_y1 - 1);
              TIDE_DTYPE dey_dz = (ey_src[li + tile_x] - ey_src[li]) * rdy_t;
              if (in_pml_y) {
                TIDE_DTYPE const m_new =
                    __ldg(&byh[gy]) * m_ey_z_s[li] + __ldg(&ayh[gy]) * dey_dz;
                m_ey_z_s[li] = m_new;
                dey_dz = dey_dz / __ldg(&kyh[gy]) + m_new;
              }
              hx_s[li] -= cq_val * dey_dz;
            }

            if (gx < nx_i - kFdPad) {
              bool const in_pml_x =
                  gx < pml_x0 || gx >= tide_max<int64_t>(pml_x0, pml_x1 - 1);
              TIDE_DTYPE dey_dx = (ey_src[li + 1] - ey_src[li]) * rdx_t;
              if (in_pml_x) {
                TIDE_DTYPE const m_new =
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
            TIDE_DTYPE const ca_val = ca_batched ? ca[gi] : ca[gj];
            TIDE_DTYPE const cb_val = cb_batched ? cb[gi] : cb[gj];
            bool const in_pml_y = gy < pml_y0 || gy >= pml_y1;
            bool const in_pml_x = gx < pml_x0 || gx >= pml_x1;

            TIDE_DTYPE dhz_dx = (hz_s[li] - hz_s[li - 1]) * rdx_t;
            TIDE_DTYPE dhx_dz = (hx_s[li] - hx_s[li - tile_x]) * rdy_t;

            if (in_pml_x) {
              TIDE_DTYPE const m_new =
                  __ldg(&bx[gx]) * m_hz_x_s[li] + __ldg(&ax[gx]) * dhz_dx;
              m_hz_x_s[li] = m_new;
              dhz_dx = dhz_dx / __ldg(&kx[gx]) + m_new;
            }
            if (in_pml_y) {
              TIDE_DTYPE const m_new =
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
        TIDE_DTYPE const *const f_step =
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
        TIDE_DTYPE *const r_step =
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

    TIDE_DTYPE *tmp = ey_src;
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
      ey[gi] = ey_src[li];
      hx[gi] = hx_s[li];
      hz[gi] = hz_s[li];
      m_ey_x[gi] = m_ey_x_s[li];
      m_ey_z[gi] = m_ey_z_s[li];
      m_hx_z[gi] = m_hx_z_s[li];
      m_hz_x[gi] = m_hz_x_s[li];
    }
  }
}

template <int ILP>
__global__ __launch_bounds__(256, 1) void forward_kernel_ebisu_tb_top_pml_y_oop(
    TIDE_DTYPE const *__restrict__ ca, TIDE_DTYPE const *__restrict__ cb,
    TIDE_DTYPE const *__restrict__ cq, TIDE_DTYPE const *__restrict__ f,
    TIDE_DTYPE const *__restrict__ ey_in, TIDE_DTYPE const *__restrict__ hx_in,
    TIDE_DTYPE const *__restrict__ hz_in,
    TIDE_DTYPE const *__restrict__ m_ey_x_in,
    TIDE_DTYPE const *__restrict__ m_ey_z_in,
    TIDE_DTYPE const *__restrict__ m_hx_z_in,
    TIDE_DTYPE const *__restrict__ m_hz_x_in, TIDE_DTYPE *__restrict__ ey_out,
    TIDE_DTYPE *__restrict__ hx_out, TIDE_DTYPE *__restrict__ hz_out,
    TIDE_DTYPE *__restrict__ m_ey_x_out, TIDE_DTYPE *__restrict__ m_ey_z_out,
    TIDE_DTYPE *__restrict__ m_hx_z_out, TIDE_DTYPE *__restrict__ m_hz_x_out,
    TIDE_DTYPE const *__restrict__ ay, TIDE_DTYPE const *__restrict__ ayh,
    TIDE_DTYPE const *__restrict__ ax, TIDE_DTYPE const *__restrict__ axh,
    TIDE_DTYPE const *__restrict__ by, TIDE_DTYPE const *__restrict__ byh,
    TIDE_DTYPE const *__restrict__ bx, TIDE_DTYPE const *__restrict__ bxh,
    TIDE_DTYPE const *__restrict__ ky, TIDE_DTYPE const *__restrict__ kyh,
    TIDE_DTYPE const *__restrict__ kx, TIDE_DTYPE const *__restrict__ kxh,
    int const *__restrict__ single_source_path,
    int const *__restrict__ single_source_block,
    int const *__restrict__ single_source_li,
    int const *__restrict__ single_receiver_path,
    int const *__restrict__ single_receiver_block,
    int const *__restrict__ single_receiver_li,
    int64_t const *__restrict__ sources_i,
    int64_t const *__restrict__ receivers_i, TIDE_DTYPE *__restrict__ r,
    int64_t const core_y, int64_t const core_x, int64_t const halo,
    int64_t const k_steps, bool const use_async_copy,
    int2 const *__restrict__ tile_coords) {
  extern __shared__ unsigned char smem_raw[];
  TIDE_DTYPE *ey_a = reinterpret_cast<TIDE_DTYPE *>(smem_raw);
  int const core_y_i = static_cast<int>(core_y);
  int const core_x_i = static_cast<int>(core_x);
  int const halo_i = static_cast<int>(halo);
  int const k_steps_i = static_cast<int>(k_steps);
  int const ny_i = static_cast<int>(ny);
  int const nx_i = static_cast<int>(nx);
  int const tile_y = core_y_i + 2 * halo_i;
  int const tile_x = core_x_i + 2 * halo_i;
  int const tile_elems = tile_y * tile_x;
  TIDE_DTYPE *ey_b = ey_a + tile_elems;
  TIDE_DTYPE *hx_s = ey_b + tile_elems;
  TIDE_DTYPE *hz_s = hx_s + tile_elems;
  TIDE_DTYPE const rdy_t = static_cast<TIDE_DTYPE>(rdy);
  TIDE_DTYPE const rdx_t = static_cast<TIDE_DTYPE>(rdx);
  int const domain_y_begin = kFdPad;
  int const domain_x_begin = kFdPad;
  int const domain_y_end = ny_i - kFdPad + 1;
  int const domain_x_end = nx_i - kFdPad + 1;

  int64_t const shot_idx = static_cast<int64_t>(blockIdx.z);
  if (shot_idx >= n_shots || tile_coords == nullptr) {
    return;
  }

  int2 const tile = tile_coords[blockIdx.x];
  bool const x_face = tile.x < 0;
  int const tile_x_idx = x_face ? -tile.x - 1 : tile.x;
  int const tile_y_idx = tile.y;
  int const out_y0 = domain_y_begin + tile_y_idx * core_y_i;
  int const out_x0 = domain_x_begin + tile_x_idx * core_x_i;
  int const compact_block_idx = static_cast<int>(blockIdx.x);

  int64_t const shot_offset = shot_idx * shot_numel;
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
      if (use_async_copy) {
        tm_async_copy_global_to_shared(ey_a + li, ey_in + gi);
        tm_async_copy_global_to_shared(hx_s + li, hx_in + gi);
        tm_async_copy_global_to_shared(hz_s + li, hz_in + gi);
      } else {
        ey_a[li] = ey_in[gi];
        hx_s[li] = hx_in[gi];
        hz_s[li] = hz_in[gi];
      }
    }
  }
  if (use_async_copy) {
    tm_async_copy_commit_and_wait();
  }
  __syncthreads();

  TIDE_DTYPE *ey_src = ey_a;
  TIDE_DTYPE *ey_dst = ey_b;
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
            TIDE_DTYPE const cq_val = cq_batched ? cq[gi] : cq[gj];

            if (gy < ny_i - kFdPad) {
              TIDE_DTYPE dey_dz = (ey_src[li + tile_x] - ey_src[li]) * rdy_t;
              if (!x_face) {
                bool const in_pml_y =
                    gy < pml_y0 || gy >= tide_max<int64_t>(pml_y0, pml_y1 - 1);
                if (in_pml_y) {
                  TIDE_DTYPE const m_prev =
                      step == 0 ? m_ey_z_in[gi] : m_ey_z_out[gi];
                  TIDE_DTYPE const m_new =
                      __ldg(&byh[gy]) * m_prev + __ldg(&ayh[gy]) * dey_dz;
                  m_ey_z_out[gi] = m_new;
                  dey_dz = dey_dz / __ldg(&kyh[gy]) + m_new;
                }
              }
              hx_s[li] -= cq_val * dey_dz;
            }

            TIDE_DTYPE dey_dx = (ey_src[li + 1] - ey_src[li]) * rdx_t;
            if (x_face && gx < nx_i - kFdPad) {
              bool const in_pml_x =
                  gx < pml_x0 || gx >= tide_max<int64_t>(pml_x0, pml_x1 - 1);
              if (in_pml_x) {
                TIDE_DTYPE const m_prev =
                    step == 0 ? m_ey_x_in[gi] : m_ey_x_out[gi];
                TIDE_DTYPE const m_new =
                    __ldg(&bxh[gx]) * m_prev + __ldg(&axh[gx]) * dey_dx;
                m_ey_x_out[gi] = m_new;
                dey_dx = dey_dx / __ldg(&kxh[gx]) + m_new;
              }
            }
            hz_s[li] += cq_val * dey_dx;
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
            TIDE_DTYPE const ca_val = ca_batched ? ca[gi] : ca[gj];
            TIDE_DTYPE const cb_val = cb_batched ? cb[gi] : cb[gj];

            TIDE_DTYPE dhz_dx = (hz_s[li] - hz_s[li - 1]) * rdx_t;
            TIDE_DTYPE dhx_dz = (hx_s[li] - hx_s[li - tile_x]) * rdy_t;

            if (x_face) {
              bool const in_pml_x = gx < pml_x0 || gx >= pml_x1;
              if (in_pml_x) {
                TIDE_DTYPE const m_prev =
                    step == 0 ? m_hz_x_in[gi] : m_hz_x_out[gi];
                TIDE_DTYPE const m_new =
                    __ldg(&bx[gx]) * m_prev + __ldg(&ax[gx]) * dhz_dx;
                m_hz_x_out[gi] = m_new;
                dhz_dx = dhz_dx / __ldg(&kx[gx]) + m_new;
              }
            } else {
              bool const in_pml_y = gy < pml_y0 || gy >= pml_y1;
              if (in_pml_y) {
                TIDE_DTYPE const m_prev =
                    step == 0 ? m_hx_z_in[gi] : m_hx_z_out[gi];
                TIDE_DTYPE const m_new =
                    __ldg(&by[gy]) * m_prev + __ldg(&ay[gy]) * dhx_dz;
                m_hx_z_out[gi] = m_new;
                dhx_dz = dhx_dz / __ldg(&ky[gy]) + m_new;
              }
            }

            ey_dst[li] = ca_val * ey_src[li] + cb_val * (dhz_dx - dhx_dz);
          }
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      if (n_sources_per_shot == 1 && single_source_path != nullptr &&
          single_source_block != nullptr && single_source_li != nullptr &&
          f != nullptr) {
        int const src_block = single_source_block[shot_idx];
        int const src_li = single_source_li[shot_idx];
        if (single_source_path[shot_idx] == kTMEbisuPathFace &&
            src_block == compact_block_idx && src_li >= 0) {
          int const ly = src_li / tile_x;
          int const lx = src_li - ly * tile_x;
          if (ly > lo && ly < hi_y - 1 && lx > lo && lx < hi_x - 1) {
            TIDE_DTYPE const *const f_step =
                f + static_cast<int64_t>(step) * n_shots * n_sources_per_shot;
            ey_dst[src_li] += f_step[shot_idx * n_sources_per_shot];
          }
        }
      } else if (n_sources_per_shot > 0 && sources_i != nullptr && f != nullptr) {
        TIDE_DTYPE const *const f_step =
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
      if (n_receivers_per_shot == 1 && single_receiver_path != nullptr &&
          single_receiver_block != nullptr && single_receiver_li != nullptr &&
          r != nullptr) {
        int const rec_block = single_receiver_block[shot_idx];
        int const rec_li = single_receiver_li[shot_idx];
        if (single_receiver_path[shot_idx] == kTMEbisuPathFace &&
            rec_block == compact_block_idx && rec_li >= 0) {
          TIDE_DTYPE *const r_step =
              r + static_cast<int64_t>(step) * n_shots * n_receivers_per_shot;
          r_step[shot_idx * n_receivers_per_shot] = ey_dst[rec_li];
        }
      } else if (n_receivers_per_shot > 0 && receivers_i != nullptr &&
                 r != nullptr) {
        TIDE_DTYPE *const r_step =
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

    TIDE_DTYPE *tmp = ey_src;
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
      ey_out[gi] = ey_src[li];
      hx_out[gi] = hx_s[li];
      hz_out[gi] = hz_s[li];
      if (x_face) {
        bool const in_pml_x = gx < pml_x0 || gx >= pml_x1;
        if (!in_pml_x) {
          m_ey_x_out[gi] = m_ey_x_in[gi];
          m_hz_x_out[gi] = m_hz_x_in[gi];
        }
        m_ey_z_out[gi] = m_ey_z_in[gi];
        m_hx_z_out[gi] = m_hx_z_in[gi];
      } else {
        m_ey_x_out[gi] = m_ey_x_in[gi];
        bool const in_pml_y = gy < pml_y0 || gy >= pml_y1;
        if (!in_pml_y) {
          m_ey_z_out[gi] = m_ey_z_in[gi];
          m_hx_z_out[gi] = m_hx_z_in[gi];
        }
        m_hz_x_out[gi] = m_hz_x_in[gi];
      }
    }
  }
}

template <int ILP>
__global__ __launch_bounds__(256, 1) void forward_kernel_ebisu_tb_x_pml_oop(
    TIDE_DTYPE const *__restrict__ ca, TIDE_DTYPE const *__restrict__ cb,
    TIDE_DTYPE const *__restrict__ cq, TIDE_DTYPE const *__restrict__ f,
    TIDE_DTYPE const *__restrict__ ey_in, TIDE_DTYPE const *__restrict__ hx_in,
    TIDE_DTYPE const *__restrict__ hz_in,
    TIDE_DTYPE const *__restrict__ m_ey_x_in,
    TIDE_DTYPE const *__restrict__ m_ey_z_in,
    TIDE_DTYPE const *__restrict__ m_hx_z_in,
    TIDE_DTYPE const *__restrict__ m_hz_x_in, TIDE_DTYPE *__restrict__ ey_out,
    TIDE_DTYPE *__restrict__ hx_out, TIDE_DTYPE *__restrict__ hz_out,
    TIDE_DTYPE *__restrict__ m_ey_x_out, TIDE_DTYPE *__restrict__ m_ey_z_out,
    TIDE_DTYPE *__restrict__ m_hx_z_out, TIDE_DTYPE *__restrict__ m_hz_x_out,
    TIDE_DTYPE const *__restrict__ ax, TIDE_DTYPE const *__restrict__ axh,
    TIDE_DTYPE const *__restrict__ bx, TIDE_DTYPE const *__restrict__ bxh,
    TIDE_DTYPE const *__restrict__ kx, TIDE_DTYPE const *__restrict__ kxh,
    int64_t const *__restrict__ sources_i,
    int64_t const *__restrict__ receivers_i, TIDE_DTYPE *__restrict__ r,
    int64_t const core_y, int64_t const core_x, int64_t const halo,
    int64_t const k_steps, int2 const *__restrict__ tile_coords) {
  extern __shared__ unsigned char smem_raw[];
  TIDE_DTYPE *ey_a = reinterpret_cast<TIDE_DTYPE *>(smem_raw);
  int const core_y_i = static_cast<int>(core_y);
  int const core_x_i = static_cast<int>(core_x);
  int const halo_i = static_cast<int>(halo);
  int const k_steps_i = static_cast<int>(k_steps);
  int const ny_i = static_cast<int>(ny);
  int const nx_i = static_cast<int>(nx);
  int const tile_y = core_y_i + 2 * halo_i;
  int const tile_x = core_x_i + 2 * halo_i;
  int const tile_elems = tile_y * tile_x;
  TIDE_DTYPE *ey_b = ey_a + tile_elems;
  TIDE_DTYPE *hx_s = ey_b + tile_elems;
  TIDE_DTYPE *hz_s = hx_s + tile_elems;
  TIDE_DTYPE *m_ey_x_s = hz_s + tile_elems;
  TIDE_DTYPE *m_hz_x_s = m_ey_x_s + tile_elems;
  TIDE_DTYPE const rdy_t = static_cast<TIDE_DTYPE>(rdy);
  TIDE_DTYPE const rdx_t = static_cast<TIDE_DTYPE>(rdx);
  int const domain_y_begin = kFdPad;
  int const domain_x_begin = kFdPad;
  int const domain_y_end = ny_i - kFdPad + 1;
  int const domain_x_end = nx_i - kFdPad + 1;

  int64_t const shot_idx = static_cast<int64_t>(blockIdx.z);
  if (shot_idx >= n_shots || tile_coords == nullptr) {
    return;
  }

  int2 const tile = tile_coords[blockIdx.x];
  int const tile_x_idx = tile.x;
  int const tile_y_idx = tile.y;
  int const out_y0 = domain_y_begin + tile_y_idx * core_y_i;
  int const out_x0 = domain_x_begin + tile_x_idx * core_x_i;

  int64_t const shot_offset = shot_idx * shot_numel;
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
      ey_a[li] = ey_in[gi];
      hx_s[li] = hx_in[gi];
      hz_s[li] = hz_in[gi];
      m_ey_x_s[li] = m_ey_x_in[gi];
      m_hz_x_s[li] = m_hz_x_in[gi];
    }
  }
  __syncthreads();

  TIDE_DTYPE *ey_src = ey_a;
  TIDE_DTYPE *ey_dst = ey_b;
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
            TIDE_DTYPE const cq_val = cq_batched ? cq[gi] : cq[gj];

            if (gy < ny_i - kFdPad) {
              hx_s[li] -= cq_val * (ey_src[li + tile_x] - ey_src[li]) * rdy_t;
            }

            if (gx < nx_i - kFdPad) {
              bool const in_pml_x =
                  gx < pml_x0 || gx >= tide_max<int64_t>(pml_x0, pml_x1 - 1);
              TIDE_DTYPE dey_dx = (ey_src[li + 1] - ey_src[li]) * rdx_t;
              if (in_pml_x) {
                TIDE_DTYPE const m_new =
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
            TIDE_DTYPE const ca_val = ca_batched ? ca[gi] : ca[gj];
            TIDE_DTYPE const cb_val = cb_batched ? cb[gi] : cb[gj];
            bool const in_pml_x = gx < pml_x0 || gx >= pml_x1;

            TIDE_DTYPE dhz_dx = (hz_s[li] - hz_s[li - 1]) * rdx_t;
            TIDE_DTYPE dhx_dz = (hx_s[li] - hx_s[li - tile_x]) * rdy_t;

            if (in_pml_x) {
              TIDE_DTYPE const m_new =
                  __ldg(&bx[gx]) * m_hz_x_s[li] + __ldg(&ax[gx]) * dhz_dx;
              m_hz_x_s[li] = m_new;
              dhz_dx = dhz_dx / __ldg(&kx[gx]) + m_new;
            }

            ey_dst[li] = ca_val * ey_src[li] + cb_val * (dhz_dx - dhx_dz);
          }
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      if (n_sources_per_shot > 0 && sources_i != nullptr && f != nullptr) {
        TIDE_DTYPE const *const f_step =
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
        TIDE_DTYPE *const r_step =
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

    TIDE_DTYPE *tmp = ey_src;
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
      ey_out[gi] = ey_src[li];
      hx_out[gi] = hx_s[li];
      hz_out[gi] = hz_s[li];
      m_ey_x_out[gi] = m_ey_x_s[li];
      m_ey_z_out[gi] = m_ey_z_in[gi];
      m_hx_z_out[gi] = m_hx_z_in[gi];
      m_hz_x_out[gi] = m_hz_x_s[li];
    }
  }
}

__global__ void setup_tm_ebisu_single_io_lookup_kernel(
    int64_t const *__restrict__ sources_i,
    int64_t const *__restrict__ receivers_i,
    int const *__restrict__ no_pml_lookup, int const *__restrict__ face_lookup,
    int const *__restrict__ full_lookup, int *__restrict__ source_path,
    int *__restrict__ source_block, int *__restrict__ source_li,
    int *__restrict__ receiver_path, int *__restrict__ receiver_block,
    int *__restrict__ receiver_li, int64_t const core_y,
    int64_t const core_x, int64_t const halo, int64_t const tiles_x) {
  int64_t const shot_idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  if (shot_idx >= n_shots) {
    return;
  }

  auto compute_entry = [&](int64_t const point, int *path_out, int *block_out,
                           int *li_out) {
    int path = kTMEbisuPathNone;
    int block = -1;
    int li = -1;
    if (point >= 0) {
      int const ny_i = static_cast<int>(ny);
      int const nx_i = static_cast<int>(nx);
      int const y = static_cast<int>(point / nx_i);
      int const x = static_cast<int>(point - static_cast<int64_t>(y) * nx_i);
      int const tile_y_idx = (y - kFdPad) / static_cast<int>(core_y);
      int const tile_x_idx = (x - kFdPad) / static_cast<int>(core_x);
      if (y >= kFdPad && x >= kFdPad && tile_y_idx >= 0 && tile_x_idx >= 0 &&
          tile_x_idx < static_cast<int>(tiles_x)) {
        int const dense_idx = tile_y_idx * static_cast<int>(tiles_x) + tile_x_idx;
        if (no_pml_lookup != nullptr) {
          block = no_pml_lookup[dense_idx];
          if (block >= 0) {
            path = kTMEbisuPathNoPml;
          }
        }
        if (path == kTMEbisuPathNone && face_lookup != nullptr) {
          block = face_lookup[dense_idx];
          if (block >= 0) {
            path = kTMEbisuPathFace;
          }
        }
        if (path == kTMEbisuPathNone && full_lookup != nullptr) {
          block = full_lookup[dense_idx];
          if (block >= 0) {
            path = kTMEbisuPathFull;
          }
        }
        if (path != kTMEbisuPathNone) {
          int const out_y0 = kFdPad + tile_y_idx * static_cast<int>(core_y);
          int const out_x0 = kFdPad + tile_x_idx * static_cast<int>(core_x);
          int const load_y0 = out_y0 - static_cast<int>(halo);
          int const load_x0 = out_x0 - static_cast<int>(halo);
          int const tile_x_total =
              static_cast<int>(core_x) + 2 * static_cast<int>(halo);
          li = (y - load_y0) * tile_x_total + (x - load_x0);
        }
      }
    }
    path_out[shot_idx] = path;
    block_out[shot_idx] = block;
    li_out[shot_idx] = li;
  };

  compute_entry(sources_i != nullptr ? sources_i[shot_idx] : -1, source_path,
                source_block, source_li);
  compute_entry(receivers_i != nullptr ? receivers_i[shot_idx] : -1,
                receiver_path, receiver_block, receiver_li);
}

template <int ILP>
__global__ __launch_bounds__(256, 1) void forward_kernel_ebisu_tb_full_oop(
    TIDE_DTYPE const *__restrict__ ca, TIDE_DTYPE const *__restrict__ cb,
    TIDE_DTYPE const *__restrict__ cq, TIDE_DTYPE const *__restrict__ f,
    TIDE_DTYPE const *__restrict__ ey_in, TIDE_DTYPE const *__restrict__ hx_in,
    TIDE_DTYPE const *__restrict__ hz_in,
    TIDE_DTYPE const *__restrict__ m_ey_x_in,
    TIDE_DTYPE const *__restrict__ m_ey_z_in,
    TIDE_DTYPE const *__restrict__ m_hx_z_in,
    TIDE_DTYPE const *__restrict__ m_hz_x_in, TIDE_DTYPE *__restrict__ ey_out,
    TIDE_DTYPE *__restrict__ hx_out, TIDE_DTYPE *__restrict__ hz_out,
    TIDE_DTYPE *__restrict__ m_ey_x_out, TIDE_DTYPE *__restrict__ m_ey_z_out,
    TIDE_DTYPE *__restrict__ m_hx_z_out, TIDE_DTYPE *__restrict__ m_hz_x_out,
    TIDE_DTYPE const *__restrict__ ay, TIDE_DTYPE const *__restrict__ ayh,
    TIDE_DTYPE const *__restrict__ ax, TIDE_DTYPE const *__restrict__ axh,
    TIDE_DTYPE const *__restrict__ by, TIDE_DTYPE const *__restrict__ byh,
    TIDE_DTYPE const *__restrict__ bx, TIDE_DTYPE const *__restrict__ bxh,
    TIDE_DTYPE const *__restrict__ ky, TIDE_DTYPE const *__restrict__ kyh,
    TIDE_DTYPE const *__restrict__ kx, TIDE_DTYPE const *__restrict__ kxh,
    int const *__restrict__ single_source_path,
    int const *__restrict__ single_source_block,
    int const *__restrict__ single_source_li,
    int const *__restrict__ single_receiver_path,
    int const *__restrict__ single_receiver_block,
    int const *__restrict__ single_receiver_li,
    int64_t const *__restrict__ sources_i,
    int64_t const *__restrict__ receivers_i, TIDE_DTYPE *__restrict__ r,
    int64_t const core_y, int64_t const core_x, int64_t const halo,
    int64_t const k_steps, int2 const *__restrict__ tile_coords) {
  extern __shared__ unsigned char smem_raw[];
  TIDE_DTYPE *ey_a = reinterpret_cast<TIDE_DTYPE *>(smem_raw);
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
  TIDE_DTYPE *ey_b = ey_a + tile_elems;
  TIDE_DTYPE *hx_s = ey_b + tile_elems;
  TIDE_DTYPE *hz_s = hx_s + tile_elems;
  TIDE_DTYPE *m_ey_x_s = hz_s + tile_elems;
  TIDE_DTYPE *m_ey_z_s = m_ey_x_s + tile_elems;
  TIDE_DTYPE *m_hx_z_s = m_ey_z_s + tile_elems;
  TIDE_DTYPE *m_hz_x_s = m_hx_z_s + tile_elems;
  TIDE_DTYPE const rdy_t = static_cast<TIDE_DTYPE>(rdy);
  TIDE_DTYPE const rdx_t = static_cast<TIDE_DTYPE>(rdx);
  int const domain_y_begin = kFdPad;
  int const domain_x_begin = kFdPad;
  int const domain_y_end = ny_i - kFdPad + 1;
  int const domain_x_end = nx_i - kFdPad + 1;

  int64_t const shot_idx = static_cast<int64_t>(blockIdx.z);
  if (shot_idx >= n_shots) {
    return;
  }

  int tile_y_idx = static_cast<int>(blockIdx.y);
  int tile_x_idx = static_cast<int>(blockIdx.x);
  if (tile_coords != nullptr) {
    int2 const tile = tile_coords[blockIdx.x];
    tile_x_idx = tile.x;
    tile_y_idx = tile.y;
  }
  int const out_y0 = domain_y_begin + tile_y_idx * core_y_i;
  int const out_x0 = domain_x_begin + tile_x_idx * core_x_i;
  if (tile_coords == nullptr &&
      ebisu_tile_fully_inside_no_pml(out_y0, out_x0, core_y_i, core_x_i,
                                     halo_i)) {
    return;
  }

  int64_t const shot_offset = shot_idx * shot_numel;
  int const load_y0 = out_y0 - halo_i;
  int const load_x0 = out_x0 - halo_i;
  int const tid = static_cast<int>(threadIdx.x);
  int const block_threads = static_cast<int>(blockDim.x);
  int const compact_block_idx = tile_coords != nullptr ? static_cast<int>(blockIdx.x)
                                                       : -1;

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
      ey_a[li] = ey_in[gi];
      hx_s[li] = hx_in[gi];
      hz_s[li] = hz_in[gi];
      m_ey_x_s[li] = m_ey_x_in[gi];
      m_ey_z_s[li] = m_ey_z_in[gi];
      m_hx_z_s[li] = m_hx_z_in[gi];
      m_hz_x_s[li] = m_hz_x_in[gi];
    }
  }
  __syncthreads();

  TIDE_DTYPE *ey_src = ey_a;
  TIDE_DTYPE *ey_dst = ey_b;
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
            TIDE_DTYPE const cq_val = cq_batched ? cq[gi] : cq[gj];

            if (gy < ny_i - kFdPad) {
              bool const in_pml_y =
                  gy < pml_y0 || gy >= tide_max<int64_t>(pml_y0, pml_y1 - 1);
              TIDE_DTYPE dey_dz = (ey_src[li + tile_x] - ey_src[li]) * rdy_t;
              if (in_pml_y) {
                TIDE_DTYPE const m_new =
                    __ldg(&byh[gy]) * m_ey_z_s[li] + __ldg(&ayh[gy]) * dey_dz;
                m_ey_z_s[li] = m_new;
                dey_dz = dey_dz / __ldg(&kyh[gy]) + m_new;
              }
              hx_s[li] -= cq_val * dey_dz;
            }

            if (gx < nx_i - kFdPad) {
              bool const in_pml_x =
                  gx < pml_x0 || gx >= tide_max<int64_t>(pml_x0, pml_x1 - 1);
              TIDE_DTYPE dey_dx = (ey_src[li + 1] - ey_src[li]) * rdx_t;
              if (in_pml_x) {
                TIDE_DTYPE const m_new =
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
            TIDE_DTYPE const ca_val = ca_batched ? ca[gi] : ca[gj];
            TIDE_DTYPE const cb_val = cb_batched ? cb[gi] : cb[gj];
            bool const in_pml_y = gy < pml_y0 || gy >= pml_y1;
            bool const in_pml_x = gx < pml_x0 || gx >= pml_x1;

            TIDE_DTYPE dhz_dx = (hz_s[li] - hz_s[li - 1]) * rdx_t;
            TIDE_DTYPE dhx_dz = (hx_s[li] - hx_s[li - tile_x]) * rdy_t;

            if (in_pml_x) {
              TIDE_DTYPE const m_new =
                  __ldg(&bx[gx]) * m_hz_x_s[li] + __ldg(&ax[gx]) * dhz_dx;
              m_hz_x_s[li] = m_new;
              dhz_dx = dhz_dx / __ldg(&kx[gx]) + m_new;
            }
            if (in_pml_y) {
              TIDE_DTYPE const m_new =
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
        TIDE_DTYPE const *const f_step =
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
        TIDE_DTYPE *const r_step =
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

    TIDE_DTYPE *tmp = ey_src;
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
      ey_out[gi] = ey_src[li];
      hx_out[gi] = hx_s[li];
      hz_out[gi] = hz_s[li];
      m_ey_x_out[gi] = m_ey_x_s[li];
      m_ey_z_out[gi] = m_ey_z_s[li];
      m_hx_z_out[gi] = m_hx_z_s[li];
      m_hz_x_out[gi] = m_hz_x_s[li];
    }
  }
}

template <typename T>
__device__ __forceinline__ void forward_h_no_pml_cell(
    ::tide::GridParams<T> const &params, T const *__restrict__ cq_ptr,
    T const *__restrict__ ey, T *__restrict__ hx, T *__restrict__ hz,
    int64_t const y, int64_t const x, int64_t const shot_idx) {
  if (y < kFdPad || x < kFdPad || y >= params.ny - kFdPad + 1 ||
      x >= params.nx - kFdPad + 1 || shot_idx >= params.n_shots) {
    return;
  }
  int64_t const j = y * params.nx + x;
  int64_t const i = shot_idx * params.shot_numel + j;
  T const cq_val = params.cq_batched ? cq_ptr[i] : cq_ptr[j];
  ::tide::GlobalFieldAccessor<T> ey_acc(ey, params.nx);
  if (y < params.ny - kFdPad) {
    T const dey_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_yh1(
        ey_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);
    hx[i] -= cq_val * dey_dz;
  }
  if (x < params.nx - kFdPad) {
    T const dey_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_xh1(
        ey_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);
    hz[i] += cq_val * dey_dx;
  }
}

template <typename T>
__device__ __forceinline__ void forward_e_no_pml_cell(
    ::tide::GridParams<T> const &params, T const *__restrict__ ca_ptr,
    T const *__restrict__ cb_ptr, T const *__restrict__ hx,
    T const *__restrict__ hz, T *__restrict__ ey, int64_t const y,
    int64_t const x, int64_t const shot_idx) {
  if (y < kFdPad || x < kFdPad || y >= params.ny - kFdPad + 1 ||
      x >= params.nx - kFdPad + 1 || shot_idx >= params.n_shots) {
    return;
  }
  int64_t const j = y * params.nx + x;
  int64_t const i = shot_idx * params.shot_numel + j;
  T const ca_val = params.ca_batched ? ca_ptr[i] : ca_ptr[j];
  T const cb_val = params.cb_batched ? cb_ptr[i] : cb_ptr[j];
  ::tide::GlobalFieldAccessor<T> hz_acc(hz, params.nx);
  ::tide::GlobalFieldAccessor<T> hx_acc(hx, params.nx);
  T const dhz_dx = ::tide::DiffForward<TIDE_STENCIL>::diff_x1(
      hz_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);
  T const dhx_dz = ::tide::DiffForward<TIDE_STENCIL>::diff_y1(
      hx_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);
  ey[i] = ca_val * ey[i] + cb_val * (dhz_dx - dhx_dz);
}

template <int ILP>
__global__ void forward_kernel_fused_ksteps_no_pml(
    TIDE_DTYPE const *__restrict__ ca, TIDE_DTYPE const *__restrict__ cb,
    TIDE_DTYPE const *__restrict__ cq, TIDE_DTYPE *__restrict__ ey,
    TIDE_DTYPE *__restrict__ hx, TIDE_DTYPE *__restrict__ hz,
    TIDE_DTYPE const *__restrict__ ay, TIDE_DTYPE const *__restrict__ ayh,
    TIDE_DTYPE const *__restrict__ ax, TIDE_DTYPE const *__restrict__ axh,
    TIDE_DTYPE const *__restrict__ by, TIDE_DTYPE const *__restrict__ byh,
    TIDE_DTYPE const *__restrict__ bx, TIDE_DTYPE const *__restrict__ bxh,
    TIDE_DTYPE const *__restrict__ ky, TIDE_DTYPE const *__restrict__ kyh,
    TIDE_DTYPE const *__restrict__ kx, TIDE_DTYPE const *__restrict__ kxh,
    int64_t const k_steps) {
  cg::grid_group grid = cg::this_grid();
  ::tide::GridParams<TIDE_DTYPE> params = {
      ay,          ayh,         ax,          axh,        by,     byh,
      bx,          bxh,         ky,          kyh,        kx,     kxh,
      static_cast<TIDE_DTYPE>(rdy), static_cast<TIDE_DTYPE>(rdx), n_shots,
      ny,          nx,          shot_numel,  0,          ny,     0,
      nx,          ca_batched,  cb_batched,  cq_batched};

  int64_t const total = n_shots * shot_numel;
  int64_t const tid =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t const stride = (int64_t)blockDim.x * (int64_t)gridDim.x;
  for (int64_t step = 0; step < k_steps; ++step) {
    for (int64_t base = tid; base < total; base += stride * ILP) {
#pragma unroll
      for (int lane = 0; lane < ILP; ++lane) {
        int64_t const idx = base + (int64_t)lane * stride;
        if (idx >= total) {
          continue;
        }
        int64_t const shot_idx = idx / shot_numel;
        int64_t const j = idx - shot_idx * shot_numel;
        int64_t const y = j / nx;
        int64_t const x = j - y * nx;
        forward_h_no_pml_cell(params, cq, ey, hx, hz, y, x, shot_idx);
      }
    }
    grid.sync();
    for (int64_t base = tid; base < total; base += stride * ILP) {
#pragma unroll
      for (int lane = 0; lane < ILP; ++lane) {
        int64_t const idx = base + (int64_t)lane * stride;
        if (idx >= total) {
          continue;
        }
        int64_t const shot_idx = idx / shot_numel;
        int64_t const j = idx - shot_idx * shot_numel;
        int64_t const y = j / nx;
        int64_t const x = j - y * nx;
        forward_e_no_pml_cell(params, ca, cb, hx, hz, ey, y, x, shot_idx);
      }
    }
    grid.sync();
  }
}

static inline size_t ring_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode_h,
    size_t const bytes_per_step_store) {
  if (storage_mode_h == STORAGE_DEVICE) {
    return (size_t)step_idx * bytes_per_step_store;
  }
  if (storage_mode_h == STORAGE_CPU) {
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

template <int ILP>
static inline cudaError_t launch_tm_fused_no_pml_kernel(
    dim3 const dim_block, int64_t const grid_blocks,
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE *const ey, TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz, TIDE_DTYPE const *const ay,
    TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const ax,
    TIDE_DTYPE const *const axh, TIDE_DTYPE const *const by,
    TIDE_DTYPE const *const byh, TIDE_DTYPE const *const bx,
    TIDE_DTYPE const *const bxh, TIDE_DTYPE const *const ky,
    TIDE_DTYPE const *const kyh, TIDE_DTYPE const *const kx,
    TIDE_DTYPE const *const kxh, int64_t const k_steps) {
  void *args[] = {(void *)&ca,  (void *)&cb,  (void *)&cq,  (void *)&ey,
                  (void *)&hx,  (void *)&hz,  (void *)&ay,  (void *)&ayh,
                  (void *)&ax,  (void *)&axh, (void *)&by,  (void *)&byh,
                  (void *)&bx,  (void *)&bxh, (void *)&ky,  (void *)&kyh,
                  (void *)&kx,  (void *)&kxh, (void *)&k_steps};
  return cudaLaunchCooperativeKernel(
      (void *)forward_kernel_fused_ksteps_no_pml<ILP>,
      dim3(to_dim_u32(grid_blocks), 1, 1), dim_block, args);
}

template <int ILP>
static inline cudaError_t launch_tm_ebisu_no_pml_kernel(
    dim3 const dim_grid, dim3 const dim_block, size_t const shared_bytes,
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE *const ey, TIDE_DTYPE *const hx,
    TIDE_DTYPE *const hz, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const k_steps) {
  forward_kernel_ebisu_tb_no_pml<ILP><<<dim_grid, dim_block, shared_bytes>>>(
      ca, cb, cq, ey, hx, hz, core_y, core_x, halo, k_steps);
  return cudaGetLastError();
}

template <int ILP>
static inline cudaError_t launch_tm_ebisu_no_pml_oop_kernel(
    dim3 const dim_grid, dim3 const dim_block, size_t const shared_bytes,
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const ey_in, TIDE_DTYPE const *const hx_in,
    TIDE_DTYPE const *const hz_in, TIDE_DTYPE *const ey_out,
    TIDE_DTYPE *const hx_out, TIDE_DTYPE *const hz_out,
    int const *const single_source_path, int const *const single_source_block,
    int const *const single_source_li,
    int const *const single_receiver_path,
    int const *const single_receiver_block,
    int const *const single_receiver_li,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    TIDE_DTYPE *const r, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const k_steps,
    int2 const *const tile_coords = nullptr) {
  forward_kernel_ebisu_tb_no_pml_oop<ILP>
      <<<dim_grid, dim_block, shared_bytes>>>(
          ca, cb, cq, f, ey_in, hx_in, hz_in, ey_out, hx_out, hz_out,
          single_source_path, single_source_block, single_source_li,
          single_receiver_path, single_receiver_block, single_receiver_li,
          sources_i, receivers_i, r, core_y, core_x, halo, k_steps, tile_coords);
  return cudaGetLastError();
}

template <int ILP>
static inline cudaError_t launch_tm_ebisu_full_kernel(
    dim3 const dim_grid, dim3 const dim_block, size_t const shared_bytes,
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f,
    TIDE_DTYPE *const ey, TIDE_DTYPE *const hx, TIDE_DTYPE *const hz,
    TIDE_DTYPE *const m_ey_x, TIDE_DTYPE *const m_ey_z,
    TIDE_DTYPE *const m_hx_z, TIDE_DTYPE *const m_hz_x,
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const by, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    TIDE_DTYPE *const r, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const k_steps) {
  forward_kernel_ebisu_tb_full<ILP><<<dim_grid, dim_block, shared_bytes>>>(
      ca, cb, cq, f, ey, hx, hz, m_ey_x, m_ey_z, m_hx_z, m_hz_x, ay, ayh, ax,
      axh, by, byh, bx, bxh, ky, kyh, kx, kxh, sources_i, receivers_i, r,
      core_y, core_x, halo, k_steps);
  return cudaGetLastError();
}

template <int ILP>
static inline cudaError_t launch_tm_ebisu_full_oop_kernel(
    dim3 const dim_grid, dim3 const dim_block, size_t const shared_bytes,
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const ey_in, TIDE_DTYPE const *const hx_in,
    TIDE_DTYPE const *const hz_in, TIDE_DTYPE const *const m_ey_x_in,
    TIDE_DTYPE const *const m_ey_z_in, TIDE_DTYPE const *const m_hx_z_in,
    TIDE_DTYPE const *const m_hz_x_in, TIDE_DTYPE *const ey_out,
    TIDE_DTYPE *const hx_out, TIDE_DTYPE *const hz_out,
    TIDE_DTYPE *const m_ey_x_out, TIDE_DTYPE *const m_ey_z_out,
    TIDE_DTYPE *const m_hx_z_out, TIDE_DTYPE *const m_hz_x_out,
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const by, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int const *const single_source_path, int const *const single_source_block,
    int const *const single_source_li,
    int const *const single_receiver_path,
    int const *const single_receiver_block,
    int const *const single_receiver_li,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    TIDE_DTYPE *const r, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const k_steps,
    int2 const *const tile_coords = nullptr) {
  forward_kernel_ebisu_tb_full_oop<ILP><<<dim_grid, dim_block, shared_bytes>>>(
      ca, cb, cq, f, ey_in, hx_in, hz_in, m_ey_x_in, m_ey_z_in, m_hx_z_in,
      m_hz_x_in, ey_out, hx_out, hz_out, m_ey_x_out, m_ey_z_out, m_hx_z_out,
      m_hz_x_out, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh,
      single_source_path, single_source_block, single_source_li,
      single_receiver_path, single_receiver_block, single_receiver_li,
      sources_i, receivers_i, r, core_y, core_x, halo, k_steps, tile_coords);
  return cudaGetLastError();
}

template <int ILP>
static inline cudaError_t launch_tm_ebisu_top_pml_y_oop_kernel(
    dim3 const dim_grid, dim3 const dim_block, size_t const shared_bytes,
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const ey_in, TIDE_DTYPE const *const hx_in,
    TIDE_DTYPE const *const hz_in, TIDE_DTYPE const *const m_ey_x_in,
    TIDE_DTYPE const *const m_ey_z_in, TIDE_DTYPE const *const m_hx_z_in,
    TIDE_DTYPE const *const m_hz_x_in, TIDE_DTYPE *const ey_out,
    TIDE_DTYPE *const hx_out, TIDE_DTYPE *const hz_out,
    TIDE_DTYPE *const m_ey_x_out, TIDE_DTYPE *const m_ey_z_out,
    TIDE_DTYPE *const m_hx_z_out, TIDE_DTYPE *const m_hz_x_out,
    TIDE_DTYPE const *const ay, TIDE_DTYPE const *const ayh,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const by, TIDE_DTYPE const *const byh,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int const *const single_source_path, int const *const single_source_block,
    int const *const single_source_li,
    int const *const single_receiver_path,
    int const *const single_receiver_block,
    int const *const single_receiver_li,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    TIDE_DTYPE *const r, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const k_steps, bool const use_async_copy,
    int2 const *const tile_coords) {
  forward_kernel_ebisu_tb_top_pml_y_oop<ILP>
      <<<dim_grid, dim_block, shared_bytes>>>(
          ca, cb, cq, f, ey_in, hx_in, hz_in, m_ey_x_in, m_ey_z_in, m_hx_z_in,
          m_hz_x_in, ey_out, hx_out, hz_out, m_ey_x_out, m_ey_z_out,
          m_hx_z_out, m_hz_x_out, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
          kyh, kx, kxh, single_source_path, single_source_block,
          single_source_li, single_receiver_path, single_receiver_block,
          single_receiver_li, sources_i, receivers_i, r, core_y, core_x,
          halo, k_steps, use_async_copy, tile_coords);
  return cudaGetLastError();
}

template <int ILP>
static inline cudaError_t launch_tm_ebisu_x_pml_oop_kernel(
    dim3 const dim_grid, dim3 const dim_block, size_t const shared_bytes,
    TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const ey_in, TIDE_DTYPE const *const hx_in,
    TIDE_DTYPE const *const hz_in, TIDE_DTYPE const *const m_ey_x_in,
    TIDE_DTYPE const *const m_ey_z_in, TIDE_DTYPE const *const m_hx_z_in,
    TIDE_DTYPE const *const m_hz_x_in, TIDE_DTYPE *const ey_out,
    TIDE_DTYPE *const hx_out, TIDE_DTYPE *const hz_out,
    TIDE_DTYPE *const m_ey_x_out, TIDE_DTYPE *const m_ey_z_out,
    TIDE_DTYPE *const m_hx_z_out, TIDE_DTYPE *const m_hz_x_out,
    TIDE_DTYPE const *const ax, TIDE_DTYPE const *const axh,
    TIDE_DTYPE const *const bx, TIDE_DTYPE const *const bxh,
    TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
    int64_t const *const sources_i, int64_t const *const receivers_i,
    TIDE_DTYPE *const r, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const k_steps,
    int2 const *const tile_coords) {
  forward_kernel_ebisu_tb_x_pml_oop<ILP>
      <<<dim_grid, dim_block, shared_bytes>>>(
          ca, cb, cq, f, ey_in, hx_in, hz_in, m_ey_x_in, m_ey_z_in, m_hx_z_in,
          m_hz_x_in, ey_out, hx_out, hz_out, m_ey_x_out, m_ey_z_out,
          m_hx_z_out, m_hz_x_out, ax, axh, bx, bxh, kx, kxh, sources_i,
          receivers_i, r, core_y, core_x, halo, k_steps, tile_coords);
  return cudaGetLastError();
}

static inline cudaError_t launch_tm_ebisu_single_io_lookup_kernel(
    int64_t const n_shots_h, int64_t const *const sources_i,
    int64_t const *const receivers_i, int const *const no_pml_lookup,
    int const *const face_lookup, int const *const full_lookup,
    int *const source_path, int *const source_block, int *const source_li,
    int *const receiver_path, int *const receiver_block,
    int *const receiver_li, int64_t const core_y, int64_t const core_x,
    int64_t const halo, int64_t const tiles_x) {
  dim3 const dim_block(128, 1, 1);
  dim3 const dim_grid(to_dim_u32((n_shots_h + dim_block.x - 1) / dim_block.x),
                      1, 1);
  setup_tm_ebisu_single_io_lookup_kernel<<<dim_grid, dim_block>>>(
      sources_i, receivers_i, no_pml_lookup, face_lookup, full_lookup,
      source_path, source_block, source_li, receiver_path, receiver_block,
      receiver_li, core_y, core_x, halo, tiles_x);
  return cudaGetLastError();
}

static inline bool can_use_tm_ebisu_no_pml_path(
    TMEbisuRuntimeConfig const &ebisu_cfg, bool const has_dispersion,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const ny_h, int64_t const nx_h) {
  if (!ebisu_cfg.enabled || ebisu_cfg.steps <= 0) {
    return false;
  }
  if (has_dispersion || n_sources_per_shot_h != 0 || n_receivers_per_shot_h != 0) {
    return false;
  }
  if constexpr (TIDE_STENCIL != 2) {
    return false;
  }
  return pml_y0_h == kFdPad && pml_x0_h == kFdPad && pml_y1_h == ny_h &&
         pml_x1_h == nx_h;
}

static inline bool can_use_tm_ebisu_general_path(
    TMEbisuRuntimeConfig const &ebisu_cfg, bool const has_dispersion) {
  if (!ebisu_cfg.enabled || ebisu_cfg.steps <= 0 || has_dispersion) {
    return false;
  }
  if constexpr (TIDE_STENCIL != 2) {
    return false;
  }
  return true;
}

static inline bool can_use_tm_fused_no_pml_path(
    TMFusedRuntimeConfig const &fused_cfg, bool const has_dispersion,
    int64_t const n_sources_per_shot_h, int64_t const n_receivers_per_shot_h,
    int64_t const pml_y0_h, int64_t const pml_x0_h, int64_t const pml_y1_h,
    int64_t const pml_x1_h, int64_t const ny_h, int64_t const nx_h) {
  if (!fused_cfg.enabled || fused_cfg.steps <= 0) {
    return false;
  }
  if (has_dispersion || n_sources_per_shot_h != 0 || n_receivers_per_shot_h != 0) {
    return false;
  }
  if constexpr (TIDE_STENCIL != 2) {
    return false;
  }
  return pml_y0_h == kFdPad && pml_x0_h == kFdPad && pml_y1_h == ny_h &&
         pml_x1_h == nx_h;
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
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;

  int64_t const shot_numel_h = ny_h * nx_h;
  static DeviceConstantCache2D constant_cache{};
  sync_device_constants_if_needed(
      constant_cache, rdy_h, rdx_h, n_shots_h, ny_h, nx_h, shot_numel_h,
      n_sources_per_shot_h, n_receivers_per_shot_h, pml_y0_h, pml_x0_h,
      pml_y1_h, pml_x1_h, ca_batched_h, cb_batched_h, cq_batched_h, device);

  TMForwardLaunchConfig const launch_cfg = make_tm_forward_launch_config(
      n_shots_h, ny_h, nx_h, n_sources_per_shot_h, n_receivers_per_shot_h);

  TMEbisuRuntimeConfig const ebisu_cfg = read_tm_ebisu_runtime_config();
  bool const debug_path = read_env_flag("TIDE_TM_DEBUG_PATH");
  bool const try_ebisu_no_pml = can_use_tm_ebisu_no_pml_path(
      ebisu_cfg, has_dispersion, n_sources_per_shot_h, n_receivers_per_shot_h,
      pml_y0_h, pml_x0_h, pml_y1_h, pml_x1_h, ny_h, nx_h);
  if (try_ebisu_no_pml) {
    int64_t const halo = ebisu_cfg.steps;
    int64_t const tile_x = ebisu_cfg.tile_x + 2 * halo;
    int64_t const tile_y = ebisu_cfg.tile_y + 2 * halo;
    size_t const shared_bytes =
        (size_t)tile_x * (size_t)tile_y * 4 * sizeof(TIDE_DTYPE);
    int max_optin_shared = 0;
    tide::cuda_check_or_abort(
        cudaDeviceGetAttribute(&max_optin_shared,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
        __FILE__, __LINE__);
    if ((int64_t)tile_x > 0 && (int64_t)tile_y > 0 &&
        shared_bytes <= (size_t)max_optin_shared) {
      int max_blocks_per_sm = 0;
      cudaError_t occ_err = cudaSuccess;
      switch (ebisu_cfg.ilp) {
      case 1:
        tide::cuda_check_or_abort(
            cudaFuncSetAttribute(
                forward_kernel_ebisu_tb_no_pml<1>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes),
            __FILE__, __LINE__);
        occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, forward_kernel_ebisu_tb_no_pml<1>, 256,
            shared_bytes);
        break;
      case 2:
        tide::cuda_check_or_abort(
            cudaFuncSetAttribute(
                forward_kernel_ebisu_tb_no_pml<2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes),
            __FILE__, __LINE__);
        occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, forward_kernel_ebisu_tb_no_pml<2>, 256,
            shared_bytes);
        break;
      default:
        tide::cuda_check_or_abort(
            cudaFuncSetAttribute(
                forward_kernel_ebisu_tb_no_pml<4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes),
            __FILE__, __LINE__);
        occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, forward_kernel_ebisu_tb_no_pml<4>, 256,
            shared_bytes);
        break;
      }
      tide::cuda_check_or_abort(occ_err, __FILE__, __LINE__);

      if (max_blocks_per_sm > 0) {
        int64_t const interior_y = ny_h - 2 * kFdPad;
        int64_t const interior_x = nx_h - 2 * kFdPad;
        dim3 const ebisu_grid(
            to_dim_u32((interior_x + ebisu_cfg.tile_x - 1) / ebisu_cfg.tile_x),
            to_dim_u32((interior_y + ebisu_cfg.tile_y - 1) / ebisu_cfg.tile_y),
            to_dim_u32(n_shots_h));
        dim3 const ebisu_block(256, 1, 1);
        int64_t remaining = nt;
        while (remaining > 0) {
          int64_t const chunk_steps =
              tide_min<int64_t>(remaining, ebisu_cfg.steps);
          cudaError_t launch_err = cudaSuccess;
          switch (ebisu_cfg.ilp) {
          case 1:
            launch_err = launch_tm_ebisu_no_pml_kernel<1>(
                ebisu_grid, ebisu_block, shared_bytes, ca, cb, cq, ey, hx, hz,
                ebisu_cfg.tile_y, ebisu_cfg.tile_x, chunk_steps, chunk_steps);
            break;
          case 2:
            launch_err = launch_tm_ebisu_no_pml_kernel<2>(
                ebisu_grid, ebisu_block, shared_bytes, ca, cb, cq, ey, hx, hz,
                ebisu_cfg.tile_y, ebisu_cfg.tile_x, chunk_steps, chunk_steps);
            break;
          default:
            launch_err = launch_tm_ebisu_no_pml_kernel<4>(
                ebisu_grid, ebisu_block, shared_bytes, ca, cb, cq, ey, hx, hz,
                ebisu_cfg.tile_y, ebisu_cfg.tile_x, chunk_steps, chunk_steps);
            break;
          }
          tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
          remaining -= chunk_steps;
        }
        if (debug_path) {
          std::fprintf(stderr,
                       "TIDE TM path: ebisu steps=%lld tile=%lldx%lld ilp=%lld "
                       "shared=%zuB\n",
                       (long long)ebisu_cfg.steps, (long long)ebisu_cfg.tile_x,
                       (long long)ebisu_cfg.tile_y, (long long)ebisu_cfg.ilp,
                       shared_bytes);
        }
        tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
        return;
      }
      if (debug_path) {
        std::fprintf(stderr,
                     "TIDE TM ebisu fallback: occupancy blocks_per_sm=%d\n",
                     max_blocks_per_sm);
      }
    } else if (debug_path) {
      std::fprintf(stderr,
                   "TIDE TM ebisu fallback: shared=%zuB max_optin=%dB tile=%lldx%lld halo=%lld\n",
                   shared_bytes, max_optin_shared, (long long)tile_x,
                   (long long)tile_y, (long long)halo);
    }
  } else {
    bool const try_ebisu_general =
        can_use_tm_ebisu_general_path(ebisu_cfg, has_dispersion);
    if (try_ebisu_general) {
      bool const use_face_pml_specialization = ebisu_cfg.face_pml;
      int64_t const halo = ebisu_cfg.steps;
      int64_t const tile_x = ebisu_cfg.tile_x + 2 * halo;
      int64_t const tile_y = ebisu_cfg.tile_y + 2 * halo;
      size_t const shared_bytes_full =
          (size_t)tile_x * (size_t)tile_y * 8 * sizeof(TIDE_DTYPE);
      size_t const shared_bytes_face =
          use_face_pml_specialization
              ? (size_t)tile_x * (size_t)tile_y * 4 * sizeof(TIDE_DTYPE)
              : 0;
      size_t const shared_bytes_no_pml =
          (size_t)tile_x * (size_t)tile_y * 4 * sizeof(TIDE_DTYPE);
      int max_optin_shared = 0;
      tide::cuda_check_or_abort(
          cudaDeviceGetAttribute(&max_optin_shared,
                                 cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
          __FILE__, __LINE__);
      if ((int64_t)tile_x > 0 && (int64_t)tile_y > 0 &&
          shared_bytes_full <= (size_t)max_optin_shared &&
          (!use_face_pml_specialization ||
           shared_bytes_face <= (size_t)max_optin_shared) &&
          shared_bytes_no_pml <= (size_t)max_optin_shared) {
        int max_blocks_per_sm_full = 0;
        int max_blocks_per_sm_y_pml = 0;
        int max_blocks_per_sm_x_pml = 0;
        int max_blocks_per_sm_no_pml = 0;
        cudaError_t occ_err_full = cudaSuccess;
        cudaError_t occ_err_y_pml = cudaSuccess;
        cudaError_t occ_err_x_pml = cudaSuccess;
        cudaError_t occ_err_no_pml = cudaSuccess;
        switch (ebisu_cfg.ilp) {
        case 1:
          tide::cuda_check_or_abort(
              cudaFuncSetAttribute(forward_kernel_ebisu_tb_full_oop<1>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shared_bytes_full),
              __FILE__, __LINE__);
          tide::cuda_check_or_abort(
              cudaFuncSetAttribute(forward_kernel_ebisu_tb_no_pml_oop<1>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shared_bytes_no_pml),
              __FILE__, __LINE__);
          if (use_face_pml_specialization) {
            tide::cuda_check_or_abort(
                cudaFuncSetAttribute(forward_kernel_ebisu_tb_top_pml_y_oop<1>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)shared_bytes_face),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaFuncSetAttribute(forward_kernel_ebisu_tb_x_pml_oop<1>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)shared_bytes_face),
                __FILE__, __LINE__);
          }
          occ_err_full = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &max_blocks_per_sm_full, forward_kernel_ebisu_tb_full_oop<1>, 256,
              shared_bytes_full);
          if (use_face_pml_specialization) {
            occ_err_y_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm_y_pml,
                forward_kernel_ebisu_tb_top_pml_y_oop<1>, 256,
                shared_bytes_face);
            occ_err_x_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm_x_pml,
                forward_kernel_ebisu_tb_x_pml_oop<1>, 256, shared_bytes_face);
          } else {
            max_blocks_per_sm_y_pml = 1;
            max_blocks_per_sm_x_pml = 1;
          }
          occ_err_no_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &max_blocks_per_sm_no_pml, forward_kernel_ebisu_tb_no_pml_oop<1>,
              256, shared_bytes_no_pml);
          break;
        case 2:
          tide::cuda_check_or_abort(
              cudaFuncSetAttribute(forward_kernel_ebisu_tb_full_oop<2>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shared_bytes_full),
              __FILE__, __LINE__);
          tide::cuda_check_or_abort(
              cudaFuncSetAttribute(forward_kernel_ebisu_tb_no_pml_oop<2>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shared_bytes_no_pml),
              __FILE__, __LINE__);
          if (use_face_pml_specialization) {
            tide::cuda_check_or_abort(
                cudaFuncSetAttribute(forward_kernel_ebisu_tb_top_pml_y_oop<2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)shared_bytes_face),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaFuncSetAttribute(forward_kernel_ebisu_tb_x_pml_oop<2>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)shared_bytes_face),
                __FILE__, __LINE__);
          }
          occ_err_full = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &max_blocks_per_sm_full, forward_kernel_ebisu_tb_full_oop<2>, 256,
              shared_bytes_full);
          if (use_face_pml_specialization) {
            occ_err_y_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm_y_pml,
                forward_kernel_ebisu_tb_top_pml_y_oop<2>, 256,
                shared_bytes_face);
            occ_err_x_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm_x_pml,
                forward_kernel_ebisu_tb_x_pml_oop<2>, 256, shared_bytes_face);
          } else {
            max_blocks_per_sm_y_pml = 1;
            max_blocks_per_sm_x_pml = 1;
          }
          occ_err_no_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &max_blocks_per_sm_no_pml, forward_kernel_ebisu_tb_no_pml_oop<2>,
              256, shared_bytes_no_pml);
          break;
        default:
          tide::cuda_check_or_abort(
              cudaFuncSetAttribute(forward_kernel_ebisu_tb_full_oop<4>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shared_bytes_full),
              __FILE__, __LINE__);
          tide::cuda_check_or_abort(
              cudaFuncSetAttribute(forward_kernel_ebisu_tb_no_pml_oop<4>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)shared_bytes_no_pml),
              __FILE__, __LINE__);
          if (use_face_pml_specialization) {
            tide::cuda_check_or_abort(
                cudaFuncSetAttribute(forward_kernel_ebisu_tb_top_pml_y_oop<4>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)shared_bytes_face),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaFuncSetAttribute(forward_kernel_ebisu_tb_x_pml_oop<4>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)shared_bytes_face),
                __FILE__, __LINE__);
          }
          occ_err_full = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &max_blocks_per_sm_full, forward_kernel_ebisu_tb_full_oop<4>, 256,
              shared_bytes_full);
          if (use_face_pml_specialization) {
            occ_err_y_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm_y_pml,
                forward_kernel_ebisu_tb_top_pml_y_oop<4>, 256,
                shared_bytes_face);
            occ_err_x_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm_x_pml,
                forward_kernel_ebisu_tb_x_pml_oop<4>, 256, shared_bytes_face);
          } else {
            max_blocks_per_sm_y_pml = 1;
            max_blocks_per_sm_x_pml = 1;
          }
          occ_err_no_pml = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &max_blocks_per_sm_no_pml, forward_kernel_ebisu_tb_no_pml_oop<4>,
              256, shared_bytes_no_pml);
          break;
        }
        tide::cuda_check_or_abort(occ_err_full, __FILE__, __LINE__);
        if (use_face_pml_specialization) {
          tide::cuda_check_or_abort(occ_err_y_pml, __FILE__, __LINE__);
          tide::cuda_check_or_abort(occ_err_x_pml, __FILE__, __LINE__);
        }
        tide::cuda_check_or_abort(occ_err_no_pml, __FILE__, __LINE__);

        if (max_blocks_per_sm_full > 0 && max_blocks_per_sm_no_pml > 0 &&
            max_blocks_per_sm_y_pml > 0 && max_blocks_per_sm_x_pml > 0) {
          int64_t const domain_y = ny_h - 2 * kFdPad + 1;
          int64_t const domain_x = nx_h - 2 * kFdPad + 1;
          dim3 const ebisu_grid(
              to_dim_u32((domain_x + ebisu_cfg.tile_x - 1) / ebisu_cfg.tile_x),
              to_dim_u32((domain_y + ebisu_cfg.tile_y - 1) / ebisu_cfg.tile_y),
              to_dim_u32(n_shots_h));
          dim3 const ebisu_block(256, 1, 1);
          size_t const field_bytes =
              (size_t)n_shots_h * (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
          static TMEbisuWorkspace ebisu_workspace{};
          ensure_tm_ebisu_workspace(ebisu_workspace, device, field_bytes);
          auto seed_workspace = [&](TIDE_DTYPE *dst, TIDE_DTYPE *src) {
            tide::cuda_check_or_abort(
                cudaMemcpy(dst, src, field_bytes, cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
          };
          seed_workspace(ebisu_workspace.ey, ey);
          seed_workspace(ebisu_workspace.hx, hx);
          seed_workspace(ebisu_workspace.hz, hz);
          seed_workspace(ebisu_workspace.m_ey_x, m_ey_x);
          seed_workspace(ebisu_workspace.m_ey_z, m_ey_z);
          seed_workspace(ebisu_workspace.m_hx_z, m_hx_z);
          seed_workspace(ebisu_workspace.m_hz_x, m_hz_x);

          std::vector<int2> no_pml_tiles_host;
          std::vector<int2> y_pml_tiles_host;
          std::vector<int2> x_pml_tiles_host;
          std::vector<int2> face_pml_tiles_host;
          std::vector<int2> full_tiles_host;
          std::vector<int> no_pml_tile_lookup_host;
          std::vector<int> face_tile_lookup_host;
          std::vector<int> full_tile_lookup_host;
          build_tm_ebisu_tile_lists(
              no_pml_tiles_host, y_pml_tiles_host, x_pml_tiles_host,
              full_tiles_host, domain_y, domain_x, ebisu_cfg.tile_y,
              ebisu_cfg.tile_x, ebisu_cfg.steps, pml_y0_h, pml_x0_h, pml_y1_h,
              pml_x1_h, use_face_pml_specialization);
          face_pml_tiles_host = y_pml_tiles_host;
          face_pml_tiles_host.reserve(y_pml_tiles_host.size() +
                                      x_pml_tiles_host.size());
          for (int2 const tile : x_pml_tiles_host) {
            face_pml_tiles_host.push_back(
                make_int2(-tile.x - 1, tile.y));
          }
          bool const use_single_source_fast =
              n_sources_per_shot_h == 1 && sources_i != nullptr &&
              (!no_pml_tiles_host.empty() || !face_pml_tiles_host.empty() ||
               !full_tiles_host.empty());
          bool const use_single_receiver_fast =
              n_receivers_per_shot_h == 1 && receivers_i != nullptr &&
              (!no_pml_tiles_host.empty() || !face_pml_tiles_host.empty() ||
               !full_tiles_host.empty());
          bool const use_single_io_fast =
              use_single_source_fast || use_single_receiver_fast;
          if (use_single_io_fast) {
            build_tm_ebisu_face_tile_lookup(no_pml_tile_lookup_host,
                                            no_pml_tiles_host, domain_y,
                                            domain_x, ebisu_cfg.tile_y,
                                            ebisu_cfg.tile_x);
            build_tm_ebisu_face_tile_lookup(face_tile_lookup_host,
                                            face_pml_tiles_host, domain_y,
                                            domain_x, ebisu_cfg.tile_y,
                                            ebisu_cfg.tile_x);
            build_tm_ebisu_face_tile_lookup(full_tile_lookup_host,
                                            full_tiles_host, domain_y, domain_x,
                                            ebisu_cfg.tile_y,
                                            ebisu_cfg.tile_x);
          }
          ensure_tm_ebisu_tile_capacity(ebisu_workspace,
                                        no_pml_tiles_host.size(),
                                        face_pml_tiles_host.size(),
                                        0,
                                        full_tiles_host.size());
          if (use_single_io_fast) {
            ensure_tm_ebisu_lookup_capacity(
                &ebisu_workspace.no_pml_tile_lookup,
                ebisu_workspace.no_pml_tile_lookup_capacity,
                no_pml_tile_lookup_host.size());
            ensure_tm_ebisu_lookup_capacity(
                &ebisu_workspace.face_tile_lookup,
                ebisu_workspace.face_tile_lookup_capacity,
                face_tile_lookup_host.size());
            ensure_tm_ebisu_lookup_capacity(
                &ebisu_workspace.full_tile_lookup,
                ebisu_workspace.full_tile_lookup_capacity,
                full_tile_lookup_host.size());
            ensure_tm_ebisu_shot_capacity(
                ebisu_workspace, static_cast<size_t>(n_shots_h));
          }
          if (!no_pml_tiles_host.empty()) {
            tide::cuda_check_or_abort(
                cudaMemcpy(ebisu_workspace.no_pml_tiles,
                           no_pml_tiles_host.data(),
                           no_pml_tiles_host.size() * sizeof(int2),
                           cudaMemcpyHostToDevice),
                __FILE__, __LINE__);
          }
          if (!face_pml_tiles_host.empty()) {
            tide::cuda_check_or_abort(
                cudaMemcpy(ebisu_workspace.y_pml_tiles,
                           face_pml_tiles_host.data(),
                           face_pml_tiles_host.size() * sizeof(int2),
                           cudaMemcpyHostToDevice),
                __FILE__, __LINE__);
          }
          if (!full_tiles_host.empty()) {
            tide::cuda_check_or_abort(
                cudaMemcpy(ebisu_workspace.full_tiles, full_tiles_host.data(),
                           full_tiles_host.size() * sizeof(int2),
                cudaMemcpyHostToDevice),
                __FILE__, __LINE__);
          }
          if (use_single_io_fast) {
            if (!no_pml_tile_lookup_host.empty()) {
              tide::cuda_check_or_abort(
                  cudaMemcpy(ebisu_workspace.no_pml_tile_lookup,
                             no_pml_tile_lookup_host.data(),
                             no_pml_tile_lookup_host.size() * sizeof(int),
                             cudaMemcpyHostToDevice),
                  __FILE__, __LINE__);
            }
            tide::cuda_check_or_abort(
                cudaMemcpy(ebisu_workspace.face_tile_lookup,
                           face_tile_lookup_host.data(),
                           face_tile_lookup_host.size() * sizeof(int),
                           cudaMemcpyHostToDevice),
                __FILE__, __LINE__);
            if (!full_tile_lookup_host.empty()) {
              tide::cuda_check_or_abort(
                  cudaMemcpy(ebisu_workspace.full_tile_lookup,
                             full_tile_lookup_host.data(),
                             full_tile_lookup_host.size() * sizeof(int),
                             cudaMemcpyHostToDevice),
                  __FILE__, __LINE__);
            }
            int64_t const tiles_x =
                (domain_x + ebisu_cfg.tile_x - 1) / ebisu_cfg.tile_x;
            tide::cuda_check_or_abort(
                launch_tm_ebisu_single_io_lookup_kernel(
                    n_shots_h,
                    use_single_source_fast ? sources_i : nullptr,
                    use_single_receiver_fast ? receivers_i : nullptr,
                    ebisu_workspace.no_pml_tile_lookup,
                    ebisu_workspace.face_tile_lookup,
                    ebisu_workspace.full_tile_lookup,
                    ebisu_workspace.single_source_path,
                    ebisu_workspace.single_source_block,
                    ebisu_workspace.single_source_li,
                    ebisu_workspace.single_receiver_path,
                    ebisu_workspace.single_receiver_block,
                    ebisu_workspace.single_receiver_li, ebisu_cfg.tile_y,
                    ebisu_cfg.tile_x, ebisu_cfg.steps, tiles_x),
                __FILE__, __LINE__);
          }

          TIDE_DTYPE *ey_curr = ey;
          TIDE_DTYPE *hx_curr = hx;
          TIDE_DTYPE *hz_curr = hz;
          TIDE_DTYPE *m_ey_x_curr = m_ey_x;
          TIDE_DTYPE *m_ey_z_curr = m_ey_z;
          TIDE_DTYPE *m_hx_z_curr = m_hx_z;
          TIDE_DTYPE *m_hz_x_curr = m_hz_x;
          TIDE_DTYPE *ey_next = ebisu_workspace.ey;
          TIDE_DTYPE *hx_next = ebisu_workspace.hx;
          TIDE_DTYPE *hz_next = ebisu_workspace.hz;
          TIDE_DTYPE *m_ey_x_next = ebisu_workspace.m_ey_x;
          TIDE_DTYPE *m_ey_z_next = ebisu_workspace.m_ey_z;
          TIDE_DTYPE *m_hx_z_next = ebisu_workspace.m_hx_z;
          TIDE_DTYPE *m_hz_x_next = ebisu_workspace.m_hz_x;

          int64_t remaining = nt;
          int64_t t = start_t;
          while (remaining > 0) {
            int64_t const chunk_steps =
                tide_min<int64_t>(remaining, ebisu_cfg.steps);
            bool const use_compact_tiles = chunk_steps == ebisu_cfg.steps;
            int64_t const chunk_tile_x = ebisu_cfg.tile_x + 2 * chunk_steps;
            int64_t const chunk_tile_y = ebisu_cfg.tile_y + 2 * chunk_steps;
            size_t const chunk_shared_no_pml =
                (size_t)chunk_tile_x * (size_t)chunk_tile_y * 4 *
                sizeof(TIDE_DTYPE);
            size_t const chunk_shared_face =
                (size_t)chunk_tile_x * (size_t)chunk_tile_y * 4 *
                sizeof(TIDE_DTYPE);
            size_t const chunk_shared_full =
                (size_t)chunk_tile_x * (size_t)chunk_tile_y * 8 *
                sizeof(TIDE_DTYPE);
            dim3 const no_pml_grid =
                use_compact_tiles
                    ? dim3(to_dim_u32((int64_t)no_pml_tiles_host.size()), 1,
                           to_dim_u32(n_shots_h))
                    : ebisu_grid;
            dim3 const full_grid =
                use_compact_tiles
                    ? dim3(to_dim_u32((int64_t)full_tiles_host.size()), 1,
                           to_dim_u32(n_shots_h))
                    : ebisu_grid;
            dim3 const face_pml_grid =
                use_compact_tiles
                    ? dim3(to_dim_u32((int64_t)face_pml_tiles_host.size()), 1,
                           to_dim_u32(n_shots_h))
                    : dim3(1, 1, 1);
            int2 const *const no_pml_tiles =
                use_compact_tiles ? ebisu_workspace.no_pml_tiles : nullptr;
            int2 const *const face_pml_tiles =
                use_compact_tiles ? ebisu_workspace.y_pml_tiles : nullptr;
            int2 const *const full_tiles =
                use_compact_tiles ? ebisu_workspace.full_tiles : nullptr;
            int const *const single_source_path =
                (use_compact_tiles && use_single_source_fast)
                    ? ebisu_workspace.single_source_path
                    : nullptr;
            int const *const single_source_block =
                (use_compact_tiles && use_single_source_fast)
                    ? ebisu_workspace.single_source_block
                    : nullptr;
            int const *const single_source_li =
                (use_compact_tiles && use_single_source_fast)
                    ? ebisu_workspace.single_source_li
                    : nullptr;
            int const *const single_receiver_path =
                (use_compact_tiles && use_single_receiver_fast)
                    ? ebisu_workspace.single_receiver_path
                    : nullptr;
            int const *const single_receiver_block =
                (use_compact_tiles && use_single_receiver_fast)
                    ? ebisu_workspace.single_receiver_block
                    : nullptr;
            int const *const single_receiver_li =
                (use_compact_tiles && use_single_receiver_fast)
                    ? ebisu_workspace.single_receiver_li
                    : nullptr;
            TIDE_DTYPE const *const f_t =
                (n_sources_per_shot_h > 0 && f != nullptr)
                    ? (f + t * n_shots_h * n_sources_per_shot_h)
                    : nullptr;
            TIDE_DTYPE *const r_t =
                (n_receivers_per_shot_h > 0 && r != nullptr)
                    ? (r + t * n_shots_h * n_receivers_per_shot_h)
                    : nullptr;
            cudaError_t launch_err = cudaSuccess;
            switch (ebisu_cfg.ilp) {
            case 1:
              if (!use_compact_tiles || !no_pml_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_no_pml_oop_kernel<1>(
                    no_pml_grid, ebisu_block, chunk_shared_no_pml, ca, cb, cq,
                    f_t, ey_curr, hx_curr, hz_curr, ey_next, hx_next, hz_next,
                    single_source_path, single_source_block, single_source_li,
                    single_receiver_path, single_receiver_block,
                    single_receiver_li,
                    sources_i, receivers_i, r_t, ebisu_cfg.tile_y,
                    ebisu_cfg.tile_x, chunk_steps, chunk_steps, no_pml_tiles);
                tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
              }
              if (use_face_pml_specialization && use_compact_tiles &&
                  !face_pml_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_top_pml_y_oop_kernel<1>(
                    face_pml_grid, ebisu_block, chunk_shared_face, ca,
                    cb, cq, f_t, ey_curr, hx_curr, hz_curr, m_ey_x_curr,
                    m_ey_z_curr, m_hx_z_curr, m_hz_x_curr, ey_next, hx_next,
                    hz_next, m_ey_x_next, m_ey_z_next, m_hx_z_next,
                    m_hz_x_next, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
                    kyh, kx, kxh, single_source_path, single_source_block,
                    single_source_li, single_receiver_path,
                    single_receiver_block, single_receiver_li, sources_i,
                    receivers_i, r_t, ebisu_cfg.tile_y, ebisu_cfg.tile_x,
                    chunk_steps, chunk_steps, ebisu_cfg.face_async_copy,
                    face_pml_tiles);
                tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
              }
              if (!use_compact_tiles || !full_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_full_oop_kernel<1>(
                    full_grid, ebisu_block, chunk_shared_full, ca, cb, cq, f_t,
                    ey_curr, hx_curr, hz_curr, m_ey_x_curr, m_ey_z_curr,
                    m_hx_z_curr, m_hz_x_curr, ey_next, hx_next, hz_next,
                    m_ey_x_next, m_ey_z_next, m_hx_z_next, m_hz_x_next, ay,
                    ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh,
                    single_source_path, single_source_block, single_source_li,
                    single_receiver_path, single_receiver_block,
                    single_receiver_li,
                    sources_i, receivers_i, r_t, ebisu_cfg.tile_y,
                    ebisu_cfg.tile_x, chunk_steps, chunk_steps, full_tiles);
              }
              break;
            case 2:
              if (!use_compact_tiles || !no_pml_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_no_pml_oop_kernel<2>(
                    no_pml_grid, ebisu_block, chunk_shared_no_pml, ca, cb, cq,
                    f_t, ey_curr, hx_curr, hz_curr, ey_next, hx_next, hz_next,
                    single_source_path, single_source_block, single_source_li,
                    single_receiver_path, single_receiver_block,
                    single_receiver_li,
                    sources_i, receivers_i, r_t, ebisu_cfg.tile_y,
                    ebisu_cfg.tile_x, chunk_steps, chunk_steps, no_pml_tiles);
                tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
              }
              if (use_face_pml_specialization && use_compact_tiles &&
                  !face_pml_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_top_pml_y_oop_kernel<2>(
                    face_pml_grid, ebisu_block, chunk_shared_face, ca,
                    cb, cq, f_t, ey_curr, hx_curr, hz_curr, m_ey_x_curr,
                    m_ey_z_curr, m_hx_z_curr, m_hz_x_curr, ey_next, hx_next,
                    hz_next, m_ey_x_next, m_ey_z_next, m_hx_z_next,
                    m_hz_x_next, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
                    kyh, kx, kxh, single_source_path, single_source_block,
                    single_source_li, single_receiver_path,
                    single_receiver_block, single_receiver_li, sources_i,
                    receivers_i, r_t, ebisu_cfg.tile_y, ebisu_cfg.tile_x,
                    chunk_steps, chunk_steps, ebisu_cfg.face_async_copy,
                    face_pml_tiles);
                tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
              }
              if (!use_compact_tiles || !full_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_full_oop_kernel<2>(
                    full_grid, ebisu_block, chunk_shared_full, ca, cb, cq, f_t,
                    ey_curr, hx_curr, hz_curr, m_ey_x_curr, m_ey_z_curr,
                    m_hx_z_curr, m_hz_x_curr, ey_next, hx_next, hz_next,
                    m_ey_x_next, m_ey_z_next, m_hx_z_next, m_hz_x_next, ay,
                    ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh,
                    single_source_path, single_source_block, single_source_li,
                    single_receiver_path, single_receiver_block,
                    single_receiver_li,
                    sources_i, receivers_i, r_t, ebisu_cfg.tile_y,
                    ebisu_cfg.tile_x, chunk_steps, chunk_steps, full_tiles);
              }
              break;
            default:
              if (!use_compact_tiles || !no_pml_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_no_pml_oop_kernel<4>(
                    no_pml_grid, ebisu_block, chunk_shared_no_pml, ca, cb, cq,
                    f_t, ey_curr, hx_curr, hz_curr, ey_next, hx_next, hz_next,
                    single_source_path, single_source_block, single_source_li,
                    single_receiver_path, single_receiver_block,
                    single_receiver_li,
                    sources_i, receivers_i, r_t, ebisu_cfg.tile_y,
                    ebisu_cfg.tile_x, chunk_steps, chunk_steps, no_pml_tiles);
                tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
              }
              if (use_face_pml_specialization && use_compact_tiles &&
                  !face_pml_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_top_pml_y_oop_kernel<4>(
                    face_pml_grid, ebisu_block, chunk_shared_face, ca,
                    cb, cq, f_t, ey_curr, hx_curr, hz_curr, m_ey_x_curr,
                    m_ey_z_curr, m_hx_z_curr, m_hz_x_curr, ey_next, hx_next,
                    hz_next, m_ey_x_next, m_ey_z_next, m_hx_z_next,
                    m_hz_x_next, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
                    kyh, kx, kxh, single_source_path, single_source_block,
                    single_source_li, single_receiver_path,
                    single_receiver_block, single_receiver_li, sources_i,
                    receivers_i, r_t, ebisu_cfg.tile_y, ebisu_cfg.tile_x,
                    chunk_steps, chunk_steps, ebisu_cfg.face_async_copy,
                    face_pml_tiles);
                tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
              }
              if (!use_compact_tiles || !full_tiles_host.empty()) {
                launch_err = launch_tm_ebisu_full_oop_kernel<4>(
                    full_grid, ebisu_block, chunk_shared_full, ca, cb, cq, f_t,
                    ey_curr, hx_curr, hz_curr, m_ey_x_curr, m_ey_z_curr,
                    m_hx_z_curr, m_hz_x_curr, ey_next, hx_next, hz_next,
                    m_ey_x_next, m_ey_z_next, m_hx_z_next, m_hz_x_next, ay,
                    ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh,
                    single_source_path, single_source_block, single_source_li,
                    single_receiver_path, single_receiver_block,
                    single_receiver_li,
                    sources_i, receivers_i, r_t, ebisu_cfg.tile_y,
                    ebisu_cfg.tile_x, chunk_steps, chunk_steps, full_tiles);
              }
              break;
            }
            tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
            TIDE_DTYPE *tmp = ey_curr;
            ey_curr = ey_next;
            ey_next = tmp;
            tmp = hx_curr;
            hx_curr = hx_next;
            hx_next = tmp;
            tmp = hz_curr;
            hz_curr = hz_next;
            hz_next = tmp;
            tmp = m_ey_x_curr;
            m_ey_x_curr = m_ey_x_next;
            m_ey_x_next = tmp;
            tmp = m_ey_z_curr;
            m_ey_z_curr = m_ey_z_next;
            m_ey_z_next = tmp;
            tmp = m_hx_z_curr;
            m_hx_z_curr = m_hx_z_next;
            m_hx_z_next = tmp;
            tmp = m_hz_x_curr;
            m_hz_x_curr = m_hz_x_next;
            m_hz_x_next = tmp;
            remaining -= chunk_steps;
            t += chunk_steps;
          }
          if (ey_curr != ey) {
            tide::cuda_check_or_abort(
                cudaMemcpy(ey, ey_curr, field_bytes, cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaMemcpy(hx, hx_curr, field_bytes, cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaMemcpy(hz, hz_curr, field_bytes, cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaMemcpy(m_ey_x, m_ey_x_curr, field_bytes,
                           cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaMemcpy(m_ey_z, m_ey_z_curr, field_bytes,
                           cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaMemcpy(m_hx_z, m_hx_z_curr, field_bytes,
                           cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
            tide::cuda_check_or_abort(
                cudaMemcpy(m_hz_x, m_hz_x_curr, field_bytes,
                           cudaMemcpyDeviceToDevice),
                __FILE__, __LINE__);
          }
          if (debug_path) {
            std::fprintf(stderr,
                         "TIDE TM path: ebisu-hybrid steps=%lld tile=%lldx%lld ilp=%lld "
                         "shared_no_pml=%zuB shared_face=%zuB shared_full=%zuB "
                         "no_pml_tiles=%zu face_tiles=%zu y_pml_tiles=%zu x_pml_tiles=%zu full_tiles=%zu\n",
                         (long long)ebisu_cfg.steps, (long long)ebisu_cfg.tile_x,
                         (long long)ebisu_cfg.tile_y, (long long)ebisu_cfg.ilp,
                         shared_bytes_no_pml, shared_bytes_face,
                         shared_bytes_full, no_pml_tiles_host.size(),
                         face_pml_tiles_host.size(),
                         y_pml_tiles_host.size(), x_pml_tiles_host.size(),
                         full_tiles_host.size());
          }
          tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
          return;
        }
        if (debug_path) {
          std::fprintf(stderr,
                       "TIDE TM ebisu-hybrid fallback: occupancy no_pml=%d y_face=%d x_face=%d full=%d\n",
                       max_blocks_per_sm_no_pml, max_blocks_per_sm_y_pml,
                       max_blocks_per_sm_x_pml, max_blocks_per_sm_full);
        }
      } else if (debug_path) {
        std::fprintf(stderr,
                     "TIDE TM ebisu-hybrid fallback: shared_no_pml=%zuB shared_face=%zuB shared_full=%zuB max_optin=%dB tile=%lldx%lld halo=%lld\n",
                     shared_bytes_no_pml, shared_bytes_face,
                     shared_bytes_full, max_optin_shared, (long long)tile_x,
                     (long long)tile_y, (long long)halo);
      }
    } else if (debug_path && ebisu_cfg.enabled) {
      std::fprintf(stderr,
                   "TIDE TM ebisu fallback: unsupported workload "
                   "disp=%d src=%lld rec=%lld pml=[%lld,%lld,%lld,%lld] ny=%lld nx=%lld\n",
                   has_dispersion ? 1 : 0, (long long)n_sources_per_shot_h,
                   (long long)n_receivers_per_shot_h, (long long)pml_y0_h,
                   (long long)pml_x0_h, (long long)pml_y1_h, (long long)pml_x1_h,
                   (long long)ny_h, (long long)nx_h);
    }
  }

  TMFusedRuntimeConfig const fused_cfg = read_tm_fused_runtime_config();
  bool const try_fused_no_pml = can_use_tm_fused_no_pml_path(
      fused_cfg, has_dispersion, n_sources_per_shot_h, n_receivers_per_shot_h,
      pml_y0_h, pml_x0_h, pml_y1_h, pml_x1_h, ny_h, nx_h);
  if (try_fused_no_pml) {
    int cooperative_launch = 0;
    tide::cuda_check_or_abort(
        cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch,
                               device),
        __FILE__, __LINE__);
    if (cooperative_launch != 0) {
      int sm_count = 0;
      tide::cuda_check_or_abort(
          cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device),
          __FILE__, __LINE__);
      dim3 const fused_block((unsigned)fused_cfg.threads, 1, 1);
      int max_blocks_per_sm = 0;
      cudaError_t occ_err = cudaSuccess;
      switch (fused_cfg.ilp) {
      case 1:
        occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, forward_kernel_fused_ksteps_no_pml<1>,
            (int)fused_block.x, 0);
        break;
      case 2:
        occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, forward_kernel_fused_ksteps_no_pml<2>,
            (int)fused_block.x, 0);
        break;
      default:
        occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, forward_kernel_fused_ksteps_no_pml<4>,
            (int)fused_block.x, 0);
        break;
      }
      tide::cuda_check_or_abort(occ_err, __FILE__, __LINE__);
      int64_t const active_blocks_per_sm =
          tide_max<int64_t>(1, tide_min<int64_t>(fused_cfg.blocks_per_sm,
                                                 (int64_t)max_blocks_per_sm));
      int64_t const grid_blocks =
          tide_max<int64_t>(1, (int64_t)sm_count * active_blocks_per_sm);

      int64_t remaining = nt;
      while (remaining > 0) {
        int64_t const chunk_steps =
            tide_min<int64_t>(remaining, fused_cfg.steps);
        cudaError_t launch_err = cudaSuccess;
        switch (fused_cfg.ilp) {
        case 1:
          launch_err = launch_tm_fused_no_pml_kernel<1>(
              fused_block, grid_blocks, ca, cb, cq, ey, hx, hz, ay, ayh, ax, axh,
              by, byh, bx, bxh, ky, kyh, kx, kxh, chunk_steps);
          break;
        case 2:
          launch_err = launch_tm_fused_no_pml_kernel<2>(
              fused_block, grid_blocks, ca, cb, cq, ey, hx, hz, ay, ayh, ax, axh,
              by, byh, bx, bxh, ky, kyh, kx, kxh, chunk_steps);
          break;
        default:
          launch_err = launch_tm_fused_no_pml_kernel<4>(
              fused_block, grid_blocks, ca, cb, cq, ey, hx, hz, ay, ayh, ax, axh,
              by, byh, bx, bxh, ky, kyh, kx, kxh, chunk_steps);
          break;
        }
        tide::cuda_check_or_abort(launch_err, __FILE__, __LINE__);
        remaining -= chunk_steps;
      }
      if (debug_path) {
        std::fprintf(stderr,
                     "TIDE TM path: fused steps=%lld threads=%lld ilp=%lld "
                     "blocks_per_sm=%lld\n",
                     (long long)fused_cfg.steps, (long long)fused_cfg.threads,
                     (long long)fused_cfg.ilp,
                     (long long)fused_cfg.blocks_per_sm);
      }
      tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
      return;
    }
  }

  if (debug_path) {
    std::fprintf(stderr, "TIDE TM path: baseline\n");
  }

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);
    if (has_dispersion) {
      forward_kernel_e_debye<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_prev, debye_cp, polarization,
          n_poles_h, ay, ayh, ax, axh, by, byh, bx, bxh, ky, kyh, kx, kxh);
    } else {
      forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (has_dispersion) {
      update_polarization_debye<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
          ey_prev, ey, debye_a, debye_b, polarization, n_poles_h);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<launch_cfg.dimGridReceivers,
                            launch_cfg.dimBlockReceivers>>>(
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
    int64_t const pml_x1_h, int64_t const n_threads, int64_t const device) {

  cudaSetDevice(device);
  (void)n_threads;

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
  bool const storage_bf16_h =
      (!kFieldIsHalf) && (storage_format_h == STORAGE_FORMAT_BF16);
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

  auto run_step = [&](int64_t t) {
    forward_kernel_h<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
        cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
        kyh, kx, kxh);

    bool const store_step = ((t % step_ratio_h) == 0);
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    bool const want_store = store_ey || store_curl;
    if (want_store) {
      int64_t const step_idx = t / step_ratio_h;
      size_t const store1_offset = ring_storage_offset_bytes(
          step_idx, storage_mode_h, bytes_per_step_store);

      void *__restrict const ey_store_1_t =
          (uint8_t *)ey_store_1 + store1_offset;
      void *__restrict const ey_store_3_t =
          (uint8_t *)ey_store_3 + (storage_mode_h == STORAGE_CPU
                                       ? (size_t)step_idx * bytes_per_step_store
                                       : 0);

      void *__restrict const curl_store_1_t =
          (uint8_t *)curl_store_1 + store1_offset;
      void *__restrict const curl_store_3_t =
          (uint8_t *)curl_store_3 +
          (storage_mode_h == STORAGE_CPU
               ? (size_t)step_idx * bytes_per_step_store
               : 0);

      if (storage_bf16_h) {
        forward_kernel_e_with_storage_bf16<<<launch_cfg.dimGrid,
                                             launch_cfg.dimBlock>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (__nv_bfloat16 *)ey_store_1_t : nullptr,
            store_curl ? (__nv_bfloat16 *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      } else {
        forward_kernel_e_with_storage<<<launch_cfg.dimGrid,
                                        launch_cfg.dimBlock>>>(
            ca, cb, hx, hz, ey, m_hx_z, m_hz_x,
            store_ey ? (TIDE_DTYPE *)ey_store_1_t : nullptr,
            store_curl ? (TIDE_DTYPE *)curl_store_1_t : nullptr, ay, ayh, ax,
            axh, by, byh, bx, bxh, ky, kyh, kx, kxh, store_ey, store_curl);
      }

      if (storage_mode_h == STORAGE_CPU) {
        if (store_ey) {
          tide::cuda_check_or_abort(
              cudaMemcpy(ey_store_3_t, ey_store_1_t, bytes_per_step_store,
                         cudaMemcpyDeviceToHost),
              __FILE__, __LINE__);
        }
        if (store_curl) {
          tide::cuda_check_or_abort(
              cudaMemcpy(curl_store_3_t, curl_store_1_t, bytes_per_step_store,
                         cudaMemcpyDeviceToHost),
              __FILE__, __LINE__);
        }
      } else {
        if (store_ey) {
          storage_save_snapshot_gpu(
              ey_store_1_t, ey_store_3_t, fp_ey, storage_mode_h, step_idx,
              (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
        }
        if (store_curl) {
          storage_save_snapshot_gpu(
              curl_store_1_t, curl_store_3_t, fp_curl, storage_mode_h, step_idx,
              (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
        }
      }
    } else {
      forward_kernel_e<<<launch_cfg.dimGrid, launch_cfg.dimBlock>>>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh);
    }

    if (n_sources_per_shot_h > 0) {
      add_sources_ey<<<launch_cfg.dimGridSources, launch_cfg.dimBlockSources>>>(
          ey, f + t * n_shots_h * n_sources_per_shot_h, sources_i);
    }

    if (n_receivers_per_shot_h > 0) {
      record_receivers_ey<<<launch_cfg.dimGridReceivers,
                            launch_cfg.dimBlockReceivers>>>(
          r + t * n_shots_h * n_receivers_per_shot_h, ey, receivers_i);
    }
  };

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    run_step(t);
  }

  if (fp_ey != nullptr)
    fclose(fp_ey);
  if (fp_curl != nullptr)
    fclose(fp_curl);

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
    int64_t const device) {

  cudaSetDevice(device);
  (void)dt_h;
  (void)n_threads;

  int64_t const shot_numel_h = ny_h * nx_h;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp_h * (size_t)n_shots_h;
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

  FILE *fp_ey = nullptr;
  FILE *fp_curl = nullptr;
  if (storage_mode_h == STORAGE_DISK) {
    if (ca_requires_grad)
      fp_ey = fopen(ey_filenames[0], "rb");
    if (cb_requires_grad)
      fp_curl = fopen(curl_filenames[0], "rb");
  }

  // Time reversed loop
  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    // Inject adjoint source (receiver residual) at receiver locations
    // Use add_adjoint_sources_ey which checks n_receivers_per_shot
    if (n_receivers_per_shot_h > 0) {
      add_adjoint_sources_ey<<<launch_cfg.dimGridReceivers,
                               launch_cfg.dimBlockReceivers>>>(
          lambda_ey, grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

    // Record adjoint field at source locations for source gradient
    // Use record_adjoint_at_sources which checks n_sources_per_shot
    if (n_sources_per_shot_h > 0) {
      record_adjoint_at_sources<<<launch_cfg.dimGridSources,
                                  launch_cfg.dimBlockSources>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h, lambda_ey, sources_i);
    }

    int64_t const store_idx = t / step_ratio_h;
    bool const do_grad = (t % step_ratio_h) == 0;
    bool const grad_ey = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;

    size_t const store1_offset = ring_storage_offset_bytes(
        store_idx, storage_mode_h, bytes_per_step_store);
    size_t const store3_offset = cpu_linear_storage_offset_bytes(
        store_idx, storage_mode_h, bytes_per_step_store);

    void *__restrict const ey_store_1_t = (uint8_t *)ey_store_1 + store1_offset;
    void *__restrict const ey_store_3_t = (uint8_t *)ey_store_3 + store3_offset;

    void *__restrict const curl_store_1_t =
        (uint8_t *)curl_store_1 + store1_offset;
    void *__restrict const curl_store_3_t =
        (uint8_t *)curl_store_3 + store3_offset;

    if (storage_mode_h == STORAGE_CPU && (grad_ey || grad_curl)) {
      if (grad_ey) {
        tide::cuda_check_or_abort(
            cudaMemcpy((void *)ey_store_1_t, (void *)ey_store_3_t,
                       bytes_per_step_store, cudaMemcpyHostToDevice),
            __FILE__, __LINE__);
      }
      if (grad_curl) {
        tide::cuda_check_or_abort(
            cudaMemcpy((void *)curl_store_1_t, (void *)curl_store_3_t,
                       bytes_per_step_store, cudaMemcpyHostToDevice),
            __FILE__, __LINE__);
      }
    } else if (storage_mode_h == STORAGE_DISK) {
      if (grad_ey) {
        storage_load_snapshot_gpu(
            (void *)ey_store_1_t, (void *)ey_store_3_t, fp_ey, storage_mode_h,
            store_idx, (size_t)shot_bytes_uncomp_h, (size_t)n_shots_h);
      }
      if (grad_curl) {
        storage_load_snapshot_gpu(
            (void *)curl_store_1_t, (void *)curl_store_3_t, fp_curl,
            storage_mode_h, store_idx, (size_t)shot_bytes_uncomp_h,
            (size_t)n_shots_h);
      }
    }

    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_h_update_m_exact<<<dimGridBoundary,
                                                dimBlockBoundary>>>(
          cb, lambda_ey, m_lambda_ey_x, m_lambda_ey_z, by, bx,
          boundary_layout);
    }
    if (has_interior) {
      backward_kernel_lambda_h_apply_exact_interior<<<dimGridInterior,
                                                      dimBlock>>>(
          cb, lambda_ey, lambda_hx, lambda_hz, interior_y_begin, interior_y_end,
          interior_x_begin, interior_x_end);
    }
    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_h_apply_exact_boundary<<<dimGridBoundary,
                                                      dimBlockBoundary>>>(
          cb, lambda_ey, m_lambda_ey_x, m_lambda_ey_z, lambda_hx, lambda_hz,
          ay, ax, by, bx, ky, kx, boundary_layout);
    }

    if (boundary_layout.total_count > 0) {
      backward_kernel_lambda_e_update_m_exact<<<dimGridBoundary,
                                                dimBlockBoundary>>>(
          cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x, byh, bxh,
          boundary_layout);
    }

    if (grad_ey || grad_curl) {
      if (storage_bf16_h) {
        if (has_interior) {
          backward_kernel_lambda_e_apply_exact_bf16_interior<<<dimGridInterior,
                                                                dimBlock>>>(
              ca, cq, lambda_hx, lambda_hz, lambda_ey,
              grad_ey ? (__nv_bfloat16 const *)ey_store_1_t : nullptr,
              grad_curl ? (__nv_bfloat16 const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h,
              interior_y_begin, interior_y_end, interior_x_begin,
              interior_x_end);
        }
        if (boundary_layout.total_count > 0) {
          backward_kernel_lambda_e_apply_exact_bf16_boundary
              <<<dimGridBoundary, dimBlockBoundary>>>(
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
                                                           dimBlock>>>(
              ca, cq, lambda_hx, lambda_hz, lambda_ey,
              grad_ey ? (TIDE_DTYPE const *)ey_store_1_t : nullptr,
              grad_curl ? (TIDE_DTYPE const *)curl_store_1_t : nullptr,
              grad_ca_shot, grad_cb_shot, grad_ey, grad_curl, step_ratio_h,
              interior_y_begin, interior_y_end, interior_x_begin,
              interior_x_end);
        }
        if (boundary_layout.total_count > 0) {
          backward_kernel_lambda_e_apply_exact_boundary<<<dimGridBoundary,
                                                          dimBlockBoundary>>>(
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
                                                                dimBlock>>>(
            ca, cq, lambda_hx, lambda_hz, lambda_ey, interior_y_begin,
            interior_y_end, interior_x_begin, interior_x_end);
      }
      if (boundary_layout.total_count > 0) {
        backward_kernel_lambda_e_apply_exact_boundary_nograd<<<dimGridBoundary,
                                                                dimBlockBoundary>>>(
            ca, cq, lambda_hx, lambda_hz, m_lambda_hx_z, m_lambda_hz_x,
            lambda_ey, ayh, axh, byh, bxh, kyh, kxh, boundary_layout);
      }
    }

  }

  if (fp_ey != nullptr)
    fclose(fp_ey);
  if (fp_curl != nullptr)
    fclose(fp_curl);

  // Combine per-shot gradients (only if not batched - batched case keeps
  // per-shot grads)
  dim3 dimBlock_combine(32, 32, 1);
  dim3 dimGrid_combine(
      (nx_h - 2 * kFdPad + dimBlock_combine.x - 1) / dimBlock_combine.x,
      (ny_h - 2 * kFdPad + dimBlock_combine.y - 1) / dimBlock_combine.y, 1);

  if (ca_requires_grad && !ca_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_ca, grad_ca_shot);
  }
  if (cb_requires_grad && !cb_batched_h) {
    combine_grad<<<dimGrid_combine, dimBlock_combine>>>(grad_cb, grad_cb_shot);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

} // namespace FUNC(Inst)
