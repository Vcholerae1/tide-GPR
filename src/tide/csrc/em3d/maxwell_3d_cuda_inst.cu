/*
 * Maxwell 3D CUDA backend implementation.
 *
 * This file provides CUDA forward and backward propagation kernels for the
 * 3D Maxwell staggered-grid + CPML scheme.
 */

#undef DIFFZ1
#undef DIFFY1
#undef DIFFX1
#undef DIFFZH1
#undef DIFFYH1
#undef DIFFXH1
#undef FD_PAD

#ifdef STAGGERED_GRID_H
#undef STAGGERED_GRID_H
#endif
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_bf16.h>
#define TIDE_STAGGERED_GRID_3D 1
#include "staggered_grid.h"
#undef TIDE_STAGGERED_GRID_3D

#define ND_INDEX(i, dz, dy, dx) ((i) + (dz) * ny * nx + (dy) * nx + (dx))
#define ND_INDEX_J(j, dz, dy, dx) ((j) + (dz) * ny * nx + (dy) * nx + (dx))

#define EX(dz, dy, dx) ex[ND_INDEX(i, dz, dy, dx)]
#define EY(dz, dy, dx) ey[ND_INDEX(i, dz, dy, dx)]
#define EZ(dz, dy, dx) ez[ND_INDEX(i, dz, dy, dx)]
#define HX(dz, dy, dx) hx[ND_INDEX(i, dz, dy, dx)]
#define HY(dz, dy, dx) hy[ND_INDEX(i, dz, dy, dx)]
#define HZ(dz, dy, dx) hz[ND_INDEX(i, dz, dy, dx)]
#define DEX(dz, dy, dx) dex[ND_INDEX(i, dz, dy, dx)]
#define DEY(dz, dy, dx) dey[ND_INDEX(i, dz, dy, dx)]
#define DEZ(dz, dy, dx) dez[ND_INDEX(i, dz, dy, dx)]
#define DHX(dz, dy, dx) dhx[ND_INDEX(i, dz, dy, dx)]
#define DHY(dz, dy, dx) dhy[ND_INDEX(i, dz, dy, dx)]
#define DHZ(dz, dy, dx) dhz[ND_INDEX(i, dz, dy, dx)]
#define CB_I(dz, dy, dx) \
  (cb_batched ? cb[ND_INDEX(i, dz, dy, dx)] : cb[ND_INDEX_J(j, dz, dy, dx)])
#define CQ_I(dz, dy, dx) \
  (cq_batched ? cq[ND_INDEX(i, dz, dy, dx)] : cq[ND_INDEX_J(j, dz, dy, dx)])

namespace FUNC(Inst) {

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

static inline size_t ring_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode,
    size_t const bytes_per_step_store) {
  if (storage_mode == STORAGE_DEVICE) {
    return static_cast<size_t>(step_idx) * bytes_per_step_store;
  }
  if (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) {
    return static_cast<size_t>(step_idx % NUM_BUFFERS) * bytes_per_step_store;
  }
  return 0;
}

static inline size_t cpu_linear_storage_offset_bytes(
    int64_t const step_idx, int64_t const storage_mode,
    size_t const bytes_per_step_store) {
  if (storage_mode == STORAGE_CPU) {
    return static_cast<size_t>(step_idx) * bytes_per_step_store;
  }
  if (storage_mode == STORAGE_DISK) {
    return static_cast<size_t>(step_idx % NUM_BUFFERS) * bytes_per_step_store;
  }
  return 0;
}

template <typename T>
__host__ __device__ __forceinline__ T tide_max(T a, T b) {
  return a > b ? a : b;
}

constexpr TIDE_DTYPE kEp0 = (TIDE_DTYPE)8.8541878128e-12;

__constant__ TIDE_DTYPE rdz;
__constant__ TIDE_DTYPE rdy;
__constant__ TIDE_DTYPE rdx;

__constant__ int64_t n_shots;
__constant__ int64_t nz;
__constant__ int64_t ny;
__constant__ int64_t nx;
__constant__ int64_t shot_numel;
__constant__ int64_t n_sources_per_shot;
__constant__ int64_t n_receivers_per_shot;

__constant__ int64_t pml_z0;
__constant__ int64_t pml_y0;
__constant__ int64_t pml_x0;
__constant__ int64_t pml_z1;
__constant__ int64_t pml_y1;
__constant__ int64_t pml_x1;

__constant__ bool ca_batched;
__constant__ bool cb_batched;
__constant__ bool cq_batched;
__constant__ TIDE_DTYPE ca_uniform_coeff;
__constant__ TIDE_DTYPE cb_uniform_coeff;
__constant__ TIDE_DTYPE cq_uniform_coeff;

template <typename T>
__device__ __forceinline__ void atomic_add_tide(T *addr, T val) {
  if constexpr (std::is_same<T, double>::value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    atomicAdd(addr, val);
#else
    unsigned long long int *address_as_ull = (unsigned long long int *)addr;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
      assumed = old;
      old = atomicCAS(
          address_as_ull,
          assumed,
          __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
  } else {
    static_assert(std::is_same<T, float>::value,
                  "atomic_add_tide only supports float/double");
    atomicAdd(addr, val);
  }
}

template <typename SnapshotT>
__device__ __forceinline__ SnapshotT encode_snapshot(TIDE_DTYPE value);

template <>
__device__ __forceinline__ TIDE_DTYPE encode_snapshot<TIDE_DTYPE>(
    TIDE_DTYPE value) {
  return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 encode_snapshot<__nv_bfloat16>(
    TIDE_DTYPE value) {
  return __float2bfloat16(static_cast<float>(value));
}

template <typename SnapshotT>
__device__ __forceinline__ TIDE_DTYPE decode_snapshot(SnapshotT value);

template <>
__device__ __forceinline__ TIDE_DTYPE decode_snapshot<TIDE_DTYPE>(
    TIDE_DTYPE value) {
  return value;
}

template <>
__device__ __forceinline__ TIDE_DTYPE decode_snapshot<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return static_cast<TIDE_DTYPE>(__bfloat162float(value));
}

__global__ void add_sources_component(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const f,
    int64_t const *__restrict const sources_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_sources_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const src = sources_i[idx];
  if (src >= 0) {
    int64_t const shot_idx = idx / n_sources_per_shot;
    field[shot_idx * shot_numel + src] += f[idx];
  }
}

__global__ void record_receivers_component(
    TIDE_DTYPE *__restrict const r,
    TIDE_DTYPE const *__restrict const field,
    int64_t const *__restrict const receivers_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_receivers_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const rec = receivers_i[idx];
  if (rec >= 0) {
    int64_t const shot_idx = idx / n_receivers_per_shot;
    r[idx] = field[shot_idx * shot_numel + rec];
  } else {
    r[idx] = (TIDE_DTYPE)0;
  }
}

template <typename T>
__host__ __device__ __forceinline__ T tide_min(T a, T b) {
  return a < b ? a : b;
}

struct LinearCellIndex3D {
  int shot_idx;
  int j;
  int z;
  int y;
  int x;
  int nz_i;
  int ny_i;
  int nx_i;
};

__device__ __forceinline__ LinearCellIndex3D decode_linear_cell_index_3d(
    int64_t const linear_idx) {
  LinearCellIndex3D idx{};
  idx.nz_i = static_cast<int>(nz);
  idx.ny_i = static_cast<int>(ny);
  idx.nx_i = static_cast<int>(nx);
  int64_t const shot_idx64 = linear_idx / shot_numel;
  int64_t const j64 = linear_idx - shot_idx64 * shot_numel;
  idx.shot_idx = static_cast<int>(shot_idx64);
  idx.j = static_cast<int>(j64);
  int const yz_stride = idx.ny_i * idx.nx_i;
  idx.z = idx.j / yz_stride;
  int const rem = idx.j - idx.z * yz_stride;
  idx.y = rem / idx.nx_i;
  idx.x = rem - idx.y * idx.nx_i;
  return idx;
}

__device__ __forceinline__ LinearCellIndex3D decode_linear_cell_index_3d_i32(
    int const linear_idx) {
  LinearCellIndex3D idx{};
  idx.nz_i = static_cast<int>(nz);
  idx.ny_i = static_cast<int>(ny);
  idx.nx_i = static_cast<int>(nx);
  int const shot_numel_i = idx.nz_i * idx.ny_i * idx.nx_i;
  int const n_shots_i = static_cast<int>(n_shots);
  idx.shot_idx = n_shots_i == 1 ? 0 : linear_idx / shot_numel_i;
  idx.j = n_shots_i == 1 ? linear_idx : linear_idx - idx.shot_idx * shot_numel_i;
  int const yz_stride = idx.ny_i * idx.nx_i;
  idx.z = idx.j / yz_stride;
  int const rem = idx.j - idx.z * yz_stride;
  idx.y = rem / idx.nx_i;
  idx.x = rem - idx.y * idx.nx_i;
  return idx;
}

__device__ __forceinline__ bool current_cell_index_3d(
    LinearCellIndex3D *const idx, int64_t *const linear_idx) {
  if (gridDim.y == 1 && gridDim.z == 1) {
    int64_t const i =
        (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t const total = n_shots * shot_numel;
    if (i >= total) {
      return false;
    }
    if (total <= 2147483647LL) {
      *idx = decode_linear_cell_index_3d_i32(static_cast<int>(i));
      *linear_idx = i;
      return true;
    }
    *idx = decode_linear_cell_index_3d(i);
    *linear_idx = i;
    return true;
  }

  int const nx_i = static_cast<int>(nx);
  int const ny_i = static_cast<int>(ny);
  int const nz_i = static_cast<int>(nz);
  int const n_shots_i = static_cast<int>(n_shots);
  int shot_idx = 0;
  int block_z = blockIdx.z;
  if (n_shots_i != 1) {
    int const blocks_z_per_shot = gridDim.z / n_shots_i;
    if (blocks_z_per_shot <= 0) {
      return false;
    }
    shot_idx = blockIdx.z / blocks_z_per_shot;
    block_z = blockIdx.z - shot_idx * blocks_z_per_shot;
  }
  int const x = blockIdx.x * blockDim.x + threadIdx.x;
  int const y = blockIdx.y * blockDim.y + threadIdx.y;
  int const z = block_z * blockDim.z + threadIdx.z;
  if (shot_idx >= n_shots_i || z >= nz_i || y >= ny_i || x >= nx_i) {
    return false;
  }

  int const j = (z * ny_i + y) * nx_i + x;
  idx->shot_idx = shot_idx;
  idx->j = j;
  idx->z = z;
  idx->y = y;
  idx->x = x;
  idx->nz_i = nz_i;
  idx->ny_i = ny_i;
  idx->nx_i = nx_i;
  *linear_idx = (int64_t)shot_idx * (int64_t)nz_i * (int64_t)ny_i *
                    (int64_t)nx_i +
                (int64_t)j;
  return true;
}

__device__ __forceinline__ bool is_active_cell_3d(LinearCellIndex3D const &idx) {
  return idx.z >= FD_PAD && idx.z < idx.nz_i - FD_PAD + 1 && idx.y >= FD_PAD &&
         idx.y < idx.ny_i - FD_PAD + 1 && idx.x >= FD_PAD &&
         idx.x < idx.nx_i - FD_PAD + 1;
}

__device__ __forceinline__ bool is_physical_cell_3d(
    LinearCellIndex3D const &idx) {
  return idx.z >= static_cast<int>(pml_z0) &&
         idx.z < static_cast<int>(pml_z1) &&
         idx.y >= static_cast<int>(pml_y0) &&
         idx.y < static_cast<int>(pml_y1) &&
         idx.x >= static_cast<int>(pml_x0) &&
         idx.x < static_cast<int>(pml_x1);
}

__device__ __forceinline__ int64_t physical_snapshot_index_3d(
    LinearCellIndex3D const &idx) {
  int const phys_z = static_cast<int>(pml_z1 - pml_z0);
  int const phys_y = static_cast<int>(pml_y1 - pml_y0);
  int const phys_x = static_cast<int>(pml_x1 - pml_x0);
  int const z = idx.z - static_cast<int>(pml_z0);
  int const y = idx.y - static_cast<int>(pml_y0);
  int const x = idx.x - static_cast<int>(pml_x0);
  return (((int64_t)idx.shot_idx * (int64_t)phys_z + (int64_t)z) *
              (int64_t)phys_y +
          (int64_t)y) *
             (int64_t)phys_x +
         (int64_t)x;
}

__device__ __forceinline__ bool decode_physical_snapshot_index_3d(
    LinearCellIndex3D *const idx, int64_t *const full_linear_idx,
    int64_t const compact_idx) {
  int const nz_i = static_cast<int>(nz);
  int const ny_i = static_cast<int>(ny);
  int const nx_i = static_cast<int>(nx);
  int const phys_z = static_cast<int>(pml_z1 - pml_z0);
  int const phys_y = static_cast<int>(pml_y1 - pml_y0);
  int const phys_x = static_cast<int>(pml_x1 - pml_x0);
  if (phys_z <= 0 || phys_y <= 0 || phys_x <= 0) {
    return false;
  }

  int64_t const cells_per_shot =
      (int64_t)phys_z * (int64_t)phys_y * (int64_t)phys_x;
  int64_t const total = n_shots * cells_per_shot;
  if (compact_idx >= total) {
    return false;
  }

  int64_t const shot_idx64 = compact_idx / cells_per_shot;
  int64_t rem = compact_idx - shot_idx64 * cells_per_shot;
  int const z_local =
      static_cast<int>(rem / ((int64_t)phys_y * (int64_t)phys_x));
  rem -= (int64_t)z_local * (int64_t)phys_y * (int64_t)phys_x;
  int const y_local = static_cast<int>(rem / (int64_t)phys_x);
  int const x_local = static_cast<int>(rem - (int64_t)y_local *
                                                 (int64_t)phys_x);

  int const z = static_cast<int>(pml_z0) + z_local;
  int const y = static_cast<int>(pml_y0) + y_local;
  int const x = static_cast<int>(pml_x0) + x_local;
  int const j = (z * ny_i + y) * nx_i + x;
  idx->shot_idx = static_cast<int>(shot_idx64);
  idx->j = j;
  idx->z = z;
  idx->y = y;
  idx->x = x;
  idx->nz_i = nz_i;
  idx->ny_i = ny_i;
  idx->nx_i = nx_i;
  *full_linear_idx = (int64_t)idx->shot_idx * (int64_t)nz_i *
                         (int64_t)ny_i * (int64_t)nx_i +
                     (int64_t)j;
  return true;
}

template <bool UNIFORM_COEFF = false>
__global__ void forward_kernel_h(
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
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh) {
  int64_t i = 0;
  LinearCellIndex3D idx{};
  if (!current_cell_index_3d(&idx, &i)) {
    return;
  }

  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;

  if (!is_active_cell_3d(idx)) {
    return;
  }

#define EX_L(dz, dy, dx) EX(dz, dy, dx)
#define EY_L(dz, dy, dx) EY(dz, dy, dx)
#define EZ_L(dz, dy, dx) EZ(dz, dy, dx)

  TIDE_DTYPE const cq_val =
      UNIFORM_COEFF ? cq_uniform_coeff : (cq_batched ? cq[i] : cq[j]);

  int const pml_z0h = static_cast<int>(pml_z0);
  int const pml_z1h = tide_max(pml_z0h, static_cast<int>(pml_z1) - 1);
  int const pml_y0h = static_cast<int>(pml_y0);
  int const pml_y1h = tide_max(pml_y0h, static_cast<int>(pml_y1) - 1);
  int const pml_x0h = static_cast<int>(pml_x0);
  int const pml_x1h = tide_max(pml_x0h, static_cast<int>(pml_x1) - 1);

  bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
  bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
  bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

  TIDE_DTYPE dEy_dz_pml = 0;
  TIDE_DTYPE dEz_dy_pml = 0;
  TIDE_DTYPE dEz_dx_pml = 0;
  TIDE_DTYPE dEx_dz_pml = 0;
  TIDE_DTYPE dEx_dy_pml = 0;
  TIDE_DTYPE dEy_dx_pml = 0;

  if (z < idx.nz_i - FD_PAD) {
    TIDE_DTYPE dEy_dz = DIFFZH1(EY_L);
    if (pml_z_h) {
      m_ey_z[i] = bzh[z] * m_ey_z[i] + azh[z] * dEy_dz;
      dEy_dz = dEy_dz / kzh[z] + m_ey_z[i];
    }
    dEy_dz_pml = dEy_dz;
  }

  if (y < idx.ny_i - FD_PAD) {
    TIDE_DTYPE dEz_dy = DIFFYH1(EZ_L);
    if (pml_y_h) {
      m_ez_y[i] = byh[y] * m_ez_y[i] + ayh[y] * dEz_dy;
      dEz_dy = dEz_dy / kyh[y] + m_ez_y[i];
    }
    dEz_dy_pml = dEz_dy;
  }

  if (x < idx.nx_i - FD_PAD) {
    TIDE_DTYPE dEz_dx = DIFFXH1(EZ_L);
    if (pml_x_h) {
      m_ez_x[i] = bxh[x] * m_ez_x[i] + axh[x] * dEz_dx;
      dEz_dx = dEz_dx / kxh[x] + m_ez_x[i];
    }
    dEz_dx_pml = dEz_dx;
  }

  if (z < idx.nz_i - FD_PAD) {
    TIDE_DTYPE dEx_dz = DIFFZH1(EX_L);
    if (pml_z_h) {
      m_ex_z[i] = bzh[z] * m_ex_z[i] + azh[z] * dEx_dz;
      dEx_dz = dEx_dz / kzh[z] + m_ex_z[i];
    }
    dEx_dz_pml = dEx_dz;
  }

  if (y < idx.ny_i - FD_PAD) {
    TIDE_DTYPE dEx_dy = DIFFYH1(EX_L);
    if (pml_y_h) {
      m_ex_y[i] = byh[y] * m_ex_y[i] + ayh[y] * dEx_dy;
      dEx_dy = dEx_dy / kyh[y] + m_ex_y[i];
    }
    dEx_dy_pml = dEx_dy;
  }

  if (x < idx.nx_i - FD_PAD) {
    TIDE_DTYPE dEy_dx = DIFFXH1(EY_L);
    if (pml_x_h) {
      m_ey_x[i] = bxh[x] * m_ey_x[i] + axh[x] * dEy_dx;
      dEy_dx = dEy_dx / kxh[x] + m_ey_x[i];
    }
    dEy_dx_pml = dEy_dx;
  }

  hx[i] -= cq_val * (dEy_dz_pml - dEz_dy_pml);
  hy[i] -= cq_val * (dEz_dx_pml - dEx_dz_pml);
  hz[i] -= cq_val * (dEx_dy_pml - dEy_dx_pml);

#undef EX_L
#undef EY_L
#undef EZ_L
}

template <bool UNIFORM_COEFF = false>
__global__ void forward_kernel_e(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx) {
  int64_t i = 0;
  LinearCellIndex3D idx{};
  if (!current_cell_index_3d(&idx, &i)) {
    return;
  }

  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;

  if (!is_active_cell_3d(idx)) {
    return;
  }

#define HX_L(dz, dy, dx) HX(dz, dy, dx)
#define HY_L(dz, dy, dx) HY(dz, dy, dx)
#define HZ_L(dz, dy, dx) HZ(dz, dy, dx)

  TIDE_DTYPE const ca_val =
      UNIFORM_COEFF ? ca_uniform_coeff : (ca_batched ? ca[i] : ca[j]);
  TIDE_DTYPE const cb_val =
      UNIFORM_COEFF ? cb_uniform_coeff : (cb_batched ? cb[i] : cb[j]);

  int const pml_z0i = static_cast<int>(pml_z0);
  int const pml_z1i = static_cast<int>(pml_z1);
  int const pml_y0i = static_cast<int>(pml_y0);
  int const pml_y1i = static_cast<int>(pml_y1);
  int const pml_x0i = static_cast<int>(pml_x0);
  int const pml_x1i = static_cast<int>(pml_x1);
  bool const pml_z_v = (z < pml_z0i) || (z >= pml_z1i);
  bool const pml_y_v = (y < pml_y0i) || (y >= pml_y1i);
  bool const pml_x_v = (x < pml_x0i) || (x >= pml_x1i);

  TIDE_DTYPE dHy_dz = DIFFZ1(HY_L);
  TIDE_DTYPE dHz_dy = DIFFY1(HZ_L);
  TIDE_DTYPE dHz_dx = DIFFX1(HZ_L);
  TIDE_DTYPE dHx_dz = DIFFZ1(HX_L);
  TIDE_DTYPE dHx_dy = DIFFY1(HX_L);
  TIDE_DTYPE dHy_dx = DIFFX1(HY_L);

  if (pml_z_v) {
    m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
    dHy_dz = dHy_dz / kz[z] + m_hy_z[i];

    m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
    dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
  }

  if (pml_y_v) {
    m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
    dHz_dy = dHz_dy / ky[y] + m_hz_y[i];

    m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
    dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
  }

  if (pml_x_v) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
    dHz_dx = dHz_dx / kx[x] + m_hz_x[i];

    m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
    dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
  }

  ex[i] = ca_val * ex[i] + cb_val * (dHy_dz - dHz_dy);
  ey[i] = ca_val * ey[i] + cb_val * (dHz_dx - dHx_dz);
  ez[i] = ca_val * ez[i] + cb_val * (dHx_dy - dHy_dx);

#undef HX_L
#undef HY_L
#undef HZ_L
}

__global__ void forward_kernel_e_debye(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const debye_cp,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE const *__restrict const pol_ex,
    TIDE_DTYPE const *__restrict const pol_ey,
    TIDE_DTYPE const *__restrict const pol_ez,
    TIDE_DTYPE *__restrict const ex_prev,
    TIDE_DTYPE *__restrict const ey_prev,
    TIDE_DTYPE *__restrict const ez_prev,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx,
    int64_t const n_poles) {
  int64_t i = 0;
  LinearCellIndex3D idx{};
  if (!current_cell_index_3d(&idx, &i)) {
    return;
  }

  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;

  if (!is_active_cell_3d(idx)) {
    return;
  }

#define HX_L(dz, dy, dx) HX(dz, dy, dx)
#define HY_L(dz, dy, dx) HY(dz, dy, dx)
#define HZ_L(dz, dy, dx) HZ(dz, dy, dx)

  TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
  TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];

  int const pml_z0i = static_cast<int>(pml_z0);
  int const pml_z1i = static_cast<int>(pml_z1);
  int const pml_y0i = static_cast<int>(pml_y0);
  int const pml_y1i = static_cast<int>(pml_y1);
  int const pml_x0i = static_cast<int>(pml_x0);
  int const pml_x1i = static_cast<int>(pml_x1);
  bool const pml_z_v = (z < pml_z0i) || (z >= pml_z1i);
  bool const pml_y_v = (y < pml_y0i) || (y >= pml_y1i);
  bool const pml_x_v = (x < pml_x0i) || (x >= pml_x1i);

  TIDE_DTYPE dHy_dz = DIFFZ1(HY_L);
  TIDE_DTYPE dHz_dy = DIFFY1(HZ_L);
  TIDE_DTYPE dHz_dx = DIFFX1(HZ_L);
  TIDE_DTYPE dHx_dz = DIFFZ1(HX_L);
  TIDE_DTYPE dHx_dy = DIFFY1(HX_L);
  TIDE_DTYPE dHy_dx = DIFFX1(HY_L);

  if (pml_z_v) {
    m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
    dHy_dz = dHy_dz / kz[z] + m_hy_z[i];

    m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
    dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
  }

  if (pml_y_v) {
    m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
    dHz_dy = dHz_dy / ky[y] + m_hz_y[i];

    m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
    dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
  }

  if (pml_x_v) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
    dHz_dx = dHz_dx / kx[x] + m_hz_x[i];

    m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
    dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
  }

  TIDE_DTYPE ex_next = ca_val * ex[i] + cb_val * (dHy_dz - dHz_dy);
  TIDE_DTYPE ey_next = ca_val * ey[i] + cb_val * (dHz_dx - dHx_dz);
  TIDE_DTYPE ez_next = ca_val * ez[i] + cb_val * (dHx_dy - dHy_dx);

  TIDE_DTYPE pol_term_x = 0;
  TIDE_DTYPE pol_term_y = 0;
  TIDE_DTYPE pol_term_z = 0;
  int const n_poles_i = static_cast<int>(n_poles);
  for (int pole = 0; pole < n_poles_i; ++pole) {
    int64_t const coeff_idx = pole * shot_numel + j;
    int64_t const pol_idx =
        ((int64_t)idx.shot_idx * n_poles_i + pole) * shot_numel + j;
    TIDE_DTYPE const cp = debye_cp[coeff_idx];
    pol_term_x += cp * pol_ex[pol_idx];
    pol_term_y += cp * pol_ey[pol_idx];
    pol_term_z += cp * pol_ez[pol_idx];
  }

  ex[i] = ex_next + pol_term_x;
  ey[i] = ey_next + pol_term_y;
  ez[i] = ez_next + pol_term_z;

#undef HX_L
#undef HY_L
#undef HZ_L
}

__global__ void update_polarization_debye_3d(
    TIDE_DTYPE const *__restrict const prev_field,
    TIDE_DTYPE const *__restrict const field,
    TIDE_DTYPE const *__restrict const debye_a,
    TIDE_DTYPE const *__restrict const debye_b,
    TIDE_DTYPE *__restrict const polarization,
    int64_t const n_poles) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_poles * shot_numel;
  if (i >= total) {
    return;
  }

  int64_t const shot_pole = i / shot_numel;
  int64_t const j = i - shot_pole * shot_numel;
  int64_t const shot_idx = shot_pole / n_poles;
  int64_t const pole = shot_pole - shot_idx * n_poles;
  int64_t const coeff_idx = pole * shot_numel + j;
  int64_t const field_idx = shot_idx * shot_numel + j;

  polarization[i] = debye_a[coeff_idx] * polarization[i] +
                    debye_b[coeff_idx] * (field[field_idx] + prev_field[field_idx]);
}

template <typename SnapshotT, bool PHYSICAL_STORAGE = false>
__global__ void forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx,
    SnapshotT *__restrict const ex_store,
    SnapshotT *__restrict const ey_store,
    SnapshotT *__restrict const ez_store,
    SnapshotT *__restrict const curl_x_store,
    SnapshotT *__restrict const curl_y_store,
    SnapshotT *__restrict const curl_z_store,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t i = 0;
  LinearCellIndex3D idx{};
  if (!current_cell_index_3d(&idx, &i)) {
    return;
  }

  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;

  if (!is_active_cell_3d(idx)) {
    return;
  }

#define HX_S(dz, dy, dx) HX(dz, dy, dx)
#define HY_S(dz, dy, dx) HY(dz, dy, dx)
#define HZ_S(dz, dy, dx) HZ(dz, dy, dx)

  TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
  TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];

  int const pml_z0i = static_cast<int>(pml_z0);
  int const pml_z1i = static_cast<int>(pml_z1);
  int const pml_y0i = static_cast<int>(pml_y0);
  int const pml_y1i = static_cast<int>(pml_y1);
  int const pml_x0i = static_cast<int>(pml_x0);
  int const pml_x1i = static_cast<int>(pml_x1);
  bool const pml_z_v = (z < pml_z0i) || (z >= pml_z1i);
  bool const pml_y_v = (y < pml_y0i) || (y >= pml_y1i);
  bool const pml_x_v = (x < pml_x0i) || (x >= pml_x1i);

  TIDE_DTYPE dHy_dz = DIFFZ1(HY_S);
  TIDE_DTYPE dHz_dy = DIFFY1(HZ_S);
  TIDE_DTYPE dHz_dx = DIFFX1(HZ_S);
  TIDE_DTYPE dHx_dz = DIFFZ1(HX_S);
  TIDE_DTYPE dHx_dy = DIFFY1(HX_S);
  TIDE_DTYPE dHy_dx = DIFFX1(HY_S);

  if (pml_z_v) {
    m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
    dHy_dz = dHy_dz / kz[z] + m_hy_z[i];
    m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
    dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
  }
  if (pml_y_v) {
    m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
    dHz_dy = dHz_dy / ky[y] + m_hz_y[i];
    m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
    dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
  }
  if (pml_x_v) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
    dHz_dx = dHz_dx / kx[x] + m_hz_x[i];
    m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
    dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
  }

  TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
  TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
  TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

  int64_t store_i = i;
  bool write_snapshot = true;
  if constexpr (PHYSICAL_STORAGE) {
    write_snapshot = is_physical_cell_3d(idx);
    if (write_snapshot) {
      store_i = physical_snapshot_index_3d(idx);
    }
  }

  if (write_snapshot && ca_requires_grad && ex_store != nullptr) {
    ex_store[store_i] = encode_snapshot<SnapshotT>(ex[i]);
    ey_store[store_i] = encode_snapshot<SnapshotT>(ey[i]);
    ez_store[store_i] = encode_snapshot<SnapshotT>(ez[i]);
  }
  if (write_snapshot && cb_requires_grad && curl_x_store != nullptr) {
    curl_x_store[store_i] = encode_snapshot<SnapshotT>(curl_x);
    curl_y_store[store_i] = encode_snapshot<SnapshotT>(curl_y);
    curl_z_store[store_i] = encode_snapshot<SnapshotT>(curl_z);
  }

  ex[i] = ca_val * ex[i] + cb_val * curl_x;
  ey[i] = ca_val * ey[i] + cb_val * curl_y;
  ez[i] = ca_val * ez[i] + cb_val * curl_z;

#undef HX_S
#undef HY_S
#undef HZ_S
}

template <typename SnapshotT>
__global__ void store_final_e_3d(
    TIDE_DTYPE const *__restrict const ex,
    TIDE_DTYPE const *__restrict const ey,
    TIDE_DTYPE const *__restrict const ez,
    SnapshotT *__restrict const ex_store,
    SnapshotT *__restrict const ey_store,
    SnapshotT *__restrict const ez_store) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  LinearCellIndex3D const idx = decode_linear_cell_index_3d(i);
  if (!is_active_cell_3d(idx)) {
    return;
  }

  ex_store[i] = encode_snapshot<SnapshotT>(ex[i]);
  ey_store[i] = encode_snapshot<SnapshotT>(ey[i]);
  ez_store[i] = encode_snapshot<SnapshotT>(ez[i]);
}

__global__ void born_forward_kernel_h(
    TIDE_DTYPE const *__restrict const cq,
    TIDE_DTYPE const *__restrict const dex,
    TIDE_DTYPE const *__restrict const dey,
    TIDE_DTYPE const *__restrict const dez,
    TIDE_DTYPE *__restrict const dhx,
    TIDE_DTYPE *__restrict const dhy,
    TIDE_DTYPE *__restrict const dhz,
    TIDE_DTYPE *__restrict const dm_ey_z,
    TIDE_DTYPE *__restrict const dm_ez_y,
    TIDE_DTYPE *__restrict const dm_ez_x,
    TIDE_DTYPE *__restrict const dm_ex_z,
    TIDE_DTYPE *__restrict const dm_ex_y,
    TIDE_DTYPE *__restrict const dm_ey_x,
    TIDE_DTYPE const *__restrict const azh,
    TIDE_DTYPE const *__restrict const bzh,
    TIDE_DTYPE const *__restrict const ayh,
    TIDE_DTYPE const *__restrict const byh,
    TIDE_DTYPE const *__restrict const axh,
    TIDE_DTYPE const *__restrict const bxh,
    TIDE_DTYPE const *__restrict const kzh,
    TIDE_DTYPE const *__restrict const kyh,
    TIDE_DTYPE const *__restrict const kxh) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  LinearCellIndex3D const idx = decode_linear_cell_index_3d(i);
  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;

  if (!is_active_cell_3d(idx)) {
    return;
  }

#define DEX_L(dz, dy, dx) DEX(dz, dy, dx)
#define DEY_L(dz, dy, dx) DEY(dz, dy, dx)
#define DEZ_L(dz, dy, dx) DEZ(dz, dy, dx)

  TIDE_DTYPE const cq_val = CQ_I(0, 0, 0);

  int const pml_z0h = static_cast<int>(pml_z0);
  int const pml_z1h = tide_max(pml_z0h, static_cast<int>(pml_z1) - 1);
  int const pml_y0h = static_cast<int>(pml_y0);
  int const pml_y1h = tide_max(pml_y0h, static_cast<int>(pml_y1) - 1);
  int const pml_x0h = static_cast<int>(pml_x0);
  int const pml_x1h = tide_max(pml_x0h, static_cast<int>(pml_x1) - 1);

  bool const pml_z_h = (z < pml_z0h) || (z >= pml_z1h);
  bool const pml_y_h = (y < pml_y0h) || (y >= pml_y1h);
  bool const pml_x_h = (x < pml_x0h) || (x >= pml_x1h);

  TIDE_DTYPE dDEy_dz_pml = 0;
  TIDE_DTYPE dDEz_dy_pml = 0;
  TIDE_DTYPE dDEz_dx_pml = 0;
  TIDE_DTYPE dDEx_dz_pml = 0;
  TIDE_DTYPE dDEx_dy_pml = 0;
  TIDE_DTYPE dDEy_dx_pml = 0;

  if (z < idx.nz_i - FD_PAD) {
    TIDE_DTYPE dDEy_dz = DIFFZH1(DEY_L);
    if (pml_z_h) {
      dm_ey_z[i] = bzh[z] * dm_ey_z[i] + azh[z] * dDEy_dz;
      dDEy_dz = dDEy_dz / kzh[z] + dm_ey_z[i];
    }
    dDEy_dz_pml = dDEy_dz;
  }
  if (y < idx.ny_i - FD_PAD) {
    TIDE_DTYPE dDEz_dy = DIFFYH1(DEZ_L);
    if (pml_y_h) {
      dm_ez_y[i] = byh[y] * dm_ez_y[i] + ayh[y] * dDEz_dy;
      dDEz_dy = dDEz_dy / kyh[y] + dm_ez_y[i];
    }
    dDEz_dy_pml = dDEz_dy;
  }
  if (x < idx.nx_i - FD_PAD) {
    TIDE_DTYPE dDEz_dx = DIFFXH1(DEZ_L);
    if (pml_x_h) {
      dm_ez_x[i] = bxh[x] * dm_ez_x[i] + axh[x] * dDEz_dx;
      dDEz_dx = dDEz_dx / kxh[x] + dm_ez_x[i];
    }
    dDEz_dx_pml = dDEz_dx;
  }
  if (z < idx.nz_i - FD_PAD) {
    TIDE_DTYPE dDEx_dz = DIFFZH1(DEX_L);
    if (pml_z_h) {
      dm_ex_z[i] = bzh[z] * dm_ex_z[i] + azh[z] * dDEx_dz;
      dDEx_dz = dDEx_dz / kzh[z] + dm_ex_z[i];
    }
    dDEx_dz_pml = dDEx_dz;
  }
  if (y < idx.ny_i - FD_PAD) {
    TIDE_DTYPE dDEx_dy = DIFFYH1(DEX_L);
    if (pml_y_h) {
      dm_ex_y[i] = byh[y] * dm_ex_y[i] + ayh[y] * dDEx_dy;
      dDEx_dy = dDEx_dy / kyh[y] + dm_ex_y[i];
    }
    dDEx_dy_pml = dDEx_dy;
  }
  if (x < idx.nx_i - FD_PAD) {
    TIDE_DTYPE dDEy_dx = DIFFXH1(DEY_L);
    if (pml_x_h) {
      dm_ey_x[i] = bxh[x] * dm_ey_x[i] + axh[x] * dDEy_dx;
      dDEy_dx = dDEy_dx / kxh[x] + dm_ey_x[i];
    }
    dDEy_dx_pml = dDEy_dx;
  }

  dhx[i] -= cq_val * (dDEy_dz_pml - dDEz_dy_pml);
  dhy[i] -= cq_val * (dDEz_dx_pml - dDEx_dz_pml);
  dhz[i] -= cq_val * (dDEx_dy_pml - dDEy_dx_pml);

#undef DEX_L
#undef DEY_L
#undef DEZ_L
}

__global__ void born_forward_kernel_e(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const dex,
    TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const dez,
    TIDE_DTYPE const *__restrict const dhx,
    TIDE_DTYPE const *__restrict const dhy,
    TIDE_DTYPE const *__restrict const dhz,
    TIDE_DTYPE *__restrict const dm_hy_z,
    TIDE_DTYPE *__restrict const dm_hz_y,
    TIDE_DTYPE *__restrict const dm_hz_x,
    TIDE_DTYPE *__restrict const dm_hx_z,
    TIDE_DTYPE *__restrict const dm_hx_y,
    TIDE_DTYPE *__restrict const dm_hy_x,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  LinearCellIndex3D const idx = decode_linear_cell_index_3d(i);
  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;

  if (!is_active_cell_3d(idx)) {
    return;
  }

#define HX_S(dz, dy, dx) HX(dz, dy, dx)
#define HY_S(dz, dy, dx) HY(dz, dy, dx)
#define HZ_S(dz, dy, dx) HZ(dz, dy, dx)
#define DHX_S(dz, dy, dx) DHX(dz, dy, dx)
#define DHY_S(dz, dy, dx) DHY(dz, dy, dx)
#define DHZ_S(dz, dy, dx) DHZ(dz, dy, dx)

  TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
  TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];
  TIDE_DTYPE const dca_val = dca[j];
  TIDE_DTYPE const dcb_val = dcb[j];

  int const pml_z0i = static_cast<int>(pml_z0);
  int const pml_z1i = static_cast<int>(pml_z1);
  int const pml_y0i = static_cast<int>(pml_y0);
  int const pml_y1i = static_cast<int>(pml_y1);
  int const pml_x0i = static_cast<int>(pml_x0);
  int const pml_x1i = static_cast<int>(pml_x1);
  bool const pml_z_v = (z < pml_z0i) || (z >= pml_z1i);
  bool const pml_y_v = (y < pml_y0i) || (y >= pml_y1i);
  bool const pml_x_v = (x < pml_x0i) || (x >= pml_x1i);

  TIDE_DTYPE dHy_dz = DIFFZ1(HY_S);
  TIDE_DTYPE dHz_dy = DIFFY1(HZ_S);
  TIDE_DTYPE dHz_dx = DIFFX1(HZ_S);
  TIDE_DTYPE dHx_dz = DIFFZ1(HX_S);
  TIDE_DTYPE dHx_dy = DIFFY1(HX_S);
  TIDE_DTYPE dHy_dx = DIFFX1(HY_S);

  if (pml_z_v) {
    m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
    dHy_dz = dHy_dz / kz[z] + m_hy_z[i];
    m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
    dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
  }
  if (pml_y_v) {
    m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
    dHz_dy = dHz_dy / ky[y] + m_hz_y[i];
    m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
    dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
  }
  if (pml_x_v) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
    dHz_dx = dHz_dx / kx[x] + m_hz_x[i];
    m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
    dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
  }

  TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
  TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
  TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

  TIDE_DTYPE ddHy_dz = DIFFZ1(DHY_S);
  TIDE_DTYPE ddHz_dy = DIFFY1(DHZ_S);
  TIDE_DTYPE ddHz_dx = DIFFX1(DHZ_S);
  TIDE_DTYPE ddHx_dz = DIFFZ1(DHX_S);
  TIDE_DTYPE ddHx_dy = DIFFY1(DHX_S);
  TIDE_DTYPE ddHy_dx = DIFFX1(DHY_S);

  if (pml_z_v) {
    dm_hy_z[i] = bz[z] * dm_hy_z[i] + az[z] * ddHy_dz;
    ddHy_dz = ddHy_dz / kz[z] + dm_hy_z[i];
    dm_hx_z[i] = bz[z] * dm_hx_z[i] + az[z] * ddHx_dz;
    ddHx_dz = ddHx_dz / kz[z] + dm_hx_z[i];
  }
  if (pml_y_v) {
    dm_hz_y[i] = by[y] * dm_hz_y[i] + ay[y] * ddHz_dy;
    ddHz_dy = ddHz_dy / ky[y] + dm_hz_y[i];
    dm_hx_y[i] = by[y] * dm_hx_y[i] + ay[y] * ddHx_dy;
    ddHx_dy = ddHx_dy / ky[y] + dm_hx_y[i];
  }
  if (pml_x_v) {
    dm_hz_x[i] = bx[x] * dm_hz_x[i] + ax[x] * ddHz_dx;
    ddHz_dx = ddHz_dx / kx[x] + dm_hz_x[i];
    dm_hy_x[i] = bx[x] * dm_hy_x[i] + ax[x] * ddHy_dx;
    ddHy_dx = ddHy_dx / kx[x] + dm_hy_x[i];
  }

  TIDE_DTYPE const dcurl_x = ddHy_dz - ddHz_dy;
  TIDE_DTYPE const dcurl_y = ddHz_dx - ddHx_dz;
  TIDE_DTYPE const dcurl_z = ddHx_dy - ddHy_dx;

  TIDE_DTYPE const ex_old = ex[i];
  TIDE_DTYPE const ey_old = ey[i];
  TIDE_DTYPE const ez_old = ez[i];

  dex[i] = ca_val * dex[i] + cb_val * dcurl_x + dca_val * ex_old + dcb_val * curl_x;
  dey[i] = ca_val * dey[i] + cb_val * dcurl_y + dca_val * ey_old + dcb_val * curl_y;
  dez[i] = ca_val * dez[i] + cb_val * dcurl_z + dca_val * ez_old + dcb_val * curl_z;

  ex[i] = ca_val * ex[i] + cb_val * curl_x;
  ey[i] = ca_val * ey[i] + cb_val * curl_y;
  ez[i] = ca_val * ez[i] + cb_val * curl_z;

#undef HX_S
#undef HY_S
#undef HZ_S
#undef DHX_S
#undef DHY_S
#undef DHZ_S
}

template <typename SnapshotT>
__global__ void born_forward_kernel_e_with_storage(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE *__restrict const ex,
    TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const ez,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hy,
    TIDE_DTYPE const *__restrict const hz,
    TIDE_DTYPE *__restrict const m_hy_z,
    TIDE_DTYPE *__restrict const m_hz_y,
    TIDE_DTYPE *__restrict const m_hz_x,
    TIDE_DTYPE *__restrict const m_hx_z,
    TIDE_DTYPE *__restrict const m_hx_y,
    TIDE_DTYPE *__restrict const m_hy_x,
    TIDE_DTYPE *__restrict const dex,
    TIDE_DTYPE *__restrict const dey,
    TIDE_DTYPE *__restrict const dez,
    TIDE_DTYPE const *__restrict const dhx,
    TIDE_DTYPE const *__restrict const dhy,
    TIDE_DTYPE const *__restrict const dhz,
    TIDE_DTYPE *__restrict const dm_hy_z,
    TIDE_DTYPE *__restrict const dm_hz_y,
    TIDE_DTYPE *__restrict const dm_hz_x,
    TIDE_DTYPE *__restrict const dm_hx_z,
    TIDE_DTYPE *__restrict const dm_hx_y,
    TIDE_DTYPE *__restrict const dm_hy_x,
    TIDE_DTYPE const *__restrict const az,
    TIDE_DTYPE const *__restrict const bz,
    TIDE_DTYPE const *__restrict const ay,
    TIDE_DTYPE const *__restrict const by,
    TIDE_DTYPE const *__restrict const ax,
    TIDE_DTYPE const *__restrict const bx,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx,
    SnapshotT *__restrict const ex_store,
    SnapshotT *__restrict const ey_store,
    SnapshotT *__restrict const ez_store,
    SnapshotT *__restrict const curl_x_store,
    SnapshotT *__restrict const curl_y_store,
    SnapshotT *__restrict const curl_z_store,
    SnapshotT *__restrict const dex_store,
    SnapshotT *__restrict const dey_store,
    SnapshotT *__restrict const dez_store,
    SnapshotT *__restrict const dcurl_x_store,
    SnapshotT *__restrict const dcurl_y_store,
    SnapshotT *__restrict const dcurl_z_store,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  LinearCellIndex3D const idx = decode_linear_cell_index_3d(i);
  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;

  if (!is_active_cell_3d(idx)) {
    return;
  }

#define HX_S(dz, dy, dx) HX(dz, dy, dx)
#define HY_S(dz, dy, dx) HY(dz, dy, dx)
#define HZ_S(dz, dy, dx) HZ(dz, dy, dx)
#define DHX_S(dz, dy, dx) DHX(dz, dy, dx)
#define DHY_S(dz, dy, dx) DHY(dz, dy, dx)
#define DHZ_S(dz, dy, dx) DHZ(dz, dy, dx)

  TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
  TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];
  TIDE_DTYPE const dca_val = dca[j];
  TIDE_DTYPE const dcb_val = dcb[j];

  int const pml_z0i = static_cast<int>(pml_z0);
  int const pml_z1i = static_cast<int>(pml_z1);
  int const pml_y0i = static_cast<int>(pml_y0);
  int const pml_y1i = static_cast<int>(pml_y1);
  int const pml_x0i = static_cast<int>(pml_x0);
  int const pml_x1i = static_cast<int>(pml_x1);
  bool const pml_z_v = (z < pml_z0i) || (z >= pml_z1i);
  bool const pml_y_v = (y < pml_y0i) || (y >= pml_y1i);
  bool const pml_x_v = (x < pml_x0i) || (x >= pml_x1i);

  TIDE_DTYPE dHy_dz = DIFFZ1(HY_S);
  TIDE_DTYPE dHz_dy = DIFFY1(HZ_S);
  TIDE_DTYPE dHz_dx = DIFFX1(HZ_S);
  TIDE_DTYPE dHx_dz = DIFFZ1(HX_S);
  TIDE_DTYPE dHx_dy = DIFFY1(HX_S);
  TIDE_DTYPE dHy_dx = DIFFX1(HY_S);

  if (pml_z_v) {
    m_hy_z[i] = bz[z] * m_hy_z[i] + az[z] * dHy_dz;
    dHy_dz = dHy_dz / kz[z] + m_hy_z[i];
    m_hx_z[i] = bz[z] * m_hx_z[i] + az[z] * dHx_dz;
    dHx_dz = dHx_dz / kz[z] + m_hx_z[i];
  }
  if (pml_y_v) {
    m_hz_y[i] = by[y] * m_hz_y[i] + ay[y] * dHz_dy;
    dHz_dy = dHz_dy / ky[y] + m_hz_y[i];
    m_hx_y[i] = by[y] * m_hx_y[i] + ay[y] * dHx_dy;
    dHx_dy = dHx_dy / ky[y] + m_hx_y[i];
  }
  if (pml_x_v) {
    m_hz_x[i] = bx[x] * m_hz_x[i] + ax[x] * dHz_dx;
    dHz_dx = dHz_dx / kx[x] + m_hz_x[i];
    m_hy_x[i] = bx[x] * m_hy_x[i] + ax[x] * dHy_dx;
    dHy_dx = dHy_dx / kx[x] + m_hy_x[i];
  }

  TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
  TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
  TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

  if (ca_requires_grad && ex_store != nullptr) {
    ex_store[i] = encode_snapshot<SnapshotT>(ex[i]);
    ey_store[i] = encode_snapshot<SnapshotT>(ey[i]);
    ez_store[i] = encode_snapshot<SnapshotT>(ez[i]);
  }
  if (cb_requires_grad && curl_x_store != nullptr) {
    curl_x_store[i] = encode_snapshot<SnapshotT>(curl_x);
    curl_y_store[i] = encode_snapshot<SnapshotT>(curl_y);
    curl_z_store[i] = encode_snapshot<SnapshotT>(curl_z);
  }

  TIDE_DTYPE ddHy_dz = DIFFZ1(DHY_S);
  TIDE_DTYPE ddHz_dy = DIFFY1(DHZ_S);
  TIDE_DTYPE ddHz_dx = DIFFX1(DHZ_S);
  TIDE_DTYPE ddHx_dz = DIFFZ1(DHX_S);
  TIDE_DTYPE ddHx_dy = DIFFY1(DHX_S);
  TIDE_DTYPE ddHy_dx = DIFFX1(DHY_S);

  if (pml_z_v) {
    dm_hy_z[i] = bz[z] * dm_hy_z[i] + az[z] * ddHy_dz;
    ddHy_dz = ddHy_dz / kz[z] + dm_hy_z[i];
    dm_hx_z[i] = bz[z] * dm_hx_z[i] + az[z] * ddHx_dz;
    ddHx_dz = ddHx_dz / kz[z] + dm_hx_z[i];
  }
  if (pml_y_v) {
    dm_hz_y[i] = by[y] * dm_hz_y[i] + ay[y] * ddHz_dy;
    ddHz_dy = ddHz_dy / ky[y] + dm_hz_y[i];
    dm_hx_y[i] = by[y] * dm_hx_y[i] + ay[y] * ddHx_dy;
    ddHx_dy = ddHx_dy / ky[y] + dm_hx_y[i];
  }
  if (pml_x_v) {
    dm_hz_x[i] = bx[x] * dm_hz_x[i] + ax[x] * ddHz_dx;
    ddHz_dx = ddHz_dx / kx[x] + dm_hz_x[i];
    dm_hy_x[i] = bx[x] * dm_hy_x[i] + ax[x] * ddHy_dx;
    ddHy_dx = ddHy_dx / kx[x] + dm_hy_x[i];
  }

  TIDE_DTYPE const dcurl_x = ddHy_dz - ddHz_dy;
  TIDE_DTYPE const dcurl_y = ddHz_dx - ddHx_dz;
  TIDE_DTYPE const dcurl_z = ddHx_dy - ddHy_dx;

  if (dex_store != nullptr) {
    dex_store[i] = encode_snapshot<SnapshotT>(dex[i]);
    dey_store[i] = encode_snapshot<SnapshotT>(dey[i]);
    dez_store[i] = encode_snapshot<SnapshotT>(dez[i]);
  }
  if (dcurl_x_store != nullptr) {
    dcurl_x_store[i] = encode_snapshot<SnapshotT>(dcurl_x);
    dcurl_y_store[i] = encode_snapshot<SnapshotT>(dcurl_y);
    dcurl_z_store[i] = encode_snapshot<SnapshotT>(dcurl_z);
  }

  TIDE_DTYPE const ex_old = ex[i];
  TIDE_DTYPE const ey_old = ey[i];
  TIDE_DTYPE const ez_old = ez[i];

  dex[i] = ca_val * dex[i] + cb_val * dcurl_x + dca_val * ex_old + dcb_val * curl_x;
  dey[i] = ca_val * dey[i] + cb_val * dcurl_y + dca_val * ey_old + dcb_val * curl_y;
  dez[i] = ca_val * dez[i] + cb_val * dcurl_z + dca_val * ez_old + dcb_val * curl_z;

  ex[i] = ca_val * ex[i] + cb_val * curl_x;
  ey[i] = ca_val * ey[i] + cb_val * curl_y;
  ez[i] = ca_val * ez[i] + cb_val * curl_z;

#undef HX_S
#undef HY_S
#undef HZ_S
#undef DHX_S
#undef DHY_S
#undef DHZ_S
}

__global__ void add_adjoint_receivers_component(
    TIDE_DTYPE *__restrict const field,
    TIDE_DTYPE const *__restrict const grad_r,
    int64_t const *__restrict const receivers_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_receivers_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const rec = receivers_i[idx];
  if (rec >= 0) {
    int64_t const shot_idx = idx / n_receivers_per_shot;
    field[shot_idx * shot_numel + rec] += grad_r[idx];
  }
}

__global__ void record_adjoint_at_sources_component(
    TIDE_DTYPE *__restrict const grad_f,
    TIDE_DTYPE const *__restrict const field,
    int64_t const *__restrict const sources_i) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * n_sources_per_shot;
  if (idx >= total) {
    return;
  }
  int64_t const src = sources_i[idx];
  if (src >= 0) {
    int64_t const shot_idx = idx / n_sources_per_shot;
    grad_f[idx] = field[shot_idx * shot_numel + src];
  } else {
    grad_f[idx] = (TIDE_DTYPE)0;
  }
}

template <typename SnapshotT, bool PHYSICAL_STORAGE = false>
__global__ void coeff_grad_3d(
    TIDE_DTYPE const *__restrict const lambda_ex,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const lambda_ez,
    SnapshotT const *__restrict const ex_store,
    SnapshotT const *__restrict const ey_store,
    SnapshotT const *__restrict const ez_store,
    SnapshotT const *__restrict const curl_x_store,
    SnapshotT const *__restrict const curl_y_store,
    SnapshotT const *__restrict const curl_z_store,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    bool const grad_ca_step,
    bool const grad_cb_step,
    int64_t const step_ratio_eff) {
  int64_t const store_i =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t i = store_i;
  LinearCellIndex3D idx{};
  if constexpr (PHYSICAL_STORAGE) {
    if (!decode_physical_snapshot_index_3d(&idx, &i, store_i)) {
      return;
    }
  } else {
    int64_t const total = n_shots * shot_numel;
    if (store_i >= total) {
      return;
    }
    idx = decode_linear_cell_index_3d(i);
  }
  int const j = idx.j;
  if (!is_active_cell_3d(idx)) {
    return;
  }

  TIDE_DTYPE const lex_curr = lambda_ex[i];
  TIDE_DTYPE const ley_curr = lambda_ey[i];
  TIDE_DTYPE const lez_curr = lambda_ez[i];

  if (grad_ca_step && grad_ca != nullptr && ex_store != nullptr &&
      ey_store != nullptr && ez_store != nullptr) {
    TIDE_DTYPE const acc_ca = lex_curr * decode_snapshot(ex_store[store_i]) +
                              ley_curr * decode_snapshot(ey_store[store_i]) +
                              lez_curr * decode_snapshot(ez_store[store_i]);
    TIDE_DTYPE const scaled = acc_ca * (TIDE_DTYPE)step_ratio_eff;
    if (ca_batched) {
      grad_ca[i] += scaled;
    } else if (grad_ca_shot != nullptr) {
      grad_ca_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_ca[j], scaled);
    }
  }

  if (grad_cb_step && grad_cb != nullptr && curl_x_store != nullptr &&
      curl_y_store != nullptr && curl_z_store != nullptr) {
    TIDE_DTYPE const acc_cb =
        lex_curr * decode_snapshot(curl_x_store[store_i]) +
        ley_curr * decode_snapshot(curl_y_store[store_i]) +
        lez_curr * decode_snapshot(curl_z_store[store_i]);
    TIDE_DTYPE const scaled = acc_cb * (TIDE_DTYPE)step_ratio_eff;
    if (cb_batched) {
      grad_cb[i] += scaled;
    } else if (grad_cb_shot != nullptr) {
      grad_cb_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_cb[j], scaled);
    }
  }
}

template <typename SnapshotT>
__global__ void coeff_grad_eonly_3d(
    TIDE_DTYPE const *__restrict const lambda_ex,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const lambda_ez,
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    SnapshotT const *__restrict const ex_store,
    SnapshotT const *__restrict const ey_store,
    SnapshotT const *__restrict const ez_store,
    SnapshotT const *__restrict const ex_next_store,
    SnapshotT const *__restrict const ey_next_store,
    SnapshotT const *__restrict const ez_next_store,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    bool const grad_ca_step,
    bool const grad_cb_step,
    int64_t const step_ratio_eff) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  LinearCellIndex3D const idx = decode_linear_cell_index_3d(i);
  int const j = idx.j;
  if (!is_active_cell_3d(idx)) {
    return;
  }

  if (ex_store == nullptr || ey_store == nullptr || ez_store == nullptr) {
    return;
  }

  TIDE_DTYPE const ex_curr = decode_snapshot(ex_store[i]);
  TIDE_DTYPE const ey_curr = decode_snapshot(ey_store[i]);
  TIDE_DTYPE const ez_curr = decode_snapshot(ez_store[i]);
  TIDE_DTYPE const lex_curr = lambda_ex[i];
  TIDE_DTYPE const ley_curr = lambda_ey[i];
  TIDE_DTYPE const lez_curr = lambda_ez[i];

  if (grad_ca_step && grad_ca != nullptr) {
    TIDE_DTYPE const acc_ca =
        lex_curr * ex_curr + ley_curr * ey_curr + lez_curr * ez_curr;
    TIDE_DTYPE const scaled = acc_ca * (TIDE_DTYPE)step_ratio_eff;
    if (ca_batched) {
      grad_ca[i] += scaled;
    } else if (grad_ca_shot != nullptr) {
      grad_ca_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_ca[j], scaled);
    }
  }

  if (grad_cb_step && grad_cb != nullptr && ex_next_store != nullptr &&
      ey_next_store != nullptr && ez_next_store != nullptr) {
    TIDE_DTYPE const ca_val = ca_batched ? ca[i] : ca[j];
    TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];
    TIDE_DTYPE const curl_x =
        (decode_snapshot(ex_next_store[i]) - ca_val * ex_curr) / cb_val;
    TIDE_DTYPE const curl_y =
        (decode_snapshot(ey_next_store[i]) - ca_val * ey_curr) / cb_val;
    TIDE_DTYPE const curl_z =
        (decode_snapshot(ez_next_store[i]) - ca_val * ez_curr) / cb_val;
    TIDE_DTYPE const acc_cb =
        lex_curr * curl_x + ley_curr * curl_y + lez_curr * curl_z;
    TIDE_DTYPE const scaled = acc_cb * (TIDE_DTYPE)step_ratio_eff;
    if (cb_batched) {
      grad_cb[i] += scaled;
    } else if (grad_cb_shot != nullptr) {
      grad_cb_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_cb[j], scaled);
    }
  }
}

template <typename SnapshotT>
__global__ void material_grad_direct_eonly_3d(
    TIDE_DTYPE const *__restrict const lambda_ex,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const lambda_ez,
    TIDE_DTYPE const *__restrict const cb,
    SnapshotT const *__restrict const ex_store,
    SnapshotT const *__restrict const ey_store,
    SnapshotT const *__restrict const ez_store,
    SnapshotT const *__restrict const ex_next_store,
    SnapshotT const *__restrict const ey_next_store,
    SnapshotT const *__restrict const ez_next_store,
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE *__restrict const grad_eps_shot,
    TIDE_DTYPE *__restrict const grad_sigma_shot,
    bool const grad_eps_step,
    bool const grad_sigma_step,
    TIDE_DTYPE const dt_h,
    int64_t const step_ratio_eff) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  LinearCellIndex3D const idx = decode_linear_cell_index_3d(i);
  int const j = idx.j;
  if (!is_active_cell_3d(idx)) {
    return;
  }
  if (ex_store == nullptr || ey_store == nullptr || ez_store == nullptr ||
      ex_next_store == nullptr || ey_next_store == nullptr ||
      ez_next_store == nullptr) {
    return;
  }

  TIDE_DTYPE const cb_val = cb_batched ? cb[i] : cb[j];
  TIDE_DTYPE const scale_step = (TIDE_DTYPE)step_ratio_eff;
  TIDE_DTYPE const ex_curr = decode_snapshot(ex_store[i]);
  TIDE_DTYPE const ey_curr = decode_snapshot(ey_store[i]);
  TIDE_DTYPE const ez_curr = decode_snapshot(ez_store[i]);
  TIDE_DTYPE const ex_next = decode_snapshot(ex_next_store[i]);
  TIDE_DTYPE const ey_next = decode_snapshot(ey_next_store[i]);
  TIDE_DTYPE const ez_next = decode_snapshot(ez_next_store[i]);
  TIDE_DTYPE const lex_curr = lambda_ex[i];
  TIDE_DTYPE const ley_curr = lambda_ey[i];
  TIDE_DTYPE const lez_curr = lambda_ez[i];

  if (grad_eps_step && grad_eps != nullptr) {
    TIDE_DTYPE const acc_eps =
        lex_curr * (ex_curr - ex_next) + ley_curr * (ey_curr - ey_next) +
        lez_curr * (ez_curr - ez_next);
    TIDE_DTYPE const scaled =
        acc_eps * cb_val * ((TIDE_DTYPE)1 / dt_h) * kEp0 * scale_step;
    if (ca_batched) {
      grad_eps[i] += scaled;
    } else if (grad_eps_shot != nullptr) {
      grad_eps_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_eps[j], scaled);
    }
  }

  if (grad_sigma_step && grad_sigma != nullptr) {
    TIDE_DTYPE const acc_sigma =
        lex_curr * (ex_curr + ex_next) + ley_curr * (ey_curr + ey_next) +
        lez_curr * (ez_curr + ez_next);
    TIDE_DTYPE const scaled =
        -((TIDE_DTYPE)0.5) * acc_sigma * cb_val * scale_step;
    if (ca_batched) {
      grad_sigma[i] += scaled;
    } else if (grad_sigma_shot != nullptr) {
      grad_sigma_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_sigma[j], scaled);
    }
  }
}

template <typename SnapshotT>
__global__ void born_bggrad_prepare_direct_3d(
    TIDE_DTYPE const *__restrict const dca,
    TIDE_DTYPE const *__restrict const dcb,
    TIDE_DTYPE const *__restrict const lambda_ex,
    TIDE_DTYPE const *__restrict const lambda_ey,
    TIDE_DTYPE const *__restrict const lambda_ez,
    TIDE_DTYPE const *__restrict const lambda_hx,
    TIDE_DTYPE const *__restrict const lambda_hy,
    TIDE_DTYPE const *__restrict const lambda_hz,
    TIDE_DTYPE const *__restrict const m_lambda_hz_y,
    TIDE_DTYPE const *__restrict const m_lambda_hy_z,
    TIDE_DTYPE const *__restrict const m_lambda_hx_z,
    TIDE_DTYPE const *__restrict const m_lambda_hz_x,
    TIDE_DTYPE const *__restrict const m_lambda_hy_x,
    TIDE_DTYPE const *__restrict const m_lambda_hx_y,
    TIDE_DTYPE const *__restrict const kz,
    TIDE_DTYPE const *__restrict const ky,
    TIDE_DTYPE const *__restrict const kx,
    SnapshotT const *__restrict const dex_store,
    SnapshotT const *__restrict const dey_store,
    SnapshotT const *__restrict const dez_store,
    SnapshotT const *__restrict const dcurl_x_store,
    SnapshotT const *__restrict const dcurl_y_store,
    SnapshotT const *__restrict const dcurl_z_store,
    TIDE_DTYPE *__restrict const grad_ca,
    TIDE_DTYPE *__restrict const grad_cb,
    TIDE_DTYPE *__restrict const grad_ca_shot,
    TIDE_DTYPE *__restrict const grad_cb_shot,
    TIDE_DTYPE *__restrict const eta_source_ex,
    TIDE_DTYPE *__restrict const eta_source_ey,
    TIDE_DTYPE *__restrict const eta_source_ez,
    bool const direct_ca_step,
    bool const direct_cb_step,
    int64_t const step_ratio_eff) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }

  LinearCellIndex3D const idx = decode_linear_cell_index_3d(i);
  int const j = idx.j;
  int const z = idx.z;
  int const y = idx.y;
  int const x = idx.x;
  if (!is_active_cell_3d(idx)) {
    return;
  }

#define LHX_BG(dz, dy, dx) lambda_hx[ND_INDEX(i, dz, dy, dx)]
#define LHY_BG(dz, dy, dx) lambda_hy[ND_INDEX(i, dz, dy, dx)]
#define LHZ_BG(dz, dy, dx) lambda_hz[ND_INDEX(i, dz, dy, dx)]

  TIDE_DTYPE const lex_curr = lambda_ex[i];
  TIDE_DTYPE const ley_curr = lambda_ey[i];
  TIDE_DTYPE const lez_curr = lambda_ez[i];
  TIDE_DTYPE const dca_val = dca[j];
  TIDE_DTYPE const dcb_val = dcb[j];

  bool const pml_z_v = (z < pml_z0) || (z >= pml_z1);
  bool const pml_y_v = (y < pml_y0) || (y >= pml_y1);
  bool const pml_x_v = (x < pml_x0) || (x >= pml_x1);

  TIDE_DTYPE dHy_dz = DIFFZ1(LHY_BG);
  TIDE_DTYPE dHz_dy = DIFFY1(LHZ_BG);
  TIDE_DTYPE dHz_dx = DIFFX1(LHZ_BG);
  TIDE_DTYPE dHx_dz = DIFFZ1(LHX_BG);
  TIDE_DTYPE dHx_dy = DIFFY1(LHX_BG);
  TIDE_DTYPE dHy_dx = DIFFX1(LHY_BG);

  if (pml_z_v) {
    dHy_dz = dHy_dz / kz[z] + m_lambda_hy_z[i];
    dHx_dz = dHx_dz / kz[z] + m_lambda_hx_z[i];
  }
  if (pml_y_v) {
    dHz_dy = dHz_dy / ky[y] + m_lambda_hz_y[i];
    dHx_dy = dHx_dy / ky[y] + m_lambda_hx_y[i];
  }
  if (pml_x_v) {
    dHz_dx = dHz_dx / kx[x] + m_lambda_hz_x[i];
    dHy_dx = dHy_dx / kx[x] + m_lambda_hy_x[i];
  }

  TIDE_DTYPE const curl_x = dHy_dz - dHz_dy;
  TIDE_DTYPE const curl_y = dHz_dx - dHx_dz;
  TIDE_DTYPE const curl_z = dHx_dy - dHy_dx;

  eta_source_ex[i] = dca_val * lex_curr + dcb_val * curl_x;
  eta_source_ey[i] = dca_val * ley_curr + dcb_val * curl_y;
  eta_source_ez[i] = dca_val * lez_curr + dcb_val * curl_z;

  if (direct_ca_step && grad_ca != nullptr && dex_store != nullptr &&
      dey_store != nullptr && dez_store != nullptr) {
    TIDE_DTYPE const acc_ca = lex_curr * decode_snapshot(dex_store[i]) +
                              ley_curr * decode_snapshot(dey_store[i]) +
                              lez_curr * decode_snapshot(dez_store[i]);
    TIDE_DTYPE const scaled = acc_ca * (TIDE_DTYPE)step_ratio_eff;
    if (ca_batched) {
      grad_ca[i] += scaled;
    } else if (grad_ca_shot != nullptr) {
      grad_ca_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_ca[j], scaled);
    }
  }

  if (direct_cb_step && grad_cb != nullptr && dcurl_x_store != nullptr &&
      dcurl_y_store != nullptr && dcurl_z_store != nullptr) {
    TIDE_DTYPE const acc_cb = lex_curr * decode_snapshot(dcurl_x_store[i]) +
                              ley_curr * decode_snapshot(dcurl_y_store[i]) +
                              lez_curr * decode_snapshot(dcurl_z_store[i]);
    TIDE_DTYPE const scaled = acc_cb * (TIDE_DTYPE)step_ratio_eff;
    if (cb_batched) {
      grad_cb[i] += scaled;
    } else if (grad_cb_shot != nullptr) {
      grad_cb_shot[i] += scaled;
    } else {
      atomic_add_tide(&grad_cb[j], scaled);
    }
  }

#undef LHX_BG
#undef LHY_BG
#undef LHZ_BG
}

__global__ void add_eta_source_3d(
    TIDE_DTYPE *__restrict const eta_ex,
    TIDE_DTYPE *__restrict const eta_ey,
    TIDE_DTYPE *__restrict const eta_ez,
    TIDE_DTYPE const *__restrict const eta_source_ex,
    TIDE_DTYPE const *__restrict const eta_source_ey,
    TIDE_DTYPE const *__restrict const eta_source_ez) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = n_shots * shot_numel;
  if (i >= total) {
    return;
  }
  eta_ex[i] += eta_source_ex[i];
  eta_ey[i] += eta_source_ey[i];
  eta_ez[i] += eta_source_ez[i];
}

__global__ void combine_grad_shot_3d(TIDE_DTYPE *__restrict const grad,
                                     TIDE_DTYPE const *__restrict const grad_shot) {
  int64_t idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (idx >= shot_numel) {
    return;
  }

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
    TIDE_DTYPE *__restrict const grad_eps,
    TIDE_DTYPE *__restrict const grad_sigma,
    TIDE_DTYPE const dt_h,
    bool const ca_requires_grad,
    bool const cb_requires_grad) {
  int64_t idx = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  int64_t total = ca_batched ? (n_shots * shot_numel) : shot_numel;
  if (idx >= total) {
    return;
  }

  int64_t shot_idx = ca_batched ? (idx / shot_numel) : 0;
  int64_t j = ca_batched ? (idx - shot_idx * shot_numel) : idx;
  int64_t out_idx = ca_batched ? idx : j;
  int64_t ca_idx = ca_batched ? idx : j;
  int64_t cb_idx = cb_batched ? (shot_idx * shot_numel + j) : j;

  TIDE_DTYPE const ca_val = ca[ca_idx];
  TIDE_DTYPE const cb_val = cb[cb_idx];
  TIDE_DTYPE const cb_sq = cb_val * cb_val;
  TIDE_DTYPE const inv_dt = (TIDE_DTYPE)1 / dt_h;

  TIDE_DTYPE grad_ca_val = 0;
  TIDE_DTYPE grad_cb_val = 0;
  if (ca_requires_grad && grad_ca != nullptr) {
    grad_ca_val = grad_ca[out_idx];
  }
  if (cb_requires_grad && grad_cb != nullptr) {
    grad_cb_val = grad_cb[out_idx];
  }

  TIDE_DTYPE const dca_de = ((TIDE_DTYPE)1 - ca_val) * cb_val * inv_dt;
  TIDE_DTYPE const dcb_de = -cb_sq * inv_dt;
  TIDE_DTYPE const dca_ds = -((TIDE_DTYPE)0.5) * ((TIDE_DTYPE)1 + ca_val) * cb_val;
  TIDE_DTYPE const dcb_ds = -((TIDE_DTYPE)0.5) * cb_sq;

  if (grad_eps != nullptr) {
    TIDE_DTYPE const grad_e = grad_ca_val * dca_de + grad_cb_val * dcb_de;
    grad_eps[out_idx] = grad_e * kEp0;
  }
  if (grad_sigma != nullptr) {
    grad_sigma[out_idx] = grad_ca_val * dca_ds + grad_cb_val * dcb_ds;
  }
}

static void set_constants(
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const shot_numel_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h) {
  struct ConstantCache3D {
    bool initialized = false;
    TIDE_DTYPE rdz_h = (TIDE_DTYPE)0;
    TIDE_DTYPE rdy_h = (TIDE_DTYPE)0;
    TIDE_DTYPE rdx_h = (TIDE_DTYPE)0;
    int64_t n_shots_h = -1;
    int64_t nz_h = -1;
    int64_t ny_h = -1;
    int64_t nx_h = -1;
    int64_t shot_numel_h = -1;
    int64_t n_sources_per_shot_h = -1;
    int64_t n_receivers_per_shot_h = -1;
    int64_t pml_z0_h = -1;
    int64_t pml_y0_h = -1;
    int64_t pml_x0_h = -1;
    int64_t pml_z1_h = -1;
    int64_t pml_y1_h = -1;
    int64_t pml_x1_h = -1;
    bool ca_batched_h = false;
    bool cb_batched_h = false;
    bool cq_batched_h = false;
  };
  static ConstantCache3D cache{};
  if (cache.initialized && cache.rdz_h == rdz_h && cache.rdy_h == rdy_h &&
      cache.rdx_h == rdx_h && cache.n_shots_h == n_shots_h &&
      cache.nz_h == nz_h && cache.ny_h == ny_h && cache.nx_h == nx_h &&
      cache.shot_numel_h == shot_numel_h &&
      cache.n_sources_per_shot_h == n_sources_per_shot_h &&
      cache.n_receivers_per_shot_h == n_receivers_per_shot_h &&
      cache.pml_z0_h == pml_z0_h && cache.pml_y0_h == pml_y0_h &&
      cache.pml_x0_h == pml_x0_h && cache.pml_z1_h == pml_z1_h &&
      cache.pml_y1_h == pml_y1_h && cache.pml_x1_h == pml_x1_h &&
      cache.ca_batched_h == ca_batched_h &&
      cache.cb_batched_h == cb_batched_h &&
      cache.cq_batched_h == cq_batched_h) {
    return;
  }

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(rdz, &rdz_h, sizeof(TIDE_DTYPE)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(rdy, &rdy_h, sizeof(TIDE_DTYPE)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(rdx, &rdx_h, sizeof(TIDE_DTYPE)),
                            __FILE__, __LINE__);

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(n_shots, &n_shots_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(nz, &nz_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(ny, &ny_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(nx, &nx_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(shot_numel, &shot_numel_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(
      cudaMemcpyToSymbol(n_sources_per_shot, &n_sources_per_shot_h, sizeof(int64_t)),
      __FILE__, __LINE__);
  tide::cuda_check_or_abort(
      cudaMemcpyToSymbol(n_receivers_per_shot, &n_receivers_per_shot_h, sizeof(int64_t)),
      __FILE__, __LINE__);

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_z0, &pml_z0_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_y0, &pml_y0_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_x0, &pml_x0_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_z1, &pml_z1_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_y1, &pml_y1_h, sizeof(int64_t)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(pml_x1, &pml_x1_h, sizeof(int64_t)),
                            __FILE__, __LINE__);

  tide::cuda_check_or_abort(cudaMemcpyToSymbol(ca_batched, &ca_batched_h, sizeof(bool)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(cb_batched, &cb_batched_h, sizeof(bool)),
                            __FILE__, __LINE__);
  tide::cuda_check_or_abort(cudaMemcpyToSymbol(cq_batched, &cq_batched_h, sizeof(bool)),
                            __FILE__, __LINE__);

  cache.initialized = true;
  cache.rdz_h = rdz_h;
  cache.rdy_h = rdy_h;
  cache.rdx_h = rdx_h;
  cache.n_shots_h = n_shots_h;
  cache.nz_h = nz_h;
  cache.ny_h = ny_h;
  cache.nx_h = nx_h;
  cache.shot_numel_h = shot_numel_h;
  cache.n_sources_per_shot_h = n_sources_per_shot_h;
  cache.n_receivers_per_shot_h = n_receivers_per_shot_h;
  cache.pml_z0_h = pml_z0_h;
  cache.pml_y0_h = pml_y0_h;
  cache.pml_x0_h = pml_x0_h;
  cache.pml_z1_h = pml_z1_h;
  cache.pml_y1_h = pml_y1_h;
  cache.pml_x1_h = pml_x1_h;
  cache.ca_batched_h = ca_batched_h;
  cache.cb_batched_h = cb_batched_h;
  cache.cq_batched_h = cq_batched_h;
}

static void set_uniform_coeff_constants(
    TIDE_DTYPE const ca_uniform_h,
    TIDE_DTYPE const cb_uniform_h,
    TIDE_DTYPE const cq_uniform_h) {
  struct UniformCoeffCache3D {
    bool initialized = false;
    TIDE_DTYPE ca_uniform_h = (TIDE_DTYPE)0;
    TIDE_DTYPE cb_uniform_h = (TIDE_DTYPE)0;
    TIDE_DTYPE cq_uniform_h = (TIDE_DTYPE)0;
  };
  static UniformCoeffCache3D cache{};
  if (cache.initialized && cache.ca_uniform_h == ca_uniform_h &&
      cache.cb_uniform_h == cb_uniform_h &&
      cache.cq_uniform_h == cq_uniform_h) {
    return;
  }

  tide::cuda_check_or_abort(
      cudaMemcpyToSymbol(ca_uniform_coeff, &ca_uniform_h, sizeof(TIDE_DTYPE)),
      __FILE__, __LINE__);
  tide::cuda_check_or_abort(
      cudaMemcpyToSymbol(cb_uniform_coeff, &cb_uniform_h, sizeof(TIDE_DTYPE)),
      __FILE__, __LINE__);
  tide::cuda_check_or_abort(
      cudaMemcpyToSymbol(cq_uniform_coeff, &cq_uniform_h, sizeof(TIDE_DTYPE)),
      __FILE__, __LINE__);

  cache.initialized = true;
  cache.ca_uniform_h = ca_uniform_h;
  cache.cb_uniform_h = cb_uniform_h;
  cache.cq_uniform_h = cq_uniform_h;
}

struct ScalarLaunchConfig3D {
  int threads_cells;
  int64_t blocks_cells;
  dim3 cell_grid;
  dim3 cell_block;
  int spatial_threads_cells;
  int spatial_block_x;
  int spatial_block_y;
  int spatial_block_z;
  int64_t spatial_blocks_z_per_shot;
  bool spatial_cells;
  int threads_sr;
  int64_t blocks_sources;
  int64_t blocks_receivers;
};

static inline int normalize_cell_threads_3d(int64_t const n_threads) {
  if (n_threads <= 0) {
    return 256;
  }
  if (n_threads <= 32) {
    return 32;
  }
  if (n_threads <= 64) {
    return 64;
  }
  if (n_threads <= 128) {
    return 128;
  }
  if (n_threads <= 256) {
    return 256;
  }
  return 512;
}

static inline dim3 make_spatial_cell_block_3d(int const threads_cells) {
  if (threads_cells <= 32) {
    return dim3(32, 1, 1);
  }
  if (threads_cells <= 64) {
    return dim3(32, 2, 1);
  }
  if (threads_cells <= 128) {
    return dim3(32, 4, 1);
  }
  if (threads_cells <= 256) {
    return dim3(32, 8, 1);
  }
  return dim3(32, 8, 2);
}

static inline int64_t ceil_div_i64(int64_t const num, int64_t const den) {
  return (num + den - 1) / den;
}

static inline bool spatial_cell_launch_requested_3d() {
  char const *const value = std::getenv("TIDE_EM3D_SPATIAL_LAUNCH");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static inline ScalarLaunchConfig3D make_scalar_launch_config_3d(
    int64_t const n_shots_h, int64_t const nz_h, int64_t const ny_h,
    int64_t const nx_h, int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h, int64_t const n_threads_h,
    bool const enable_spatial_cells) {
  ScalarLaunchConfig3D cfg{};
  cfg.threads_cells = normalize_cell_threads_3d(n_threads_h);
  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const total_cells = n_shots_h * shot_numel_h;
  cfg.blocks_cells = (total_cells + cfg.threads_cells - 1) / cfg.threads_cells;
  cfg.spatial_cells = enable_spatial_cells && n_threads_h > 0;
  if (cfg.spatial_cells) {
    cfg.cell_block = make_spatial_cell_block_3d(cfg.threads_cells);
    cfg.spatial_threads_cells =
        (int)cfg.cell_block.x * (int)cfg.cell_block.y * (int)cfg.cell_block.z;
    cfg.spatial_block_x = (int)cfg.cell_block.x;
    cfg.spatial_block_y = (int)cfg.cell_block.y;
    cfg.spatial_block_z = (int)cfg.cell_block.z;
    cfg.spatial_blocks_z_per_shot = ceil_div_i64(nz_h, cfg.cell_block.z);
    cfg.cell_grid =
        dim3((unsigned)ceil_div_i64(nx_h, cfg.cell_block.x),
             (unsigned)ceil_div_i64(ny_h, cfg.cell_block.y),
             (unsigned)(n_shots_h * cfg.spatial_blocks_z_per_shot));
  } else {
    cfg.cell_block = dim3((unsigned)cfg.threads_cells, 1, 1);
    cfg.cell_grid = dim3((unsigned)cfg.blocks_cells, 1, 1);
    cfg.spatial_threads_cells = 0;
    cfg.spatial_block_x = 0;
    cfg.spatial_block_y = 0;
    cfg.spatial_block_z = 0;
    cfg.spatial_blocks_z_per_shot = 0;
  }

  cfg.threads_sr = 256;
  int64_t const total_sources = n_shots_h * n_sources_per_shot_h;
  int64_t const total_receivers = n_shots_h * n_receivers_per_shot_h;
  cfg.blocks_sources =
      total_sources > 0 ? (total_sources + cfg.threads_sr - 1) / cfg.threads_sr
                        : 1;
  cfg.blocks_receivers =
      total_receivers > 0 ? (total_receivers + cfg.threads_sr - 1) / cfg.threads_sr
                          : 1;
  return cfg;
}
}  // namespace

extern "C" void FUNC(forward)(
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
    TIDE_DTYPE const *const debye_a,
    TIDE_DTYPE const *const debye_b,
    TIDE_DTYPE const *const debye_cp,
    TIDE_DTYPE *const pol_ex,
    TIDE_DTYPE *const pol_ey,
    TIDE_DTYPE *const pol_ez,
    TIDE_DTYPE *const ex_prev,
    TIDE_DTYPE *const ey_prev,
    TIDE_DTYPE *const ez_prev,
    TIDE_DTYPE *const r,
    int64_t const n_poles,
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
    bool const uniform_coeffs,
    TIDE_DTYPE const ca_uniform,
    TIDE_DTYPE const cb_uniform,
    TIDE_DTYPE const cq_uniform,
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    bool const has_dispersion,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle) {
  (void)dt_h;
  (void)step_ratio_h;
  (void)execution_backend;

  cudaSetDevice((int)device);
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *source_field = ey;
  if (source_component == 0) {
    source_field = ex;
  } else if (source_component == 2) {
    source_field = ez;
  }

  TIDE_DTYPE const *receiver_field = ey;
  if (receiver_component == 0) {
    receiver_field = ex;
  } else if (receiver_component == 2) {
    receiver_field = ez;
  }

  bool const uniform_coeffs_h = uniform_coeffs && !has_dispersion;
  if (uniform_coeffs_h) {
    set_uniform_coeff_constants(ca_uniform, cb_uniform, cq_uniform);
  }
  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, nz_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h, n_threads,
      spatial_cell_launch_requested_3d() && n_shots_h == 1 &&
          !has_dispersion && TIDE_STENCIL == 2);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    if (uniform_coeffs_h) {
      forward_kernel_h<true>
          <<<launch_cfg.cell_grid, launch_cfg.cell_block, 0, stream_compute>>>(
              cq,
              ex,
              ey,
              ez,
              hx,
              hy,
              hz,
              m_ey_z,
              m_ez_y,
              m_ez_x,
              m_ex_z,
              m_ex_y,
              m_ey_x,
              azh,
              bzh,
              ayh,
              byh,
              axh,
              bxh,
              kzh,
              kyh,
              kxh);
    } else {
      forward_kernel_h<><<<launch_cfg.cell_grid, launch_cfg.cell_block, 0,
                    stream_compute>>>(
          cq,
          ex,
          ey,
          ez,
          hx,
          hy,
          hz,
          m_ey_z,
          m_ez_y,
          m_ez_x,
          m_ex_z,
          m_ex_y,
          m_ey_x,
          azh,
          bzh,
          ayh,
          byh,
          axh,
          bxh,
          kzh,
          kyh,
          kxh);
    }

    if (has_dispersion) {
      size_t const field_bytes =
          (size_t)(n_shots_h * shot_numel_h) * sizeof(TIDE_DTYPE);
      tide::cuda_check_or_abort(
          cudaMemcpyAsync(ex_prev, ex, field_bytes, cudaMemcpyDeviceToDevice,
                          stream_compute),
          __FILE__, __LINE__);
      tide::cuda_check_or_abort(
          cudaMemcpyAsync(ey_prev, ey, field_bytes, cudaMemcpyDeviceToDevice,
                          stream_compute),
          __FILE__, __LINE__);
      tide::cuda_check_or_abort(
          cudaMemcpyAsync(ez_prev, ez, field_bytes, cudaMemcpyDeviceToDevice,
                          stream_compute),
          __FILE__, __LINE__);
      forward_kernel_e_debye<<<launch_cfg.cell_grid,
                               launch_cfg.cell_block, 0,
                               stream_compute>>>(
          ca,
          cb,
          debye_cp,
          ex,
          ey,
          ez,
          hx,
          hy,
          hz,
          m_hy_z,
          m_hz_y,
          m_hz_x,
          m_hx_z,
          m_hx_y,
          m_hy_x,
          pol_ex,
          pol_ey,
          pol_ez,
          ex_prev,
          ey_prev,
          ez_prev,
          az,
          bz,
          ay,
          by,
          ax,
          bx,
          kz,
          ky,
          kx,
          n_poles);
    } else {
      if (uniform_coeffs_h) {
        forward_kernel_e<true>
            <<<launch_cfg.cell_grid, launch_cfg.cell_block, 0, stream_compute>>>(
                ca,
                cb,
                ex,
                ey,
                ez,
                hx,
                hy,
                hz,
                m_hy_z,
                m_hz_y,
                m_hz_x,
                m_hx_z,
                m_hx_y,
                m_hy_x,
                az,
                bz,
                ay,
                by,
                ax,
                bx,
                kz,
                ky,
                kx);
      } else {
        forward_kernel_e<><<<launch_cfg.cell_grid, launch_cfg.cell_block, 0,
                      stream_compute>>>(
            ca,
            cb,
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
            m_hy_z,
            m_hz_y,
            m_hz_x,
            m_hx_z,
            m_hx_y,
            m_hy_x,
            az,
            bz,
            ay,
            by,
            ax,
            bx,
            kz,
            ky,
            kx);
      }
    }

    if (n_sources_per_shot_h > 0 && f != nullptr && sources_i != nullptr) {
      add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                              launch_cfg.threads_sr, 0, stream_compute>>>(
          source_field,
          f + t * n_shots_h * n_sources_per_shot_h,
          sources_i);
    }

    if (has_dispersion) {
      int64_t const total_pol = n_shots_h * n_poles * shot_numel_h;
      int64_t const blocks_pol =
          total_pol > 0 ? (total_pol + launch_cfg.threads_cells - 1) /
                              launch_cfg.threads_cells
                        : 1;
      update_polarization_debye_3d<<<(unsigned)blocks_pol,
                                     launch_cfg.threads_cells, 0,
                                     stream_compute>>>(
          ex_prev, ex, debye_a, debye_b, pol_ex, n_poles);
      update_polarization_debye_3d<<<(unsigned)blocks_pol,
                                     launch_cfg.threads_cells, 0,
                                     stream_compute>>>(
          ey_prev, ey, debye_a, debye_b, pol_ey, n_poles);
      update_polarization_debye_3d<<<(unsigned)blocks_pol,
                                     launch_cfg.threads_cells, 0,
                                     stream_compute>>>(
          ez_prev, ez, debye_a, debye_b, pol_ez, n_poles);
    }

    if (n_receivers_per_shot_h > 0 && r != nullptr && receivers_i != nullptr) {
      record_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                   launch_cfg.threads_sr, 0,
                                   stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h,
          receiver_field,
          receivers_i);
    }
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(forward_with_storage)(
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
    TIDE_DTYPE *const store_1,
    TIDE_DTYPE *const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *const store_3,
    TIDE_DTYPE *const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *const store_5,
    TIDE_DTYPE *const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *const store_7,
    TIDE_DTYPE *const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *const store_9,
    TIDE_DTYPE *const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *const store_11,
    TIDE_DTYPE *const store_12,
    char **store_filenames_6,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)dt_h;

  cudaSetDevice((int)device);
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const physical_nz_h = pml_z1_h - pml_z0_h;
  int64_t const physical_ny_h = pml_y1_h - pml_y0_h;
  int64_t const physical_nx_h = pml_x1_h - pml_x0_h;
  int64_t const physical_numel_h =
      physical_nz_h > 0 && physical_ny_h > 0 && physical_nx_h > 0
          ? physical_nz_h * physical_ny_h * physical_nx_h
          : 0;
  int64_t const step_ratio_eff = step_ratio_h > 0 ? step_ratio_h : 1;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp * (size_t)n_shots_h;
  size_t const full_bytes_per_shot = (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  size_t const bf16_bytes_per_shot = (size_t)shot_numel_h * sizeof(__nv_bfloat16);
  size_t const physical_full_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(TIDE_DTYPE);
  size_t const physical_bf16_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(__nv_bfloat16);
  bool const storage_bf16 = storage_format_h == STORAGE_FORMAT_BF16;
  bool const storage_full = storage_format_h == STORAGE_FORMAT_FULL;
  bool const storage_full_domain =
      ((storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot) ||
       (storage_bf16 && shot_bytes_uncomp == (int64_t)bf16_bytes_per_shot));
  bool const storage_physical =
      (physical_numel_h > 0 && physical_numel_h != shot_numel_h) &&
      ((storage_full &&
        shot_bytes_uncomp == (int64_t)physical_full_bytes_per_shot) ||
       (storage_bf16 &&
        shot_bytes_uncomp == (int64_t)physical_bf16_bytes_per_shot));
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      (storage_full_domain || storage_physical);
  bool const host_backed_storage =
      (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) &&
      storage_full_domain;
  bool const eonly_snapshot =
      (execution_backend == 1 || execution_backend == 2) && storage_direct &&
      !storage_physical &&
      step_ratio_eff == 1 &&
      ca_requires_grad && cb_requires_grad;
  bool const use_storage_pipeline =
      host_backed_storage && (ca_requires_grad || cb_requires_grad) &&
      stream_storage != nullptr && stream_storage != stream_compute;

  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *source_field = ey;
  if (source_component == 0) {
    source_field = ex;
  } else if (source_component == 2) {
    source_field = ez;
  }

  TIDE_DTYPE const *receiver_field = ey;
  if (receiver_component == 0) {
    receiver_field = ex;
  } else if (receiver_component == 2) {
    receiver_field = ez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, nz_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h, n_threads,
      spatial_cell_launch_requested_3d() && n_shots_h == 1 &&
          TIDE_STENCIL == 2);

  void *const device_stores[6] = {store_1, store_3, store_5,
                                  store_7, store_9, store_11};
  void *const host_stores[6] = {store_2, store_4, store_6,
                                store_8, store_10, store_12};
  char **const store_filenames[6] = {store_filenames_1, store_filenames_2,
                                     store_filenames_3, store_filenames_4,
                                     store_filenames_5, store_filenames_6};
  bool const component_required[6] = {ca_requires_grad, ca_requires_grad,
                                      ca_requires_grad,
                                      cb_requires_grad && !eonly_snapshot,
                                      cb_requires_grad && !eonly_snapshot,
                                      cb_requires_grad && !eonly_snapshot};
  void *async_disk_handles[6] = {};
  if (storage_mode == STORAGE_DISK) {
    for (int i = 0; i < 6; ++i) {
      if (component_required[i] && store_filenames[i] != nullptr) {
        async_disk_handles[i] =
            storage_async_disk_open(store_filenames[i][0], true, NUM_BUFFERS);
      }
    }
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

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    bool const do_store =
        (storage_direct || host_backed_storage) && ((t % step_ratio_eff) == 0);
    int slot = 0;
    cudaEvent_t slot_storage_done = nullptr;
    cudaEvent_t slot_compute_done = nullptr;
    if (do_store && use_storage_pipeline) {
      slot = (int)((t / step_ratio_eff) % NUM_BUFFERS);
      slot_storage_done = storage_done_events.events[slot];
      slot_compute_done = compute_done_events.events[slot];
      tide::cuda_check_or_abort(
          cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
          __LINE__);
    }

    forward_kernel_h<false><<<launch_cfg.cell_grid, launch_cfg.cell_block, 0,
                       stream_compute>>>(
        cq,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        m_ey_z,
        m_ez_y,
        m_ez_x,
        m_ex_z,
        m_ex_y,
        m_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    if (do_store) {
      int64_t const store_idx = t / step_ratio_eff;
      size_t const device_offset = ring_storage_offset_bytes(
          store_idx, storage_mode, bytes_per_step_store);
      size_t const host_offset = cpu_linear_storage_offset_bytes(
          store_idx, storage_mode, bytes_per_step_store);
      void *const ex_store_t =
          (ca_requires_grad && device_stores[0] != nullptr)
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(device_stores[0]) + device_offset)
              : nullptr;
      void *const ey_store_t =
          (ca_requires_grad && device_stores[1] != nullptr)
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(device_stores[1]) + device_offset)
              : nullptr;
      void *const ez_store_t =
          (ca_requires_grad && device_stores[2] != nullptr)
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(device_stores[2]) + device_offset)
              : nullptr;
      void *const curl_x_store_t =
          (cb_requires_grad && !eonly_snapshot && device_stores[3] != nullptr)
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(device_stores[3]) + device_offset)
              : nullptr;
      void *const curl_y_store_t =
          (cb_requires_grad && !eonly_snapshot && device_stores[4] != nullptr)
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(device_stores[4]) + device_offset)
              : nullptr;
      void *const curl_z_store_t =
          (cb_requires_grad && !eonly_snapshot && device_stores[5] != nullptr)
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(device_stores[5]) + device_offset)
              : nullptr;

      if (storage_bf16) {
        if (storage_physical) {
          forward_kernel_e_with_storage<__nv_bfloat16, true>
              <<<launch_cfg.cell_grid, launch_cfg.cell_block, 0, stream_compute>>>(
                  ca,
                  cb,
                  ex,
                  ey,
                  ez,
                  hx,
                  hy,
                  hz,
                  m_hy_z,
                  m_hz_y,
                  m_hz_x,
                  m_hx_z,
                  m_hx_y,
                  m_hy_x,
                  az,
                  bz,
                  ay,
                  by,
                  ax,
                  bx,
                  kz,
                  ky,
                  kx,
                  reinterpret_cast<__nv_bfloat16 *>(ex_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(ey_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(ez_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(curl_x_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(curl_y_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(curl_z_store_t),
                  ca_requires_grad,
                  cb_requires_grad && !eonly_snapshot);
        } else {
          forward_kernel_e_with_storage<__nv_bfloat16, false>
              <<<launch_cfg.cell_grid, launch_cfg.cell_block, 0, stream_compute>>>(
                  ca,
                  cb,
                  ex,
                  ey,
                  ez,
                  hx,
                  hy,
                  hz,
                  m_hy_z,
                  m_hz_y,
                  m_hz_x,
                  m_hx_z,
                  m_hx_y,
                  m_hy_x,
                  az,
                  bz,
                  ay,
                  by,
                  ax,
                  bx,
                  kz,
                  ky,
                  kx,
                  reinterpret_cast<__nv_bfloat16 *>(ex_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(ey_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(ez_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(curl_x_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(curl_y_store_t),
                  reinterpret_cast<__nv_bfloat16 *>(curl_z_store_t),
                  ca_requires_grad,
                  cb_requires_grad && !eonly_snapshot);
        }
      } else {
        if (storage_physical) {
          forward_kernel_e_with_storage<TIDE_DTYPE, true>
              <<<launch_cfg.cell_grid, launch_cfg.cell_block, 0, stream_compute>>>(
                  ca,
                  cb,
                  ex,
                  ey,
                  ez,
                  hx,
                  hy,
                  hz,
                  m_hy_z,
                  m_hz_y,
                  m_hz_x,
                  m_hx_z,
                  m_hx_y,
                  m_hy_x,
                  az,
                  bz,
                  ay,
                  by,
                  ax,
                  bx,
                  kz,
                  ky,
                  kx,
                  reinterpret_cast<TIDE_DTYPE *>(ex_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(ey_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(ez_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(curl_x_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(curl_y_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(curl_z_store_t),
                  ca_requires_grad,
                  cb_requires_grad && !eonly_snapshot);
        } else {
          forward_kernel_e_with_storage<TIDE_DTYPE, false>
              <<<launch_cfg.cell_grid, launch_cfg.cell_block, 0, stream_compute>>>(
                  ca,
                  cb,
                  ex,
                  ey,
                  ez,
                  hx,
                  hy,
                  hz,
                  m_hy_z,
                  m_hz_y,
                  m_hz_x,
                  m_hx_z,
                  m_hx_y,
                  m_hy_x,
                  az,
                  bz,
                  ay,
                  by,
                  ax,
                  bx,
                  kz,
                  ky,
                  kx,
                  reinterpret_cast<TIDE_DTYPE *>(ex_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(ey_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(ez_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(curl_x_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(curl_y_store_t),
                  reinterpret_cast<TIDE_DTYPE *>(curl_z_store_t),
                  ca_requires_grad,
                  cb_requires_grad && !eonly_snapshot);
        }
      }

      if (host_backed_storage) {
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

        void *const current_device[6] = {ex_store_t, ey_store_t, ez_store_t,
                                         curl_x_store_t, curl_y_store_t,
                                         curl_z_store_t};
        for (int i = 0; i < 6; ++i) {
          if (!component_required[i] || current_device[i] == nullptr ||
              host_stores[i] == nullptr) {
            continue;
          }
          void *const current_host =
              reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[i]) +
                                       host_offset);
          if (storage_mode == STORAGE_DISK) {
            int64_t const file_offset =
                store_idx * (int64_t)bytes_per_step_store;
            storage_async_disk_wait_slot(async_disk_handles[i], slot);
            tide::cuda_check_or_abort(
                cudaMemcpyAsync(current_host, current_device[i],
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
                async_disk_handles[i], slot, current_host, bytes_per_step_store,
                file_offset, ready_event);
          } else {
            storage_copy_snapshot_d2h(current_device[i], current_host,
                                      (size_t)shot_bytes_uncomp,
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
      forward_kernel_e<false><<<launch_cfg.cell_grid, launch_cfg.cell_block, 0,
                         stream_compute>>>(
          ca,
          cb,
          ex,
          ey,
          ez,
          hx,
          hy,
          hz,
          m_hy_z,
          m_hz_y,
          m_hz_x,
          m_hx_z,
          m_hx_y,
          m_hy_x,
          az,
          bz,
          ay,
          by,
          ax,
          bx,
          kz,
          ky,
          kx);
    }

    if (n_sources_per_shot_h > 0 && f != nullptr && sources_i != nullptr) {
      add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                              launch_cfg.threads_sr, 0, stream_compute>>>(
          source_field,
          f + t * n_shots_h * n_sources_per_shot_h,
          sources_i);
    }
    if (n_receivers_per_shot_h > 0 && r != nullptr && receivers_i != nullptr) {
      record_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                   launch_cfg.threads_sr, 0, stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h,
          receiver_field,
          receivers_i);
    }
  }

  if (eonly_snapshot && store_7 != nullptr && store_9 != nullptr &&
      store_11 != nullptr) {
    if (storage_bf16) {
      store_final_e_3d<__nv_bfloat16>
          <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
             stream_compute>>>(
              ex,
              ey,
              ez,
              reinterpret_cast<__nv_bfloat16 *>(store_7),
              reinterpret_cast<__nv_bfloat16 *>(store_9),
              reinterpret_cast<__nv_bfloat16 *>(store_11));
    } else {
      store_final_e_3d<TIDE_DTYPE>
          <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
             stream_compute>>>(ex, ey, ez, store_7, store_9, store_11);
    }
  }

  if (use_storage_pipeline) {
    tide::cuda_check_or_abort(cudaStreamSynchronize(stream_storage), __FILE__,
                              __LINE__);
  }
  for (void *&handle : async_disk_handles) {
    storage_async_disk_close(handle);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(born_forward)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb,
    TIDE_DTYPE const *const f0,
    TIDE_DTYPE const *const df,
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
    TIDE_DTYPE *const dex,
    TIDE_DTYPE *const dey,
    TIDE_DTYPE *const dez,
    TIDE_DTYPE *const dhx,
    TIDE_DTYPE *const dhy,
    TIDE_DTYPE *const dhz,
    TIDE_DTYPE *const dm_hz_y,
    TIDE_DTYPE *const dm_hy_z,
    TIDE_DTYPE *const dm_hx_z,
    TIDE_DTYPE *const dm_hz_x,
    TIDE_DTYPE *const dm_hy_x,
    TIDE_DTYPE *const dm_hx_y,
    TIDE_DTYPE *const dm_ey_z,
    TIDE_DTYPE *const dm_ez_y,
    TIDE_DTYPE *const dm_ez_x,
    TIDE_DTYPE *const dm_ex_z,
    TIDE_DTYPE *const dm_ex_y,
    TIDE_DTYPE *const dm_ey_x,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle) {
  (void)dt_h;
  (void)step_ratio_h;
  (void)n_threads;
  (void)execution_backend;

  cudaSetDevice((int)device);
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *source_field_bg = ey;
  TIDE_DTYPE *source_field_sc = dey;
  if (source_component == 0) {
    source_field_bg = ex;
    source_field_sc = dex;
  } else if (source_component == 2) {
    source_field_bg = ez;
    source_field_sc = dez;
  }

  TIDE_DTYPE const *receiver_field_sc = dey;
  if (receiver_component == 0) {
    receiver_field_sc = dex;
  } else if (receiver_component == 2) {
    receiver_field_sc = dez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, nz_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h, n_threads, false);

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    forward_kernel_h<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        m_ey_z,
        m_ez_y,
        m_ez_x,
        m_ex_z,
        m_ex_y,
        m_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    born_forward_kernel_h<<<(unsigned)launch_cfg.blocks_cells,
                            launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        dex,
        dey,
        dez,
        dhx,
        dhy,
        dhz,
        dm_ey_z,
        dm_ez_y,
        dm_ez_x,
        dm_ex_z,
        dm_ex_y,
        dm_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    born_forward_kernel_e<<<(unsigned)launch_cfg.blocks_cells,
                            launch_cfg.threads_cells, 0, stream_compute>>>(
        ca,
        cb,
        dca,
        dcb,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        m_hy_z,
        m_hz_y,
        m_hz_x,
        m_hx_z,
        m_hx_y,
        m_hy_x,
        dex,
        dey,
        dez,
        dhx,
        dhy,
        dhz,
        dm_hy_z,
        dm_hz_y,
        dm_hz_x,
        dm_hx_z,
        dm_hx_y,
        dm_hy_x,
        az,
        bz,
        ay,
        by,
        ax,
        bx,
        kz,
        ky,
        kx);

    if (n_sources_per_shot_h > 0 && sources_i != nullptr) {
      if (f0 != nullptr) {
        add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                                launch_cfg.threads_sr, 0, stream_compute>>>(
            source_field_bg,
            f0 + t * n_shots_h * n_sources_per_shot_h,
            sources_i);
      }
      if (df != nullptr) {
        add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                                launch_cfg.threads_sr, 0, stream_compute>>>(
            source_field_sc,
            df + t * n_shots_h * n_sources_per_shot_h,
            sources_i);
      }
    }

    if (n_receivers_per_shot_h > 0 && r != nullptr && receivers_i != nullptr) {
      record_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                   launch_cfg.threads_sr, 0, stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h,
          receiver_field_sc,
          receivers_i);
    }
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(born_forward_with_storage)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb,
    TIDE_DTYPE const *const f0,
    TIDE_DTYPE const *const df,
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
    TIDE_DTYPE *const dex,
    TIDE_DTYPE *const dey,
    TIDE_DTYPE *const dez,
    TIDE_DTYPE *const dhx,
    TIDE_DTYPE *const dhy,
    TIDE_DTYPE *const dhz,
    TIDE_DTYPE *const dm_hz_y,
    TIDE_DTYPE *const dm_hy_z,
    TIDE_DTYPE *const dm_hx_z,
    TIDE_DTYPE *const dm_hz_x,
    TIDE_DTYPE *const dm_hy_x,
    TIDE_DTYPE *const dm_hx_y,
    TIDE_DTYPE *const dm_ey_z,
    TIDE_DTYPE *const dm_ez_y,
    TIDE_DTYPE *const dm_ez_x,
    TIDE_DTYPE *const dm_ex_z,
    TIDE_DTYPE *const dm_ex_y,
    TIDE_DTYPE *const dm_ey_x,
    TIDE_DTYPE *const r,
    TIDE_DTYPE *const store_1,
    TIDE_DTYPE *const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *const store_3,
    TIDE_DTYPE *const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *const store_5,
    TIDE_DTYPE *const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *const store_7,
    TIDE_DTYPE *const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *const store_9,
    TIDE_DTYPE *const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *const store_11,
    TIDE_DTYPE *const store_12,
    char **store_filenames_6,
    void *const dstore_ex,
    void *const dstore_ey,
    void *const dstore_ez,
    void *const dstore_curl_x,
    void *const dstore_curl_y,
    void *const dstore_curl_z,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)dt_h;
  (void)n_threads;
  (void)execution_backend;

  cudaSetDevice((int)device);
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const step_ratio_eff = step_ratio_h > 0 ? step_ratio_h : 1;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp * (size_t)n_shots_h;
  size_t const full_bytes_per_shot = (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  size_t const bf16_bytes_per_shot = (size_t)shot_numel_h * sizeof(__nv_bfloat16);
  bool const storage_bf16 = storage_format_h == STORAGE_FORMAT_BF16;
  bool const storage_full = storage_format_h == STORAGE_FORMAT_FULL;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      ((storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot) ||
       (storage_bf16 && shot_bytes_uncomp == (int64_t)bf16_bytes_per_shot));
  bool const storage_full_domain =
      storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot;
  bool const host_backed_storage =
      (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) &&
      storage_full_domain;
  bool const use_storage_pipeline =
      host_backed_storage && (ca_requires_grad || cb_requires_grad) &&
      stream_storage != nullptr && stream_storage != stream_compute;

  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *source_field_bg = ey;
  TIDE_DTYPE *source_field_sc = dey;
  if (source_component == 0) {
    source_field_bg = ex;
    source_field_sc = dex;
  } else if (source_component == 2) {
    source_field_bg = ez;
    source_field_sc = dez;
  }

  TIDE_DTYPE const *receiver_field_sc = dey;
  if (receiver_component == 0) {
    receiver_field_sc = dex;
  } else if (receiver_component == 2) {
    receiver_field_sc = dez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, nz_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h, n_threads, false);

  void *const host_stores[6] = {store_2, store_4, store_6,
                                store_8, store_10, store_12};
  char **const store_filenames[6] = {store_filenames_1, store_filenames_2,
                                     store_filenames_3, store_filenames_4,
                                     store_filenames_5, store_filenames_6};
  bool const component_required[6] = {ca_requires_grad, ca_requires_grad,
                                      ca_requires_grad, cb_requires_grad,
                                      cb_requires_grad, cb_requires_grad};
  void *async_disk_handles[6] = {};
  if (storage_mode == STORAGE_DISK) {
    for (int i = 0; i < 6; ++i) {
      if (component_required[i] && store_filenames[i] != nullptr) {
        async_disk_handles[i] =
            storage_async_disk_open(store_filenames[i][0], true, NUM_BUFFERS);
      }
    }
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

  for (int64_t t = start_t; t < start_t + nt; ++t) {
    bool const do_store =
        (storage_direct || host_backed_storage) && ((t % step_ratio_eff) == 0);
    int slot = 0;
    cudaEvent_t slot_storage_done = nullptr;
    cudaEvent_t slot_compute_done = nullptr;
    if (do_store && use_storage_pipeline) {
      slot = (int)((t / step_ratio_eff) % NUM_BUFFERS);
      slot_storage_done = storage_done_events.events[slot];
      slot_compute_done = compute_done_events.events[slot];
      tide::cuda_check_or_abort(
          cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
          __LINE__);
    }
    int64_t const store_idx = do_store ? (t / step_ratio_eff) : 0;
    size_t const device_offset =
        do_store ? ring_storage_offset_bytes(store_idx, storage_mode,
                                             bytes_per_step_store)
                 : 0;
    size_t const direct_device_offset =
        do_store ? (size_t)store_idx * bytes_per_step_store : 0;
    size_t const host_offset =
        do_store ? cpu_linear_storage_offset_bytes(store_idx, storage_mode,
                                                   bytes_per_step_store)
                 : 0;

    forward_kernel_h<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        m_ey_z,
        m_ez_y,
        m_ez_x,
        m_ex_z,
        m_ex_y,
        m_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    born_forward_kernel_h<<<(unsigned)launch_cfg.blocks_cells,
                            launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        dex,
        dey,
        dez,
        dhx,
        dhy,
        dhz,
        dm_ey_z,
        dm_ez_y,
        dm_ez_x,
        dm_ex_z,
        dm_ex_y,
        dm_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    if (do_store) {
      void *const ex_store_t =
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(store_1) + device_offset);
      void *const ey_store_t =
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(store_3) + device_offset);
      void *const ez_store_t =
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(store_5) + device_offset);
      void *const curl_x_store_t =
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(store_7) + device_offset);
      void *const curl_y_store_t =
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(store_9) + device_offset);
      void *const curl_z_store_t =
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(store_11) + device_offset);
      void *const dex_store_t =
          dstore_ex != nullptr
              ? reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(dstore_ex) +
                                         direct_device_offset)
              : nullptr;
      void *const dey_store_t =
          dstore_ey != nullptr
              ? reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(dstore_ey) +
                                         direct_device_offset)
              : nullptr;
      void *const dez_store_t =
          dstore_ez != nullptr
              ? reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(dstore_ez) +
                                         direct_device_offset)
              : nullptr;
      void *const dcurl_x_store_t =
          dstore_curl_x != nullptr
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(dstore_curl_x) +
                    direct_device_offset)
              : nullptr;
      void *const dcurl_y_store_t =
          dstore_curl_y != nullptr
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(dstore_curl_y) +
                    direct_device_offset)
              : nullptr;
      void *const dcurl_z_store_t =
          dstore_curl_z != nullptr
              ? reinterpret_cast<void *>(
                    reinterpret_cast<uint8_t *>(dstore_curl_z) +
                    direct_device_offset)
              : nullptr;

      if (storage_bf16) {
        born_forward_kernel_e_with_storage<__nv_bfloat16>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                ca,
                cb,
                dca,
                dcb,
                ex,
                ey,
                ez,
                hx,
                hy,
                hz,
                m_hy_z,
                m_hz_y,
                m_hz_x,
                m_hx_z,
                m_hx_y,
                m_hy_x,
                dex,
                dey,
                dez,
                dhx,
                dhy,
                dhz,
                dm_hy_z,
                dm_hz_y,
                dm_hz_x,
                dm_hx_z,
                dm_hx_y,
                dm_hy_x,
                az,
                bz,
                ay,
                by,
                ax,
                bx,
                kz,
                ky,
                kx,
                reinterpret_cast<__nv_bfloat16 *>(ex_store_t),
                reinterpret_cast<__nv_bfloat16 *>(ey_store_t),
                reinterpret_cast<__nv_bfloat16 *>(ez_store_t),
                reinterpret_cast<__nv_bfloat16 *>(curl_x_store_t),
                reinterpret_cast<__nv_bfloat16 *>(curl_y_store_t),
                reinterpret_cast<__nv_bfloat16 *>(curl_z_store_t),
                reinterpret_cast<__nv_bfloat16 *>(dex_store_t),
                reinterpret_cast<__nv_bfloat16 *>(dey_store_t),
                reinterpret_cast<__nv_bfloat16 *>(dez_store_t),
                reinterpret_cast<__nv_bfloat16 *>(dcurl_x_store_t),
                reinterpret_cast<__nv_bfloat16 *>(dcurl_y_store_t),
                reinterpret_cast<__nv_bfloat16 *>(dcurl_z_store_t),
                ca_requires_grad,
                cb_requires_grad);
      } else {
        born_forward_kernel_e_with_storage<TIDE_DTYPE>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                ca,
                cb,
                dca,
                dcb,
                ex,
                ey,
                ez,
                hx,
                hy,
                hz,
                m_hy_z,
                m_hz_y,
                m_hz_x,
                m_hx_z,
                m_hx_y,
                m_hy_x,
                dex,
                dey,
                dez,
                dhx,
                dhy,
                dhz,
                dm_hy_z,
                dm_hz_y,
                dm_hz_x,
                dm_hx_z,
                dm_hx_y,
                dm_hy_x,
                az,
                bz,
                ay,
                by,
                ax,
                bx,
                kz,
                ky,
                kx,
                reinterpret_cast<TIDE_DTYPE *>(ex_store_t),
                reinterpret_cast<TIDE_DTYPE *>(ey_store_t),
                reinterpret_cast<TIDE_DTYPE *>(ez_store_t),
                reinterpret_cast<TIDE_DTYPE *>(curl_x_store_t),
                reinterpret_cast<TIDE_DTYPE *>(curl_y_store_t),
                reinterpret_cast<TIDE_DTYPE *>(curl_z_store_t),
                reinterpret_cast<TIDE_DTYPE *>(dex_store_t),
                reinterpret_cast<TIDE_DTYPE *>(dey_store_t),
                reinterpret_cast<TIDE_DTYPE *>(dez_store_t),
                reinterpret_cast<TIDE_DTYPE *>(dcurl_x_store_t),
                reinterpret_cast<TIDE_DTYPE *>(dcurl_y_store_t),
                reinterpret_cast<TIDE_DTYPE *>(dcurl_z_store_t),
                ca_requires_grad,
                cb_requires_grad);
      }

      if (host_backed_storage) {
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

        void *const current_device[6] = {ex_store_t, ey_store_t, ez_store_t,
                                         curl_x_store_t, curl_y_store_t,
                                         curl_z_store_t};
        for (int i = 0; i < 6; ++i) {
          if (!component_required[i] || current_device[i] == nullptr ||
              host_stores[i] == nullptr) {
            continue;
          }
          void *const current_host =
              reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[i]) +
                                       host_offset);
          if (storage_mode == STORAGE_DISK) {
            int64_t const file_offset =
                store_idx * (int64_t)bytes_per_step_store;
            storage_async_disk_wait_slot(async_disk_handles[i], slot);
            tide::cuda_check_or_abort(
                cudaMemcpyAsync(current_host, current_device[i],
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
                async_disk_handles[i], slot, current_host, bytes_per_step_store,
                file_offset, ready_event);
            storage_async_disk_wait_slot(async_disk_handles[i], slot);
          } else {
            storage_copy_snapshot_d2h(current_device[i], current_host,
                                      (size_t)shot_bytes_uncomp,
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
      born_forward_kernel_e<<<(unsigned)launch_cfg.blocks_cells,
                              launch_cfg.threads_cells, 0, stream_compute>>>(
          ca,
          cb,
          dca,
          dcb,
          ex,
          ey,
          ez,
          hx,
          hy,
          hz,
          m_hy_z,
          m_hz_y,
          m_hz_x,
          m_hx_z,
          m_hx_y,
          m_hy_x,
          dex,
          dey,
          dez,
          dhx,
          dhy,
          dhz,
          dm_hy_z,
          dm_hz_y,
          dm_hz_x,
          dm_hx_z,
          dm_hx_y,
          dm_hy_x,
          az,
          bz,
          ay,
          by,
          ax,
          bx,
          kz,
          ky,
          kx);
    }

    if (n_sources_per_shot_h > 0 && sources_i != nullptr) {
      if (f0 != nullptr) {
        add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                                launch_cfg.threads_sr, 0, stream_compute>>>(
            source_field_bg,
            f0 + t * n_shots_h * n_sources_per_shot_h,
            sources_i);
      }
      if (df != nullptr) {
        add_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                                launch_cfg.threads_sr, 0, stream_compute>>>(
            source_field_sc,
            df + t * n_shots_h * n_sources_per_shot_h,
            sources_i);
      }
    }

    if (n_receivers_per_shot_h > 0 && r != nullptr && receivers_i != nullptr) {
      record_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                   launch_cfg.threads_sr, 0, stream_compute>>>(
          r + t * n_shots_h * n_receivers_per_shot_h,
          receiver_field_sc,
          receivers_i);
    }
  }

  if (use_storage_pipeline) {
    tide::cuda_check_or_abort(cudaStreamSynchronize(stream_storage), __FILE__,
                              __LINE__);
  }
  for (void *&handle : async_disk_handles) {
    storage_async_disk_close(handle);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(backward)(
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
    TIDE_DTYPE *const store_1,
    TIDE_DTYPE *const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *const store_3,
    TIDE_DTYPE *const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *const store_5,
    TIDE_DTYPE *const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *const store_7,
    TIDE_DTYPE *const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *const store_9,
    TIDE_DTYPE *const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *const store_11,
    TIDE_DTYPE *const store_12,
    char **store_filenames_6,
    TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_eps,
    TIDE_DTYPE *const grad_sigma,
    TIDE_DTYPE *const grad_ca_shot,
    TIDE_DTYPE *const grad_cb_shot,
    bool const zero_grad_on_entry,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  cudaSetDevice((int)device);
  (void)n_threads;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const physical_nz_h = pml_z1_h - pml_z0_h;
  int64_t const physical_ny_h = pml_y1_h - pml_y0_h;
  int64_t const physical_nx_h = pml_x1_h - pml_x0_h;
  int64_t const physical_numel_h =
      physical_nz_h > 0 && physical_ny_h > 0 && physical_nx_h > 0
          ? physical_nz_h * physical_ny_h * physical_nx_h
          : 0;
  int64_t const step_ratio_eff = step_ratio_h > 0 ? step_ratio_h : 1;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp * (size_t)n_shots_h;
  size_t const full_bytes_per_shot = (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  size_t const bf16_bytes_per_shot = (size_t)shot_numel_h * sizeof(__nv_bfloat16);
  size_t const physical_full_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(TIDE_DTYPE);
  size_t const physical_bf16_bytes_per_shot =
      (size_t)physical_numel_h * sizeof(__nv_bfloat16);
  bool const storage_bf16 = storage_format_h == STORAGE_FORMAT_BF16;
  bool const storage_full = storage_format_h == STORAGE_FORMAT_FULL;
  bool const storage_full_domain =
      ((storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot) ||
       (storage_bf16 && shot_bytes_uncomp == (int64_t)bf16_bytes_per_shot));
  bool const storage_physical =
      (physical_numel_h > 0 && physical_numel_h != shot_numel_h) &&
      ((storage_full &&
        shot_bytes_uncomp == (int64_t)physical_full_bytes_per_shot) ||
       (storage_bf16 &&
        shot_bytes_uncomp == (int64_t)physical_bf16_bytes_per_shot));
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      (storage_full_domain || storage_physical);
  bool const host_backed_storage =
      (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) &&
      storage_full_domain;
  bool const eonly_snapshot =
      (execution_backend == 1 || execution_backend == 2) && storage_direct &&
      !storage_physical &&
      step_ratio_eff == 1 &&
      ca_requires_grad && cb_requires_grad;
  bool const direct_material_grad = execution_backend == 2 && eonly_snapshot;
  bool const use_storage_pipeline =
      host_backed_storage && (ca_requires_grad || cb_requires_grad) &&
      stream_storage != nullptr && stream_storage != stream_compute;
  bool const reduce_grad_ca =
      !direct_material_grad && ca_requires_grad && !ca_batched_h &&
      grad_ca != nullptr && grad_ca_shot != nullptr;
  bool const reduce_grad_cb =
      !direct_material_grad && cb_requires_grad && !cb_batched_h &&
      grad_cb != nullptr && grad_cb_shot != nullptr;
  bool const reduce_grad_eps =
      direct_material_grad && !ca_batched_h && grad_eps != nullptr &&
      grad_ca_shot != nullptr;
  bool const reduce_grad_sigma =
      direct_material_grad && !ca_batched_h && grad_sigma != nullptr &&
      grad_cb_shot != nullptr;

  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *lambda_src_field = lambda_ey;
  if (source_component == 0) {
    lambda_src_field = lambda_ex;
  } else if (source_component == 2) {
    lambda_src_field = lambda_ez;
  }

  TIDE_DTYPE *lambda_recv_field = lambda_ey;
  if (receiver_component == 0) {
    lambda_recv_field = lambda_ex;
  } else if (receiver_component == 2) {
    lambda_recv_field = lambda_ez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, nz_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h, n_threads, false);
  int64_t const blocks_physical_cells =
      ((int64_t)n_shots_h * physical_numel_h + launch_cfg.threads_cells - 1) /
      launch_cfg.threads_cells;

  void *const device_stores[6] = {store_1, store_3, store_5,
                                  store_7, store_9, store_11};
  void *const host_stores[6] = {store_2, store_4, store_6,
                                store_8, store_10, store_12};
  char **const store_filenames[6] = {store_filenames_1, store_filenames_2,
                                     store_filenames_3, store_filenames_4,
                                     store_filenames_5, store_filenames_6};
  bool const component_required[6] = {ca_requires_grad, ca_requires_grad,
                                      ca_requires_grad,
                                      cb_requires_grad && !eonly_snapshot,
                                      cb_requires_grad && !eonly_snapshot,
                                      cb_requires_grad && !eonly_snapshot};
  void *async_disk_handles[6] = {};
  if (storage_mode == STORAGE_DISK) {
    for (int i = 0; i < 6; ++i) {
      if (component_required[i] && store_filenames[i] != nullptr) {
        async_disk_handles[i] =
            storage_async_disk_open(store_filenames[i][0], false, NUM_BUFFERS);
      }
    }
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

  int64_t const first_store_idx = (start_t - 1) / step_ratio_eff;
  int64_t const first_t_in_chunk = start_t - nt;
  int64_t const last_store_idx =
      (first_t_in_chunk + step_ratio_eff - 1) / step_ratio_eff;
  if (storage_mode == STORAGE_DISK) {
    int64_t const stored_steps_in_chunk =
        first_store_idx >= last_store_idx ? (first_store_idx - last_store_idx + 1)
                                          : 0;
    int64_t const prefetch_count =
        tide_min<int64_t>(NUM_BUFFERS, stored_steps_in_chunk);
    for (int64_t i = 0; i < prefetch_count; ++i) {
      int64_t const store_idx = first_store_idx - i;
      int const slot = (int)(store_idx % NUM_BUFFERS);
      size_t const host_offset = cpu_linear_storage_offset_bytes(
          store_idx, storage_mode, bytes_per_step_store);
      int64_t const file_offset = store_idx * (int64_t)bytes_per_step_store;
      for (int comp = 0; comp < 6; ++comp) {
        if (!component_required[comp] || host_stores[comp] == nullptr) {
          continue;
        }
        void *const host_ptr =
            reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[comp]) +
                                     host_offset);
        storage_async_disk_enqueue_read(async_disk_handles[comp], slot, host_ptr,
                                        bytes_per_step_store, file_offset,
                                        nullptr);
      }
    }
  }

  if (zero_grad_on_entry) {
    if (grad_f != nullptr && nt > 0 && n_shots_h > 0 &&
        n_sources_per_shot_h > 0) {
      tide::cuda_check_or_abort(
          cudaMemset(grad_f, 0,
                     (size_t)nt * (size_t)n_shots_h *
                         (size_t)n_sources_per_shot_h * sizeof(TIDE_DTYPE)),
          __FILE__, __LINE__);
    }
    if (!direct_material_grad && ca_requires_grad && grad_ca != nullptr) {
      tide::cuda_check_or_abort(
          cudaMemset(grad_ca, 0,
                     (size_t)(ca_batched_h ? n_shots_h : 1) *
                         (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
          __FILE__, __LINE__);
    }
    if (!direct_material_grad && cb_requires_grad && grad_cb != nullptr) {
      tide::cuda_check_or_abort(
          cudaMemset(grad_cb, 0,
                     (size_t)(cb_batched_h ? n_shots_h : 1) *
                         (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
          __FILE__, __LINE__);
    }
    if ((ca_requires_grad || cb_requires_grad || direct_material_grad) &&
        grad_eps != nullptr) {
      tide::cuda_check_or_abort(
          cudaMemset(grad_eps, 0,
                     (size_t)(ca_batched_h ? n_shots_h : 1) *
                         (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
          __FILE__, __LINE__);
    }
    if ((ca_requires_grad || cb_requires_grad || direct_material_grad) &&
        grad_sigma != nullptr) {
      tide::cuda_check_or_abort(
          cudaMemset(grad_sigma, 0,
                     (size_t)(ca_batched_h ? n_shots_h : 1) *
                         (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
          __FILE__, __LINE__);
    }
  }
  if (reduce_grad_ca || reduce_grad_eps) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_ca_shot, 0,
                   (size_t)n_shots_h * (size_t)shot_numel_h *
                       sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (reduce_grad_cb || reduce_grad_sigma) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_cb_shot, 0,
                   (size_t)n_shots_h * (size_t)shot_numel_h *
                       sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    bool const do_grad = ((t % step_ratio_eff) == 0);
    bool const direct_material_step =
        do_grad && direct_material_grad &&
        (grad_eps != nullptr || grad_sigma != nullptr) && storage_direct &&
        store_1 != nullptr && store_3 != nullptr && store_5 != nullptr &&
        store_7 != nullptr && store_9 != nullptr && store_11 != nullptr;
    bool const grad_ca_step =
        do_grad && !direct_material_grad && ca_requires_grad &&
        ((storage_direct && store_1 != nullptr && store_3 != nullptr &&
          store_5 != nullptr) ||
         (host_backed_storage && store_1 != nullptr && store_2 != nullptr &&
          store_3 != nullptr && store_4 != nullptr && store_5 != nullptr &&
          store_6 != nullptr));
    bool const grad_cb_step =
        do_grad && !direct_material_grad && cb_requires_grad &&
        (eonly_snapshot
             ? (storage_direct && store_1 != nullptr && store_3 != nullptr &&
                store_5 != nullptr && store_7 != nullptr && store_9 != nullptr &&
                store_11 != nullptr)
             : ((storage_direct && store_7 != nullptr && store_9 != nullptr &&
                 store_11 != nullptr) ||
                (host_backed_storage && store_7 != nullptr &&
                 store_8 != nullptr && store_9 != nullptr &&
                 store_10 != nullptr && store_11 != nullptr &&
                 store_12 != nullptr)));
    bool const want_load = direct_material_step || grad_ca_step || grad_cb_step;
    int slot = 0;
    cudaEvent_t slot_storage_done = nullptr;
    cudaEvent_t slot_compute_done = nullptr;
    cudaStream_t load_stream = stream_compute;

    int64_t const store_idx = t / step_ratio_eff;
    if (want_load) {
      slot = (int)(store_idx % NUM_BUFFERS);
      if (use_storage_pipeline) {
        slot_storage_done = storage_done_events.events[slot];
        slot_compute_done = compute_done_events.events[slot];
      }
    }

    size_t const device_offset = ring_storage_offset_bytes(
        store_idx, storage_mode, bytes_per_step_store);
    size_t const host_offset = cpu_linear_storage_offset_bytes(
        store_idx, storage_mode, bytes_per_step_store);

    void const *const ex_store =
        (direct_material_step || grad_ca_step || (eonly_snapshot && grad_cb_step))
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(device_stores[0]) + device_offset)
            : nullptr;
    void const *const ey_store =
        (direct_material_step || grad_ca_step || (eonly_snapshot && grad_cb_step))
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(device_stores[1]) + device_offset)
            : nullptr;
    void const *const ez_store =
        (direct_material_step || grad_ca_step || (eonly_snapshot && grad_cb_step))
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(device_stores[2]) + device_offset)
            : nullptr;
    void const *const curl_x_store =
        (grad_cb_step && !eonly_snapshot)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(device_stores[3]) + device_offset)
            : nullptr;
    void const *const curl_y_store =
        (grad_cb_step && !eonly_snapshot)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(device_stores[4]) + device_offset)
            : nullptr;
    void const *const curl_z_store =
        (grad_cb_step && !eonly_snapshot)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(device_stores[5]) + device_offset)
            : nullptr;
    size_t const next_device_offset = ring_storage_offset_bytes(
        store_idx + 1, storage_mode, bytes_per_step_store);
    bool const eonly_use_final_next =
        eonly_snapshot && (direct_material_step || grad_cb_step) &&
        (t + 1 == start_t);
    void const *const ex_next_store =
        (eonly_snapshot && (direct_material_step || grad_cb_step))
            ? (eonly_use_final_next
                   ? reinterpret_cast<void const *>(device_stores[3])
                   : reinterpret_cast<void const *>(
                         reinterpret_cast<uint8_t *>(device_stores[0]) +
                         next_device_offset))
            : nullptr;
    void const *const ey_next_store =
        (eonly_snapshot && (direct_material_step || grad_cb_step))
            ? (eonly_use_final_next
                   ? reinterpret_cast<void const *>(device_stores[4])
                   : reinterpret_cast<void const *>(
                         reinterpret_cast<uint8_t *>(device_stores[1]) +
                         next_device_offset))
            : nullptr;
    void const *const ez_next_store =
        (eonly_snapshot && (direct_material_step || grad_cb_step))
            ? (eonly_use_final_next
                   ? reinterpret_cast<void const *>(device_stores[5])
                   : reinterpret_cast<void const *>(
                         reinterpret_cast<uint8_t *>(device_stores[2]) +
                         next_device_offset))
            : nullptr;

    if (storage_mode == STORAGE_CPU && want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      void *const host_ptrs[6] = {
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[0]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[1]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[2]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[3]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[4]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[5]) +
                                   host_offset),
      };
      void const *const device_ptrs[6] = {ex_store,    ey_store,   ez_store,
                                          curl_x_store, curl_y_store, curl_z_store};
      for (int i = 0; i < 6; ++i) {
        if (!component_required[i] || device_ptrs[i] == nullptr ||
            host_stores[i] == nullptr) {
          continue;
        }
        storage_copy_snapshot_h2d(
            const_cast<void *>(device_ptrs[i]), host_ptrs[i],
            (size_t)shot_bytes_uncomp, (size_t)n_shots_h, load_stream);
      }
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, stream_storage), __FILE__,
            __LINE__);
      }
    } else if (storage_mode == STORAGE_DISK && want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      void *const host_ptrs[6] = {
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[0]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[1]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[2]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[3]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[4]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[5]) +
                                   host_offset),
      };
      void const *const device_ptrs[6] = {ex_store,    ey_store,   ez_store,
                                          curl_x_store, curl_y_store, curl_z_store};
      for (int i = 0; i < 6; ++i) {
        if (!component_required[i] || device_ptrs[i] == nullptr ||
            host_stores[i] == nullptr) {
          continue;
        }
        storage_async_disk_wait_slot(async_disk_handles[i], slot);
        tide::cuda_check_or_abort(
            cudaMemcpyAsync(const_cast<void *>(device_ptrs[i]), host_ptrs[i],
                            bytes_per_step_store, cudaMemcpyHostToDevice,
                            load_stream),
            __FILE__, __LINE__);
      }
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, load_stream), __FILE__,
            __LINE__);
      }
    }

    forward_kernel_h<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_ey_z,
        m_lambda_ez_y,
        m_lambda_ez_x,
        m_lambda_ex_z,
        m_lambda_ex_y,
        m_lambda_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    forward_kernel_e<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        ca,
        cb,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_hy_z,
        m_lambda_hz_y,
        m_lambda_hz_x,
        m_lambda_hx_z,
        m_lambda_hx_y,
        m_lambda_hy_x,
        az,
        bz,
        ay,
        by,
        ax,
        bx,
        kz,
          ky,
          kx);

    if (n_receivers_per_shot_h > 0 && grad_r != nullptr && receivers_i != nullptr) {
      add_adjoint_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                        launch_cfg.threads_sr, 0, stream_compute>>>(
          lambda_recv_field,
          grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

    if (n_sources_per_shot_h > 0 && grad_f != nullptr && sources_i != nullptr) {
      record_adjoint_at_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                                            launch_cfg.threads_sr, 0,
                                            stream_compute>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h,
          lambda_src_field,
          sources_i);
    }

    if (want_load && use_storage_pipeline) {
      tide::cuda_check_or_abort(
          cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
          __LINE__);
    }

    if (want_load) {
      if (storage_bf16) {
        if (direct_material_step) {
          material_grad_direct_eonly_3d<__nv_bfloat16>
              <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
                 stream_compute>>>(
                  lambda_ex,
                  lambda_ey,
                  lambda_ez,
                  cb,
                  reinterpret_cast<__nv_bfloat16 const *>(ex_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ey_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ez_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ex_next_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ey_next_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ez_next_store),
                  grad_eps,
                  grad_sigma,
                  grad_ca_shot,
                  grad_cb_shot,
                  grad_eps != nullptr,
                  grad_sigma != nullptr,
                  dt_h,
                  step_ratio_eff);
        } else if (eonly_snapshot) {
          coeff_grad_eonly_3d<__nv_bfloat16>
              <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
                 stream_compute>>>(
                  lambda_ex,
                  lambda_ey,
                  lambda_ez,
                  ca,
                  cb,
                  reinterpret_cast<__nv_bfloat16 const *>(ex_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ey_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ez_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ex_next_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ey_next_store),
                  reinterpret_cast<__nv_bfloat16 const *>(ez_next_store),
                  grad_ca,
                  grad_cb,
                  grad_ca_shot,
                  grad_cb_shot,
                  grad_ca_step,
                  grad_cb_step,
                  step_ratio_eff);
        } else {
          if (storage_physical) {
            coeff_grad_3d<__nv_bfloat16, true>
                <<<(unsigned)blocks_physical_cells, launch_cfg.threads_cells, 0,
                   stream_compute>>>(
                    lambda_ex,
                    lambda_ey,
                    lambda_ez,
                    reinterpret_cast<__nv_bfloat16 const *>(ex_store),
                    reinterpret_cast<__nv_bfloat16 const *>(ey_store),
                    reinterpret_cast<__nv_bfloat16 const *>(ez_store),
                    reinterpret_cast<__nv_bfloat16 const *>(curl_x_store),
                    reinterpret_cast<__nv_bfloat16 const *>(curl_y_store),
                    reinterpret_cast<__nv_bfloat16 const *>(curl_z_store),
                    grad_ca,
                    grad_cb,
                    grad_ca_shot,
                    grad_cb_shot,
                    grad_ca_step,
                    grad_cb_step,
                    step_ratio_eff);
          } else {
            coeff_grad_3d<__nv_bfloat16, false>
                <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
                   stream_compute>>>(
                    lambda_ex,
                    lambda_ey,
                    lambda_ez,
                    reinterpret_cast<__nv_bfloat16 const *>(ex_store),
                    reinterpret_cast<__nv_bfloat16 const *>(ey_store),
                    reinterpret_cast<__nv_bfloat16 const *>(ez_store),
                    reinterpret_cast<__nv_bfloat16 const *>(curl_x_store),
                    reinterpret_cast<__nv_bfloat16 const *>(curl_y_store),
                    reinterpret_cast<__nv_bfloat16 const *>(curl_z_store),
                    grad_ca,
                    grad_cb,
                    grad_ca_shot,
                    grad_cb_shot,
                    grad_ca_step,
                    grad_cb_step,
                    step_ratio_eff);
          }
        }
      } else {
        if (direct_material_step) {
          material_grad_direct_eonly_3d<TIDE_DTYPE>
              <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
                 stream_compute>>>(
                  lambda_ex,
                  lambda_ey,
                  lambda_ez,
                  cb,
                  reinterpret_cast<TIDE_DTYPE const *>(ex_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ey_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ez_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ex_next_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ey_next_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ez_next_store),
                  grad_eps,
                  grad_sigma,
                  grad_ca_shot,
                  grad_cb_shot,
                  grad_eps != nullptr,
                  grad_sigma != nullptr,
                  dt_h,
                  step_ratio_eff);
        } else if (eonly_snapshot) {
          coeff_grad_eonly_3d<TIDE_DTYPE>
              <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
                 stream_compute>>>(
                  lambda_ex,
                  lambda_ey,
                  lambda_ez,
                  ca,
                  cb,
                  reinterpret_cast<TIDE_DTYPE const *>(ex_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ey_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ez_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ex_next_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ey_next_store),
                  reinterpret_cast<TIDE_DTYPE const *>(ez_next_store),
                  grad_ca,
                  grad_cb,
                  grad_ca_shot,
                  grad_cb_shot,
                  grad_ca_step,
                  grad_cb_step,
                  step_ratio_eff);
        } else {
          if (storage_physical) {
            coeff_grad_3d<TIDE_DTYPE, true>
                <<<(unsigned)blocks_physical_cells, launch_cfg.threads_cells, 0,
                   stream_compute>>>(
                    lambda_ex,
                    lambda_ey,
                    lambda_ez,
                    reinterpret_cast<TIDE_DTYPE const *>(ex_store),
                    reinterpret_cast<TIDE_DTYPE const *>(ey_store),
                    reinterpret_cast<TIDE_DTYPE const *>(ez_store),
                    reinterpret_cast<TIDE_DTYPE const *>(curl_x_store),
                    reinterpret_cast<TIDE_DTYPE const *>(curl_y_store),
                    reinterpret_cast<TIDE_DTYPE const *>(curl_z_store),
                    grad_ca,
                    grad_cb,
                    grad_ca_shot,
                    grad_cb_shot,
                    grad_ca_step,
                    grad_cb_step,
                    step_ratio_eff);
          } else {
            coeff_grad_3d<TIDE_DTYPE, false>
                <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
                   stream_compute>>>(
                    lambda_ex,
                    lambda_ey,
                    lambda_ez,
                    reinterpret_cast<TIDE_DTYPE const *>(ex_store),
                    reinterpret_cast<TIDE_DTYPE const *>(ey_store),
                    reinterpret_cast<TIDE_DTYPE const *>(ez_store),
                    reinterpret_cast<TIDE_DTYPE const *>(curl_x_store),
                    reinterpret_cast<TIDE_DTYPE const *>(curl_y_store),
                    reinterpret_cast<TIDE_DTYPE const *>(curl_z_store),
                    grad_ca,
                    grad_cb,
                    grad_ca_shot,
                    grad_cb_shot,
                    grad_ca_step,
                    grad_cb_step,
                    step_ratio_eff);
          }
        }
      }
    }

    if (want_load && use_storage_pipeline) {
      tide::cuda_check_or_abort(
          cudaEventRecord(slot_compute_done, stream_compute), __FILE__,
          __LINE__);
    }

    if (want_load && storage_mode == STORAGE_DISK) {
      int64_t const future_store_idx = store_idx - NUM_BUFFERS;
      if (future_store_idx >= last_store_idx) {
        size_t const future_host_offset = cpu_linear_storage_offset_bytes(
            future_store_idx, storage_mode, bytes_per_step_store);
        int64_t const future_file_offset =
            future_store_idx * (int64_t)bytes_per_step_store;
        for (int comp = 0; comp < 6; ++comp) {
          if (!component_required[comp] || host_stores[comp] == nullptr) {
            continue;
          }
          TIDE_DTYPE *const future_host = reinterpret_cast<TIDE_DTYPE *>(
              reinterpret_cast<uint8_t *>(host_stores[comp]) + future_host_offset);
          cudaEvent_t ready_event = nullptr;
          tide::cuda_check_or_abort(
              cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming),
              __FILE__, __LINE__);
          tide::cuda_check_or_abort(
              cudaEventRecord(ready_event, load_stream), __FILE__, __LINE__);
          storage_async_disk_enqueue_read(
              async_disk_handles[comp], slot, future_host, bytes_per_step_store,
              future_file_offset, ready_event);
        }
      }
    }
  }

  if (reduce_grad_ca || reduce_grad_cb || reduce_grad_eps ||
      reduce_grad_sigma) {
    int const threads_reduce = 256;
    int64_t const blocks_reduce =
        (shot_numel_h + threads_reduce - 1) / threads_reduce;
    if (reduce_grad_ca) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_ca, grad_ca_shot);
    }
    if (reduce_grad_cb) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_cb, grad_cb_shot);
    }
    if (reduce_grad_eps) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_eps, grad_ca_shot);
    }
    if (reduce_grad_sigma) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_sigma, grad_cb_shot);
    }
  }

  if (!direct_material_grad && (grad_eps != nullptr || grad_sigma != nullptr) &&
      (ca_requires_grad || cb_requires_grad)) {
    int64_t const total_conv = (ca_batched_h ? n_shots_h : 1) * shot_numel_h;
    int const threads_conv = 256;
    int64_t const blocks_conv = (total_conv + threads_conv - 1) / threads_conv;
    convert_grad_ca_cb_to_eps_sigma_3d<<<(unsigned)blocks_conv, threads_conv, 0,
                                         stream_compute>>>(
        ca,
        cb,
        grad_ca,
        grad_cb,
        grad_eps,
        grad_sigma,
        dt_h,
        ca_requires_grad,
        cb_requires_grad);
  }

  for (void *&handle : async_disk_handles) {
    storage_async_disk_close(handle);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(checkpoint_revolve_backward)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const f,
    TIDE_DTYPE const *const grad_r,
    TIDE_DTYPE *const checkpoint_pool,
    TIDE_DTYPE *const scratch_wavefields,
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
    TIDE_DTYPE *const store_1,
    TIDE_DTYPE *const store_2,
    char **store_filenames_1,
    TIDE_DTYPE *const store_3,
    TIDE_DTYPE *const store_4,
    char **store_filenames_2,
    TIDE_DTYPE *const store_5,
    TIDE_DTYPE *const store_6,
    char **store_filenames_3,
    TIDE_DTYPE *const store_7,
    TIDE_DTYPE *const store_8,
    char **store_filenames_4,
    TIDE_DTYPE *const store_9,
    TIDE_DTYPE *const store_10,
    char **store_filenames_5,
    TIDE_DTYPE *const store_11,
    TIDE_DTYPE *const store_12,
    char **store_filenames_6,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    int64_t const segment_steps_h,
    int64_t const num_segments_h,
    int64_t const pool_slots,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  (void)step_ratio_h;
  (void)start_t;

  cudaSetDevice((int)device);
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);

  if (nt <= 0 || n_shots_h <= 0 || nz_h <= 0 || ny_h <= 0 || nx_h <= 0) {
    return;
  }
  if (checkpoint_pool == nullptr || scratch_wavefields == nullptr ||
      pool_slots <= 0) {
    std::fprintf(stderr,
                 "checkpoint_revolve_backward received invalid checkpoint pool\n");
    std::abort();
  }

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const state_numel = n_shots_h * shot_numel_h;
  constexpr int64_t checkpoint_fields = 18;
  size_t const state_bytes =
      (size_t)checkpoint_fields * (size_t)state_numel * sizeof(TIDE_DTYPE);
  TIDE_DTYPE *const scratch_current = scratch_wavefields;
  TIDE_DTYPE *const scratch_leaf =
      scratch_wavefields + checkpoint_fields * state_numel;
  int64_t const segment_steps =
      segment_steps_h > 0 ? segment_steps_h : (nt > 0 ? nt : 1);
  int64_t const computed_segments =
      (nt + segment_steps - 1) / segment_steps;
  int64_t const num_segments =
      num_segments_h > 0 ? num_segments_h : computed_segments;
  if (num_segments <= 0) {
    return;
  }

  auto state_base = [&](int64_t const slot) -> TIDE_DTYPE * {
    return checkpoint_pool + slot * checkpoint_fields * state_numel;
  };
  auto field = [&](TIDE_DTYPE *const base, int const field_idx) -> TIDE_DTYPE * {
    return base + (int64_t)field_idx * state_numel;
  };
  auto copy_state = [&](TIDE_DTYPE *const dst, TIDE_DTYPE const *const src) {
    tide::cuda_check_or_abort(
        cudaMemcpyAsync(dst, src, state_bytes, cudaMemcpyDeviceToDevice,
                        stream_compute),
        __FILE__, __LINE__);
  };
  auto offset_steps = [&](TIDE_DTYPE const *const base,
                          int64_t const step) -> TIDE_DTYPE const * {
    if (base == nullptr || step <= 0) {
      return base;
    }
    return base + step * n_shots_h * n_sources_per_shot_h;
  };
  auto offset_grad_r = [&](int64_t const step) -> TIDE_DTYPE const * {
    if (grad_r == nullptr || n_receivers_per_shot_h <= 0 || step <= 0) {
      return grad_r;
    }
    return grad_r + step * n_shots_h * n_receivers_per_shot_h;
  };
  auto offset_grad_f = [&](int64_t const step) -> TIDE_DTYPE * {
    if (grad_f == nullptr || n_sources_per_shot_h <= 0 || step <= 0) {
      return grad_f;
    }
    return grad_f + step * n_shots_h * n_sources_per_shot_h;
  };

  auto run_forward = [&](TIDE_DTYPE *const state, int64_t const step_start,
                         int64_t const steps, bool const with_storage) {
    if (steps <= 0) {
      return;
    }
    TIDE_DTYPE const *const source_ptr =
        n_sources_per_shot_h > 0 ? offset_steps(f, step_start) : f;
    FUNC(forward_with_storage)(
        ca,
        cb,
        cq,
        source_ptr,
        field(state, 0),
        field(state, 1),
        field(state, 2),
        field(state, 3),
        field(state, 4),
        field(state, 5),
        field(state, 6),
        field(state, 7),
        field(state, 8),
        field(state, 9),
        field(state, 10),
        field(state, 11),
        field(state, 12),
        field(state, 13),
        field(state, 14),
        field(state, 15),
        field(state, 16),
        field(state, 17),
        nullptr,
        with_storage ? store_1 : nullptr,
        with_storage ? store_2 : nullptr,
        with_storage ? store_filenames_1 : nullptr,
        with_storage ? store_3 : nullptr,
        with_storage ? store_4 : nullptr,
        with_storage ? store_filenames_2 : nullptr,
        with_storage ? store_5 : nullptr,
        with_storage ? store_6 : nullptr,
        with_storage ? store_filenames_3 : nullptr,
        with_storage ? store_7 : nullptr,
        with_storage ? store_8 : nullptr,
        with_storage ? store_filenames_4 : nullptr,
        with_storage ? store_9 : nullptr,
        with_storage ? store_10 : nullptr,
        with_storage ? store_filenames_5 : nullptr,
        with_storage ? store_11 : nullptr,
        with_storage ? store_12 : nullptr,
        with_storage ? store_filenames_6 : nullptr,
        az,
        bz,
        azh,
        bzh,
        ay,
        by,
        ayh,
        byh,
        ax,
        bx,
        axh,
        bxh,
        kz,
        kzh,
        ky,
        kyh,
        kx,
        kxh,
        sources_i,
        receivers_i,
        rdz_h,
        rdy_h,
        rdx_h,
        dt_h,
        steps,
        n_shots_h,
        nz_h,
        ny_h,
        nx_h,
        n_sources_per_shot_h,
        n_receivers_per_shot_h,
        1,
        with_storage ? storage_mode : (int64_t)STORAGE_NONE,
        storage_format_h,
        shot_bytes_uncomp,
        with_storage ? ca_requires_grad : false,
        with_storage ? cb_requires_grad : false,
        ca_batched_h,
        cb_batched_h,
        cq_batched_h,
        0,
        pml_z0_h,
        pml_y0_h,
        pml_x0_h,
        pml_z1_h,
        pml_y1_h,
        pml_x1_h,
        source_component,
        receiver_component,
        n_threads,
        device,
        execution_backend,
        compute_stream_handle,
        storage_stream_handle);
  };

  auto run_backward = [&](int64_t const segment_idx,
                          int64_t const step_start,
                          int64_t const steps) {
    if (steps <= 0) {
      return;
    }
    bool const zero_grad_on_entry = segment_idx == num_segments - 1;
    FUNC(backward)(
        ca,
        cb,
        cq,
        offset_grad_r(step_start),
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_ey_z,
        m_lambda_ez_y,
        m_lambda_ez_x,
        m_lambda_ex_z,
        m_lambda_ex_y,
        m_lambda_ey_x,
        m_lambda_hz_y,
        m_lambda_hy_z,
        m_lambda_hx_z,
        m_lambda_hz_x,
        m_lambda_hy_x,
        m_lambda_hx_y,
        store_1,
        store_2,
        store_filenames_1,
        store_3,
        store_4,
        store_filenames_2,
        store_5,
        store_6,
        store_filenames_3,
        store_7,
        store_8,
        store_filenames_4,
        store_9,
        store_10,
        store_filenames_5,
        store_11,
        store_12,
        store_filenames_6,
        offset_grad_f(step_start),
        grad_ca,
        grad_cb,
        grad_eps,
        grad_sigma,
        grad_ca_shot,
        grad_cb_shot,
        zero_grad_on_entry,
        az,
        bz,
        azh,
        bzh,
        ay,
        by,
        ayh,
        byh,
        ax,
        bx,
        axh,
        bxh,
        kz,
        kzh,
        ky,
        kyh,
        kx,
        kxh,
        sources_i,
        receivers_i,
        rdz_h,
        rdy_h,
        rdx_h,
        dt_h,
        steps,
        n_shots_h,
        nz_h,
        ny_h,
        nx_h,
        n_sources_per_shot_h,
        n_receivers_per_shot_h,
        1,
        storage_mode,
        storage_format_h,
        shot_bytes_uncomp,
        ca_requires_grad,
        cb_requires_grad,
        ca_batched_h,
        cb_batched_h,
        cq_batched_h,
        steps,
        pml_z0_h,
        pml_y0_h,
        pml_x0_h,
        pml_z1_h,
        pml_y1_h,
        pml_x1_h,
        source_component,
        receiver_component,
        n_threads,
        device,
        execution_backend,
        compute_stream_handle,
        storage_stream_handle);
  };

#ifdef TIDE_HAS_REFERENCE_REVOLVE
  if (num_segments > (int64_t)MAXINT || pool_slots > (int64_t)MAXINT) {
    std::fprintf(stderr,
                 "checkpoint_revolve_backward segment count exceeds reference "
                 "Revolve int limits\n");
    std::abort();
  }

  ::Revolve revolve((int)num_segments, (int)pool_slots);
  revolve.set_info(0);
  copy_state(scratch_current, state_base(0));
  for (;;) {
    ACTION::action const action = revolve.revolve();
    if (action == ACTION::terminate) {
      break;
    }
    if (action == ACTION::error) {
      std::fprintf(stderr,
                   "checkpoint_revolve_backward reference Revolve error %d\n",
                   revolve.getinfo());
      std::abort();
    }

    int64_t const check = (int64_t)revolve.getcheck();
    int64_t const capo = (int64_t)revolve.getcapo();
    if (action == ACTION::takeshot) {
      if (check < 0 || check >= pool_slots) {
        std::fprintf(stderr,
                     "checkpoint_revolve_backward invalid takeshot slot\n");
        std::abort();
      }
      copy_state(state_base(check), scratch_current);
    } else if (action == ACTION::restore) {
      if (check < 0 || check >= pool_slots) {
        std::fprintf(stderr,
                     "checkpoint_revolve_backward invalid restore slot\n");
        std::abort();
      }
      copy_state(scratch_current, state_base(check));
    } else if (action == ACTION::advance) {
      int64_t const old_capo = (int64_t)revolve.getoldcapo();
      int64_t const step_start = old_capo * segment_steps;
      int64_t const step_stop = tide_min<int64_t>(capo * segment_steps, nt);
      if (step_stop > step_start) {
        run_forward(scratch_current, step_start, step_stop - step_start, false);
      }
    } else if (action == ACTION::firsturn || action == ACTION::youturn) {
      int64_t const step_start = capo * segment_steps;
      if (step_start < nt) {
        int64_t const steps =
            tide_min<int64_t>(segment_steps, nt - step_start);
        copy_state(scratch_leaf, scratch_current);
        run_forward(scratch_leaf, step_start, steps, true);
        run_backward(capo, step_start, steps);
      }
    }
  }
#else
  auto process_interval = [&](auto &&self, int64_t const lo_seg,
                              int64_t const hi_seg, int64_t const slot,
                              int64_t const depth) -> void {
    if (hi_seg <= lo_seg) {
      return;
    }
    if (hi_seg - lo_seg == 1) {
      int64_t const step_start = lo_seg * segment_steps;
      if (step_start >= nt) {
        return;
      }
      int64_t const steps =
          tide_min<int64_t>(segment_steps, nt - step_start);
      copy_state(scratch_wavefields, state_base(slot));
      run_forward(scratch_wavefields, step_start, steps, true);
      run_backward(lo_seg, step_start, steps);
      return;
    }

    int64_t const split_seg = lo_seg + (hi_seg - lo_seg) / 2;
    int64_t const next_slot = depth + 1;
    if (next_slot >= pool_slots) {
      std::fprintf(stderr,
                   "checkpoint_revolve_backward checkpoint pool too small\n");
      std::abort();
    }
    int64_t const lo_step = lo_seg * segment_steps;
    int64_t const split_step =
        tide_min<int64_t>(split_seg * segment_steps, nt);
    copy_state(state_base(next_slot), state_base(slot));
    run_forward(state_base(next_slot), lo_step, split_step - lo_step, false);
    self(self, split_seg, hi_seg, next_slot, depth + 1);
    self(self, lo_seg, split_seg, slot, depth);
  };

  process_interval(process_interval, 0, num_segments, 0, 0);
#endif
  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(born_backward_bggrad)(
    TIDE_DTYPE const *const ca,
    TIDE_DTYPE const *const cb,
    TIDE_DTYPE const *const cq,
    TIDE_DTYPE const *const dca,
    TIDE_DTYPE const *const dcb,
    TIDE_DTYPE const *const f0,
    TIDE_DTYPE const *const df,
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
    TIDE_DTYPE *const eta_ex,
    TIDE_DTYPE *const eta_ey,
    TIDE_DTYPE *const eta_ez,
    TIDE_DTYPE *const eta_hx,
    TIDE_DTYPE *const eta_hy,
    TIDE_DTYPE *const eta_hz,
    TIDE_DTYPE *const m_eta_ey_z,
    TIDE_DTYPE *const m_eta_ez_y,
    TIDE_DTYPE *const m_eta_ez_x,
    TIDE_DTYPE *const m_eta_ex_z,
    TIDE_DTYPE *const m_eta_ex_y,
    TIDE_DTYPE *const m_eta_ey_x,
    TIDE_DTYPE *const m_eta_hz_y,
    TIDE_DTYPE *const m_eta_hy_z,
    TIDE_DTYPE *const m_eta_hx_z,
    TIDE_DTYPE *const m_eta_hz_x,
    TIDE_DTYPE *const m_eta_hy_x,
    TIDE_DTYPE *const m_eta_hx_y,
    void *const store_1,
    void *const store_2,
    char **store_filenames_1,
    void *const store_3,
    void *const store_4,
    char **store_filenames_2,
    void *const store_5,
    void *const store_6,
    char **store_filenames_3,
    void *const store_7,
    void *const store_8,
    char **store_filenames_4,
    void *const store_9,
    void *const store_10,
    char **store_filenames_5,
    void *const store_11,
    void *const store_12,
    char **store_filenames_6,
    void const *const dstore_ex,
    void const *const dstore_ey,
    void const *const dstore_ez,
    void const *const dstore_curl_x,
    void const *const dstore_curl_y,
    void const *const dstore_curl_z,
    TIDE_DTYPE *const grad_f0,
    TIDE_DTYPE *const grad_df,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
    TIDE_DTYPE *const grad_dca,
    TIDE_DTYPE *const grad_dcb,
    TIDE_DTYPE *const grad_ca_shot,
    TIDE_DTYPE *const grad_cb_shot,
    TIDE_DTYPE *const grad_dca_shot,
    TIDE_DTYPE *const grad_dcb_shot,
    TIDE_DTYPE *const eta_source_ex,
    TIDE_DTYPE *const eta_source_ey,
    TIDE_DTYPE *const eta_source_ez,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const dca_requires_grad,
    bool const dcb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  cudaSetDevice((int)device);
  (void)n_threads;
  (void)execution_backend;
  (void)dt_h;
  (void)f0;
  (void)df;

  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const store_size_h = n_shots_h * shot_numel_h;
  int64_t const step_ratio_eff = step_ratio_h > 0 ? step_ratio_h : 1;
  size_t const full_bytes_per_shot = (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  size_t const bf16_bytes_per_shot = (size_t)shot_numel_h * sizeof(__nv_bfloat16);
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp * (size_t)n_shots_h;
  bool const storage_bf16 = storage_format_h == STORAGE_FORMAT_BF16;
  bool const storage_full = storage_format_h == STORAGE_FORMAT_FULL;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      ((storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot) ||
       (storage_bf16 && shot_bytes_uncomp == (int64_t)bf16_bytes_per_shot));
  bool const storage_full_domain =
      storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot;
  bool const host_backed_storage =
      (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) &&
      storage_full_domain;
  if (!storage_direct && !host_backed_storage) {
    std::fprintf(stderr,
                 "born_backward_bggrad currently supports full-precision "
                 "host-backed storage or full-precision/bf16 device storage "
                 "for 3D CUDA.\n");
    std::abort();
  }
  bool const use_storage_pipeline =
      host_backed_storage &&
      (ca_requires_grad || cb_requires_grad || dca_requires_grad ||
       dcb_requires_grad) &&
      stream_storage != nullptr && stream_storage != stream_compute;

  bool const reduce_grad_ca =
      ca_requires_grad && !ca_batched_h && grad_ca != nullptr &&
      grad_ca_shot != nullptr;
  bool const reduce_grad_cb =
      cb_requires_grad && !cb_batched_h && grad_cb != nullptr &&
      grad_cb_shot != nullptr;
  bool const reduce_grad_dca =
      dca_requires_grad && !ca_batched_h && grad_dca != nullptr &&
      grad_dca_shot != nullptr;
  bool const reduce_grad_dcb =
      dcb_requires_grad && !cb_batched_h && grad_dcb != nullptr &&
      grad_dcb_shot != nullptr;

  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *lambda_src_field = lambda_ey;
  TIDE_DTYPE *eta_src_field = eta_ey;
  if (source_component == 0) {
    lambda_src_field = lambda_ex;
    eta_src_field = eta_ex;
  } else if (source_component == 2) {
    lambda_src_field = lambda_ez;
    eta_src_field = eta_ez;
  }

  TIDE_DTYPE *lambda_recv_field = lambda_ey;
  if (receiver_component == 0) {
    lambda_recv_field = lambda_ex;
  } else if (receiver_component == 2) {
    lambda_recv_field = lambda_ez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, nz_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h, n_threads, false);

  void *const host_stores[6] = {store_2, store_4, store_6,
                                store_8, store_10, store_12};
  char **const store_filenames[6] = {store_filenames_1, store_filenames_2,
                                     store_filenames_3, store_filenames_4,
                                     store_filenames_5, store_filenames_6};
  bool const component_required[6] = {
      ca_requires_grad || dca_requires_grad,
      ca_requires_grad || dca_requires_grad,
      ca_requires_grad || dca_requires_grad,
      cb_requires_grad || dcb_requires_grad,
      cb_requires_grad || dcb_requires_grad,
      cb_requires_grad || dcb_requires_grad};
  void *async_disk_handles[6] = {};
  if (storage_mode == STORAGE_DISK) {
    for (int i = 0; i < 6; ++i) {
      if (component_required[i] && store_filenames[i] != nullptr) {
        async_disk_handles[i] =
            storage_async_disk_open(store_filenames[i][0], false, NUM_BUFFERS);
      }
    }
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

  int64_t const first_store_idx = (start_t - 1) / step_ratio_eff;
  int64_t const first_t_in_chunk = start_t - nt;
  int64_t const last_store_idx =
      (first_t_in_chunk + step_ratio_eff - 1) / step_ratio_eff;
  if (storage_mode == STORAGE_DISK) {
    int64_t const stored_steps_in_chunk =
        first_store_idx >= last_store_idx ? (first_store_idx - last_store_idx + 1)
                                          : 0;
    int64_t const prefetch_count =
        tide_min<int64_t>(NUM_BUFFERS, stored_steps_in_chunk);
    for (int64_t i = 0; i < prefetch_count; ++i) {
      int64_t const store_idx = first_store_idx - i;
      int const slot = (int)(store_idx % NUM_BUFFERS);
      size_t const host_offset = cpu_linear_storage_offset_bytes(
          store_idx, storage_mode, bytes_per_step_store);
      int64_t const file_offset = store_idx * (int64_t)bytes_per_step_store;
      for (int comp = 0; comp < 6; ++comp) {
        if (!component_required[comp] || host_stores[comp] == nullptr) {
          continue;
        }
        void *const host_ptr =
            reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[comp]) +
                                     host_offset);
        storage_async_disk_enqueue_read(async_disk_handles[comp], slot, host_ptr,
                                        bytes_per_step_store, file_offset,
                                        nullptr);
      }
    }
  }

  auto zero_tensor = [&](TIDE_DTYPE *ptr, size_t count) {
    if (ptr == nullptr || count == 0) {
      return;
    }
    tide::cuda_check_or_abort(
        cudaMemsetAsync(ptr, 0, count * sizeof(TIDE_DTYPE), stream_compute),
        __FILE__, __LINE__);
  };
  size_t const state_count = (size_t)store_size_h;
  zero_tensor(lambda_ex, state_count);
  zero_tensor(lambda_ey, state_count);
  zero_tensor(lambda_ez, state_count);
  zero_tensor(lambda_hx, state_count);
  zero_tensor(lambda_hy, state_count);
  zero_tensor(lambda_hz, state_count);
  zero_tensor(eta_ex, state_count);
  zero_tensor(eta_ey, state_count);
  zero_tensor(eta_ez, state_count);
  zero_tensor(eta_hx, state_count);
  zero_tensor(eta_hy, state_count);
  zero_tensor(eta_hz, state_count);
  zero_tensor(m_lambda_ey_z, state_count);
  zero_tensor(m_lambda_ez_y, state_count);
  zero_tensor(m_lambda_ez_x, state_count);
  zero_tensor(m_lambda_ex_z, state_count);
  zero_tensor(m_lambda_ex_y, state_count);
  zero_tensor(m_lambda_ey_x, state_count);
  zero_tensor(m_lambda_hz_y, state_count);
  zero_tensor(m_lambda_hy_z, state_count);
  zero_tensor(m_lambda_hx_z, state_count);
  zero_tensor(m_lambda_hz_x, state_count);
  zero_tensor(m_lambda_hy_x, state_count);
  zero_tensor(m_lambda_hx_y, state_count);
  zero_tensor(m_eta_ey_z, state_count);
  zero_tensor(m_eta_ez_y, state_count);
  zero_tensor(m_eta_ez_x, state_count);
  zero_tensor(m_eta_ex_z, state_count);
  zero_tensor(m_eta_ex_y, state_count);
  zero_tensor(m_eta_ey_x, state_count);
  zero_tensor(m_eta_hz_y, state_count);
  zero_tensor(m_eta_hy_z, state_count);
  zero_tensor(m_eta_hx_z, state_count);
  zero_tensor(m_eta_hz_x, state_count);
  zero_tensor(m_eta_hy_x, state_count);
  zero_tensor(m_eta_hx_y, state_count);
  zero_tensor(eta_source_ex, state_count);
  zero_tensor(eta_source_ey, state_count);
  zero_tensor(eta_source_ez, state_count);

  size_t const src_count =
      (size_t)nt * (size_t)n_shots_h * (size_t)n_sources_per_shot_h;
  zero_tensor(grad_f0, src_count);
  zero_tensor(grad_df, src_count);
  if (ca_requires_grad) {
    zero_tensor(grad_ca,
                (size_t)(ca_batched_h ? n_shots_h : 1) *
                    (size_t)shot_numel_h);
  }
  if (cb_requires_grad) {
    zero_tensor(grad_cb,
                (size_t)(cb_batched_h ? n_shots_h : 1) *
                    (size_t)shot_numel_h);
  }
  if (dca_requires_grad) {
    zero_tensor(grad_dca,
                (size_t)(ca_batched_h ? n_shots_h : 1) *
                    (size_t)shot_numel_h);
  }
  if (dcb_requires_grad) {
    zero_tensor(grad_dcb,
                (size_t)(cb_batched_h ? n_shots_h : 1) *
                    (size_t)shot_numel_h);
  }
  if (reduce_grad_ca) {
    zero_tensor(grad_ca_shot, state_count);
  }
  if (reduce_grad_cb) {
    zero_tensor(grad_cb_shot, state_count);
  }
  if (reduce_grad_dca) {
    zero_tensor(grad_dca_shot, state_count);
  }
  if (reduce_grad_dcb) {
    zero_tensor(grad_dcb_shot, state_count);
  }

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    bool const do_grad = ((t % step_ratio_eff) == 0);
    bool const grad_ca_step =
        do_grad && ca_requires_grad &&
        ((storage_direct && store_1 != nullptr && store_3 != nullptr &&
          store_5 != nullptr) ||
         (host_backed_storage && store_1 != nullptr && store_2 != nullptr &&
          store_3 != nullptr && store_4 != nullptr && store_5 != nullptr &&
          store_6 != nullptr));
    bool const grad_cb_step =
        do_grad && cb_requires_grad &&
        ((storage_direct && store_7 != nullptr && store_9 != nullptr &&
          store_11 != nullptr) ||
         (host_backed_storage && store_7 != nullptr && store_8 != nullptr &&
          store_9 != nullptr && store_10 != nullptr && store_11 != nullptr &&
          store_12 != nullptr));
    bool const grad_dca_step =
        do_grad && dca_requires_grad &&
        ((storage_direct && store_1 != nullptr && store_3 != nullptr &&
          store_5 != nullptr) ||
         (host_backed_storage && store_1 != nullptr && store_2 != nullptr &&
          store_3 != nullptr && store_4 != nullptr && store_5 != nullptr &&
          store_6 != nullptr));
    bool const grad_dcb_step =
        do_grad && dcb_requires_grad &&
        ((storage_direct && store_7 != nullptr && store_9 != nullptr &&
          store_11 != nullptr) ||
         (host_backed_storage && store_7 != nullptr && store_8 != nullptr &&
          store_9 != nullptr && store_10 != nullptr && store_11 != nullptr &&
          store_12 != nullptr));
    bool const direct_ca_step =
        do_grad && ca_requires_grad && dstore_ex != nullptr &&
        dstore_ey != nullptr && dstore_ez != nullptr;
    bool const direct_cb_step =
        do_grad && cb_requires_grad && dstore_curl_x != nullptr &&
        dstore_curl_y != nullptr && dstore_curl_z != nullptr;
    bool const want_load =
        grad_ca_step || grad_cb_step || grad_dca_step || grad_dcb_step;
    int slot = 0;
    cudaEvent_t slot_storage_done = nullptr;
    cudaEvent_t slot_compute_done = nullptr;
    cudaStream_t load_stream = stream_compute;

    int64_t const store_idx = t / step_ratio_eff;
    if (want_load) {
      slot = (int)(store_idx % NUM_BUFFERS);
      if (use_storage_pipeline) {
        slot_storage_done = storage_done_events.events[slot];
        slot_compute_done = compute_done_events.events[slot];
      }
    }
    size_t const device_offset =
        ring_storage_offset_bytes(store_idx, storage_mode, bytes_per_step_store);
    size_t const direct_device_offset =
        (size_t)store_idx * bytes_per_step_store;
    size_t const host_offset = cpu_linear_storage_offset_bytes(
        store_idx, storage_mode, bytes_per_step_store);

    void const *const ex_store =
        (grad_ca_step || grad_dca_step)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_1) + device_offset)
            : nullptr;
    void const *const ey_store =
        (grad_ca_step || grad_dca_step)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_3) + device_offset)
            : nullptr;
    void const *const ez_store =
        (grad_ca_step || grad_dca_step)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_5) + device_offset)
            : nullptr;
    void const *const curl_x_store =
        (grad_cb_step || grad_dcb_step)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_7) + device_offset)
            : nullptr;
    void const *const curl_y_store =
        (grad_cb_step || grad_dcb_step)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_9) + device_offset)
            : nullptr;
    void const *const curl_z_store =
        (grad_cb_step || grad_dcb_step)
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_11) + device_offset)
            : nullptr;
    void const *const dex_store =
        direct_ca_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t const *>(dstore_ex) +
                  direct_device_offset)
            : nullptr;
    void const *const dey_store =
        direct_ca_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t const *>(dstore_ey) +
                  direct_device_offset)
            : nullptr;
    void const *const dez_store =
        direct_ca_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t const *>(dstore_ez) +
                  direct_device_offset)
            : nullptr;
    void const *const dcurl_x_store =
        direct_cb_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t const *>(dstore_curl_x) +
                  direct_device_offset)
            : nullptr;
    void const *const dcurl_y_store =
        direct_cb_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t const *>(dstore_curl_y) +
                  direct_device_offset)
            : nullptr;
    void const *const dcurl_z_store =
        direct_cb_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t const *>(dstore_curl_z) +
                  direct_device_offset)
            : nullptr;

    if (storage_mode == STORAGE_CPU && want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      void *const host_ptrs[6] = {
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[0]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[1]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[2]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[3]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[4]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[5]) +
                                   host_offset),
      };
      void const *const device_ptrs[6] = {ex_store,    ey_store,   ez_store,
                                          curl_x_store, curl_y_store, curl_z_store};
      for (int i = 0; i < 6; ++i) {
        if (!component_required[i] || device_ptrs[i] == nullptr ||
            host_stores[i] == nullptr) {
          continue;
        }
        storage_copy_snapshot_h2d(
            const_cast<void *>(device_ptrs[i]), host_ptrs[i],
            (size_t)shot_bytes_uncomp, (size_t)n_shots_h, load_stream);
      }
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, stream_storage), __FILE__,
            __LINE__);
      }
    } else if (storage_mode == STORAGE_DISK && want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      void *const host_ptrs[6] = {
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[0]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[1]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[2]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[3]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[4]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[5]) +
                                   host_offset),
      };
      void const *const device_ptrs[6] = {ex_store,    ey_store,   ez_store,
                                          curl_x_store, curl_y_store, curl_z_store};
      int64_t const file_offset = store_idx * (int64_t)bytes_per_step_store;
      for (int i = 0; i < 6; ++i) {
        if (!component_required[i] || device_ptrs[i] == nullptr ||
            host_stores[i] == nullptr) {
          continue;
        }
        storage_async_disk_wait_slot(async_disk_handles[i], slot);
        storage_async_disk_enqueue_read(async_disk_handles[i], slot, host_ptrs[i],
                                        bytes_per_step_store, file_offset,
                                        nullptr);
        storage_async_disk_wait_slot(async_disk_handles[i], slot);
        tide::cuda_check_or_abort(
            cudaMemcpyAsync(const_cast<void *>(device_ptrs[i]), host_ptrs[i],
                            bytes_per_step_store, cudaMemcpyHostToDevice,
                            load_stream),
            __FILE__, __LINE__);
      }
      tide::cuda_check_or_abort(cudaStreamSynchronize(load_stream), __FILE__,
                                __LINE__);
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, load_stream), __FILE__,
            __LINE__);
      }
    }

    if (want_load && use_storage_pipeline) {
      tide::cuda_check_or_abort(
          cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
          __LINE__);
    }

    forward_kernel_h<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_ey_z,
        m_lambda_ez_y,
        m_lambda_ez_x,
        m_lambda_ex_z,
        m_lambda_ex_y,
        m_lambda_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    forward_kernel_e<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        ca,
        cb,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_hy_z,
        m_lambda_hz_y,
        m_lambda_hz_x,
        m_lambda_hx_z,
        m_lambda_hx_y,
        m_lambda_hy_x,
        az,
        bz,
        ay,
        by,
        ax,
        bx,
        kz,
          ky,
          kx);

    forward_kernel_h<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        eta_ex,
        eta_ey,
        eta_ez,
        eta_hx,
        eta_hy,
        eta_hz,
        m_eta_ey_z,
        m_eta_ez_y,
        m_eta_ez_x,
        m_eta_ex_z,
        m_eta_ex_y,
        m_eta_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    forward_kernel_e<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        ca,
        cb,
        eta_ex,
        eta_ey,
        eta_ez,
        eta_hx,
        eta_hy,
        eta_hz,
        m_eta_hy_z,
        m_eta_hz_y,
        m_eta_hz_x,
        m_eta_hx_z,
        m_eta_hx_y,
        m_eta_hy_x,
        az,
        bz,
        ay,
        by,
        ax,
        bx,
        kz,
          ky,
          kx);

    if (n_receivers_per_shot_h > 0 && grad_r != nullptr &&
        receivers_i != nullptr) {
      add_adjoint_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                        launch_cfg.threads_sr, 0,
                                        stream_compute>>>(
          lambda_recv_field,
          grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

    if (do_grad) {
      if (want_load && use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
            __LINE__);
      }
      if (storage_bf16) {
        born_bggrad_prepare_direct_3d<__nv_bfloat16>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                dca,
                dcb,
                lambda_ex,
                lambda_ey,
                lambda_ez,
                lambda_hx,
                lambda_hy,
                lambda_hz,
                m_lambda_hz_y,
                m_lambda_hy_z,
                m_lambda_hx_z,
                m_lambda_hz_x,
                m_lambda_hy_x,
                m_lambda_hx_y,
                kz,
                ky,
                kx,
                reinterpret_cast<__nv_bfloat16 const *>(dex_store),
                reinterpret_cast<__nv_bfloat16 const *>(dey_store),
                reinterpret_cast<__nv_bfloat16 const *>(dez_store),
                reinterpret_cast<__nv_bfloat16 const *>(dcurl_x_store),
                reinterpret_cast<__nv_bfloat16 const *>(dcurl_y_store),
                reinterpret_cast<__nv_bfloat16 const *>(dcurl_z_store),
                grad_ca,
                grad_cb,
                grad_ca_shot,
                grad_cb_shot,
                eta_source_ex,
                eta_source_ey,
                eta_source_ez,
                direct_ca_step,
                direct_cb_step,
                step_ratio_eff);
      } else {
        born_bggrad_prepare_direct_3d<TIDE_DTYPE>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                dca,
                dcb,
                lambda_ex,
                lambda_ey,
                lambda_ez,
                lambda_hx,
                lambda_hy,
                lambda_hz,
                m_lambda_hz_y,
                m_lambda_hy_z,
                m_lambda_hx_z,
                m_lambda_hz_x,
                m_lambda_hy_x,
                m_lambda_hx_y,
                kz,
                ky,
                kx,
                reinterpret_cast<TIDE_DTYPE const *>(dex_store),
                reinterpret_cast<TIDE_DTYPE const *>(dey_store),
                reinterpret_cast<TIDE_DTYPE const *>(dez_store),
                reinterpret_cast<TIDE_DTYPE const *>(dcurl_x_store),
                reinterpret_cast<TIDE_DTYPE const *>(dcurl_y_store),
                reinterpret_cast<TIDE_DTYPE const *>(dcurl_z_store),
                grad_ca,
                grad_cb,
                grad_ca_shot,
                grad_cb_shot,
                eta_source_ex,
                eta_source_ey,
                eta_source_ez,
                direct_ca_step,
                direct_cb_step,
                step_ratio_eff);
      }

      if (storage_bf16) {
        coeff_grad_3d<__nv_bfloat16>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                lambda_ex,
                lambda_ey,
                lambda_ez,
                reinterpret_cast<__nv_bfloat16 const *>(ex_store),
                reinterpret_cast<__nv_bfloat16 const *>(ey_store),
                reinterpret_cast<__nv_bfloat16 const *>(ez_store),
                reinterpret_cast<__nv_bfloat16 const *>(curl_x_store),
                reinterpret_cast<__nv_bfloat16 const *>(curl_y_store),
                reinterpret_cast<__nv_bfloat16 const *>(curl_z_store),
                grad_dca,
                grad_dcb,
                grad_dca_shot,
                grad_dcb_shot,
                grad_dca_step,
                grad_dcb_step,
                step_ratio_eff);
      } else {
        coeff_grad_3d<TIDE_DTYPE>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                lambda_ex,
                lambda_ey,
                lambda_ez,
                reinterpret_cast<TIDE_DTYPE const *>(ex_store),
                reinterpret_cast<TIDE_DTYPE const *>(ey_store),
                reinterpret_cast<TIDE_DTYPE const *>(ez_store),
                reinterpret_cast<TIDE_DTYPE const *>(curl_x_store),
                reinterpret_cast<TIDE_DTYPE const *>(curl_y_store),
                reinterpret_cast<TIDE_DTYPE const *>(curl_z_store),
                grad_dca,
                grad_dcb,
                grad_dca_shot,
                grad_dcb_shot,
                grad_dca_step,
                grad_dcb_step,
                step_ratio_eff);
      }

      add_eta_source_3d<<<(unsigned)launch_cfg.blocks_cells,
                          launch_cfg.threads_cells, 0, stream_compute>>>(
          eta_ex,
          eta_ey,
          eta_ez,
          eta_source_ex,
          eta_source_ey,
          eta_source_ez);
    }

    if (n_sources_per_shot_h > 0 && sources_i != nullptr) {
      if (grad_df != nullptr) {
        record_adjoint_at_sources_component<<<
            (unsigned)launch_cfg.blocks_sources,
            launch_cfg.threads_sr,
            0,
            stream_compute>>>(
            grad_df + t * n_shots_h * n_sources_per_shot_h,
            lambda_src_field,
            sources_i);
      }
      if (grad_f0 != nullptr) {
        record_adjoint_at_sources_component<<<
            (unsigned)launch_cfg.blocks_sources,
            launch_cfg.threads_sr,
            0,
            stream_compute>>>(
            grad_f0 + t * n_shots_h * n_sources_per_shot_h,
            eta_src_field,
            sources_i);
      }
    }
  }

  if (reduce_grad_ca || reduce_grad_cb || reduce_grad_dca || reduce_grad_dcb) {
    int const threads_reduce = 256;
    int64_t const blocks_reduce =
        (shot_numel_h + threads_reduce - 1) / threads_reduce;
    if (reduce_grad_ca) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_ca, grad_ca_shot);
    }
    if (reduce_grad_cb) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_cb, grad_cb_shot);
    }
    if (reduce_grad_dca) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_dca, grad_dca_shot);
    }
    if (reduce_grad_dcb) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_dcb, grad_dcb_shot);
    }
  }

  if (host_backed_storage) {
    tide::cuda_check_or_abort(cudaStreamSynchronize(stream_compute), __FILE__,
                              __LINE__);
  }

  for (void *&handle : async_disk_handles) {
    storage_async_disk_close(handle);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

extern "C" void FUNC(born_backward)(
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
    void *const store_1,
    void *const store_2,
    char **store_filenames_1,
    void *const store_3,
    void *const store_4,
    char **store_filenames_2,
    void *const store_5,
    void *const store_6,
    char **store_filenames_3,
    void *const store_7,
    void *const store_8,
    char **store_filenames_4,
    void *const store_9,
    void *const store_10,
    char **store_filenames_5,
    void *const store_11,
    void *const store_12,
    char **store_filenames_6,
    TIDE_DTYPE *const grad_f,
    TIDE_DTYPE *const grad_ca,
    TIDE_DTYPE *const grad_cb,
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
    TIDE_DTYPE const rdz_h,
    TIDE_DTYPE const rdy_h,
    TIDE_DTYPE const rdx_h,
    TIDE_DTYPE const dt_h,
    int64_t const nt,
    int64_t const n_shots_h,
    int64_t const nz_h,
    int64_t const ny_h,
    int64_t const nx_h,
    int64_t const n_sources_per_shot_h,
    int64_t const n_receivers_per_shot_h,
    int64_t const step_ratio_h,
    int64_t const storage_mode,
    int64_t const storage_format_h,
    int64_t const shot_bytes_uncomp,
    bool const ca_requires_grad,
    bool const cb_requires_grad,
    bool const ca_batched_h,
    bool const cb_batched_h,
    bool const cq_batched_h,
    int64_t const start_t,
    int64_t const pml_z0_h,
    int64_t const pml_y0_h,
    int64_t const pml_x0_h,
    int64_t const pml_z1_h,
    int64_t const pml_y1_h,
    int64_t const pml_x1_h,
    int64_t const source_component,
    int64_t const receiver_component,
    int64_t const n_threads,
    int64_t const device,
    int64_t const execution_backend,
    void *const compute_stream_handle,
    void *const storage_stream_handle) {
  cudaSetDevice((int)device);
  (void)n_threads;
  (void)execution_backend;
  (void)dt_h;
  cudaStream_t const stream_compute =
      resolve_cuda_stream(compute_stream_handle);
  cudaStream_t const stream_storage =
      resolve_cuda_stream(storage_stream_handle);

  int64_t const shot_numel_h = nz_h * ny_h * nx_h;
  int64_t const step_ratio_eff = step_ratio_h > 0 ? step_ratio_h : 1;
  size_t const bytes_per_step_store =
      (size_t)shot_bytes_uncomp * (size_t)n_shots_h;
  size_t const full_bytes_per_shot = (size_t)shot_numel_h * sizeof(TIDE_DTYPE);
  size_t const bf16_bytes_per_shot = (size_t)shot_numel_h * sizeof(__nv_bfloat16);
  bool const storage_bf16 = storage_format_h == STORAGE_FORMAT_BF16;
  bool const storage_full = storage_format_h == STORAGE_FORMAT_FULL;
  bool const storage_direct =
      (storage_mode == STORAGE_DEVICE) &&
      ((storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot) ||
       (storage_bf16 && shot_bytes_uncomp == (int64_t)bf16_bytes_per_shot));
  bool const storage_full_domain =
      storage_full && shot_bytes_uncomp == (int64_t)full_bytes_per_shot;
  bool const host_backed_storage =
      (storage_mode == STORAGE_CPU || storage_mode == STORAGE_DISK) &&
      storage_full_domain;
  bool const use_storage_pipeline =
      host_backed_storage && (ca_requires_grad || cb_requires_grad) &&
      stream_storage != nullptr && stream_storage != stream_compute;
  bool const reduce_grad_ca =
      ca_requires_grad && !ca_batched_h && grad_ca != nullptr &&
      grad_ca_shot != nullptr;
  bool const reduce_grad_cb =
      cb_requires_grad && !cb_batched_h && grad_cb != nullptr &&
      grad_cb_shot != nullptr;

  set_constants(
      rdz_h,
      rdy_h,
      rdx_h,
      n_shots_h,
      nz_h,
      ny_h,
      nx_h,
      shot_numel_h,
      n_sources_per_shot_h,
      n_receivers_per_shot_h,
      pml_z0_h,
      pml_y0_h,
      pml_x0_h,
      pml_z1_h,
      pml_y1_h,
      pml_x1_h,
      ca_batched_h,
      cb_batched_h,
      cq_batched_h);

  TIDE_DTYPE *lambda_src_field = lambda_ey;
  if (source_component == 0) {
    lambda_src_field = lambda_ex;
  } else if (source_component == 2) {
    lambda_src_field = lambda_ez;
  }

  TIDE_DTYPE *lambda_recv_field = lambda_ey;
  if (receiver_component == 0) {
    lambda_recv_field = lambda_ex;
  } else if (receiver_component == 2) {
    lambda_recv_field = lambda_ez;
  }

  ScalarLaunchConfig3D const launch_cfg = make_scalar_launch_config_3d(
      n_shots_h, nz_h, ny_h, nx_h, n_sources_per_shot_h,
      n_receivers_per_shot_h, n_threads, false);

  void *const host_stores[6] = {store_2, store_4, store_6,
                                store_8, store_10, store_12};
  char **const store_filenames[6] = {store_filenames_1, store_filenames_2,
                                     store_filenames_3, store_filenames_4,
                                     store_filenames_5, store_filenames_6};
  bool const component_required[6] = {ca_requires_grad, ca_requires_grad,
                                      ca_requires_grad, cb_requires_grad,
                                      cb_requires_grad, cb_requires_grad};
  void *async_disk_handles[6] = {};
  if (storage_mode == STORAGE_DISK) {
    for (int i = 0; i < 6; ++i) {
      if (component_required[i] && store_filenames[i] != nullptr) {
        async_disk_handles[i] =
            storage_async_disk_open(store_filenames[i][0], false, NUM_BUFFERS);
      }
    }
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

  int64_t const first_store_idx = (start_t - 1) / step_ratio_eff;
  int64_t const first_t_in_chunk = start_t - nt;
  int64_t const last_store_idx =
      (first_t_in_chunk + step_ratio_eff - 1) / step_ratio_eff;
  if (storage_mode == STORAGE_DISK) {
    int64_t const stored_steps_in_chunk =
        first_store_idx >= last_store_idx ? (first_store_idx - last_store_idx + 1)
                                          : 0;
    int64_t const prefetch_count =
        tide_min<int64_t>(NUM_BUFFERS, stored_steps_in_chunk);
    for (int64_t i = 0; i < prefetch_count; ++i) {
      int64_t const store_idx = first_store_idx - i;
      int const slot = (int)(store_idx % NUM_BUFFERS);
      size_t const host_offset = cpu_linear_storage_offset_bytes(
          store_idx, storage_mode, bytes_per_step_store);
      int64_t const file_offset = store_idx * (int64_t)bytes_per_step_store;
      for (int comp = 0; comp < 6; ++comp) {
        if (!component_required[comp] || host_stores[comp] == nullptr) {
          continue;
        }
        void *const host_ptr =
            reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[comp]) +
                                     host_offset);
        storage_async_disk_enqueue_read(async_disk_handles[comp], slot, host_ptr,
                                        bytes_per_step_store, file_offset,
                                        nullptr);
      }
    }
  }

  if (grad_f != nullptr && nt > 0 && n_shots_h > 0 && n_sources_per_shot_h > 0) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_f, 0,
                   (size_t)nt * (size_t)n_shots_h *
                       (size_t)n_sources_per_shot_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (ca_requires_grad && grad_ca != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_ca, 0,
                   (size_t)(ca_batched_h ? n_shots_h : 1) *
                       (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (cb_requires_grad && grad_cb != nullptr) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_cb, 0,
                   (size_t)(cb_batched_h ? n_shots_h : 1) *
                       (size_t)shot_numel_h * sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (reduce_grad_ca) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_ca_shot, 0,
                   (size_t)n_shots_h * (size_t)shot_numel_h *
                       sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }
  if (reduce_grad_cb) {
    tide::cuda_check_or_abort(
        cudaMemset(grad_cb_shot, 0,
                   (size_t)n_shots_h * (size_t)shot_numel_h *
                       sizeof(TIDE_DTYPE)),
        __FILE__, __LINE__);
  }

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    bool const do_grad = ((t % step_ratio_eff) == 0);
    bool const grad_ca_step =
        do_grad && ca_requires_grad &&
        ((storage_direct && store_1 != nullptr && store_3 != nullptr &&
          store_5 != nullptr) ||
         (host_backed_storage && store_1 != nullptr && store_2 != nullptr &&
          store_3 != nullptr && store_4 != nullptr && store_5 != nullptr &&
          store_6 != nullptr));
    bool const grad_cb_step =
        do_grad && cb_requires_grad &&
        ((storage_direct && store_7 != nullptr && store_9 != nullptr &&
          store_11 != nullptr) ||
         (host_backed_storage && store_7 != nullptr && store_8 != nullptr &&
          store_9 != nullptr && store_10 != nullptr && store_11 != nullptr &&
          store_12 != nullptr));
    bool const want_load = grad_ca_step || grad_cb_step;
    int slot = 0;
    cudaEvent_t slot_storage_done = nullptr;
    cudaEvent_t slot_compute_done = nullptr;
    cudaStream_t load_stream = stream_compute;

    int64_t const store_idx = t / step_ratio_eff;
    if (want_load) {
      slot = (int)(store_idx % NUM_BUFFERS);
      if (use_storage_pipeline) {
        slot_storage_done = storage_done_events.events[slot];
        slot_compute_done = compute_done_events.events[slot];
      }
    }
    size_t const device_offset = ring_storage_offset_bytes(
        store_idx, storage_mode, bytes_per_step_store);
    size_t const host_offset = cpu_linear_storage_offset_bytes(
        store_idx, storage_mode, bytes_per_step_store);

    void const *const ex_store =
        grad_ca_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_1) + device_offset)
            : nullptr;
    void const *const ey_store =
        grad_ca_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_3) + device_offset)
            : nullptr;
    void const *const ez_store =
        grad_ca_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_5) + device_offset)
            : nullptr;
    void const *const curl_x_store =
        grad_cb_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_7) + device_offset)
            : nullptr;
    void const *const curl_y_store =
        grad_cb_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_9) + device_offset)
            : nullptr;
    void const *const curl_z_store =
        grad_cb_step
            ? reinterpret_cast<void const *>(
                  reinterpret_cast<uint8_t *>(store_11) + device_offset)
            : nullptr;

    if (storage_mode == STORAGE_CPU && want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      void *const host_ptrs[6] = {
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[0]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[1]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[2]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[3]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[4]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[5]) +
                                   host_offset),
      };
      void const *const device_ptrs[6] = {ex_store,    ey_store,   ez_store,
                                          curl_x_store, curl_y_store, curl_z_store};
      for (int i = 0; i < 6; ++i) {
        if (!component_required[i] || device_ptrs[i] == nullptr ||
            host_stores[i] == nullptr) {
          continue;
        }
        storage_copy_snapshot_h2d(
            const_cast<void *>(device_ptrs[i]), host_ptrs[i],
            (size_t)shot_bytes_uncomp, (size_t)n_shots_h, load_stream);
      }
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, stream_storage), __FILE__,
            __LINE__);
      }
    } else if (storage_mode == STORAGE_DISK && want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_storage, slot_compute_done, 0),
            __FILE__, __LINE__);
        load_stream = stream_storage;
      }
      void *const host_ptrs[6] = {
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[0]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[1]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[2]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[3]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[4]) +
                                   host_offset),
          reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(host_stores[5]) +
                                   host_offset),
      };
      void const *const device_ptrs[6] = {ex_store,    ey_store,   ez_store,
                                          curl_x_store, curl_y_store, curl_z_store};
      int64_t const file_offset = store_idx * (int64_t)bytes_per_step_store;
      for (int i = 0; i < 6; ++i) {
        if (!component_required[i] || device_ptrs[i] == nullptr ||
            host_stores[i] == nullptr) {
          continue;
        }
        storage_async_disk_wait_slot(async_disk_handles[i], slot);
        storage_async_disk_enqueue_read(async_disk_handles[i], slot, host_ptrs[i],
                                        bytes_per_step_store, file_offset,
                                        nullptr);
        storage_async_disk_wait_slot(async_disk_handles[i], slot);
        tide::cuda_check_or_abort(
            cudaMemcpyAsync(const_cast<void *>(device_ptrs[i]), host_ptrs[i],
                            bytes_per_step_store, cudaMemcpyHostToDevice,
                            load_stream),
            __FILE__, __LINE__);
      }
      tide::cuda_check_or_abort(cudaStreamSynchronize(load_stream), __FILE__,
                                __LINE__);
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaEventRecord(slot_storage_done, load_stream), __FILE__,
            __LINE__);
      }
    }

    if (want_load && use_storage_pipeline) {
      tide::cuda_check_or_abort(
          cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
          __LINE__);
    }

    forward_kernel_h<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        cq,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_ey_z,
        m_lambda_ez_y,
        m_lambda_ez_x,
        m_lambda_ex_z,
        m_lambda_ex_y,
        m_lambda_ey_x,
        azh,
        bzh,
        ayh,
        byh,
        axh,
        bxh,
        kzh,
        kyh,
        kxh);

    forward_kernel_e<false><<<(unsigned)launch_cfg.blocks_cells,
                       launch_cfg.threads_cells, 0, stream_compute>>>(
        ca,
        cb,
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_hy_z,
        m_lambda_hz_y,
        m_lambda_hz_x,
        m_lambda_hx_z,
        m_lambda_hx_y,
        m_lambda_hy_x,
        az,
        bz,
        ay,
        by,
        ax,
        bx,
        kz,
          ky,
          kx);

    if (n_receivers_per_shot_h > 0 && grad_r != nullptr &&
        receivers_i != nullptr) {
      add_adjoint_receivers_component<<<(unsigned)launch_cfg.blocks_receivers,
                                        launch_cfg.threads_sr, 0,
                                        stream_compute>>>(
          lambda_recv_field,
          grad_r + t * n_shots_h * n_receivers_per_shot_h,
          receivers_i);
    }

    if (n_sources_per_shot_h > 0 && grad_f != nullptr && sources_i != nullptr) {
      record_adjoint_at_sources_component<<<(unsigned)launch_cfg.blocks_sources,
                                            launch_cfg.threads_sr, 0,
                                            stream_compute>>>(
          grad_f + t * n_shots_h * n_sources_per_shot_h,
          lambda_src_field,
          sources_i);
    }

    if (want_load) {
      if (use_storage_pipeline) {
        tide::cuda_check_or_abort(
            cudaStreamWaitEvent(stream_compute, slot_storage_done, 0), __FILE__,
            __LINE__);
      }
      if (storage_bf16) {
        coeff_grad_3d<__nv_bfloat16>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                lambda_ex,
                lambda_ey,
                lambda_ez,
                reinterpret_cast<__nv_bfloat16 const *>(ex_store),
                reinterpret_cast<__nv_bfloat16 const *>(ey_store),
                reinterpret_cast<__nv_bfloat16 const *>(ez_store),
                reinterpret_cast<__nv_bfloat16 const *>(curl_x_store),
                reinterpret_cast<__nv_bfloat16 const *>(curl_y_store),
                reinterpret_cast<__nv_bfloat16 const *>(curl_z_store),
                grad_ca,
                grad_cb,
                grad_ca_shot,
                grad_cb_shot,
                grad_ca_step,
                grad_cb_step,
                step_ratio_eff);
      } else {
        coeff_grad_3d<TIDE_DTYPE>
            <<<(unsigned)launch_cfg.blocks_cells, launch_cfg.threads_cells, 0,
               stream_compute>>>(
                lambda_ex,
                lambda_ey,
                lambda_ez,
                reinterpret_cast<TIDE_DTYPE const *>(ex_store),
                reinterpret_cast<TIDE_DTYPE const *>(ey_store),
                reinterpret_cast<TIDE_DTYPE const *>(ez_store),
                reinterpret_cast<TIDE_DTYPE const *>(curl_x_store),
                reinterpret_cast<TIDE_DTYPE const *>(curl_y_store),
                reinterpret_cast<TIDE_DTYPE const *>(curl_z_store),
                grad_ca,
                grad_cb,
                grad_ca_shot,
                grad_cb_shot,
                grad_ca_step,
                grad_cb_step,
                step_ratio_eff);
      }
    }
  }

  if (reduce_grad_ca || reduce_grad_cb) {
    int const threads_reduce = 256;
    int64_t const blocks_reduce =
        (shot_numel_h + threads_reduce - 1) / threads_reduce;
    if (reduce_grad_ca) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_ca, grad_ca_shot);
    }
    if (reduce_grad_cb) {
      combine_grad_shot_3d<<<(unsigned)blocks_reduce, threads_reduce, 0,
                             stream_compute>>>(grad_cb, grad_cb_shot);
    }
  }

  if (host_backed_storage) {
    tide::cuda_check_or_abort(cudaStreamSynchronize(stream_compute), __FILE__,
                              __LINE__);
  }

  for (void *&handle : async_disk_handles) {
    storage_async_disk_close(handle);
  }

  tide::cuda_check_or_abort(cudaPeekAtLastError(), __FILE__, __LINE__);
}

#undef ND_INDEX
#undef ND_INDEX_J
#undef EX
#undef EY
#undef EZ
#undef HX
#undef HY
#undef HZ
#undef CB_I
#undef CQ_I
#undef gpuErrchk

} // namespace FUNC(Inst)
