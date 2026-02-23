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

#define TIDE_FD_PAD (tide::StencilTraits<TIDE_STENCIL>::FD_PAD)
#define EY(dy, dx) ey[idx + (dy) * nx + (dx)]
#define HX(dy, dx) hx[idx + (dy) * nx + (dx)]
#define HZ(dy, dx) hz[idx + (dy) * nx + (dx)]

namespace FUNC(Inst) {
static inline void clear_work_halo(TIDE_DTYPE *__restrict const work,
                                   int64_t const shot_offset, int64_t const ny,
                                   int64_t const nx) {
  size_t const row_bytes = (size_t)nx * sizeof(TIDE_DTYPE);

  for (int64_t y = 0; y < TIDE_FD_PAD; ++y) {
    memset(work + shot_offset + y * nx, 0, row_bytes);
  }
  for (int64_t y = ny - TIDE_FD_PAD + 1; y < ny; ++y) {
    memset(work + shot_offset + y * nx, 0, row_bytes);
  }

  for (int64_t y = TIDE_FD_PAD; y <= ny - TIDE_FD_PAD; ++y) {
    TIDE_DTYPE *row_ptr = work + shot_offset + y * nx;
    TIDE_OMP_SIMD
    for (int64_t x = 0; x < TIDE_FD_PAD; ++x) {
      row_ptr[x] = (TIDE_DTYPE)0;
    }
    TIDE_OMP_SIMD
    for (int64_t x = nx - TIDE_FD_PAD + 1; x < nx; ++x) {
      row_ptr[x] = (TIDE_DTYPE)0;
    }
  }
}





static void forward_kernel_h(
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const cq_batched) {

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,   ax,    axh,        by,     byh,    bx,
      bxh,     ky,    kyh,   kx,         kxh,    rdy,    rdx,
      n_shots, ny,    nx,    shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  false, false, cq_batched};

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_INDEX y;
  TIDE_OMP_INDEX x;

  

  TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (y = 0; y < ny - TIDE_FD_PAD + 1; ++y) {
      for (x = 0; x < nx - TIDE_FD_PAD + 1; ++x) {
        forward_kernel_h_core<TIDE_DTYPE, TIDE_STENCIL>(
            params, cq, ey, hx, hz, m_ey_x, m_ey_z, y, x, shot_idx);
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cb_batched, bool const ca_requires_grad,
    bool const cb_requires_grad, TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const curl_h_store) {

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false};

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_INDEX y;
  TIDE_OMP_INDEX x;

  

  TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (y = 0; y < ny - TIDE_FD_PAD + 1; ++y) {
      for (x = 0; x < nx - TIDE_FD_PAD + 1; ++x) {
        forward_kernel_e_with_storage_core<TIDE_DTYPE, TIDE_DTYPE,
                                           TIDE_STENCIL>(
            params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_store, curl_h_store,
            ca_requires_grad, cb_requires_grad, y, x, shot_idx);
      }
    }
  }
}

static void forward_kernel_e_with_storage_bf16(
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cb_batched, bool const ca_requires_grad,
    bool const cb_requires_grad, tide_bfloat16 *__restrict const ey_store,
    tide_bfloat16 *__restrict const curl_h_store) {

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,         axh,        by,     byh,    bx,
      bxh,     ky,         kyh,        kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,         shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, cb_batched, false};

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_INDEX y;
  TIDE_OMP_INDEX x;

  

  TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (y = 0; y < ny - TIDE_FD_PAD + 1; ++y) {
      for (x = 0; x < nx - TIDE_FD_PAD + 1; ++x) {
        forward_kernel_e_with_storage_core<TIDE_DTYPE, tide_bfloat16,
                                           TIDE_STENCIL>(
            params, ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ey_store, curl_h_store,
            ca_requires_grad, cb_requires_grad, y, x, shot_idx);
      }
    }
  }
}

static inline void forward_kernel_e_with_storage_dispatch(
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cb_batched, bool const ca_requires_grad,
    bool const cb_requires_grad, TIDE_DTYPE *__restrict const ey_store,
    TIDE_DTYPE *__restrict const curl_store) {
  forward_kernel_e_with_storage(
      ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh,
      ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0, pml_y1,
      pml_x0, pml_x1, ca_batched, cb_batched, ca_requires_grad, cb_requires_grad,
      ey_store, curl_store);
}

static inline void forward_kernel_e_with_storage_dispatch(
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cb_batched, bool const ca_requires_grad,
    bool const cb_requires_grad, tide_bfloat16 *__restrict const ey_store,
    tide_bfloat16 *__restrict const curl_store) {
  forward_kernel_e_with_storage_bf16(
      ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh,
      ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0, pml_y1,
      pml_x0, pml_x1, ca_batched, cb_batched, ca_requires_grad, cb_requires_grad,
      ey_store, curl_store);
}

template <typename StoreT>
static inline void forward_step_with_storage(
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cb_batched,
    StoreT *__restrict const ey_store_1, StoreT *__restrict const curl_store_1,
    bool const store_ey, bool const store_curl, int64_t const storage_mode,
    FILE *const *const fp_ey, FILE *const *const fp_curl,
    int64_t const store_offset, int64_t const step_idx,
    int64_t const shot_bytes_uncomp) {
  StoreT *const ey_store_1_t = store_ey ? (ey_store_1 + store_offset) : NULL;
  StoreT *const curl_store_1_t =
      store_curl ? (curl_store_1 + store_offset) : NULL;

  forward_kernel_e_with_storage_dispatch(
      ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh,
      ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0, pml_y1,
      pml_x0, pml_x1, ca_batched, cb_batched, store_ey, store_curl,
      ey_store_1_t, curl_store_1_t);

  if (storage_mode == STORAGE_DISK) {
    if (store_ey) {
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        storage_save_snapshot_cpu((void *)(ey_store_1_t + shot * shot_numel),
                                  fp_ey[shot], storage_mode, step_idx,
                                  (size_t)shot_bytes_uncomp);
      }
    }
    if (store_curl) {
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        storage_save_snapshot_cpu(
            (void *)(curl_store_1_t + shot * shot_numel), fp_curl[shot],
            storage_mode, step_idx, (size_t)shot_bytes_uncomp);
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
    void
    FUNC(forward)(
        TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
        TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f,
        TIDE_DTYPE *const ey, TIDE_DTYPE *const hx, TIDE_DTYPE *const hz,
        TIDE_DTYPE *const m_ey_x, TIDE_DTYPE *const m_ey_z,
        TIDE_DTYPE *const m_hx_z, TIDE_DTYPE *const m_hz_x, TIDE_DTYPE *const r,
        TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
        TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
        TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
        TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
        TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
        TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
        int64_t const *const sources_i, int64_t const *const receivers_i,
        TIDE_DTYPE const rdy, TIDE_DTYPE const rdx, TIDE_DTYPE const dt,
        int64_t const nt, int64_t const n_shots, int64_t const ny,
        int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot, int64_t const step_ratio,
        bool const ca_batched, bool const cb_batched, bool const cq_batched,
        int64_t const start_t, int64_t const pml_y0, int64_t const pml_x0,
        int64_t const pml_y1, int64_t const pml_x1, int64_t const n_threads,
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
    forward_kernel_h(cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh,
                     bx, bxh, ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx,
                     shot_numel, pml_y0, pml_y1, pml_x0, pml_x1, cq_batched);

    forward_kernel_e_with_storage(
        ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx, bxh,
        ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0, pml_y1,
        pml_x0, pml_x1, ca_batched, cb_batched, false,
        false, // No storage for standard forward
        NULL, NULL);

    if (n_sources_per_shot > 0) {
      add_sources_ey(ey, f + t * n_shots * n_sources_per_shot, sources_i,
                     n_shots, shot_numel, n_sources_per_shot);
    }

    if (n_receivers_per_shot > 0) {
      record_receivers_ey(r + t * n_shots * n_receivers_per_shot, ey,
                          receivers_i, n_shots, shot_numel,
                          n_receivers_per_shot);
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
    void
    FUNC(forward_with_storage)(
        TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
        TIDE_DTYPE const *const cq, TIDE_DTYPE const *const f,
        TIDE_DTYPE *const ey, TIDE_DTYPE *const hx, TIDE_DTYPE *const hz,
        TIDE_DTYPE *const m_ey_x, TIDE_DTYPE *const m_ey_z,
        TIDE_DTYPE *const m_hx_z, TIDE_DTYPE *const m_hz_x, TIDE_DTYPE *const r,
        TIDE_DTYPE *const ey_store_1, void *const ey_store_3,
        char const *const *const ey_filenames, TIDE_DTYPE *const curl_store_1,
        void *const curl_store_3, char const *const *const curl_filenames,
        TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
        TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
        TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
        TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
        TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
        TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
        int64_t const *const sources_i, int64_t const *const receivers_i,
        TIDE_DTYPE const rdy, TIDE_DTYPE const rdx, TIDE_DTYPE const dt,
        int64_t const nt, int64_t const n_shots, int64_t const ny,
        int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, int64_t const shot_bytes_uncomp,
        bool const ca_requires_grad, bool const cb_requires_grad,
        bool const ca_batched, bool const cb_batched, bool const cq_batched,
        int64_t const start_t, int64_t const pml_y0, int64_t const pml_x0,
        int64_t const pml_y1, int64_t const pml_x1, int64_t const n_threads,
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
    forward_kernel_h(cq, ey, hx, hz, m_ey_x, m_ey_z, ay, ayh, ax, axh, by, byh,
                     bx, bxh, ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx,
                     shot_numel, pml_y0, pml_y1, pml_x0, pml_x1, cq_batched);

    bool const store_step = ((t % step_ratio) == 0);
    bool const store_ey = store_step && ca_requires_grad;
    bool const store_curl = store_step && cb_requires_grad;
    int64_t const step_idx = t / step_ratio;

    int64_t const store_offset =
        (storage_mode == STORAGE_DEVICE ? step_idx * store_size : 0);

    if (storage_bf16) {
      forward_step_with_storage<tide_bfloat16>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0,
          pml_y1, pml_x0, pml_x1, ca_batched, cb_batched,
          (tide_bfloat16 *)ey_store_1, (tide_bfloat16 *)curl_store_1, store_ey,
          store_curl, storage_mode, fp_ey, fp_curl, store_offset, step_idx,
          shot_bytes_uncomp);
    } else {
      forward_step_with_storage<TIDE_DTYPE>(
          ca, cb, hx, hz, ey, m_hx_z, m_hz_x, ay, ayh, ax, axh, by, byh, bx,
          bxh, ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0,
          pml_y1, pml_x0, pml_x1, ca_batched, cb_batched, ey_store_1,
          curl_store_1, store_ey, store_curl, storage_mode, fp_ey, fp_curl,
          store_offset, step_idx, shot_bytes_uncomp);
    }

    if (n_sources_per_shot > 0) {
      add_sources_ey(ey, f + t * n_shots * n_sources_per_shot, sources_i,
                     n_shots, shot_numel, n_sources_per_shot);
    }

    if (n_receivers_per_shot > 0) {
      record_receivers_ey(r + t * n_shots * n_receivers_per_shot, ey,
                          receivers_i, n_shots, shot_numel,
                          n_receivers_per_shot);
    }
  }

  if (fp_ey != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot)
      fclose(fp_ey[shot]);
    free(fp_ey);
  }
  if (fp_curl != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot)
      fclose(fp_curl[shot]);
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const cb_batched, TIDE_DTYPE *__restrict const work_x,
    TIDE_DTYPE *__restrict const work_z) {

  (void)work_x;
  (void)work_z; // No longer needed with new formulation

  GridParams<TIDE_DTYPE> params = {
      ay,     ayh,    ax,     axh,    by,    byh,        bx,   bxh, ky,
      kyh,    kx,     kxh,    rdy,    rdx,   n_shots,    ny,   nx,  shot_numel,
      pml_y0, pml_y1, pml_x0, pml_x1, false, cb_batched, false};

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_INDEX y;
  TIDE_OMP_INDEX x;

  

  TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (y = 0; y < ny - TIDE_FD_PAD + 1; ++y) {
      for (x = 0; x < nx - TIDE_FD_PAD + 1; ++x) {
        backward_kernel_lambda_h_core<TIDE_DTYPE, TIDE_STENCIL>(
            params, cb, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x,
            m_lambda_ey_z, y, x, shot_idx);
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
    TIDE_DTYPE *__restrict const grad_ca, TIDE_DTYPE *__restrict const grad_cb,
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cq_batched, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio,
    TIDE_DTYPE *__restrict const work_x, TIDE_DTYPE *__restrict const work_z) {

  (void)work_x;
  (void)work_z; // No longer needed with new formulation

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,    axh,        by,     byh,    bx,
      bxh,     ky,         kyh,   kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,    shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, false, cq_batched};

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_INDEX y;
  TIDE_OMP_INDEX x;

  

  TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (y = 0; y < ny - TIDE_FD_PAD + 1; ++y) {
      for (x = 0; x < nx - TIDE_FD_PAD + 1; ++x) {
        backward_kernel_lambda_e_with_grad_core<TIDE_DTYPE, TIDE_DTYPE,
                                                TIDE_STENCIL>(
            params, ca, cq, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
            m_lambda_hz_x, ey_store, curl_h_store, grad_ca, grad_cb,
            ca_requires_grad, cb_requires_grad, step_ratio, y, x, shot_idx);
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
    TIDE_DTYPE *__restrict const grad_ca, TIDE_DTYPE *__restrict const grad_cb,
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
    TIDE_DTYPE const *__restrict const kxh, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cq_batched, bool const ca_requires_grad,
    bool const cb_requires_grad, int64_t const step_ratio,
    TIDE_DTYPE *__restrict const work_x, TIDE_DTYPE *__restrict const work_z) {

  (void)work_x;
  (void)work_z; // No longer needed with new formulation

  GridParams<TIDE_DTYPE> params = {
      ay,      ayh,        ax,    axh,        by,     byh,    bx,
      bxh,     ky,         kyh,   kx,         kxh,    rdy,    rdx,
      n_shots, ny,         nx,    shot_numel, pml_y0, pml_y1, pml_x0,
      pml_x1,  ca_batched, false, cq_batched};

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_INDEX y;
  TIDE_OMP_INDEX x;

  

  TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    for (y = 0; y < ny - TIDE_FD_PAD + 1; ++y) {
      for (x = 0; x < nx - TIDE_FD_PAD + 1; ++x) {
        backward_kernel_lambda_e_with_grad_core<TIDE_DTYPE, tide_bfloat16,
                                                TIDE_STENCIL>(
            params, ca, cq, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z,
            m_lambda_hz_x, ey_store, curl_h_store, grad_ca, grad_cb,
            ca_requires_grad, cb_requires_grad, step_ratio, y, x, shot_idx);
      }
    }
  }
}

static void inverse_kernel_e_and_curl(
    TIDE_DTYPE const *__restrict const ca,
    TIDE_DTYPE const *__restrict const cb,
    TIDE_DTYPE const *__restrict const hx,
    TIDE_DTYPE const *__restrict const hz, TIDE_DTYPE *__restrict const ey,
    TIDE_DTYPE *__restrict const curl_h_out, TIDE_DTYPE const rdy,
    TIDE_DTYPE const rdx, int64_t const n_shots, int64_t const ny,
    int64_t const nx, int64_t const shot_numel, int64_t const pml_y0,
    int64_t const pml_y1, int64_t const pml_x0, int64_t const pml_x1,
    bool const ca_batched, bool const cb_batched) {

  int64_t const y0 = MAX(TIDE_FD_PAD, pml_y0);
  int64_t const y1 = MIN(ny - TIDE_FD_PAD + 1, pml_y1);
  int64_t const x0 = MAX(TIDE_FD_PAD, pml_x0);
  int64_t const x1 = MIN(nx - TIDE_FD_PAD + 1, pml_x1);

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
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
    TIDE_DTYPE const *__restrict const ey, TIDE_DTYPE *__restrict const hx,
    TIDE_DTYPE *__restrict const hz, TIDE_DTYPE const rdy, TIDE_DTYPE const rdx,
    int64_t const n_shots, int64_t const ny, int64_t const nx,
    int64_t const shot_numel, int64_t const pml_y0, int64_t const pml_y1,
    int64_t const pml_x0, int64_t const pml_x1, bool const cq_batched) {

  int64_t const y0 = MAX(TIDE_FD_PAD, pml_y0);
  int64_t const y1 = MIN(ny - TIDE_FD_PAD + 1, pml_y1);
  int64_t const x0 = MAX(TIDE_FD_PAD, pml_x0);
  int64_t const x1 = MIN(nx - TIDE_FD_PAD + 1, pml_x1);

  TIDE_OMP_INDEX shot_idx;
  TIDE_OMP_PARALLEL_FOR_IF(n_shots >= TIDE_OMP_MIN_PARALLEL_SHOTS)
  for (shot_idx = 0; shot_idx < n_shots; ++shot_idx) {
    int64_t const shot_offset = shot_idx * shot_numel;
    TIDE_DTYPE const *__restrict const cq_ptr =
        cq_batched ? (cq + shot_offset) : cq;
    TIDE_OMP_SIMD_COLLAPSE2
    for (int64_t y = y0; y < y1; ++y) {
      for (int64_t x = x0; x < x1; ++x) {
        int64_t const idx = IDX(y, x);
        TIDE_DTYPE const cq_val = cq_ptr[idx];

        if (y < ny - TIDE_FD_PAD) {
          TIDE_DTYPE const dey_dz = DIFFYH1(EY);
          HX(0, 0) += cq_val * dey_dz;
        }
        if (x < nx - TIDE_FD_PAD) {
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
    void
    FUNC(backward)(
        TIDE_DTYPE const *const ca, TIDE_DTYPE const *const cb,
        TIDE_DTYPE const *const cq, TIDE_DTYPE const *const grad_r,
        TIDE_DTYPE *const lambda_ey, TIDE_DTYPE *const lambda_hx,
        TIDE_DTYPE *const lambda_hz, TIDE_DTYPE *const m_lambda_ey_x,
        TIDE_DTYPE *const m_lambda_ey_z, TIDE_DTYPE *const m_lambda_hx_z,
        TIDE_DTYPE *const m_lambda_hz_x, TIDE_DTYPE *const ey_store_1,
        void *const ey_store_3, char const *const *const ey_filenames,
        TIDE_DTYPE *const curl_store_1, void *const curl_store_3,
        char const *const *const curl_filenames, TIDE_DTYPE *const grad_f,
        TIDE_DTYPE *const grad_ca, TIDE_DTYPE *const grad_cb,
        TIDE_DTYPE *const grad_ca_shot,
        TIDE_DTYPE *const grad_cb_shot,
        TIDE_DTYPE const *const ay, TIDE_DTYPE const *const by,
        TIDE_DTYPE const *const ayh, TIDE_DTYPE const *const byh,
        TIDE_DTYPE const *const ax, TIDE_DTYPE const *const bx,
        TIDE_DTYPE const *const axh, TIDE_DTYPE const *const bxh,
        TIDE_DTYPE const *const ky, TIDE_DTYPE const *const kyh,
        TIDE_DTYPE const *const kx, TIDE_DTYPE const *const kxh,
        int64_t const *const sources_i, int64_t const *const receivers_i,
        TIDE_DTYPE const rdy, TIDE_DTYPE const rdx, TIDE_DTYPE const dt,
        int64_t const nt, int64_t const n_shots, int64_t const ny,
        int64_t const nx, int64_t const n_sources_per_shot,
        int64_t const n_receivers_per_shot, int64_t const step_ratio,
        int64_t const storage_mode, int64_t const shot_bytes_uncomp,
        bool const ca_requires_grad, bool const cb_requires_grad,
        bool const ca_batched, bool const cb_batched, bool const cq_batched,
        int64_t const start_t, int64_t const pml_y0, int64_t const pml_x0,
        int64_t const pml_y1, int64_t const pml_x1, int64_t const n_threads,
        int64_t const device /* unused for CPU */) {

  (void)device;
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
  bool const storage_bf16 = (shot_bytes_uncomp == shot_numel * 2);
  bool const reduce_grad_ca = ca_requires_grad && !ca_batched;
  bool const reduce_grad_cb = cb_requires_grad && !cb_batched;
  TIDE_DTYPE *grad_ca_accum = grad_ca;
  TIDE_DTYPE *grad_cb_accum = grad_cb;
  if (reduce_grad_ca) {
    grad_ca_accum = grad_ca_shot;
    memset(grad_ca_accum, 0, (size_t)store_size * sizeof(TIDE_DTYPE));
  }
  if (reduce_grad_cb) {
    grad_cb_accum = grad_cb_shot;
    memset(grad_cb_accum, 0, (size_t)store_size * sizeof(TIDE_DTYPE));
  }
  TIDE_DTYPE *work_x = NULL;
  TIDE_DTYPE *work_z = NULL;

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
  // Backward order is: record(adjoint) -> source_inject(adjoint) ->
  // E_update(adjoint) -> H_update(adjoint) Which translates to: grad_r_inject
  // -> grad_f_record -> λ_E_update(grad_accum) -> λ_H_update

  for (int64_t t = start_t - 1; t >= start_t - nt; --t) {
    // Determine storage index for this time step
    int64_t const store_idx = t / step_ratio;
    bool const do_grad = (t % step_ratio) == 0;
    bool const grad_ey = do_grad && ca_requires_grad;
    bool const grad_curl = do_grad && cb_requires_grad;

    int64_t const store_offset =
        (storage_mode == STORAGE_DEVICE ? store_idx * store_size : 0);

    if (storage_bf16) {
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
      TIDE_DTYPE *const ey_store_1_t = ey_store_1 + store_offset;
      TIDE_DTYPE *const curl_store_1_t = curl_store_1 + store_offset;

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
      add_sources_ey(lambda_ey, grad_r + t * n_shots * n_receivers_per_shot,
                     receivers_i, n_shots, shot_numel, n_receivers_per_shot);
    }

    // Record adjoint source gradient using λ_Ey^{t+1} (adjoint of source
    // injection)
    if (n_sources_per_shot > 0) {
      record_receivers_ey(grad_f + t * n_shots * n_sources_per_shot, lambda_ey,
                          sources_i, n_shots, shot_numel, n_sources_per_shot);
    }

    // Backward λ_H fields update
    backward_kernel_lambda_h(cb, lambda_ey, lambda_hx, lambda_hz, m_lambda_ey_x,
                             m_lambda_ey_z, ay, ayh, ax, axh, by, byh, bx, bxh,
                             ky, kyh, kx, kxh, rdy, rdx, n_shots, ny, nx,
                             shot_numel, pml_y0, pml_y1, pml_x0, pml_x1,
                             cb_batched, work_x, work_z);

    // Backward λ_Ey update with gradient accumulation
    // This computes: λ_Ey^n = C_a * λ_Ey^{n+1} + C_q * curl_λH
    // And accumulates: grad_ca += λ_Ey^{n+1} * E_y^n, grad_cb += λ_Ey^{n+1} *
    // curl_H^n
    if (storage_bf16 && (grad_ey || grad_curl)) {
      tide_bfloat16 *const ey_store_1_t =
          (tide_bfloat16 *)ey_store_1 + store_offset;
      tide_bfloat16 *const curl_store_1_t =
          (tide_bfloat16 *)curl_store_1 + store_offset;
      backward_kernel_lambda_e_with_grad_bf16(
          ca, cq, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z, m_lambda_hz_x,
          grad_ey ? ey_store_1_t : NULL, grad_curl ? curl_store_1_t : NULL,
          grad_ca_accum, grad_cb_accum, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
          kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0, pml_y1,
          pml_x0, pml_x1, ca_batched, cq_batched, grad_ey, grad_curl, step_ratio,
          work_x, work_z);
    } else {
      TIDE_DTYPE *const ey_store_1_t =
          storage_bf16 ? NULL : (ey_store_1 + store_offset);
      TIDE_DTYPE *const curl_store_1_t =
          storage_bf16 ? NULL : (curl_store_1 + store_offset);
      backward_kernel_lambda_e_with_grad(
          ca, cq, lambda_hx, lambda_hz, lambda_ey, m_lambda_hx_z, m_lambda_hz_x,
          grad_ey ? ey_store_1_t : NULL, grad_curl ? curl_store_1_t : NULL,
          grad_ca_accum, grad_cb_accum, ay, ayh, ax, axh, by, byh, bx, bxh, ky,
          kyh, kx, kxh, rdy, rdx, n_shots, ny, nx, shot_numel, pml_y0, pml_y1,
          pml_x0, pml_x1, ca_batched, cq_batched, grad_ey, grad_curl, step_ratio,
          work_x, work_z);
    }
  }

  if (reduce_grad_ca) {
    TIDE_OMP_INDEX j;
    TIDE_OMP_PARALLEL_FOR_IF(shot_numel >= 4096)
    for (j = 0; j < shot_numel; ++j) {
      TIDE_DTYPE sum = (TIDE_DTYPE)0;
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        sum += grad_ca_accum[shot * shot_numel + j];
      }
      grad_ca[j] += sum;
    }
  }

  if (reduce_grad_cb) {
    TIDE_OMP_INDEX j;
    TIDE_OMP_PARALLEL_FOR_IF(shot_numel >= 4096)
    for (j = 0; j < shot_numel; ++j) {
      TIDE_DTYPE sum = (TIDE_DTYPE)0;
      for (int64_t shot = 0; shot < n_shots; ++shot) {
        sum += grad_cb_accum[shot * shot_numel + j];
      }
      grad_cb[j] += sum;
    }
  }

  if (fp_ey != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot)
      fclose(fp_ey[shot]);
    free(fp_ey);
  }
  if (fp_curl != NULL) {
    for (int64_t shot = 0; shot < n_shots; ++shot)
      fclose(fp_curl[shot]);
    free(fp_curl);
  }
#ifdef _OPENMP
  if (n_threads > 0) {
    omp_set_num_threads(prev_threads);
  }
#endif
}

#undef TIDE_FD_PAD
#undef EY
#undef HX
#undef HZ


} // namespace FUNC(Inst)
