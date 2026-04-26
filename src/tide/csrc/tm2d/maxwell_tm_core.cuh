#ifndef MAXWELL_TM_CORE_CUH
#define MAXWELL_TM_CORE_CUH

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

#if defined(__CUDACC__)
#define TIDE_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define TIDE_HOST_DEVICE inline
#endif

// 2D indexing macro
#define ND_INDEX(i, dy, dx) (i + (dy) * nx + (dx))

namespace tide {

template <typename scalar_t> struct GridParams {
  scalar_t const *ay;
  scalar_t const *ayh;
  scalar_t const *ax;
  scalar_t const *axh;
  scalar_t const *by;
  scalar_t const *byh;
  scalar_t const *bx;
  scalar_t const *bxh;
  scalar_t const *ky;
  scalar_t const *kyh;
  scalar_t const *kx;
  scalar_t const *kxh;
  scalar_t rdy;
  scalar_t rdx;
  int64_t n_shots;
  int64_t ny;
  int64_t nx;
  int64_t shot_numel;
  int64_t pml_y0;
  int64_t pml_y1;
  int64_t pml_x0;
  int64_t pml_x1;
  bool ca_batched;
  bool cb_batched;
  bool cq_batched;
};

// Compile-time stencil properties
template <int STENCIL_ORDER> struct StencilTraits;

template <> struct StencilTraits<2> {
  static constexpr int FD_PAD = 1;
};
template <> struct StencilTraits<4> {
  static constexpr int FD_PAD = 2;
};
template <> struct StencilTraits<6> {
  static constexpr int FD_PAD = 3;
};
template <> struct StencilTraits<8> {
  static constexpr int FD_PAD = 4;
};

// Forward finite differences
template <int STENCIL_ORDER> struct DiffForward;

template <> struct DiffForward<2> {
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_y1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdy) {
    return (F(i, y, x) - F(i, y - 1, x)) * rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_x1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdx) {
    return (F(i, y, x) - F(i, y, x - 1)) * rdx;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_yh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdy) {
    return (F(i, y + 1, x) - F(i, y, x)) * rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_xh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdx) {
    return (F(i, y, x + 1) - F(i, y, x)) * rdx;
  }
};

template <> struct DiffForward<4> {
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_y1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdy) {
    return ((T)(9.0 / 8.0) * (F(i, y, x) - F(i, y - 1, x)) +
            (T)(-1.0 / 24.0) * (F(i, y + 1, x) - F(i, y - 2, x))) *
           rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_x1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdx) {
    return ((T)(9.0 / 8.0) * (F(i, y, x) - F(i, y, x - 1)) +
            (T)(-1.0 / 24.0) * (F(i, y, x + 1) - F(i, y, x - 2))) *
           rdx;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_yh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdy) {
    return ((T)(9.0 / 8.0) * (F(i, y + 1, x) - F(i, y, x)) +
            (T)(-1.0 / 24.0) * (F(i, y + 2, x) - F(i, y - 1, x))) *
           rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_xh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdx) {
    return ((T)(9.0 / 8.0) * (F(i, y, x + 1) - F(i, y, x)) +
            (T)(-1.0 / 24.0) * (F(i, y, x + 2) - F(i, y, x - 1))) *
           rdx;
  }
};

template <> struct DiffForward<6> {
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_y1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdy) {
    return ((T)(75.0 / 64.0) * (F(i, y, x) - F(i, y - 1, x)) +
            (T)(-25.0 / 384.0) * (F(i, y + 1, x) - F(i, y - 2, x)) +
            (T)(3.0 / 640.0) * (F(i, y + 2, x) - F(i, y - 3, x))) *
           rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_x1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdx) {
    return ((T)(75.0 / 64.0) * (F(i, y, x) - F(i, y, x - 1)) +
            (T)(-25.0 / 384.0) * (F(i, y, x + 1) - F(i, y, x - 2)) +
            (T)(3.0 / 640.0) * (F(i, y, x + 2) - F(i, y, x - 3))) *
           rdx;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_yh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdy) {
    return ((T)(75.0 / 64.0) * (F(i, y + 1, x) - F(i, y, x)) +
            (T)(-25.0 / 384.0) * (F(i, y + 2, x) - F(i, y - 1, x)) +
            (T)(3.0 / 640.0) * (F(i, y + 3, x) - F(i, y - 2, x))) *
           rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_xh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdx) {
    return ((T)(75.0 / 64.0) * (F(i, y, x + 1) - F(i, y, x)) +
            (T)(-25.0 / 384.0) * (F(i, y, x + 2) - F(i, y, x - 1)) +
            (T)(3.0 / 640.0) * (F(i, y, x + 3) - F(i, y, x - 2))) *
           rdx;
  }
};

template <> struct DiffForward<8> {
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_y1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdy) {
    return ((T)(1225.0 / 1024.0) * (F(i, y, x) - F(i, y - 1, x)) +
            (T)(-245.0 / 3072.0) * (F(i, y + 1, x) - F(i, y - 2, x)) +
            (T)(49.0 / 5120.0) * (F(i, y + 2, x) - F(i, y - 3, x)) +
            (T)(-5.0 / 7168.0) * (F(i, y + 3, x) - F(i, y - 4, x))) *
           rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_x1(Accessor F, int64_t i, int y, int x, int nx,
                                    T rdx) {
    return ((T)(1225.0 / 1024.0) * (F(i, y, x) - F(i, y, x - 1)) +
            (T)(-245.0 / 3072.0) * (F(i, y, x + 1) - F(i, y, x - 2)) +
            (T)(49.0 / 5120.0) * (F(i, y, x + 2) - F(i, y, x - 3)) +
            (T)(-5.0 / 7168.0) * (F(i, y, x + 3) - F(i, y, x - 4))) *
           rdx;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_yh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdy) {
    return ((T)(1225.0 / 1024.0) * (F(i, y + 1, x) - F(i, y, x)) +
            (T)(-245.0 / 3072.0) * (F(i, y + 2, x) - F(i, y - 1, x)) +
            (T)(49.0 / 5120.0) * (F(i, y + 3, x) - F(i, y - 2, x)) +
            (T)(-5.0 / 7168.0) * (F(i, y + 4, x) - F(i, y - 3, x))) *
           rdy;
  }
  template <typename Accessor, typename T>
  static TIDE_HOST_DEVICE T diff_xh1(Accessor F, int64_t i, int y, int x,
                                     int nx, T rdx) {
    return ((T)(1225.0 / 1024.0) * (F(i, y, x + 1) - F(i, y, x)) +
            (T)(-245.0 / 3072.0) * (F(i, y, x + 2) - F(i, y, x - 1)) +
            (T)(49.0 / 5120.0) * (F(i, y, x + 3) - F(i, y, x - 2)) +
            (T)(-5.0 / 7168.0) * (F(i, y, x + 4) - F(i, y, x - 3))) *
           rdx;
  }
};

// Adjoint finite differences (transpose operators)
template <int STENCIL_ORDER> struct DiffAdjoint;

template <> struct DiffAdjoint<2> {
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_y1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdy) {
    return (c_acc(i, y, x) * f_acc(i, y, x) -
            c_acc(i, y + 1, x) * f_acc(i, y + 1, x)) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_x1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdx) {
    return (c_acc(i, y, x) * f_acc(i, y, x) -
            c_acc(i, y, x + 1) * f_acc(i, y, x + 1)) *
           rdx;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_yh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdy) {
    return (c_acc(i, y - 1, x) * f_acc(i, y - 1, x) -
            c_acc(i, y, x) * f_acc(i, y, x)) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_xh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdx) {
    return (c_acc(i, y, x - 1) * f_acc(i, y, x - 1) -
            c_acc(i, y, x) * f_acc(i, y, x)) *
           rdx;
  }
};

template <> struct DiffAdjoint<4> {
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_y1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdy) {
    return ((T)(9.0 / 8.0) * (c_acc(i, y, x) * f_acc(i, y, x) -
                              c_acc(i, y + 1, x) * f_acc(i, y + 1, x)) +
            (T)(-1.0 / 24.0) * (c_acc(i, y - 1, x) * f_acc(i, y - 1, x) -
                                c_acc(i, y + 2, x) * f_acc(i, y + 2, x))) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_x1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdx) {
    return ((T)(9.0 / 8.0) * (c_acc(i, y, x) * f_acc(i, y, x) -
                              c_acc(i, y, x + 1) * f_acc(i, y, x + 1)) +
            (T)(-1.0 / 24.0) * (c_acc(i, y, x - 1) * f_acc(i, y, x - 1) -
                                c_acc(i, y, x + 2) * f_acc(i, y, x + 2))) *
           rdx;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_yh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdy) {
    return ((T)(9.0 / 8.0) * (c_acc(i, y - 1, x) * f_acc(i, y - 1, x) -
                              c_acc(i, y, x) * f_acc(i, y, x)) +
            (T)(-1.0 / 24.0) * (c_acc(i, y - 2, x) * f_acc(i, y - 2, x) -
                                c_acc(i, y + 1, x) * f_acc(i, y + 1, x))) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_xh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdx) {
    return ((T)(9.0 / 8.0) * (c_acc(i, y, x - 1) * f_acc(i, y, x - 1) -
                              c_acc(i, y, x) * f_acc(i, y, x)) +
            (T)(-1.0 / 24.0) * (c_acc(i, y, x - 2) * f_acc(i, y, x - 2) -
                                c_acc(i, y, x + 1) * f_acc(i, y, x + 1))) *
           rdx;
  }
};

template <> struct DiffAdjoint<6> {
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_y1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdy) {
    return ((T)(75.0 / 64.0) * (c_acc(i, y, x) * f_acc(i, y, x) -
                                c_acc(i, y + 1, x) * f_acc(i, y + 1, x)) +
            (T)(-25.0 / 384.0) * (c_acc(i, y - 1, x) * f_acc(i, y - 1, x) -
                                  c_acc(i, y + 2, x) * f_acc(i, y + 2, x)) +
            (T)(3.0 / 640.0) * (c_acc(i, y - 2, x) * f_acc(i, y - 2, x) -
                                c_acc(i, y + 3, x) * f_acc(i, y + 3, x))) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_x1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdx) {
    return ((T)(75.0 / 64.0) * (c_acc(i, y, x) * f_acc(i, y, x) -
                                c_acc(i, y, x + 1) * f_acc(i, y, x + 1)) +
            (T)(-25.0 / 384.0) * (c_acc(i, y, x - 1) * f_acc(i, y, x - 1) -
                                  c_acc(i, y, x + 2) * f_acc(i, y, x + 2)) +
            (T)(3.0 / 640.0) * (c_acc(i, y, x - 2) * f_acc(i, y, x - 2) -
                                c_acc(i, y, x + 3) * f_acc(i, y, x + 3))) *
           rdx;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_yh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdy) {
    return ((T)(75.0 / 64.0) * (c_acc(i, y - 1, x) * f_acc(i, y - 1, x) -
                                c_acc(i, y, x) * f_acc(i, y, x)) +
            (T)(-25.0 / 384.0) * (c_acc(i, y - 2, x) * f_acc(i, y - 2, x) -
                                  c_acc(i, y + 1, x) * f_acc(i, y + 1, x)) +
            (T)(3.0 / 640.0) * (c_acc(i, y - 3, x) * f_acc(i, y - 3, x) -
                                c_acc(i, y + 2, x) * f_acc(i, y + 2, x))) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_xh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdx) {
    return ((T)(75.0 / 64.0) * (c_acc(i, y, x - 1) * f_acc(i, y, x - 1) -
                                c_acc(i, y, x) * f_acc(i, y, x)) +
            (T)(-25.0 / 384.0) * (c_acc(i, y, x - 2) * f_acc(i, y, x - 2) -
                                  c_acc(i, y, x + 1) * f_acc(i, y, x + 1)) +
            (T)(3.0 / 640.0) * (c_acc(i, y, x - 3) * f_acc(i, y, x - 3) -
                                c_acc(i, y, x + 2) * f_acc(i, y, x + 2))) *
           rdx;
  }
};

template <> struct DiffAdjoint<8> {
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_y1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdy) {
    return ((T)(1225.0 / 1024.0) * (c_acc(i, y, x) * f_acc(i, y, x) -
                                    c_acc(i, y + 1, x) * f_acc(i, y + 1, x)) +
            (T)(-245.0 / 3072.0) * (c_acc(i, y - 1, x) * f_acc(i, y - 1, x) -
                                    c_acc(i, y + 2, x) * f_acc(i, y + 2, x)) +
            (T)(49.0 / 5120.0) * (c_acc(i, y - 2, x) * f_acc(i, y - 2, x) -
                                  c_acc(i, y + 3, x) * f_acc(i, y + 3, x)) +
            (T)(-5.0 / 7168.0) * (c_acc(i, y - 3, x) * f_acc(i, y - 3, x) -
                                  c_acc(i, y + 4, x) * f_acc(i, y + 4, x))) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_x1_adj(C c_acc, F f_acc, int64_t i, int y,
                                        int x, int nx, T rdx) {
    return ((T)(1225.0 / 1024.0) * (c_acc(i, y, x) * f_acc(i, y, x) -
                                    c_acc(i, y, x + 1) * f_acc(i, y, x + 1)) +
            (T)(-245.0 / 3072.0) * (c_acc(i, y, x - 1) * f_acc(i, y, x - 1) -
                                    c_acc(i, y, x + 2) * f_acc(i, y, x + 2)) +
            (T)(49.0 / 5120.0) * (c_acc(i, y, x - 2) * f_acc(i, y, x - 2) -
                                  c_acc(i, y, x + 3) * f_acc(i, y, x + 3)) +
            (T)(-5.0 / 7168.0) * (c_acc(i, y, x - 3) * f_acc(i, y, x - 3) -
                                  c_acc(i, y, x + 4) * f_acc(i, y, x + 4))) *
           rdx;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_yh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdy) {
    return ((T)(1225.0 / 1024.0) * (c_acc(i, y - 1, x) * f_acc(i, y - 1, x) -
                                    c_acc(i, y, x) * f_acc(i, y, x)) +
            (T)(-245.0 / 3072.0) * (c_acc(i, y - 2, x) * f_acc(i, y - 2, x) -
                                    c_acc(i, y + 1, x) * f_acc(i, y + 1, x)) +
            (T)(49.0 / 5120.0) * (c_acc(i, y - 3, x) * f_acc(i, y - 3, x) -
                                  c_acc(i, y + 2, x) * f_acc(i, y + 2, x)) +
            (T)(-5.0 / 7168.0) * (c_acc(i, y - 4, x) * f_acc(i, y - 4, x) -
                                  c_acc(i, y + 3, x) * f_acc(i, y + 3, x))) *
           rdy;
  }
  template <typename C, typename F, typename T>
  static TIDE_HOST_DEVICE T diff_xh1_adj(C c_acc, F f_acc, int64_t i, int y,
                                         int x, int nx, T rdx) {
    return ((T)(1225.0 / 1024.0) * (c_acc(i, y, x - 1) * f_acc(i, y, x - 1) -
                                    c_acc(i, y, x) * f_acc(i, y, x)) +
            (T)(-245.0 / 3072.0) * (c_acc(i, y, x - 2) * f_acc(i, y, x - 2) -
                                    c_acc(i, y, x + 1) * f_acc(i, y, x + 1)) +
            (T)(49.0 / 5120.0) * (c_acc(i, y, x - 3) * f_acc(i, y, x - 3) -
                                  c_acc(i, y, x + 2) * f_acc(i, y, x + 2)) +
            (T)(-5.0 / 7168.0) * (c_acc(i, y, x - 4) * f_acc(i, y, x - 4) -
                                  c_acc(i, y, x + 3) * f_acc(i, y, x + 3))) *
           rdx;
  }
};

// Accessor for simple pointers
template <typename T> struct FieldAccessor {
  T const *ptr;
  TIDE_HOST_DEVICE FieldAccessor(T const *p) : ptr(p) {}
  TIDE_HOST_DEVICE T operator()(int64_t base, int y, int x) const {
    return ptr[base + y]; // 1D accessor: y is the offset from base.
  }
};

template <typename T> struct GlobalFieldAccessor {
  T const *ptr;
  int nx;
  TIDE_HOST_DEVICE GlobalFieldAccessor(T const *p, int w) : ptr(p), nx(w) {}
  TIDE_HOST_DEVICE T operator()(int64_t base, int y, int x) const {
    return ptr[base + y * nx + x]; // base is shot_idx * shot_numel
  }
};

// Simple Constant Accessor (for adjoint operator signature matching)
struct ConstAccessor {
  TIDE_HOST_DEVICE int operator()(int64_t, int, int) const { return 1; }
};

template <typename StoreT> struct SnapshotCodec {
  template <typename T>
  static TIDE_HOST_DEVICE StoreT encode(T const value) {
    return static_cast<StoreT>(value);
  }

  template <typename T>
  static TIDE_HOST_DEVICE T decode(StoreT const value) {
    return static_cast<T>(value);
  }
};

template <> struct SnapshotCodec<uint16_t> {
  template <typename T>
  static TIDE_HOST_DEVICE uint16_t encode(T const value) {
#if defined(__CUDACC__)
    // CUDA path does not use uint16_t snapshot storage.
    return static_cast<uint16_t>(value);
#else
    return tide_float_to_bf16(static_cast<float>(value));
#endif
  }

  template <typename T>
  static TIDE_HOST_DEVICE T decode(uint16_t const value) {
#if defined(__CUDACC__)
    return static_cast<T>(value);
#else
    return static_cast<T>(tide_bf16_to_float(value));
#endif
  }
};

#if defined(__CUDACC__)
template <> struct SnapshotCodec<__half> {
  template <typename T>
  static TIDE_HOST_DEVICE __half encode(T const value) {
    return __float2half(static_cast<float>(value));
  }

  template <typename T>
  static TIDE_HOST_DEVICE T decode(__half const value) {
    return static_cast<T>(__half2float(value));
  }
};

template <> struct SnapshotCodec<__nv_bfloat16> {
  template <typename T>
  static TIDE_HOST_DEVICE __nv_bfloat16 encode(T const value) {
    return __float2bfloat16(static_cast<float>(value));
  }

  template <typename T>
  static TIDE_HOST_DEVICE T decode(__nv_bfloat16 const value) {
    return static_cast<T>(__bfloat162float(value));
  }
};
#endif

template <typename StoreT, typename T>
static TIDE_HOST_DEVICE StoreT encode_snapshot(T const value) {
  return SnapshotCodec<StoreT>::template encode<T>(value);
}

template <typename StoreT, typename T>
static TIDE_HOST_DEVICE T decode_snapshot(StoreT const value) {
  return SnapshotCodec<StoreT>::template decode<T>(value);
}

// Update H fields (Hx and Hz)
template <typename T, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void
forward_kernel_h_core(GridParams<T> const &params, T const *cq_ptr, T const *ey,
                      T *hx, T *hz, T *m_ey_x, T *m_ey_z, int64_t y, int64_t x,
                      int64_t shot_idx) {

  int const FD_PAD = StencilTraits<STENCIL_ORDER>::FD_PAD;

  if (y >= FD_PAD && x >= FD_PAD && y < params.ny - FD_PAD + 1 &&
      x < params.nx - FD_PAD + 1 && shot_idx < params.n_shots) {
    int64_t const pml_y0h = params.pml_y0;
    int64_t const pml_y1h =
        params.pml_y1 > params.pml_y0 ? params.pml_y1 - 1 : params.pml_y0;
    int64_t const pml_x0h = params.pml_x0;
    int64_t const pml_x1h =
        params.pml_x1 > params.pml_x0 ? params.pml_x1 - 1 : params.pml_x0;

    int64_t j = y * params.nx + x;
    int64_t i = shot_idx * params.shot_numel + j;

    T const cq_val = params.cq_batched ? cq_ptr[i] : cq_ptr[j];

    // Update Hx: Hx = Hx - cq * dEy/dz
    if (y < params.ny - FD_PAD) {
      bool pml_y = y < pml_y0h || y >= pml_y1h;
      GlobalFieldAccessor<T> ey_acc(ey, params.nx);

      T dey_dz = DiffForward<STENCIL_ORDER>::diff_yh1(
          ey_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);

      if (pml_y) {
        m_ey_z[i] = params.byh[y] * m_ey_z[i] + params.ayh[y] * dey_dz;
        dey_dz = dey_dz / params.kyh[y] + m_ey_z[i];
      }

      hx[i] -= cq_val * dey_dz;
    }

    // Update Hz: Hz = Hz + cq * dEy/dx
    if (x < params.nx - FD_PAD) {
      bool pml_x = x < pml_x0h || x >= pml_x1h;
      GlobalFieldAccessor<T> ey_acc(ey, params.nx);

      T dey_dx = DiffForward<STENCIL_ORDER>::diff_xh1(
          ey_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);

      if (pml_x) {
        m_ey_x[i] = params.bxh[x] * m_ey_x[i] + params.axh[x] * dey_dx;
        dey_dx = dey_dx / params.kxh[x] + m_ey_x[i];
      }

      hz[i] += cq_val * dey_dx;
    }
  }
}

// Update E field (Ey) - standard version
template <typename T, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void
forward_kernel_e_core(GridParams<T> const &params, T const *ca_ptr,
                      T const *cb_ptr, T const *hx, T const *hz, T *ey,
                      T *m_hx_z, T *m_hz_x, int64_t y, int64_t x,
                      int64_t shot_idx) {

  int const FD_PAD = StencilTraits<STENCIL_ORDER>::FD_PAD;

  if (y >= FD_PAD && x >= FD_PAD && y < params.ny - FD_PAD + 1 &&
      x < params.nx - FD_PAD + 1 && shot_idx < params.n_shots) {
    int64_t j = y * params.nx + x;
    int64_t i = shot_idx * params.shot_numel + j;

    T const ca_val = params.ca_batched ? ca_ptr[i] : ca_ptr[j];
    T const cb_val = params.cb_batched ? cb_ptr[i] : cb_ptr[j];

    bool pml_y = y < params.pml_y0 || y >= params.pml_y1;
    bool pml_x = x < params.pml_x0 || x >= params.pml_x1;

    GlobalFieldAccessor<T> L_HZ(hz, params.nx);
    GlobalFieldAccessor<T> L_HX(hx, params.nx);

    T dhz_dx = DiffForward<STENCIL_ORDER>::diff_x1(
        L_HZ, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);
    T dhx_dz = DiffForward<STENCIL_ORDER>::diff_y1(
        L_HX, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);

    if (pml_x) {
      m_hz_x[i] = params.bx[x] * m_hz_x[i] + params.ax[x] * dhz_dx;
      dhz_dx = dhz_dx / params.kx[x] + m_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = params.by[y] * m_hx_z[i] + params.ay[y] * dhx_dz;
      dhx_dz = dhx_dz / params.ky[y] + m_hx_z[i];
    }

    T curl_h = dhz_dx - dhx_dz;

    ey[i] = ca_val * ey[i] + cb_val * curl_h;
  }
}

// Update E field (Ey) with generic storage saving mechanics
template <typename T, typename StoreT, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void forward_kernel_e_with_storage_core(
    GridParams<T> const &params, T const *ca_ptr, T const *cb_ptr, T const *hx,
    T const *hz, T *ey, T *m_hx_z, T *m_hz_x, StoreT *ey_store,
    StoreT *curl_h_store, bool ca_requires_grad, bool cb_requires_grad,
    int64_t y, int64_t x, int64_t shot_idx) {

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;

  if (y >= FD_PAD && x >= FD_PAD && y < params.ny - FD_PAD + 1 &&
      x < params.nx - FD_PAD + 1 && shot_idx < params.n_shots) {
    int64_t j = y * params.nx + x;
    int64_t i = shot_idx * params.shot_numel + j;

    T const ca_val = params.ca_batched ? ca_ptr[i] : ca_ptr[j];
    T const cb_val = params.cb_batched ? cb_ptr[i] : cb_ptr[j];

    bool pml_y = y < params.pml_y0 || y >= params.pml_y1;
    bool pml_x = x < params.pml_x0 || x >= params.pml_x1;

    GlobalFieldAccessor<T> L_HZ(hz, params.nx);
    GlobalFieldAccessor<T> L_HX(hx, params.nx);

    T dhz_dx = DiffForward<STENCIL_ORDER>::diff_x1(
        L_HZ, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);
    T dhx_dz = DiffForward<STENCIL_ORDER>::diff_y1(
        L_HX, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);

    if (pml_x) {
      m_hz_x[i] = params.bx[x] * m_hz_x[i] + params.ax[x] * dhz_dx;
      dhz_dx = dhz_dx / params.kx[x] + m_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = params.by[y] * m_hx_z[i] + params.ay[y] * dhx_dz;
      dhx_dz = dhx_dz / params.ky[y] + m_hx_z[i];
    }

    T curl_h = dhz_dx - dhx_dz;

    if (ca_requires_grad && ey_store != nullptr) {
      ey_store[i] = encode_snapshot<StoreT, T>(ey[i]);
    }
    if (cb_requires_grad && curl_h_store != nullptr) {
      curl_h_store[i] = encode_snapshot<StoreT, T>(curl_h);
    }

    ey[i] = ca_val * ey[i] + cb_val * curl_h;
  }
}

// Update background and scattered E fields for Born propagation.
template <typename T, typename StoreT, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void forward_kernel_e_born_with_storage_core(
    GridParams<T> const &params, T const *ca_ptr, T const *cb_ptr,
    T const *dca_ptr, T const *dcb_ptr, T const *hx, T const *hz, T *ey,
    T *m_hx_z, T *m_hz_x, T const *dhx, T const *dhz, T *dey, T *dm_hx_z,
    T *dm_hz_x, StoreT *ey_store, StoreT *curl_h_store, T *dey_store,
    T *dcurl_h_store,
    bool ca_requires_grad, bool cb_requires_grad, int64_t y, int64_t x,
    int64_t shot_idx) {

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;

  if (y >= FD_PAD && x >= FD_PAD && y < params.ny - FD_PAD + 1 &&
      x < params.nx - FD_PAD + 1 && shot_idx < params.n_shots) {
    int64_t const j = y * params.nx + x;
    int64_t const i = shot_idx * params.shot_numel + j;

    T const ca_val = params.ca_batched ? ca_ptr[i] : ca_ptr[j];
    T const cb_val = params.cb_batched ? cb_ptr[i] : cb_ptr[j];
    T const dca_val = params.ca_batched ? dca_ptr[i] : dca_ptr[j];
    T const dcb_val = params.cb_batched ? dcb_ptr[i] : dcb_ptr[j];

    bool const pml_y = y < params.pml_y0 || y >= params.pml_y1;
    bool const pml_x = x < params.pml_x0 || x >= params.pml_x1;

    GlobalFieldAccessor<T> bg_hz_acc(hz, params.nx);
    GlobalFieldAccessor<T> bg_hx_acc(hx, params.nx);

    T dhz_dx = DiffForward<STENCIL_ORDER>::diff_x1(
        bg_hz_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);
    T dhx_dz = DiffForward<STENCIL_ORDER>::diff_y1(
        bg_hx_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);

    if (pml_x) {
      m_hz_x[i] = params.bx[x] * m_hz_x[i] + params.ax[x] * dhz_dx;
      dhz_dx = dhz_dx / params.kx[x] + m_hz_x[i];
    }

    if (pml_y) {
      m_hx_z[i] = params.by[y] * m_hx_z[i] + params.ay[y] * dhx_dz;
      dhx_dz = dhx_dz / params.ky[y] + m_hx_z[i];
    }

    T const curl_h = dhz_dx - dhx_dz;

    GlobalFieldAccessor<T> sc_hz_acc(dhz, params.nx);
    GlobalFieldAccessor<T> sc_hx_acc(dhx, params.nx);

    T ddhz_dx = DiffForward<STENCIL_ORDER>::diff_x1(
        sc_hz_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);
    T ddhx_dz = DiffForward<STENCIL_ORDER>::diff_y1(
        sc_hx_acc, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);

    if (pml_x) {
      dm_hz_x[i] = params.bx[x] * dm_hz_x[i] + params.ax[x] * ddhz_dx;
      ddhz_dx = ddhz_dx / params.kx[x] + dm_hz_x[i];
    }

    if (pml_y) {
      dm_hx_z[i] = params.by[y] * dm_hx_z[i] + params.ay[y] * ddhx_dz;
      ddhx_dz = ddhx_dz / params.ky[y] + dm_hx_z[i];
    }

    T const dcurl_h = ddhz_dx - ddhx_dz;
    T const ey_n = ey[i];
    T const dey_n = dey[i];

    if (ca_requires_grad && ey_store != nullptr) {
      ey_store[i] = encode_snapshot<StoreT, T>(ey_n);
    }
    if (cb_requires_grad && curl_h_store != nullptr) {
      curl_h_store[i] = encode_snapshot<StoreT, T>(curl_h);
    }
    if (dey_store != nullptr) {
      dey_store[i] = dey_n;
    }
    if (dcurl_h_store != nullptr) {
      dcurl_h_store[i] = dcurl_h;
    }

    ey[i] = ca_val * ey_n + cb_val * curl_h;
    dey[i] = ca_val * dey_n + cb_val * dcurl_h + dca_val * ey_n +
             dcb_val * curl_h;
  }
}

// Direct bggrad preparation for the full Hessian path. This accumulates the
// local ca/cb direct term and emits the transposed dcb * lambda contribution
// into alpha_h* without requiring scattered CPML memory reconstruction.
template <typename T, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void born_background_prepare_direct_core(
    GridParams<T> const &params, T const *dca_ptr, T const *dcb_ptr,
    T const *lambda_sc_ey, T const *dey_store, T const *dcurl_h_store,
    T *grad_ca_shot, T *grad_cb_shot, T *eta_source_old, T *alpha_hz_x,
    T *alpha_hx_z, int64_t step_ratio_val, int64_t y, int64_t x,
    int64_t shot_idx) {

  if (y < 0 || x < 0 || y >= params.ny || x >= params.nx ||
      shot_idx >= params.n_shots) {
    return;
  }

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;
  int64_t const j = y * params.nx + x;
  int64_t const i = shot_idx * params.shot_numel + j;

  alpha_hz_x[i] = static_cast<T>(0);
  alpha_hx_z[i] = static_cast<T>(0);
  eta_source_old[i] = static_cast<T>(0);

  if (y < FD_PAD || x < FD_PAD || y >= params.ny - FD_PAD + 1 ||
      x >= params.nx - FD_PAD + 1) {
    return;
  }

  T const dca_val = params.ca_batched ? dca_ptr[i] : dca_ptr[j];
  T const dcb_val = params.cb_batched ? dcb_ptr[i] : dcb_ptr[j];
  T const lambda_curr = lambda_sc_ey[i];

  if (grad_ca_shot != nullptr && dey_store != nullptr) {
    grad_ca_shot[i] += lambda_curr * dey_store[i] * static_cast<T>(step_ratio_val);
  }
  if (grad_cb_shot != nullptr && dcurl_h_store != nullptr) {
    grad_cb_shot[i] +=
        lambda_curr * dcurl_h_store[i] * static_cast<T>(step_ratio_val);
  }

  eta_source_old[i] = dca_val * lambda_curr;

  T const beta_x = dcb_val * lambda_curr;
  if (x >= params.pml_x0 && x < params.pml_x1) {
    alpha_hz_x[i] = beta_x;
  } else {
    alpha_hz_x[i] = beta_x / params.kx[x] + params.ax[x] * beta_x;
  }

  T const beta_z = -dcb_val * lambda_curr;
  if (y >= params.pml_y0 && y < params.pml_y1) {
    alpha_hx_z[i] = beta_z;
  } else {
    alpha_hx_z[i] = beta_z / params.ky[y] + params.ay[y] * beta_z;
  }
}

// Reverse of the scattered E update. Produces transpose fluxes into the
// scattered H adjoint variables and accumulates gradients with respect to the
// Born coefficients dca / dcb.
template <typename T, typename StoreT, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void born_backward_prepare_e_core(
    GridParams<T> const &params, T const *ca_ptr, T const *cb_ptr, T *lambda_ey,
    T *m_lambda_hx_z, T *m_lambda_hz_x, StoreT const *ey_store,
    StoreT const *curl_h_store, T *grad_ca_shot, T *grad_cb_shot,
    T *alpha_hz_x, T *alpha_hx_z, bool ca_requires_grad, bool cb_requires_grad,
    int64_t step_ratio_val, int64_t y, int64_t x, int64_t shot_idx) {

  if (y < 0 || x < 0 || y >= params.ny || x >= params.nx ||
      shot_idx >= params.n_shots) {
    return;
  }

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;
  int64_t const j = y * params.nx + x;
  int64_t const i = shot_idx * params.shot_numel + j;
  alpha_hz_x[i] = static_cast<T>(0);
  alpha_hx_z[i] = static_cast<T>(0);

  if (y < FD_PAD || x < FD_PAD || y >= params.ny - FD_PAD + 1 ||
      x >= params.nx - FD_PAD + 1) {
    return;
  }

  T const ca_val = params.ca_batched ? ca_ptr[i] : ca_ptr[j];
  T const cb_val = params.cb_batched ? cb_ptr[i] : cb_ptr[j];
  T const lambda_ey_curr = lambda_ey[i];

  if (ca_requires_grad && ey_store != nullptr && grad_ca_shot != nullptr) {
    T const ey_n = decode_snapshot<StoreT, T>(ey_store[i]);
    grad_ca_shot[i] +=
        lambda_ey_curr * ey_n * static_cast<T>(step_ratio_val);
  }

  if (cb_requires_grad && curl_h_store != nullptr && grad_cb_shot != nullptr) {
    T const curl_h_n = decode_snapshot<StoreT, T>(curl_h_store[i]);
    grad_cb_shot[i] +=
        lambda_ey_curr * curl_h_n * static_cast<T>(step_ratio_val);
  }

  lambda_ey[i] = ca_val * lambda_ey_curr;

  T beta_x = cb_val * lambda_ey_curr;
  if (x >= params.pml_x0 && x < params.pml_x1) {
    alpha_hz_x[i] = beta_x;
  } else {
    T const gamma_x = m_lambda_hz_x[i] + beta_x;
    alpha_hz_x[i] = beta_x / params.kx[x] + params.ax[x] * gamma_x;
    m_lambda_hz_x[i] = params.bx[x] * gamma_x;
  }

  T beta_z = -cb_val * lambda_ey_curr;
  if (y >= params.pml_y0 && y < params.pml_y1) {
    alpha_hx_z[i] = beta_z;
  } else {
    T const gamma_z = m_lambda_hx_z[i] + beta_z;
    alpha_hx_z[i] = beta_z / params.ky[y] + params.ay[y] * gamma_z;
    m_lambda_hx_z[i] = params.by[y] * gamma_z;
  }
}

// Apply the transposed scattered E update to the scattered H adjoint fields.
template <typename T, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void born_backward_apply_e_to_h_core(
    GridParams<T> const &params, T const *alpha_hz_x, T const *alpha_hx_z,
    T *lambda_hx, T *lambda_hz, int64_t y, int64_t x, int64_t shot_idx) {

  if (y < 0 || x < 0 || y >= params.ny || x >= params.nx ||
      shot_idx >= params.n_shots) {
    return;
  }

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;
  if (y < FD_PAD || x < FD_PAD || y >= params.ny - FD_PAD + 1 ||
      x >= params.nx - FD_PAD + 1) {
    return;
  }

  int64_t const i = shot_idx * params.shot_numel + y * params.nx + x;
  int64_t const shot_offset = shot_idx * params.shot_numel;
  GlobalFieldAccessor<T> alpha_x_acc(alpha_hz_x, params.nx);
  GlobalFieldAccessor<T> alpha_z_acc(alpha_hx_z, params.nx);
  ConstAccessor ones;

  if (y < params.ny - FD_PAD) {
    lambda_hx[i] += DiffAdjoint<STENCIL_ORDER>::diff_y1_adj(
        alpha_z_acc, ones, shot_offset, (int)y, (int)x, params.nx, params.rdy);
  }
  if (x < params.nx - FD_PAD) {
    lambda_hz[i] += DiffAdjoint<STENCIL_ORDER>::diff_x1_adj(
        alpha_x_acc, ones, shot_offset, (int)y, (int)x, params.nx, params.rdx);
  }
}

// Reverse of the scattered H update. Produces transpose fluxes into the
// scattered E adjoint field.
template <typename T, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void born_backward_prepare_h_core(
    GridParams<T> const &params, T const *cq_ptr, T const *lambda_hx,
    T const *lambda_hz, T *m_lambda_ey_x, T *m_lambda_ey_z, T *alpha_ey_x,
    T *alpha_ey_z, int64_t y, int64_t x, int64_t shot_idx) {

  if (y < 0 || x < 0 || y >= params.ny || x >= params.nx ||
      shot_idx >= params.n_shots) {
    return;
  }

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;
  int64_t const j = y * params.nx + x;
  int64_t const i = shot_idx * params.shot_numel + j;
  alpha_ey_x[i] = static_cast<T>(0);
  alpha_ey_z[i] = static_cast<T>(0);

  if (y < FD_PAD || x < FD_PAD || y >= params.ny - FD_PAD + 1 ||
      x >= params.nx - FD_PAD + 1) {
    return;
  }

  T const cq_val = params.cq_batched ? cq_ptr[i] : cq_ptr[j];
  int64_t const pml_y0h = params.pml_y0;
  int64_t const pml_y1h =
      params.pml_y1 > params.pml_y0 ? params.pml_y1 - 1 : params.pml_y0;
  int64_t const pml_x0h = params.pml_x0;
  int64_t const pml_x1h =
      params.pml_x1 > params.pml_x0 ? params.pml_x1 - 1 : params.pml_x0;

  if (y < params.ny - FD_PAD) {
    T const beta_z = -cq_val * lambda_hx[i];
    if (y >= pml_y0h && y < pml_y1h) {
      alpha_ey_z[i] = beta_z;
    } else {
      T const gamma_z = m_lambda_ey_z[i] + beta_z;
      alpha_ey_z[i] = beta_z / params.kyh[y] + params.ayh[y] * gamma_z;
      m_lambda_ey_z[i] = params.byh[y] * gamma_z;
    }
  }

  if (x < params.nx - FD_PAD) {
    T const beta_x = cq_val * lambda_hz[i];
    if (x >= pml_x0h && x < pml_x1h) {
      alpha_ey_x[i] = beta_x;
    } else {
      T const gamma_x = m_lambda_ey_x[i] + beta_x;
      alpha_ey_x[i] = beta_x / params.kxh[x] + params.axh[x] * gamma_x;
      m_lambda_ey_x[i] = params.bxh[x] * gamma_x;
    }
  }
}

// Apply the transposed scattered H update to the scattered E adjoint field.
template <typename T, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void born_backward_apply_h_to_e_core(
    GridParams<T> const &params, T const *alpha_ey_x, T const *alpha_ey_z,
    T *lambda_ey, int64_t y, int64_t x, int64_t shot_idx) {

  if (y < 0 || x < 0 || y >= params.ny || x >= params.nx ||
      shot_idx >= params.n_shots) {
    return;
  }

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;
  if (y < FD_PAD || x < FD_PAD || y >= params.ny - FD_PAD + 1 ||
      x >= params.nx - FD_PAD + 1) {
    return;
  }

  int64_t const i = shot_idx * params.shot_numel + y * params.nx + x;
  int64_t const shot_offset = shot_idx * params.shot_numel;
  GlobalFieldAccessor<T> alpha_x_acc(alpha_ey_x, params.nx);
  GlobalFieldAccessor<T> alpha_z_acc(alpha_ey_z, params.nx);
  ConstAccessor ones;

  lambda_ey[i] += DiffAdjoint<STENCIL_ORDER>::diff_xh1_adj(
      alpha_x_acc, ones, shot_offset, (int)y, (int)x, params.nx, params.rdx);
  lambda_ey[i] += DiffAdjoint<STENCIL_ORDER>::diff_yh1_adj(
      alpha_z_acc, ones, shot_offset, (int)y, (int)x, params.nx, params.rdy);
}

// Backward kernel: Update adjoint λ_H fields
template <typename T, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void
backward_kernel_lambda_h_core(GridParams<T> const &params, T const *cb_ptr,
                              T const *lambda_ey, T *lambda_hx, T *lambda_hz,
                              T *m_lambda_ey_x, T *m_lambda_ey_z, int64_t y,
                              int64_t x, int64_t shot_idx) {

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;

  if (y >= FD_PAD && x >= FD_PAD && y < params.ny - FD_PAD + 1 &&
      x < params.nx - FD_PAD + 1 && shot_idx < params.n_shots) {
    int64_t const pml_y0h = params.pml_y0;
    int64_t const pml_y1h =
        params.pml_y1 > params.pml_y0 ? params.pml_y1 - 1 : params.pml_y0;
    int64_t const pml_x0h = params.pml_x0;
    int64_t const pml_x1h =
        params.pml_x1 > params.pml_x0 ? params.pml_x1 - 1 : params.pml_x0;

    int64_t j = y * params.nx + x;
    int64_t i = shot_idx * params.shot_numel + j;

    T const cb_val = params.cb_batched ? cb_ptr[i] : cb_ptr[j];

    GlobalFieldAccessor<T> L_EY(lambda_ey, params.nx);

    if (y < params.ny - FD_PAD) {
      bool pml_y = y < pml_y0h || y >= pml_y1h;

      T d_lambda_ey_dz = DiffForward<STENCIL_ORDER>::diff_y1(
          L_EY, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);

      if (pml_y) {
        m_lambda_ey_z[i] =
            params.byh[y] * m_lambda_ey_z[i] + params.ayh[y] * d_lambda_ey_dz;
        d_lambda_ey_dz = d_lambda_ey_dz / params.kyh[y] + m_lambda_ey_z[i];
      }

      lambda_hx[i] -= cb_val * d_lambda_ey_dz;
    }

    if (x < params.nx - FD_PAD) {
      bool pml_x = x < pml_x0h || x >= pml_x1h;

      T d_lambda_ey_dx = DiffForward<STENCIL_ORDER>::diff_x1(
          L_EY, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);

      if (pml_x) {
        m_lambda_ey_x[i] =
            params.bxh[x] * m_lambda_ey_x[i] + params.axh[x] * d_lambda_ey_dx;
        d_lambda_ey_dx = d_lambda_ey_dx / params.kxh[x] + m_lambda_ey_x[i];
      }

      lambda_hz[i] += cb_val * d_lambda_ey_dx;
    }
  }
}

// Backward kernel: Update adjoint λ_Ey field with per-shot gradient
// accumulation
template <typename T, typename StoreT, int STENCIL_ORDER>
static TIDE_HOST_DEVICE void backward_kernel_lambda_e_with_grad_core(
    GridParams<T> const &params, T const *ca_ptr, T const *cq_ptr,
    T const *lambda_hx, T const *lambda_hz, T *lambda_ey, T *m_lambda_hx_z,
    T *m_lambda_hz_x, StoreT const *ey_store, StoreT const *curl_h_store,
    T *grad_ca_shot, T *grad_cb_shot, bool ca_requires_grad,
    bool cb_requires_grad, int64_t step_ratio_val, int64_t y, int64_t x,
    int64_t shot_idx) {

  int const FD_PAD = tide::StencilTraits<STENCIL_ORDER>::FD_PAD;

  // diff_yh1/diff_xh1 read y+1/x+1, so the adjoint E update needs the
  // positive-side halo and cannot touch the last interior row/column.
  if (y >= FD_PAD && x >= FD_PAD && y < params.ny - FD_PAD &&
      x < params.nx - FD_PAD && shot_idx < params.n_shots) {
    int64_t j = y * params.nx + x;
    int64_t i = shot_idx * params.shot_numel + j;

    T const ca_val = params.ca_batched ? ca_ptr[i] : ca_ptr[j];
    T const cq_val = params.cq_batched ? cq_ptr[i] : cq_ptr[j];

    bool pml_y = y < params.pml_y0 || y >= params.pml_y1;
    bool pml_x = x < params.pml_x0 || x >= params.pml_x1;

    GlobalFieldAccessor<T> L_HZ(lambda_hz, params.nx);
    GlobalFieldAccessor<T> L_HX(lambda_hx, params.nx);

    T d_lambda_hz_dx = DiffForward<STENCIL_ORDER>::diff_xh1(
        L_HZ, shot_idx * params.shot_numel, y, x, params.nx, params.rdx);
    T d_lambda_hx_dz = DiffForward<STENCIL_ORDER>::diff_yh1(
        L_HX, shot_idx * params.shot_numel, y, x, params.nx, params.rdy);

    if (pml_x) {
      m_lambda_hz_x[i] =
          params.bx[x] * m_lambda_hz_x[i] + params.ax[x] * d_lambda_hz_dx;
      d_lambda_hz_dx = d_lambda_hz_dx / params.kx[x] + m_lambda_hz_x[i];
    }

    if (pml_y) {
      m_lambda_hx_z[i] =
          params.by[y] * m_lambda_hx_z[i] + params.ay[y] * d_lambda_hx_dz;
      d_lambda_hx_dz = d_lambda_hx_dz / params.ky[y] + m_lambda_hx_z[i];
    }

    T curl_lambda_h = d_lambda_hz_dx - d_lambda_hx_dz;
    T lambda_ey_curr = lambda_ey[i]; // λ_Ey^{n+1}

    lambda_ey[i] = ca_val * lambda_ey_curr + cq_val * curl_lambda_h;

    if (ca_requires_grad && ey_store != nullptr && grad_ca_shot != nullptr) {
      T ey_n = decode_snapshot<StoreT, T>(ey_store[i]);
      // Accumulate grad_ca += λ_Ey^{n+1} * E_y^n
      grad_ca_shot[i] +=
          lambda_ey_curr * ey_n * static_cast<T>(step_ratio_val);
    }

    if (cb_requires_grad && curl_h_store != nullptr &&
        grad_cb_shot != nullptr) {
      T curl_h_n = decode_snapshot<StoreT, T>(curl_h_store[i]);
      // Accumulate grad_cb += λ_Ey^{n+1} * curl_H^n
      grad_cb_shot[i] +=
          lambda_ey_curr * curl_h_n * static_cast<T>(step_ratio_val);
    }
  }
}

} // namespace tide

#endif // MAXWELL_TM_CORE_CUH
