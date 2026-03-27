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

  if (y >= FD_PAD && x >= FD_PAD && y < params.ny - FD_PAD + 1 &&
      x < params.nx - FD_PAD + 1 && shot_idx < params.n_shots) {
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
