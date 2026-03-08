#ifndef COMMON_GPU_H
#define COMMON_GPU_H

#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include <cuda_runtime.h>

namespace tide {

inline cudaError_t cuda_check(cudaError_t code, const char *file, int line,
                              bool abort = true) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s %s %d\n",
                 cudaGetErrorString(code), file, line);
    if (abort) {
      std::exit(code);
    }
  }
  return code;
}

inline void cuda_check_or_abort(cudaError_t code, const char *file, int line) {
  (void)cuda_check(code, file, line, true);
}

inline void cuda_check_last_error(const char *file, int line,
                                  bool abort = true) {
  (void)cuda_check(cudaGetLastError(), file, line, abort);
}

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  (void)cuda_check(code, file, line, abort);
}

template <typename T> __device__ __forceinline__ T atomic_add(T *address, T val);

template <>
__device__ __forceinline__ float atomic_add<float>(float *address, float val) {
  return atomicAdd(address, val);
}

template <>
__device__ __forceinline__ double atomic_add<double>(double *address,
                                                     double val) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
  unsigned long long int *address_as_ull =
      reinterpret_cast<unsigned long long int *>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#else
  return atomicAdd(address, val);
#endif
}

} // namespace tide

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  tide::gpuAssert(code, file, line, abort);
}

#endif
