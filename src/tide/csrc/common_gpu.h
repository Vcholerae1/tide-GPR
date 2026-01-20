#ifndef COMMON_GPU_H
#define COMMON_GPU_H

#include <cuda_runtime.h>

// Define TIDE_DTYPE_FLOAT based on TIDE_DTYPE (for vectorized load optimization)
// Note: TIDE_DTYPE_FLOAT is now defined in CMakelists.txt for better compatibility
#ifndef TIDE_DTYPE_FLOAT
#ifdef TIDE_DTYPE
#if TIDE_DTYPE == float
#define TIDE_DTYPE_FLOAT 1
#elif TIDE_DTYPE == double
#define TIDE_DTYPE_FLOAT 0
#endif
#endif
#endif

// Macro to check for CUDA kernel errors
#define CHECK_KERNEL_ERROR                                              \
  {                                                                     \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA kernel error: %s at %s:%d\n",              \
              cudaGetErrorString(err), __FILE__, __LINE__);            \
      exit(EXIT_FAILURE);                                              \
    }                                                                   \
  }

// GPU error checking helper function
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Custom atomicAdd for double precision (needed for older compute capabilities)
// This is a software emulation using atomicCAS
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

#endif 
