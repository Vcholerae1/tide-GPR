#ifndef COMMON_CPU_H
#define COMMON_CPU_H

#include <stdint.h>
#include <stdbool.h>

#ifndef TIDE_DTYPE
#define TIDE_DTYPE float
#endif

#ifndef TIDE_STENCIL
#define TIDE_STENCIL 4
#endif

#if defined(_OPENMP)
#define TIDE_OMP_INDEX int64_t
#define TIDE_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2)")
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp parallel for collapse(4)")
#if defined(_MSC_VER)
#define TIDE_OMP_SIMD
#define TIDE_OMP_SIMD_COLLAPSE2
#else
#define TIDE_OMP_SIMD _Pragma("omp simd")
#define TIDE_OMP_SIMD_COLLAPSE2 _Pragma("omp simd collapse(2)")
#endif
#else
#define TIDE_OMP_INDEX int64_t
#define TIDE_OMP_PARALLEL_FOR
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4
#define TIDE_OMP_SIMD
#define TIDE_OMP_SIMD_COLLAPSE2
#endif

#endif
