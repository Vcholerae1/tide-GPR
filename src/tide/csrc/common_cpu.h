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
#ifndef TIDE_OMP_MIN_PARALLEL_SHOTS
#define TIDE_OMP_MIN_PARALLEL_SHOTS 1
#endif
#define TIDE_OMP_PRAGMA(x) _Pragma(#x)
#define TIDE_OMP_PARALLEL_FOR_IF(cond) TIDE_OMP_PRAGMA(omp parallel for schedule(static) if(cond))
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2_IF(cond) \
    TIDE_OMP_PRAGMA(omp parallel for collapse(2) schedule(static) if(cond))
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(cond) \
    TIDE_OMP_PRAGMA(omp parallel for collapse(3) schedule(static) if(cond))
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4_IF(cond) \
    TIDE_OMP_PRAGMA(omp parallel for collapse(4) schedule(static) if(cond))
#define TIDE_OMP_PARALLEL_FOR TIDE_OMP_PARALLEL_FOR_IF(1)
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2 TIDE_OMP_PARALLEL_FOR_COLLAPSE2_IF(1)
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE3 TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(1)
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4 TIDE_OMP_PARALLEL_FOR_COLLAPSE4_IF(1)
#define TIDE_OMP_SIMD _Pragma("omp simd")
#define TIDE_OMP_SIMD_COLLAPSE2 _Pragma("omp simd collapse(2)")
#else
#define TIDE_OMP_INDEX int64_t
#define TIDE_OMP_MIN_PARALLEL_SHOTS 8
#define TIDE_OMP_PARALLEL_FOR_IF(cond)
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2_IF(cond)
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE3_IF(cond)
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4_IF(cond)
#define TIDE_OMP_PARALLEL_FOR
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE2
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE3
#define TIDE_OMP_PARALLEL_FOR_COLLAPSE4
#define TIDE_OMP_SIMD
#define TIDE_OMP_SIMD_COLLAPSE2
#endif

#endif
