/*
 * Maxwell 3D CPU backend entrypoint.
 *
 * This translation unit instantiates all supported (stencil, dtype)
 * combinations from maxwell_3d_inst.cpp.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common_cpu.h"
#include "storage_utils.h"

#define CAT_I(name, accuracy, dtype, device) \
  maxwell_3d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) \
  CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, cpu)

#ifdef __cplusplus
#define TIDE_EXTERN_C extern "C"
#else
#define TIDE_EXTERN_C
#endif

#ifdef _WIN32
#define TIDE_EXPORT __declspec(dllexport)
#else
#define TIDE_EXPORT
#endif

#include "maxwell_3d_cpu_instantiations.inc"

#undef FUNC
#undef CAT
#undef CAT_I
#undef TIDE_EXPORT
#undef TIDE_EXTERN_C
