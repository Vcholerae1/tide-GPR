/*
 * Maxwell 3D CUDA backend entrypoint.
 *
 * This translation unit instantiates all supported (stencil, dtype)
 * combinations from maxwell_3d_cuda_inst.cu.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

#include <cuda_runtime.h>

#include "common_gpu.h"
#include "storage_utils.h"

#ifndef TIDE_DEVICE
#define TIDE_DEVICE cuda
#endif

#define CAT_I(name, accuracy, dtype, device) \
  maxwell_3d_##accuracy##_##dtype##_##name##_##device
#define CAT(name, accuracy, dtype, device) \
  CAT_I(name, accuracy, dtype, device)
#define FUNC(name) CAT(name, TIDE_STENCIL, TIDE_DTYPE, TIDE_DEVICE)

#include "maxwell_3d_cuda_instantiations.inc"

#undef FUNC
#undef CAT
#undef CAT_I
#undef TIDE_DEVICE
