/*
 * TM2D CUDA mixed-precision implementation for compute_precision='fp16_scaled'.
 *
 * Public coefficients and PML profiles remain float32, while field-state and
 * snapshot payloads are stored in fp16. All stencil, CPML, and gradient math
 * is accumulated in float32.
 */

#include <cuda_fp16.h>

#define TIDE_DEVICE cuda

#ifndef NUM_BUFFERS
#define NUM_BUFFERS 3
#endif

#include "common_gpu.h"
#include "storage_utils.h"

#include "maxwell_tm_cuda_fp16s_instantiations.inc"
