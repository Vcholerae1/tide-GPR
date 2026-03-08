#define TIDE_STENCIL 2
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 2
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE double
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#if TIDE_ENABLE_HALF_KERNELS
#define TIDE_STENCIL 2
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE half
#include "maxwell_tm_cuda_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE
#endif
