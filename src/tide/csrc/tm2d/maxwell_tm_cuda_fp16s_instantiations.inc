#define TIDE_STENCIL 2
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_fp16s_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_fp16s_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_fp16s_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE float
#include "maxwell_tm_cuda_fp16s_inst.cu"
#undef TIDE_STENCIL
#undef TIDE_DTYPE
