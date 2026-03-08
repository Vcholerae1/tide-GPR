#undef TIDE_STENCIL
#undef TIDE_DTYPE
#define TIDE_STENCIL 2
#define TIDE_DTYPE float
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE float
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE float
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE float
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 2
#define TIDE_DTYPE double
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 4
#define TIDE_DTYPE double
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 6
#define TIDE_DTYPE double
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE

#define TIDE_STENCIL 8
#define TIDE_DTYPE double
#include "maxwell_tm_inst.cpp"
#undef TIDE_STENCIL
#undef TIDE_DTYPE
