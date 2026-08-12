---
title: "C/CUDA Sources (`tide.csrc`)"
description: "Build layout, ABI conventions, and symbol families for native C and CUDA kernels."
---

Native kernels and CMake build live under `src/tide/csrc`.

## Build Convention

`csrc` now enforces out-of-source CMake builds to keep source directories clean.

```bash
cmake -S src/tide/csrc -B build/csrc -DCMAKE_BUILD_TYPE=Release
cmake --build build/csrc -j
```

The shared library is emitted to `src/tide/` as `libtide_C.{so|dylib|dll}`.

## Directory Layout

- `src/tide/csrc/CMakeLists.txt`
  - backend build entrypoint
- `src/tide/csrc/tm2d/`
  - 2D TM CPU/CUDA kernels (`maxwell.cpp`, `maxwell.cu`)
  - TM core kernels (`maxwell_tm_core.cuh`)
  - TM instantiation units (`maxwell_tm_inst.cpp`, `maxwell_tm_cuda_inst.cu`)
  - instantiation manifests (`maxwell_tm_cpu_instantiations.inc`, `maxwell_tm_cuda_instantiations.inc`)
- `src/tide/csrc/em3d/`
  - 3D CPU/CUDA kernels (`maxwell_3d.cpp`, `maxwell_3d.cu`)
- `src/tide/csrc/common/`
  - shared CPU/GPU utility headers
  - finite-difference stencil headers (`staggered_grid*.h`)
- `src/tide/csrc/storage/`
  - snapshot storage utilities (`storage_utils.c`, `storage_utils.cu`, `storage_utils.h`)

## Exported Symbol Families

- 2D TM:
  - `maxwell_tm_<stencil>_<dtype>_{forward,forward_with_storage,backward}_{cpu|cuda}`
- 3D:
  - `maxwell_3d_<stencil>_<dtype>_{forward,forward_with_storage,backward}_{cpu|cuda}`

Symbol lookup and ctypes signatures are defined in `src/tide/backend_utils.py`.

## Supported specialization axes

Native symbols are specialized by:

- Propagator family: TM2D or EM3D.
- Finite-difference order: 2, 4, 6, or 8.
- Scalar dtype: float32 or float64.
- Device implementation: CPU or CUDA.
- Pass: forward, stored forward, tangent, adjoint, or incremental adjoint where
  supported.

The Python dispatch layer resolves a symbol only after the central capability
matrix accepts the requested operation.

## Adding a kernel specialization

1. Implement or instantiate the kernel without changing the public operator
   contract.
2. Export a symbol that follows the existing family naming convention.
3. Add its ordered ABI declaration to `backend_utils`.
4. Add a capability row only when the complete public path is reachable.
5. Compare receiver data and derivatives with the reference implementation.
6. Cover missing-symbol and unsupported-combination behavior.

Generated build files and CMake caches belong under `build/`, not under
`src/tide/csrc`.

## ABI invariants

Python and native declarations must agree on argument order, pointer
nullability, scalar width, and ownership. Tensors passed to native code must be
contiguous and remain alive for the duration of the call. Native code must not
retain borrowed tensor pointers after returning unless an explicit owner
controls that lifetime.

Use the structured operator smoke tests after any ABI change. A successful
shared-library build does not prove symbol names or `ctypes` signatures are
correct.
