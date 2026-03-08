# C/CUDA Sources (`tide.csrc`)

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
