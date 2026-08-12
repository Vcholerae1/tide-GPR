---
title: "Build from Source"
description: "Build editable Python installations and native TIDE backends reproducibly."
---

This page documents reliable local build workflows for Python + native backend.

## Requirements
- Python >= 3.12
- CMake >= 3.28 (optional)
- CUDA Toolkit (optional)

Recommended tools:
- uv for environment and packaging workflow
- Ninja or Make for faster CMake builds

## Build Steps
```bash
git clone https://github.com/vcholerae1/tide.git
cd tide
uv build
```

This builds the Python package and triggers native extension packaging.

### Rebuild Native Backend Only (`csrc`)

```bash
cmake -S src/tide/csrc -B build/csrc -DCMAKE_BUILD_TYPE=Release
cmake --build build/csrc -j
```

If needed, clean and rebuild:

```bash
rm -rf build/csrc
cmake -S src/tide/csrc -B build/csrc -DCMAKE_BUILD_TYPE=Release
cmake --build build/csrc -j
```

Notes:
- Do not configure CMake inside `src/tide/csrc` directly.
- Backend CMake now rejects in-source builds by design.

## Verify Build

Use Python to verify native backend loading:

```python
from tide import backend_utils

print(backend_utils.is_backend_available())
print(backend_utils.get_library_path())
```

## Notes

Common environment variables and flags:
- CMAKE_BUILD_TYPE=Release for optimized kernels
- CMAKE_CUDA_ARCHITECTURES to pin target GPUs
- CC/CXX to select host compilers

## Troubleshooting

1. Shared library not found:
	- rebuild backend and confirm output path under src/tide
2. CUDA symbols missing:
	- verify PyTorch CUDA build and CUDA toolkit compatibility
3. Compiler mismatch:
	- use consistent host compiler versions for C++ and CUDA toolchains

## Editable development environment

```bash
uv sync --group dev
uv run python -c "import tide; print(tide.__version__)"
uv run pytest -q tests/test_public_api.py
```

The package uses `scikit-build-core` and `cibuildwheel`. A wheel build packages
Python sources and the native library for the target platform. An editable
environment is useful for Python changes, but rebuild the native target after
changing C, C++, CUDA, CMake, or ABI declarations.

## CPU-only native build

When CUDA is not required, configure a clean build directory with the compiler
toolchain visible to CMake. Inspect the configure summary to confirm that CUDA
was disabled intentionally rather than missed unexpectedly.

After building, test both library loading and a small forced-native CPU
propagation. `is_backend_available()` alone proves only that a shared library
loaded.

## CUDA architecture selection

Set `CMAKE_CUDA_ARCHITECTURES` to the deployment GPU architectures when
building CUDA kernels:

```bash
cmake -S src/tide/csrc -B build/csrc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/csrc -j
```

A library can load successfully and still lack code for the current GPU.
Record the selected architecture list with released wheels and benchmark
artifacts.

## Focused verification

Use a staged build check:

```bash
uv run pytest -q tests/test_public_api.py
uv run pytest -q tests/test_forward_physics.py
uv run pytest -q tests/test_backend_parity_matrix.py
```

On a CUDA machine, add the CUDA-marked parity cases. On CPU-only hosts, skipped
CUDA tests are expected but do not verify CUDA packaging.

## Diagnosing loader errors

When the library path exists but loading fails:

1. Inspect the original loader exception.
2. Check unresolved system dependencies for the shared library.
3. Confirm Python architecture matches the compiled library.
4. Confirm compiler runtime and C++ ABI compatibility.
5. Confirm CUDA runtime dependencies are available for a CUDA build.
6. Rebuild from a clean out-of-source directory after toolchain changes.

Do not copy a shared library from another environment unless its platform,
Python package layout, compiler ABI, and CUDA targets are known to match.
