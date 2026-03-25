# Build from Source

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
