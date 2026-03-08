# Build from Source

TODO: Provide detailed build instructions and troubleshooting tips.

## Requirements
- Python >= 3.12
- CMake >= 3.28 (optional)
- CUDA Toolkit (optional)

## Build Steps
```bash
git clone https://github.com/vcholerae1/tide.git
cd tide
uv build
```

### Rebuild Native Backend Only (`csrc`)

```bash
cmake -S src/tide/csrc -B build/csrc -DCMAKE_BUILD_TYPE=Release
cmake --build build/csrc -j
```

Notes:
- Do not configure CMake inside `src/tide/csrc` directly.
- Backend CMake now rejects in-source builds by design.

## Notes
TODO: Document build options and environment variables.
