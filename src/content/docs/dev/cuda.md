---
title: "CUDA Notes"
description: "Distinguish driver, PyTorch, and native-kernel CUDA support and diagnose failures."
---

This page summarizes CUDA-specific setup and failure modes.

## Compatibility Checklist

1. NVIDIA driver installed and visible in nvidia-smi.
2. PyTorch build has CUDA support.
3. CUDA toolkit available for native build workflows.
4. CMake can detect nvcc during csrc build.

## Typical Setup
- Ensure CUDA Toolkit is installed and on PATH.
- Verify `nvcc --version`.
- Confirm your PyTorch build has CUDA enabled.

Useful Python checks:

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

## Troubleshooting

1. torch.cuda.is_available() is False:
	- install matching CUDA-enabled torch wheel
	- verify driver installation

2. Native backend loads but CUDA symbols are unavailable:
	- rebuild csrc with a valid CUDA toolkit and visible nvcc
	- check CMake output for CUDA detection messages

3. Runtime illegal memory access or launch failures:
	- validate tensor shapes and bounds for source/receiver indices
	- reduce workload size and reproduce with one shot for isolation

4. Performance lower than expected:
	- test storage_mode=device first
	- profile with realistic n_shots and nt
	- verify kernels are not falling back to Python backend

## Distinguish three CUDA layers

CUDA availability has three independent parts:

1. The NVIDIA driver can expose a GPU.
2. PyTorch can allocate CUDA tensors.
3. TIDE's native library contains compatible CUDA symbols.

`torch.cuda.is_available()` checks the second layer, not the third. Use
`backend_utils.cuda_build_arches()` and a forced-native smoke test to verify the
TIDE layer.

```python
import torch
import tide
from tide import backend_utils

print(torch.cuda.get_device_name())
print(torch.version.cuda)
print(backend_utils.get_library_path())
print(backend_utils.cuda_build_arches())
```

## Forced-native smoke test

Set `backend=BackendPreference.NATIVE` and
`fallback=FallbackPolicy.ERROR` on a small operator. This prevents a missing
CUDA specialization from being hidden by reference execution. Check receiver
shape, finite values, and one supported gradient.

## Asynchronous errors

CUDA kernels launch asynchronously. An illegal access can be reported by a
later operation, making the stack trace misleading. Reproduce with one shot,
a small grid, and synchronized execution. Confirm source and receiver bounds,
contiguity, dtype, and stencil specialization before investigating performance.

## Memory accounting

Device memory includes more than model tensors:

- Live electric, magnetic, and CPML fields.
- Padded material coefficients.
- Source and receiver buffers.
- Forward snapshots for reverse propagation.
- Autograd and optimizer state.
- CUDA context and allocator reserve.

Measure peak allocated and reserved memory around the complete operation. Leave
headroom rather than setting automatic snapshot limits to total VRAM.

## Performance verification

Warm up the same forward or derivative path before timing. Synchronize before
and after the measured region. Report model shape, padded shape, shots, internal
step ratio, stencil, dtype, storage mode, and whether the measurement includes
backward propagation.

A faster CUDA result is acceptable only after comparison with the reference
path at an explicit numerical tolerance.
