# CUDA Notes

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
