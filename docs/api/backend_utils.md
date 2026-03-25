# Module: tide.backend_utils

Internal backend interop and C/CUDA bindings.
This module is mostly internal and used by tide.maxwell to resolve native symbols.

## Functions
- is_backend_available
- get_dll
- get_library_path
- cuda_build_arches
- get_backend_function
- tensor_to_ptr
- ensure_contiguous

## Internal Signature Templates

The module defines declarative ctypes signature templates for:
- maxwell_tm forward, forward_with_storage, backward
- maxwell_3d forward, forward_with_storage, backward

These templates are cached and assigned lazily by get_backend_function.

## Typical Internal Flow

1. Probe and load native shared library.
2. Validate requested propagator/pass/accuracy/dtype/device combination.
3. Build native symbol name.
4. Resolve symbol from shared library.
5. Bind argtypes/restype and return callable C function pointer.

## Notes

- tensor_to_ptr handles wrapped tensors used by torch.func transforms.
- ensure_contiguous returns contiguous storage when required by native kernels.
