---
title: "Module: tide.backend_utils"
description: "Native-library discovery, ABI lookup, and tensor interop utilities."
---

Internal backend interop and C/CUDA bindings.
This module is mostly internal and used by tide.maxwell to resolve native symbols.

## Functions
- is_backend_available
- get_dll
- get_library_path
- cuda_build_arches
- get_backend_function
- backend_signature
- tensor_to_ptr
- ensure_contiguous

## Internal Signature Templates

The module defines declarative ctypes signature templates for:
- maxwell_tm forward, forward_with_storage, backward
- maxwell_3d forward, forward_with_storage, backward

These templates are validated once, exposed by `backend_signature`, and assigned
lazily by `get_backend_function`.

## Typical Internal Flow

1. Probe and load native shared library.
2. Validate requested propagator/pass/accuracy/dtype/device combination.
3. Build native symbol name.
4. Resolve symbol from shared library.
5. Bind argtypes/restype and return callable C function pointer.

## Notes

- tensor_to_ptr handles wrapped tensors used by torch.func transforms.
- ensure_contiguous returns contiguous storage when required by native kernels.

## User-Facing Use

Most users only need:

- `is_backend_available()`
- `get_library_path()`

Use these functions during installation checks and before investigating backend-specific performance behavior.

## Availability checks

```python
from tide import backend_utils

if backend_utils.is_backend_available():
    print(backend_utils.get_library_path())
    print(backend_utils.cuda_build_arches())
```

`is_backend_available` is the non-raising probe. `get_dll` and
`get_library_path` raise when no native library can be loaded. A present shared
library still may not contain the symbol, CUDA architecture, dtype, stencil, or
derivative pass required by a particular request.

## Function resolution

`get_backend_function` builds a symbol request from propagator, pass, stencil
accuracy, dtype, device, and optional variant. It validates the combination,
resolves the C symbol, installs the declared `ctypes` signature, and returns a
callable pointer.

`backend_signature` returns the ordered logical argument names used to verify
ABI declarations. It is intended for backend tests and diagnostics, not for
application dispatch.

## Tensor interop

`ensure_contiguous` returns the original tensor when its layout already meets
the native contract and creates a contiguous copy otherwise.
`tensor_to_ptr(None)` produces a null pointer representation. For tensors, the
caller must keep the tensor alive and must not use the pointer outside the
native call's lifetime.

Application code should let structured Maxwell operators perform these steps.
