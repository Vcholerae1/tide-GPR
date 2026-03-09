import ctypes
import itertools
import pathlib
import platform
from ctypes import c_bool, c_double, c_float, c_int64, c_void_p
from typing import Any

import torch

type CFunctionPointer = Any

# Platform-specific shared library extension
SO_EXT = {"Linux": "so", "Darwin": "dylib", "Windows": "dll"}.get(platform.system())
if SO_EXT is None:
    raise RuntimeError("Unsupported OS or platform type")

# Supported configurations for backend function variants.
_SUPPORTED_ACCURACIES = (2, 4, 6, 8)
_SUPPORTED_DEVICES = ("cpu", "cuda")
_SUPPORTED_BACKEND_DTYPES = ("float", "double")

# Mapping from torch dtypes to backend dtype strings and from backend dtype strings to C types.
_TORCH_DTYPE_TO_BACKEND_DTYPE: dict[torch.dtype, str] = {
    torch.float32: "float",
    torch.float64: "double",
}

_BACKEND_DTYPE_TO_CTYPE: dict[str, type] = {
    "float": c_float,
    "double": c_double,
}


def _candidate_lib_paths() -> list[pathlib.Path]:
    """Return likely shared-library locations for standard uv/pip installs."""
    lib_name = f"libtide_C.{SO_EXT}"
    module_dir = pathlib.Path(__file__).resolve().parent
    candidates = [
        module_dir / lib_name,  # Most common: package directory
        module_dir / "tide" / lib_name,  # Nested package layout
        module_dir.parent / "tide.libs" / lib_name,  # Wheel external libs
    ]

    # Keep discovery order while removing duplicates.
    return list(dict.fromkeys(candidates))


_dll: ctypes.CDLL | None = None
_lib_path: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent / f"libtide_C.{SO_EXT}"
)
_backend_probed = False

USE_OPENMP = False


def _load_backend_once() -> None:
    global _dll, _lib_path, _backend_probed, USE_OPENMP
    if _backend_probed:
        return
    _backend_probed = True

    for candidate in _candidate_lib_paths():
        if not candidate.exists():
            continue
        try:
            _dll = ctypes.CDLL(str(candidate))
            _lib_path = candidate
            break
        except OSError:
            continue

    USE_OPENMP = _dll is not None and hasattr(_dll, "omp_get_num_threads")


def is_backend_available() -> bool:
    """Check if the C/CUDA backend is available."""
    _load_backend_once()
    return _dll is not None


def get_dll() -> ctypes.CDLL:
    """Get the loaded DLL, raising an error if not available."""
    _load_backend_once()
    if _dll is None:
        raise RuntimeError(
            f"C/CUDA backend not available. Please compile the library first. "
            f"Expected library at: {_lib_path}"
        )
    return _dll


def get_library_path() -> pathlib.Path:
    """Return the resolved shared library path."""
    _load_backend_once()
    return _lib_path


def cuda_build_arches() -> str | None:
    """Return the CUDA arch list the backend was compiled for, if available."""
    _load_backend_once()
    if _dll is None or not hasattr(_dll, "tide_cuda_arches"):
        return None
    func = _dll.tide_cuda_arches
    func.restype = ctypes.c_char_p
    value = func()
    if not value:
        return None
    return value.decode("utf-8", errors="replace")


# Each argtype spec is a declarative list of (ctype, count, comment) triples.
# FLOAT_TYPE is a placeholder that _get_argtypes() substitutes with the actual
# float ctypes type (c_float or c_double) at resolution time.
FLOAT_TYPE: type = c_float

# Short aliases to keep specs readable.
_P = c_void_p
_F = FLOAT_TYPE  # float placeholder (substituted by _get_argtypes)
_I = c_int64
_B = c_bool

# (ctype, count, comment) — comment documents the argument names in order.
type _Spec = list[tuple[Any, int, str]]


def _build(spec: _Spec) -> list[Any]:
    """Flatten a declarative spec into a flat ctypes argtypes list."""
    return [ctype for ctype, n, _ in spec for _ in range(n)]


# ---------------------------------------------------------------------------
# TM (2-D) propagators
# ---------------------------------------------------------------------------

_TM_COMMON_TAIL: _Spec = [
    (_F, 2, "rdy, rdx"),
    (_F, 1, "dt"),
    (_I, 1, "nt"),
    (_I, 1, "n_shots"),
    (_I, 2, "ny, nx"),
    (_I, 2, "n_sources_per_shot, n_receivers_per_shot"),
    (_I, 1, "step_ratio"),
]

_TM_BATCHED_FLAGS: _Spec = [
    (_B, 3, "ca_batched, cb_batched, cq_batched"),
    (_I, 1, "start_t"),
    (_I, 4, "pml_y0, pml_x0, pml_y1, pml_x1"),
    (_I, 1, "n_threads"),
    (_I, 1, "device"),
]

_TM_STORAGE_TAIL: _Spec = [
    (_I, 2, "storage_mode, shot_bytes_uncomp"),
    (_B, 2, "ca_requires_grad, cb_requires_grad"),
]

_TM_PML_PROFILES: _Spec = [
    (_P, 8, "ay, by, ayh, byh, ax, bx, axh, bxh"),
    (_P, 4, "ky, kyh, kx, kxh"),
    (_P, 2, "sources_i, receivers_i"),
]

_TM_FORWARD_SPEC: _Spec = [
    (_P, 3, "ca, cb, cq"),
    (_P, 1, "f"),
    (_P, 3, "ey, hx, hz"),
    (_P, 4, "m_ey_x, m_ey_z, m_hx_z, m_hz_x"),
    (_P, 1, "r"),
    *_TM_PML_PROFILES,
    *_TM_COMMON_TAIL,
    *_TM_BATCHED_FLAGS,
]

_TM_FORWARD_WITH_STORAGE_SPEC: _Spec = [
    (_P, 3, "ca, cb, cq"),
    (_P, 1, "f"),
    (_P, 3, "ey, hx, hz"),
    (_P, 4, "m_ey_x, m_ey_z, m_hx_z, m_hz_x"),
    (_P, 1, "r"),
    (
        _P,
        6,
        "ey_store_1, ey_store_3, ey_filenames, curl_store_1, curl_store_3, curl_filenames",
    ),
    *_TM_PML_PROFILES,
    *_TM_COMMON_TAIL,
    *_TM_STORAGE_TAIL,
    *_TM_BATCHED_FLAGS,
]

_TM_BACKWARD_SPEC: _Spec = [
    (_P, 3, "ca, cb, cq"),
    (_P, 1, "grad_r"),
    (_P, 3, "lambda_ey, lambda_hx, lambda_hz"),
    (_P, 4, "m_lambda_ey_x, m_lambda_ey_z, m_lambda_hx_z, m_lambda_hz_x"),
    (
        _P,
        6,
        "ey_store_1, ey_store_3, ey_filenames, curl_store_1, curl_store_3, curl_filenames",
    ),
    (_P, 1, "grad_f"),
    (_P, 2, "grad_ca, grad_cb"),
    (_P, 2, "grad_ca_shot, grad_cb_shot"),
    *_TM_PML_PROFILES,
    *_TM_COMMON_TAIL,
    *_TM_STORAGE_TAIL,
    *_TM_BATCHED_FLAGS,
]

# ---------------------------------------------------------------------------
# 3-D propagators
# ---------------------------------------------------------------------------

_3D_COMMON_TAIL: _Spec = [
    (_F, 3, "rdz, rdy, rdx"),
    (_F, 1, "dt"),
    (_I, 1, "nt"),
    (_I, 1, "n_shots"),
    (_I, 3, "nz, ny, nx"),
    (_I, 2, "n_sources_per_shot, n_receivers_per_shot"),
    (_I, 1, "step_ratio"),
]

_3D_BATCHED_FLAGS: _Spec = [
    (_B, 3, "ca_batched, cb_batched, cq_batched"),
    (_I, 1, "start_t"),
    (_I, 6, "pml_z0, pml_y0, pml_x0, pml_z1, pml_y1, pml_x1"),
    (_I, 2, "source_component, receiver_component"),
    (_I, 1, "n_threads"),
    (_I, 1, "device"),
]

_3D_STORAGE_TAIL: _Spec = [
    (_I, 2, "storage_mode, shot_bytes_uncomp"),
    (_B, 2, "ca_requires_grad, cb_requires_grad"),
]

_3D_PML_PROFILES: _Spec = [
    (_P, 12, "az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh"),
    (_P, 6, "kz, kzh, ky, kyh, kx, kxh"),
    (_P, 2, "sources_i, receivers_i"),
]

_3D_FIELDS: _Spec = [
    (_P, 6, "ex, ey, ez, hx, hy, hz"),
    (
        _P,
        12,
        "m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y, "
        "m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x",
    ),
]

_3D_FORWARD_SPEC: _Spec = [
    (_P, 3, "ca, cb, cq"),
    (_P, 1, "f"),
    *_3D_FIELDS,
    (_P, 1, "r"),
    *_3D_PML_PROFILES,
    *_3D_COMMON_TAIL,
    *_3D_BATCHED_FLAGS,
]

_3D_FORWARD_WITH_STORAGE_SPEC: _Spec = [
    (_P, 3, "ca, cb, cq"),
    (_P, 1, "f"),
    *_3D_FIELDS,
    (_P, 1, "r"),
    (
        _P,
        18,
        "ex_s1,ex_s3,ex_fn, ey_s1,ey_s3,ey_fn, ez_s1,ez_s3,ez_fn, "
        "cx_s1,cx_s3,cx_fn, cy_s1,cy_s3,cy_fn, cz_s1,cz_s3,cz_fn",
    ),
    *_3D_PML_PROFILES,
    *_3D_COMMON_TAIL,
    *_3D_STORAGE_TAIL,
    *_3D_BATCHED_FLAGS,
]

_3D_ADJ_FIELDS: _Spec = [
    (_P, 6, "lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz"),
    (
        _P,
        12,
        "m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x, m_lambda_ex_z, "
        "m_lambda_ex_y, m_lambda_ey_x, m_lambda_hz_y, m_lambda_hy_z, "
        "m_lambda_hx_z, m_lambda_hz_x, m_lambda_hy_x, m_lambda_hx_y",
    ),
]

_3D_BACKWARD_SPEC: _Spec = [
    (_P, 3, "ca, cb, cq"),
    (_P, 1, "grad_r"),
    *_3D_ADJ_FIELDS,
    (
        _P,
        18,
        "ex_s1,ex_s3,ex_fn, ey_s1,ey_s3,ey_fn, ez_s1,ez_s3,ez_fn, "
        "cx_s1,cx_s3,cx_fn, cy_s1,cy_s3,cy_fn, cz_s1,cz_s3,cz_fn",
    ),
    (_P, 1, "grad_f"),
    (_P, 4, "grad_ca, grad_cb, grad_eps, grad_sigma"),
    (_P, 2, "grad_ca_shot, grad_cb_shot"),
    *_3D_PML_PROFILES,
    *_3D_COMMON_TAIL,
    *_3D_STORAGE_TAIL,
    *_3D_BATCHED_FLAGS,
]

# Flat template registry.
_TEMPLATE_SPECS: dict[str, _Spec] = {
    "maxwell_tm_forward": _TM_FORWARD_SPEC,
    "maxwell_tm_forward_with_storage": _TM_FORWARD_WITH_STORAGE_SPEC,
    "maxwell_tm_backward": _TM_BACKWARD_SPEC,
    "maxwell_3d_forward": _3D_FORWARD_SPEC,
    "maxwell_3d_forward_with_storage": _3D_FORWARD_WITH_STORAGE_SPEC,
    "maxwell_3d_backward": _3D_BACKWARD_SPEC,
}

_ARGTYPES_CACHE: dict[tuple[str, str], list[Any]] = {}
_ARGTYPES_INITIALIZED = False


def _torch_dtype_to_backend_dtype(dtype: torch.dtype) -> str:
    try:
        return _TORCH_DTYPE_TO_BACKEND_DTYPE[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported dtype {dtype}") from exc


def _template_argtypes(template_name: str, backend_dtype: str) -> list[Any]:
    cache_key = (template_name, backend_dtype)
    cached = _ARGTYPES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        spec = _TEMPLATE_SPECS[template_name]
    except KeyError as exc:
        raise KeyError(f"Unknown backend template {template_name!r}.") from exc

    float_type = _BACKEND_DTYPE_TO_CTYPE[backend_dtype]
    argtypes = [float_type if ctype is FLOAT_TYPE else ctype for ctype in _build(spec)]
    _ARGTYPES_CACHE[cache_key] = argtypes
    return argtypes


def _assign_argtypes_for_variant(
    propagator: str,
    accuracy: int,
    backend_dtype: str,
    pass_name: str,
) -> None:
    if _dll is None:
        return

    template_name = f"{propagator}_{pass_name}"
    argtypes = _template_argtypes(template_name, backend_dtype)

    for device_name in _SUPPORTED_DEVICES:
        func_name = f"{propagator}_{accuracy}_{backend_dtype}_{pass_name}_{device_name}"
        func = getattr(_dll, func_name, None)
        if func is None:
            continue
        func.argtypes = argtypes
        func.restype = None


def _initialize_argtypes() -> None:
    global _ARGTYPES_INITIALIZED
    if _ARGTYPES_INITIALIZED or _dll is None:
        return

    for propagator, pass_name, accuracy, backend_dtype in itertools.product(
        ("maxwell_tm", "maxwell_3d"),
        ("forward", "forward_with_storage", "backward"),
        _SUPPORTED_ACCURACIES,
        _SUPPORTED_BACKEND_DTYPES,
    ):
        _assign_argtypes_for_variant(propagator, accuracy, backend_dtype, pass_name)

    _ARGTYPES_INITIALIZED = True


def get_backend_function(
    propagator: str,
    pass_name: str,
    accuracy: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CFunctionPointer:
    """Selects and returns the appropriate backend C/CUDA function.

    Args:
        propagator: The name of the propagator (e.g., "maxwell_tm").
        pass_name: The name of the pass (e.g., "forward", "backward").
        accuracy: The finite-difference accuracy order.
        dtype: The torch.dtype of the tensors.
        device: The torch.device the tensors are on.

    Returns:
        The backend function pointer.

    Raises:
        AttributeError: If the function is not found in the shared library.
        TypeError: If the dtype is not torch.float32 or torch.float64.
        RuntimeError: If the backend is not available.

    """
    dll = get_dll()

    _initialize_argtypes()

    dtype_str = _torch_dtype_to_backend_dtype(dtype)

    device_str = device.type

    func_name = f"{propagator}_{accuracy}_{dtype_str}_{pass_name}_{device_str}"

    try:
        return getattr(dll, func_name)
    except AttributeError as e:
        raise AttributeError(f"Backend function {func_name} not found.") from e


def tensor_to_ptr(tensor: torch.Tensor | None) -> int:
    """Convert a PyTorch tensor to a C pointer (int).

    Args:
        tensor: The tensor to convert, or None.

    Returns:
        The data pointer as an integer, or 0 if tensor is None.

    """
    if tensor is None:
        return 0
    if torch._C._functorch.is_functorch_wrapped_tensor(tensor):
        tensor = torch._C._functorch.get_unwrapped(tensor)
    return tensor.data_ptr()


def ensure_contiguous(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Ensure a tensor is contiguous in memory.

    Args:
        tensor: The tensor to check, or None.

    Returns:
        A contiguous version of the tensor, or None.

    """
    if tensor is None:
        return None
    return tensor.contiguous()


# Argtypes are assigned lazily in get_backend_function().
