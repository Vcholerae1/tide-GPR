import ctypes
import pathlib
import platform
from ctypes import c_bool, c_double, c_float, c_int64, c_void_p
from typing import Any, Callable, Optional, TypeAlias, list

import torch

CFunctionPointer: TypeAlias = Any

# Platform-specific shared library extension
SO_EXT = {"Linux": "so", "Darwin": "dylib", "Windows": "dll"}.get(platform.system())
if SO_EXT is None:
    raise RuntimeError("Unsupported OS or platform type")

# Try to load the shared library
_lib_path = pathlib.Path(__file__).resolve().parent / f"libtide_C.{SO_EXT}"
_dll: Optional[ctypes.CDLL] = None

try:
    _dll = ctypes.CDLL(str(_lib_path))
except OSError:
    # Library not compiled yet, will use Python backend
    pass


def is_backend_available() -> bool:
    """Check if the C/CUDA backend is available."""
    return _dll is not None


def get_dll() -> ctypes.CDLL:
    """Get the loaded DLL, raising an error if not available."""
    if _dll is None:
        raise RuntimeError(
            f"C/CUDA backend not available. Please compile the library first. "
            f"Expected library at: {_lib_path}"
        )
    return _dll


# Check if was compiled with OpenMP support
USE_OPENMP = _dll is not None and hasattr(_dll, "omp_get_num_threads")

# Define ctypes argument type templates to reduce repetition while preserving order.
# A placeholder will be replaced by the appropriate float type (c_float or c_double).
FLOAT_TYPE: type = c_float


def get_maxwell_tm_forward_template() -> list[Any]:
    """Returns the argtype template for the Maxwell TM forward propagator."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Source
    args += [c_void_p]  # f (source amplitudes)
    # Fields
    args += [c_void_p] * 3  # ey, hx, hz
    # PML memory variables
    args += [c_void_p] * 4  # m_ey_x, m_ey_z, m_hx_z, m_hz_x
    # Recorded data
    args += [c_void_p]  # r (receiver amplitudes)
    # PML profiles
    args += [c_void_p] * 8  # ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 4  # ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 2  # rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 2  # ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    args += [c_int64]  # step_ratio
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # Start time
    args += [c_int64]  # start_t
    # PML boundaries
    args += [c_int64] * 4  # pml_y0, pml_x0, pml_y1, pml_x1
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


def get_maxwell_tm_backward_template() -> list[Any]:
    """Returns the argtype template for the Maxwell TM backward propagator (v2 with ASM)."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Gradient of receiver data
    args += [c_void_p]  # grad_r
    # Adjoint fields (lambda)
    args += [c_void_p] * 3  # lambda_ey, lambda_hx, lambda_hz
    # Adjoint PML memory variables
    args += [c_void_p] * 4  # m_lambda_ey_x, m_lambda_ey_z, m_lambda_hx_z, m_lambda_hz_x
    # Stored forward values for gradient (Ey and curl_H)
    # For each: store_1, store_3, filenames (char**)
    args += [c_void_p] * 6
    # Gradient outputs
    args += [c_void_p]  # grad_f
    args += [c_void_p] * 4  # grad_ca, grad_cb, grad_eps, grad_sigma
    args += [c_void_p] * 2  # grad_ca_shot, grad_cb_shot (per-shot workspace)
    # PML profiles
    args += [c_void_p] * 8  # ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 4  # ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 2  # rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 2  # ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    args += [c_int64]  # step_ratio
    # Storage mode
    args += [c_int64] * 2  # storage_mode, shot_bytes_uncomp
    # Requires grad flags
    args += [c_bool] * 2  # ca_requires_grad, cb_requires_grad
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # Start time
    args += [c_int64]  # start_t
    # PML boundaries for adjoint propagation
    args += [c_int64] * 4  # pml_y0, pml_x0, pml_y1, pml_x1
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


def get_maxwell_tm_forward_with_storage_template() -> list[Any]:
    """Returns the argtype template for Maxwell TM forward with storage (for ASM backward)."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Source
    args += [c_void_p]  # f (source amplitudes)
    # Fields
    args += [c_void_p] * 3  # ey, hx, hz
    # PML memory variables
    args += [c_void_p] * 4  # m_ey_x, m_ey_z, m_hx_z, m_hz_x
    # Recorded data
    args += [c_void_p]  # r (receiver amplitudes)
    # Storage for backward (Ey and curl_H)
    # For each: store_1, store_3, filenames (char**)
    args += [c_void_p] * 6
    # PML profiles
    args += [c_void_p] * 8  # ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 4  # ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 2  # rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 2  # ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    args += [c_int64]  # step_ratio
    # Storage mode
    args += [c_int64] * 2  # storage_mode, shot_bytes_uncomp
    # Requires grad flags
    args += [c_bool] * 2  # ca_requires_grad, cb_requires_grad
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # Start time
    args += [c_int64]  # start_t
    # PML boundaries
    args += [c_int64] * 4  # pml_y0, pml_x0, pml_y1, pml_x1
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


def get_maxwell_tm_forward_with_boundary_storage_template() -> list[Any]:
    """Returns the argtype template for Maxwell TM forward with boundary storage."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Source
    args += [c_void_p]  # f
    # Fields
    args += [c_void_p] * 3  # ey, hx, hz
    # PML memory variables
    args += [c_void_p] * 4  # m_ey_x, m_ey_z, m_hx_z, m_hz_x
    # Recorded data
    args += [c_void_p]  # r
    # Boundary storage: Ey, Hx, Hz (store_1, store_3, filenames)
    args += [c_void_p] * 9
    # Boundary indices + size
    args += [c_void_p]  # boundary_indices
    args += [c_int64]  # boundary_numel
    # PML profiles
    args += [c_void_p] * 8  # ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 4  # ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 2  # rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 2  # ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    # Storage mode
    args += [c_int64] * 2  # storage_mode, shot_bytes_uncomp
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # PML boundaries
    args += [c_int64] * 4  # pml_y0, pml_x0, pml_y1, pml_x1
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


def get_maxwell_tm_backward_with_boundary_template() -> list[Any]:
    """Returns the argtype template for Maxwell TM backward with boundary storage."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Source (scaled) and grad_r
    args += [c_void_p] * 2  # f, grad_r
    # Reconstructed forward fields and curl(H)
    args += [c_void_p] * 4  # ey, hx, hz, curl_h
    # Adjoint fields (lambda)
    args += [c_void_p] * 3  # lambda_ey, lambda_hx, lambda_hz
    # Adjoint PML memory variables
    args += [c_void_p] * 4  # m_lambda_ey_x, m_lambda_ey_z, m_lambda_hx_z, m_lambda_hz_x
    # Boundary storage: Ey, Hx, Hz (store_1, store_3, filenames)
    args += [c_void_p] * 9
    # Boundary indices + size
    args += [c_void_p]  # boundary_indices
    args += [c_int64]  # boundary_numel
    # Gradient outputs
    args += [c_void_p]  # grad_f
    args += [c_void_p] * 4  # grad_ca, grad_cb, grad_eps, grad_sigma
    args += [c_void_p] * 2  # grad_ca_shot, grad_cb_shot
    # PML profiles
    args += [c_void_p] * 8  # ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 4  # ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 2  # rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 2  # ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    # Storage mode
    args += [c_int64] * 2  # storage_mode, shot_bytes_uncomp
    # Requires grad flags
    args += [c_bool] * 2  # ca_requires_grad, cb_requires_grad
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # PML boundaries
    args += [c_int64] * 4  # pml_y0, pml_x0, pml_y1, pml_x1
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


def get_maxwell_3d_forward_template() -> list[Any]:
    """Returns the argtype template for the 3D Maxwell forward propagator."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Source
    args += [c_void_p]  # f (source amplitudes)
    # Fields
    args += [c_void_p] * 6  # ex, ey, ez, hx, hy, hz
    # PML memory variables
    args += (
        [c_void_p] * 12
    )  # m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y, m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x
    # Recorded data
    args += [c_void_p]  # r
    # PML profiles
    args += [c_void_p] * 12  # az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 6  # kz, kzh, ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 3  # rdz, rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 3  # nz, ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    args += [c_int64]  # step_ratio
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # Start time
    args += [c_int64]  # start_t
    # PML boundaries
    args += [c_int64] * 6  # pml_z0, pml_y0, pml_x0, pml_z1, pml_y1, pml_x1
    # Source/receiver component
    args += [c_int64] * 2  # source_component, receiver_component
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


def get_maxwell_3d_forward_with_storage_template() -> list[Any]:
    """Returns the argtype template for 3D Maxwell forward with storage (ASM)."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Source
    args += [c_void_p]  # f (source amplitudes)
    # Fields
    args += [c_void_p] * 6  # ex, ey, ez, hx, hy, hz
    # PML memory variables
    args += (
        [c_void_p] * 12
    )  # m_hz_y, m_hy_z, m_hx_z, m_hz_x, m_hy_x, m_hx_y, m_ey_z, m_ez_y, m_ez_x, m_ex_z, m_ex_y, m_ey_x
    # Recorded data
    args += [c_void_p]  # r
    # Storage for backward (Ex/Ey/Ez and curl(H) components)
    # For each: store_1, store_3, filenames (char**)
    args += [c_void_p] * 18
    # PML profiles
    args += [c_void_p] * 12  # az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 6  # kz, kzh, ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 3  # rdz, rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 3  # nz, ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    args += [c_int64]  # step_ratio
    # Storage mode
    args += [c_int64] * 2  # storage_mode, shot_bytes_uncomp
    # Requires grad flags
    args += [c_bool] * 2  # ca_requires_grad, cb_requires_grad
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # Start time
    args += [c_int64]  # start_t
    # PML boundaries
    args += [c_int64] * 6  # pml_z0, pml_y0, pml_x0, pml_z1, pml_y1, pml_x1
    # Source/receiver component
    args += [c_int64] * 2  # source_component, receiver_component
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


def get_maxwell_3d_backward_template() -> list[Any]:
    """Returns the argtype template for 3D Maxwell backward propagator (ASM)."""
    args: list[Any] = []
    # Material parameters
    args += [c_void_p] * 3  # ca, cb, cq
    # Gradient of receiver data
    args += [c_void_p]  # grad_r
    # Adjoint fields (lambda)
    args += [
        c_void_p
    ] * 6  # lambda_ex, lambda_ey, lambda_ez, lambda_hx, lambda_hy, lambda_hz
    # Adjoint PML memory variables
    args += (
        [c_void_p] * 12
    )  # m_lambda_ey_z, m_lambda_ez_y, m_lambda_ez_x, m_lambda_ex_z, m_lambda_ex_y, m_lambda_ey_x, m_lambda_hz_y, m_lambda_hy_z, m_lambda_hx_z, m_lambda_hz_x, m_lambda_hy_x, m_lambda_hx_y
    # Stored forward values (Ex/Ey/Ez and curl(H) components)
    # For each: store_1, store_3, filenames (char**)
    args += [c_void_p] * 18
    # Gradient outputs
    args += [c_void_p]  # grad_f
    args += [c_void_p] * 4  # grad_ca, grad_cb, grad_eps, grad_sigma
    args += [c_void_p] * 2  # grad_ca_shot, grad_cb_shot
    # PML profiles
    args += [c_void_p] * 12  # az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh
    # Kappa profiles
    args += [c_void_p] * 6  # kz, kzh, ky, kyh, kx, kxh
    # Source and receiver indices
    args += [c_void_p] * 2  # sources_i, receivers_i
    # Grid spacing
    args += [FLOAT_TYPE] * 3  # rdz, rdy, rdx
    # Time step
    args += [FLOAT_TYPE]  # dt
    # Sizes
    args += [c_int64]  # nt
    args += [c_int64]  # n_shots
    args += [c_int64] * 3  # nz, ny, nx
    args += [c_int64] * 2  # n_sources_per_shot, n_receivers_per_shot
    args += [c_int64]  # step_ratio
    # Storage mode
    args += [c_int64] * 2  # storage_mode, shot_bytes_uncomp
    # Requires grad flags
    args += [c_bool] * 2  # ca_requires_grad, cb_requires_grad
    # Batched flags
    args += [c_bool] * 3  # ca_batched, cb_batched, cq_batched
    # Start time
    args += [c_int64]  # start_t
    # PML boundaries
    args += [c_int64] * 6  # pml_z0, pml_y0, pml_x0, pml_z1, pml_y1, pml_x1
    # Source/receiver component
    args += [c_int64] * 2  # source_component, receiver_component
    # OpenMP threads (CPU only)
    args += [c_int64]  # n_threads
    # Device (for CUDA)
    args += [c_int64]  # device
    return args


# Template registry
templates: dict[str, Callable[[], list[Any]]] = {
    "maxwell_tm_forward": get_maxwell_tm_forward_template,
    "maxwell_tm_forward_with_storage": get_maxwell_tm_forward_with_storage_template,
    "maxwell_tm_forward_with_boundary_storage": get_maxwell_tm_forward_with_boundary_storage_template,
    "maxwell_tm_backward": get_maxwell_tm_backward_template,
    "maxwell_tm_backward_with_boundary": get_maxwell_tm_backward_with_boundary_template,
    "maxwell_3d_forward": get_maxwell_3d_forward_template,
    "maxwell_3d_forward_with_storage": get_maxwell_3d_forward_with_storage_template,
    "maxwell_3d_backward": get_maxwell_3d_backward_template,
}


def _get_argtypes(template_name: str, float_type: type) -> list[Any]:
    """Generates a concrete argtype list from a template and a float type.

    Args:
        template_name: The name of the argtype template to use.
        float_type: The `ctypes` float type (`c_float` or `c_double`)
            to substitute into the template.

    Returns:
        list[Any]: A list of `ctypes` types representing the argument
            signature for a C function.

    """
    template = templates[template_name]()
    return [float_type if t is FLOAT_TYPE else t for t in template]


def _assign_argtypes(
    propagator: str,
    accuracy: int,
    dtype: str,
    direction: str,
) -> None:
    """Dynamically assigns ctypes argtypes to a given C function.

    Args:
        propagator: The name of the propagator (e.g., "maxwell_tm").
        accuracy: The finite-difference accuracy order (e.g., 2, 4, 6, 8).
        dtype: The data type as a string (e.g., "float", "double").
        direction: The direction of propagation (e.g., "forward", "backward").

    """
    if _dll is None:
        return

    template_name = f"{propagator}_{direction}"
    float_type = c_float if dtype == "float" else c_double
    argtypes = _get_argtypes(template_name, float_type)

    for device in ["cpu", "cuda"]:
        func_name = f"{propagator}_{accuracy}_{dtype}_{direction}_{device}"
        try:
            func = getattr(_dll, func_name)
            func.argtypes = argtypes
            func.restype = None  # All C functions return void
        except AttributeError:
            continue


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

    if dtype == torch.float32:
        dtype_str = "float"
    elif dtype == torch.float64:
        dtype_str = "double"
    else:
        raise TypeError(f"Unsupported dtype {dtype}")

    device_str = device.type

    func_name = f"{propagator}_{accuracy}_{dtype_str}_{pass_name}_{device_str}"

    try:
        return getattr(dll, func_name)
    except AttributeError as e:
        raise AttributeError(f"Backend function {func_name} not found.") from e


def tensor_to_ptr(tensor: Optional[torch.Tensor]) -> int:
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


def ensure_contiguous(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Ensure a tensor is contiguous in memory.

    Args:
        tensor: The tensor to check, or None.

    Returns:
        A contiguous version of the tensor, or None.

    """
    if tensor is None:
        return None
    return tensor.contiguous()


# Initialize argtypes for all available functions when the module loads
if _dll is not None:
    for current_accuracy in [2, 4, 6, 8]:
        for current_dtype in ["float", "double"]:
            _assign_argtypes("maxwell_tm", current_accuracy, current_dtype, "forward")
            _assign_argtypes(
                "maxwell_tm", current_accuracy, current_dtype, "forward_with_storage"
            )
            _assign_argtypes(
                "maxwell_tm",
                current_accuracy,
                current_dtype,
                "forward_with_boundary_storage",
            )
            _assign_argtypes("maxwell_tm", current_accuracy, current_dtype, "backward")
            _assign_argtypes(
                "maxwell_tm", current_accuracy, current_dtype, "backward_with_boundary"
            )
            _assign_argtypes("maxwell_3d", current_accuracy, current_dtype, "forward")
            _assign_argtypes(
                "maxwell_3d", current_accuracy, current_dtype, "forward_with_storage"
            )
            _assign_argtypes("maxwell_3d", current_accuracy, current_dtype, "backward")
