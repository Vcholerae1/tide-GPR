from typing import Union

import torch


def reverse_pad(pad: list[int]) -> list[int]:
    """Reverse the padding order for use with torch.nn.functional.pad.

    PyTorch's pad function expects padding in reverse order (last dim first).
    This function converts [y0, y1, x0, x1] to [x0, x1, y0, y1].

    Args:
        pad: Padding values in [y0, y1, x0, x1] format.

    Returns:
        Padding values in PyTorch format [x0, x1, y0, y1].
    """
    # For 2D: [y0, y1, x0, x1] -> [x0, x1, y0, y1]
    result = []
    for i in range(len(pad) // 2 - 1, -1, -1):
        result.extend([pad[i * 2], pad[i * 2 + 1]])
    return result


def create_or_pad(
    tensor: torch.Tensor,
    pad: Union[int, list[int]],
    device: torch.device,
    dtype: torch.dtype,
    size: tuple[int, ...],
    mode: str = "constant",
) -> torch.Tensor:
    """Creates a zero tensor of specified size or pads an existing tensor.

    If the input tensor is empty (numel == 0), a new zero tensor with the
    given size is created. Otherwise, the tensor is padded according to
    the specified mode.

    This is a unified padding function that supports:
    - Zero padding (mode='constant') for wavefields
    - Replicate padding (mode='replicate') for models

    Args:
        tensor: The input tensor to be created or padded.
        pad: The padding to apply. Can be an integer (for uniform padding)
            or a list of integers [y0, y1, x0, x1] for per-side padding.
        device: The PyTorch device for the tensor.
        dtype: The PyTorch data type for the tensor.
        size: The desired size of the tensor if it needs to be created.
        mode: Padding mode ('constant', 'replicate', 'reflect', 'circular').
            Default is 'constant' (zero padding)

    Returns:
        The created or padded tensor.

    Example:
        >>> # Create a zero tensor of size [2, 110, 110] (batch=2, with padding)
        >>> wf = create_or_pad(torch.empty(0), 5, device, dtype, (2, 110, 110))
        >>>
        >>> # Pad a wavefield with zeros [2, 100, 100] -> [2, 110, 110]
        >>> wf_padded = create_or_pad(wf, [5, 5, 5, 5], device, dtype, (2, 110, 110))
        >>>
        >>> # Pad a model with replicate mode [100, 100] -> [110, 110]
        >>> eps_padded = create_or_pad(eps, [5, 5, 5, 5], device, dtype, (110, 110), mode='replicate')
    """
    if isinstance(pad, int):
        # Convert single int to [pad, pad, pad, pad, ...] for each spatial dimension
        # size includes batch dimension if len > 2, so spatial ndim = len(size) - 1 or len(size)
        ndim = len(size) - 1 if len(size) > 2 else len(size)
        pad = [pad] * ndim * 2

    if tensor.numel() == 0:
        return torch.zeros(size, device=device, dtype=dtype)

    if max(pad) == 0:
        return tensor.clone()

    # Reverse padding for PyTorch's pad function
    reversed_pad = reverse_pad(pad)

    # For non-constant padding modes (replicate, reflect, circular),
    # PyTorch requires:
    # - 2D spatial padding: 3D or 4D input
    # - 3D spatial padding: 4D or 5D input
    original_ndim = tensor.ndim
    needs_unsqueeze = original_ndim in {2, 3} and mode != "constant"

    if needs_unsqueeze:
        tensor = tensor.unsqueeze(0)

    result = torch.nn.functional.pad(tensor, reversed_pad, mode=mode)

    if needs_unsqueeze:
        result = result.squeeze(0)

    # PyTorch's autograd system automatically tracks gradients through operations.
    # Explicitly calling requires_grad_() is incompatible with torch.func transforms.
    # Simply return the result; gradient tracking is handled automatically.
    return result


def zero_interior(
    tensor: torch.Tensor,
    fd_pad: Union[int, list[int]],
    pml_width: list[int],
    dim: int,
) -> torch.Tensor:
    """Zero out the interior region of a tensor (keeping only PML regions).

    This is used for CPML auxiliary variables which should only be non-zero
    in the PML regions. Setting the interior to zero allows the propagator
    to skip unnecessary PML calculations in those regions.

    Args:
        tensor: The input tensor with shape [batch, ny, nx].
        fd_pad: Finite difference padding. Can be an int or list [y0, y1, x0, x1].
        pml_width: The width of PML regions [top, bottom, left, right].
        dim: The spatial dimension to zero (0 for y, 1 for x).

    Returns:
        The tensor with interior region zeroed out.
    """
    shape = tensor.shape[1:]  # Spatial dimensions (without batch)
    ndim = len(shape)

    if isinstance(fd_pad, int):
        fd_pad = [fd_pad] * 2 * ndim

    # Calculate interior slice for the specified dimension
    interior_start = fd_pad[dim * 2] + pml_width[dim * 2]
    interior_end = shape[dim] - pml_width[dim * 2 + 1] - fd_pad[dim * 2 + 1]

    # Zero out the interior
    if dim == 0:  # y dimension
        tensor[:, interior_start:interior_end, :].fill_(0)
    else:  # x dimension
        tensor[:, :, interior_start:interior_end].fill_(0)

    return tensor
