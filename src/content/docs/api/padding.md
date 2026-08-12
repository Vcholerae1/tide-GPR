---
title: "Module: tide.padding"
description: "Pad material fields and recover physical-domain views safely."
---

Padding and masking helpers for model/field setup.

## Functions
- reverse_pad
- create_or_pad
- zero_interior

## reverse_pad

Converts natural spatial side ordering into torch.nn.functional.pad ordering.

Example:
- input [y0, y1, x0, x1]
- output [x0, x1, y0, y1]

## create_or_pad

Behavior:
- if input tensor is empty, creates a zero tensor of requested size
- otherwise applies torch padding with mode constant, replicate, reflect, or circular

Useful for:
- creating initial fields lazily
- applying model padding and FD halo padding

## zero_interior

Zeroes the interior region for CPML auxiliaries, preserving only PML zones.

Supports:
- 2D tensors with spatial dims [ny, nx]
- 3D tensors with spatial dims [nz, ny, nx]

## Examples

```python
natural = [2, 3, 4, 5]  # y0, y1, x0, x1
torch_order = tide.reverse_pad(natural)
assert torch_order == [4, 5, 2, 3]
```

`create_or_pad` allocates a zero tensor when no initial tensor is supplied by
the caller, or delegates to PyTorch padding for an existing tensor. The
requested `device`, `dtype`, and `size` define the allocation branch.

`zero_interior` is intended for CPML auxiliary state. It preserves boundary
slabs and clears cells that should not carry CPML memory. It is not a general
model mask and should not be used to impose inversion constraints.

## Ordering rule

TIDE configuration uses natural axis-side order. `torch.nn.functional.pad`
starts from the last dimension, so `reverse_pad` is required before passing a
multi-axis width list to PyTorch. A width list with the correct values in the
wrong order can remain shape-valid while padding the wrong sides.
