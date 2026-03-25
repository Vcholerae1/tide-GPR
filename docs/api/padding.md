# Module: tide.padding

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
