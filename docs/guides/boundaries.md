# Boundaries and PML

TIDE uses CPML (Convolutional Perfectly Matched Layer) boundaries to absorb outgoing waves and reduce reflections from computational edges.

## pml_width

Accepted formats:
- 2D:
	- int: same width on all sides
	- [top, bottom, left, right]
- 3D:
	- int: same width on all sides
	- [z0, z1, y0, y1, x0, x1]

Larger PML widths reduce boundary reflection but increase memory and compute cost.

## fd_pad

Finite-difference padding is extra halo space required by stencil order.

Interaction with PML:
- FD padding and PML are both outside the physical model interior.
- Callback views expose full, pml, and inner regions to separate these areas.
- CPML auxiliary fields are zeroed in non-PML interior regions.

## Accuracy and Stencils

Supported stencil orders: 2, 4, 6, 8.

Trade-offs:
- Higher order reduces numerical dispersion at fixed grid spacing.
- Higher order increases arithmetic cost and halo width.
- Order 2 is fastest per step but may need finer grids for equivalent accuracy.

Practical recommendation:
- Start with stencil=2 for quick iteration.
- Increase to 4 or 6 for production-quality inversion when dispersion error dominates.
