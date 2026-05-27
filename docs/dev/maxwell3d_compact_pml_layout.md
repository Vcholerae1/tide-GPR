# Maxwell 3D Compact CPML Layout Prototype

This note defines the first compact CPML memory layout prototype for the 3D
Maxwell CUDA path. It is intentionally a layout/parity prototype, not a
claim that the main H/E kernels are faster.

## Directional Slabs

Each CPML auxiliary field belongs to exactly one PML direction. Edge and
corner cells reuse the slabs for every active direction; there are no separate
edge or corner arrays.

| Direction | Auxiliary fields | Compact shape |
|---|---|---|
| z | `m_hy_z`, `m_hx_z`, `m_ey_z`, `m_ex_z` | `n_shots x nz_compact x ny x nx` |
| y | `m_hz_y`, `m_hx_y`, `m_ez_y`, `m_ex_y` | `n_shots x nz x ny_compact x nx` |
| x | `m_hz_x`, `m_hy_x`, `m_ez_x`, `m_ey_x` | `n_shots x nz x ny x nx_compact` |

The compact indices preserve the same cells that `zero_interior` keeps in the
current full-domain implementation:

```text
left slab:  [0, fd_pad_low + pml_low)
right slab: [size - fd_pad_high - pml_high, size)
```

This means the compact layout is a storage change only. It does not alter the
CPML recurrence, the Yee staggering, or edge/corner physics.

## Current Status

- Implemented layout descriptor and pack/unpack helpers in
  `src/tide/maxwell/compact_pml.py`.
- Added parity tests that verify `unpack(pack(full_zeroed_aux)) == full_zeroed_aux`.
- Added compact-vs-full CPML auxiliary memory estimates to
  `tools/profile_maxwell3d_cuda_6q.py`.

The CUDA kernels still use full-domain CPML arrays. Wiring this layout into the
propagator requires replacing `m_*[i]` accesses with direction-specific compact
indexing and deciding how the public API should expose final auxiliary fields.
