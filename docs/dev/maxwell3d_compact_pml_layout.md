# Maxwell 3D Compact CPML Layout Prototype

This page is retained only as an archived design note. The compact CPML runtime
path is not part of the main branch.

The prototype explored storing CPML auxiliary fields only in directional PML
slabs instead of full-domain tensors. The intended layout was:

| Direction | Auxiliary fields | Compact shape |
|---|---|---|
| z | `m_hy_z`, `m_hx_z`, `m_ey_z`, `m_ex_z` | `n_shots x nz_compact x ny x nx` |
| y | `m_hz_y`, `m_hx_y`, `m_ez_y`, `m_ex_y` | `n_shots x nz x ny_compact x nx` |
| x | `m_hz_x`, `m_hy_x`, `m_ez_x`, `m_ey_x` | `n_shots x nz x ny x nx_compact` |

It was not merged into the main runtime because it did not yet provide a clear
end-to-end win: the public API still returns full-domain auxiliary fields, the
CUDA kernels would need direction-specific indexing, and the extra layout logic
increased maintenance and testing cost.

Use the archive tag `archive/experimental-snapshot-2026-06-13` to inspect the
removed prototype code.
