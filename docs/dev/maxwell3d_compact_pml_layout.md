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

The base compact indices preserve the same cells that `zero_interior` keeps in
the current full-domain implementation:

```text
left slab:  [0, fd_pad_low + pml_low)
right slab: [size - fd_pad_high - pml_high, size)
```

The electric auxiliary fields updated by the H kernel (`m_ey_z`, `m_ex_z`,
`m_ez_y`, `m_ex_y`, `m_ez_x`, `m_ey_x`) also include one extra high-side cell
because the half-cell PML profile starts at `pml_*1 - 1` on the high side in the
CUDA kernels. This is intentionally tracked per auxiliary field rather than per
direction.

This means the compact layout is a storage change only. It does not alter the
CPML recurrence, the Yee staggering, or edge/corner physics.

## PML Cell Ordering

The experimental split launch path builds two ordered PML cell lists in Python:

- H update: half-cell high-side PML starts at `pml_*1 - 1`.
- E update: regular high-side PML starts at `pml_*1`.

Both lists use a union decomposition that avoids edge/corner duplicates while
keeping x contiguous inside each rectangular piece:

```text
1. z slabs over all active y/x
2. y slabs over non-z-PML active z and all active x
3. x slabs over non-z/y-PML active z/y
```

The CUDA split path then runs an interior kernel with no CPML auxiliary access,
followed by an indexed PML kernel over only those ordered cells.

## Archived Status

The compact CPML runtime experiment was removed from the codebase on
2026-06-06. This document is retained as an implementation and profiling record
only.

Removed pieces:

- `src/tide/maxwell/compact_pml.py` layout descriptor and pack/unpack helpers.
- Compact CPML parity tests.
- CUDA forward-only compact auxiliary tensors behind `TIDE_EM3D_COMPACT_CPML`.
- Experimental interior/PML split launch behind `TIDE_EM3D_PML_SPLIT` and
  `TIDE_EM3D_INTERIOR_TILED`.
- Compact-vs-full CPML auxiliary memory estimates in
  `tools/profile_maxwell3d_cuda_6q.py`.

## 2026-05-28 Profile Result

On RTX 4070, 64³ model, stencil 2, pml 8, nt 500, one shot:

| Mode | Mean time | Cell-steps/s | Peak allocated |
|---|---:|---:|---:|
| default | 29.51 ms | 9.01 G | 92.6 MB |
| compact CPML | 30.62 ms | 8.68 G | 93.1 MB |
| PML split | 41.34 ms | 6.43 G | 96.7 MB |
| PML split + compact CPML | 44.02 ms | 6.04 G | 97.3 MB |

On the same model with pml 16 and nt 300:

| Mode | Mean time | Cell-steps/s | Peak allocated |
|---|---:|---:|---:|
| default | 67.48 ms | 4.06 G | 156.6 MB |
| PML split | 112.15 ms | 2.44 G | 166.6 MB |

The split launch is therefore an experiment, not a recommended fast path. The
extra H/E PML kernels and indexed cell decoding cost more than the branch work
removed from the interior kernels. Compact CPML is still useful as a layout
prototype, but with the current public API it unpacks final auxiliary fields
back to full-domain tensors, so peak allocated memory does not drop yet.

## Uniform Coefficient Fast Path

The next useful memory-traffic reduction is not CPML-specific. For uniform
material models, `ca`, `cb`, and `cq` are constant across the padded domain, but
the default kernels still load those coefficient arrays for every cell update.
`TIDE_EM3D_UNIFORM_COEFFS=1` makes Python verify that all three coefficient
fields are uniform, then passes scalar values to CUDA. The H/E kernels use a
template specialization that reads these values from constant memory instead of
loading `cq[j]`, `ca[j]`, and `cb[j]` from global memory.

On RTX 4070, 64³ model, stencil 2, pml 8, nt 500, one shot:

| Mode | Mean time | Cell-steps/s | Peak allocated |
|---|---:|---:|---:|
| default | 29.85 ms | 8.90 G | 92.6 MB |
| uniform coefficients | 27.68 ms | 9.60 G | 93.1 MB |

On the same model with pml 16 and nt 300:

| Mode | Mean time | Cell-steps/s | Peak allocated |
|---|---:|---:|---:|
| default | 61.63 ms | 4.44 G | 156.6 MB |
| uniform coefficients | 45.94 ms | 5.96 G | 157.5 MB |

This is the first measured fast path in this series that consistently improves
runtime. It is only valid for uniform coefficients and is ignored for Debye or
gradient paths.

## Gradient Snapshot Storage

For gradient workloads, the broader memory-traffic lever is snapshot storage,
not material-coefficient loads. The native 3D CUDA gradient path normally stores
six full-domain snapshot components per saved time step. The e-only snapshot
backend stores the three E fields plus final E, then reconstructs curl terms in
the adjoint pass. It is available with:

```python
tide.maxwell3d(
    ...,
    python_backend=False,
    execution_backend="eonly_snapshot",
    storage_mode="device",
)
```

It can be combined with `storage_compression="bf16"` to reduce snapshot bytes
further while keeping the field update and adjoint arithmetic in fp32.

Measured with `tools/profile_maxwell3d_gradient_storage.py` on RTX 4070,
heterogeneous 64³ model, pml 8, nt 120, one shot, 16 receivers, gradients for
both `epsilon` and `sigma`:

| Mode | Mean time | Cell-steps/s | Peak allocated | Snapshot estimate | Grad rel L2 vs full |
|---|---:|---:|---:|---:|---:|
| full fp32 snapshots | 34.48 ms | 1.85 G | 1.66 GB | 1.53 GB | 0 |
| bf16 snapshots | 26.24 ms | 2.43 G | 898 MB | 765 MB | ~4.6e-4 |
| e-only fp32 snapshots | 29.47 ms | 2.16 G | 905 MB | 772 MB | ~1.6e-7 |
| e-only bf16 snapshots | 22.77 ms | 2.80 G | 520 MB | 386 MB | ~1.1e-3 |

This is a better next gradient direction than jumping directly to fp16 compute:
it targets the saved-state traffic and memory footprint while leaving numerical
compute in fp32. True fp16/half2 field computation should remain a separate
explicit numerical mode with gradient dot-test and stability validation.

### Checkpoint/Recompute Prototype

`execution_backend="checkpoint_recompute"` is an experimental CUDA-only
prototype for larger batched-shot gradient steps. It stores the 18 wavefield and
CPML state tensors only at segment boundaries, replays one segment at a time
during backward, and feeds the replayed segment snapshots into the existing
native adjoint. `storage_chunk_steps` controls the segment length; when it is
zero, the CUDA path chooses a simple `sqrt(6 * nt)` segment length. The prototype
requires `storage_mode="device"`, `model_gradient_sampling_interval=1`, and no
forward/backward callbacks.

`execution_backend="checkpoint_revolve"` keeps the same leaf segment replay but
moves the checkpoint scheduler into the CUDA backend. When
`reference/revolve/revolve.cpp` is available at build time, the native backend
uses the reference Offline Revolve action stream over coarse leaf segments:
`takeshot` writes a full 18-field checkpoint slot, `restore` reloads one,
`advance` recomputes without snapshot storage, and `firsturn`/`youturn` replay
one segment with ordinary E/curl snapshots before calling the existing adjoint.
The checkpoint pool uses `ceil(log2(num_segments)) + 1` full-state slots plus
one current-state scratch and one leaf replay scratch.

The expected tradeoff is lower peak snapshot memory, enabling more shots per
batch, in exchange for extra forward replay over the time axis. The segmented
prototype is usually faster when all segment-boundary checkpoints fit; the
Revolve-style prototype is the capacity path when checkpoint memory is the
limiting factor.
Use `tools/profile_maxwell3d_gradient_storage.py --modes full eonly_bf16
checkpoint_bf16 revolve_bf16 --storage-chunk-steps 32` to compare end-to-end
training-step time and peak allocation on a representative workload.

## Direct Material-Gradient Prototype

`execution_backend="direct_material_grad"` is a CUDA-only prototype that reuses
the e-only snapshot layout but bypasses the `grad_ca/grad_cb` intermediate
fields. The backward kernel directly accumulates material gradients from
`E_t`, `E_{t+1}`, `lambda_E`, and `cb`; the source-cell correction is applied
directly to `grad_epsilon` / `grad_sigma`.

For the same 64³ / pml 8 / nt 120 workload with both `epsilon` and `sigma`
gradients:

| Mode | Mean time | Peak allocated | Grad rel L2 vs full |
|---|---:|---:|---:|
| e-only fp32 | 30.07 ms | 905 MB | ~1.8e-7 |
| direct fp32 | 29.36 ms | 900 MB | ~6.3e-7 |
| e-only bf16 | 23.86 ms | 520 MB | ~1.1e-3 |
| direct bf16 | 22.56 ms | 515 MB | ~1.1e-3 |

For nt 300, the both-gradient gain is not stable: direct fp32 was slightly
slower than e-only fp32, and direct bf16 was only about 0.7% faster. The
prototype is more useful when only one material gradient is requested:

| Workload | e-only fp32 | direct fp32 | e-only bf16 | direct bf16 |
|---|---:|---:|---:|---:|
| epsilon-only, nt 120 | 28.58 ms | 26.09 ms | 22.13 ms | 21.85 ms |
| sigma-only, nt 120 | 30.18 ms | 27.71 ms | 24.00 ms | 20.66 ms |

This suggests a full migration should be conditional rather than global:
direct material-gradient is promising for single-material-gradient paths, while
e-only bf16 remains the stronger default candidate for broad both-gradient
training steps until Nsight confirms the direct kernel is removing a dominant
kernel-time share.
