# Module: tide.maxwell

## Public API
- class MaxwellTM
- class Maxwell3D
- class BornTM
- class Born3D
- function maxwelltm
- function maxwell3d
- function borntm
- function born3d

## maxwelltm

Signature highlights:

```python
maxwelltm(
    epsilon, sigma, mu,
    grid_spacing, dt,
    source_amplitude, source_location, receiver_location,
    stencil=2, pml_width=20,
    ...,
    python_backend=False,
    storage_mode="device",
    storage_compression=False,
)
```

Key inputs:
- epsilon, sigma, mu: shape [ny, nx] or [B, ny, nx]
- source_amplitude: [n_shots, n_sources, nt] or [B, n_shots, n_sources, nt]
- source_location: [n_shots, n_sources, 2] or [B, n_shots, n_sources, 2]
- receiver_location: [n_shots, n_receivers, 2] or [B, n_shots, n_receivers, 2]

Return tuple:
- Ey, Hx, Hz
- m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
- receiver_amplitudes with shape [nt, n_shots, n_receivers] for shared models
- receiver_amplitudes with shape [nt, B, n_shots, n_receivers] for batched models

Important behavior:
- TIDE checks CFL stability and may use an internal smaller time step.
- If internal sub-stepping is used, source signals are upsampled and receiver traces are downsampled automatically.
- save_snapshots defaults to auto behavior based on gradient requirements.
- Batched models are supported by the native backend and by `python_backend=True`.
- In batched-model Python mode, TIDE uses an internal `vmap` over the model axis.
- Batched-model Python mode does not support forward/backward callbacks.

## maxwell3d

Signature highlights:

```python
maxwell3d(
    epsilon, sigma, mu,
    grid_spacing, dt,
    source_amplitude, source_location, receiver_location,
    source_component="ey", receiver_component="ey",
    python_backend=False,
    storage_mode="device",
)
```

Key inputs:
- epsilon, sigma, mu: shape [nz, ny, nx] or [B, nz, ny, nx]
- source_location: [n_shots, n_sources, 3] or [B, n_shots, n_sources, 3]
- receiver_location: [n_shots, n_receivers, 3] or [B, n_shots, n_receivers, 3]
- source_component and receiver_component: one of ex, ey, ez

Return tuple:
- Ex, Ey, Ez, Hx, Hy, Hz
- 12 CPML memory tensors
- receiver_amplitudes shaped [nt, n_shots, n_receivers] for shared models
- receiver_amplitudes shaped [nt, B, n_shots, n_receivers] for batched models

## borntm / born3d

The Born propagators now follow a unified two-wavefield layout similar to
Deepwave's `ScalarBorn`.

Key inputs:
- background model tensors: `epsilon`, `sigma`, `mu`
- perturbation tensors: `depsilon` / `dsigma` or `dca` / `dcb`
- `receiver_location` for scattered traces
- `bg_receiver_location` for background traces
- optional background and scattered initial wavefields

Return tuple:
- background final state tensors, in the same order as `maxwelltm` or `maxwell3d`
- scattered final state tensors, in the same order as the background states
- `bg_receiver_amplitudes`
- `receiver_amplitudes`

Autograd behavior:
- use PyTorch autograd directly on the perturbation inputs to obtain Born
  Jacobian-vector or vector-Jacobian products
- the explicit `borntm_adjoint` and `born3d_adjoint` APIs were removed
- native backend gradients are supported for perturbation inputs and supported
  background/source-gradient paths
- `storage_compression="bf16"` stores saved Born backward snapshots in bf16 for
  float32 native workflows that support compressed storage. This includes TM2D
  CPU/CUDA and 3D CUDA scattered direct snapshots used by background-gradient
  adjoints
- unsupported gradient requests, such as some initial-wavefield gradients, fall
  back to the Python reference path

## Class Wrappers

- MaxwellTM and Maxwell3D are torch.nn.Module wrappers that store background model tensors and call maxwelltm/maxwell3d in forward.
- BornTM and Born3D are torch.nn.Module wrappers that store the background model plus an optional Born perturbation and call the unified two-wavefield borntm/born3d propagators in forward.
- Useful when integrating with training loops that repeatedly propagate on the same model object.

## Choosing The Right Maxwell Entry Point

- Use `maxwelltm` for the fastest onboarding path and most 2D examples.
- Use `maxwell3d` when component selection and full 3D geometry matter.
- Use `borntm` or `born3d` when you want a Deepwave-style two-wavefield Born propagator that returns both background and scattered states.
- Use `MaxwellTM`, `Maxwell3D`, `BornTM`, or `Born3D` when you want model tensors stored inside a `torch.nn.Module`.

See:

- `guides/api-orientation.md`
- `guides/modeling.md`
- `guides/inversion.md`

## Advanced or Internal Functions
- prepare_parameters
- maxwell_func
- maxwell_python
- update_E
- update_H
- maxwell_c_cuda

## Autograd Functions
- MaxwellTMForwardFunc
- MaxwellTMForwardBoundaryFunc

## Internal Helpers
- _register_ctx_handle
- _get_ctx_handle
- _release_ctx_handle
- _compute_boundary_indices_flat
