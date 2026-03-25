# Module: tide.maxwell

## Public API
- class MaxwellTM
- class Maxwell3D
- function maxwelltm
- function maxwell3d

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
- epsilon, sigma, mu: shape [ny, nx]
- source_amplitude: [n_shots, n_sources, nt]
- source_location: [n_shots, n_sources, 2]
- receiver_location: [n_shots, n_receivers, 2]

Return tuple:
- Ey, Hx, Hz
- m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x
- receiver_amplitudes with shape [nt, n_shots, n_receivers]

Important behavior:
- TIDE checks CFL stability and may use an internal smaller time step.
- If internal sub-stepping is used, source signals are upsampled and receiver traces are downsampled automatically.
- save_snapshots defaults to auto behavior based on gradient requirements.

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
- epsilon, sigma, mu: shape [nz, ny, nx]
- source_location: [n_shots, n_sources, 3]
- receiver_location: [n_shots, n_receivers, 3]
- source_component and receiver_component: one of ex, ey, ez, hx, hy, hz

Return tuple:
- Ex, Ey, Ez, Hx, Hy, Hz
- 12 CPML memory tensors
- receiver_amplitudes

## Class Wrappers

- MaxwellTM and Maxwell3D are torch.nn.Module wrappers that store model tensors and call maxwelltm/maxwell3d in forward.
- Useful when integrating with training loops that repeatedly propagate on the same model object.

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
