# API Orientation

Use this page to decide which public API layer matches your workflow.

## Functional APIs

- `tide.maxwelltm(...)`
- `tide.maxwell3d(...)`
- `tide.borntm(...)`
- `tide.born3d(...)`

Use the functional APIs when:

- you want the shortest path from tensors to receiver data,
- you are scripting experiments directly,
- you do not need to keep a model object around.

## Module APIs

- `tide.MaxwellTM(...)`
- `tide.Maxwell3D(...)`
- `tide.BornTM(...)`
- `tide.Born3D(...)`

Use the module APIs when:

- you want model parameters stored in a reusable `torch.nn.Module`,
- you are integrating with optimizers and training loops,
- you want a model object that can be moved between devices.

## Supporting APIs

- `tide.ricker` for source design
- `tide.CallbackState` for forward and backward callbacks
- `tide.DebyeDispersion` for dispersive-material inputs
- `tide.backend_utils` for backend availability checks

## Recommended Reading Order

1. Start with `tide.maxwelltm(...)` unless you already need 3D geometry.
2. Move to `tide.maxwell3d(...)` when component selection or full 3D layouts matter.
3. Use `tide.borntm(...)` or `tide.born3d(...)` when you want a unified Born propagator that advances background and scattered wavefields together.
4. Adopt `MaxwellTM(...)`, `Maxwell3D(...)`, `BornTM(...)`, or `Born3D(...)` when you want a reusable model object inside a training or inversion loop.
5. Read `guides/configuration.md` before tuning storage, callbacks, or backend behavior.
