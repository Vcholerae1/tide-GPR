# Execution Capability Matrix

The authoritative rows are exposed by
`tide.core.backends.backend_capabilities()`. This table is the current stable
baseline; a new row must be added in code and covered by matrix tests before
the table is updated.

Gradient targets are the inputs a solver may be asked to back-propagate into:
`epsilon`, `sigma`, and `mu` are background-model tensors, `perturbation`
covers the Born perturbation inputs (`depsilon`, `dsigma`, `dca`, `dcb`),
`source` is the source-amplitude wavelet, and `state` is any initial wavefield
or Born-derivative state tensor.

| Backend | Dimension | Operations | Devices | Dtypes | Compute modes | Storage | Gradient targets | Callbacks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Python reference | TM2D | forward | CPU, CUDA | float32, float64 | native | auto, device, CPU, disk, none | epsilon, sigma, mu, perturbation, source, state | yes |
| Python reference | TM2D | born | CPU, CUDA | float32, float64 | native | auto, device, CPU, disk, none | epsilon, sigma, mu, perturbation, source, state | no |
| Python reference | TM2D | hvp | CPU, CUDA | float32, float64 | native | device, CPU, disk | epsilon, sigma, mu, perturbation, source, state | no |
| Python reference | TM2D | linearization | CPU, CUDA | float32, float64 | native | device, CPU, disk | epsilon, sigma, mu, perturbation, source, state | no |
| Python reference | EM3D | forward | CPU, CUDA | float32, float64 | native | auto, device, CPU, disk, none | epsilon, sigma, mu, perturbation, source, state | yes |
| Python reference | EM3D | born | CPU, CUDA | float32, float64 | native | device, none | epsilon, sigma, mu, perturbation, source, state | no |
| Python reference | EM3D | hvp | CPU, CUDA | float32, float64 | native | device | epsilon, sigma, mu, perturbation, source, state | no |
| Native | TM2D | forward | CPU, CUDA | float32, float64 | native | auto, device, CPU, disk, none | epsilon, sigma, source | yes |
| Native | TM2D | born | CPU, CUDA | float32, float64 | native | auto, device, CPU, disk, none | epsilon, sigma, perturbation | no |
| Native | TM2D | hvp | CPU, CUDA | float32, float64 | native | device, CPU, disk | epsilon, sigma | no |
| Native | TM2D | linearization | CPU, CUDA | float32, float64 | native | device, CPU, disk | epsilon, sigma | no |
| Native | EM3D | forward | CPU, CUDA | float32, float64 | native | auto, device, CPU, disk, none | epsilon, sigma, source | yes |
| Native | EM3D | born | CPU, CUDA | float32, float64 | native | device, none | epsilon, sigma, perturbation | no |
| Native | EM3D | hvp | CPU, CUDA | float32, float64 | native | device | epsilon, sigma | no |

Notes on cells that are deliberately narrower than a naive reading:

- Linearization is a TM2D-only feature (`linearize_maxwelltm`); there is no EM3D
  linearization context, so EM3D rows do not list it.
- Storage cells mirror what the public API can express per operation. Native
  EM3D Born snapshots are device-only (`storage_mode="device"` or `"none"`),
  native EM3D HVP always executes with device snapshots (no storage argument
  is exposed), and native TM2D HVP/linearization accept `device`, `cpu`, and
  `disk`. The Python reference ignores storage, so every reachable mode is
  executable there.
- Native Born differentiates with respect to the `epsilon`/`sigma` background
  model and the Born perturbation. Perturbation gradients require snapshot
  storage: a plan requesting `perturbation` with `storage_mode="none"` is
  rejected by the capability matrix or routed to the Python reference.
- Plans that require gradients with respect to `mu`, source amplitudes, or
  initial wavefields are routed by `select_backend` to the Python reference
  backend, or rejected with `fallback="error"`.
- Native forward with dispersion supports inference only: dispersion combined
  with any gradient target is rejected by the capability matrix (or routed to
  the reference). Residual runtime-only conditions (torch.func transforms,
  missing ABI symbols, dispersion with explicit snapshot storage) honor the
  fallback policy in the adapter: they raise under `fallback="error"` and
  switch to the reference only under `fallback="reference"`.
- Callbacks are wired only on forward entry points, so non-forward rows report
  `no`.

Some operation-specific constraints are intentionally represented by the
central decision object rather than by the kernel wrappers: Python HVP and
linearization currently limit gradient sampling, and native TM2D full HVP has
storage restrictions. The Python and native TM2D sampling-interval messages
are covered by `tests/test_core_plan.py` and `tests/test_api_wrappers.py`, and
the native TM2D full-HVP storage restriction is covered by
`tests/test_core_plan.py` (`test_native_tm2d_full_hvp_requires_device_storage`).
