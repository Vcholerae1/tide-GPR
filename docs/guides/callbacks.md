# Callbacks

Callbacks provide controlled access to simulation state during propagation without modifying solver internals.

## CallbackState
- Access wavefields, models, and gradients by name.
- Views: full, pml, inner.

Minimal example:

```python
def monitor(state):
	ey = state.get_wavefield("Ey", view="inner")
	if state.step % 20 == 0:
		print(state.step, float(ey.abs().max()))
```

Field names differ by solver:
- 2D includes Ey, Hx, Hz and CPML memory tensors.
- 3D includes Ex, Ey, Ez, Hx, Hy, Hz and corresponding CPML memories.

## Forward Callback

Runs during forward propagation every callback_frequency steps.

Use cases:
- runtime diagnostics
- energy or amplitude monitoring
- lightweight logging and sanity checks

## Backward Callback

Runs during adjoint/backward propagation.

Use cases:
- inspect gradient magnitudes and support masks
- detect exploding/vanishing gradient regions
- debug inversion behavior

Best practices:
- keep callback work lightweight to avoid slowing propagation
- avoid mutating tensor shapes or device placement
- when heavy post-processing is needed, store compact summaries and process offline
