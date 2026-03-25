# Module: tide.callbacks

Callback state objects for inspecting forward and backward propagation.

## Classes
- CallbackState

## Types
- Callback

## Functions
- create_callback_state

## Callback Type

Callback is a callable with this pattern:

```python
def callback(state: CallbackState) -> None:
	...
```

## CallbackState

Core properties:
- step: current time index
- nt: total time steps
- dt: time step size
- is_backward: whether this is the adjoint/backward pass

Accessor methods:
- get_wavefield(name, view="inner")
- get_model(name, view="inner")
- get_gradient(name, view="inner")

View options:
- full: full padded domain
- pml: model plus PML region
- inner: physical model interior

## create_callback_state

Factory helper for constructing CallbackState with consistent metadata.

Typical use:
- Users usually do not call this directly.
- It is mainly useful for testing custom callback handlers.

## Practical Notes

- Use forward_callback for monitoring wave propagation statistics.
- Use backward_callback to inspect gradients and adjoint wavefields.
- Avoid expensive Python-side operations every step; use callback_frequency to thin callback cadence.
