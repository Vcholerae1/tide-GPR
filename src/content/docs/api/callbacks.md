---
title: "Module: tide.callbacks"
description: "Inspect model fields, wavefields, and gradients during propagation."
---

Callback state objects for inspecting forward and backward propagation.

## Classes
- CallbackState

## Types
- Callback


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

## Practical Notes

- Use forward_callback for monitoring wave propagation statistics.
- Use backward_callback to inspect gradients and adjoint wavefields.
- Avoid expensive Python-side operations every step; use callback_frequency to thin callback cadence.

## Accessor behavior

```python
field = state.get_wavefield("Ey", view="inner")
material = state.get_model("epsilon", view="inner")
gradient = state.get_gradient("epsilon", view="inner")
```

An unknown name or unavailable gradient raises rather than returning an
unrelated tensor. The selected view slices the same logical quantity to the
physical interior, CPML extent, or full padded domain.

`CallbackState.dt` describes the callback-visible propagation interval.
`grid_spacing`, `fd_pad`, and `pml_width` provide the metadata needed to map
array locations back to the computational domain.

## Wiring callbacks

Structured operators accept an `Observers` value when a forward or linearized
operation exposes observers:

```python
observers = tide.Observers(
    forward=monitor,
    frequency=20,
)
result = operator(model, observers=observers)
```

Check the selected backend capability. Callback support is advertised on
forward rows and may be unavailable for tangent or second-order operations.

## Performance contract

Callbacks run synchronously with propagation. Moving a large CUDA tensor to CPU
inside the callback synchronizes the device. Prefer device-side reductions,
sparse cadence, and deferred visualization.
