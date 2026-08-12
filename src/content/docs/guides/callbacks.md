---
title: "Callbacks"
description: "Inspect forward fields, adjoint fields, models, and gradients without modifying solver kernels."
---

Callbacks provide read-oriented access to propagation state at selected time steps. They are useful for diagnosing geometry, boundary absorption, field growth, and gradient support. They are not intended for implementing physics inside Python.

## Callback state

A callback receives one `CallbackState` with:

| Property | Meaning |
| --- | --- |
| `step` | Current internal propagation step exposed to the callback |
| `nt` | Total step count for the current pass |
| `dt` | Internal callback time interval |
| `is_backward` | `True` during reverse or adjoint propagation |
| `wavefields` | Named field tensors available on this solver path |
| `models` | Named padded material tensors |
| `gradients` | Named gradient tensors when available |
| `grid_spacing` | Per-axis spatial spacing |

Use accessors instead of indexing dictionaries directly:

```python
def monitor(state: tide.CallbackState) -> None:
    ey = state.get_wavefield("Ey", view="inner")
    maximum = ey.detach().abs().amax()
    print(state.step, float(maximum.cpu()))
```

Field names depend on the solver. TM2D exposes `Ey`, `Hx`, and `Hz`; EM3D exposes all six electric and magnetic components. CPML memory variables may also be present, but they are implementation-facing diagnostics rather than physical receiver quantities.

## Spatial views

- `inner` returns the physical model interior.
- `pml` includes the physical model and CPML cells.
- `full` also includes finite-difference halo padding.

Use `inner` for physical plots and amplitude checks. Use `pml` to study attenuation near boundaries. Use `full` only when debugging padding or stencil indexing.

## Forward monitoring

A useful forward callback records scalar summaries rather than full wavefields:

```python
history = []

def record_forward(state: tide.CallbackState) -> None:
    ey = state.get_wavefield("Ey", view="inner")
    history.append(
        {
            "step": state.step,
            "max_abs": float(ey.detach().abs().amax().cpu()),
            "energy": float(ey.detach().square().sum().cpu()),
        }
    )
```

The amplitude should remain finite. After source injection ends, interior energy should leave the domain and be absorbed by CPML. Monotonic growth in a passive homogeneous model is a strong sign of a stability, material, or source-scaling problem.

## Backward monitoring

During a loss backward pass, a callback can reveal whether adjoint energy reaches the intended region and whether a gradient is concentrated only around acquisition points. Use the same axis, view, and scaling conventions as forward plots. A backward field is an adjoint state, not a second physical experiment.

## Callback frequency

Calling Python every time step can serialize device execution and force host synchronization. Start with a sparse frequency that produces roughly 20 to 50 observations over the run. Increase temporal density only around a diagnosed event.

Avoid these operations inside a callback:

- Plotting each frame.
- Writing a full tensor to disk every step.
- Calling `.cpu()` on several large fields.
- Changing tensor shapes, strides, dtype, or device.
- Mutating the model or wavefield in place.

If a full animation is required, store a spatially downsampled field at a sparse cadence and render after propagation.

## A compact snapshot collector

```python
frames = []

def collect(state: tide.CallbackState) -> None:
    if state.step % 25 != 0:
        return
    ey = state.get_wavefield("Ey", view="inner")
    frames.append(ey.detach()[0, ::2, ::2].to("cpu"))
```

The leading index above selects one shot. Adapt it for batched models and confirm the callback-visible shape before slicing.

## Interpreting common patterns

| Observation | Likely question to investigate |
| --- | --- |
| Field is zero everywhere | Source component, location, amplitude, or timing |
| Energy appears on wrong axis | Coordinate order or component staggering |
| Late field grows near an edge | CPML width, material at boundary, or instability |
| Forward field is finite but gradient is not | Objective scaling, adjoint source, or stored-state precision |
| Runtime changes greatly when callback is enabled | Host synchronization or excessive callback cadence |

## Backend constraints

Callback support is part of the backend capability matrix. Some batched-model reference modes do not support callbacks. Requesting a callback can therefore change backend selection or raise under an error fallback policy. Verify the selected path rather than assuming callbacks are free diagnostics.

See the [callback API](/tide-GPR/api/callbacks/) for accessors and [boundaries](/tide-GPR/guides/boundaries/) for interpreting padded views.
