---
title: "Sources and Receivers"
description: "Tensor layouts, coordinate conventions, component selection, and batching for Maxwell acquisitions."
---

An acquisition connects physical survey geometry to the discrete grid. TIDE represents it with three tensors: source amplitudes, source locations, and receiver locations. Their shot axes must agree, locations use integer grid indices, and the time axis appears in different positions on input and output.

## Tensor contract

For a shared material model:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `source_amplitude` | `[n_shots, n_sources, nt]` | Same floating dtype as model |
| `source_location` | `[n_shots, n_sources, ndim]` | `torch.long` |
| `receiver_location` | `[n_shots, n_receivers, ndim]` | `torch.long` |
| Returned receiver data | `[nt, n_shots, n_receivers]` | Same floating dtype as model |

Coordinates use `[y, x]` in 2D and `[z, y, x]` in 3D. Every coordinate must satisfy `0 <= index < model_size` on its corresponding axis.

:::note[Why time moves]
Source amplitudes keep time last because signal operations normally act on the final dimension. Receiver data keeps time first to match TIDE's propagation and legacy receiver convention. Use named variables and explicit `movedim` calls when integrating code that expects another layout.
:::

## Building a multi-shot line acquisition

The workflow helper turns one-dimensional source and receiver coordinates into solver-ready tensors:

```python
import torch
import tide

n_shots = 12
source_x = torch.arange(n_shots) * 6 + 12
receiver_x = torch.arange(80) + 8

acquisition = tide.workflow.line_acquisition_2d(
    source_x,
    receiver_x,
    source_depth=5,
    receiver_depth=5,
    receiver_mode="shared",
)

wavelet = tide.ricker(freq=6.0e8, length=600, dt=4.0e-11)
source_amplitude = tide.workflow.expand_source_amplitude(
    wavelet,
    n_shots=n_shots,
)

print(acquisition.source_location.shape)    # [12, 1, 2]
print(acquisition.receiver_location.shape)  # [12, 80, 2]
print(source_amplitude.shape)                # [12, 1, 600]
```

With `receiver_mode="shared"`, every shot records the same receiver line. With `receiver_mode="paired"`, receiver coordinates are paired by shot, which is useful for moving source-receiver configurations.

## Multiple sources per shot

The second source-amplitude axis allows several injection points to share one shot. Their amplitudes are added during the same propagation:

```python
source_amplitude = wavelet.repeat(n_shots, 2, 1)
source_amplitude[:, 1].mul_(-1.0)

source_location = torch.empty(n_shots, 2, 2, dtype=torch.long)
source_location[:, 0, 0] = source_depth
source_location[:, 0, 1] = source_x
source_location[:, 1, 0] = source_depth
source_location[:, 1, 1] = source_x + 1
```

This pattern can represent a simple discrete dipole or a spatially distributed source. The effective source depends on field staggering and selected source component, so verify polarity and placement in a homogeneous model.

## Source-free continuation

For propagation without a new source, pass `source_amplitude=None` and provide `nt` on `Experiment`:

```python
experiment = tide.Experiment(
    tide.Acquisition(source_location=None, receiver_location=receiver_location),
    source_amplitude=None,
    nt=200,
)
```

`nt` is otherwise inferred from the final source-amplitude dimension.

## Component selection in 3D

`Experiment.source_component` selects the injected electric or magnetic component, and `receiver_component` selects the recorded component. Their defaults are `"ey"`. The component name changes the physical experiment, not only the output label. Place sources and receivers with the staggered component locations in mind, especially when comparing cross-components.

## Converting physical coordinates

If the physical origin is `origin` and spacing is `spacing`, a basic cell-index conversion is

```python
indices = torch.round((coordinates - origin) / spacing).to(torch.long)
```

For anisotropic spacing, apply the conversion independently per axis. Decide once whether a coordinate refers to cell centers or a particular staggered field location. Record that convention beside the survey loader. Silent half-cell shifts can create phase errors that resemble modeling or inversion problems.

## Shot batching

The shot axis is independent across propagations for a shared model. Split it when all shots do not fit in memory:

```python
for shot_indices in tide.workflow.split_shots(n_shots, batch_size=3):
    batch = tide.workflow.take_shot_batch(
        source_amplitude=source_amplitude,
        source_location=acquisition.source_location,
        receiver_location=acquisition.receiver_location,
        shot_indices=shot_indices,
    )
    # Rebuild only Experiment with batch.source_amplitude and batch locations.
```

`tide.workflow.run_shot_batches` performs this pattern and concatenates receiver data along the correct shot axis while preserving autograd connections.

## Common failures

### Mismatched shot counts

All three acquisition tensors must describe the same number of shots. A shared receiver layout is still expanded to a shot-indexed tensor by the workflow helper.

### Wrong coordinate order

An `[x, y]` tensor may remain in bounds and still place the acquisition incorrectly. Plot or print several discrete coordinates before a long run.

### Wrong location dtype

Locations are indices and must use an integer dtype. Do not place them on a floating tensor merely to match material dtype.

### Source outside useful bandwidth

A source can be numerically valid but poorly resolved by the grid. Inspect its spectrum and compare the highest useful frequency with the grid-resolution study.

### Receiver traces compared on different time axes

Observed and predicted data must use the same sampling interval, sample count, timing convention, and preprocessing. TIDE's internal CFL sub-stepping does not change the returned user time axis.
