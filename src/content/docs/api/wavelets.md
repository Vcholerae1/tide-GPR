---
title: "Module: tide.wavelets"
description: "Generate source wavelets with explicit sampling and timing controls."
---

Wavelet generators used for source design in FDTD propagation.

## Functions
- ricker

## ricker

Signature:

```python
ricker(freq, length, dt, peak_time=None, dtype=None, device=None)
```

Parameters:
- freq: dominant frequency in Hz, must be > 0
- length: number of samples, must be > 0
- dt: time sample interval in seconds, must be non-zero
- peak_time: optional peak time, default is 1/freq
- dtype/device: optional torch dtype and device

Returns:
- torch.Tensor with shape [length]

Typical usage:

```python
src = tide.ricker(8e8, length=1000, dt=4e-11).view(1, 1, -1)
```

## Definition and timing

The returned source is

$$
w(t)=\left(1-2\pi^2f^2(t-t_0)^2\right)
\exp\left(-\pi^2f^2(t-t_0)^2\right).
$$

`freq` is the dominant frequency and `peak_time` is $t_0$. The default
`peak_time=1/freq` leaves a causal-looking lead-in, but it does not guarantee
that the finite trace contains negligible endpoint energy. Choose `length` and
timing from the full source spectrum required by the experiment.

## Expanding to shots

```python
wavelet = tide.ricker(
    freq=8.0e8,
    length=600,
    dt=4.0e-11,
    dtype=epsilon.dtype,
    device=epsilon.device,
)
source_amplitude = tide.workflow.expand_source_amplitude(
    wavelet,
    n_shots=16,
    n_sources=1,
)
```

The generator returns one dimension. Maxwell experiments require
`[shots, sources, nt]`, so reshape or use `expand_source_amplitude`.

## Validation

Frequency and length must be positive. `dt` must be non-zero. Match dtype and
device to the material tensors to avoid an implicit copy during setup.
