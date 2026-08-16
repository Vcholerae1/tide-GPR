---
title: "Module: tide.wavelets"
description: "Generate source wavelets with explicit sampling and timing controls."
---

Wavelet generators used for source design in FDTD propagation.

## Functions

- `ricker`
- `gaussian`
- `gaussian_derivative`
- `morlet`
- `sine_burst`

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

## Other pulse shapes

All generators use the same `freq`, `length`, `dt`, `peak_time`, `dtype`, and
`device` arguments as `ricker`. When omitted, `peak_time` defaults to `1/freq`.

```python
gaussian(freq, length, dt, peak_time=None, dtype=None, device=None)
gaussian_derivative(freq, length, dt, peak_time=None, dtype=None, device=None)
morlet(freq, length, dt, peak_time=None, cycles=3.0, dtype=None, device=None)
sine_burst(freq, length, dt, peak_time=None, cycles=3.0, dtype=None, device=None)
```

`gaussian` returns a unit Gaussian pulse. `gaussian_derivative` returns its
first derivative normalized to unit magnitude. `morlet` returns a real,
Gaussian-windowed cosine. `sine_burst` returns a cosine limited to `cycles`
periods by a Hann envelope.

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

Frequency, sample interval, length, and cycle count must be positive. Match
dtype and device to the material tensors to avoid an implicit copy during setup.
