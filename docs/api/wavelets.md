# Module: tide.wavelets

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
