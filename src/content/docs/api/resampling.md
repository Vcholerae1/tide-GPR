---
title: "Module: tide.resampling"
description: "Upsample sources and downsample receiver traces around internal CFL sub-stepping."
---

Signal resampling utilities used by CFL-driven internal sub-stepping.

## Functions
- cosine_taper_end
- zero_last_element_of_final_dimension
- upsample
- downsample
- downsample_and_movedim

## upsample

Low-pass upsampling on the last dimension.

Typical use:
- Called internally when CFL requires internal dt < user dt.
- Can be used manually for source preprocessing.

## downsample

Frequency-limited downsampling on the last dimension.

Typical use:
- Called internally to bring receiver traces back to user sampling interval.

## downsample_and_movedim

Convenience wrapper:
- expects receiver_amplitudes shaped [nt, n_shots, n_receivers]
- processes time on the last axis internally
- returns [n_shots, n_receivers, nt_downsampled]

## Signatures

```python
upsample(signal, step_ratio, freq_taper_frac=0.0,
         time_pad_frac=0.0, time_taper=False)

downsample(signal, step_ratio, freq_taper_frac=0.0,
           time_pad_frac=0.0, time_taper=False, shift=0.0)
```

Both functions operate on the final tensor dimension and preserve all leading
dimensions. `step_ratio=1` is the identity case. A larger ratio changes only
the time-axis length.

## Round-trip check

```python
fine = tide.upsample(source, step_ratio=3)
recovered = tide.downsample(fine, step_ratio=3)
torch.testing.assert_close(recovered, source, rtol=1e-4, atol=1e-6)
```

Use a tolerance appropriate for signal bandwidth and taper settings. A signal
with energy near the new Nyquist limit cannot be downsampled without loss.

`freq_taper_frac` softens the spectral cutoff. `time_pad_frac` reduces circular
FFT interaction between trace ends. `time_taper=True` applies an end taper
before padding. These choices affect endpoint samples and should stay
consistent between observed and predicted-data preprocessing.
