---
title: "Module: tide.validation"
description: "Validate public numerical options before simulation planning."
---

Validation helpers for user-facing parameters.

## Functions
- validate_model_gradient_sampling_interval
- validate_freq_taper_frac
- validate_time_pad_frac

## validate_model_gradient_sampling_interval

- Input must be int
- Value must be >= 0

## validate_freq_taper_frac

- Converts input to float
- Requires 0.0 <= value <= 1.0

## validate_time_pad_frac

- Converts input to float
- Requires 0.0 <= value <= 1.0

These helpers are used by maxwelltm and maxwell3d before propagation starts.

## Return and failure behavior

Each function returns the normalized value. Invalid types or ranges raise
before propagation:

| Function | Accepted value |
| --- | --- |
| `validate_model_gradient_sampling_interval` | Integer greater than or equal to zero |
| `validate_freq_taper_frac` | Float in `[0, 1]` |
| `validate_time_pad_frac` | Float in `[0, 1]` |

```python
interval = tide.validate_model_gradient_sampling_interval(2)
taper = tide.validate_freq_taper_frac(0.1)
padding = tide.validate_time_pad_frac(0.25)
```

These guards validate parameter domains, not numerical suitability. A taper of
one is accepted by range validation but may be inappropriate for a particular
signal. Pair them with waveform and derivative checks.
