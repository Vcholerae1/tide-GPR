# Module: tide.validation

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
