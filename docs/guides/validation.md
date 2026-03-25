# Validation and Stability

TIDE includes built-in checks for stability and parameter consistency.

## CFL Condition

Use cfl_condition(grid_spacing, dt, max_vel) to evaluate whether dt is stable.

Return values:
- inner_dt: stable internal time step
- step_ratio: number of internal steps per external step

maxwelltm and maxwell3d call this internally, but explicit checks are useful when designing survey parameters.

## Sampling and Resampling

Resampling helpers provide deterministic signal transforms used in CFL sub-stepping:
- upsample: low-pass upsample on last time axis
- downsample: low-pass downsample on last time axis
- downsample_and_movedim: convenience transform for receiver tensors

Use freq_taper_frac and time_pad_frac to reduce FFT edge artifacts.

## Padding Helpers

- create_or_pad: create zeros or pad existing tensors
- reverse_pad: convert natural pad order into torch.nn.functional.pad order
- zero_interior: keep CPML activity near boundaries while clearing non-PML interior

## Validation Helpers

- validate_model_gradient_sampling_interval
- validate_freq_taper_frac
- validate_time_pad_frac

These are simple but effective front-line guards for user-facing numeric controls.
