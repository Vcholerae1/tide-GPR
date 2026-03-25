# Module: tide

Top-level exports from src/tide/__init__.py.

Versioned package entry point that re-exports high-value modules and functions.

## Typical Import Pattern

```python
import tide

*_, receivers = tide.maxwelltm(...)
wavelet = tide.ricker(...)
```

## Modules
- callbacks
- resampling
- cfl
- padding
- validation
- maxwell
- staggered
- utils
- wavelets

## Classes
- MaxwellTM
- Maxwell3D
- CallbackState
- DebyeDispersion

## Types
- Callback

## Functions
- maxwelltm
- maxwell3d
- create_callback_state
- upsample
- downsample
- downsample_and_movedim
- cfl_condition
- validate_model_gradient_sampling_interval
- validate_freq_taper_frac
- validate_time_pad_frac
- create_or_pad
- zero_interior
- reverse_pad
- ricker

## Notes

- maxwelltm and maxwell3d are the recommended functional APIs for most workflows.
- MaxwellTM and Maxwell3D classes are convenient wrappers when model parameters are reused across calls.
- Validation and padding helpers are stable utilities used internally and are also safe for user preprocessing.
