# Module: tide.resampling

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
