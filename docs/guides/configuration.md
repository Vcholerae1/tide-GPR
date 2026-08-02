# Configuration Reference

This page centralizes the runtime controls that most strongly affect correctness, memory use, and performance.

## Storage Controls

- `storage_mode`: `device`, `cpu`, `disk`, `none`, or `auto`
- `storage_compression`: `none` or `bf16`
- `storage_bytes_limit_device`
- `storage_bytes_limit_host`
- `storage_chunk_steps`

Use `storage_mode="device"` first for speed, then step down to `cpu`, `disk`, or `auto` when memory pressure forces it.

## Backend Controls

- `python_backend=False` for the native path when available
- `python_backend=True`, `"eager"`, `"jit"`, or `"compile"` for the reference path
- `python_backend="native"` to require the native path
- `fallback="reference"` (default) or `fallback="error"`
- `tide.backend_utils.is_backend_available()` to check native backend visibility

Every public Maxwell entry point first compiles a shared `SimulationPlan`. The
plan makes backend selection and fallback explicit; `fallback="error"` prevents
an unavailable native library from silently switching to Python.

## Callback Controls

- `forward_callback`
- `backward_callback`
- `callback_frequency`

Callbacks are useful for diagnostics and monitoring, but they add overhead and should stay lightweight.

## Numerical Controls

- `stencil`
- `pml_width`
- `freq_taper_frac`
- `time_pad_frac`
- `time_taper`
- `model_gradient_sampling_interval`

These controls affect stability, memory use, resampling behavior, and numerical fidelity.

## 3D-Specific Controls

- `source_component`
- `receiver_component`

Use these when the injected or recorded field component matters for the physical setup.

## Dispersive Materials

- `DebyeDispersion(delta_epsilon=..., tau=...)`
- enforce `dt < min(tau)`

Always validate dispersive workflows on a small case before scaling them up.
