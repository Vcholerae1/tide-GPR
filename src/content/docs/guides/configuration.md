---
title: "Configuration Reference"
description: "Execution, storage, numerical, callback, and dispersive controls for structured Maxwell operators."
---

TIDE groups settings by ownership. `Discretization` defines numerical space and time, `Experiment` defines acquisition and signal handling, `ExecutionOptions` defines backend policy, and `StorageOptions` defines derivative snapshot placement. Keeping these groups explicit makes a run reproducible and prevents incompatible settings from being scattered across solver calls.

## Discretization

```python
discretization = tide.Discretization(
    spacing=(0.02, 0.02),
    dt=4.0e-11,
    stencil=4,
    boundary=tide.CPML((12, 12, 12, 12)),
    max_velocity=None,
)
```

| Field | Meaning | Guidance |
| --- | --- | --- |
| `spacing` | Scalar or per-axis cell spacing in meters | Match model axis order |
| `dt` | User source and receiver sampling interval in seconds | Internal sub-stepping may reduce it |
| `stencil` | Finite-difference order: 2, 4, 6, or 8 | Validate dispersion versus cost |
| `boundary` | Per-side CPML width | Test late-time reflection |
| `max_velocity` | Optional velocity bound for planning | Must cover every model used with the operator |

Changing spacing or stencil changes the discrete physical problem. Treat these values as part of result provenance, not only performance knobs.

## Experiment

```python
experiment = tide.Experiment(
    tide.Acquisition(source_location, receiver_location),
    source_amplitude,
    source_component="ey",
    receiver_component="ey",
    frequency_taper_fraction=0.0,
    time_padding_fraction=0.0,
    time_taper=False,
)
```

`nt` is inferred from `source_amplitude.shape[-1]`. For source-free propagation, set `source_amplitude=None` and provide `nt` explicitly.

Frequency tapering and time padding are signal-conditioning controls used by internal resampling. They can reduce FFT edge artifacts, but they do not repair an under-resolved grid or inconsistent observed data. Apply the same timing assumptions throughout the workflow.

## Execution policy

```python
execution = tide.ExecutionOptions(
    backend=tide.BackendPreference.AUTO,
    fallback=tide.FallbackPolicy.ERROR,
    reference_mode="eager",
    n_threads=None,
)
```

| Field | Values | Effect |
| --- | --- | --- |
| `backend` | `AUTO`, `REFERENCE`, `NATIVE` | Preferred implementation family |
| `fallback` | `ERROR`, `REFERENCE` | Behavior when native capability is unavailable |
| `reference_mode` | `eager`, `jit`, `compile` | Python reference execution mode |
| `n_threads` | Positive integer or `None` | Native CPU thread request |

Use `fallback=ERROR` for benchmarks and production runs that require a specific backend. Use `REFERENCE` while developing portable examples. A fallback can preserve functionality, but it changes performance and may change supported storage or callback combinations.

## Snapshot storage

```python
storage = tide.StorageOptions(
    mode=tide.StorageMode.AUTO,
    path="./tide-storage",
    compression="bf16",
    bytes_limit_device=4 * 1024**3,
    bytes_limit_host=24 * 1024**3,
    chunk_steps=0,
)
```

| Mode | Location | Main trade-off |
| --- | --- | --- |
| `DEVICE` | Compute device | Fastest, highest device memory use |
| `CPU` | Host memory | Lower VRAM, transfer overhead |
| `DISK` | Files under `path` | Lowest memory pressure, highest latency |
| `NONE` | No stored snapshots | Only valid for operations that do not need them |
| `AUTO` | Selected from byte limits | Convenient, but limits must reflect the actual machine |

BF16 compression reduces snapshot traffic and capacity at the cost of stored-state precision. It does not change the model or arithmetic dtype. Verify gradient impact for the actual objective before enabling it broadly.

## Gradient sampling

`model_gradient_sampling_interval` belongs to the Maxwell operator:

```python
operator = tide.MaxwellTM(
    discretization,
    experiment,
    execution=execution,
    storage=storage,
    model_gradient_sampling_interval=2,
)
```

A larger interval reduces the number of time samples used for model-gradient accumulation. This can reduce work and storage pressure, but it is an approximation knob. Compare gradients and inversion behavior against interval 1 before increasing it.

## Callbacks and observers

Forward and backward callbacks expose `CallbackState` at a chosen frequency. Callbacks execute Python code during propagation, so frequent tensor transfers, plotting, and synchronization can dominate runtime. Collect compact device-side statistics when possible and move only summaries to the host.

Not every backend and batched-model mode supports callbacks. Backend selection includes callback capability in its compatibility decision.

## Debye dispersion

```python
dispersion = tide.DebyeDispersion(
    delta_epsilon=delta_epsilon,
    tau=tau,
)
model = tide.EMModel(epsilon, sigma, mu, dispersion=dispersion)
```

`delta_epsilon` and `tau` must be compatible with the material shape and time step. The relaxation time must remain larger than the integration step required by the implementation. Validate a single homogeneous dispersive material against its expected attenuation and phase response before combining it with inversion.

## Reproducible configuration record

For every reported run, record:

- TIDE, PyTorch, Python, compiler, and CUDA versions.
- Device name and native backend availability.
- Model shape, dtype, and batch shape.
- Spacing, `dt`, stencil, CPML widths, and CFL step ratio.
- Source frequency, sample count, components, and geometry.
- Backend, fallback, storage, compression, and gradient sampling settings.

This information is usually more useful than a single runtime number or loss curve.
