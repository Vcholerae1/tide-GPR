---
title: "Validation and Stability"
description: "CFL stability, convergence tests, derivative checks, and operational validation for Maxwell simulations."
---

A successful kernel call proves only that inputs passed structural validation. Trust in a simulation comes from several independent checks: stable time stepping, grid convergence, boundary tests, physical benchmarks, derivative tests, and backend parity.

## CFL condition

Explicit FDTD updates have a maximum stable time step determined by wave speed, spacing, dimension, and stencil. TIDE calculates a stable internal interval and an integer sub-step ratio:

```python
from tide.cfl import cfl_condition

inner_dt, step_ratio = cfl_condition(
    grid_spacing=(0.02, 0.02),
    dt=4.0e-11,
    max_vel=299_792_458.0,
)
print(inner_dt, step_ratio)
```

If `step_ratio` is larger than one, TIDE upsamples the source, advances with `inner_dt`, then downsamples receiver data to the requested sampling interval. This preserves the public time axis but increases runtime.

CFL stability does not guarantee accuracy. A stable grid can still have unacceptable numerical dispersion.

## Resolution convergence

Run the same physical problem at two or more spatial resolutions. When spacing changes, convert physical source and receiver coordinates to the new indices and preserve the same source and receiver time axis. Compare:

- Direct-arrival time.
- Phase over the useful bandwidth.
- Peak and integrated amplitude.
- Reflected-event timing.
- Receiver residual norm after interpolation to a common representation.

A result is grid-converged only relative to a stated tolerance and quantity of interest.

## Boundary test

In a homogeneous model, any late coherent return from the edge is numerical. Compare at least two CPML widths or model extents. The direct arrival should remain fixed, while a boundary return should move or decrease.

## Analytic and physical checks

Simple cases are more informative than a visually complicated synthetic model:

- Homogeneous-medium travel time tests velocity and coordinate conversion.
- Interface reflection tests impedance contrast, polarity, and timing.
- Conductive homogeneous media test attenuation trends.
- Debye materials test frequency-dependent phase and attenuation.
- Symmetric geometry tests expected trace symmetry.

Use float64 for strict small-scale numerical studies when the selected backend supports it. Do not infer production float32 error only from one dtype comparison.

## Directional derivative test

For a scalar objective $\Phi$ and model-space direction $v$:

```python
h = 1.0e-3
with torch.no_grad():
    plus = objective(model_plus_hv)
    minus = objective(model_minus_hv)
finite_difference = (plus - minus) / (2 * h)

autograd_directional = torch.sum(epsilon.grad * v_epsilon)
```

Evaluate several logarithmically spaced `h` values. A correct derivative normally shows a region where the discrepancy decreases with `h`. Very small steps eventually suffer floating-point cancellation.

## Adjoint dot-product test

For arbitrary model direction $v$ and receiver cotangent $r$, verify

$$
\langle Jv, r\rangle \approx \langle v, J^\top r\rangle.
$$

```python
with operator.linearize(model) as linearized:
    jv = linearized.jvp(direction).receiver_data
    jtr = linearized.vjp(receiver_cotangent)

left = torch.sum(jv * receiver_cotangent)
right = torch.sum(direction.epsilon * jtr.epsilon)
```

Include every active material component on the right-hand side. Match dtype, backend, storage, and compression to the intended workload.

## Backend parity

Run the same small deterministic case on reference and native paths. Compare receiver traces, final state, and gradients with tolerances appropriate for dtype and storage compression. Parity isolates implementation differences from modeling choices.

## Signal resampling checks

`upsample` and `downsample` operate on the last axis. Frequency taper and time padding reduce spectral edge artifacts but can alter endpoints. For a representative wavelet:

1. Upsample and downsample with the intended ratio.
2. Compare the recovered waveform in time and frequency.
3. Inspect the final samples for wraparound or taper artifacts.
4. Repeat with the shortest trace used by the workflow.

## Validation order

Use the cheapest discriminating test first:

1. Shapes, bounds, dtype, and units.
2. CFL ratio and finite values.
3. Homogeneous travel time.
4. Boundary reflection.
5. Grid or stencil convergence.
6. Directional derivative and adjoint identity.
7. Backend and storage parity.
8. Full inverse workflow on synthetic observed data.

Passing a later test does not replace an earlier one. An inversion can reduce its loss while fitting a systematic modeling error.
