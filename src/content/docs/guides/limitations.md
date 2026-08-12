---
title: "Limitations"
description: "Current physical, derivative, backend, storage, and workflow boundaries in TIDE."
---

TIDE supports differentiable 2D TM and full 3D Maxwell propagation, but support is not identical across every derivative target, storage mode, backend, callback, and material model. Treat the selected combination as the unit of support.

## Physical scope

- `MaxwellTM` represents 2D transverse magnetic propagation with `Ey`, `Hx`, and `Hz` fields.
- `Maxwell3D` represents six-component Cartesian Maxwell propagation.
- Source and receiver positions are integer grid indices. Fractional-location interpolation is not part of the structured public acquisition API.
- CPML is the documented absorbing boundary. Other physical boundary conditions require separate validation and are not implied by a zero CPML width.
- The optional dispersion model is single-pole Debye dispersion, not a general constitutive library.

## Material derivatives

The structured derivative API accepts `epsilon`, `sigma`, and `mu` targets, but native derivative capability is narrower than reference capability.

- Native forward VJP supports `epsilon`, `sigma`, and source gradients.
- Native JVP supports perturbations in `epsilon` and `sigma`.
- Native second VJP supports `epsilon` and `sigma`.
- Requests involving `mu`, initial state, or unsupported source derivatives require a compatible reference path or raise under `FallbackPolicy.ERROR`.
- TM2D tangent propagation for a `mu` direction is not available.

Always make derivative targets explicit when a workflow uses only a subset. This reduces ambiguity in backend planning and avoids retaining gradients for unused fields.

## Storage restrictions

Storage support depends on operation and dimension:

- Forward and VJP paths accept automatic, device, CPU, disk, or no snapshot storage where the backend row allows it.
- Native TM2D JVP supports automatic, device, CPU, disk, and none.
- Native EM3D JVP uses device or none.
- Native TM2D second VJP supports device, CPU, or disk.
- Native EM3D second VJP requires device storage.

Reference capability is broader for several combinations, but a reference fallback can be substantially slower.

## Callbacks

Callbacks are supported on forward entry points. JVP and second VJP capability rows do not advertise callback support. Batched-model and reference execution combinations may be narrower. Requesting callbacks is part of backend selection, not an independent afterthought.

## Dispersion

Native dispersive forward propagation is inference-oriented. Combining dispersion with gradient targets can route to the reference backend when allowed or raise when fallback is disabled. Validate `dt` against relaxation times and compare a homogeneous dispersive response before using spatially varying parameters.

## Numerical limitations

- CFL sub-stepping protects stability but does not guarantee adequate spatial resolution.
- Higher stencil order reduces some dispersion error but does not remove the need for convergence tests.
- CPML is not perfectly reflection-free for every angle, material, and bandwidth.
- BF16 snapshot compression and sparse model-gradient sampling approximate derivative state.
- Float32 and float64 results can diverge more in long, high-Q, or ill-conditioned workflows.

## Inverse-problem limitations

Differentiability does not make an inverse problem identifiable. Limited bandwidth, sparse aperture, source uncertainty, parameter cross-talk, and noise can produce non-unique models. TIDE does not choose regularization, constraints, preprocessing, or stopping rules on the user's behalf.

Graph-space and Sinkhorn receiver losses are computationally heavier than MSE and introduce their own scaling choices. They should be treated as objective functions to validate, not automatic remedies for cycle skipping.

## Operational boundaries

- Large 3D derivative runs can exceed device and host memory even when forward inference fits.
- Disk-backed snapshots depend strongly on local filesystem performance.
- `torch.compile` startup and recompilation costs can dominate short reference runs.
- Native library availability depends on platform, compiler, CUDA toolkit, and built architectures.
- Hardware-specific benchmark numbers are not portable evidence of application performance.

## Before relying on a configuration

1. Query the live capability matrix with `backend_capabilities`.
2. Run a small forward case with `FallbackPolicy.ERROR` on the intended backend.
3. Run a directional derivative or adjoint test for the exact derivative targets.
4. Compare storage compression and sampling choices against a full-precision baseline.
5. Measure memory and runtime with the intended shot batch size.
6. Record every numerical and execution option with the result.

The [execution capability matrix](/tide-GPR/dev/feature-matrix/) is the detailed contract. The [verification guide](/tide-GPR/guides/verification/) provides concrete checks.
