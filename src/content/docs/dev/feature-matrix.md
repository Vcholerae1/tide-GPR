---
title: "Execution Capability Matrix"
description: "The stable backend combinations exposed by TIDE's live planning contract."
---

The executable source of truth is `tide.core.backends.backend_capabilities()`. The table below mirrors the current rows. A new combination is supported only after code, capability selection, numerical tests, and public documentation agree.

## Operation vocabulary

| Operation | Meaning |
| --- | --- |
| `forward` | Nonlinear propagation $F(m)$ |
| `jvp` | Tangent action $J(m)v$ |
| `vjp` | Adjoint action $J(m)^\top r$ |
| `second_vjp` | Nonlinear second-order action $(DJ(m)[v])^\top r$ |

`forward` and `vjp` share capability rows because standard reverse-mode differentiation begins with the stored or recomputed forward trajectory.

## Stable rows

All rows support CPU and CUDA devices, float32 and float64 dtypes, subject to native library availability on the selected machine.

| Backend | Dimension | Operations | Storage modes | Gradient targets | Callbacks | Reusable background |
| --- | --- | --- | --- | --- | --- | --- |
| Reference | TM2D | forward, vjp | auto, device, CPU, disk, none | epsilon, sigma, mu, perturbation, source, state | yes | no |
| Reference | TM2D | jvp | auto, device, CPU, disk, none | epsilon, sigma, mu, perturbation, source, state | no | no |
| Reference | TM2D | second_vjp | device, CPU, disk | epsilon, sigma, mu, perturbation, source, state | no | no |
| Reference | EM3D | forward, vjp | auto, device, CPU, disk, none | epsilon, sigma, mu, perturbation, source, state | yes | no |
| Reference | EM3D | jvp | device, none | epsilon, sigma, mu, perturbation, source, state | no | no |
| Reference | EM3D | second_vjp | device | epsilon, sigma, mu, perturbation, source, state | no | no |
| Native | TM2D | forward, vjp | auto, device, CPU, disk, none | epsilon, sigma, source | yes | no |
| Native | TM2D | jvp | auto, device, CPU, disk, none | epsilon, sigma, perturbation | no | yes |
| Native | TM2D | second_vjp | device, CPU, disk | epsilon, sigma | no | no |
| Native | EM3D | forward, vjp | auto, device, CPU, disk, none | epsilon, sigma, source | yes | no |
| Native | EM3D | jvp | device, none | epsilon, sigma, perturbation | no | no |
| Native | EM3D | second_vjp | device | epsilon, sigma | no | no |

## Gradient targets

- `epsilon`, `sigma`, and `mu` refer to background material tensors.
- `perturbation` refers to `EMDirection` fields used by JVP and differentiated tangent workflows.
- `source` refers to source-amplitude samples.
- `state` refers to initial field or derivative-state tensors.

A capability row lists the largest target set that the backend family can accept. Individual physics adapters may impose a narrower operation-specific rule. For example, TM2D JVP does not currently accept a `mu` direction.

## Storage interpretation

Storage cells describe accepted public policy for that operation:

- `auto` resolves from configured device and host byte limits.
- `device` stores eligible trajectory state on the compute device.
- `cpu` stores on the host.
- `disk` stores below the configured path.
- `none` avoids snapshot allocation where the operation permits it.

Native EM3D JVP accepts device or no snapshot storage. Native EM3D second VJP requires device storage. Reference capability is not universally broader: it intentionally follows the same EM3D derivative storage boundaries for operations that lack another implemented path.

## Fallback behavior

Backend selection evaluates the complete plan. A request can be rejected because of operation, gradient target, storage, callbacks, device, dtype, dispersion, batched-model layout, or native availability.

- `FallbackPolicy.ERROR` raises with the unsupported reason.
- `FallbackPolicy.REFERENCE` selects the reference backend only if a reference row covers the full request.
- `BackendPreference.NATIVE` combined with error fallback is the appropriate choice when benchmarking native execution.

No solver adapter should invent a fallback after the central decision.

## Runtime-only checks

The matrix captures stable declarative capability. Some conditions remain dependent on the loaded binary or runtime object:

- Required native ABI symbol exists.
- CUDA architecture is compatible with the built library.
- Tensor layout is contiguous where the ABI requires it.
- Dispersion and snapshot representation are compatible.
- `torch.func` transform state can use the selected adapter.

These checks must honor the same fallback policy and report a descriptive reason.

## Querying the live matrix

```python
import tide
from tide.core.backends import backend_capabilities

for preference in (
    tide.BackendPreference.REFERENCE,
    tide.BackendPreference.NATIVE,
):
    print(preference.value)
    for row in backend_capabilities(preference).matrix:
        print(row)
```

When this output and the rendered table disagree, treat the Python output as authoritative and update the documentation in the same change.
