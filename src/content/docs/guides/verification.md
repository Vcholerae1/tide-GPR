---
title: "Verification"
description: "A staged procedure for verifying installation, forward physics, gradients, and backend parity."
---

Use this page after installation and whenever a production configuration changes. The goal is not only to prove that TIDE imports, but to prove that the intended backend, physics, storage policy, and derivatives behave consistently.

## 1. Record the environment

```python
import platform
import torch
import tide
from tide import backend_utils

print("python:", platform.python_version())
print("tide:", tide.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("native backend:", backend_utils.is_backend_available())
print("library path:", backend_utils.get_library_path())
print("cuda arches:", backend_utils.cuda_build_arches())
```

If `get_library_path()` raises because the backend is unavailable, record that result instead of treating the reference backend as native execution.

## 2. Run the minimal forward model

Run the example from [Getting Started](/tide-GPR/getting-started/) and confirm:

- Receiver shape is `[nt, n_shots, n_receivers]`.
- Every receiver value is finite.
- A non-zero source produces a non-zero trace.
- Repeating the deterministic call produces the same result.
- Moving the receiver changes the arrival time as expected.

A shape assertion is useful:

```python
assert receivers.shape == (nt, 1, 1)
assert torch.isfinite(receivers).all()
assert receivers.abs().max() > 0
```

## 3. Check physical travel time

For a homogeneous non-magnetic material, estimate

$$
v \approx c_0 / \sqrt{\epsilon_r}.
$$

Convert source-receiver index separation to meters, divide by `v`, and compare with the first clear arrival. Grid dispersion and source wavelet width prevent exact equality, but a large discrepancy usually indicates an axis, spacing, or unit error.

## 4. Check CPML behavior

Repeat the homogeneous run with two boundary widths or model extents. The direct arrival should remain at the same time. A late edge reflection should decrease or move. Inspect callback snapshots when a trace alone cannot identify the path.

## 5. Check a scalar gradient

```python
epsilon = epsilon.detach().clone().requires_grad_(True)
predicted = operator(tide.EMModel(epsilon, sigma, mu)).receiver_data
loss = predicted.square().mean()
loss.backward()

assert epsilon.grad is not None
assert torch.isfinite(epsilon.grad).all()
assert epsilon.grad.abs().max() > 0
```

This proves autograd connectivity and finite values. It does not prove derivative accuracy.

## 6. Run a directional derivative test

Choose a normalized random direction `v`, then compare autograd with centered finite differences for several `h` values:

```python
v = torch.randn_like(epsilon)
v = v / v.norm()

def evaluate(eps):
    data = operator(tide.EMModel(eps, sigma, mu)).receiver_data
    return data.square().mean()

with torch.no_grad():
    finite_difference = (
        evaluate(epsilon + h * v) - evaluate(epsilon - h * v)
    ) / (2 * h)

autograd_value = torch.sum(epsilon.grad * v)
relative_error = (
    (finite_difference - autograd_value).abs()
    / autograd_value.abs().clamp_min(1.0e-12)
)
```

Inspect the error trend across `h`, not one arbitrary value.

## 7. Check JVP and VJP consistency

```python
cotangent = torch.randn_like(operator(model).receiver_data)
direction = tide.EMDirection(epsilon=torch.randn_like(model.epsilon))

with operator.linearize(model) as linearized:
    jv = linearized.jvp(direction).receiver_data
    jtr = linearized.vjp(cotangent)

left = torch.sum(jv * cotangent)
right = torch.sum(direction.epsilon * jtr.epsilon)
torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-6)
```

Adjust tolerances for dtype, backend, problem size, and compressed storage. If multiple direction components are active, include each inner product on the right.

## 8. Compare runtime paths

On a small case supported by both paths, compare reference and native receiver data and gradients. Force the backend and use `FallbackPolicy.ERROR` so a missing capability cannot silently change the comparison.

The project test suite includes focused checks:

```bash
uv run pytest -q tests/test_forward_physics.py
uv run pytest -q tests/test_gradients.py
uv run pytest -q tests/test_discrete_adjoint.py
uv run pytest -q tests/test_backend_parity_matrix.py
```

CUDA-only cases skip on hosts without CUDA. A skip is not evidence that the CUDA path works on another machine.

## 9. Verify the intended storage policy

Repeat one gradient with full-precision device snapshots and the intended production storage settings. Compare gradient norms, directional derivatives, and one optimizer step. This catches precision or I/O configuration effects that a forward-only check cannot see.

## Acceptance record

A useful verification note includes:

- Exact command or script revision.
- Environment and backend output.
- Model, acquisition, and discretization sizes.
- Expected and observed tensor shapes.
- Travel-time and boundary checks.
- Directional or adjoint error.
- Reference-native tolerances and result.
- Storage policy and peak memory.

Keep this record beside benchmark or inversion results so later changes can be evaluated against the same contract.
