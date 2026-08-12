---
title: "Feature Lifecycle"
description: "Promote solver capabilities from research code to the supported API."
---

Every new execution feature follows the same promotion path. This keeps the
number of backend combinations bounded as the solver grows.

## Required before merge

1. State whether the feature is stable, experimental, or explicitly rejected.
2. Add its supported cells to the capability matrix: dimension, operation,
   device, dtype, compute mode, storage mode, callback support, and gradient
   support.
3. Provide a reference implementation or a test that proves the unsupported
   combination fails with a useful error.
4. Add numerical parity tests for every newly supported native cell.
5. Document the public behavior, limitations, and fallback policy.
6. Add a small benchmark when the feature changes a kernel or memory path.

## Promotion rule

An experiment can move into the stable API only after it survives the CPU
reference suite, the native parity suite, and the relevant storage/precision
matrix. Large datasets and generated results are inputs to experiments, not
package contents.

## Review questions

- Does this add a public option or can it remain an internal plan field?
- Which existing capability cells change?
- What happens when the native backend is unavailable?
- Can the change be rolled back without changing stored data?
- Does it introduce a new dependency direction or duplicate an existing
  normalization path?

## Documentation visibility

Unstable runtime paths are not published as website routes. Keep design notes,
hardware profiles, generated reports, and one-off benchmark narratives outside
the documentation content collection.

The website documents a feature only after it has:

1. A supported public name and ownership boundary.
2. An executable capability row.
3. A reference behavior or explicit unsupported error.
4. Numerical and lifecycle tests.
5. User guidance that explains inputs, outputs, failure modes, and verification.

Developer documentation may explain the promotion process, but it must not
advertise an unreleased path as something users can select.

## Review sequence

Review a feature in this order:

1. Public contract and mathematical meaning.
2. Reference implementation and observable result.
3. Capability and fallback behavior.
4. Native implementation and parity.
5. Storage and resource lifetime.
6. Workflow integration.
7. Documentation and focused benchmark.

This order prevents an optimized kernel from defining the API by accident.

## Removal rule

When a feature is rejected or withdrawn, remove its capability rows, public
exports, call sites, tests that imply support, and website references together.
Keep historical investigation in version control or an external research
record, not in the stable navigation.
