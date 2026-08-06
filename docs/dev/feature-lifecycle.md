# Feature Lifecycle

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
