# Verification

Use this page to confirm that your installation and chosen runtime path are behaving as expected.

## Backend Availability

```python
from tide import backend_utils

print("backend available:", backend_utils.is_backend_available())
print("library path:", backend_utils.get_library_path())
```

## Minimal Forward Smoke Test

Run the 2D example from `docs/getting-started.md` and confirm the receiver tensor has shape `[nt, n_shots, n_receivers]`.

## Gradient Sanity

Run:

```bash
uv run pytest tests/test_gradients.py
```

Expected:

- finite-difference and backend-parity checks pass where supported
- gradient sampling interval regressions stay covered

Notes:

- some cases are skipped automatically when CUDA is unavailable
- use `-k` to narrow to a specific gradient scenario during local debugging

## Runtime Notes

- this command set is intended for environments where the documented examples can access the required backend
- CUDA-only verification steps should be skipped on CPU-only hosts
- larger inversion examples are reference workflows, not required smoke tests
