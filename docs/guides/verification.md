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
uv run python examples/example_gradient_dot_fd_validation.py --backend c
```

Expected:

- Taylor remainder decreases with step size
- directional finite-difference checks stay close to adjoint gradients

Notes:

- the script auto-selects CUDA when available
- append `--device cpu` if you want the same check on CPU

## Runtime Notes

- this command set is intended for environments where the documented examples can access the required backend
- CUDA-only verification steps should be skipped on CPU-only hosts
- larger inversion examples are reference workflows, not required smoke tests
