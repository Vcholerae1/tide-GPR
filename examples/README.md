# Examples

These are small, stable examples for learning the public TIDE API. Run them
from the repository root with `uv run python -m ...`.

| Goal | Command |
| --- | --- |
| Minimal 2D TM forward model | `uv run python -m examples.modeling.forward_2d` |
| Minimal shot-batched 2D inversion | `uv run python -m examples.inversion.fwi_2d` |

Long-running comparisons, paper reproductions, parameter sweeps, diagnostics,
and benchmarks live outside this directory. See `experiments/README.md`,
`benchmarks/`, and `tools/`.

Generated files belong under `artifacts/`; local model data belongs under
`data/`. Neither is tracked by Git.
