# Examples

This is the single home for maintained TIDE examples, from minimal tutorials
to reproducible research workflows. Run commands from the repository root.

| Goal | Command |
| --- | --- |
| Minimal 2D TM forward model | `uv run python -m examples.modeling.forward_2d` |
| Minimal shot-batched 2D inversion | `uv run python -m examples.inversion.fwi_2d` |
| 3D HashGrid implicit full-waveform inversion | `uv run --extra experiments python -m examples.inversion.implicit_3d --help` |
| 4-GPU single-band HashGrid 3D inversion | `uv run --extra experiments torchrun --standalone --nproc-per-node=4 -m examples.inversion.implicit_3d --distributed --expected-world-size=4 --frequency-mhz=900 --output-dir=artifacts/tide-runs/hydro_3d_hashgrid_4gpu` |
| Paper tutorials | See `examples/paper/README.md` |
| Implicit FWI methods | See `examples/implicit_fwi/README.md` |

Only examples documented here are maintained. One-off studies, superseded
comparisons, local datasets, and generated results do not belong in this tree.
Stable performance benchmarks live under `benchmarks/`.

Generated files belong under `artifacts/`; local model data belongs under
`data/`. Neither is tracked by Git.
