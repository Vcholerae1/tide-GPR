# TIDE Documentation

This directory is the entry point for user guides, API reference, runnable examples, and developer notes.

Recommended reading order:
1. overview.md
2. getting-started.md
3. guides/modeling.md
4. guides/sources-receivers.md
5. guides/storage.md
6. api/index.md

## Start Here
- overview.md
- getting-started.md

## Guides
- guides/modeling.md
- guides/sources-receivers.md
- guides/boundaries.md
- guides/storage.md
- guides/callbacks.md
- guides/inversion.md
- guides/performance.md
- guides/validation.md

## Examples
- examples/example_multiscale_filtered.md
- examples/example_multiscale_random_sources.md
- examples/wavefield_animation.md
- examples/benchmark_maxwell.md

## API Reference
- api/index.md
- api/tide.md
- api/callbacks.md
- api/resampling.md
- api/cfl.md
- api/padding.md
- api/validation.md
- api/maxwell.md
- api/wavelets.md
- api/staggered.md
- api/utils.md
- api/storage.md
- api/backend_utils.md
- api/csrc.md

## Developer Docs
- dev/build.md
- dev/cuda.md

Notes:
- Paths in this directory are written for repository-local browsing.
- The API pages focus on practical signatures, shapes, and constraints rather than implementation internals.

## MkDocs Preview

Local preview:

```bash
uv sync --group docs
uv run mkdocs serve
```

Static build:

```bash
uv run mkdocs build
```

GitHub Pages:
- This repository provides a workflow at .github/workflows/docs.yml.
- In repository settings, set Pages source to GitHub Actions.
