# TIDE Documentation

This directory is the entry point for user guides, API reference, runnable examples, and developer notes.

Recommended reading order:
1. getting-started.md
2. guides/api-orientation.md
3. guides/modeling.md
4. guides/inversion.md
5. guides/configuration.md
6. guides/limitations.md
7. guides/verification.md

## Start Here
- getting-started.md
- guides/api-orientation.md
- overview.md

## Guides
- guides/api-orientation.md
- guides/modeling.md
- guides/inversion.md
- guides/configuration.md
- guides/sources-receivers.md
- guides/boundaries.md
- guides/storage.md
- guides/callbacks.md
- guides/performance.md
- guides/validation.md
- guides/limitations.md
- guides/verification.md

## Examples
- examples/index.md
- examples/example_checkpoint.md
- examples/example_multiscale_crosscorr_wrong_wavelet.md
- examples/example_multiscale_joint_eps_sigma.md
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
- The documentation is organized for onboarding first, then workflow guides, then API and advanced operational reference.

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
