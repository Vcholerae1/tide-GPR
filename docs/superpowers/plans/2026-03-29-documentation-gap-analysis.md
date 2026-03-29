# Documentation Gap Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework TIDE documentation so new users can move from install to first forward model to inversion-oriented examples, while documenting the implemented configuration, limitations, and verification surface.

**Architecture:** Use a layered documentation pass. First, rebuild the entry points and docs navigation. Next, add onboarding, API illustration, and operations reference pages. Finally, document the prioritized examples and verify the docs build and user-facing commands against the current repository.

**Tech Stack:** Markdown, MkDocs Material, Python 3.12, uv, existing TIDE examples and API docs

---

## File Structure

### Existing files to modify

- `README.md`
- `mkdocs.yml`
- `docs/README.md`
- `docs/index.md`
- `docs/overview.md`
- `docs/getting-started.md`
- `docs/guides/modeling.md`
- `docs/guides/inversion.md`
- `docs/guides/storage.md`
- `docs/guides/callbacks.md`
- `docs/guides/performance.md`
- `docs/guides/validation.md`
- `docs/api/index.md`
- `docs/api/tide.md`
- `docs/api/maxwell.md`
- `docs/api/storage.md`
- `docs/api/backend_utils.md`

### New files to create

- `docs/guides/api-orientation.md`
- `docs/guides/configuration.md`
- `docs/guides/limitations.md`
- `docs/guides/verification.md`
- `docs/examples/index.md`
- `docs/examples/example_checkpoint.md`
- `docs/examples/example_multiscale_crosscorr_wrong_wavelet.md`
- `docs/examples/example_multiscale_joint_eps_sigma.md`

### Files used for validation context

- `examples/example_checkpoint.py`
- `examples/example_multiscale_crosscorr_wrong_wavelet.py`
- `examples/example_multiscale_joint_eps_sigma.py`
- `examples/example_wavefield_animation.py`
- `src/tide/__init__.py`
- `src/tide/maxwell/tm2d.py`
- `src/tide/maxwell/maxwell3d.py`
- `src/tide/storage.py`
- `src/tide/backend_utils.py`

### Files intentionally deferred in the first pass

- `docs/dev/build.md`
- `docs/dev/cuda.md`
- `docs/examples/benchmark_maxwell.md`
- `docs/examples/example_multiscale_filtered.md`
- `docs/examples/example_multiscale_random_sources.md`
- `docs/examples/wavefield_animation.md`

Reason: the approved spec prioritizes onboarding, API illustration, configuration, limitations, verification, and three high-value example workflows before a broader example and developer-doc pass.

### Task 1: Rebuild Entry Points And Navigation

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/index.md`
- Modify: `mkdocs.yml`
- Test: `uv run mkdocs build`

- [ ] **Step 1: Rewrite the README documentation map and learning-path section**

Replace the current short "Documentation" section in `README.md` with:

```md
## Documentation

Recommended reading path:

1. `docs/getting-started.md` for installation and the first 2D forward run
2. `docs/guides/api-orientation.md` for when to use `tide.maxwelltm`, `tide.maxwell3d`, `MaxwellTM`, and `Maxwell3D`
3. `docs/guides/modeling.md` and `docs/guides/inversion.md` for day-to-day workflows
4. `docs/guides/configuration.md` for storage, callbacks, backend, and CFL-related controls
5. `docs/guides/limitations.md` and `docs/guides/verification.md` before enabling advanced features broadly

Key example docs:

- `docs/examples/example_checkpoint.md`
- `docs/examples/example_multiscale_crosscorr_wrong_wavelet.md`
- `docs/examples/example_multiscale_joint_eps_sigma.md`
```

- [ ] **Step 2: Add a concise feature/support matrix to the README**

Insert this table below the "Features" section in `README.md`:

```md
## Feature Matrix

| Capability | Entry Point | Status | Notes |
| --- | --- | --- | --- |
| 2D TM forward modeling | `tide.maxwelltm` | Stable | Primary onboarding path |
| 2D TM inversion / autograd | `tide.maxwelltm`, `MaxwellTM` | Stable | Uses PyTorch autograd |
| 3D forward modeling | `tide.maxwell3d` | Stable | Supports component selection |
| 3D inversion / gradients | `tide.maxwell3d`, `Maxwell3D` | Stable with constraints | See limitations page |
| Snapshot storage modes | `storage_mode=*` | Stable | Device / CPU / disk / auto |
| Callbacks | `forward_callback`, `backward_callback` | Stable | Keep callback work light |
| Debye dispersion | `DebyeDispersion` | Advanced | Requires explicit time-step validation |
| CUDA graph mode | `maxwell3d(..., cuda_graph=True)` | Experimental | Forward-only CUDA path |
```

- [ ] **Step 3: Rewrite the docs landing page to route users by goal**

Replace `docs/index.md` with:

```md
# TIDE Documentation Home

Use this site in the following order:

1. [Getting Started](getting-started.md) for installation and the first forward run
2. [API Orientation](guides/api-orientation.md) for choosing functional vs module APIs
3. [Modeling](guides/modeling.md) and [Inversion Workflow](guides/inversion.md) for common workflows
4. [Configuration Reference](guides/configuration.md) before tuning storage, callbacks, backend mode, or CFL settings
5. [Limitations](guides/limitations.md) and [Verification](guides/verification.md) before relying on advanced combinations

Looking for examples:

- [Examples Overview](examples/index.md)
- [Checkpointing](examples/example_checkpoint.md)
- [Wrong-Wavelet Multiscale Inversion](examples/example_multiscale_crosscorr_wrong_wavelet.md)
- [Joint Epsilon + Sigma Inversion](examples/example_multiscale_joint_eps_sigma.md)
```

- [ ] **Step 4: Align `docs/README.md` with the new navigation model**

Replace the "Recommended reading order" and section lists in `docs/README.md` with:

```md
Recommended reading order:
1. `getting-started.md`
2. `guides/api-orientation.md`
3. `guides/modeling.md`
4. `guides/inversion.md`
5. `guides/configuration.md`
6. `guides/limitations.md`
7. `guides/verification.md`

## Examples
- `examples/index.md`
- `examples/example_checkpoint.md`
- `examples/example_multiscale_crosscorr_wrong_wavelet.md`
- `examples/example_multiscale_joint_eps_sigma.md`
```

- [ ] **Step 5: Update MkDocs navigation to expose the new docs flow**

Edit the `nav:` section in `mkdocs.yml` so the Guides and Examples groups contain:

```yaml
nav:
  - Home: index.md
  - Overview: overview.md
  - Getting Started: getting-started.md
  - Guides:
      - API Orientation: guides/api-orientation.md
      - Modeling: guides/modeling.md
      - Inversion Workflow: guides/inversion.md
      - Configuration Reference: guides/configuration.md
      - Sources and Receivers: guides/sources-receivers.md
      - Boundaries and PML: guides/boundaries.md
      - Storage and Checkpointing: guides/storage.md
      - Callbacks: guides/callbacks.md
      - Performance Tips: guides/performance.md
      - Stability and Validation: guides/validation.md
      - Limitations: guides/limitations.md
      - Verification: guides/verification.md
  - API:
      - Index: api/index.md
      - Top-level Module: api/tide.md
      - Maxwell: api/maxwell.md
      - Wavelets: api/wavelets.md
      - Callbacks: api/callbacks.md
      - Resampling: api/resampling.md
      - CFL: api/cfl.md
      - Padding: api/padding.md
      - Validation: api/validation.md
      - Staggered: api/staggered.md
      - Utils: api/utils.md
      - Storage: api/storage.md
      - Backend Utils: api/backend_utils.md
      - C/CUDA: api/csrc.md
  - Examples:
      - Overview: examples/index.md
      - Checkpointing: examples/example_checkpoint.md
      - Wrong-Wavelet Inversion: examples/example_multiscale_crosscorr_wrong_wavelet.md
      - Joint Epsilon + Sigma Inversion: examples/example_multiscale_joint_eps_sigma.md
      - Multiscale Filtered: examples/example_multiscale_filtered.md
      - Random Source Encoding: examples/example_multiscale_random_sources.md
      - Wavefield Animation: examples/wavefield_animation.md
      - Benchmark: examples/benchmark_maxwell.md
```

- [ ] **Step 6: Build the docs site to validate the new navigation**

Run: `uv run mkdocs build`
Expected: PASS with a line similar to `Documentation built in ... seconds`

- [ ] **Step 7: Commit the entry-point and navigation changes**

```bash
git add README.md docs/README.md docs/index.md mkdocs.yml
git commit -m "docs(docs): reorganize entry points and navigation"
```

### Task 2: Add Onboarding And API Illustration Guides

**Files:**
- Modify: `docs/overview.md`
- Modify: `docs/getting-started.md`
- Create: `docs/guides/api-orientation.md`
- Modify: `docs/guides/modeling.md`
- Modify: `docs/guides/inversion.md`
- Test: `uv run mkdocs build`

- [ ] **Step 1: Rewrite `docs/overview.md` around workflow concepts**

Replace the current body of `docs/overview.md` with a structure like:

```md
# Project Overview

TIDE is a PyTorch-first electromagnetic modeling and inversion library built around finite-difference time-domain Maxwell solvers.

## What You Can Do With TIDE

- Run 2D TM forward simulations with `tide.maxwelltm`
- Run 3D forward simulations with `tide.maxwell3d`
- Compute gradients with respect to `epsilon` and `sigma`
- Build inversion loops in raw PyTorch or with `MaxwellTM` / `Maxwell3D`
- Control memory and runtime with storage, callback, and backend options

## Recommended Learning Order

1. Run a small 2D forward example
2. Learn the solver inputs and tensor shapes
3. Read the inversion workflow guide
4. Study one inversion-oriented example
5. Review configuration, limitations, and verification before scaling up
```

- [ ] **Step 2: Rewrite `docs/getting-started.md` to include an explicit first-run path**

Ensure the page contains these sections:

````md
## First Success Criteria

You are done with this page when you can:

- import `tide`
- verify backend availability
- run one small 2D forward simulation
- identify where inversion and API docs live next

## Minimal 2D Forward Run

```python
import torch
import tide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = torch.full((96, 96), 4.0, device=device)
sigma = torch.zeros_like(epsilon)
mu = torch.ones_like(epsilon)
dt = 4e-11
nt = 300
src = tide.ricker(8e8, nt, dt, device=device).view(1, 1, nt)
src_loc = torch.tensor([[[20, 48]]], device=device, dtype=torch.long)
rec_loc = torch.tensor([[[20, 60]]], device=device, dtype=torch.long)

*_, receivers = tide.maxwelltm(
    epsilon=epsilon,
    sigma=sigma,
    mu=mu,
    grid_spacing=0.02,
    dt=dt,
    source_amplitude=src,
    source_location=src_loc,
    receiver_location=rec_loc,
    pml_width=10,
)

print(receivers.shape)
```

## What To Read Next

- `guides/api-orientation.md`
- `guides/modeling.md`
- `guides/inversion.md`
- `examples/index.md`
````

- [ ] **Step 3: Create the API orientation page**

Create `docs/guides/api-orientation.md` with:

```md
# API Orientation

Use this page to decide which public API layer matches your workflow.

## Functional APIs

- `tide.maxwelltm(...)`
- `tide.maxwell3d(...)`

Use the functional APIs when:

- you want the shortest path from tensors to receiver data,
- you are scripting experiments directly,
- you do not need to keep a model object around.

## Module APIs

- `tide.MaxwellTM(...)`
- `tide.Maxwell3D(...)`

Use the module APIs when:

- you want model parameters stored in a reusable `torch.nn.Module`,
- you are integrating with optimizers and training loops,
- you want a model object that can be moved between devices.

## Supporting APIs

- `tide.ricker` for source design
- `tide.CallbackState` for callbacks
- `tide.DebyeDispersion` for dispersive materials
- `tide.backend_utils` for backend availability checks
```

- [ ] **Step 4: Expand the modeling guide with a real parameter checklist**

Add the following sections to `docs/guides/modeling.md`:

```md
## Forward Modeling Checklist

- Choose 2D TM or 3D based on the problem geometry
- Build `epsilon`, `sigma`, and `mu` on the target device
- Confirm source and receiver tensor shapes
- Pick `pml_width` and `stencil` deliberately
- Start with a small domain before scaling up

## Choosing 2D vs 3D

Use 2D TM when:

- the survey is effectively planar,
- you need faster iteration,
- you are learning the API for the first time.

Use 3D when:

- component selection matters,
- the geometry is not well represented in 2D,
- you are ready to pay the additional compute and memory cost.
```

- [ ] **Step 5: Expand the inversion guide with an API-oriented loop example**

Add this section to `docs/guides/inversion.md`:

````md
## Minimal PyTorch Inversion Skeleton

```python
import torch
import tide

epsilon = torch.full((96, 96), 4.0, requires_grad=True)
sigma = torch.zeros_like(epsilon, requires_grad=True)
mu = torch.ones_like(epsilon)
optimizer = torch.optim.Adam([epsilon, sigma], lr=1e-2)

for _ in range(10):
    optimizer.zero_grad()
    pred = tide.maxwelltm(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=src,
        source_location=src_loc,
        receiver_location=rec_loc,
        pml_width=10,
    )[-1]
    loss = (pred - observed).pow(2).mean()
    loss.backward()
    optimizer.step()
```

See `docs/examples/example_multiscale_joint_eps_sigma.md` for a larger staged workflow and `docs/examples/example_multiscale_crosscorr_wrong_wavelet.md` for a wavelet-mismatch-aware objective.
````

- [ ] **Step 6: Build the docs site after the onboarding and guide edits**

Run: `uv run mkdocs build`
Expected: PASS with no missing-page errors

- [ ] **Step 7: Commit the onboarding and guide changes**

```bash
git add docs/overview.md docs/getting-started.md docs/guides/api-orientation.md docs/guides/modeling.md docs/guides/inversion.md
git commit -m "docs(guides): add onboarding and api orientation"
```

### Task 3: Add Configuration, Limitations, And Verification Reference

**Files:**
- Create: `docs/guides/configuration.md`
- Create: `docs/guides/limitations.md`
- Create: `docs/guides/verification.md`
- Modify: `docs/guides/storage.md`
- Modify: `docs/guides/callbacks.md`
- Modify: `docs/guides/performance.md`
- Modify: `docs/guides/validation.md`
- Test: `uv run mkdocs build`

- [ ] **Step 1: Create the configuration reference page**

Create `docs/guides/configuration.md` with this structure:

```md
# Configuration Reference

This page centralizes the runtime controls that most strongly affect correctness, memory use, and performance.

## Storage Controls

- `storage_mode`: `device`, `cpu`, `disk`, `none`, `auto`
- `storage_compression`: `none` or `bf16`
- `storage_bytes_limit_device`
- `storage_bytes_limit_host`
- `storage_chunk_steps`

## Backend Controls

- `python_backend=False` for the native path when available
- string backend modes where supported on the Python path
- `tide.backend_utils.is_backend_available()` to check native backend visibility

## Callback Controls

- `forward_callback`
- `backward_callback`
- `callback_frequency`

## Numerical Controls

- `stencil`
- `pml_width`
- `freq_taper_frac`
- `time_pad_frac`
- `time_taper`
- `model_gradient_sampling_interval`

## 3D-Specific Controls

- `source_component`
- `receiver_component`

## Dispersive Materials

- `DebyeDispersion(delta_epsilon=..., tau=...)`
- enforce `dt < min(tau)`
```

- [ ] **Step 2: Strengthen the storage guide to point at the new reference page**

Append this section to `docs/guides/storage.md`:

```md
## Related Controls

See `guides/configuration.md` for:

- `storage_bytes_limit_device`
- `storage_bytes_limit_host`
- `storage_chunk_steps`
- backend-specific storage caveats
```

- [ ] **Step 3: Strengthen the callbacks and performance guides around user-facing constraints**

Add to `docs/guides/callbacks.md`:

```md
## Callback Constraints

- Keep callbacks lightweight and side-effect aware
- Do not resize or relocate callback-visible tensors
- Prefer summary statistics over heavy per-step plotting
- Increase `callback_frequency` when callback overhead dominates runtime
```

Add to `docs/guides/performance.md`:

```md
## Read This Before Enabling Advanced Modes

Before enabling advanced runtime options broadly:

1. Confirm correctness on a small case
2. Read `guides/limitations.md`
3. Run the checks in `guides/verification.md`
```

- [ ] **Step 4: Create the limitations page**

Create `docs/guides/limitations.md` with:

```md
# Limitations

This page separates stable behavior from advanced and constrained combinations.

## Stable

- 2D TM forward modeling
- 2D TM gradients with standard storage modes
- 3D forward modeling with documented source and receiver components

## Advanced Or Experimental

- Debye dispersion workflows
- CUDA graph mode on the 3D CUDA forward path
- large inversion workloads that depend heavily on storage tuning

## Known Constraints

- some Python backend modes do not support all storage modes
- CUDA graph mode does not apply uniformly across all solver paths
- feature support differs between 2D and 3D runtime paths

Always pair advanced modes with the checks in `guides/verification.md`.
```

- [ ] **Step 5: Create the verification page with concrete commands**

Create `docs/guides/verification.md` with:

````md
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

## Advanced Feature Verification

Run:

```bash
uv run python examples/benchmark_maxwell3d_cuda_graph.py --verify
```

Expected:

- graph and non-graph receiver traces match before timing begins
````

- [ ] **Step 6: Cross-link the validation guide to the new verification page**

Append this section to `docs/guides/validation.md`:

```md
## Operational Verification

Numerical validation helpers are only one part of trust-building. For installation and workflow-level checks, continue with `guides/verification.md`.
```

- [ ] **Step 7: Build the docs site after the reference-page additions**

Run: `uv run mkdocs build`
Expected: PASS with the new pages included in navigation

- [ ] **Step 8: Commit the configuration, limitations, and verification docs**

```bash
git add docs/guides/configuration.md docs/guides/limitations.md docs/guides/verification.md docs/guides/storage.md docs/guides/callbacks.md docs/guides/performance.md docs/guides/validation.md
git commit -m "docs(guides): add configuration and verification reference"
```

### Task 4: Document The Prioritized Example Workflows

**Files:**
- Create: `docs/examples/index.md`
- Create: `docs/examples/example_checkpoint.md`
- Create: `docs/examples/example_multiscale_crosscorr_wrong_wavelet.md`
- Create: `docs/examples/example_multiscale_joint_eps_sigma.md`
- Test: `uv run mkdocs build`

- [ ] **Step 1: Create an examples overview page that routes users by learning goal**

Create `docs/examples/index.md` with:

```md
# Examples Overview

Use these examples by goal:

- Learn memory trade-offs: `example_checkpoint.py`
- Study inversion failure from wavelet mismatch: `example_multiscale_crosscorr_wrong_wavelet.py`
- Study staged joint inversion of `epsilon` and `sigma`: `example_multiscale_joint_eps_sigma.py`

Previously documented examples remain available for filtered inversion, random source encoding, and wavefield visualization.
```

- [ ] **Step 2: Add the checkpointing example page**

Create `docs/examples/example_checkpoint.md` with:

```md
# Example: Checkpointing

Script: `examples/example_checkpoint.py`

## Goal

Show how PyTorch checkpointing can reduce activation memory during 2D TM inversion workflows.

## What It Demonstrates

- splitting propagation into segments
- recomputing forward activations during backward
- trading runtime for memory
- capturing wavefield snapshots for inspection

## Practical Notes

- start on a small case before increasing `nt` or shot count
- checkpointing reduces memory but increases compute time
- use this pattern only after confirming the non-checkpointed workflow is correct
```

- [ ] **Step 3: Add the wrong-wavelet inversion example page**

Create `docs/examples/example_multiscale_crosscorr_wrong_wavelet.md` with:

```md
# Example: Multiscale Inversion With Wavelet Mismatch

Script: `examples/example_multiscale_crosscorr_wrong_wavelet.py`

## Goal

Show how inversion behavior changes when the observed and inversion wavelets do not match exactly.

## What It Demonstrates

- multiscale inversion under source mismatch
- cross-correlation style objectives
- comparing observed and inversion wavelets directly

## Expected Outputs

- wavelet comparison figure
- inversion progress figures
- diagnostics that make mismatch effects visible
```

- [ ] **Step 4: Add the joint epsilon-plus-sigma inversion page**

Create `docs/examples/example_multiscale_joint_eps_sigma.md` with:

```md
# Example: Joint Epsilon And Sigma Inversion

Script: `examples/example_multiscale_joint_eps_sigma.py`

## Goal

Document the staged workflow for jointly updating relative permittivity and conductivity.

## What It Demonstrates

- joint parameter inversion
- empirical conductivity parameterization
- staged low-pass schedules
- shot batching and PDE accounting

## Runtime Assumptions

- CUDA is strongly recommended
- the example is intended as a larger workflow, not a minimal smoke test
- reduce `nt`, `n_shots`, and batch size first when debugging
```

- [ ] **Step 5: Build the docs site after adding the example pages**

Run: `uv run mkdocs build`
Expected: PASS with no missing-file references under `Examples`

- [ ] **Step 6: Commit the example documentation pages**

```bash
git add docs/examples/index.md docs/examples/example_checkpoint.md docs/examples/example_multiscale_crosscorr_wrong_wavelet.md docs/examples/example_multiscale_joint_eps_sigma.md
git commit -m "docs(examples): add inversion workflow example pages"
```

### Task 5: Align The API Reference With The New Learning Path

**Files:**
- Modify: `docs/api/index.md`
- Modify: `docs/api/tide.md`
- Modify: `docs/api/maxwell.md`
- Modify: `docs/api/storage.md`
- Modify: `docs/api/backend_utils.md`
- Test: `uv run mkdocs build`

- [ ] **Step 1: Rewrite the API index around entry points instead of module inventory**

Replace the current lead section of `docs/api/index.md` with:

```md
# API Reference

Start here if you already know which workflow you want:

1. `tide.maxwelltm` for 2D TM forward modeling and inversion
2. `tide.maxwell3d` for 3D forward modeling and inversion
3. `tide.MaxwellTM` / `tide.Maxwell3D` for reusable module-based workflows
4. `tide.ricker` for source generation
5. `tide.CallbackState` for forward and backward callbacks
6. `tide.DebyeDispersion` and storage/backend helpers for advanced workflows
```

- [ ] **Step 2: Add API-orientation cross-links to the top-level module docs**

Append this section to `docs/api/tide.md`:

```md
## Where To Go Next

- Read `guides/api-orientation.md` for choosing functional vs module APIs
- Read `guides/configuration.md` before tuning storage or backend behavior
- Read `guides/limitations.md` and `guides/verification.md` before relying on advanced combinations
```

- [ ] **Step 3: Expand the Maxwell API page with decision guidance**

Append this section to `docs/api/maxwell.md`:

```md
## Choosing The Right Maxwell Entry Point

- Use `maxwelltm` for the fastest onboarding path and most 2D examples
- Use `maxwell3d` when component selection and full 3D geometry matter
- Use `MaxwellTM` or `Maxwell3D` when you want model tensors stored inside a `torch.nn.Module`

See:

- `guides/api-orientation.md`
- `guides/modeling.md`
- `guides/inversion.md`
```

- [ ] **Step 4: Add user-facing warnings to the storage and backend reference pages**

Append to `docs/api/storage.md`:

```md
## See Also

- `guides/configuration.md` for how storage settings interact with memory limits and chunking
- `guides/limitations.md` for backend-specific constraints
```

Append to `docs/api/backend_utils.md`:

```md
## User-Facing Use

Most users only need:

- `is_backend_available()`
- `get_library_path()`

Use these functions during installation checks and before investigating backend-specific performance behavior.
```

- [ ] **Step 5: Build the docs site after API-reference alignment**

Run: `uv run mkdocs build`
Expected: PASS with no broken internal links

- [ ] **Step 6: Commit the API-reference updates**

```bash
git add docs/api/index.md docs/api/tide.md docs/api/maxwell.md docs/api/storage.md docs/api/backend_utils.md
git commit -m "docs(api): align reference with onboarding flow"
```

### Task 6: Verify The Documentation Against Real Commands And Finalize

**Files:**
- Modify: `docs/guides/verification.md`
- Modify: `docs/examples/example_checkpoint.md`
- Modify: `docs/examples/example_multiscale_crosscorr_wrong_wavelet.md`
- Modify: `docs/examples/example_multiscale_joint_eps_sigma.md`
- Test: `uv run mkdocs build`
- Test: `uv run python - <<'PY' ... PY`

- [ ] **Step 1: Run the backend availability snippet from the verification guide**

Run:

```bash
uv run python - <<'PY'
from tide import backend_utils
print("backend available:", backend_utils.is_backend_available())
print("library path:", backend_utils.get_library_path())
PY
```

Expected:

- Python exits successfully
- output includes `backend available:` and `library path:`

- [ ] **Step 2: Run the gradient-validation example in the documented form**

Run: `uv run python examples/example_gradient_dot_fd_validation.py --backend c`
Expected:

- script starts successfully
- output includes Taylor-test style diagnostics
- no immediate import or argument errors

- [ ] **Step 3: Run the CUDA-graph verification command if CUDA is available**

Run: `uv run python examples/benchmark_maxwell3d_cuda_graph.py --verify`
Expected:

- on CUDA hosts, graph and baseline traces match before benchmark output
- on CPU-only hosts, record that this step is skipped because CUDA is unavailable

- [ ] **Step 4: Update the verification and example docs with any command caveats discovered during the runs**

If commands need caveats, patch the pages with text like:

```md
## Runtime Notes

- this command is intended for environments with the native backend available
- CUDA-only verification steps should be skipped on CPU-only hosts
- larger inversion examples are documentation references, not required smoke tests
```

- [ ] **Step 5: Run the final docs build**

Run: `uv run mkdocs build`
Expected: PASS with the site generated under `site/`

- [ ] **Step 6: Commit the verification updates and final doc adjustments**

```bash
git add docs/guides/verification.md docs/examples/example_checkpoint.md docs/examples/example_multiscale_crosscorr_wrong_wavelet.md docs/examples/example_multiscale_joint_eps_sigma.md
git commit -m "docs(verification): validate documented workflows"
```
