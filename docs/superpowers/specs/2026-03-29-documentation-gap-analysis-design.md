# Documentation Gap Analysis Design

## Summary

This design defines a documentation planning pass for TIDE that focuses on
already-implemented features that are under-documented, hard to discover, or not
explained from a new-user perspective. The primary audience is new users
learning forward modeling, inversion workflows, and practical API usage.

The recommended approach is a layered documentation model:

1. onboarding-first entry points,
2. curated workflow and example coverage,
3. consolidated configuration, limitations, and verification reference.

The goal is to improve learnability and trust without changing solver behavior
or introducing new features.

## Context

The repository already contains a meaningful documentation skeleton:

- top-level onboarding in `README.md`,
- MkDocs pages under `docs/`,
- API reference pages,
- runnable examples under `examples/`,
- developer notes for build and CUDA setup.

However, the implemented feature surface is broader than the effective user
documentation surface.

Observed gaps:

- `README.md` explains core solvers but does not give a strong learning path for
  first forward modeling, inversion, and API exploration.
- The docs site documents only a subset of example scripts, while the repository
  contains additional user-facing workflows such as checkpointing, gradient
  validation, 3D comparison workflows, joint inversion, 3D visualization, and
  CUDA graph benchmarking.
- Important configuration and support details exist in code paths and warning
  behavior but are not centralized in user-facing docs.
- Limitations and support boundaries are not clearly documented in one place.
- Verification guidance is too thin for users to confirm that installation,
  backend availability, gradients, and advanced features are working correctly.

## Audience

Primary audience:

- new users learning forward modeling and inversion in TIDE,
- users who need API illustration rather than raw symbol listings.

Secondary audience:

- power users who need configuration and backend behavior clarified after they
  are already productive with the core workflows.

This priority means the first documentation pass should optimize for discovery,
learning order, and practical examples before deep internal reference coverage.

## Goals

- Make the entry path from installation to first successful simulation obvious.
- Show how forward modeling, inversion, and key APIs relate to each other.
- Document implemented examples that teach important user workflows.
- Consolidate user-facing configuration controls into predictable docs pages.
- State limitations, experimental features, and unsupported combinations
  explicitly.
- Provide verification steps that help users establish confidence in their
  installation and workflows.

## Non-Goals

- No new solver or backend features.
- No attempt to fully document every internal helper or backend detail.
- No major information architecture rewrite unless required for discoverability.
- No requirement to turn every script into a full tutorial in the first pass.

## Recommended Documentation Architecture

### 1. Entry Points

The top-level documentation surfaces should guide a new user through the first
few decisions without requiring source inspection.

This includes:

- `README.md` as the concise product and onboarding surface,
- `docs/index.md` as the docs landing page,
- `docs/getting-started.md` as the first hands-on path,
- `docs/overview.md` as the conceptual orientation page.

These pages should explain:

- what TIDE is for,
- the main supported workflows,
- where to start for 2D forward modeling,
- where inversion guidance lives,
- how to navigate examples and API docs,
- where advanced topics live.

### 2. Workflow Guides

The guide layer should connect concepts to actual practice. It should explain
how model setup, sources, receivers, boundaries, storage, callbacks, stability,
and inversion fit together in normal usage.

This layer should also include explicit API illustration for the public solver
entry points and related user-facing helpers:

- `tide.maxwelltm`,
- `tide.maxwell3d`,
- `MaxwellTM`,
- `Maxwell3D`,
- `CallbackState`,
- `DebyeDispersion`,
- backend and validation helpers where users are expected to touch them.

### 3. Example Coverage

Example pages should reflect the workflows users can actually run today.

The current docs cover only part of the example directory. The documentation
plan should prioritize example pages for scripts that are either:

- core learning paths,
- representative of inversion workflows,
- useful for validation and trust building,
- differentiating features of the repository.

Examples with clear user-facing documentation value include:

- `examples/example_checkpoint.py`,
- `examples/example_gradient_dot_fd_validation.py`,
- `examples/example_maxwell3d_analytic_compare.py`,
- `examples/example_maxwell3d_dispersion_analytic_compare.py`,
- `examples/example_maxwell3d_petrophysical_survey.py`,
- `examples/example_multiscale_crosscorr_wrong_wavelet.py`,
- `examples/example_multiscale_joint_eps_sigma.py`,
- `examples/example_wavefield_animation_3d.py`,
- `examples/benchmark_maxwell3d_cuda_graph.py`.

Each documented example should explain:

- why the script exists,
- what concept it teaches,
- key inputs or flags,
- expected outputs,
- any major runtime or hardware assumptions,
- where it fits in the learning journey.

### 4. Configuration Reference

Configuration knowledge currently appears across README text, guide pages, and
code. The documentation plan should consolidate user-facing controls into one
predictable reference area.

The first-pass scope should include:

- storage modes and compression,
- auto-storage byte limits,
- storage chunking behavior,
- backend selection and backend availability checks,
- callback hooks and callback cadence,
- 3D source and receiver component selection,
- CFL-driven internal resampling knobs,
- dispersion input shape and time-step constraints.

### 5. Limitations and Support Boundaries

A dedicated limitations page should separate:

- stable behavior,
- experimental behavior,
- unsupported or constrained combinations.

This page should make code-implied boundaries visible to users, including cases
such as:

- where CUDA graph applies,
- where Python backends do not support some storage modes,
- where advanced features have backend or gradient restrictions,
- where feature support differs between 2D and 3D paths.

### 6. Verification Guidance

Users need a clear way to answer:

- did installation work,
- is the native backend available,
- do core examples run,
- do gradients appear numerically sane,
- does an advanced feature like CUDA graph preserve correctness.

The documentation plan should define a verification page or section that covers:

- install and backend checks,
- first example validation,
- gradient sanity validation,
- example-based validation for inversion workflows,
- advanced-feature equivalence checks where relevant.

## Proposed Deliverables

The later documentation plan should cover work in these categories:

1. README improvements.
2. Core docs page improvements.
3. Example page expansion and docs nav updates.
4. Configuration reference additions or restructuring.
5. New limitations page.
6. New verification guide.

Each category should be broken into practical implementation tasks, with a clear
distinction between:

- must-have first-pass work,
- optional second-pass refinements.

## Scope Boundaries for the Planning Phase

The planning phase should stay disciplined:

- prioritize documentation for already-implemented features,
- avoid speculative future docs,
- keep the first pass practical enough to complete in one focused doc cycle,
- preserve the current MkDocs structure where possible,
- only add new pages when consolidation or discoverability clearly benefits.

## Risks

### Risk: Over-documenting internals

If the plan treats every internal helper as a first-class user feature, the
docs will become harder to navigate.

Mitigation:

- keep the first pass centered on public workflows and public APIs,
- move internal details behind explicit advanced or developer sections.

### Risk: Example sprawl

If every example gets the same depth, the documentation workload grows quickly
and delays the most important onboarding fixes.

Mitigation:

- prioritize examples by user value,
- allow concise reference-style example pages where a full tutorial is not
  warranted.

### Risk: Hidden support boundaries remain scattered

If limitations are left embedded in code and warnings, user trust remains low.

Mitigation:

- include an explicit limitations deliverable in the documentation plan,
- tie advanced docs to documented support boundaries and verification steps.

## Acceptance Criteria

The resulting documentation plan should ensure that:

- a new user can follow one clear path from install to a first forward run,
- inversion-oriented learning resources are easy to find,
- public API usage is illustrated rather than merely listed,
- high-value implemented examples are documented or explicitly deferred with
  rationale,
- user-facing configuration controls are documented in one predictable place,
- stable, experimental, and unsupported combinations are explicit,
- verification steps are documented for installation health and numerical trust.

## Planning Hand-off

The next step after approval of this design is to create a documentation plan
that covers:

- README,
- docs pages,
- examples,
- configuration,
- limitations,
- verification steps.

That plan should be implementation-oriented, file-specific, and sized for an
engineer who has little prior context about this repository.
