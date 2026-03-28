# TM2D Performance Workflow and Keep-Revert Agent Loop Design

## Goal

Establish a heavy-duty local performance engineering workflow for TM2D full-waveform inversion on a single-GPU development machine, starting with an RTX 4070. The workflow should optimize for end-to-end `loss + grad` wall time on a representative workload derived from `examples/example_multiscale_joint_eps_sigma.py`, while preserving numerical and physics-facing correctness.

The first target is not full autonomy. The first target is a reliable, repeatable closed loop that can:

1. Measure a stable representative TM2D FWI workload.
2. Attribute cost across end-to-end evaluation, stages, and hotspot kernels.
3. Let an agent modify one hotspot implementation at a time.
4. Automatically keep or revert a candidate change based on correctness and KPI thresholds.

## Representative Workloads

The workflow should standardize a small set of named workloads instead of profiling ad hoc scripts.

- `smoke`: fast local validation for the workflow itself.
- `representative`: TM2D FWI-style evaluation aligned with the current main case: `ny=400`, `nx=200`, `nt=1800`, `n_shots=100`, batched shots, GPU execution.
- `stress`: larger or more memory-constrained variant used to expose scaling and storage limits.

`representative` is the primary optimization target and the default benchmark for keep-revert decisions.

## Success Metrics

Primary KPI:

- End-to-end wall time for one complete objective and gradient evaluation over the full representative workload.

Secondary KPIs:

- Peak GPU memory.
- Effective cost per shot and per shot-batch.
- Forward time share.
- Backward time share.
- Top kernel time share.
- Numerical deviation versus baseline.

The workflow should always report the primary KPI first. Kernel-level improvements only count as wins if they improve the end-to-end KPI or unlock a materially larger batch size.

## Recommended Approach

Adopt an AutoKernel-style workflow shape, but not full AutoKernel-style autonomy in the first version.

The first version should be a constrained semi-automatic loop:

1. Profile and rank hotspots for the representative workload.
2. Select one hotspot kernel or implementation path.
3. Allow the agent to edit only the files associated with that hotspot.
4. Run a fixed evaluation harness.
5. Keep the change only if correctness passes and KPI thresholds are met.
6. Otherwise revert automatically and record the failed experiment.

This keeps the system aligned with the realities of TIDE, where bottlenecks may sit in CUDA kernels, shot batching, storage, backward logic, or orchestration, and where correctness must include more than simple tensor equality.

## System Design

### 1. Benchmark Layer

Create a dedicated TM2D performance benchmark entrypoint instead of using the example script directly.

Responsibilities:

- Load a named workload configuration.
- Run one complete objective and gradient evaluation.
- Emit stable timing and memory metrics.
- Save machine-readable results for comparison.

Inputs should include:

- workload name
- device
- dtype
- storage mode
- shot batch size
- warmup and repeat counts
- optional profiling mode

Outputs should include:

- JSON summary
- optional profiler artifacts
- concise terminal summary

### 2. Attribution Layer

The benchmark must expose cost at three levels.

Level A: end-to-end

- full evaluation wall time
- peak memory

Level B: stage breakdown

- model and source preparation
- shot batching overhead
- forward propagation
- filtering and loss computation
- backward propagation
- gradient collection or optimizer-side preparation

Level C: hotspot breakdown

- top CUDA kernels by time
- launch counts
- memory or compute bound evidence where available

The attribution order should be:

1. end-to-end timing
2. stage timing
3. operator and kernel hotspots

This prevents early overfitting to a single kernel before the whole evaluation is understood.

### 3. Profiling Layer

The default local flow should degrade cleanly across tool availability.

Default:

- `torch.profiler` for operator and CUDA kernel ranking

Advanced:

- `nsys` for timeline and synchronization analysis
- `ncu` for hotspot kernel microanalysis

`ncu` analysis should focus on a short list of stable questions:

- Is the kernel memory-bound or compute-bound?
- What is the achieved DRAM throughput?
- What is the achieved SM throughput?
- What are the dominant stall reasons?
- Is occupancy limited by registers, shared memory, or launch shape?

### 4. Correctness Layer

Every candidate change in the keep-revert loop must pass a fixed correctness harness before speedup is credited.

The harness should include:

- smoke execution
- shape sweep for supported TM2D configurations
- receiver trace comparison against baseline
- gradient consistency checks for the affected path
- determinism or bounded-repeatability checks where applicable
- numerical stability checks on representative and edge-case inputs

For TIDE, end-to-end verification should be treated as mandatory, not optional. A fast kernel benchmark without representative TM2D validation is insufficient.

### 5. Keep-Revert Agent Layer

The first autonomous loop should be intentionally constrained.

Constraints:

- One hotspot target at a time.
- One bounded edit set at a time.
- No cross-cutting refactors inside optimization experiments.
- No keep decision without benchmark and verification evidence.

Inputs:

- hotspot target
- allowed file set
- benchmark command
- verification command
- improvement threshold

Decision rule:

- `keep` if correctness passes and end-to-end KPI improves beyond threshold.
- `revert` if correctness fails, runtime regresses, or the result is inconclusive.

A change that improves only an isolated microbenchmark but does not improve the representative end-to-end workload should not be kept by default.

### 6. Result Tracking Layer

Each experiment should produce a durable record.

Minimum record contents:

- git commit or parent commit
- target hotspot
- edited files
- benchmark configuration
- wall time and memory deltas
- top hotspot deltas
- correctness status
- keep or revert decision

Artifacts should be stored under a stable local directory such as:

- `artifacts/perf/<date>/<gpu>/<workload>/`

This gives the agent and the developer a shared history and reduces repeated dead-end experiments.

## File and Tool Layout

Suggested first-pass layout:

- `scripts/perf/benchmark_tm2d_fwi.py`
- `scripts/perf/run_torch_profiler.sh`
- `scripts/perf/run_nsys.sh`
- `scripts/perf/run_ncu.sh`
- `scripts/perf/summarize_perf.py`
- `scripts/perf/verify_tm2d_perf_candidate.py`
- `artifacts/perf/`
- `docs/guides/performance_tm2d.md`

The benchmark and verification scripts should be first-class project tools, not one-off local notes.

## Decision Rules for Optimization

The workflow should standardize how profile evidence turns into next actions.

- If end-to-end time is dominated by forward and backward kernels, prioritize kernel optimization.
- If batch size is constrained by memory, prioritize storage, snapshots, precision, and memory traffic before deeper kernel work.
- If kernel time is fragmented across many launches, prioritize batching, fusion, or orchestration overhead.
- If `ncu` shows high DRAM utilization with memory-related stalls, treat the kernel as memory-bound.
- If `ncu` shows low occupancy or high register pressure, treat launch shape and register use as first-class candidates.
- If a hotspot improvement does not move the primary KPI, re-rank the system using end-to-end measurements instead of continuing to optimize the same kernel blindly.

## Rollout Plan

### Phase 1: Fix the Benchmark Target

Extract the representative TM2D FWI objective-and-gradient path from `examples/example_multiscale_joint_eps_sigma.py` into a dedicated benchmark script with named workload presets.

Deliverables:

- benchmark script
- named workload configs
- JSON output format

### Phase 2: Add Layered Attribution

Add end-to-end timing, stage timing, and profiler entrypoints.

Deliverables:

- stable timing output
- peak memory measurement
- stage breakdown
- `torch.profiler` integration
- optional `nsys` and `ncu` wrappers

### Phase 3: Add Fixed Verification

Build the correctness harness required for keep-revert decisions.

Deliverables:

- smoke verification
- representative trace comparison
- gradient verification for affected paths
- result summary for pass or fail

### Phase 4: Add Semi-Automatic Keep-Revert Loop

Introduce a controlled agent workflow that targets one hotspot at a time and records keep or revert decisions.

Deliverables:

- experiment manifest
- allowed-file targeting
- keep or revert rule
- experiment result logging

### Phase 5: Evolve Toward Multi-Hotspot Orchestration

After the constrained loop is stable, add hotspot scheduling and longer-running autonomous search.

This phase is explicitly out of scope for the first implementation.

## Risks and Guardrails

Main risks:

- Optimizing isolated kernels that do not improve the real TM2D FWI workload.
- Treating unstable local measurements as meaningful regressions or wins.
- Letting autonomous edits escape the intended hotspot boundary.
- Accepting numerically unsafe changes because they benchmark well.

Guardrails:

- Keep one representative workload fixed.
- Use warmup and repeated measurements.
- Require end-to-end verification before keep.
- Restrict the editable file set per experiment.
- Prefer small, reversible experiments over broad rewrites.

## Testing Strategy

The workflow itself should be tested in layers.

- Unit tests for result parsing and summarization.
- Smoke tests for benchmark entrypoints.
- Verification tests for correctness harness behavior.
- Integration test for the keep-revert decision path using a lightweight workload.

The heavy representative workload should remain available for real performance work, but the automation around it should also have smaller tests so the infrastructure is maintainable.

## Non-Goals

The first version does not aim to:

- provide CI-grade performance gates across heterogeneous GPUs
- autonomously optimize multiple hotspots in parallel
- replace human review of broad architectural changes
- generalize immediately to Maxwell3D

## Recommendation Summary

Build a TIDE-specific, AutoKernel-inspired performance system centered on the TM2D representative FWI workload. Start with a benchmark and verification backbone, then add a constrained keep-revert agent loop for one hotspot at a time. Delay full autonomous multi-hotspot orchestration until the benchmark, attribution, and correctness layers are stable and trusted.
