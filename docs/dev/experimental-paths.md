# Experimental Paths

The main branch keeps only the stable public API and its direct implementation
paths. Larger experiments are preserved in the archive tag
`archive/experimental-snapshot-2026-06-13` instead of remaining importable from
the package.

## What Was Archived

- large inversion and LSRTM example scripts and generated result bundles
- optimizer prototypes under `tide.optim`
- CUDA profiling, sweep, and comparison scripts
- compact CPML layout and split-PML launch prototypes
- alternate snapshot, checkpoint, and direct material-gradient runtime paths

## Why They Are Not In Main

These paths were useful for research iteration, but they were not ready to
carry as supported package surface:

- their APIs were not stable enough to document as user-facing behavior;
- backend support differed across CPU, CUDA, Python fallback, and storage modes;
- the test matrix was too broad for routine release verification;
- some paths depended on large local datasets or generated artifacts;
- keeping them beside the stable solver made the codebase harder to audit.

The archive tag keeps the work recoverable without making it part of the main
branch maintenance contract.
