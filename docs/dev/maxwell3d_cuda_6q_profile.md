# Maxwell 3D CUDA 6Q Profile

This page is retained as an archived profiling note. The profiling scripts and
experimental launch variants used for the original measurement were removed
from the main branch and are available from
`archive/experimental-snapshot-2026-06-13`.

The useful conclusion remains: the native 3D CUDA forward path was primarily
memory-throughput-bound on the measured workload, with the H/E update kernels
dominating runtime. Future optimization work should start from a fresh profile
against the current main branch before adding new runtime paths.
