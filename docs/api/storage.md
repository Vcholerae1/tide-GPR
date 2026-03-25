# Module: tide.storage

Snapshot storage helpers for backward propagation.

## Functions
- storage_mode_to_int

## Classes
- TemporaryStorage

## Storage Modes

- device: snapshots stay on compute device, fastest and most memory-heavy
- cpu: host-backed storage, lower device memory use
- disk: file-backed storage, smallest memory footprint and highest latency
- none: disable snapshot storage

The user-facing APIs also accept auto, which chooses mode based on configured byte limits.

## storage_mode_to_int

Maps mode strings to backend integer constants used by native kernels.

## Compression

- storage_compression supports none and bf16 on default compute path
- bf16 storage is valid for float32 workflows

## TemporaryStorage

- Creates an isolated temporary directory under the specified base path
- Generates one file path per logical shot buffer
- Cleans up directory tree on close or object destruction
