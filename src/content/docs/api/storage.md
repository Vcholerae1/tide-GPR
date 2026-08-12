---
title: "Module: tide.storage"
description: "Allocate and manage device, host, and disk snapshot storage."
---

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

## See Also

- [Configuration](/tide-GPR/guides/configuration/) explains memory limits and
  chunking.
- [Limitations](/tide-GPR/guides/limitations/) lists backend-specific
  constraints.

## Structured storage policy

Application code should construct `StorageOptions` rather than call backend
integer helpers:

```python
storage = tide.StorageOptions(
    mode=tide.StorageMode.CPU,
    compression=False,
    bytes_limit_device=None,
    bytes_limit_host=None,
    chunk_steps=0,
)
```

Internally, `resolve_snapshot_storage` converts this policy into an immutable
`SnapshotStorageSpec` containing normalized mode, representation, sample count,
and shot shape. `SnapshotAllocator` then owns the concrete device, host, or disk
buffers.

## TemporaryStorage lifecycle

```python
temporary = TemporaryStorage("/fast/local/path", num_files=4)
try:
    paths = temporary.get_filenames()
finally:
    temporary.close()
```

Use higher-level derivative sessions when possible. Their context manager owns
temporary storage and closes it even when a derivative operation raises.

## Capacity planning

`auto` placement considers configured byte limits, not every allocation in the
process. Reserve headroom for live fields, model gradients, optimizer state,
receiver tensors, and runtime libraries. Validate the resolved mode during a
small run before scaling the shot batch.
