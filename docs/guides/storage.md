# Storage and Gradient Checkpointing

TODO: Describe snapshot storage and its impact on memory and compute.

## storage_mode
- device: store on the same device as compute.
- cpu: store on host memory.
- disk: store on local disk.
- none: disable storage.
TODO: Document any "auto" behavior and defaults.

## storage_compression
TODO: Document compression options (e.g., BF16 boundary storage).

## TemporaryStorage
TODO: Explain temporary file lifecycle for disk mode.
