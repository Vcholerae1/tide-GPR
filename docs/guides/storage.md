# Storage and Gradient Checkpointing

Backward propagation needs forward-time snapshots. Storage mode controls where those snapshots live and strongly affects speed/memory trade-offs.

## storage_mode
- device: store on the same device as compute.
- cpu: store on host memory.
- disk: store on local disk.
- none: disable storage.
- auto: select device/cpu/disk based on configured memory limits.

Default behavior in high-level APIs is device unless auto is explicitly requested.

## storage_compression

Supported values:
- False / "none": full precision snapshots
- True / "bf16": BF16 compressed snapshots (default compute path)

Compression reduces memory and I/O pressure at the cost of precision.

## TemporaryStorage

Disk mode writes snapshot chunks under a temporary per-run directory.
TemporaryStorage is responsible for:
- creating unique directory names
- allocating shot file paths
- cleanup on close/destructor

## Tuning Checklist

1. If GPU memory is sufficient, use storage_mode=device for best speed.
2. If GPU OOM occurs, switch to cpu first.
3. If host memory is also constrained, use disk.
4. For auto mode, set storage_bytes_limit_device and storage_bytes_limit_host.
5. Use bf16 compression when memory pressure dominates and accuracy remains acceptable.
