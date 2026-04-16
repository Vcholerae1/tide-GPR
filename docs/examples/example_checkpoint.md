# Example: Checkpointing

Script: `examples/example_checkpoint.py`

## Goal

Show how PyTorch checkpointing can reduce activation memory during 2D TM inversion workflows.

## What It Demonstrates

- splitting propagation into time segments
- recomputing forward activations during backward
- trading runtime for memory
- capturing wavefield snapshots for inspection

## Practical Notes

- start on a small case before increasing `nt` or shot count
- checkpointing reduces memory but increases compute time
- the script also saves wavefield snapshots and animations, so expect plotting overhead
- use this pattern only after confirming the non-checkpointed workflow is correct
