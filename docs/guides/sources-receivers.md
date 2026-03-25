# Sources and Receivers

This guide summarizes source and receiver tensor conventions for both 2D and 3D APIs.

## Source Amplitude
- Expected shape: [n_shots, n_sources, nt].
- Typical dtype: float32 on CPU or CUDA.

Time axis convention:
- Time is the last axis in source_amplitude.
- In returned receiver data, time is the first axis: [nt, n_shots, n_receivers].

## Source Locations
- Expected shape: [n_shots, n_sources, ndim].
- Coordinate order:
	- 2D: [y, x]
	- 3D: [z, y, x]
- dtype should be torch.long/int64.

## Receiver Locations
- Expected shape: [n_shots, n_receivers, ndim].
- Coordinate order follows source conventions.

## Batching

- Shot axis is the first axis in source_amplitude, source_location, and receiver_location.
- Multiple shots run in one call when these tensors share the same n_shots.
- Output receiver_amplitudes preserves shot order.

Example shape mapping:
- source_amplitude: [8, 2, 1500]
- source_location: [8, 2, 2]
- receiver_location: [8, 32, 2]
- receiver output: [1500, 8, 32]

## Validation Rules

- Each coordinate must satisfy 0 <= idx < model_size along that dimension.
- Mismatched n_shots across source and receiver tensors will fail at runtime.
- For source-free propagation, pass source_amplitude=None and provide nt explicitly.
