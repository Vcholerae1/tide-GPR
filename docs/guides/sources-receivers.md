# Sources and Receivers

TODO: Document source and receiver conventions, shapes, and coordinate order.

## Source Amplitude
- Expected shape: [n_shots, n_sources, nt].
- Typical dtype: float32 on CPU or CUDA.
TODO: Document time axis convention and normalization.

## Source Locations
- Expected shape: [n_shots, n_sources, ndim].
- Coordinate order: (y, x) for 2D.

## Receiver Locations
- Expected shape: [n_shots, n_receivers, ndim].
- Coordinate order: (y, x) for 2D.

## Batching
TODO: Explain how shots are batched and how indices map to outputs.
