# Example: Wavefield Animation

Script: `examples/wavefield_animation.py`

## Goal
Visualize the temporal evolution of electromagnetic fields from a forward simulation.

## Inputs

- 2D model tensors (epsilon, sigma, mu)
- Source wavelet and source/receiver locations
- Visualization settings (frame stride, color limits, output path)

## Steps

1. Run forward propagation and collect field snapshots over time.
2. Convert field tensors to plotting frames.
3. Render frames with consistent scaling.
4. Export animation (for example GIF or MP4 depending on script settings).

## Outputs

- Wavefield animation file in examples/outputs or configured output directory.
- Optional static previews for selected time steps.

## Practical Tips

- Start with small grid and short nt to validate plotting pipeline.
- Fix color range across frames to avoid misleading amplitude interpretation.
