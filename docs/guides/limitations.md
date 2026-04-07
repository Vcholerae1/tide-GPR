# Limitations

This page separates stable behavior from advanced and constrained combinations.

## Stable

- 2D TM forward modeling
- 2D TM gradients with the documented storage modes
- 3D forward modeling with documented source and receiver components

## Advanced

- Debye dispersion workflows
- larger inversion workloads that depend heavily on storage tuning

## Known Constraints

- some Python backend modes do not support all storage modes
- feature support differs between 2D and 3D runtime paths
- advanced modes should be validated on a small case before being used broadly

Always pair advanced modes with the checks in `guides/verification.md`.
