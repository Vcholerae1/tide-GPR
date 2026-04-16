# Example: Joint Epsilon And Sigma Inversion

Script: `examples/example_multiscale_joint_eps_sigma.py`

## Goal

Document the staged workflow for jointly updating relative permittivity and conductivity.

## What It Demonstrates

- joint parameter inversion
- empirical conductivity parameterization
- staged low-pass schedules on observed data
- shot batching and PDE accounting
- block-preconditioned PLBFGS style updates

## Runtime Assumptions

- CUDA is strongly recommended
- the example depends on the `OverThrust.npy` model and the external `sotb_wrapper` package
- this is intended as a larger workflow reference, not a minimal verification step

## When To Use It

Read this example after the inversion guide if you want to study a more realistic staged inversion that updates both `epsilon` and `sigma` together.
