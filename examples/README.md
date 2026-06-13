# tide-GPR examples

This repository collects small GPR modeling and inversion examples built on top
of `tide-gpr`.

## Born RTM / LSRTM

The repository includes a 2D TM-mode Born imaging example on the bundled
`examples/OverThrust.npy` model. The workflow follows the usual RTM/LSRTM split:
full-wave `tide.maxwelltm` generates `d_obs` and `d0`, then the linearized
residual `d_obs - d0` is migrated/inverted with `tide.borntm` plus PyTorch
autograd for adjoints.

The example will:

- generate full-wave data and form the linearized residual `d_obs - d0`
- build an RTM image with the Born adjoint
- run damped LSRTM with conjugate gradients on the normal equations

Run it from the repo root:

```bash
uv run python examples/example_born_rtm_lsrtm_tm.py --check-adjoint
```

Pure CPU / Python backend fallback:

```bash
uv run python examples/example_born_rtm_lsrtm_tm.py --device cpu --backend python
```

Larger OverThrust crop:

```bash
uv run python examples/example_born_rtm_lsrtm_tm.py --nz 120 --nx 160 --nshots 8 --lsrtm-iters 4
```

Outputs are written to `outputs/born_rtm_lsrtm_tm/`.

## LSRTM with Tide MaxwellTM

`examples/example_deepwave_style_lsrtm_tm.py` generates observed data in a true
permittivity model, subtracts the response of a uniform initial model to
attenuate direct arrivals, computes an RTM adjoint image, and then iteratively
fits the linearized scattered data.

With `tide-gpr==0.0.26`, this example uses `tide.borntm` directly for the
linearized forward action. The adjoint image is formed with PyTorch autograd
with respect to `depsilon`.
By default it uses the same model path as `2d/example_multiscale_filtered_multi.py`:
`multi_freq_inv/data/OverThrust.npy`. If that file is absent, it falls back to
`examples/OverThrust.npy`.

```bash
uv run python examples/example_deepwave_style_lsrtm_tm.py
```

Edit the configuration block at the top of the script to change settings.
For a fast smoke run, set `MODEL_SOURCE = "synthetic"`, `NZ = 32`, `NX = 44`,
`NSHOTS = 4`, `PML_WIDTH = 4`, and `CG_ITERS = 2`.
Set `CHECK_ADJOINT = True` to run the Born adjoint check.
The initial model is uniform below the air layer; `BACKGROUND_EPSILON = None`
uses the subsurface mean of the true model, or set it to a positive value.

Outputs are written to `outputs/deepwave_style_lsrtm_tm/`.
The summary plots are saved as `example_lsrtm_scattered.png` and `example_lsrtm.png`
in that directory.
If `NT = None`, the script estimates a time length that reaches the
bottom of the model and returns to the surface receivers.
The default time step is `DT = 3.5e-11`; the script prints CFL plus
points-per-wavelength diagnostics at startup.
The default acquisition is one source and one receiver per shot, matching the
multi-frequency 2D examples: `NSHOTS = 100`, `D_SOURCE = 4`,
`FIRST_SOURCE = 0`, and `RECEIVER_OFFSET = 1`.
Shot processing is mini-batched with `BATCH_SIZE = 8` by default.
The default stencil is 4th order.
Shot gathers are displayed with a symmetric 92nd percentile red-blue scale;
RTM and LSRTM images are displayed with a gray scale.
For the one-source/one-receiver default acquisition, receiver-data plots use
time versus shot number instead of stretching a single receiver trace.
The LSRTM solve uses conjugate gradients on the damped normal equations
(`CG_ITERS`) because the Born problem is linear.

## Minimum-Phase Deconvolution

`examples/example_min_phase_deconv_tide.py` builds a small sparse-scatterer 2D
GPR model, uses `tide.maxwelltm` to generate a common-offset gather with a
minimum-phase source wavelet, subtracts a homogeneous-background response to
isolate reflections, and applies statistical spiking deconvolution.

```bash
uv run python examples/example_min_phase_deconv_tide.py
```

Outputs are written to `outputs/min_phase_deconv_tide/`.

## TV-L1 regularization

`regularization/tv_l1.py` provides a matrix-free PyTorch implementation of the
anisotropic TV operator used by `SetIntersectionProjection.jl`:

- 2D model shape `(..., nz, nx)`: TV parts are `(D_z m, D_x m)`
- 3D model shape `(..., nz, ny, nx)`: TV parts are `(D_z m, D_y m, D_x m)`
- `active_mask` excludes inactive cells and differences crossing the mask
  boundary, which is useful for keeping the GPR air layer out of the TV term

`examples/example_ifwi_gpr_eps_fourier_features.py` and
`2d/cross_inv_single_freq_common.py` use the Julia-style model-space projection
path. When enabled, the inversion model is projected onto a TV-L1 ball before
wave propagation:

```python
TV_L1_PROJECTION = False
TV_L1_EPSILON_RADIUS = None
TV_L1_SIGMA_RADIUS = None
TV_L1_EPSILON_RADIUS_FRACTION = 1.0
TV_L1_SIGMA_RADIUS_FRACTION = 1.0
```

If an absolute radius is `None`, the radius is computed as
`radius_fraction * TV_L1(start_model)` on the subsurface mask. Gradients use a
straight-through approximation through the projection so the implicit network
can still train while the physics sees the projected model.

For explicit tensor-model FWI, `project_tv_l1_ball` implements the constrained
projection `min_x 0.5 ||x-m||_2^2` subject to `||TV x||_1 <= radius` with
matrix-free ADMM and CG.
