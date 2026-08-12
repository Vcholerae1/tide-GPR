---
title: "Modeling"
description: "Choose electromagnetic parameters, grid resolution, dimensionality, and numerical settings for a TIDE simulation."
---

A reliable simulation starts before the first kernel launch. The model must represent the intended physics, the grid must resolve the shortest wavelength of interest, the time step must be compatible with explicit FDTD propagation, and the acquisition must fit inside the physical model interior.

## Material parameters

TIDE uses relative material tensors and SI units:

| Tensor | Physical quantity | Unit | Common baseline |
| --- | --- | --- | --- |
| `epsilon` | Relative permittivity $\epsilon_r$ | Dimensionless | Greater than zero |
| `sigma` | Electrical conductivity $\sigma$ | S/m | Zero or positive in passive media |
| `mu` | Relative permeability $\mu_r$ | Dimensionless | Often `1.0` |

All three tensors must have identical shape, dtype, and device. `epsilon` and `mu` must be positive. A passive material normally has non-negative conductivity.

The local electromagnetic wave speed, ignoring dispersion, is approximately

$$
v = \frac{c_0}{\sqrt{\epsilon_r\mu_r}}.
$$

Increasing permittivity therefore lowers propagation speed and shortens wavelength at a fixed frequency. Conductivity attenuates the field and changes phase. Do not treat `epsilon` and `sigma` as interchangeable ways to alter amplitude.

## Grid resolution

Let $f_{max}$ be the highest frequency that must be represented and $v_{min}$ the minimum expected phase velocity. The shortest wavelength is

$$
\lambda_{min}=\frac{v_{min}}{f_{max}}.
$$

Choose spacing so that several grid cells represent this wavelength. The required cells per wavelength depend on stencil order and acceptable phase error. A higher-order stencil reduces numerical dispersion at fixed spacing, but costs more arithmetic and a wider halo.

A practical workflow is:

1. Estimate `f_max` from the source spectrum after any filtering.
2. Estimate the slowest material from the largest plausible `epsilon * mu`.
3. Start with a conservative spacing and stencil 4.
4. Repeat the same homogeneous or layered test at a finer grid.
5. Accept the coarser grid only when arrival time, phase, and amplitude agree within the study's tolerance.

This convergence comparison is more reliable than adopting one universal cells-per-wavelength number.

## 2D or 3D

Use `MaxwellTM` when the geometry and material are invariant in the omitted direction and the TM polarization is appropriate. It evolves `Ey`, `Hx`, and `Hz` on `[y, x]` grids. Use it for rapid survey design, gradient tests, and problems whose physics is genuinely planar.

Use `Maxwell3D` when sources, receivers, targets, or polarization cannot be represented by a 2D slice. It evolves `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, and `Hz` on `[z, y, x]` grids. Memory grows with the volume, number of shots, number of stored time levels, and CPML thickness, so validate the numerical setup in 2D or on a small 3D volume first.

:::caution[2D is a different physical model]
A 2D run is not merely a cheaper 3D run. Geometrical spreading, polarization content, and out-of-plane scattering differ. Use 2D only when that approximation is defensible.
:::

## Shapes and batching

| Quantity | Shared model | Batched models |
| --- | --- | --- |
| 2D material | `[ny, nx]` | `[batch, ny, nx]` |
| 3D material | `[nz, ny, nx]` | `[batch, nz, ny, nx]` |
| Source amplitude | `[shots, sources, nt]` | `[batch, shots, sources, nt]` |
| 2D location | `[shots, points, 2]` | `[batch, shots, points, 2]` |
| 3D location | `[shots, points, 3]` | `[batch, shots, points, 3]` |
| Receiver output | `[nt, shots, receivers]` | `[nt, batch, shots, receivers]` |

A shared model means every shot propagates through the same material tensors. A batched model evaluates several models in one call. Do not add a leading singleton model axis unless you intend to use batched-model semantics.

## A layered 2D model

```python
import torch
import tide

ny, nx = 120, 180
epsilon = torch.full((ny, nx), 4.0)
epsilon[55:] = 6.5
epsilon[85:] = 9.0

sigma = torch.zeros_like(epsilon)
sigma[85:] = 0.01
mu = torch.ones_like(epsilon)

model = tide.EMModel(epsilon, sigma, mu)
```

The horizontal interface positions above are grid indices. If `spacing=0.02`, row 55 corresponds to 1.1 m from the chosen grid origin. Keep the index-to-coordinate conversion in one place in the survey code so model construction and acquisition geometry cannot drift apart.

## Time sampling and CFL sub-stepping

`Discretization.dt` is the source and receiver sampling interval exposed to user code. TIDE calculates a stable internal time step from spacing and maximum velocity. If required, it advances several internal steps per external sample, upsamples the source, and downsamples receiver traces.

Sub-stepping protects stability but is not free. A large step ratio multiplies propagation work. If a run is unexpectedly slow, inspect the CFL ratio before tuning kernels or increasing shot batch size.

## Boundary and interior design

CPML lies outside the physical model after padding. Keep sources and receivers in the physical model coordinates accepted by the API. Place targets far enough from the PML that the absorbing layer does not alter the time window of interest. Increase CPML width or adjust the experiment when late boundary reflections contaminate receiver traces.

## Constraints for inversion

Direct optimizer updates can produce nonphysical values. Safer parameterizations map an unconstrained tensor into a physical interval:

```python
class BoundedPermittivity(torch.nn.Module):
    def __init__(self, initial, lower, upper):
        super().__init__()
        scaled = (initial - lower) / (upper - lower)
        self.raw = torch.nn.Parameter(torch.logit(scaled))
        self.lower = lower
        self.upper = upper

    def forward(self):
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.raw)
```

PyTorch backpropagates through this transformation. The optimizer updates `raw`, while the Maxwell operator always receives positive, bounded permittivity.

## Model validation checklist

Before a production run:

1. Confirm units for spacing, `dt`, conductivity, and source frequency.
2. Confirm `[y, x]` or `[z, y, x]` axis order.
3. Check that all source and receiver indices are inside model bounds.
4. Compare a small result at two grid resolutions or stencil orders.
5. Inspect a homogeneous-medium arrival against the expected velocity.
6. Verify that CPML reflections do not enter the analysis window.
7. Run a directional derivative test before trusting inversion gradients.

Continue with [sources and receivers](/tide-GPR/guides/sources-receivers/), [boundaries and CPML](/tide-GPR/guides/boundaries/), and [validation](/tide-GPR/guides/validation/).
