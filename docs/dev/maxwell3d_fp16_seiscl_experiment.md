# Maxwell 3D FP16 and SeisCL-style half2 experiment

_RTX 4070 validation of FP16 field storage and paired-cell execution · 2026-07-20_

---

## 📋 Abstract

This experiment tested whether Tide's 3D Maxwell forward solver benefits from
FP16 wavefield storage and whether SeisCL's paired-cell `half2` strategy adds a
material gain. FP16 I/O is useful for medium and large CUDA workloads, reaching
`1.13–1.34x` over native FP32 in the tested cases. Pairing adjacent x-cells adds
only `1.00–1.10x` over scalar FP16, with the largest gain when CPML is disabled.
It also accumulates more long-record error. The scalar FP16 path is worth
retaining as an experimental forward mode; the `half2` path should remain
behind `TIDE_EM3D_FP16_HALF2=1` and should not become a public compute mode.

## 🔬 Methodology

### Implementation under test

Three forward paths were compared:

| Path | Primary fields | Arithmetic | Cells per thread | CPML memories |
| --- | --- | --- | ---: | --- |
| Native | FP32 | FP32 | 1 | FP32 |
| Scalar FP16 I/O | FP16 | FP32 | 1 | FP32 |
| Paired FP16 I/O | FP16 | FP32 `float2` | 2 x-cells | FP32 |

The paired path follows the central mechanism in
[`header_FD_fp16.cl`](../../references/SeisCL/src/header_FD_fp16.cl): two
adjacent values are loaded as a pair, converted to FP32 lanes for stencil
arithmetic, and rounded back to FP16. SeisCL also stores its `psi` absorbing
boundary state using the precision-selected paired type, as visible in
[`update_v3D_half2.cl`](../../references/SeisCL/src/update_v3D_half2.cl). Tide's
prototype deliberately leaves all 12 CPML memories in FP32 to isolate paired
primary-field execution.

### Hardware and workload

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070, 12 GB |
| Compute capability | 8.9 |
| CUDA | 13.0 |
| PyTorch | 2.12.0+cu130 |
| Default time steps | 300 |
| Default stencil | Fourth order |
| Timing | Synchronized CUDA events, median after warmup |

The main benchmark uses a two-layer permittivity model. Each measurement
includes padding, propagation, receiver recording, and conversion of returned
FP16 field states to the public FP32 output format.

### Reproduction

Run the comparison benchmark with:

```bash
uv run python benchmarks/maxwell3d_fp16_io.py \
  --shape 70,70,70 --nt 300 --shots 1 --pml-width 20 --stencil 4
```

Run the generic launch benchmark with paired execution enabled using:

```bash
uv run python benchmarks/maxwell3d_cuda_launch.py \
  --compute-mode fp16_io --fp16-half2 --shots 4
```

## 📊 Findings

### End-to-end speed

| Model / shots | Native ms | Scalar FP16 ms | Paired FP16 ms | Scalar speedup | Paired speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32³`, PML 10, 1 | 15.12 | 23.67 | 25.45 | `0.64x` | `0.59x` |
| `70³`, PML 20, 1 | 123.61 | 99.04 | 96.04 | `1.25x` | `1.29x` |
| `70³`, PML 20, 4 | 470.99 | 415.48 | 390.45 | `1.13x` | `1.21x` |
| `120³`, PML 20, 1 | 324.28 | 285.45 | 273.01 | `1.14x` | `1.19x` |
| `70³`, no PML, 1 | 14.41 | 11.63 | 10.58 | `1.24x` | `1.36x` |

The small workload is dominated by orchestration and returned-state handling,
so reduced field traffic does not pay back its conversion overhead. On useful
3D workloads, scalar FP16 consistently helps. Paired execution is most useful
without CPML: it adds `1.10x` over scalar FP16 there, versus only `1.03x` on the
default one-shot CPML workload. This isolates the remaining FP32 CPML traffic
as a principal limiter.

### Stencil sensitivity

| Stencil | Native ms | Scalar FP16 ms | Paired FP16 ms | Paired vs native |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 119.61 | 89.02 | 88.94 | `1.34x` |
| 4 | 123.61 | 99.04 | 96.04 | `1.29x` |
| 8 | 130.63 | 112.88 | 107.92 | `1.21x` |

The benefit declines as stencil order increases because more neighbor values
must be converted and combined per output cell. The paired implementation
remains fastest, but its incremental value over scalar FP16 is small.

### Numerical behavior

The CUDA test suite verifies paired-versus-scalar FP16 behavior for stencil
orders 2, 4, 6, and 8. All seven FP16 tests pass. Short cases are bitwise equal,
but long propagation exposes different FP32 operation ordering:

| Steps / frequency | Scalar relative L2 | Paired relative L2 | Paired correlation |
| --- | ---: | ---: | ---: |
| 300 / 160 MHz | 0.00062 | 0.00062 | 0.9999999 |
| 1,000 / 160 MHz | 0.00131 | 0.00157 | 0.9999988 |
| 3,000 / 80 MHz | 0.00314 | 0.00598 | 0.9999821 |

Both FP16 paths stayed finite in the layered conductive test. Paired execution
approximately doubled the long-record relative error compared with scalar
FP16, even though waveform correlation remained high. This makes scalar FP16
the safer default experimental path.

### Memory behavior

FP16 reduces the six primary working fields internally, but the public solver
returns FP32 states and keeps material coefficients plus 12 CPML memories in
FP32. Consequently, measured end-to-end peak incremental allocation did not
fall: for the `70³` one-shot case it was about 132 MiB for native and 136 MiB
for both FP16 variants. FP16 currently provides a bandwidth optimization, not
an observable peak-memory reduction at the API boundary.

## 💡 Interpretation

The experiment validates FP16 field storage but does not validate a broad
"make 3D half precision" direction. The measured hierarchy is:

1. Reducing primary-field traffic provides the main gain
2. Pairing two cells reduces indexing and conversion overhead modestly
3. FP32 CPML traffic caps the paired path on boundary-heavy padded domains
4. Returning FP32 states removes the user-visible memory benefit
5. Longer propagation needs stricter accuracy checks than the current short tests

The next high-information experiment is FP16 storage for CPML auxiliary state
with FP32 computation, matching SeisCL more closely. It must be evaluated first
for absorbing-boundary reflection and long-record stability. Native-half
arithmetic and adjoint support should remain out of scope until that experiment
shows a clear end-to-end gain.

## 🎯 Decision

- Keep `compute_mode="fp16_io"` as CUDA forward-only experimental functionality
- Keep paired execution behind `TIDE_EM3D_FP16_HALF2=1`
- Do not make paired execution the default or add a public compute-mode value
- Do not extend FP16 to gradients, dispersion, or callbacks from this evidence
- Require workload-specific accuracy checks before using FP16-generated data

