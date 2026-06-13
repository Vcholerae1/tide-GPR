"""3D image-domain LSRTM with 3D PSF probes.

This example is the image-domain counterpart of ``lsrtm_pcgnr_over_new.py``.
It generates the same 3D modelling data and RTM image from Tide, then replaces
the data-domain PCGNR solve with an SMIwiz-style image-domain PSF solve:

    min_m 0.5 || H_psf m - m_rtm ||^2

The PSF probe is a sparse 3D comb in ``[z, y, x]``. The image-domain operator
uses local trilinear PSF interpolation and includes crossline ``x`` coupling.
The image residual is explicitly normalized with source illumination estimated
from the initial model.

Run from the repository root:

    uv run python examples/image_domain_lsrtm_pcgnr_over_new.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tide
import torch

from tide import backend_utils


C0 = 299_792_458.0
SHOT_DISPLAY_QUANTILE = 0.92

DEVICE = "auto"
DTYPE = "float32"
BACKEND = "native"

MODEL_PATH = Path("examples/overthrust_new.h5")
DATASET = "m"
NZ = 100
NY = 200
NX = 50

NT = 1200
N_LINES = 1
SHOTS_PER_LINE = 99
BATCH_SIZE = 1
D_SOURCE = 2
FIRST_SOURCE = 1
RECEIVER_OFFSET = 2
LINE_XS = None

DX = 0.02
DT = 2.5e-11
FREQ = 5e8
AIR_LAYERS = 5
PML_WIDTH = 10
STENCIL = 4

SMOOTH_SIGMA = 10.0
SMOOTH_TRUNCATE = 3.0
MASK_NONIMAGING_ZONE = True
MODEL_TAPER_CELLS = 0
MODEL_GRADIENT_SAMPLING_INTERVAL = 2
ILLUMINATION_WAVEFIELD_SAMPLING_INTERVAL = 8

PSF_CW_Z = 10
PSF_CW_Y = 10
PSF_CW_X = 5
IMAGE_DOMAIN_ITERS = 8
PRECONDITION_DAMPING = 1e-3
SYMMETRIZE_PSF = True
IMAGE_DOMAIN_LATERAL_TAPER_CELLS = 3
DEBLUR_IMAGE_DOMAIN = True
DEBLUR_FILTER_Z = 5
DEBLUR_FILTER_Y = 5
DEBLUR_FILTER_X = 3
DEBLUR_PATCH_Z = 19
DEBLUR_PATCH_Y = 19
DEBLUR_PATCH_X = 9
DEBLUR_DAMPING = 1e-3
ILLUMINATION_COMPENSATION = True
ILLUMINATION_COMPENSATION_FLOOR = 1e-3
ILLUMINATION_COMPENSATION_POWER = 1.0
ILLUMINATION_COMPENSATION_MAX_GAIN = 50.0

DATA_MUTE_MARGIN_CYCLES = 1.0
DATA_MUTE_TAPER_SAMPLES = 12
NO_DATA_MUTE = False

STORAGE_MODE = "device"
STORAGE_COMPRESSION = "bf16"
CHECK_ADJOINT = False
CHECK_PSF_ADJOINT = True
OUTPUT_DIR = Path("outputs/over_image_domain_lsrtm_1lines_sigma10")


@dataclass(frozen=True, slots=True)
class ImageDomain3DConfig:
    device: str
    dtype: str
    backend: str
    model_path: Path
    dataset: str
    nz: int | None
    ny: int | None
    nx: int | None
    nt: int | None
    n_lines: int
    shots_per_line: int
    batch_size: int
    d_source: int
    first_source: int
    receiver_offset: int
    line_xs: tuple[int, ...] | None
    dx: float
    dt: float
    freq: float
    air_layers: int
    pml_width: int
    stencil: int
    smooth_sigma: float
    smooth_truncate: float
    mask_nonimaging_zone: bool
    model_taper_cells: int
    model_gradient_sampling_interval: int
    illumination_wavefield_sampling_interval: int
    psf_cw_z: int
    psf_cw_y: int
    psf_cw_x: int
    image_domain_iters: int
    precondition_damping: float
    symmetrize_psf: bool
    image_domain_lateral_taper_cells: int
    deblur_image_domain: bool
    deblur_filter_z: int
    deblur_filter_y: int
    deblur_filter_x: int
    deblur_patch_z: int
    deblur_patch_y: int
    deblur_patch_x: int
    deblur_damping: float
    illumination_compensation: bool
    illumination_compensation_floor: float
    illumination_compensation_power: float
    illumination_compensation_max_gain: float
    data_mute_margin_cycles: float
    data_mute_taper_samples: int
    no_data_mute: bool
    storage_mode: str
    storage_compression: str
    check_adjoint: bool
    check_psf_adjoint: bool
    output_dir: Path


CONFIG = ImageDomain3DConfig(
    device=DEVICE,
    dtype=DTYPE,
    backend=BACKEND,
    model_path=MODEL_PATH,
    dataset=DATASET,
    nz=NZ,
    ny=NY,
    nx=NX,
    nt=NT,
    n_lines=N_LINES,
    shots_per_line=SHOTS_PER_LINE,
    batch_size=BATCH_SIZE,
    d_source=D_SOURCE,
    first_source=FIRST_SOURCE,
    receiver_offset=RECEIVER_OFFSET,
    line_xs=LINE_XS,
    dx=DX,
    dt=DT,
    freq=FREQ,
    air_layers=AIR_LAYERS,
    pml_width=PML_WIDTH,
    stencil=STENCIL,
    smooth_sigma=SMOOTH_SIGMA,
    smooth_truncate=SMOOTH_TRUNCATE,
    mask_nonimaging_zone=MASK_NONIMAGING_ZONE,
    model_taper_cells=MODEL_TAPER_CELLS,
    model_gradient_sampling_interval=MODEL_GRADIENT_SAMPLING_INTERVAL,
    illumination_wavefield_sampling_interval=ILLUMINATION_WAVEFIELD_SAMPLING_INTERVAL,
    psf_cw_z=PSF_CW_Z,
    psf_cw_y=PSF_CW_Y,
    psf_cw_x=PSF_CW_X,
    image_domain_iters=IMAGE_DOMAIN_ITERS,
    precondition_damping=PRECONDITION_DAMPING,
    symmetrize_psf=SYMMETRIZE_PSF,
    image_domain_lateral_taper_cells=IMAGE_DOMAIN_LATERAL_TAPER_CELLS,
    deblur_image_domain=DEBLUR_IMAGE_DOMAIN,
    deblur_filter_z=DEBLUR_FILTER_Z,
    deblur_filter_y=DEBLUR_FILTER_Y,
    deblur_filter_x=DEBLUR_FILTER_X,
    deblur_patch_z=DEBLUR_PATCH_Z,
    deblur_patch_y=DEBLUR_PATCH_Y,
    deblur_patch_x=DEBLUR_PATCH_X,
    deblur_damping=DEBLUR_DAMPING,
    illumination_compensation=ILLUMINATION_COMPENSATION,
    illumination_compensation_floor=ILLUMINATION_COMPENSATION_FLOOR,
    illumination_compensation_power=ILLUMINATION_COMPENSATION_POWER,
    illumination_compensation_max_gain=ILLUMINATION_COMPENSATION_MAX_GAIN,
    data_mute_margin_cycles=DATA_MUTE_MARGIN_CYCLES,
    data_mute_taper_samples=DATA_MUTE_TAPER_SAMPLES,
    no_data_mute=NO_DATA_MUTE,
    storage_mode=STORAGE_MODE,
    storage_compression=STORAGE_COMPRESSION,
    check_adjoint=CHECK_ADJOINT,
    check_psf_adjoint=CHECK_PSF_ADJOINT,
    output_dir=OUTPUT_DIR,
)


@dataclass(slots=True)
class ImageDomain3DCase:
    epsilon_true: torch.Tensor
    epsilon_background: torch.Tensor
    sigma: torch.Tensor
    mu: torch.Tensor
    depsilon_true: torch.Tensor
    source_amplitude: torch.Tensor
    source_locations: torch.Tensor
    receiver_locations: torch.Tensor
    active_mask: torch.Tensor
    model_weights: torch.Tensor
    line_mask: torch.Tensor
    data_weights: torch.Tensor
    model_gradient_sampling_interval: int
    illumination_wavefield_sampling_interval: int
    batch_size: int
    dx: float
    dt: float
    freq: float
    pml_width: int
    stencil: int
    max_vel: float
    python_backend: bool | Literal["eager", "jit", "compile"]
    storage_mode: str
    storage_compression: str
    model_label: str
    line_xs: tuple[int, ...]


@dataclass(slots=True)
class ImageDomainResult:
    image: torch.Tensor
    image_residual: torch.Tensor
    normal_residual: torch.Tensor
    loss_history: list[float]
    relative_residual_history: list[float]
    status: str
    iterations: int


@dataclass(frozen=True, slots=True)
class DeblurFilterBank3D:
    centers: torch.Tensor
    filters: torch.Tensor
    filter_shape: tuple[int, int, int]
    patch_shape: tuple[int, int, int]
    damping: float


@dataclass(frozen=True, slots=True)
class CenterInterpolationPlan3D:
    shape: tuple[int, int, int]
    z0: torch.Tensor
    y0: torch.Tensor
    x0: torch.Tensor
    z1: torch.Tensor
    y1: torch.Tensor
    x1: torch.Tensor
    wz: torch.Tensor
    wy: torch.Tensor
    wx: torch.Tensor


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f} s"


def timed_stage(label: str, device: torch.device, fn):
    sync_device(device)
    start = perf_counter()
    result = fn()
    sync_device(device)
    elapsed = perf_counter() - start
    print(f"{label} time: {format_seconds(elapsed)}")
    return result, elapsed


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_backend(
    backend_arg: str,
) -> bool | Literal["eager", "jit", "compile"]:
    if backend_arg == "python":
        raise SystemExit(
            "The Python backend does not support this 3D image-domain example. "
            "Use BACKEND='native' with CUDA."
        )
    if backend_arg == "native":
        if backend_utils is None or not backend_utils.is_backend_available():
            raise SystemExit("Tide native backend was requested but is not available.")
        return False
    if backend_utils is None or not backend_utils.is_backend_available():
        raise SystemExit("Tide native backend is not available.")
    return False


def crop_center_3d(
    model: np.ndarray,
    nz: int | None,
    ny: int | None,
    nx: int | None,
) -> np.ndarray:
    if model.ndim != 3:
        raise ValueError(f"Expected a 3D model, got shape={model.shape}.")
    out_shape = (
        model.shape[0] if nz is None else nz,
        model.shape[1] if ny is None else ny,
        model.shape[2] if nx is None else nx,
    )
    if any(v <= 0 for v in out_shape):
        raise ValueError("nz, ny, and nx must be positive when provided.")

    slices: list[slice] = []
    pads: list[tuple[int, int]] = []
    for src_len, dst_len in zip(model.shape, out_shape, strict=True):
        if src_len >= dst_len:
            start = (src_len - dst_len) // 2
            slices.append(slice(start, start + dst_len))
            pads.append((0, 0))
        else:
            slices.append(slice(0, src_len))
            before = (dst_len - src_len) // 2
            pads.append((before, dst_len - src_len - before))

    cropped = model[tuple(slices)]
    if any(pad != (0, 0) for pad in pads):
        cropped = np.pad(cropped, pads, mode="edge")
    return cropped.astype(np.float32, copy=False)


def load_h5_model(
    path: Path,
    dataset: str,
    nz: int | None,
    ny: int | None,
    nx: int | None, 
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise KeyError(f"Dataset {dataset!r} not found in {path}.")
        model = h5[dataset][...]
    return crop_center_3d(model, nz, ny, nx)


def gaussian_smooth(
    epsilon: torch.Tensor,
    *,
    air_layers: int,
    sigma: float,
    truncate: float,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError("smooth_sigma must be positive.")
    if truncate <= 0.0:
        raise ValueError("smooth_truncate must be positive.")

    background = epsilon.clone()
    if air_layers >= epsilon.shape[0]:
        return background.contiguous()

    radius = max(1, int(round(truncate * sigma)))
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=epsilon.device,
        dtype=epsilon.dtype,
    )
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    subsurface = epsilon[air_layers:, :, :].unsqueeze(0).unsqueeze(0)
    kernel_z = kernel.view(1, 1, -1, 1, 1)
    kernel_y = kernel.view(1, 1, 1, -1, 1)
    kernel_x = kernel.view(1, 1, 1, 1, -1)

    smoothed = torch.nn.functional.conv3d(
        torch.nn.functional.pad(
            subsurface,
            (0, 0, 0, 0, radius, radius),
            mode="replicate",
        ),
        kernel_z,
    )
    smoothed = torch.nn.functional.conv3d(
        torch.nn.functional.pad(
            smoothed,
            (0, 0, radius, radius, 0, 0),
            mode="replicate",
        ),
        kernel_y,
    )
    smoothed = torch.nn.functional.conv3d(
        torch.nn.functional.pad(
            smoothed,
            (radius, radius, 0, 0, 0, 0),
            mode="replicate",
        ),
        kernel_x,
    )
    background[air_layers:, :, :] = smoothed.squeeze(0).squeeze(0)
    background[:air_layers, :, :] = epsilon[:air_layers, :, :]
    return background.contiguous()


def cosine_ramp(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    grid = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(torch.pi * grid)


def make_active_mask(
    nz: int,
    ny: int,
    nx: int,
    *,
    air_layers: int,
    pml_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((nz, ny, nx), device=device, dtype=dtype)
    z0 = min(air_layers + 2, pml_width)
    if z0 < nz:
        mask[z0:, :, :] = 1.0
    return mask.contiguous()


def make_model_weights(
    active_mask: torch.Tensor,
    *,
    air_layers: int,
    pml_width: int,
    taper_cells: int,
) -> torch.Tensor:
    weights = active_mask.clone()
    if taper_cells <= 0:
        return weights

    nz, ny, nx = weights.shape
    z0 = max(air_layers + 2, pml_width)
    z1 = nz - pml_width
    y0 = pml_width
    y1 = ny - pml_width
    x0 = pml_width
    x1 = nx - pml_width
    ramp = cosine_ramp(taper_cells, device=weights.device, dtype=weights.dtype)

    for i in range(min(taper_cells, max(z1 - z0, 0))):
        weights[z0 + i, y0:y1, x0:x1] *= ramp[i]
    for i in range(min(taper_cells, max(y1 - y0, 0))):
        weights[z0:z1, y0 + i, x0:x1] *= ramp[i]
        weights[z0:z1, y1 - 1 - i, x0:x1] *= ramp[i]
    for i in range(min(taper_cells, max(x1 - x0, 0))):
        weights[z0:z1, y0:y1, x0 + i] *= ramp[i]
        weights[z0:z1, y0:y1, x1 - 1 - i] *= ramp[i]
    return weights.contiguous()


def make_line_mask(
    active_mask: torch.Tensor,
    line_xs: tuple[int, ...],
) -> torch.Tensor:
    line_mask = torch.zeros_like(active_mask)
    for line_x in line_xs:
        line_mask[:, :, line_x] = active_mask[:, :, line_x]
    return line_mask.contiguous()


def resolve_model_gradient_sampling_interval(sampling_interval: int) -> int:
    sampling_interval = tide.validate_model_gradient_sampling_interval(
        sampling_interval
    )
    return max(1, sampling_interval)


def resolve_illumination_sampling_interval(
    sampling_interval: int,
    *,
    dt: float,
    freq: float,
) -> int:
    if sampling_interval < 0:
        raise ValueError("illumination_wavefield_sampling_interval must be >= 0.")
    if sampling_interval > 0:
        return sampling_interval
    if dt <= 0.0 or freq <= 0.0:
        return 1
    reference_dt_samp = 1.0 / (18.0 * freq)
    return max(1, int(np.floor(reference_dt_samp / dt)))


def downsample_receiver_data(data: torch.Tensor, sampling_interval: int) -> torch.Tensor:
    if sampling_interval <= 1 or data.numel() == 0:
        return data.contiguous()
    return tide.downsample(
        torch.movedim(data, 0, -1),
        sampling_interval,
    ).movedim(-1, 0).contiguous()


def match_data_sampling(case: ImageDomain3DCase, data: torch.Tensor) -> torch.Tensor:
    if data.shape[0] == case.data_weights.shape[0]:
        return data.contiguous()
    return downsample_receiver_data(data, case.model_gradient_sampling_interval)


def subsample_time_grid(data: torch.Tensor, sampling_interval: int) -> torch.Tensor:
    if sampling_interval <= 1 or data.numel() == 0:
        return data.contiguous()
    nt_down = data.shape[0] // sampling_interval
    return data[: nt_down * sampling_interval : sampling_interval].contiguous()


def build_geometry(
    *,
    nx: int,
    ny: int,
    air_layers: int,
    pml_width: int,
    n_lines: int,
    shots_per_line: int,
    d_source: int,
    first_source: int,
    receiver_offset: int,
    line_xs: tuple[int, ...] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if n_lines <= 0:
        raise ValueError("n_lines must be positive.")
    if shots_per_line <= 0:
        raise ValueError("shots_per_line must be positive.")
    if d_source <= 0:
        raise ValueError("d_source must be positive.")

    source_depth = max(0, air_layers - 3)
    if line_xs is None:
        x0 = min(max(pml_width, 0), nx - 1)
        x1 = nx - 1 - x0
        if x1 < x0:
            x0, x1 = 0, nx - 1
        lines = torch.linspace(x0, x1, n_lines, device=device).round().to(torch.long)
        line_values = tuple(int(v) for v in lines.tolist())
    else:
        if len(line_xs) != n_lines:
            raise ValueError(
                f"line_xs must contain n_lines={n_lines} entries, got {len(line_xs)}."
            )
        line_values = tuple(int(v) for v in line_xs)
    if any(x < 0 or x >= nx for x in line_values):
        raise ValueError(f"line_xs={line_values} contains an x outside nx={nx}.")
    if len(set(line_values)) != len(line_values):
        raise ValueError(f"line_xs={line_values} contains duplicate x positions.")

    source_y = (
        torch.arange(shots_per_line, device=device, dtype=torch.long) * d_source
        + first_source
    )
    receiver_y = source_y + receiver_offset
    if int(source_y.min()) < 0 or int(receiver_y.min()) < 0:
        raise ValueError("Source and receiver y locations must be non-negative.")
    if int(source_y.max()) >= ny or int(receiver_y.max()) >= ny:
        raise ValueError(
            f"One-source/one-receiver geometry exceeds model width ny={ny}: "
            f"max source y={int(source_y.max())}, "
            f"max receiver y={int(receiver_y.max())}."
        )

    nshots = n_lines * shots_per_line
    line_x_tensor = torch.tensor(line_values, device=device, dtype=torch.long)
    shot_line_x = line_x_tensor.repeat_interleave(shots_per_line)
    shot_source_y = source_y.repeat(n_lines)
    shot_receiver_y = receiver_y.repeat(n_lines)

    source_locations = torch.zeros((nshots, 1, 3), device=device, dtype=torch.long)
    source_locations[:, 0, 0] = source_depth
    source_locations[:, 0, 1] = shot_source_y
    source_locations[:, 0, 2] = shot_line_x

    receiver_locations = torch.zeros((nshots, 1, 3), device=device, dtype=torch.long)
    receiver_locations[:, 0, 0] = source_depth + 2
    receiver_locations[:, 0, 1] = shot_receiver_y
    receiver_locations[:, 0, 2] = shot_line_x
    print(
        f"Geometry: line_xs={line_values}, n_lines={n_lines}, "
        f"shots_per_line={shots_per_line}, nshots={nshots}, "
        f"source y=[{int(source_y.min())}, {int(source_y.max())}], "
        f"receiver y=[{int(receiver_y.min())}, {int(receiver_y.max())}]."
    )
    return source_locations, receiver_locations, line_values


def estimate_nt_for_bottom_coverage(
    epsilon: torch.Tensor,
    *,
    dx: float,
    dt: float,
    freq: float,
    source_depth: int,
) -> int:
    deepest_depth = max(epsilon.shape[0] - 1 - source_depth, 1) * dx
    slowest_velocity = C0 / torch.sqrt(epsilon.max()).item()
    two_way_time = 2.0 * deepest_depth / slowest_velocity
    wavelet_margin = 3.0 / freq
    return int(np.ceil((two_way_time + wavelet_margin) / dt))


def report_dispersion_sampling(
    epsilon: torch.Tensor,
    *,
    dx: float,
    dt: float,
    freq: float,
    stencil: int,
) -> None:
    if dx <= 0.0 or freq <= 0.0:
        return
    eps_max = float(epsilon.max().item())
    v_min = C0 / np.sqrt(eps_max)
    ppw_peak = v_min / (freq * dx)
    ppw_tail = v_min / (2.0 * freq * dx)
    samples_per_period = 1.0 / (freq * dt)
    eps_min = float(epsilon.min().clamp_min(torch.finfo(epsilon.dtype).eps).item())
    v_max = C0 / np.sqrt(eps_min)
    courant = v_max * dt * np.sqrt(3.0) / dx
    print(
        "Dispersion check: "
        f"v_min={v_min:.3e} m/s, ppw@f0={ppw_peak:.1f}, "
        f"ppw@2f0={ppw_tail:.1f}, samples/period={samples_per_period:.1f}, "
        f"3D CFL={courant:.2f}, stencil={stencil}."
    )
    if courant > 0.5:
        warnings.warn(
            f"Time step is close to the 3D CFL limit: CFL={courant:.2f}.",
            RuntimeWarning,
        )


def make_data_weights(
    *,
    nt: int,
    dt: float,
    dx: float,
    freq: float,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    mute_margin_cycles: float,
    taper_samples: int,
    enabled: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    nshots, nreceivers = receiver_locations.shape[:2]
    weights = torch.ones(
        (nt, nshots, nreceivers),
        device=source_locations.device,
        dtype=dtype,
    )
    if not enabled:
        return weights
    if taper_samples < 0:
        raise ValueError("data_mute_taper_samples must be >= 0.")

    src = source_locations[:, 0, :].to(dtype=dtype)
    rec = receiver_locations[:, :, :].to(dtype=dtype)
    offsets = ((src[:, None, :] - rec).square().sum(dim=-1).sqrt()) * dx
    direct_time = offsets / C0 + (1.0 + mute_margin_cycles) / freq
    start_samples = torch.round(direct_time / dt).to(torch.long)
    taper = cosine_ramp(max(taper_samples, 1), device=weights.device, dtype=dtype)
    for shot in range(nshots):
        for rec_idx in range(nreceivers):
            start = int(start_samples[shot, rec_idx].item())
            if start >= nt:
                continue
            weights[:start, shot, rec_idx] = 0.0
            if taper_samples > 0:
                end = min(start + taper_samples, nt)
                weights[start:end, shot, rec_idx] = taper[: end - start]
    return weights.contiguous()


def build_case(
    config: ImageDomain3DConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    python_backend: bool | Literal["eager", "jit", "compile"],
) -> ImageDomain3DCase:
    if device.type != "cuda":
        raise SystemExit("This 3D image-domain example is intended for CUDA.")
    model_gradient_sampling_interval = resolve_model_gradient_sampling_interval(
        config.model_gradient_sampling_interval
    )
    illumination_wavefield_sampling_interval = resolve_illumination_sampling_interval(
        config.illumination_wavefield_sampling_interval,
        dt=config.dt,
        freq=config.freq,
    )
    epsilon_np = load_h5_model(
        config.model_path,
        config.dataset,
        config.nz,
        config.ny,
        config.nx,
    )
    epsilon_np[: config.air_layers, :, :] = 1.0
    model_label = f"{config.model_path.stem}:{config.dataset}"
    epsilon_true = torch.as_tensor(epsilon_np, device=device, dtype=dtype).contiguous()
    dt = float(config.dt)
    report_dispersion_sampling(
        epsilon_true,
        dx=config.dx,
        dt=dt,
        freq=config.freq,
        stencil=config.stencil,
    )
    epsilon_background = gaussian_smooth(
        epsilon_true,
        air_layers=config.air_layers,
        sigma=config.smooth_sigma,
        truncate=config.smooth_truncate,
    )
    if config.air_layers < epsilon_background.shape[0]:
        background_value = float(epsilon_background[config.air_layers :, :, :].mean().cpu())
        print(
            "Smoothed initial epsilon "
            f"(sigma={config.smooth_sigma:.2f}, truncate={config.smooth_truncate:.2f}) "
            f"mean={background_value:.3f}."
        )
    sigma = torch.full_like(epsilon_background, 1e-3)
    if config.air_layers > 0:
        sigma[: config.air_layers, :, :] = 0.0
    mu = torch.ones_like(epsilon_background)
    max_vel = float((C0 / torch.sqrt(epsilon_true * mu)).max().item())
    nz_model, ny_model, nx_model = epsilon_true.shape
    if config.mask_nonimaging_zone:
        active_mask = make_active_mask(
            nz_model,
            ny_model,
            nx_model,
            air_layers=config.air_layers,
            pml_width=config.pml_width,
            device=device,
            dtype=dtype,
        )
        model_weights = make_model_weights(
            active_mask,
            air_layers=config.air_layers,
            pml_width=config.pml_width,
            taper_cells=config.model_taper_cells,
        )
    else:
        active_mask = torch.ones_like(epsilon_background)
        model_weights = torch.ones_like(epsilon_background)
    depsilon_true = (epsilon_true - epsilon_background) * active_mask
    source_locations, receiver_locations, line_xs = build_geometry(
        nx=nx_model,
        ny=ny_model,
        air_layers=config.air_layers,
        pml_width=config.pml_width,
        n_lines=config.n_lines,
        shots_per_line=config.shots_per_line,
        d_source=config.d_source,
        first_source=config.first_source,
        receiver_offset=config.receiver_offset,
        line_xs=config.line_xs,
        device=device,
    )
    line_mask = make_line_mask(model_weights, line_xs)
    estimated_nt = estimate_nt_for_bottom_coverage(
        epsilon_true,
        dx=config.dx,
        dt=dt,
        freq=config.freq,
        source_depth=int(source_locations[:, 0, 0].min().item()),
    )
    nt = estimated_nt if config.nt is None else config.nt
    if config.nt is None:
        print(f"Auto-selected nt={nt} ({nt * dt * 1e6:.1f} us) for bottom coverage.")
    elif nt < estimated_nt:
        warnings.warn(
            f"nt={nt} is shorter than the estimated bottom-coverage nt={estimated_nt}.",
            RuntimeWarning,
        )
    wavelet = tide.ricker(
        config.freq,
        nt,
        dt,
        peak_time=1.0 / config.freq,
        device=device,
        dtype=dtype,
    )
    nshots = source_locations.shape[0]
    source_amplitude = wavelet.view(1, 1, nt).repeat(nshots, 1, 1).contiguous()
    data_weights = make_data_weights(
        nt=nt,
        dt=dt,
        dx=config.dx,
        freq=config.freq,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        mute_margin_cycles=config.data_mute_margin_cycles,
        taper_samples=config.data_mute_taper_samples,
        enabled=not config.no_data_mute,
        dtype=dtype,
    )
    data_weights = subsample_time_grid(data_weights, model_gradient_sampling_interval)
    print(
        f"Mini-batch: batch_size={config.batch_size}, "
        f"n_batches={(nshots + config.batch_size - 1) // config.batch_size}, "
        f"model_gradient_sampling_interval={model_gradient_sampling_interval}, "
        f"data_nt={data_weights.shape[0]}."
    )
    return ImageDomain3DCase(
        epsilon_true=epsilon_true,
        epsilon_background=epsilon_background,
        sigma=sigma,
        mu=mu,
        depsilon_true=depsilon_true,
        source_amplitude=source_amplitude,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        active_mask=active_mask,
        model_weights=model_weights,
        line_mask=line_mask,
        data_weights=data_weights,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        illumination_wavefield_sampling_interval=illumination_wavefield_sampling_interval,
        batch_size=config.batch_size,
        dx=config.dx,
        dt=dt,
        freq=config.freq,
        pml_width=config.pml_width,
        stencil=config.stencil,
        max_vel=max_vel,
        python_backend=python_backend,
        storage_mode=config.storage_mode,
        storage_compression=config.storage_compression,
        model_label=model_label,
        line_xs=line_xs,
    )


def make_shot_batches(case: ImageDomain3DCase) -> list[torch.Tensor]:
    shot_indices = torch.arange(
        case.source_locations.shape[0],
        device=case.source_locations.device,
        dtype=torch.long,
    )
    return [
        shot_indices[i : i + case.batch_size]
        for i in range(0, shot_indices.numel(), case.batch_size)
    ]


def forward_data_batch(
    case: ImageDomain3DCase,
    epsilon: torch.Tensor,
    shot_indices: torch.Tensor,
    *,
    model_gradient_sampling_interval: int = 1,
    forward_callback=None,
    callback_frequency: int = 1,
    storage_mode: str | None = None,
    storage_compression: bool | str | None = None,
) -> torch.Tensor:
    requires_grad = torch.is_grad_enabled() and epsilon.requires_grad
    if storage_mode is None:
        storage_mode = case.storage_mode if requires_grad else "none"
    if storage_compression is None:
        storage_compression = case.storage_compression if requires_grad else False
    return tide.maxwell3d(
        epsilon,
        case.sigma,
        case.mu,
        grid_spacing=case.dx,
        dt=case.dt,
        source_amplitude=case.source_amplitude[shot_indices],
        source_location=case.source_locations[shot_indices],
        receiver_location=case.receiver_locations[shot_indices],
        pml_width=case.pml_width,
        stencil=case.stencil,
        max_vel=case.max_vel,
        source_component="ey",
        receiver_component="ey",
        python_backend=case.python_backend,
        storage_mode=storage_mode,
        storage_compression=storage_compression,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        forward_callback=forward_callback,
        callback_frequency=callback_frequency,
    )[-1]


def forward_data(
    case: ImageDomain3DCase,
    epsilon: torch.Tensor,
    *,
    model_gradient_sampling_interval: int = 1,
) -> torch.Tensor:
    batches = [
        forward_data_batch(
            case,
            epsilon,
            shot_indices,
            model_gradient_sampling_interval=model_gradient_sampling_interval,
        )
        for shot_indices in make_shot_batches(case)
    ]
    return match_data_sampling(case, torch.cat(batches, dim=1))


def add_source_energy_to_illumination(illumination: torch.Tensor, state) -> None:
    energy = None
    for component in ("Ex", "Ey", "Ez"):
        field = state.get_wavefield(component, view="inner").detach()
        component_energy = field.square()
        energy = component_energy if energy is None else energy + component_energy
    if energy is None:
        return
    if energy.ndim == illumination.ndim:
        illumination.add_(energy)
    else:
        illumination.add_(energy.sum(dim=0))


def forward_data_with_source_illumination(
    case: ImageDomain3DCase,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    illumination = torch.zeros_like(case.epsilon_background)
    batches = []
    for shot_indices in make_shot_batches(case):

        def callback(state) -> None:
            add_source_energy_to_illumination(illumination, state)

        data = forward_data_batch(
            case,
            epsilon,
            shot_indices,
            forward_callback=callback,
            callback_frequency=case.illumination_wavefield_sampling_interval,
            storage_mode="none",
            storage_compression=False,
        )
        batches.append(match_data_sampling(case, data))
    illumination = (
        illumination
        * case.dt
        * case.illumination_wavefield_sampling_interval
    ).detach()
    return torch.cat(batches, dim=1).contiguous(), illumination


def apply_data_weights(
    case: ImageDomain3DCase,
    data: torch.Tensor,
    shot_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = case.data_weights if shot_indices is None else case.data_weights[:, shot_indices, :]
    return data * weights.to(device=data.device, dtype=data.dtype)


def apply_mask(model: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (model * mask.to(device=model.device, dtype=model.dtype)).contiguous()


def born_forward_batch(
    case: ImageDomain3DCase,
    scatter: torch.Tensor,
    shot_indices: torch.Tensor,
) -> torch.Tensor:
    depsilon = case.model_weights.to(dtype=scatter.dtype) * scatter
    data = tide.born3d(
        case.epsilon_background,
        case.sigma,
        case.mu,
        grid_spacing=case.dx,
        dt=case.dt,
        source_amplitude=case.source_amplitude[shot_indices],
        source_location=case.source_locations[shot_indices],
        receiver_location=case.receiver_locations[shot_indices],
        depsilon=depsilon,
        pml_width=case.pml_width,
        stencil=case.stencil,
        max_vel=case.max_vel,
        parameterization="epsilon_sigma",
        linearize_source=True,
        source_component="ey",
        receiver_component="ey",
        python_backend=case.python_backend,
        storage_mode=case.storage_mode,
        storage_compression=case.storage_compression,
    )[-1]
    return match_data_sampling(case, data)


def born_forward_data(case: ImageDomain3DCase, scatter: torch.Tensor) -> torch.Tensor:
    batches = [
        born_forward_batch(case, scatter, shot_indices)
        for shot_indices in make_shot_batches(case)
    ]
    return torch.cat(batches, dim=1).contiguous()


def adjoint_image(
    case: ImageDomain3DCase,
    data_residual: torch.Tensor,
) -> torch.Tensor:
    image = torch.zeros_like(case.epsilon_background)
    for shot_indices in make_shot_batches(case):
        epsilon_background = case.epsilon_background.detach().clone().requires_grad_(True)
        predicted = apply_data_weights(
            case,
            match_data_sampling(
                case,
                forward_data_batch(
                    case,
                    epsilon_background,
                    shot_indices,
                    model_gradient_sampling_interval=case.model_gradient_sampling_interval,
                ),
            ),
            shot_indices,
        )
        rhs = data_residual[:, shot_indices, :].detach()
        objective = torch.sum(predicted * rhs)
        (grad_eps,) = torch.autograd.grad(objective, epsilon_background)
        image = image + (grad_eps * case.model_weights).detach()
    return image.detach()


def weighted_born_forward(case: ImageDomain3DCase, image: torch.Tensor) -> torch.Tensor:
    return apply_data_weights(case, born_forward_data(case, image)).detach()


def normal_image(case: ImageDomain3DCase, image: torch.Tensor) -> torch.Tensor:
    return adjoint_image(case, weighted_born_forward(case, image))


def relative_rms(error: torch.Tensor, reference: torch.Tensor) -> float:
    denom = reference.pow(2).mean().sqrt().clamp_min(torch.finfo(reference.dtype).eps)
    return float((error.pow(2).mean().sqrt() / denom).detach().cpu())


def cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_norm = torch.linalg.norm(lhs).clamp_min(torch.finfo(lhs.dtype).eps)
    rhs_norm = torch.linalg.norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    return float((torch.sum(lhs * rhs) / (lhs_norm * rhs_norm)).detach().cpu())


def dot_product(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.dot(lhs.reshape(-1).double(), rhs.reshape(-1).double())


def check_adjoint(case: ImageDomain3DCase) -> float:
    generator = torch.Generator(device=case.epsilon_background.device)
    generator.manual_seed(20260506)
    model_vec = torch.randn(
        case.epsilon_background.shape,
        generator=generator,
        device=case.epsilon_background.device,
        dtype=case.epsilon_background.dtype,
    )
    model_vec = model_vec * case.model_weights
    with torch.no_grad():
        data_vec = torch.randn(
            case.data_weights.shape,
            generator=generator,
            device=case.epsilon_background.device,
            dtype=case.epsilon_background.dtype,
        )
        j_model = apply_data_weights(case, born_forward_data(case, model_vec)).detach()
    jt_data = adjoint_image(case, data_vec)
    lhs = torch.sum(j_model * data_vec)
    rhs = torch.sum(model_vec * jt_data)
    denom = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(torch.finfo(lhs.dtype).eps)
    return float(((lhs - rhs).abs() / denom).detach().cpu())


def psf_probe_origins(
    case: ImageDomain3DCase,
    image_mask: torch.Tensor,
) -> tuple[int, int, int]:
    active_indices = torch.nonzero(image_mask > 0, as_tuple=False)
    if active_indices.numel() == 0:
        raise ValueError("image_mask has no active cells.")
    origin_z = int(active_indices[:, 0].min().item())
    origin_y = int(case.source_locations[:, 0, 1].min().item())
    origin_x = min(case.line_xs)
    nz, ny, nx = image_mask.shape
    return (
        min(max(origin_z, 0), nz - 1),
        min(max(origin_y, 0), ny - 1),
        min(max(origin_x, 0), nx - 1),
    )


def regular_probe_positions(length: int, origin: int, step: int) -> range:
    return range(origin % step, length, step)


def balanced_probe_origin(length: int, step: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    best_origin = 0
    best_score: tuple[int, int] | None = None
    for origin in range(min(step, length)):
        positions = range(origin, length, step)
        last = origin + (len(positions) - 1) * step
        left_margin = origin
        right_margin = length - 1 - last
        score = (abs(left_margin - right_margin), -left_margin)
        if best_score is None or score < best_score:
            best_origin = origin
            best_score = score
    return best_origin


def make_3d_psf_probe(
    case: ImageDomain3DCase,
    *,
    cw_z: int,
    cw_y: int,
    cw_x: int,
    image_mask: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    if cw_z <= 0 or cw_y <= 0 or cw_x <= 0:
        raise ValueError("PSF_CW_Z, PSF_CW_Y, and PSF_CW_X must be positive.")
    probe = torch.zeros_like(case.epsilon_background)
    active = image_mask.to(device=probe.device, dtype=torch.bool)
    nz, ny, nx = probe.shape
    origins = psf_probe_origins(case, active)
    origin_z, _, origin_x = origins
    origin_y = balanced_probe_origin(ny, cw_y)
    origins = (origin_z, origin_y, origin_x)
    for z in regular_probe_positions(nz, origin_z, cw_z):
        for y in regular_probe_positions(ny, origin_y, cw_y):
            for x in regular_probe_positions(nx, origin_x, cw_x):
                if bool(active[z, y, x]):
                    probe[z, y, x] = 1.0
    return probe.contiguous(), origins


def build_illumination_compensation(
    source_illumination: torch.Tensor,
    image_mask: torch.Tensor,
    *,
    enabled: bool,
    floor_ratio: float,
    power: float,
    max_gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if floor_ratio < 0.0:
        raise ValueError("illumination compensation floor must be >= 0.")
    if power < 0.0:
        raise ValueError("illumination compensation power must be >= 0.")
    if max_gain <= 0.0:
        raise ValueError("illumination compensation max gain must be positive.")

    active = image_mask > 0
    illumination = torch.nan_to_num(
        source_illumination.to(device=image_mask.device, dtype=image_mask.dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    illumination = torch.where(
        active,
        illumination * image_mask.to(dtype=illumination.dtype),
        torch.zeros_like(illumination),
    )
    compensation = torch.zeros_like(illumination)
    if not enabled:
        compensation[active] = 1.0
        return illumination.detach(), compensation.detach()

    usable = active & torch.isfinite(illumination) & (illumination > 0)
    if not bool(usable.any().item()):
        warnings.warn(
            "Initial-model source illumination is zero in the active image region; "
            "using identity image-domain illumination compensation.",
            RuntimeWarning,
        )
        compensation[active] = 1.0
        return illumination.detach(), compensation.detach()

    active_values = illumination[usable]
    eps = torch.finfo(illumination.dtype).eps
    floor = (active_values.max() * floor_ratio).clamp_min(eps)
    safe_illumination = illumination + floor
    scale = torch.median(safe_illumination[usable]).clamp_min(floor)
    gain = torch.pow(scale / safe_illumination, power)
    gain = torch.clamp(gain, max=max_gain)
    compensation[active] = gain[active]
    return illumination.detach(), compensation.detach()


def lateral_cosine_taper_3d(mask: torch.Tensor, cells: int) -> torch.Tensor:
    if cells < 0:
        raise ValueError("lateral taper cells must be >= 0.")
    if mask.ndim != 3:
        raise ValueError("mask must have shape [z, y, x].")
    if cells == 0 or mask.shape[1] <= 1:
        return torch.ones_like(mask)
    ny = mask.shape[1]
    y = torch.arange(ny, device=mask.device, dtype=mask.dtype)
    distance = torch.minimum(y, (ny - 1) - y).clamp_min(0)
    phase = (distance / float(cells)).clamp(0.0, 1.0)
    taper_y = 0.5 - 0.5 * torch.cos(torch.pi * phase)
    return taper_y.reshape(1, ny, 1).expand_as(mask)


def validate_positive_odd_shape(name: str, shape: tuple[int, int, int]) -> None:
    if any(v <= 0 for v in shape):
        raise ValueError(f"{name} dimensions must be positive.")
    if any(v % 2 != 1 for v in shape):
        raise ValueError(f"{name} dimensions must be odd.")


def centered_tensor_patch_3d(
    image: torch.Tensor,
    center: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape [z, y, x].")
    validate_positive_odd_shape("patch_shape", patch_shape)
    patch = torch.zeros(patch_shape, device=image.device, dtype=image.dtype)
    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for c, size, dim in zip(center, patch_shape, image.shape, strict=True):
        start = int(c) - size // 2
        stop = start + size
        src_start = max(start, 0)
        src_stop = min(stop, dim)
        dst_start = src_start - start
        dst_stop = dst_start + max(src_stop - src_start, 0)
        src_slices.append(slice(src_start, src_stop))
        dst_slices.append(slice(dst_start, dst_stop))
    if all(s.stop > s.start for s in src_slices):
        patch[tuple(dst_slices)] = image[tuple(src_slices)]
    return patch


def shift_slices_for_same_convolution(
    length: int,
    offset: int,
) -> tuple[slice, slice] | None:
    dst_start = max(0, offset)
    dst_stop = min(length, length + offset)
    if dst_stop <= dst_start:
        return None
    src_start = dst_start - offset
    src_stop = dst_stop - offset
    return slice(dst_start, dst_stop), slice(src_start, src_stop)


def same_convolution_shift_3d(
    image: torch.Tensor,
    dz: int,
    dy: int,
    dx: int,
) -> torch.Tensor:
    out = torch.zeros_like(image)
    z_slices = shift_slices_for_same_convolution(image.shape[0], dz)
    y_slices = shift_slices_for_same_convolution(image.shape[1], dy)
    x_slices = shift_slices_for_same_convolution(image.shape[2], dx)
    if z_slices is None or y_slices is None or x_slices is None:
        return out
    dst_z, src_z = z_slices
    dst_y, src_y = y_slices
    dst_x, src_x = x_slices
    out[dst_z, dst_y, dst_x] = image[src_z, src_y, src_x]
    return out


def build_same_convolution_matrix_3d(
    image_patch: torch.Tensor,
    filter_shape: tuple[int, int, int],
) -> torch.Tensor:
    if image_patch.ndim != 3:
        raise ValueError("image_patch must have shape [z, y, x].")
    validate_positive_odd_shape("filter_shape", filter_shape)
    kz, ky, kx = filter_shape
    cz, cy, cx = kz // 2, ky // 2, kx // 2
    columns: list[torch.Tensor] = []
    for fz in range(kz):
        for fy in range(ky):
            for fx in range(kx):
                shifted = same_convolution_shift_3d(
                    image_patch,
                    fz - cz,
                    fy - cy,
                    fx - cx,
                )
                columns.append(shifted.reshape(-1))
    return torch.stack(columns, dim=1)


def identity_deblur_filter_3d(
    filter_shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    validate_positive_odd_shape("filter_shape", filter_shape)
    filt = torch.zeros(filter_shape, device=device, dtype=dtype)
    filt[filter_shape[0] // 2, filter_shape[1] // 2, filter_shape[2] // 2] = 1.0
    return filt


def estimate_deblur_filter_3d(
    psf_patch: torch.Tensor,
    reference_patch: torch.Tensor,
    *,
    filter_shape: tuple[int, int, int],
    damping: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if psf_patch.shape != reference_patch.shape:
        raise ValueError("psf_patch and reference_patch must have the same shape.")
    if valid_mask is not None and valid_mask.shape != psf_patch.shape:
        raise ValueError("valid_mask must have the same shape as psf_patch.")
    validate_positive_odd_shape("filter_shape", filter_shape)
    if damping < 0.0:
        raise ValueError("damping must be >= 0.")

    original_dtype = psf_patch.dtype
    psf_work = torch.nan_to_num(psf_patch.detach().double())
    reference_work = torch.nan_to_num(reference_patch.detach().double())
    valid_work = None
    if valid_mask is not None:
        valid_work = valid_mask.to(
            device=psf_patch.device,
            dtype=torch.bool,
        ).reshape(-1)
        if not bool(valid_work.any().item()):
            return identity_deblur_filter_3d(
                filter_shape,
                device=psf_patch.device,
                dtype=original_dtype,
            )
    if psf_work.square().sum().item() == 0.0:
        return identity_deblur_filter_3d(
            filter_shape,
            device=psf_patch.device,
            dtype=original_dtype,
        )

    a = build_same_convolution_matrix_3d(psf_work, filter_shape)
    rhs = reference_work.reshape(-1)
    if valid_work is not None:
        a = a[valid_work]
        rhs = rhs[valid_work]
    ata = a.T @ a
    atb = a.T @ rhs
    diag_scale = torch.diagonal(ata).abs().max()
    if diag_scale.item() == 0.0:
        return identity_deblur_filter_3d(
            filter_shape,
            device=psf_patch.device,
            dtype=original_dtype,
        )

    eye = torch.eye(ata.shape[0], device=ata.device, dtype=ata.dtype)
    ridge = damping * diag_scale
    system = ata + ridge * eye
    try:
        filt = torch.linalg.solve(system, atb)
    except RuntimeError:
        filt = torch.linalg.lstsq(system, atb[:, None]).solution[:, 0]
    filt = torch.nan_to_num(filt, nan=0.0, posinf=0.0, neginf=0.0)
    return filt.reshape(filter_shape).to(dtype=original_dtype)


def make_deblur_filter_bank_3d(
    psf_image: torch.Tensor,
    reference_probe: torch.Tensor,
    *,
    active_mask: torch.Tensor,
    filter_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    damping: float,
) -> DeblurFilterBank3D:
    if psf_image.shape != reference_probe.shape:
        raise ValueError("psf_image and reference_probe must have the same shape.")
    if active_mask.shape != psf_image.shape:
        raise ValueError("active_mask must have the same shape as psf_image.")
    validate_positive_odd_shape("filter_shape", filter_shape)
    validate_positive_odd_shape("patch_shape", patch_shape)
    if damping < 0.0:
        raise ValueError("damping must be >= 0.")

    active = active_mask.to(device=psf_image.device, dtype=torch.bool)
    centers = torch.nonzero((reference_probe != 0) & active, as_tuple=False)
    if centers.numel() == 0:
        raise ValueError("reference_probe has no active deblurring filter centers.")

    filters = torch.empty(
        (centers.shape[0], *filter_shape),
        device=psf_image.device,
        dtype=psf_image.dtype,
    )
    with torch.no_grad():
        for index, center_tensor in enumerate(centers):
            center = tuple(int(v) for v in center_tensor.detach().cpu().tolist())
            psf_patch = centered_tensor_patch_3d(psf_image, center, patch_shape)
            ref_patch = centered_tensor_patch_3d(reference_probe, center, patch_shape)
            valid_patch = centered_tensor_patch_3d(
                active_mask.to(device=psf_image.device, dtype=psf_image.dtype),
                center,
                patch_shape,
            )
            filters[index] = estimate_deblur_filter_3d(
                psf_patch,
                ref_patch,
                filter_shape=filter_shape,
                damping=damping,
                valid_mask=valid_patch > 0,
            )
    return DeblurFilterBank3D(
        centers=centers.contiguous(),
        filters=filters.contiguous(),
        filter_shape=filter_shape,
        patch_shape=patch_shape,
        damping=float(damping),
    )


def build_center_interpolation_plan_3d(
    *,
    shape: tuple[int, int, int],
    origins: tuple[int, int, int],
    cw_z: int,
    cw_y: int,
    cw_x: int,
    device: torch.device,
    dtype: torch.dtype,
) -> CenterInterpolationPlan3D:
    if cw_z <= 0 or cw_y <= 0 or cw_x <= 0:
        raise ValueError("PSF cell widths must be positive.")
    nz, ny, nx = shape
    z, y, x = torch.meshgrid(
        torch.arange(nz, device=device),
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    origin_z, origin_y, origin_x = (int(v) for v in origins)
    grid_z = torch.div(z - origin_z, cw_z, rounding_mode="floor")
    grid_y = torch.div(y - origin_y, cw_y, rounding_mode="floor")
    grid_x = torch.div(x - origin_x, cw_x, rounding_mode="floor")
    base_z = origin_z + grid_z * cw_z
    base_y = origin_y + grid_y * cw_y
    base_x = origin_x + grid_x * cw_x
    return CenterInterpolationPlan3D(
        shape=shape,
        z0=base_z,
        y0=base_y,
        x0=base_x,
        z1=base_z + cw_z,
        y1=base_y + cw_y,
        x1=base_x + cw_x,
        wz=((z - base_z).to(dtype) / float(cw_z)).clamp(0.0, 1.0),
        wy=((y - base_y).to(dtype) / float(cw_y)).clamp(0.0, 1.0),
        wx=((x - base_x).to(dtype) / float(cw_x)).clamp(0.0, 1.0),
    )


def interpolate_center_values_with_plan_3d(
    center_values: torch.Tensor,
    *,
    centers: torch.Tensor,
    plan: CenterInterpolationPlan3D,
) -> torch.Tensor:
    if center_values.ndim != 1:
        raise ValueError("center_values must be a 1D tensor.")
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape [n, 3].")
    if center_values.shape[0] != centers.shape[0]:
        raise ValueError("center_values and centers must have the same length.")

    device = center_values.device
    dtype = center_values.dtype
    shape = plan.shape
    coeff_at_centers = torch.zeros(shape, device=device, dtype=dtype)
    valid_centers = torch.zeros(shape, device=device, dtype=torch.bool)
    if centers.numel() > 0:
        zc, yc, xc = centers.to(device=device, dtype=torch.long).unbind(dim=1)
        coeff_at_centers[zc, yc, xc] = center_values
        valid_centers[zc, yc, xc] = True

    def sample_coeff(
        z: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nz, ny, nx = shape
        valid_bounds = (
            (z >= 0)
            & (z < nz)
            & (y >= 0)
            & (y < ny)
            & (x >= 0)
            & (x < nx)
        )
        safe_z = z.clamp(0, nz - 1)
        safe_y = y.clamp(0, ny - 1)
        safe_x = x.clamp(0, nx - 1)
        valid = valid_bounds & valid_centers[safe_z, safe_y, safe_x]
        values = coeff_at_centers[safe_z, safe_y, safe_x]
        return torch.where(valid, values, torch.zeros_like(values)), valid.to(dtype)

    numerator = torch.zeros(shape, device=device, dtype=dtype)
    denominator = torch.zeros(shape, device=device, dtype=dtype)
    for weight, z, y, x in (
        ((1.0 - plan.wz) * (1.0 - plan.wy) * (1.0 - plan.wx), plan.z0, plan.y0, plan.x0),
        (plan.wz * (1.0 - plan.wy) * (1.0 - plan.wx), plan.z1, plan.y0, plan.x0),
        ((1.0 - plan.wz) * plan.wy * (1.0 - plan.wx), plan.z0, plan.y1, plan.x0),
        (plan.wz * plan.wy * (1.0 - plan.wx), plan.z1, plan.y1, plan.x0),
        ((1.0 - plan.wz) * (1.0 - plan.wy) * plan.wx, plan.z0, plan.y0, plan.x1),
        (plan.wz * (1.0 - plan.wy) * plan.wx, plan.z1, plan.y0, plan.x1),
        ((1.0 - plan.wz) * plan.wy * plan.wx, plan.z0, plan.y1, plan.x1),
        (plan.wz * plan.wy * plan.wx, plan.z1, plan.y1, plan.x1),
    ):
        values, valid = sample_coeff(z, y, x)
        effective_weight = weight * valid
        numerator = numerator + effective_weight * values
        denominator = denominator + effective_weight

    eps = torch.finfo(dtype).eps
    return torch.where(
        denominator > eps,
        numerator / denominator.clamp_min(eps),
        torch.zeros_like(numerator),
    )


def interpolate_center_value_volume_3d(
    center_values: torch.Tensor,
    *,
    centers: torch.Tensor,
    shape: tuple[int, int, int],
    origins: tuple[int, int, int],
    cw_z: int,
    cw_y: int,
    cw_x: int,
) -> torch.Tensor:
    plan = build_center_interpolation_plan_3d(
        shape=shape,
        origins=origins,
        cw_z=cw_z,
        cw_y=cw_y,
        cw_x=cw_x,
        device=center_values.device,
        dtype=center_values.dtype,
    )
    return interpolate_center_values_with_plan_3d(
        center_values,
        centers=centers,
        plan=plan,
    )


def apply_deblur_filter_bank_3d(
    image: torch.Tensor,
    filter_bank: DeblurFilterBank3D,
    *,
    active_mask: torch.Tensor,
    origins: tuple[int, int, int],
    cw_z: int,
    cw_y: int,
    cw_x: int,
) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError("image must have shape [z, y, x].")
    if active_mask.shape != image.shape:
        raise ValueError("active_mask must have the same shape as image.")
    validate_positive_odd_shape("filter_shape", filter_bank.filter_shape)

    active = active_mask.to(device=image.device, dtype=image.dtype)
    centers = filter_bank.centers.to(device=image.device, dtype=torch.long)
    filters = filter_bank.filters.to(device=image.device, dtype=image.dtype)
    kz, ky, kx = filter_bank.filter_shape
    cz, cy, cx = kz // 2, ky // 2, kx // 2
    flat_filters = filters.reshape(filters.shape[0], -1)
    plan = build_center_interpolation_plan_3d(
        shape=tuple(int(v) for v in image.shape),
        origins=origins,
        cw_z=cw_z,
        cw_y=cw_y,
        cw_x=cw_x,
        device=image.device,
        dtype=image.dtype,
    )
    out = torch.zeros_like(image)
    col = 0
    for fz in range(kz):
        for fy in range(ky):
            for fx in range(kx):
                coeff = interpolate_center_values_with_plan_3d(
                    flat_filters[:, col],
                    centers=centers,
                    plan=plan,
                )
                shifted = same_convolution_shift_3d(
                    image,
                    fz - cz,
                    fy - cy,
                    fx - cx,
                )
                out = out + coeff * shifted
                col += 1
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0) * active


def describe_deblur_filter_bank(filter_bank: DeblurFilterBank3D) -> str:
    kz, ky, kx = filter_bank.filter_shape
    center_coeff = filter_bank.filters[:, kz // 2, ky // 2, kx // 2]
    l2 = torch.linalg.vector_norm(filter_bank.filters.reshape(filter_bank.filters.shape[0], -1), dim=1)
    return (
        f"count={filter_bank.filters.shape[0]} "
        f"filter_shape={filter_bank.filter_shape} patch_shape={filter_bank.patch_shape} "
        f"damping={filter_bank.damping:.3e} "
        f"center_coeff_median={float(center_coeff.median().detach().cpu()):.3e} "
        f"l2_median={float(l2.median().detach().cpu()):.3e}"
    )


class PsfHessian3D:
    def __init__(
        self,
        psf_image: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        cw_z: int,
        cw_y: int,
        cw_x: int,
        origins: tuple[int, int, int],
        symmetrize: bool,
        row_weights: torch.Tensor | None = None,
    ) -> None:
        if psf_image.ndim != 3:
            raise ValueError("psf_image must be [z, y, x].")
        if cw_z <= 0 or cw_y <= 0 or cw_x <= 0:
            raise ValueError("PSF cell widths must be positive.")
        if active_mask.shape != psf_image.shape:
            raise ValueError("active_mask must have the same shape as psf_image.")
        self.psf = psf_image.contiguous()
        self.active = active_mask.to(device=psf_image.device, dtype=torch.bool)
        self.shape = tuple(int(v) for v in psf_image.shape)
        self.device = psf_image.device
        self.dtype = psf_image.dtype
        self.cw_z = int(cw_z)
        self.cw_y = int(cw_y)
        self.cw_x = int(cw_x)
        self.origin_z, self.origin_y, self.origin_x = (int(v) for v in origins)
        self.symmetrize = symmetrize
        if row_weights is None:
            self.row_weights = torch.ones_like(psf_image)
        else:
            if row_weights.shape != psf_image.shape:
                raise ValueError("row_weights must have the same shape as psf_image.")
            self.row_weights = row_weights.to(
                device=psf_image.device,
                dtype=psf_image.dtype,
            ).contiguous()
        nz, ny, nx = self.shape
        z, y, x = torch.meshgrid(
            torch.arange(nz, device=self.device),
            torch.arange(ny, device=self.device),
            torch.arange(nx, device=self.device),
            indexing="ij",
        )
        self.z = z
        self.y = y
        self.x = x
        self.kz = torch.div(z - self.origin_z, self.cw_z, rounding_mode="floor")
        self.ky = torch.div(y - self.origin_y, self.cw_y, rounding_mode="floor")
        self.kx = torch.div(x - self.origin_x, self.cw_x, rounding_mode="floor")
        base_z = self.origin_z + self.kz * self.cw_z
        base_y = self.origin_y + self.ky * self.cw_y
        base_x = self.origin_x + self.kx * self.cw_x
        self.wz = ((z - base_z).to(self.dtype) / float(self.cw_z)).clamp(0.0, 1.0)
        self.wy = ((y - base_y).to(self.dtype) / float(self.cw_y)).clamp(0.0, 1.0)
        self.wx = ((x - base_x).to(self.dtype) / float(self.cw_x)).clamp(0.0, 1.0)
        self.diagonal = self.row_weights * self._build_base_diagonal()

    def _sample_psf_with_valid(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nz, ny, nx = self.shape
        valid = (
            (z >= 0)
            & (z < nz)
            & (y >= 0)
            & (y < ny)
            & (x >= 0)
            & (x < nx)
        )
        values = self.psf[
            z.clamp(0, nz - 1),
            y.clamp(0, ny - 1),
            x.clamp(0, nx - 1),
        ]
        return torch.where(valid, values, torch.zeros_like(values)), valid.to(self.dtype)

    def _offset_ranges(self) -> tuple[range, range, range]:
        return (
            range(-(self.cw_z // 2), (self.cw_z + 1) // 2),
            range(-(self.cw_y // 2), (self.cw_y + 1) // 2),
            range(-(self.cw_x // 2), (self.cw_x + 1) // 2),
        )

    def _coefficient_and_shift(
        self,
        dz: int,
        dy: int,
        dx: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        nz, ny, nx = self.shape
        raw_iz = self.z + dz
        raw_iy = self.y + dy
        raw_ix = self.x + dx
        valid_shift = (
            (raw_iz >= 0)
            & (raw_iz < nz)
            & (raw_iy >= 0)
            & (raw_iy < ny)
            & (raw_ix >= 0)
            & (raw_ix < nx)
        )
        iz = raw_iz.clamp(0, nz - 1)
        iy = raw_iy.clamp(0, ny - 1)
        ix = raw_ix.clamp(0, nx - 1)
        mz0 = self.origin_z + self.kz * self.cw_z + dz
        mz1 = self.origin_z + (self.kz + 1) * self.cw_z + dz
        my0 = self.origin_y + self.ky * self.cw_y + dy
        my1 = self.origin_y + (self.ky + 1) * self.cw_y + dy
        mx0 = self.origin_x + self.kx * self.cw_x + dx
        mx1 = self.origin_x + (self.kx + 1) * self.cw_x + dx
        coeff_num = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        coeff_den = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        for weight, z, y, x in (
            ((1.0 - self.wz) * (1.0 - self.wy) * (1.0 - self.wx), mz0, my0, mx0),
            (self.wz * (1.0 - self.wy) * (1.0 - self.wx), mz1, my0, mx0),
            ((1.0 - self.wz) * self.wy * (1.0 - self.wx), mz0, my1, mx0),
            (self.wz * self.wy * (1.0 - self.wx), mz1, my1, mx0),
            ((1.0 - self.wz) * (1.0 - self.wy) * self.wx, mz0, my0, mx1),
            (self.wz * (1.0 - self.wy) * self.wx, mz1, my0, mx1),
            ((1.0 - self.wz) * self.wy * self.wx, mz0, my1, mx1),
            (self.wz * self.wy * self.wx, mz1, my1, mx1),
        ):
            values, valid = self._sample_psf_with_valid(z, y, x)
            effective_weight = weight * valid
            coeff_num = coeff_num + effective_weight * values
            coeff_den = coeff_den + effective_weight
        eps = torch.finfo(self.dtype).eps
        coeff = torch.where(
            coeff_den > eps,
            coeff_num / coeff_den.clamp_min(eps),
            torch.zeros_like(coeff_num),
        )
        same_cell = (
            valid_shift
            & (raw_iz == self.z)
            & (raw_iy == self.y)
            & (raw_ix == self.x)
        )
        coeff = torch.where(
            same_cell & (coeff < 0),
            torch.zeros((), device=self.device, dtype=self.dtype),
            coeff,
        )
        keep = (
            self.active
            & valid_shift
            & self.active[iz, iy, ix]
            & torch.isfinite(coeff)
        )
        coeff = torch.where(keep, coeff, torch.zeros_like(coeff))
        shifted_flat = (iz * ny * nx + iy * nx + ix).reshape(-1)
        return coeff, iz, iy, ix, shifted_flat

    def _forward_base_unsym(self, image: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(image)
        for dz_range, dy_range, dx_range in [self._offset_ranges()]:
            for dz in dz_range:
                for dy in dy_range:
                    for dx in dx_range:
                        coeff, iz, iy, ix, _ = self._coefficient_and_shift(dz, dy, dx)
                        out = out + coeff * image[iz, iy, ix]
        return out * self.active.to(dtype=image.dtype)

    def _adjoint_base_unsym(self, image: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(image.numel(), device=image.device, dtype=image.dtype)
        image_flat = image.reshape(-1)
        for dz_range, dy_range, dx_range in [self._offset_ranges()]:
            for dz in dz_range:
                for dy in dy_range:
                    for dx in dx_range:
                        coeff, _, _, _, shifted_flat = self._coefficient_and_shift(dz, dy, dx)
                        out.index_add_(0, shifted_flat.to(torch.long), coeff.reshape(-1) * image_flat)
        return out.reshape(self.shape) * self.active.to(dtype=image.dtype)

    def _build_base_diagonal(self) -> torch.Tensor:
        diagonal = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        for dz_range, dy_range, dx_range in [self._offset_ranges()]:
            for dz in dz_range:
                for dy in dy_range:
                    for dx in dx_range:
                        coeff, iz, iy, ix, _ = self._coefficient_and_shift(dz, dy, dx)
                        same = (iz == self.z) & (iy == self.y) & (ix == self.x)
                        diagonal = diagonal + torch.where(same, coeff, torch.zeros_like(coeff))
        return diagonal * self.active.to(dtype=self.dtype)

    def _base_matvec(self, image: torch.Tensor) -> torch.Tensor:
        if self.symmetrize:
            return 0.5 * (
                self._forward_base_unsym(image) + self._adjoint_base_unsym(image)
            )
        return self._forward_base_unsym(image)

    def _base_adjoint(self, image: torch.Tensor) -> torch.Tensor:
        if self.symmetrize:
            return self._base_matvec(image)
        return self._adjoint_base_unsym(image)

    def matvec(self, image: torch.Tensor) -> torch.Tensor:
        return self.row_weights * self._base_matvec(image)

    def adjoint(self, image: torch.Tensor) -> torch.Tensor:
        return self._base_adjoint(self.row_weights * image)


def check_psf_adjoint(
    operator: PsfHessian3D,
    image_mask: torch.Tensor,
) -> float:
    generator = torch.Generator(device=image_mask.device)
    generator.manual_seed(20260602)
    x = torch.randn(
        operator.shape,
        generator=generator,
        device=image_mask.device,
        dtype=image_mask.dtype,
    ) * image_mask
    y = torch.randn(
        operator.shape,
        generator=generator,
        device=image_mask.device,
        dtype=image_mask.dtype,
    ) * image_mask
    lhs = dot_product(operator.matvec(x), y)
    rhs = dot_product(x, operator.adjoint(y))
    denom = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(torch.finfo(torch.float64).eps)
    return float(((lhs - rhs).abs() / denom).detach().cpu())


def solve_image_domain_pcgnr(
    operator: PsfHessian3D,
    rtm_image: torch.Tensor,
    *,
    image_mask: torch.Tensor,
    iterations: int,
    precondition_damping: float,
) -> ImageDomainResult:
    if iterations < 0:
        raise ValueError("iterations must be >= 0.")
    if precondition_damping < 0.0:
        raise ValueError("precondition_damping must be >= 0.")
    active = image_mask.to(device=rtm_image.device, dtype=rtm_image.dtype)
    target = operator.row_weights * rtm_image.detach() * active
    image = torch.zeros_like(target)
    residual = (target - operator.matvec(image)) * active
    normal_residual = operator.adjoint(residual) * active
    diagonal = operator.diagonal.clamp_min(0)
    damping = precondition_damping * diagonal.max().clamp_min(torch.finfo(diagonal.dtype).eps)

    def precondition(vector: torch.Tensor) -> torch.Tensor:
        out = vector / (diagonal + damping)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0) * active

    z = precondition(normal_residual)
    direction = z.clone()
    zrt_old = dot_product(z, normal_residual)
    residual0 = dot_product(residual, residual).clamp_min(torch.finfo(torch.float64).eps)
    loss_history: list[float] = []
    relative_residual_history: list[float] = []
    status = "max_iter"
    completed = 0
    if zrt_old.item() <= 0.0:
        status = "converged"
    else:
        for iteration in range(1, iterations + 1):
            ap = operator.matvec(direction) * active
            denom = dot_product(ap, ap)
            if denom.item() <= 0.0 or not bool(torch.isfinite(denom)):
                status = "breakdown"
                break
            alpha = (zrt_old / denom).to(dtype=rtm_image.dtype)
            image = (image + alpha * direction) * active
            residual = (residual - alpha * ap) * active
            normal_residual = operator.adjoint(residual) * active
            z = precondition(normal_residual)
            zrt_new = dot_product(z, normal_residual)
            beta = (zrt_new / zrt_old).to(dtype=rtm_image.dtype)
            if not bool(torch.isfinite(beta)):
                status = "breakdown"
                break
            direction = (z + beta * direction) * active
            zrt_old = zrt_new
            completed = iteration
            residual_norm_sq = dot_product(residual, residual)
            relative = torch.sqrt(residual_norm_sq / residual0)
            loss = 0.5 * residual_norm_sq
            loss_history.append(float(loss.detach().cpu()))
            relative_residual_history.append(float(relative.detach().cpu()))
            print(
                f"Image-domain PCGNR iter {iteration:03d}/{iterations} "
                f"loss={loss.item():.6e} relative_image_residual={relative.item():.4f}"
            )
            if (residual_norm_sq / residual0).item() < 1e-6:
                status = "converged"
                break
    return ImageDomainResult(
        image=image.detach(),
        image_residual=residual.detach(),
        normal_residual=normal_residual.detach(),
        loss_history=loss_history,
        relative_residual_history=relative_residual_history,
        status=status,
        iterations=completed,
    )


def center_line_x(case: ImageDomain3DCase) -> int:
    return case.line_xs[len(case.line_xs) // 2]


def line_slice(image: torch.Tensor, line_x: int) -> torch.Tensor:
    return image[:, :, line_x]


def receiver_data_panel(data: torch.Tensor) -> torch.Tensor:
    if data.ndim != 3:
        raise ValueError("receiver data must have shape [time, shot, receiver].")
    return data[:, :, 0] if data.shape[2] == 1 else data[:, 0, :]


def symmetric_abs_quantile_limit(image: torch.Tensor, quantile: float) -> float:
    values = image.detach().cpu().numpy()
    return max(float(np.quantile(np.abs(values), quantile)), 1e-8)


def percentile_limits(image: torch.Tensor, quantile: float = 0.98) -> tuple[float, float]:
    vmax = symmetric_abs_quantile_limit(image, quantile)
    return -vmax, vmax


def plot_results(
    case: ImageDomain3DCase,
    *,
    observed_data: torch.Tensor,
    background_data: torch.Tensor,
    observed_scattered: torch.Tensor,
    rtm_image: torch.Tensor,
    psf_probe: torch.Tensor,
    psf_image: torch.Tensor,
    illumination: torch.Tensor,
    illumination_compensation: torch.Tensor,
    image_domain_lsrtm: torch.Tensor,
    result: ImageDomainResult,
    deblurred: bool,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    line_x = center_line_x(case)
    fig, axes = plt.subplots(4, 3, figsize=(15.0, 13.0), constrained_layout=True)
    psf_title = "Deblurred 3D PSF Hessian image" if deblurred else "3D PSF Hessian image"
    image_panels = [
        ("True perturbation", line_slice(case.depsilon_true, line_x), "seismic"),
        ("RTM", line_slice(rtm_image, line_x), "gray"),
        ("Source illumination", line_slice(illumination, line_x), "magma"),
        (
            "Illumination compensation",
            line_slice(illumination_compensation, line_x),
            "viridis",
        ),
        ("3D PSF probe", line_slice(psf_probe, line_x), "gray"),
        (psf_title, line_slice(psf_image, line_x), "seismic"),
        ("Image-domain LSRTM", line_slice(image_domain_lsrtm, line_x), "gray"),
        ("Image residual", line_slice(result.image_residual, line_x), "seismic"),
        ("Normal residual", line_slice(result.normal_residual, line_x), "seismic"),
    ]
    for ax, (title, image, cmap) in zip(axes.flat[:9], image_panels, strict=True):
        if title == "3D PSF probe":
            vmin, vmax = 0.0, 1.0
        elif title in {"Source illumination", "Illumination compensation"}:
            values = image.detach().cpu().numpy()
            vmin = 0.0
            vmax = max(float(np.quantile(values, 0.98)), 1e-8)
        else:
            vmin, vmax = percentile_limits(image)
        im = ax.imshow(
            image.detach().cpu().numpy(),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{title} x={line_x}")
        ax.set_xlabel("y cell")
        ax.set_ylabel("z cell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    data_panels = [
        ("Observed", receiver_data_panel(observed_data)),
        ("Background", receiver_data_panel(background_data)),
        ("Observed - background", receiver_data_panel(observed_scattered)),
    ]
    for ax, (title, data) in zip(axes.flat[9:], data_panels, strict=True):
        values = data.detach().cpu()
        vmax = symmetric_abs_quantile_limit(values, SHOT_DISPLAY_QUANTILE)
        im = ax.imshow(
            values.numpy(),
            aspect="auto",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("shot")
        ax.set_ylabel("time sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_dir / "image_domain_lsrtm_3d_overview.png", dpi=180)
    plt.close(fig)

    if result.relative_residual_history:
        fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
        ax.semilogy(
            range(1, len(result.relative_residual_history) + 1),
            result.relative_residual_history,
            marker="o",
        )
        ax.set_xlabel("PCGNR iteration")
        ax.set_ylabel("relative image residual")
        ax.grid(True, alpha=0.25)
        fig.savefig(output_dir / "image_domain_lsrtm_3d_loss.png", dpi=180)
        plt.close(fig)


def h5_write_tensor(group: h5py.Group, name: str, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    group.create_dataset(name, data=array, compression="gzip", shuffle=True)


def save_h5(
    case: ImageDomain3DCase,
    *,
    observed_data: torch.Tensor,
    background_data: torch.Tensor,
    observed_scattered: torch.Tensor,
    rtm_image: torch.Tensor,
    psf_probe: torch.Tensor,
    psf_image: torch.Tensor,
    illumination: torch.Tensor,
    illumination_compensation: torch.Tensor,
    image_domain_mask: torch.Tensor,
    image_domain_lsrtm: torch.Tensor,
    result: ImageDomainResult,
    standard_rtm_image: torch.Tensor | None,
    standard_psf_image: torch.Tensor | None,
    deblur_filter_bank: DeblurFilterBank3D | None,
    timings: dict[str, float],
    psf_cw_z: int,
    psf_cw_y: int,
    psf_cw_x: int,
    image_domain_lateral_taper_cells: int,
    deblurred: bool,
    illumination_compensation_enabled: bool,
    illumination_compensation_floor: float,
    illumination_compensation_power: float,
    illumination_compensation_max_gain: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "image_domain_lsrtm_3d_results.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["workflow"] = "image_domain_lsrtm_3d_psf_illumination_pcgnr"
        h5.attrs["model_label"] = case.model_label
        h5.attrs["dx"] = case.dx
        h5.attrs["dt"] = case.dt
        h5.attrs["freq"] = case.freq
        h5.attrs["pml_width"] = case.pml_width
        h5.attrs["stencil"] = case.stencil
        h5.attrs["storage_mode"] = case.storage_mode
        h5.attrs["storage_compression"] = case.storage_compression
        h5.attrs["line_xs"] = np.asarray(case.line_xs, dtype=np.int64)
        h5.attrs["model_gradient_sampling_interval"] = case.model_gradient_sampling_interval
        h5.attrs["source_illumination_sampling_interval"] = (
            case.illumination_wavefield_sampling_interval
        )
        h5.attrs["psf_cw_z"] = int(psf_cw_z)
        h5.attrs["psf_cw_y"] = int(psf_cw_y)
        h5.attrs["psf_cw_x"] = int(psf_cw_x)
        h5.attrs["image_domain_lateral_taper_cells"] = int(
            image_domain_lateral_taper_cells
        )
        h5.attrs["deblurred_image_domain"] = bool(deblurred)
        if deblur_filter_bank is not None:
            h5.attrs["deblur_filter_shape"] = np.asarray(
                deblur_filter_bank.filter_shape,
                dtype=np.int64,
            )
            h5.attrs["deblur_patch_shape"] = np.asarray(
                deblur_filter_bank.patch_shape,
                dtype=np.int64,
            )
            h5.attrs["deblur_damping"] = float(deblur_filter_bank.damping)
        h5.attrs["illumination_compensation_enabled"] = bool(
            illumination_compensation_enabled
        )
        h5.attrs["illumination_compensation_floor"] = float(
            illumination_compensation_floor
        )
        h5.attrs["illumination_compensation_power"] = float(
            illumination_compensation_power
        )
        h5.attrs["illumination_compensation_max_gain"] = float(
            illumination_compensation_max_gain
        )
        h5.attrs["pcgnr_status"] = result.status
        h5.attrs["pcgnr_iterations"] = result.iterations
        models = h5.create_group("models")
        h5_write_tensor(models, "epsilon_true", case.epsilon_true)
        h5_write_tensor(models, "epsilon_background", case.epsilon_background)
        h5_write_tensor(models, "depsilon_true", case.depsilon_true)
        h5_write_tensor(models, "rtm_image", rtm_image)
        if standard_rtm_image is not None:
            h5_write_tensor(models, "standard_rtm_image", standard_rtm_image)
        h5_write_tensor(models, "source_illumination", illumination)
        h5_write_tensor(models, "illumination_compensation", illumination_compensation)
        h5_write_tensor(models, "image_domain_mask", image_domain_mask)
        h5_write_tensor(models, "psf_probe", psf_probe)
        h5_write_tensor(models, "psf_image", psf_image)
        if standard_psf_image is not None:
            h5_write_tensor(models, "standard_psf_image", standard_psf_image)
        h5_write_tensor(models, "image_domain_lsrtm", image_domain_lsrtm)
        h5_write_tensor(models, "image_residual", result.image_residual)
        h5_write_tensor(models, "normal_residual", result.normal_residual)
        h5_write_tensor(models, "model_weights", case.model_weights)
        h5_write_tensor(models, "line_mask", case.line_mask)
        data = h5.create_group("data")
        h5_write_tensor(data, "observed_data", observed_data)
        h5_write_tensor(data, "background_data", background_data)
        h5_write_tensor(data, "observed_scattered", observed_scattered)
        geometry = h5.create_group("geometry")
        geometry.create_dataset("source_locations", data=case.source_locations.detach().cpu().numpy())
        geometry.create_dataset("receiver_locations", data=case.receiver_locations.detach().cpu().numpy())
        geometry.create_dataset("line_xs", data=np.asarray(case.line_xs, dtype=np.int64))
        if deblur_filter_bank is not None:
            deblur = h5.create_group("deblur")
            deblur.create_dataset(
                "centers",
                data=deblur_filter_bank.centers.detach().cpu().numpy().astype(
                    np.int64,
                    copy=False,
                ),
                compression="gzip",
                shuffle=True,
            )
            deblur.create_dataset(
                "filters",
                data=deblur_filter_bank.filters.detach().cpu().numpy().astype(
                    np.float32,
                    copy=False,
                ),
                compression="gzip",
                shuffle=True,
            )
        curves = h5.create_group("curves")
        curves.create_dataset("loss_history", data=np.asarray(result.loss_history, dtype=np.float64))
        curves.create_dataset(
            "relative_residual_history",
            data=np.asarray(result.relative_residual_history, dtype=np.float64),
        )
        timing_group = h5.create_group("timings")
        for name, seconds in timings.items():
            timing_group.attrs[name] = float(seconds)
    print(f"Wrote HDF5 results to {path}")


def validate_config(config: ImageDomain3DConfig) -> None:
    if config.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'.")
    if config.device not in {"auto", "cuda"}:
        raise ValueError("device must be 'auto' or 'cuda'.")
    if config.backend not in {"auto", "native"}:
        raise ValueError("backend must be 'auto' or 'native'.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.psf_cw_z <= 0 or config.psf_cw_y <= 0 or config.psf_cw_x <= 0:
        raise ValueError("PSF_CW_Z, PSF_CW_Y, and PSF_CW_X must be positive.")
    if config.image_domain_iters < 0:
        raise ValueError("IMAGE_DOMAIN_ITERS must be >= 0.")
    if config.precondition_damping < 0.0:
        raise ValueError("PRECONDITION_DAMPING must be >= 0.")
    if config.image_domain_lateral_taper_cells < 0:
        raise ValueError("IMAGE_DOMAIN_LATERAL_TAPER_CELLS must be >= 0.")
    validate_positive_odd_shape(
        "DEBLUR_FILTER",
        (
            config.deblur_filter_z,
            config.deblur_filter_y,
            config.deblur_filter_x,
        ),
    )
    validate_positive_odd_shape(
        "DEBLUR_PATCH",
        (
            config.deblur_patch_z,
            config.deblur_patch_y,
            config.deblur_patch_x,
        ),
    )
    if config.deblur_damping < 0.0:
        raise ValueError("DEBLUR_DAMPING must be >= 0.")
    if config.illumination_compensation_floor < 0.0:
        raise ValueError("ILLUMINATION_COMPENSATION_FLOOR must be >= 0.")
    if config.illumination_compensation_power < 0.0:
        raise ValueError("ILLUMINATION_COMPENSATION_POWER must be >= 0.")
    if config.illumination_compensation_max_gain <= 0.0:
        raise ValueError("ILLUMINATION_COMPENSATION_MAX_GAIN must be positive.")


def main(config: ImageDomain3DConfig = CONFIG) -> None:
    total_start = perf_counter()
    validate_config(config)
    device = resolve_device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    python_backend = resolve_backend(config.backend)
    print(
        "Building 3D image-domain LSRTM case "
        f"device={device} dtype={dtype} backend=native "
        f"storage_compression={config.storage_compression}"
    )
    case, build_time = timed_stage(
        "Build case",
        device,
        lambda: build_case(
            config,
            device=device,
            dtype=dtype,
            python_backend=python_backend,
        ),
    )
    timings: dict[str, float] = {"build_case": build_time}

    image_mask = case.model_weights

    def forward_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            print("Forward modelling observed and background data")
            observed_data = forward_data(case, case.epsilon_true).detach()
            background_data, source_illumination = forward_data_with_source_illumination(
                case,
                case.epsilon_background.detach(),
            )
            background_data = background_data.detach()
            source_illumination = source_illumination.detach()
            observed_scattered = apply_data_weights(
                case,
                observed_data - background_data,
            ).detach()
        return observed_data, background_data, observed_scattered, source_illumination

    (
        observed_data,
        background_data,
        observed_scattered,
        source_illumination,
    ), forward_time = timed_stage(
        "Forward modelling",
        device,
        forward_stage,
    )
    timings["forward_modelling"] = forward_time

    if config.check_adjoint:
        rel_err = check_adjoint(case)
        print(f"Born J / Maxwell VJP J^T adjoint check relative error: {rel_err:.3e}")

    print("Computing RTM image")
    rtm_image, rtm_time = timed_stage(
        "RTM imaging",
        device,
        lambda: adjoint_image(case, observed_scattered),
    )
    timings["rtm"] = rtm_time
    print(
        "RTM image against true perturbation: "
        f"relative_rms={relative_rms(rtm_image - case.depsilon_true, case.depsilon_true):.4f} "
        f"cosine={cosine_similarity(rtm_image, case.depsilon_true):.4f}"
    )

    illumination, illumination_compensation = build_illumination_compensation(
        source_illumination,
        image_mask,
        enabled=config.illumination_compensation,
        floor_ratio=config.illumination_compensation_floor,
        power=config.illumination_compensation_power,
        max_gain=config.illumination_compensation_max_gain,
    )
    image_domain_mask = image_mask
    if config.image_domain_lateral_taper_cells > 0:
        lateral_taper = lateral_cosine_taper_3d(
            image_mask,
            config.image_domain_lateral_taper_cells,
        )
        image_domain_mask = image_mask * lateral_taper
        illumination_compensation = illumination_compensation * lateral_taper
        print(
            "Image-domain lateral taper: "
            f"cells={config.image_domain_lateral_taper_cells}"
        )
    active_comp = illumination_compensation[image_domain_mask > 0]
    print(
        "Image-domain illumination compensation: "
        f"enabled={config.illumination_compensation} "
        f"min={float(active_comp.min().detach().cpu()):.3e} "
        f"median={float(active_comp.median().detach().cpu()):.3e} "
        f"max={float(active_comp.max().detach().cpu()):.3e}"
    )

    psf_probe, psf_origins = make_3d_psf_probe(
        case,
        cw_z=config.psf_cw_z,
        cw_y=config.psf_cw_y,
        cw_x=config.psf_cw_x,
        image_mask=image_mask,
    )
    print(
        f"3D PSF probe count: {int((psf_probe != 0).sum().item())} "
        f"origins={psf_origins}, cw_z={config.psf_cw_z}, "
        f"cw_y={config.psf_cw_y}, cw_x={config.psf_cw_x}"
    )
    psf_image, psf_time = timed_stage(
        "Tide PSF Hessian image J^T J m_probe",
        device,
        lambda: normal_image(case, psf_probe).detach(),
    )
    timings["psf_generation"] = psf_time
    standard_rtm_image = rtm_image.detach()
    standard_psf_image = psf_image.detach()
    rtm_target_image = rtm_image
    psf_operator_image = psf_image
    deblur_filter_bank: DeblurFilterBank3D | None = None
    if config.deblur_image_domain:
        filter_shape = (
            config.deblur_filter_z,
            config.deblur_filter_y,
            config.deblur_filter_x,
        )
        patch_shape = (
            config.deblur_patch_z,
            config.deblur_patch_y,
            config.deblur_patch_x,
        )
        print(
            "Estimating image-domain deblurring filters "
            f"filter_shape={filter_shape} patch_shape={patch_shape} "
            f"damping={config.deblur_damping:.3e}"
        )
        deblur_filter_bank, deblur_filter_time = timed_stage(
            "Estimate image-domain deblurring filters",
            device,
            lambda: make_deblur_filter_bank_3d(
                psf_image,
                psf_probe,
                active_mask=image_mask,
                filter_shape=filter_shape,
                patch_shape=patch_shape,
                damping=config.deblur_damping,
            ),
        )
        timings["deblur_filter_estimation"] = deblur_filter_time
        print(f"Deblur filter bank: {describe_deblur_filter_bank(deblur_filter_bank)}")

        def apply_deblur_stage() -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                deblurred_rtm = apply_deblur_filter_bank_3d(
                    rtm_target_image,
                    deblur_filter_bank,
                    active_mask=image_domain_mask,
                    origins=psf_origins,
                    cw_z=config.psf_cw_z,
                    cw_y=config.psf_cw_y,
                    cw_x=config.psf_cw_x,
                ).detach()
                deblurred_psf = apply_deblur_filter_bank_3d(
                    psf_operator_image,
                    deblur_filter_bank,
                    active_mask=image_domain_mask,
                    origins=psf_origins,
                    cw_z=config.psf_cw_z,
                    cw_y=config.psf_cw_y,
                    cw_x=config.psf_cw_x,
                ).detach()
                return deblurred_rtm, deblurred_psf

        (rtm_target_image, psf_operator_image), deblur_apply_time = timed_stage(
            "Apply image-domain deblurring",
            device,
            apply_deblur_stage,
        )
        timings["deblur_apply"] = deblur_apply_time
        print(
            "Deblurred RTM target against true perturbation: "
            f"relative_rms={relative_rms(rtm_target_image - case.depsilon_true, case.depsilon_true):.4f} "
            f"cosine={cosine_similarity(rtm_target_image, case.depsilon_true):.4f}"
        )
    operator, operator_time = timed_stage(
        "Build 3D PSF Hessian",
        device,
        lambda: PsfHessian3D(
            psf_operator_image,
            active_mask=image_domain_mask,
            cw_z=config.psf_cw_z,
            cw_y=config.psf_cw_y,
            cw_x=config.psf_cw_x,
            origins=psf_origins,
            symmetrize=config.symmetrize_psf,
            row_weights=illumination_compensation,
        ),
    )
    timings["psf_operator"] = operator_time
    if config.check_psf_adjoint:
        psf_rel_err = check_psf_adjoint(operator, image_domain_mask)
        print(f"3D PSF Hessian adjoint relative error: {psf_rel_err:.3e}")

    result, pcgnr_time = timed_stage(
        "Image-domain PCGNR",
        device,
        lambda: solve_image_domain_pcgnr(
            operator,
            rtm_target_image,
            image_mask=image_domain_mask,
            iterations=config.image_domain_iters,
            precondition_damping=config.precondition_damping,
        ),
    )
    timings["image_domain_pcgnr"] = pcgnr_time
    image_domain_lsrtm = apply_mask(result.image, image_domain_mask)
    print(
        f"Image-domain status={result.status} iterations={result.iterations} "
        f"relative_rms={relative_rms(image_domain_lsrtm - case.depsilon_true, case.depsilon_true):.4f} "
        f"cosine={cosine_similarity(image_domain_lsrtm, case.depsilon_true):.4f}"
    )

    plot_results(
        case,
        observed_data=observed_data,
        background_data=background_data,
        observed_scattered=observed_scattered,
        rtm_image=standard_rtm_image,
        psf_probe=psf_probe,
        psf_image=psf_operator_image,
        illumination=illumination,
        illumination_compensation=illumination_compensation,
        image_domain_lsrtm=image_domain_lsrtm,
        result=result,
        deblurred=config.deblur_image_domain,
        output_dir=config.output_dir,
    )
    total_time = perf_counter() - total_start
    timings["total"] = total_time
    save_h5(
        case,
        observed_data=observed_data,
        background_data=background_data,
        observed_scattered=observed_scattered,
        rtm_image=rtm_target_image,
        psf_probe=psf_probe,
        psf_image=psf_operator_image,
        illumination=illumination,
        illumination_compensation=illumination_compensation,
        image_domain_lsrtm=image_domain_lsrtm,
        result=result,
        standard_rtm_image=standard_rtm_image if config.deblur_image_domain else None,
        standard_psf_image=standard_psf_image if config.deblur_image_domain else None,
        deblur_filter_bank=deblur_filter_bank,
        timings=timings,
        psf_cw_z=config.psf_cw_z,
        psf_cw_y=config.psf_cw_y,
        psf_cw_x=config.psf_cw_x,
        image_domain_lateral_taper_cells=config.image_domain_lateral_taper_cells,
        deblurred=config.deblur_image_domain,
        illumination_compensation_enabled=config.illumination_compensation,
        illumination_compensation_floor=config.illumination_compensation_floor,
        illumination_compensation_power=config.illumination_compensation_power,
        illumination_compensation_max_gain=config.illumination_compensation_max_gain,
        output_dir=config.output_dir,
    )
    print("Timing summary:")
    for name, seconds in timings.items():
        print(f"  {name}: {format_seconds(seconds)}")
    print(f"Wrote plots to {config.output_dir}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
