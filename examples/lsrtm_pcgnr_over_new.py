"""3D LSRTM example for Tide Maxwell modelling.

This is the 3D counterpart of ``example_deepwave_style_lsrtm_tm.py``:

1. model observed data in the true model,
2. model background data in a smoothed initial model,
3. subtract the background data,
4. invert a scattering model with the 3D Born forward operator.

The real model is read from ``data/overthrust_crop_norm.h5`` dataset ``m``.
Tide uses the coordinate convention ``[z, y, x]`` for 3D Maxwell models.

Run from the repository root:

    uv run python scripts/
    example_deepwave_style_lsrtm_3d_multi_yline.py

The default acquisition uses five y-directed lines on different x planes, with
ten sources per line. Sources and receivers are inside the air layer.
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

# Edit these parameters before running the script.
DEVICE = "auto"  # "auto" or "cuda"; bf16 storage compression requires CUDA.
DTYPE = "float32"  # "float32" or "float64"
BACKEND = "native"  # "native" or "auto"; Python backend cannot use bf16 storage.

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

KRYLOV_ITERS = 5
REG_LAMBDA = 1e-3

DATA_MUTE_MARGIN_CYCLES = 1.0
DATA_MUTE_TAPER_SAMPLES = 12
NO_DATA_MUTE = False

STORAGE_MODE = "device"  # "device", "cpu", "disk", or "auto".
STORAGE_COMPRESSION = "bf16"
CHECK_ADJOINT = False
OUTPUT_DIR = Path("outputs/over_pcgnr_1lines_sigma10")


@dataclass(frozen=True, slots=True)
class Lsrtm3DConfig:
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
    krylov_iters: int
    reg_lambda: float
    data_mute_margin_cycles: float
    data_mute_taper_samples: int
    no_data_mute: bool
    storage_mode: str
    storage_compression: str
    check_adjoint: bool
    output_dir: Path


CONFIG = Lsrtm3DConfig(
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
    krylov_iters=KRYLOV_ITERS,
    reg_lambda=REG_LAMBDA,
    data_mute_margin_cycles=DATA_MUTE_MARGIN_CYCLES,
    data_mute_taper_samples=DATA_MUTE_TAPER_SAMPLES,
    no_data_mute=NO_DATA_MUTE,
    storage_mode=STORAGE_MODE,
    storage_compression=STORAGE_COMPRESSION,
    check_adjoint=CHECK_ADJOINT,
    output_dir=OUTPUT_DIR,
)


@dataclass(slots=True)
class Lsrtm3DCase:
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
class LsrtmOptimizationResult:
    image: torch.Tensor
    predicted_scattered: torch.Tensor
    loss_history: list[float]
    residual_history: list[float]
    final_flag: int
    optimizer_iterations: int
    objective_evaluations: int


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
            "The Python backend does not support storage_compression='bf16'. "
            "Use BACKEND='native' with CUDA."
        )
    if backend_arg == "native":
        if backend_utils is None or not backend_utils.is_backend_available():
            raise SystemExit("Tide native backend was requested but is not available.")
        return False
    if backend_utils is None or not backend_utils.is_backend_available():
        raise SystemExit(
            "Tide native backend is not available. "
            "storage_compression='bf16' requires the native CUDA path."
        )
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


def load_h5_model(path: Path, dataset: str, nz: int | None, ny: int | None, nx: int | None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise KeyError(f"Dataset {dataset!r} not found in {path}.")
        model = h5[dataset][...]
    return crop_center_3d(model, nz, ny, nx)


def gaussion_smooth(
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
        torch.nn.functional.pad(subsurface, (0, 0, 0, 0, radius, radius), mode="replicate"),
        kernel_z,
    )
    smoothed = torch.nn.functional.conv3d(
        torch.nn.functional.pad(smoothed, (0, 0, radius, radius, 0, 0), mode="replicate"),
        kernel_y,
    )
    smoothed = torch.nn.functional.conv3d(
        torch.nn.functional.pad(smoothed, (radius, radius, 0, 0, 0, 0), mode="replicate"),
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
    return mask


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


def resolve_model_gradient_sampling_interval(sampling_interval: int) -> int:
    sampling_interval = tide.validate_model_gradient_sampling_interval(
        sampling_interval
    )
    return max(1, sampling_interval)


def downsample_receiver_data(data: torch.Tensor, sampling_interval: int) -> torch.Tensor:
    if sampling_interval <= 1 or data.numel() == 0:
        return data.contiguous()
    return tide.downsample(
        torch.movedim(data, 0, -1),
        sampling_interval,
    ).movedim(-1, 0).contiguous()


def match_data_sampling(case: Lsrtm3DCase, data: torch.Tensor) -> torch.Tensor:
    if data.shape[0] == case.data_weights.shape[0]:
        return data.contiguous()
    return downsample_receiver_data(data, case.model_gradient_sampling_interval)


def subsample_time_grid(data: torch.Tensor, sampling_interval: int) -> torch.Tensor:
    if sampling_interval <= 1 or data.numel() == 0:
        return data.contiguous()
    nt_down = data.shape[0] // sampling_interval
    return data[: nt_down * sampling_interval : sampling_interval].contiguous()


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
        torch.arange(shots_per_line, device=device, dtype=torch.long) * d_source + first_source
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
        f"Geometry: multi-x y-lines, line_xs={line_values}, "
        f"n_lines={n_lines}, shots_per_line={shots_per_line}, nshots={nshots}, "
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
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    if freq <= 0.0:
        raise ValueError("freq must be positive.")
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
    config: Lsrtm3DConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    python_backend: bool | Literal["eager", "jit", "compile"],
) -> Lsrtm3DCase:
    if device.type != "cuda":
        raise SystemExit("storage_compression='bf16' for 3D LSRTM requires CUDA.")
    if config.air_layers < 0:
        raise ValueError("air_layers must be >= 0.")
    if config.pml_width < 0:
        raise ValueError("pml_width must be >= 0.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
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

    epsilon_background = gaussion_smooth(
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
    estimated_nt = estimate_nt_for_bottom_coverage(
        epsilon_true,
        dx=config.dx,
        dt=dt,
        freq=config.freq,
        source_depth=int(source_locations[:, 0, 0].min().item()),
    )
    nt = estimated_nt if config.nt is None else config.nt
    if nt <= 0:
        raise ValueError("nt must be positive.")
    if config.nt is None:
        print(f"Auto-selected nt={nt} ({nt * dt * 1e6:.1f} us) for bottom coverage.")
    elif nt < estimated_nt:
        warnings.warn(
            f"nt={nt} ({nt * dt * 1e6:.1f} us) is shorter than the "
            f"estimated bottom-coverage nt={estimated_nt} "
            f"({estimated_nt * dt * 1e6:.1f} us). Deep reflectors may be missing.",
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
    data_weights = subsample_time_grid(
        data_weights,
        model_gradient_sampling_interval,
    )
    n_batches = (nshots + config.batch_size - 1) // config.batch_size
    print(
        f"Mini-batch: batch_size={config.batch_size}, n_batches={n_batches}, "
        f"model_gradient_sampling_interval={model_gradient_sampling_interval}, "
        f"data_nt={data_weights.shape[0]}."
    )

    return Lsrtm3DCase(
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


def make_shot_batches(case: Lsrtm3DCase) -> list[torch.Tensor]:
    shot_indices = torch.arange(
        case.source_locations.shape[0],
        device=case.source_locations.device,
        dtype=torch.long,
    )
    return [
        shot_indices[i : i + case.batch_size]
        for i in range(0, shot_indices.numel(), case.batch_size)
    ]


def max_velocity(case: Lsrtm3DCase) -> float:
    return case.max_vel


def forward_data_batch(
    case: Lsrtm3DCase,
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
        max_vel=max_velocity(case),
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
    case: Lsrtm3DCase,
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


def add_source_energy_to_illumination(
    illumination: torch.Tensor,
    state,
) -> None:
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
    case: Lsrtm3DCase,
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
    case: Lsrtm3DCase,
    data: torch.Tensor,
    shot_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = case.data_weights if shot_indices is None else case.data_weights[:, shot_indices, :]
    return data * weights.to(device=data.device, dtype=data.dtype)


def apply_mask(model: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (model * mask.to(device=model.device, dtype=model.dtype)).contiguous()


def born_forward_batch(
    case: Lsrtm3DCase,
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
        max_vel=max_velocity(case),
        parameterization="epsilon_sigma",
        linearize_source=True,
        source_component="ey",
        receiver_component="ey",
        python_backend=case.python_backend,
        storage_mode=case.storage_mode,
        storage_compression=case.storage_compression,
    )[-1]
    return match_data_sampling(case, data)


def born_forward_data(case: Lsrtm3DCase, scatter: torch.Tensor) -> torch.Tensor:
    batches = [
        born_forward_batch(case, scatter, shot_indices)
        for shot_indices in make_shot_batches(case)
    ]
    return torch.cat(batches, dim=1).contiguous()


def vjp_adjoint_image(
    case: Lsrtm3DCase,
    data_residual: torch.Tensor,
) -> torch.Tensor:
    """Exact discrete A^T r via VJP of the nonlinear Maxwell forward map."""
    image = torch.zeros_like(case.epsilon_background)
    for shot_indices in make_shot_batches(case):
        epsilon_background = (
            case.epsilon_background.detach().clone().requires_grad_(True)
        )
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


def adjoint_image(case: Lsrtm3DCase, data_residual: torch.Tensor) -> torch.Tensor:
    return vjp_adjoint_image(case, data_residual)


def relative_rms(error: torch.Tensor, reference: torch.Tensor) -> float:
    denom = reference.pow(2).mean().sqrt().clamp_min(torch.finfo(reference.dtype).eps)
    return float((error.pow(2).mean().sqrt() / denom).detach().cpu())


def cosine_similarity(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_norm = torch.linalg.norm(lhs).clamp_min(torch.finfo(lhs.dtype).eps)
    rhs_norm = torch.linalg.norm(rhs).clamp_min(torch.finfo(rhs.dtype).eps)
    return float((torch.sum(lhs * rhs) / (lhs_norm * rhs_norm)).detach().cpu())


def report_rtm_metrics(
    case: Lsrtm3DCase,
    *,
    rtm_image: torch.Tensor,
) -> None:
    reference = case.depsilon_true
    model_fit = relative_rms(rtm_image - reference, reference)
    cosine = cosine_similarity(rtm_image, reference)
    print(
        "RTM image against true perturbation: "
        f"relative_rms={model_fit:.4f} cosine={cosine:.4f}"
    )


def check_adjoint(case: Lsrtm3DCase) -> float:
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


def source_illumination(case: Lsrtm3DCase) -> torch.Tensor:
    illumination = torch.zeros_like(case.epsilon_background)
    for shot_indices in make_shot_batches(case):

        def callback(state) -> None:
            add_source_energy_to_illumination(illumination, state)

        with torch.no_grad():
            _ = tide.maxwell3d(
                case.epsilon_background.detach(),
                case.sigma,
                case.mu,
                grid_spacing=case.dx,
                dt=case.dt,
                source_amplitude=case.source_amplitude[shot_indices],
                source_location=case.source_locations[shot_indices],
                receiver_location=None,
                pml_width=case.pml_width,
                stencil=case.stencil,
                max_vel=max_velocity(case),
                model_gradient_sampling_interval=1,
                source_component="ey",
                receiver_component="ey",
                storage_mode="none",
                forward_callback=callback,
                callback_frequency=case.illumination_wavefield_sampling_interval,
            )
    return (illumination * case.dt * case.illumination_wavefield_sampling_interval).detach()


def illumination_preconditioner(
    case: Lsrtm3DCase,
    source_illumination_image: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source_illumination_image is None:
        source_illumination_image = source_illumination(case)
    illumination = apply_mask(
        source_illumination_image * case.model_weights,
        case.active_mask,
    )
    active = (case.active_mask > 0) & torch.isfinite(illumination) & (illumination > 0)
    if not active.any().item():
        warnings.warn(
            "Initial-model source illumination is zero in the active model region; "
            "falling back to an identity Krylov preconditioner.",
            RuntimeWarning,
        )
        return illumination, case.active_mask.detach().clone()

    active_values = illumination[active]
    eps = torch.finfo(illumination.dtype).eps
    floor = (active_values.max() * 1e-3).clamp_min(eps)
    safe_illumination = illumination + floor
    scale = torch.median(safe_illumination[case.active_mask > 0]).clamp_min(floor)
    preconditioner = torch.zeros_like(illumination)
    preconditioner[case.active_mask > 0] = (
        scale / safe_illumination[case.active_mask > 0]
    )
    return illumination.detach(), preconditioner.detach()


def run_lsrtm(
    case: Lsrtm3DCase,
    observed_scattered: torch.Tensor,
    *,
    krylov_iters: int,
    reg_lambda: float,
    preconditioner: torch.Tensor,
    initial_adjoint_image: torch.Tensor | None = None,
) -> LsrtmOptimizationResult:
    target = observed_scattered.detach()
    data_rms = target.pow(2).mean().sqrt().clamp_min(torch.finfo(target.dtype).eps)
    data_scale = data_rms.square()
    loss_history: list[float] = []
    residual_history: list[float] = []
    latest_pred = torch.zeros_like(target)

    if krylov_iters < 0:
        raise ValueError("krylov_iters must be >= 0.")

    def weighted_forward(model: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return apply_data_weights(case, born_forward_data(case, model)).detach()

    def objective_from_state(model: torch.Tensor, residual: torch.Tensor) -> tuple[float, float]:
        data_loss = 0.5 * residual.square().sum() / data_scale
        reg_loss = 0.5 * reg_lambda * model.square().sum()
        total_loss = data_loss + reg_loss
        if not torch.isfinite(total_loss).item():
            raise RuntimeError(
                "LSRTM PCGNR objective is not finite: "
                f"{float(total_loss.detach().cpu())!r}."
            )
        residual_rms = relative_rms(residual, target)
        return float(total_loss.detach().cpu()), residual_rms

    def normal_residual(
        model: torch.Tensor,
        residual: torch.Tensor,
        *,
        adjoint_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if adjoint_residual is None:
            adjoint_residual = adjoint_image(case, residual)
        else:
            adjoint_residual = adjoint_residual.detach()
            if adjoint_residual.shape != model.shape:
                raise ValueError(
                    "initial_adjoint_image must have the same shape as the model."
                )
        rt = adjoint_residual / data_scale
        if reg_lambda > 0.0:
            rt = rt - reg_lambda * model
        return apply_mask(rt, case.active_mask)

    def apply_preconditioner(normal: torch.Tensor) -> torch.Tensor:
        return apply_mask(normal * preconditioner, case.active_mask)

    model = torch.zeros_like(case.epsilon_background)
    if krylov_iters == 0:
        latest_pred = weighted_forward(model).detach()
        return LsrtmOptimizationResult(
            image=model,
            predicted_scattered=latest_pred,
            loss_history=loss_history,
            residual_history=residual_history,
            final_flag=2,
            optimizer_iterations=0,
            objective_evaluations=0,
        )

    residual = target.clone()
    rt = normal_residual(
        model,
        residual,
        adjoint_residual=initial_adjoint_image,
    )
    z = apply_preconditioner(rt)
    direction = z.clone()
    zrt_old = torch.sum(z * rt)
    fcost, residual_rms = objective_from_state(model, residual)
    loss_history.append(fcost)
    residual_history.append(residual_rms)
    objective_evaluations = 1
    print(
        "LSRTM PCGNR iter 000 "
        f"objective={fcost:.6e} residual_rms={residual_rms:.4f}"
    )

    final_flag = 2
    optimizer_iterations = 0
    eps = torch.finfo(model.dtype).eps
    if (not torch.isfinite(zrt_old).item()) or float(zrt_old.detach().cpu()) <= eps:
        warnings.warn(
            "LSRTM PCGNR has no usable initial search direction.",
            RuntimeWarning,
        )
        final_flag = 4
    else:
        for iteration in range(1, krylov_iters + 1):
            Ap = weighted_forward(direction)
            denom = Ap.square().sum() / data_scale
            if reg_lambda > 0.0:
                denom = denom + reg_lambda * direction.square().sum()
            if (not torch.isfinite(denom).item()) or float(denom.detach().cpu()) <= eps:
                warnings.warn(
                    "LSRTM PCGNR encountered a zero denominator "
                    f"at iter={iteration}.",
                    RuntimeWarning,
                )
                final_flag = 4
                break

            numerator = torch.sum(residual * Ap) / data_scale
            if reg_lambda > 0.0:
                numerator = numerator - reg_lambda * torch.sum(model * direction)
            direction_reset = False
            if (
                (not torch.isfinite(numerator).item())
                or float(numerator.detach().cpu()) <= eps
            ):
                direction = z.clone()
                direction_reset = True
                Ap = weighted_forward(direction)
                denom = Ap.square().sum() / data_scale
                if reg_lambda > 0.0:
                    denom = denom + reg_lambda * direction.square().sum()
                numerator = torch.sum(residual * Ap) / data_scale
                if reg_lambda > 0.0:
                    numerator = numerator - reg_lambda * torch.sum(model * direction)
                if (
                    (not torch.isfinite(denom).item())
                    or float(denom.detach().cpu()) <= eps
                    or (not torch.isfinite(numerator).item())
                    or float(numerator.detach().cpu()) <= eps
                ):
                    warnings.warn(
                        "LSRTM PCGNR has no descent direction "
                        f"at iter={iteration}.",
                        RuntimeWarning,
                    )
                    final_flag = 4
                    break

            alpha = numerator / denom
            model = apply_mask(model + alpha * direction, case.active_mask)
            residual = residual - alpha * Ap
            latest_pred = (target - residual).detach()
            optimizer_iterations = iteration

            fcost, residual_rms = objective_from_state(model, residual)
            objective_evaluations += 1
            loss_history.append(fcost)
            residual_history.append(residual_rms)
            print(
                f"LSRTM PCGNR iter {iteration:03d} "
                f"objective={fcost:.6e} residual_rms={residual_rms:.4f} "
                f"alpha={float(alpha.detach().cpu()):.4e} "
                f"numerator={float(numerator.detach().cpu()):.4e}"
                f"{' reset' if direction_reset else ''}"
            )

            rt = normal_residual(model, residual)
            z = apply_preconditioner(rt)
            zrt_new = torch.sum(z * rt)
            if (not torch.isfinite(zrt_new).item()) or float(zrt_new.detach().cpu()) <= eps:
                final_flag = 2
                break
            beta = zrt_new / zrt_old
            direction = apply_mask(z + beta * direction, case.active_mask)
            zrt_old = zrt_new

    if objective_evaluations == 1:
        latest_pred = (target - residual).detach()

    return LsrtmOptimizationResult(
        image=(case.model_weights * model.detach()),
        predicted_scattered=latest_pred,
        loss_history=loss_history,
        residual_history=residual_history,
        final_flag=final_flag,
        optimizer_iterations=optimizer_iterations,
        objective_evaluations=objective_evaluations,
    )


def percentile_limits(
    image: torch.Tensor,
    q: tuple[float, float] = (0.02, 0.98),
) -> tuple[float, float]:
    values = image.detach().flatten()
    if values.numel() == 0:
        return -1.0, 1.0
    lo = torch.quantile(values, q[0])
    hi = torch.quantile(values, q[1])
    if torch.isclose(lo, hi):
        scale = values.abs().max().clamp_min(1.0)
        return float(-scale.cpu()), float(scale.cpu())
    return float(lo.cpu()), float(hi.cpu())


def symmetric_abs_quantile_limit(image: torch.Tensor, quantile: float) -> float:
    values = image.detach().abs().flatten()
    if values.numel() == 0:
        return 1.0
    return float(torch.quantile(values, quantile).clamp_min(1e-12).cpu())


def center_line_slice(case: Lsrtm3DCase, image: torch.Tensor) -> torch.Tensor:
    return image[:, :, case.line_xs[len(case.line_xs) // 2]]


def line_slice(image: torch.Tensor, line_x: int) -> torch.Tensor:
    return image[:, :, line_x]


def receiver_data_panel(data: torch.Tensor) -> torch.Tensor:
    if data.ndim != 3:
        raise ValueError("receiver data must have shape [time, shot, receiver].")
    if data.shape[2] == 1:
        return data[:, :, 0]
    return data[:, 0, :]


def plot_rtm_preview(
    case: Lsrtm3DCase,
    *,
    observed_data: torch.Tensor,
    background_data: torch.Tensor,
    observed_scattered: torch.Tensor,
    rtm_image: torch.Tensor,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    image_panels = [
        ("True perturbation", center_line_slice(case, case.depsilon_true), "seismic"),
        ("RTM", center_line_slice(case, rtm_image), "gray"),
        ("Initial epsilon", center_line_slice(case, case.epsilon_background), "viridis"),
    ]
    for ax, (title, image, cmap) in zip(axes[0], image_panels, strict=True):
        values = image.detach().cpu().numpy()
        vmax = symmetric_abs_quantile_limit(image, 0.98)
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("y cell")
        ax.set_ylabel("z cell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    data_panels = [
        ("Observed records", receiver_data_panel(observed_data)),
        ("Background records", receiver_data_panel(background_data)),
        ("Observed - background", receiver_data_panel(observed_scattered)),
    ]
    for ax, (title, data) in zip(axes[1], data_panels, strict=True):
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
        ax.set_xlabel("shot" if data.shape[1] == case.source_locations.shape[0] else "receiver")
        ax.set_ylabel("time sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Tide 3D RTM preview ({case.model_label}, line x={case.line_xs}, "
        f"display x={case.line_xs[len(case.line_xs) // 2]})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "rtm_no_illumination_preview.png", dpi=180)
    plt.close(fig)


def plot_all_line_profiles(
    case: Lsrtm3DCase,
    *,
    rtm_image: torch.Tensor,
    lsrtm_image: torch.Tensor,
    output_dir: Path,
) -> None:
    n_lines = len(case.line_xs)
    fig, axes = plt.subplots(
        n_lines,
        3,
        figsize=(12.6, 3.2 * n_lines),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    true_vmax = symmetric_abs_quantile_limit(case.depsilon_true, 0.98)
    rtm_vmax = symmetric_abs_quantile_limit(rtm_image, 0.98)
    lsrtm_vmax = symmetric_abs_quantile_limit(lsrtm_image, 0.98)
    for row, line_x in enumerate(case.line_xs):
        panels = [
            ("True", line_slice(case.depsilon_true, line_x), true_vmax),
            ("RTM", line_slice(rtm_image, line_x), rtm_vmax),
            ("LSRTM", line_slice(lsrtm_image, line_x), lsrtm_vmax),
        ]
        for col, (title, image, vmax) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                image.detach().cpu().numpy(),
                aspect="auto",
                cmap="gray",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(f"{title} x={line_x}")
            ax.set_xlabel("y cell")
            ax.set_ylabel("z cell")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"All line profiles ({case.model_label})")
    fig.tight_layout()
    fig.savefig(output_dir / "deepwave_style_lsrtm_3d_all_line_profiles.png", dpi=180)
    plt.close(fig)


def plot_results(
    case: Lsrtm3DCase,
    *,
    observed_data: torch.Tensor,
    background_data: torch.Tensor,
    observed_scattered: torch.Tensor,
    predicted_scattered: torch.Tensor,
    rtm_image: torch.Tensor,
    lsrtm_image: torch.Tensor,
    loss_history: list[float],
    residual_history: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_residual = predicted_scattered - observed_scattered

    fig, axes = plt.subplots(3, 4, figsize=(18.0, 9.4))
    model_panels = [
        ("True epsilon", center_line_slice(case, case.epsilon_true), "viridis", None),
        ("Initial epsilon", center_line_slice(case, case.epsilon_background), "viridis", None),
        (
            "True perturbation",
            center_line_slice(case, case.depsilon_true),
            "seismic",
            percentile_limits(center_line_slice(case, case.depsilon_true)),
        ),
        (
            "RTM",
            center_line_slice(case, rtm_image),
            "gray",
            percentile_limits(center_line_slice(case, rtm_image)),
        ),
        (
            "LSRTM",
            center_line_slice(case, lsrtm_image),
            "gray",
            percentile_limits(center_line_slice(case, lsrtm_image)),
        ),
    ]
    for ax, (title, image, cmap, limits) in zip(axes.flat[: len(model_panels)], model_panels, strict=False):
        kwargs = {"aspect": "auto", "cmap": cmap}
        if limits is not None:
            vmax = max(abs(limits[0]), abs(limits[1]))
            kwargs.update(vmin=-vmax, vmax=vmax)
        im = ax.imshow(image.detach().cpu().numpy(), **kwargs)
        ax.set_title(title)
        ax.set_xlabel("y cell")
        ax.set_ylabel("z cell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    data_panels = [
        ("Observed records", receiver_data_panel(observed_data)),
        ("Background records", receiver_data_panel(background_data)),
        ("Observed - background", receiver_data_panel(observed_scattered)),
        ("Predicted scatter", receiver_data_panel(predicted_scattered)),
        ("Data residual", receiver_data_panel(data_residual)),
    ]
    data_start = len(model_panels)
    data_stop = data_start + len(data_panels)
    for ax, (title, data) in zip(axes.flat[data_start:data_stop], data_panels, strict=False):
        values = data.detach().cpu()
        im = ax.imshow(
            values.numpy(),
            aspect="auto",
            cmap="seismic",
            vmin=-100,
            vmax=100,
        )
        ax.set_title(title)
        ax.set_xlabel("shot" if data.shape[1] == case.source_locations.shape[0] else "receiver")
        ax.set_ylabel("time sample")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    curve_ax = axes.flat[data_stop]
    if loss_history:
        curve_ax.plot(
            np.arange(len(loss_history)),
            loss_history,
            marker="o",
        )
        curve_ax.set_title("PCGNR objective")
        curve_ax.set_xlabel("iteration")
        curve_ax.set_ylabel("objective")
        curve_ax.grid(True, alpha=0.25)
    else:
        curve_ax.axis("off")

    residual_ax = axes.flat[data_stop + 1]
    if residual_history:
        residual_ax.plot(
            np.arange(len(residual_history)),
            residual_history,
            marker="o",
        )
        residual_ax.set_title("Data residual RMS")
        residual_ax.set_xlabel("iteration")
        residual_ax.set_ylabel("relative RMS")
        residual_ax.grid(True, alpha=0.25)
    else:
        residual_ax.axis("off")
    for ax in axes.flat[data_stop + 2 :]:
        ax.axis("off")

    fig.suptitle(
        f"Tide 3D LSRTM ({case.model_label}, line x={case.line_xs}, "
        f"display x={case.line_xs[len(case.line_xs) // 2]})"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "deepwave_style_lsrtm_3d_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharex=True, sharey=True)
    simple_panels = [
        ("True perturbation", center_line_slice(case, case.depsilon_true), "seismic"),
        ("RTM", center_line_slice(case, rtm_image), "gray"),
        ("LSRTM", center_line_slice(case, lsrtm_image), "gray"),
    ]
    for ax, (title, image, cmap) in zip(axes, simple_panels, strict=True):
        im = ax.imshow(image.detach().cpu().numpy(), aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("y cell")
        ax.set_ylabel("z cell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "deepwave_style_lsrtm_3d_models.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), sharex=True, sharey=True)
    record_panels = [
        ("Observed - background", receiver_data_panel(observed_scattered)),
        ("Predicted scatter", receiver_data_panel(predicted_scattered)),
        ("Residual", receiver_data_panel(data_residual)),
    ]
    for ax, (title, data) in zip(axes, record_panels, strict=True):
        values = data.detach().cpu()
        vmax = symmetric_abs_quantile_limit(values, SHOT_DISPLAY_QUANTILE)
        ax.imshow(values.numpy(), aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("shot")
        ax.set_ylabel("time sample")
    fig.tight_layout()
    fig.savefig(output_dir / "deepwave_style_lsrtm_3d_data_residual.png", dpi=180)
    plt.close(fig)

    if loss_history:
        plt.figure(figsize=(5.0, 3.5))
        plt.plot(np.arange(len(loss_history)), loss_history, marker="o")
        plt.xlabel("iteration")
        plt.ylabel("objective")
        plt.title("PCGNR objective")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "deepwave_style_lsrtm_3d_loss.png", dpi=180)
        plt.close()

    plot_all_line_profiles(
        case,
        rtm_image=rtm_image,
        lsrtm_image=lsrtm_image,
        output_dir=output_dir,
    )


def h5_write_tensor(group: h5py.Group, name: str, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    group.create_dataset(name, data=array, compression="gzip", shuffle=True)


def save_rtm_h5(
    case: Lsrtm3DCase,
    *,
    observed_data: torch.Tensor,
    background_data: torch.Tensor,
    observed_scattered: torch.Tensor,
    rtm_image: torch.Tensor,
    timings: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rtm_no_illumination_results.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["workflow"] = "rtm_only"
        h5.attrs["model_label"] = case.model_label
        h5.attrs["dx"] = case.dx
        h5.attrs["dt"] = case.dt
        h5.attrs["freq"] = case.freq
        h5.attrs["pml_width"] = case.pml_width
        h5.attrs["stencil"] = case.stencil
        h5.attrs["storage_mode"] = case.storage_mode
        h5.attrs["storage_compression"] = case.storage_compression
        h5.attrs["source_component"] = "ey"
        h5.attrs["receiver_component"] = "ey"
        h5.attrs["line_xs"] = np.asarray(case.line_xs, dtype=np.int64)
        h5.attrs["adjoint_operator"] = "weighted_maxwell_vjp"
        h5.attrs["model_gradient_sampling_interval"] = (
            case.model_gradient_sampling_interval
        )

        models = h5.create_group("models")
        h5_write_tensor(models, "epsilon_true", case.epsilon_true)
        h5_write_tensor(models, "epsilon_background", case.epsilon_background)
        h5_write_tensor(models, "depsilon_true", case.depsilon_true)
        h5_write_tensor(models, "rtm_image", rtm_image)

        data = h5.create_group("data")
        h5_write_tensor(data, "observed_data", observed_data)
        h5_write_tensor(data, "background_data", background_data)
        h5_write_tensor(data, "observed_scattered", observed_scattered)

        geometry = h5.create_group("geometry")
        geometry.create_dataset("source_locations", data=case.source_locations.detach().cpu().numpy())
        geometry.create_dataset("receiver_locations", data=case.receiver_locations.detach().cpu().numpy())
        geometry.create_dataset("line_xs", data=np.asarray(case.line_xs, dtype=np.int64))

        timing_group = h5.create_group("timings")
        for name, seconds in timings.items():
            timing_group.attrs[name] = float(seconds)
    print(f"Wrote RTM HDF5 results to {path}")


def save_h5(
    case: Lsrtm3DCase,
    *,
    observed_data: torch.Tensor,
    background_data: torch.Tensor,
    observed_scattered: torch.Tensor,
    rtm_image: torch.Tensor,
    result: LsrtmOptimizationResult,
    source_illumination_image: torch.Tensor,
    illumination_preconditioner_image: torch.Tensor,
    timings: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_residual = result.predicted_scattered - observed_scattered
    path = output_dir / "deepwave_style_lsrtm_3d_results.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["workflow"] = "lsrtm_pcgnr"
        h5.attrs["model_label"] = case.model_label
        h5.attrs["optimizer"] = "custom_preconditioned_cgnr"
        h5.attrs["preconditioner"] = "initial_model_source_illumination_diagonal"
        h5.attrs["adjoint_operator"] = "weighted_maxwell_vjp"
        h5.attrs["optimizer_final_flag"] = result.final_flag
        h5.attrs["optimizer_iterations"] = result.optimizer_iterations
        h5.attrs["objective_evaluations"] = result.objective_evaluations
        h5.attrs["dx"] = case.dx
        h5.attrs["dt"] = case.dt
        h5.attrs["freq"] = case.freq
        h5.attrs["pml_width"] = case.pml_width
        h5.attrs["stencil"] = case.stencil
        h5.attrs["storage_mode"] = case.storage_mode
        h5.attrs["storage_compression"] = case.storage_compression
        h5.attrs["source_component"] = "ey"
        h5.attrs["receiver_component"] = "ey"
        h5.attrs["line_xs"] = np.asarray(case.line_xs, dtype=np.int64)
        h5.attrs["model_gradient_sampling_interval"] = (
            case.model_gradient_sampling_interval
        )
        h5.attrs["source_illumination_sampling_interval"] = (
            case.illumination_wavefield_sampling_interval
        )

        models = h5.create_group("models")
        h5_write_tensor(models, "epsilon_true", case.epsilon_true)
        h5_write_tensor(models, "epsilon_background", case.epsilon_background)
        h5_write_tensor(models, "depsilon_true", case.depsilon_true)
        h5_write_tensor(models, "rtm_image", rtm_image)
        h5_write_tensor(models, "lsrtm_image", result.image)
        h5_write_tensor(models, "source_illumination", source_illumination_image)
        h5_write_tensor(
            models,
            "illumination_preconditioner",
            illumination_preconditioner_image,
        )

        data = h5.create_group("data")
        h5_write_tensor(data, "observed_data", observed_data)
        h5_write_tensor(data, "background_data", background_data)
        h5_write_tensor(data, "observed_scattered", observed_scattered)
        h5_write_tensor(data, "predicted_scattered", result.predicted_scattered)
        h5_write_tensor(data, "data_residual", data_residual)

        geometry = h5.create_group("geometry")
        geometry.create_dataset("source_locations", data=case.source_locations.detach().cpu().numpy())
        geometry.create_dataset("receiver_locations", data=case.receiver_locations.detach().cpu().numpy())
        geometry.create_dataset("line_xs", data=np.asarray(case.line_xs, dtype=np.int64))

        curves = h5.create_group("curves")
        curves.create_dataset(
            "loss_history",
            data=np.asarray(result.loss_history, dtype=np.float64),
        )
        curves.create_dataset(
            "residual_history",
            data=np.asarray(result.residual_history, dtype=np.float64),
        )

        timing_group = h5.create_group("timings")
        for name, seconds in timings.items():
            timing_group.attrs[name] = float(seconds)
    print(f"Wrote HDF5 results to {path}")


def validate_config(config: Lsrtm3DConfig) -> None:
    choices = {
        "device": (config.device, {"auto", "cuda"}),
        "dtype": (config.dtype, {"float32", "float64"}),
        "backend": (config.backend, {"auto", "native"}),
        "stencil": (config.stencil, {2, 4, 6, 8}),
        "storage_mode": (config.storage_mode, {"device", "cpu", "disk", "auto"}),
        "storage_compression": (config.storage_compression, {"bf16", "none"}),
    }
    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    if config.krylov_iters < 0:
        raise ValueError("krylov_iters must be >= 0.")
    if config.reg_lambda < 0.0:
        raise ValueError("reg_lambda must be >= 0.")


def main(config: Lsrtm3DConfig = CONFIG) -> None:
    total_start = perf_counter()
    validate_config(config)

    device = resolve_device(config.device)
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    python_backend = resolve_backend(config.backend)

    print(
        "Building 3D RTM case "
        f"device={device} dtype={dtype} backend=native "
        f"storage_compression={config.storage_compression}"
    )
    case, build_time = timed_stage(
        "Build case",
        device,
        lambda: build_case(config, device=device, dtype=dtype, python_backend=python_backend),
    )
    timings: dict[str, float] = {"build_case": build_time}

    def forward_stage() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            print("Forward modelling observed and background data")
            observed_data = forward_data(case, case.epsilon_true).detach()
            background_data, source_illumination_image = (
                forward_data_with_source_illumination(
                    case,
                    case.epsilon_background.detach(),
                )
            )
            background_data = background_data.detach()
            source_illumination_image = source_illumination_image.detach()
            observed_scattered = apply_data_weights(case, observed_data - background_data).detach()
        return observed_data, background_data, observed_scattered, source_illumination_image

    (
        observed_data,
        background_data,
        observed_scattered,
        source_illumination_image,
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

    report_rtm_metrics(
        case,
        rtm_image=rtm_image,
    )

    plot_rtm_preview(
        case,
        observed_data=observed_data,
        background_data=background_data,
        observed_scattered=observed_scattered,
        rtm_image=rtm_image,
        output_dir=config.output_dir,
    )
    save_rtm_h5(
        case,
        observed_data=observed_data,
        background_data=background_data,
        observed_scattered=observed_scattered,
        rtm_image=rtm_image,
        timings=timings,
        output_dir=config.output_dir,
    )

    print("Computing initial-model source illumination preconditioner")
    (illumination, preconditioner), preconditioner_time = timed_stage(
        "Initial-model source illumination",
        device,
        lambda: illumination_preconditioner(case, source_illumination_image),
    )
    timings["illumination_preconditioner"] = preconditioner_time

    print("Running PCGNR LSRTM")
    result, lsrtm_time = timed_stage(
        "PCGNR LSRTM",
        device,
        lambda: run_lsrtm(
            case,
            observed_scattered,
            krylov_iters=config.krylov_iters,
            reg_lambda=config.reg_lambda,
            preconditioner=preconditioner,
            initial_adjoint_image=rtm_image,
        ),
    )
    timings["pcgnr_lsrtm"] = lsrtm_time

    plot_results(
        case,
        observed_data=observed_data,
        background_data=background_data,
        observed_scattered=observed_scattered,
        predicted_scattered=result.predicted_scattered,
        rtm_image=rtm_image,
        lsrtm_image=result.image,
        loss_history=result.loss_history,
        residual_history=result.residual_history,
        output_dir=config.output_dir,
    )

    total_time = perf_counter() - total_start
    timings["total"] = total_time
    save_h5(
        case,
        observed_data=observed_data,
        background_data=background_data,
        observed_scattered=observed_scattered,
        rtm_image=rtm_image,
        result=result,
        source_illumination_image=illumination,
        illumination_preconditioner_image=preconditioner,
        timings=timings,
        output_dir=config.output_dir,
    )
    print("Timing summary:")
    for name, seconds in timings.items():
        print(f"  {name}: {format_seconds(seconds)}")
    print(f"Wrote plots to {config.output_dir}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
