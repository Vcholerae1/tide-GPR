"""Run a small 3D GPR forward example on a 50 x 50 x 50 petrophysical sub-block."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

import tide

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SimulationConfig:
    model: str = "examples/data/Petrophysical_Models.h5"
    output: str = "examples/outputs/petrophysical_survey_subcube_forward.h5"
    device: str = "auto"
    batch_size: int = 16
    max_shots: int | None = None
    dry_run: bool = False
    python_backend: bool = False
    stencil: int = 4
    source_component: str = "ey"
    receiver_component: str = "ey"
    freq: float = 100e6
    time_window_ns: float = 80.0
    dt_ns: float = 0.040
    air_thickness_m: float = 0.5
    pml_thickness_m: float = 0.60
    offset_m: float = 0.30
    inline_spacing_m: float = 0.10
    crossline_spacing_m: float = 0.10
    n_lines: int = 50
    traces_per_line: int = 47
    decimation_stride: int = 4
    crop_x_start: int = 0
    crop_y_start: int = 0
    crop_z_start: int = 0
    crop_nx: int = 50
    crop_ny: int = 50
    crop_nz: int = 60
    use_subcuboids: bool = False


CONFIG = SimulationConfig()


def _render_progress(done: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "[invalid] 0/0"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    bar = "=" * filled + "." * (width - filled)
    return f"[{bar}] {done}/{total} {ratio * 100:5.1f}%"


def _resolve_device(name: str) -> torch.device:
    value = name.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {name!r}")


def _estimate_courant_number_3d(dt: float, dz: float, dy: float, dx: float, max_vel: float) -> float:
    return float(abs(dt) * max_vel * math.sqrt((1.0 / dz**2) + (1.0 / dy**2) + (1.0 / dx**2)))


def _build_figure_paths(output_path: Path) -> dict[str, Path]:
    stem = output_path.with_suffix("")
    return {
        "geometry_xy": stem.parent / f"{stem.name}_survey_geometry_xy.png",
        "model_slices": stem.parent / f"{stem.name}_model_slices.png",
        "trace_gather": stem.parent / f"{stem.name}_trace_gather.png",
    }


def _resolve_output_path(output: str | Path) -> Path:
    requested = Path(output)
    output_dir = requested.parent / requested.stem
    return output_dir / f"{requested.stem}.h5"


def _save_survey_geometry_xy(
    path: Path,
    *,
    survey_meta: dict[str, np.ndarray | int | float],
) -> None:
    src_x = np.asarray(survey_meta["src_x_m"], dtype=np.float32)
    rec_x = np.asarray(survey_meta["rec_x_m"], dtype=np.float32)
    line_y = np.asarray(survey_meta["line_y_m"], dtype=np.float32)
    src_xx, line_yy = np.meshgrid(src_x, line_y)
    rec_xx, _ = np.meshgrid(rec_x, line_y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(src_xx.ravel(), line_yy.ravel(), s=10, c="#c44e52", label="Sources")
    ax.scatter(rec_xx.ravel(), line_yy.ravel(), s=10, c="#4c72b0", label="Receivers")
    for y_m, src_row, rec_row in zip(line_y, src_xx, rec_xx, strict=True):
        ax.plot(
            np.stack((src_row, rec_row), axis=0),
            np.full((2, src_row.size), y_m),
            color="0.75",
            linewidth=0.6,
            alpha=0.7,
        )
    ax.set_title("Survey Geometry (XY Top View)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_model_slices(
    path: Path,
    *,
    epsilon: np.ndarray,
    model_meta: dict[str, float | int],
    survey_meta: dict[str, np.ndarray | int | float],
) -> None:
    pml = int(model_meta["pml_cells"])
    air = int(model_meta["air_cells"])
    eps_inner = epsilon[pml:-pml, pml:-pml, pml:-pml]

    z_idx = min(air, eps_inner.shape[0] - 1)
    y_idx = eps_inner.shape[1] // 2
    x_idx = eps_inner.shape[2] // 2

    dx = float(model_meta["dx"])
    dy = float(model_meta["dy"])
    dz = float(model_meta["dz"])
    x0 = float(model_meta["x0_subsurface_m"])
    y0 = float(model_meta["y0_subsurface_m"])
    z0 = -float(model_meta["air_cells"]) * dz
    x_extent = [x0, x0 + eps_inner.shape[2] * dx, y0, y0 + eps_inner.shape[1] * dy]
    xz_extent = [x0, x0 + eps_inner.shape[2] * dx, z0 + eps_inner.shape[0] * dz, z0]
    yz_extent = [y0, y0 + eps_inner.shape[1] * dy, z0 + eps_inner.shape[0] * dz, z0]

    src_x = np.asarray(survey_meta["src_x_m"], dtype=np.float32)
    rec_x = np.asarray(survey_meta["rec_x_m"], dtype=np.float32)
    line_y = np.asarray(survey_meta["line_y_m"], dtype=np.float32)
    z_air_side = float(survey_meta["z_m_air_side"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im0 = axes[0].imshow(eps_inner[z_idx], origin="lower", aspect="auto", extent=x_extent, cmap="viridis")
    src_xx, line_yy = np.meshgrid(src_x, line_y)
    rec_xx, _ = np.meshgrid(rec_x, line_y)
    axes[0].scatter(src_xx.ravel(), line_yy.ravel(), s=6, c="white", alpha=0.9)
    axes[0].scatter(rec_xx.ravel(), line_yy.ravel(), s=6, c="#ffb000", alpha=0.9)
    axes[0].set_title("XY Slice")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    im1 = axes[1].imshow(eps_inner[:, y_idx, :], origin="upper", aspect="auto", extent=xz_extent, cmap="viridis")
    axes[1].scatter(src_x, np.full_like(src_x, z_air_side), s=12, c="white", alpha=0.9)
    axes[1].scatter(rec_x, np.full_like(rec_x, z_air_side), s=12, c="#ffb000", alpha=0.9)
    axes[1].set_title("XZ Slice")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("z (m)")

    im2 = axes[2].imshow(eps_inner[:, :, x_idx], origin="upper", aspect="auto", extent=yz_extent, cmap="viridis")
    axes[2].scatter(line_y, np.full_like(line_y, z_air_side), s=12, c="white", alpha=0.9)
    axes[2].set_title("YZ Slice")
    axes[2].set_xlabel("y (m)")
    axes[2].set_ylabel("z (m)")

    cbar = fig.colorbar(im2, ax=axes, shrink=0.9)
    cbar.set_label("epsilon_r")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_trace_gather(
    path: Path,
    *,
    traces: np.ndarray,
    nt: int,
    dt: float,
    survey_meta: dict[str, np.ndarray | int | float],
    n_total_shots: int,
) -> None:
    if n_total_shots == int(survey_meta["n_shots"]):
        n_lines = int(survey_meta["n_lines"])
        traces_per_line = int(survey_meta["traces_per_line"])
        gather = traces.reshape(n_lines * traces_per_line, nt)
    else:
        gather = traces

    vmax = float(np.percentile(np.abs(gather), 99.0))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(
        gather.T,
        cmap="seismic",
        aspect="auto",
        origin="upper",
        vmin=-vmax,
        vmax=vmax,
        extent=[0, gather.shape[0], nt * dt * 1e9, 0],
    )
    ax.set_title("Trace Gather")
    ax.set_xlabel("shot index")
    ax.set_ylabel("time (ns)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_petrophysical_model(
    path: str | Path,
    *,
    air_thickness_m: float,
    pml_thickness_m: float,
    decimation_stride: int,
    x_start: int,
    y_start: int,
    z_start: int,
    nx_crop: int,
    ny_crop: int,
    nz_crop: int,
    mu_r_air: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    with h5py.File(path, "r") as f:
        xvec = f["xvec"][:]
        yvec = f["yvec"][:]
        zvec = f["zvec"][:]
        full_shape = (yvec.size, xvec.size, zvec.size)

        if decimation_stride <= 0:
            raise ValueError("decimation_stride must be positive.")
        if x_start < 0 or y_start < 0 or z_start < 0:
            raise ValueError("Crop start indices must be non-negative.")
        if nx_crop <= 0 or ny_crop <= 0 or nz_crop <= 0:
            raise ValueError("Crop sizes must be positive.")
        if x_start + (nx_crop - 1) * decimation_stride >= full_shape[1]:
            raise ValueError("Requested x crop exceeds the model extent.")
        if y_start + (ny_crop - 1) * decimation_stride >= full_shape[0]:
            raise ValueError("Requested y crop exceeds the model extent.")
        if z_start + (nz_crop - 1) * decimation_stride >= full_shape[2]:
            raise ValueError("Requested z crop exceeds the model extent.")

        y_slice = slice(y_start, y_start + ny_crop * decimation_stride, decimation_stride)
        x_slice = slice(x_start, x_start + nx_crop * decimation_stride, decimation_stride)
        z_slice = slice(z_start, z_start + nz_crop * decimation_stride, decimation_stride)
        epsilon_r = f["epsilon_r"][y_slice, x_slice, z_slice]
        sigma = f["sigma"][y_slice, x_slice, z_slice]
        xvec = xvec[x_slice]
        yvec = yvec[y_slice]
        zvec = zvec[z_slice]

    if epsilon_r.shape != sigma.shape:
        raise ValueError("epsilon_r and sigma must share the same shape.")
    if epsilon_r.shape != (yvec.size, xvec.size, zvec.size):
        raise ValueError(
            "Expected datasets ordered as (y, x, z) matching yvec/xvec/zvec lengths."
        )

    dx = float(np.mean(np.diff(xvec)))
    dy = float(np.mean(np.diff(yvec)))
    dz = float(np.mean(np.diff(zvec)))
    spacing_tol = max(abs(dx), abs(dy), abs(dz)) * 1e-4
    if not np.isclose(dx, dy, atol=spacing_tol, rtol=0.0) or not np.isclose(
        dx, dz, atol=spacing_tol, rtol=0.0
    ):
        raise ValueError("This survey script expects isotropic spacing in x/y/z.")

    spacing_m = dx
    air_cells = int(round(air_thickness_m / spacing_m))
    pml_cells = int(round(pml_thickness_m / spacing_m))
    align_tol = spacing_m * 1e-4
    if not math.isclose(
        air_cells * spacing_m,
        air_thickness_m,
        rel_tol=0.0,
        abs_tol=align_tol,
    ):
        raise ValueError("air_thickness_m must align with the model spacing.")
    if not math.isclose(
        pml_cells * spacing_m,
        pml_thickness_m,
        rel_tol=0.0,
        abs_tol=align_tol,
    ):
        raise ValueError("pml_thickness_m must align with the model spacing.")

    # HDF5 is stored as (y, x, z); the solver expects (z, y, x).
    epsilon_zyx = np.transpose(epsilon_r, (2, 0, 1)).astype(np.float32, copy=False)
    sigma_zyx = np.transpose(sigma, (2, 0, 1)).astype(np.float32, copy=False)
    mu_zyx = np.ones_like(epsilon_zyx, dtype=np.float32)

    air_shape = (air_cells, epsilon_zyx.shape[1], epsilon_zyx.shape[2])
    epsilon_with_air = np.concatenate(
        [np.ones(air_shape, dtype=np.float32), epsilon_zyx],
        axis=0,
    )
    sigma_with_air = np.concatenate(
        [np.zeros(air_shape, dtype=np.float32), sigma_zyx],
        axis=0,
    )
    mu_with_air = np.concatenate(
        [np.full(air_shape, mu_r_air, dtype=np.float32), mu_zyx],
        axis=0,
    )

    pad_width = ((pml_cells, pml_cells), (pml_cells, pml_cells), (pml_cells, pml_cells))
    epsilon_full = np.pad(epsilon_with_air, pad_width, mode="edge")
    sigma_full = np.pad(sigma_with_air, pad_width, mode="edge")
    mu_full = np.pad(mu_with_air, pad_width, mode="edge")

    meta = {
        "dx": spacing_m,
        "dy": spacing_m,
        "dz": spacing_m,
        "air_cells": air_cells,
        "pml_cells": pml_cells,
        "nx_subsurface": int(xvec.size),
        "ny_subsurface": int(yvec.size),
        "nz_subsurface": int(zvec.size),
        "nx_total": int(epsilon_full.shape[2]),
        "ny_total": int(epsilon_full.shape[1]),
        "nz_total": int(epsilon_full.shape[0]),
        "x0_subsurface_m": float(xvec[0]),
        "y0_subsurface_m": float(yvec[0]),
        "z0_subsurface_m": float(zvec[0]),
        "x1_subsurface_m": float(xvec[-1]),
        "y1_subsurface_m": float(yvec[-1]),
        "z1_subsurface_m": float(zvec[-1]),
        "x_start_idx": x_start,
        "y_start_idx": y_start,
        "z_start_idx": z_start,
        "decimation_stride": decimation_stride,
        "nx_crop": nx_crop,
        "ny_crop": ny_crop,
        "nz_crop": nz_crop,
    }
    return epsilon_full, sigma_full, mu_full, meta


def _build_constant_offset_survey(
    *,
    nx_subsurface: int,
    ny_subsurface: int,
    air_cells: int,
    pml_cells: int,
    dx: float,
    dy: float,
    inline_spacing_m: float,
    crossline_spacing_m: float,
    offset_m: float,
    n_lines: int,
    traces_per_line: int,
    x0_subsurface_m: float,
    y0_subsurface_m: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray | int | float]]:
    inline_step_cells = int(round(inline_spacing_m / dx))
    crossline_step_cells = int(round(crossline_spacing_m / dy))
    offset_cells = int(round(offset_m / dx))
    align_tol = max(dx, dy) * 1e-4
    if inline_step_cells <= 0 or crossline_step_cells <= 0 or offset_cells <= 0:
        raise ValueError("Survey spacings and offset must be positive.")
    if not math.isclose(
        inline_step_cells * dx,
        inline_spacing_m,
        rel_tol=0.0,
        abs_tol=align_tol,
    ):
        raise ValueError("inline_spacing_m must align with the model spacing.")
    if not math.isclose(
        crossline_step_cells * dy,
        crossline_spacing_m,
        rel_tol=0.0,
        abs_tol=align_tol,
    ):
        raise ValueError("crossline_spacing_m must align with the model spacing.")
    if not math.isclose(offset_cells * dx, offset_m, rel_tol=0.0, abs_tol=align_tol):
        raise ValueError("offset_m must align with the model spacing.")

    src_x_idx = np.arange(0, traces_per_line * inline_step_cells, inline_step_cells, dtype=np.int64)
    rec_x_idx = src_x_idx + offset_cells
    line_y_idx = np.arange(0, n_lines * crossline_step_cells, crossline_step_cells, dtype=np.int64)

    if src_x_idx[-1] >= nx_subsurface:
        raise ValueError("Source positions extend beyond the subsurface model in x.")
    if rec_x_idx[-1] >= nx_subsurface:
        raise ValueError("Receiver positions extend beyond the subsurface model in x.")
    if line_y_idx[-1] >= ny_subsurface:
        raise ValueError("Line positions extend beyond the subsurface model in y.")

    n_shots = n_lines * traces_per_line
    source_location = torch.zeros((n_shots, 1, 3), dtype=torch.long)
    receiver_location = torch.zeros((n_shots, 1, 3), dtype=torch.long)

    # The air-ground interface is approximated by the last air cell center.
    interface_z_idx = pml_cells + air_cells - 1
    shot = 0
    for y_idx in line_y_idx:
        y_full = pml_cells + int(y_idx)
        for sx_idx, rx_idx in zip(src_x_idx, rec_x_idx, strict=True):
            source_location[shot, 0] = torch.tensor(
                [interface_z_idx, y_full, pml_cells + int(sx_idx)],
                dtype=torch.long,
            )
            receiver_location[shot, 0] = torch.tensor(
                [interface_z_idx, y_full, pml_cells + int(rx_idx)],
                dtype=torch.long,
            )
            shot += 1

    survey_meta = {
        "n_lines": n_lines,
        "traces_per_line": traces_per_line,
        "n_shots": n_shots,
        "line_y_idx": line_y_idx,
        "src_x_idx": src_x_idx,
        "rec_x_idx": rec_x_idx,
        "line_y_m": y0_subsurface_m + line_y_idx.astype(np.float32) * dy,
        "src_x_m": x0_subsurface_m + src_x_idx.astype(np.float32) * dx,
        "rec_x_m": x0_subsurface_m + rec_x_idx.astype(np.float32) * dx,
        "z_m_air_side": np.float32(-(0.5 * dx)),
        "offset_m": float(offset_m),
        "inline_spacing_m": float(inline_spacing_m),
        "crossline_spacing_m": float(crossline_spacing_m),
    }
    return source_location, receiver_location, survey_meta


def _build_subcuboid_plan(
    *,
    line_y_idx: np.ndarray,
    line_y_m: np.ndarray,
    dy: float,
    pml_cells: int,
    air_cells: int,
) -> list[dict[str, object]]:
    if line_y_idx.size == 0:
        return []

    y_min_idx = int(line_y_idx.min())
    y_max_idx = int(line_y_idx.max())
    total_span = max(y_max_idx - y_min_idx + 1, 1)
    base = max(1, total_span // 3)
    overlap = max(1, base // 3)
    model_bounds_idx = [
        (y_min_idx, min(y_min_idx + base + overlap, y_max_idx + 1)),
        (
            max(y_min_idx, y_min_idx + base - overlap),
            min(y_min_idx + 2 * base + overlap, y_max_idx + 1),
        ),
        (max(y_min_idx, y_min_idx + 2 * base - overlap), y_max_idx + 1),
    ]
    result: list[dict[str, object]] = []
    for cuboid_id, (model_y0_idx, model_y1_idx) in enumerate(model_bounds_idx, start=1):
        if model_y1_idx <= model_y0_idx:
            continue
        if cuboid_id == 1:
            line_mask = line_y_idx < model_y1_idx
        elif cuboid_id == 2:
            line_mask = (line_y_idx >= model_y0_idx) & (line_y_idx < model_y1_idx)
        else:
            line_mask = line_y_idx >= model_y0_idx
        assigned_line_indices = np.flatnonzero(line_mask)
        model_y0_m = float(model_y0_idx * dy)
        model_y1_m = float(model_y1_idx * dy)
        result.append(
            {
                "cuboid_id": cuboid_id,
                "model_y0_m": model_y0_m,
                "model_y1_m": model_y1_m,
                "line_y0_m": float(line_y_m[assigned_line_indices[0]])
                if assigned_line_indices.size
                else model_y0_m,
                "line_y1_m": float(line_y_m[assigned_line_indices[-1]])
                if assigned_line_indices.size
                else model_y1_m,
                "model_y0_idx": model_y0_idx,
                "model_y1_idx": model_y1_idx,
                "model_y0_idx_full": pml_cells + model_y0_idx,
                "model_y1_idx_full": pml_cells + model_y1_idx,
                "assigned_line_indices": assigned_line_indices,
                "assigned_line_y_m": line_y_m[assigned_line_indices].astype(np.float32),
                "interface_z_idx": pml_cells + air_cells - 1,
            }
        )
    return result


def _slice_subcuboid_model(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    *,
    y0_full: int,
    y1_full: int,
    pml_cells: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = max(0, y0_full - pml_cells)
    stop = min(epsilon.shape[1], y1_full + pml_cells + 1)
    return (
        epsilon[:, start:stop, :].contiguous(),
        sigma[:, start:stop, :].contiguous(),
        mu[:, start:stop, :].contiguous(),
    )


def _remap_locations_to_subcuboid(
    locations: torch.Tensor,
    *,
    y0_full: int,
    pml_cells: int,
) -> torch.Tensor:
    start = y0_full - pml_cells
    remapped = locations.clone()
    remapped[..., 1] = remapped[..., 1] - start
    return remapped


def main(config: SimulationConfig = CONFIG) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if config.max_shots is not None and config.max_shots <= 0:
        raise ValueError("max-shots must be positive when provided.")

    device = _resolve_device(config.device)
    output_path = _resolve_output_path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon_np, sigma_np, mu_np, model_meta = _load_petrophysical_model(
        config.model,
        air_thickness_m=config.air_thickness_m,
        pml_thickness_m=config.pml_thickness_m,
        decimation_stride=config.decimation_stride,
        x_start=config.crop_x_start,
        y_start=config.crop_y_start,
        z_start=config.crop_z_start,
        nx_crop=config.crop_nx,
        ny_crop=config.crop_ny,
        nz_crop=config.crop_nz,
    )
    source_location, receiver_location, survey_meta = _build_constant_offset_survey(
        nx_subsurface=int(model_meta["nx_subsurface"]),
        ny_subsurface=int(model_meta["ny_subsurface"]),
        air_cells=int(model_meta["air_cells"]),
        pml_cells=int(model_meta["pml_cells"]),
        dx=float(model_meta["dx"]),
        dy=float(model_meta["dy"]),
        inline_spacing_m=config.inline_spacing_m,
        crossline_spacing_m=config.crossline_spacing_m,
        offset_m=config.offset_m,
        n_lines=config.n_lines,
        traces_per_line=config.traces_per_line,
        x0_subsurface_m=float(model_meta["x0_subsurface_m"]),
        y0_subsurface_m=float(model_meta["y0_subsurface_m"]),
    )
    subcuboid_plan = _build_subcuboid_plan(
        line_y_idx=np.asarray(survey_meta["line_y_idx"], dtype=np.int64),
        line_y_m=np.asarray(survey_meta["line_y_m"], dtype=np.float32),
        dy=float(model_meta["dy"]),
        pml_cells=int(model_meta["pml_cells"]),
        air_cells=int(model_meta["air_cells"]),
    )

    nt = int(round(config.time_window_ns / config.dt_ns))
    dt = config.dt_ns * 1e-9
    max_vel = tide.utils.C0 / math.sqrt(float(np.min(epsilon_np * mu_np)))
    courant = _estimate_courant_number_3d(
        dt,
        float(model_meta["dz"]),
        float(model_meta["dy"]),
        float(model_meta["dx"]),
        max_vel,
    )
    recommended_dt = 0.9 * float(model_meta["dx"]) / (max_vel * math.sqrt(3.0))
    wavelet = tide.ricker(
        config.freq,
        nt,
        dt,
        peak_time=1.2 / config.freq,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, nt)

    n_total_shots = int(survey_meta["n_shots"])
    if config.max_shots is not None:
        n_total_shots = min(n_total_shots, config.max_shots)
        source_location = source_location[:n_total_shots]
        receiver_location = receiver_location[:n_total_shots]

    print(f"Using device: {device}")
    print(
        "Sub-block [y, x, z]: "
        f"start=({model_meta['y_start_idx']}, {model_meta['x_start_idx']}, {model_meta['z_start_idx']}), "
        f"size=({model_meta['ny_crop']}, {model_meta['nx_crop']}, {model_meta['nz_crop']}), "
        f"stride={model_meta['decimation_stride']}"
    )
    print(
        "Model grid [z, y, x]: "
        f"{epsilon_np.shape[0]} x {epsilon_np.shape[1]} x {epsilon_np.shape[2]}"
    )
    print(
        f"Survey shots: {n_total_shots} "
        f"(configured total {int(survey_meta['n_shots'])})"
    )
    print(
        f"nt={nt}, dt={config.dt_ns:.3f} ns, freq={config.freq / 1e6:.1f} MHz, "
        f"batch_size={config.batch_size}"
    )
    print(
        f"Estimated 3D Courant number: {courant:.3f} "
        f"(recommended dt <= {recommended_dt * 1e9:.3f} ns for this model)"
    )
    if courant >= 0.9:
        print(
            "Warning: dt is too close to the 3D stability limit; traces may diverge. "
            "Use a smaller --dt-ns value such as 0.040.",
            file=sys.stderr,
        )
    if config.use_subcuboids:
        print("Execution plan: overlapping y-subcuboids on the cropped sub-block")
    if config.dry_run:
        if config.use_subcuboids:
            for cuboid in subcuboid_plan:
                print(
                    "  "
                    f"cuboid {cuboid['cuboid_id']}: model y={cuboid['model_y0_m']:.1f}-"
                    f"{cuboid['model_y1_m']:.1f} m, lines y={cuboid['line_y0_m']:.1f}-"
                    f"{cuboid['line_y1_m']:.1f} m, n_lines="
                    f"{len(cuboid['assigned_line_indices'])}"
                )
        return

    epsilon = torch.from_numpy(epsilon_np).to(device=device)
    sigma = torch.from_numpy(sigma_np).to(device=device)
    mu = torch.from_numpy(mu_np).to(device=device)

    traces = np.zeros((n_total_shots, nt), dtype=np.float32)
    t0 = time.perf_counter()
    print(_render_progress(0, n_total_shots), end="\r", flush=True)
    with torch.no_grad():
        if config.use_subcuboids:
            traces_per_line = int(survey_meta["traces_per_line"])
            shots_done = 0
            for cuboid in subcuboid_plan:
                line_indices = np.asarray(cuboid["assigned_line_indices"], dtype=np.int64)
                if line_indices.size == 0:
                    continue
                shot_indices = np.concatenate(
                    [
                        np.arange(
                            line_idx * traces_per_line,
                            (line_idx + 1) * traces_per_line,
                            dtype=np.int64,
                        )
                        for line_idx in line_indices
                    ]
                )
                if config.max_shots is not None:
                    shot_indices = shot_indices[shot_indices < n_total_shots]
                if shot_indices.size == 0:
                    continue
                y0_full = int(cuboid["model_y0_idx_full"])
                y1_full = int(cuboid["model_y1_idx_full"])
                sub_epsilon, sub_sigma, sub_mu = _slice_subcuboid_model(
                    epsilon,
                    sigma,
                    mu,
                    y0_full=y0_full,
                    y1_full=y1_full,
                    pml_cells=int(model_meta["pml_cells"]),
                )
                print(
                    f"cuboid {cuboid['cuboid_id']}: "
                    f"model y={cuboid['model_y0_m']:.1f}-{cuboid['model_y1_m']:.1f} m, "
                    f"shots={shot_indices.size}"
                )
                for shot0 in range(0, shot_indices.size, config.batch_size):
                    shot1 = min(shot0 + config.batch_size, shot_indices.size)
                    batch_global_idx = shot_indices[shot0:shot1]
                    batch_src = _remap_locations_to_subcuboid(
                        source_location[batch_global_idx].to(device=device),
                        y0_full=y0_full,
                        pml_cells=int(model_meta["pml_cells"]),
                    )
                    batch_rec = _remap_locations_to_subcuboid(
                        receiver_location[batch_global_idx].to(device=device),
                        y0_full=y0_full,
                        pml_cells=int(model_meta["pml_cells"]),
                    )
                    batch_wavelet = wavelet.expand(shot1 - shot0, -1, -1).contiguous()
                    out = tide.maxwell3d(
                        epsilon=sub_epsilon,
                        sigma=sub_sigma,
                        mu=sub_mu,
                        grid_spacing=[model_meta["dz"], model_meta["dy"], model_meta["dx"]],
                        dt=dt,
                        source_amplitude=batch_wavelet,
                        source_location=batch_src,
                        receiver_location=batch_rec,
                        pml_width=int(model_meta["pml_cells"]),
                        stencil=config.stencil,
                        source_component=config.source_component,
                        receiver_component=config.receiver_component,
                        python_backend=config.python_backend,
                    )
                    batch_traces = out[-1][:, :, 0].detach().cpu().numpy().T
                    traces[batch_global_idx] = batch_traces
                    shots_done += shot1 - shot0
                    elapsed = time.perf_counter() - t0
                    print(
                        f"{_render_progress(shots_done, n_total_shots)} elapsed={elapsed:.1f}s",
                        end="\r" if shots_done < n_total_shots else "\n",
                        flush=True,
                    )
        else:
            for shot0 in range(0, n_total_shots, config.batch_size):
                shot1 = min(shot0 + config.batch_size, n_total_shots)
                batch_src = source_location[shot0:shot1].to(device=device)
                batch_rec = receiver_location[shot0:shot1].to(device=device)
                batch_wavelet = wavelet.expand(shot1 - shot0, -1, -1).contiguous()
                out = tide.maxwell3d(
                    epsilon=epsilon,
                    sigma=sigma,
                    mu=mu,
                    grid_spacing=[model_meta["dz"], model_meta["dy"], model_meta["dx"]],
                    dt=dt,
                    source_amplitude=batch_wavelet,
                    source_location=batch_src,
                    receiver_location=batch_rec,
                    pml_width=int(model_meta["pml_cells"]),
                    stencil=config.stencil,
                    source_component=config.source_component,
                    receiver_component=config.receiver_component,
                    python_backend=config.python_backend,
                )
                batch_traces = out[-1][:, :, 0].detach().cpu().numpy().T
                traces[shot0:shot1] = batch_traces
                elapsed = time.perf_counter() - t0
                print(
                    f"{_render_progress(shot1, n_total_shots)} elapsed={elapsed:.1f}s",
                    end="\r" if shot1 < n_total_shots else "\n",
                    flush=True,
                )

    elapsed = time.perf_counter() - t0
    with h5py.File(output_path, "w") as f:
        if n_total_shots == int(survey_meta["n_shots"]):
            n_lines = int(survey_meta["n_lines"])
            traces_per_line = int(survey_meta["traces_per_line"])
            dset = f.create_dataset(
                "traces",
                data=traces.reshape(n_lines, traces_per_line, nt),
                compression="gzip",
            )
            dset.attrs["layout"] = "line, trace, time"
        else:
            dset = f.create_dataset("traces", data=traces, compression="gzip")
            dset.attrs["layout"] = "shot, time"
        dset.attrs["receiver_component"] = config.receiver_component
        dset.attrs["source_component"] = config.source_component
        f.create_dataset("time_s", data=np.arange(nt, dtype=np.float32) * np.float32(dt))
        f.create_dataset("line_y_m", data=np.asarray(survey_meta["line_y_m"], dtype=np.float32))
        f.create_dataset("src_x_m", data=np.asarray(survey_meta["src_x_m"], dtype=np.float32))
        f.create_dataset("rec_x_m", data=np.asarray(survey_meta["rec_x_m"], dtype=np.float32))
        f.create_dataset(
            "source_location_zyx",
            data=source_location.cpu().numpy().reshape(n_total_shots, 3),
        )
        f.create_dataset(
            "receiver_location_zyx",
            data=receiver_location.cpu().numpy().reshape(n_total_shots, 3),
        )
        for key, value in model_meta.items():
            f.attrs[key] = value
        f.attrs["freq_hz"] = config.freq
        f.attrs["dt_s"] = dt
        f.attrs["nt"] = nt
        f.attrs["time_window_ns"] = config.time_window_ns
        f.attrs["batch_size"] = config.batch_size
        f.attrs["offset_m"] = config.offset_m
        f.attrs["inline_spacing_m"] = config.inline_spacing_m
        f.attrs["crossline_spacing_m"] = config.crossline_spacing_m
        f.attrs["simulation_seconds"] = elapsed
        f.attrs["used_python_backend"] = bool(config.python_backend)
        f.attrs["used_subcuboids"] = bool(config.use_subcuboids)
        f.attrs["n_shots_simulated"] = n_total_shots
        f.attrs["n_shots_configured"] = int(survey_meta["n_shots"])

    figure_paths = _build_figure_paths(output_path)
    _save_survey_geometry_xy(figure_paths["geometry_xy"], survey_meta=survey_meta)
    _save_model_slices(
        figure_paths["model_slices"],
        epsilon=epsilon_np,
        model_meta=model_meta,
        survey_meta=survey_meta,
    )
    _save_trace_gather(
        figure_paths["trace_gather"],
        traces=traces,
        nt=nt,
        dt=dt,
        survey_meta=survey_meta,
        n_total_shots=n_total_shots,
    )

    print(f"Saved traces to: {output_path}")
    print(f"Saved survey geometry to: {figure_paths['geometry_xy']}")
    print(f"Saved model slices to: {figure_paths['model_slices']}")
    print(f"Saved trace gather to: {figure_paths['trace_gather']}")
    print(f"Elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

