"""Multiscale filtered LSRTM for the 2D common-offset GPR setup.

This mirrors the stage schedule in ``2d/example_multiscale_filtered_common.py``
but uses the linearized Born workflow:

1. Generate full-wave data on the true model and background model.
2. Form the linearized residual ``d_lin = d_obs - d_bg`` once.
3. For each low-pass stage, solve the damped LSRTM normal equations for
   ``P * J * Mr`` with conjugate gradients, following the JUDI pattern
   ``Jp = Ml * J * Mr`` / ``lsqr!``.

The stage operator is:
    A_stage(x) = M_data * F_stage * J * M_model * x

where ``F_stage`` is a symmetric FIR low-pass filter, ``M_data`` is a direct-
arrival mute/taper in data space, and ``M_model`` is a top/boundary taper in
model space.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import tide

try:
    from tide import backend_utils
except ImportError:  # pragma: no cover
    backend_utils = None

C0 = 299_792_458.0


@dataclass(slots=True)
class LsrtmCase:
    epsilon_true: torch.Tensor
    epsilon_background: torch.Tensor
    sigma_background: torch.Tensor
    mu_background: torch.Tensor
    depsilon_true: torch.Tensor
    source_amplitude: torch.Tensor
    source_locations: torch.Tensor
    receiver_locations: torch.Tensor
    active_mask: torch.Tensor
    model_weights: torch.Tensor
    data_weights: torch.Tensor
    dx: float
    dt: float
    nt: int
    pml_width: int
    air_layer: int
    batch_size: int
    python_backend: bool
    storage_mode: str
    output_dir: Path


@dataclass(slots=True)
class StageSpec:
    key: str
    cutoff_hz: float
    desc: str
    cg_iters: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiscale filtered LSRTM for the 2D common-offset GPR setup."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--backend",
        choices=("auto", "python", "native"),
        default="auto",
        help="Execution backend for tide operators.",
    )
    parser.add_argument(
        "--storage-mode",
        choices=("device", "cpu", "disk", "none", "auto"),
        default="auto",
        help="Storage mode forwarded to tide operators when native backend is used.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("examples/OverThrust.npy"))
    parser.add_argument("--dx", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=4e-11)
    parser.add_argument("--nt", type=int, default=1800)
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--air-layer", type=int, default=3)
    parser.add_argument("--nshots", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-source", type=int, default=4)
    parser.add_argument("--first-source", type=int, default=0)
    parser.add_argument("--receiver-offset", type=int, default=1)
    parser.add_argument("--base-freq", type=float, default=600e6)
    parser.add_argument(
        "--filter-cutoffs-mhz",
        type=str,
        default="150,300,400,600",
        help="Comma-separated low-pass cutoffs for the multiscale stages.",
    )
    parser.add_argument(
        "--stage-iters",
        type=str,
        default="12,12,12,12",
        help="Comma-separated CG iteration counts for the corresponding stages.",
    )
    parser.add_argument("--background-sigma", type=float, default=20.0)
    parser.add_argument("--reg-lambda", type=float, default=5e-3)
    parser.add_argument("--model-taper-cells", type=int, default=12)
    parser.add_argument("--data-mute-margin-cycles", type=float, default=1.5)
    parser.add_argument("--data-mute-taper-samples", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/multiscale_filtered_lsrtm_common_...",
    )
    parser.add_argument(
        "--check-adjoint",
        action="store_true",
        help="Run a dot-product adjoint test for the first stage operator.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_python_backend(backend: str) -> bool:
    if backend == "python":
        return True
    if backend == "native":
        if backend_utils is None or not backend_utils.is_backend_available():
            raise SystemExit("Native tide backend requested but not available.")
        return False
    return False


def parse_csv_int_list(value: str) -> list[int]:
    out = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("Expected at least one integer in a comma-separated list.")
    return out


def build_stage_specs(base_freq: float, cutoffs_mhz: str, stage_iters: str) -> list[StageSpec]:
    cutoff_values = parse_csv_int_list(cutoffs_mhz)
    iter_values = parse_csv_int_list(stage_iters)
    if len(cutoff_values) != len(iter_values):
        raise ValueError("filter_cutoffs_mhz and stage_iters must have the same length.")

    specs: list[StageSpec] = []
    for cutoff_mhz, niter in zip(cutoff_values, iter_values, strict=True):
        if cutoff_mhz <= 0:
            raise ValueError("All stage cutoffs must be positive.")
        if niter < 0:
            raise ValueError("All stage iteration counts must be >= 0.")
        specs.append(
            StageSpec(
                key=f"lp{cutoff_mhz}",
                cutoff_hz=float(cutoff_mhz) * 1e6,
                desc=(
                    f"{base_freq / 1e6:.0f} MHz linearized data "
                    f"low-pass to {cutoff_mhz} MHz"
                ),
                cg_iters=niter,
            )
        )
    return specs


def cosine_ramp(length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.empty((0,), device=device, dtype=dtype)
    grid = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(torch.pi * grid)


def make_active_mask(
    nz: int,
    nx: int,
    air_layer: int,
    pml_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.zeros((nz, nx), device=device, dtype=dtype)
    z0 = max(air_layer + 1, pml_width)
    z1 = nz - pml_width
    x0 = pml_width
    x1 = nx - pml_width
    if z0 < z1 and x0 < x1:
        mask[z0:z1, x0:x1] = 1.0
    return mask.contiguous()


def make_model_weights(
    active_mask: torch.Tensor,
    *,
    air_layer: int,
    pml_width: int,
    taper_cells: int,
) -> torch.Tensor:
    nz, nx = active_mask.shape
    weights = active_mask.clone()
    if taper_cells <= 0:
        return weights

    z0 = max(air_layer + 1, pml_width)
    z1 = nz - pml_width
    x0 = pml_width
    x1 = nx - pml_width
    ramp = cosine_ramp(taper_cells, device=weights.device, dtype=weights.dtype)

    for i in range(min(taper_cells, max(z1 - z0, 0))):
        weights[z0 + i, x0:x1] *= ramp[i]
    for i in range(min(taper_cells, max(x1 - x0, 0))):
        weights[z0:z1, x0 + i] *= ramp[i]
        weights[z0:z1, x1 - 1 - i] *= ramp[i]
    return weights.contiguous()


def make_data_weights(
    nt: int,
    dt: float,
    dx: float,
    freq: float,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    *,
    mute_margin_cycles: float,
    taper_samples: int,
) -> torch.Tensor:
    if taper_samples < 0:
        raise ValueError("data_mute_taper_samples must be >= 0.")

    nshots, nreceivers = receiver_locations.shape[:2]
    weights = torch.ones(
        (nt, nshots, nreceivers),
        device=source_locations.device,
        dtype=torch.float32,
    )
    src_x = source_locations[:, 0, 1].to(dtype=weights.dtype)
    rec_x = receiver_locations[:, :, 1].to(dtype=weights.dtype)
    offsets = (src_x[:, None] - rec_x).abs() * dx
    direct_time = offsets / C0 + (1.0 + mute_margin_cycles) / freq
    start_samples = torch.round(direct_time / dt).to(torch.long)

    taper = cosine_ramp(max(taper_samples, 1), device=weights.device, dtype=weights.dtype)
    for shot in range(nshots):
        for rec in range(nreceivers):
            start = int(start_samples[shot, rec].item())
            if start >= nt:
                continue
            weights[:start, shot, rec] = 0.0
            if taper_samples > 0:
                end = min(start + taper_samples, nt)
                width = end - start
                weights[start:end, shot, rec] = taper[:width]
    return weights.contiguous()


def design_fir_filter(cutoff_hz: float, fs: float, numtaps: int) -> torch.Tensor:
    n = torch.arange(numtaps, dtype=torch.float32)
    window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (numtaps - 1))
    shifted = n - (numtaps - 1) / 2
    sinc = torch.sin(2 * torch.pi * (cutoff_hz / fs) * shifted) / (torch.pi * shifted)
    sinc[(numtaps - 1) // 2] = 2 * cutoff_hz / fs
    h = window * sinc
    return h / h.sum()


def apply_stage_filter(data: torch.Tensor, *, dt: float, cutoff_hz: float) -> torch.Tensor:
    if cutoff_hz <= 0.0:
        return data

    fs = 1.0 / dt
    numtaps = max(3, int(fs / cutoff_hz))
    if numtaps % 2 == 0:
        numtaps += 1
    fir_coeff = design_fir_filter(cutoff_hz, fs, numtaps).to(
        device=data.device, dtype=data.dtype
    )
    half = numtaps // 2

    if data.ndim == 1:
        data_2d = data.view(1, 1, -1)
        padded = F.pad(data_2d, (half, half), mode="constant", value=0.0)
        filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
        return filtered.view(-1)

    if data.ndim == 3:
        nt_local, nshots_local, nreceivers_local = data.shape
        reshaped = data.permute(1, 2, 0).reshape(-1, 1, nt_local)
        padded = F.pad(reshaped, (half, half), mode="constant", value=0.0)
        filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
        return filtered.view(nshots_local, nreceivers_local, nt_local).permute(2, 0, 1)

    raise ValueError(f"Unsupported data dimension {data.ndim}, expected 1D or 3D.")


def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (image * mask).contiguous()


def apply_model_weights(case: LsrtmCase, model: torch.Tensor) -> torch.Tensor:
    return (case.model_weights * model).contiguous()


def apply_data_weights(
    case: LsrtmCase,
    data: torch.Tensor,
    shot_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if shot_indices is None:
        weights = case.data_weights
    else:
        weights = case.data_weights[:, shot_indices, :]
    return (weights.to(device=data.device, dtype=data.dtype) * data).contiguous()


def make_shot_batches(case: LsrtmCase) -> list[torch.Tensor]:
    perm = torch.arange(case.source_locations.shape[0], device=case.epsilon_background.device)
    return [perm[i : i + case.batch_size] for i in range(0, perm.numel(), case.batch_size)]


def build_case(
    args: argparse.Namespace,
    *,
    device: torch.device,
    python_backend: bool,
    output_dir: Path,
) -> LsrtmCase:
    if args.air_layer < 1:
        raise ValueError("air_layer must be >= 1 so the source can be placed at air_layer - 1.")
    epsilon_true_raw = np.load(args.model_path).astype(np.float32, copy=False)
    if epsilon_true_raw.ndim != 2:
        raise ValueError(f"Expected a 2D model, got shape={epsilon_true_raw.shape}.")

    nz, nx = epsilon_true_raw.shape
    epsilon_true_np = epsilon_true_raw.copy()
    epsilon_true_np[: args.air_layer, :] = 1.0
    epsilon_background_np = gaussian_filter(epsilon_true_raw, sigma=args.background_sigma)
    epsilon_background_np = epsilon_background_np.astype(np.float32, copy=False)
    epsilon_background_np[: args.air_layer, :] = 1.0

    sigma_background_np = np.full_like(epsilon_true_np, 1e-3, dtype=np.float32)
    sigma_background_np[: args.air_layer, :] = 0.0

    epsilon_true = torch.as_tensor(epsilon_true_np, device=device)
    epsilon_background = torch.as_tensor(epsilon_background_np, device=device)
    sigma_background = torch.as_tensor(sigma_background_np, device=device)
    mu_background = torch.ones_like(epsilon_background)

    active_mask = make_active_mask(
        nz,
        nx,
        args.air_layer,
        args.pml_width,
        device=device,
        dtype=epsilon_background.dtype,
    )
    model_weights = make_model_weights(
        active_mask,
        air_layer=args.air_layer,
        pml_width=args.pml_width,
        taper_cells=args.model_taper_cells,
    )
    depsilon_true = apply_mask(epsilon_true - epsilon_background, active_mask)

    source_depth = args.air_layer - 1
    source_x = torch.arange(args.nshots, device=device, dtype=torch.long) * args.d_source + args.first_source
    receiver_x = source_x + args.receiver_offset
    if source_x.numel() == 0:
        raise ValueError("nshots must be positive.")
    if int(source_x.min()) < 0 or int(receiver_x.min()) < 0:
        raise ValueError("Source/receiver x locations must be non-negative.")
    if int(source_x.max()) >= nx or int(receiver_x.max()) >= nx:
        raise ValueError(
            f"Geometry exceeds model width nx={nx}; max source={int(source_x.max())}, "
            f"max receiver={int(receiver_x.max())}."
        )

    source_locations = torch.zeros((args.nshots, 1, 2), device=device, dtype=torch.long)
    source_locations[:, 0, 0] = source_depth
    source_locations[:, 0, 1] = source_x

    receiver_locations = torch.zeros((args.nshots, 1, 2), device=device, dtype=torch.long)
    receiver_locations[:, 0, 0] = source_depth
    receiver_locations[:, 0, 1] = receiver_x

    wavelet = tide.ricker(
        args.base_freq,
        args.nt,
        args.dt,
        peak_time=1.0 / args.base_freq,
        device=device,
        dtype=epsilon_background.dtype,
    )
    source_amplitude = wavelet.view(1, 1, args.nt).repeat(args.nshots, 1, 1).contiguous()

    data_weights = make_data_weights(
        args.nt,
        args.dt,
        args.dx,
        args.base_freq,
        source_locations,
        receiver_locations,
        mute_margin_cycles=args.data_mute_margin_cycles,
        taper_samples=args.data_mute_taper_samples,
    ).to(device=device, dtype=epsilon_background.dtype)

    return LsrtmCase(
        epsilon_true=epsilon_true,
        epsilon_background=epsilon_background,
        sigma_background=sigma_background,
        mu_background=mu_background,
        depsilon_true=depsilon_true,
        source_amplitude=source_amplitude,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        active_mask=active_mask,
        model_weights=model_weights,
        data_weights=data_weights,
        dx=args.dx,
        dt=args.dt,
        nt=args.nt,
        pml_width=args.pml_width,
        air_layer=args.air_layer,
        batch_size=args.batch_size,
        python_backend=python_backend,
        storage_mode=args.storage_mode,
        output_dir=output_dir,
    )


pde_counts: dict[str, float] = {}


def reset_pde_counts() -> None:
    pde_counts.clear()
    pde_counts.update(full_forward=0.0, born_forward=0.0, born_adjoint=0.0)


def add_pde_counts(case: LsrtmCase, *, key: str, shot_count: int) -> None:
    if shot_count <= 0:
        return
    pde_counts[key] += float(shot_count) / float(case.source_locations.shape[0])


def format_pde_counts() -> str:
    total = pde_counts["full_forward"] + pde_counts["born_forward"] + pde_counts["born_adjoint"]
    return (
        f"full_forward {pde_counts['full_forward']:.2f}, "
        f"born_forward {pde_counts['born_forward']:.2f}, "
        f"born_adjoint {pde_counts['born_adjoint']:.2f}, "
        f"total {total:.2f}"
    )


def report_pde_totals(prefix: str) -> None:
    print(f"{prefix}PDE solves (all shots = 1): {format_pde_counts()}")


def full_forward_batch(case: LsrtmCase, epsilon: torch.Tensor, shot_indices: torch.Tensor) -> torch.Tensor:
    src_amp = case.source_amplitude[shot_indices]
    src_loc = case.source_locations[shot_indices]
    rec_loc = case.receiver_locations[shot_indices]
    out = tide.maxwelltm(
        epsilon,
        case.sigma_background,
        case.mu_background,
        grid_spacing=case.dx,
        dt=case.dt,
        source_amplitude=src_amp,
        source_location=src_loc,
        receiver_location=rec_loc,
        pml_width=case.pml_width,
        python_backend=case.python_backend,
        storage_mode=case.storage_mode,
    )[-1]
    add_pde_counts(case, key="full_forward", shot_count=int(shot_indices.numel()))
    return out


def born_forward_batch(case: LsrtmCase, depsilon: torch.Tensor, shot_indices: torch.Tensor) -> torch.Tensor:
    src_amp = case.source_amplitude[shot_indices]
    src_loc = case.source_locations[shot_indices]
    rec_loc = case.receiver_locations[shot_indices]
    out = tide.borntm(
        case.epsilon_background,
        case.sigma_background,
        case.mu_background,
        grid_spacing=case.dx,
        dt=case.dt,
        source_amplitude=src_amp,
        source_location=src_loc,
        receiver_location=rec_loc,
        depsilon=apply_mask(depsilon, case.active_mask),
        pml_width=case.pml_width,
        python_backend=case.python_backend,
        storage_mode=case.storage_mode,
    )[-1]
    add_pde_counts(case, key="born_forward", shot_count=int(shot_indices.numel()))
    return out


def born_adjoint_batch(case: LsrtmCase, residual: torch.Tensor, shot_indices: torch.Tensor) -> torch.Tensor:
    src_amp = case.source_amplitude[shot_indices]
    src_loc = case.source_locations[shot_indices]
    rec_loc = case.receiver_locations[shot_indices]
    depsilon = torch.zeros_like(case.epsilon_background, requires_grad=True)
    predicted = tide.borntm(
        case.epsilon_background,
        case.sigma_background,
        case.mu_background,
        grid_spacing=case.dx,
        dt=case.dt,
        source_amplitude=src_amp,
        source_location=src_loc,
        receiver_location=rec_loc,
        depsilon=apply_mask(depsilon, case.active_mask),
        pml_width=case.pml_width,
        python_backend=case.python_backend,
        storage_mode=case.storage_mode,
    )[-1]
    objective = torch.sum(predicted * residual.detach())
    (image,) = torch.autograd.grad(objective, depsilon)
    add_pde_counts(case, key="born_adjoint", shot_count=int(shot_indices.numel()))
    return apply_mask(image, case.active_mask)


def stage_forward(case: LsrtmCase, model: torch.Tensor, stage: StageSpec) -> torch.Tensor:
    physical_model = apply_model_weights(case, model)
    batches = []
    for shot_indices in make_shot_batches(case):
        pred = born_forward_batch(case, physical_model, shot_indices)
        pred = apply_stage_filter(pred, dt=case.dt, cutoff_hz=stage.cutoff_hz)
        pred = apply_data_weights(case, pred, shot_indices)
        batches.append(pred)
    return torch.cat(batches, dim=1).contiguous()


def stage_adjoint(case: LsrtmCase, residual: torch.Tensor, stage: StageSpec) -> torch.Tensor:
    image = torch.zeros_like(case.epsilon_background)
    for shot_indices in make_shot_batches(case):
        batch_residual = residual[:, shot_indices, :]
        batch_residual = apply_data_weights(case, batch_residual, shot_indices)
        batch_residual = apply_stage_filter(batch_residual, dt=case.dt, cutoff_hz=stage.cutoff_hz)
        image = image + born_adjoint_batch(case, batch_residual, shot_indices)
    return apply_mask(apply_model_weights(case, image), case.active_mask)


def normal_operator(case: LsrtmCase, model: torch.Tensor, stage: StageSpec, reg_lambda: float) -> torch.Tensor:
    normal = stage_adjoint(case, stage_forward(case, model, stage), stage)
    return apply_mask(normal + reg_lambda * apply_mask(model, case.active_mask), case.active_mask)


def dot_product(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.sum(lhs * rhs)


def summarize_image(
    label: str,
    case: LsrtmCase,
    image: torch.Tensor,
    observed_stage_data: torch.Tensor,
    stage: StageSpec,
    reg_lambda: float,
) -> tuple[float, float, float]:
    predicted = stage_forward(case, image, stage)
    residual = predicted - observed_stage_data
    data_loss = 0.5 * residual.square().mean().item()
    reg_loss = 0.5 * reg_lambda * apply_mask(image, case.active_mask).square().mean().item()
    total_loss = data_loss + reg_loss
    data_rmse = residual.square().mean().sqrt().item()
    physical_image = apply_model_weights(case, image)
    model_rmse = (physical_image - case.depsilon_true).square().mean().sqrt().item()
    print(
        f"{label}: total_loss={total_loss:.6e}, data_rmse={data_rmse:.6e}, "
        f"model_rmse={model_rmse:.6e}"
    )
    return total_loss, data_rmse, model_rmse


def save_filter_comparison(base_data: torch.Tensor, observed_sets: dict[str, torch.Tensor], output_dir: Path) -> None:
    base_np = base_data.detach().cpu().numpy()[:, :, 0]
    vmax = float(np.max(np.abs(base_np)))
    for arr in observed_sets.values():
        vmax = max(vmax, float(np.max(np.abs(arr.detach().cpu().numpy()[:, :, 0]))))
    vmax = max(vmax, 1e-8)

    ncols = 1 + len(observed_sets)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(base_np, aspect="auto", cmap="gray", vmin=-vmax, vmax=vmax)
    axes[0].set_title("Weighted linearized data")
    axes[0].set_xlabel("Shot")
    axes[0].set_ylabel("Time sample")

    for idx, (key, data) in enumerate(observed_sets.items(), start=1):
        axes[idx].imshow(
            data.detach().cpu().numpy()[:, :, 0],
            aspect="auto",
            cmap="gray",
            vmin=-vmax,
            vmax=vmax,
        )
        axes[idx].set_title(key)
        axes[idx].set_xlabel("Shot")

    fig.tight_layout()
    fig.savefig(output_dir / "linearized_data_filter_comparison.png", dpi=180)
    plt.close(fig)


def save_model_snapshot(image: torch.Tensor, filename: Path, title: str, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(image.detach().cpu().numpy(), aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def save_summary(
    case: LsrtmCase,
    rtm_image: torch.Tensor,
    final_image: torch.Tensor,
    all_total_losses: list[float],
    all_model_rmses: list[float],
    stage_breaks: list[int],
    stage_labels: list[str],
) -> None:
    true_eps_np = case.epsilon_true.detach().cpu().numpy()
    background_np = case.epsilon_background.detach().cpu().numpy()
    true_deps_np = case.depsilon_true.detach().cpu().numpy()
    rtm_np = apply_model_weights(case, rtm_image).detach().cpu().numpy()
    final_np = apply_model_weights(case, final_image).detach().cpu().numpy()

    vmax_deps = float(np.max(np.abs(true_deps_np)))
    vmax_img = max(vmax_deps, float(np.max(np.abs(rtm_np))), float(np.max(np.abs(final_np))), 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    im = axes[0, 0].imshow(true_eps_np, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("True epsilon")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

    im = axes[0, 1].imshow(background_np, aspect="auto", cmap="viridis")
    axes[0, 1].set_title("Background epsilon")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)

    im = axes[0, 2].imshow(true_deps_np, aspect="auto", cmap="seismic", vmin=-vmax_img, vmax=vmax_img)
    axes[0, 2].set_title("True depsilon")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8)

    im = axes[1, 0].imshow(rtm_np, aspect="auto", cmap="seismic", vmin=-vmax_img, vmax=vmax_img)
    axes[1, 0].set_title("RTM image")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    im = axes[1, 1].imshow(final_np, aspect="auto", cmap="seismic", vmin=-vmax_img, vmax=vmax_img)
    axes[1, 1].set_title("Final multiscale LSRTM")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

    ax = axes[1, 2]
    ax.semilogy(all_total_losses, label="Total loss")
    ax.plot(all_model_rmses, label="Model RMSE")
    for idx, label in zip(stage_breaks, stage_labels, strict=True):
        ax.axvline(idx, color="r", linestyle="--", alpha=0.35)
        ax.text(idx, ax.get_ylim()[0], label, rotation=90, va="bottom", ha="right", fontsize=8)
    ax.set_title("Stage convergence")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(case.output_dir / "multiscale_filtered_lsrtm_summary.png", dpi=180)
    plt.close(fig)

    np.savez(
        case.output_dir / "multiscale_filtered_lsrtm_bundle.npz",
        epsilon_true=true_eps_np,
        epsilon_background=background_np,
        depsilon_true=true_deps_np,
        data_weights=case.data_weights.detach().cpu().numpy(),
        model_weights=case.model_weights.detach().cpu().numpy(),
        rtm_image=rtm_np,
        lsrtm_image=final_np,
        total_loss=np.asarray(all_total_losses, dtype=np.float64),
        model_rmse=np.asarray(all_model_rmses, dtype=np.float64),
        stage_breaks=np.asarray(stage_breaks, dtype=np.int64),
        stage_labels=np.asarray(stage_labels, dtype=object),
    )


def generate_linearized_data_sets(case: LsrtmCase, stage_specs: list[StageSpec]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observed_batches = []
    background_batches = []
    for shot_indices in make_shot_batches(case):
        observed_batches.append(full_forward_batch(case, case.epsilon_true, shot_indices))
        background_batches.append(full_forward_batch(case, case.epsilon_background, shot_indices))

    observed_full = torch.cat(observed_batches, dim=1)
    background_full = torch.cat(background_batches, dim=1)
    linearized_raw = (observed_full - background_full).contiguous()
    base_data = apply_data_weights(case, linearized_raw)

    observed_sets: dict[str, torch.Tensor] = {}
    for spec in stage_specs:
        observed_sets[spec.key] = apply_data_weights(
            case,
            apply_stage_filter(linearized_raw, dt=case.dt, cutoff_hz=spec.cutoff_hz),
        ).contiguous()

    return base_data, observed_sets


def check_stage_adjointness(case: LsrtmCase, stage: StageSpec) -> float:
    model = apply_mask(torch.randn_like(case.epsilon_background), case.active_mask)
    data = torch.randn(
        (case.nt, case.source_locations.shape[0], case.receiver_locations.shape[1]),
        device=case.epsilon_background.device,
        dtype=case.epsilon_background.dtype,
    )
    lhs = dot_product(stage_forward(case, model, stage), data)
    rhs = dot_product(model, stage_adjoint(case, data, stage))
    denom = max(abs(lhs.item()), abs(rhs.item()), 1e-12)
    return abs(lhs.item() - rhs.item()) / denom


def run_stage_cg(
    case: LsrtmCase,
    stage: StageSpec,
    observed_stage_data: torch.Tensor,
    *,
    reg_lambda: float,
    image_init: torch.Tensor,
) -> tuple[torch.Tensor, list[float], list[float]]:
    image = apply_mask(image_init.clone(), case.active_mask)
    rhs = stage_adjoint(case, observed_stage_data, stage)
    residual = apply_mask(rhs - normal_operator(case, image, stage, reg_lambda), case.active_mask)
    direction = residual.clone()
    rr = dot_product(residual, residual)

    total_losses: list[float] = []
    model_rmses: list[float] = []

    if stage.cg_iters == 0:
        return image, total_losses, model_rmses

    for iteration in range(1, stage.cg_iters + 1):
        normal_direction = normal_operator(case, direction, stage, reg_lambda)
        denom = dot_product(direction, normal_direction).item()
        if denom <= 0.0:
            raise RuntimeError(
                f"Normal operator lost positive definiteness in stage {stage.key} at iter {iteration}."
            )
        alpha = rr.item() / denom
        image = apply_mask(image + alpha * direction, case.active_mask)
        residual = apply_mask(residual - alpha * normal_direction, case.active_mask)

        total_loss, data_rmse, model_rmse = summarize_image(
            f"{stage.key} iter {iteration:02d}",
            case,
            image,
            observed_stage_data,
            stage,
            reg_lambda,
        )
        total_losses.append(total_loss)
        model_rmses.append(model_rmse)

        rr_next = dot_product(residual, residual)
        if rr_next.item() <= 1e-20:
            break
        beta = rr_next.item() / rr.item()
        direction = apply_mask(residual + beta * direction, case.active_mask)
        rr = rr_next

    return image, total_losses, model_rmses


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    python_backend = resolve_python_backend(args.backend)
    stage_specs = build_stage_specs(args.base_freq, args.filter_cutoffs_mhz, args.stage_iters)

    lowpass_tag = "-".join(spec.key.replace("lp", "") for spec in stage_specs)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs") / (
            f"multiscale_filtered_lsrtm_common_base{int(args.base_freq / 1e6)}MHz_"
            f"lp{lowpass_tag}_shots{args.nshots}_bs{args.batch_size}_nt{args.nt}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    case = build_case(args, device=device, python_backend=python_backend, output_dir=output_dir)
    reset_pde_counts()

    print(f"Using device: {device}")
    print(
        f"Loaded model shape: {tuple(case.epsilon_true.shape)}, "
        f"permittivity range: {case.epsilon_true.min().item():.2f} - {case.epsilon_true.max().item():.2f}"
    )
    print(f"Base forward frequency: {args.base_freq / 1e6:.0f} MHz")
    print("Stage schedule:")
    for spec in stage_specs:
        print(f"  {spec.key}: {spec.desc}  CG {spec.cg_iters} iters")
    print(f"Saving figures to: {case.output_dir}")

    if args.check_adjoint:
        relerr = check_stage_adjointness(case, stage_specs[0])
        print(f"Adjoint relative dot-product error ({stage_specs[0].key}) = {relerr:.6e}")
        reset_pde_counts()

    print("Generating full-wave linearized data once, then building filtered stage data...")
    time_start = time.time()
    base_data, observed_sets = generate_linearized_data_sets(case, stage_specs)
    report_pde_totals("After data generation: ")
    save_filter_comparison(base_data, observed_sets, case.output_dir)

    full_band_stage = StageSpec(key="rtm_full", cutoff_hz=0.0, desc="Unfiltered RTM", cg_iters=0)
    rtm_image = stage_adjoint(case, base_data, full_band_stage)
    summarize_image("RTM summary", case, rtm_image, base_data, full_band_stage, args.reg_lambda)

    image = torch.zeros_like(case.epsilon_background)
    all_total_losses: list[float] = []
    all_model_rmses: list[float] = []
    stage_breaks: list[int] = []
    stage_labels: list[str] = []
    vmax_stage = float(torch.max(case.depsilon_true.abs()).item())

    for stage in stage_specs:
        print(f"\n==== Stage {stage.key}: {stage.desc} ====")
        forward_start = pde_counts["born_forward"]
        adjoint_start = pde_counts["born_adjoint"]
        image, stage_losses, stage_model_rmses = run_stage_cg(
            case,
            stage,
            observed_sets[stage.key],
            reg_lambda=args.reg_lambda,
            image_init=image,
        )
        if stage_losses:
            all_total_losses.extend(stage_losses)
            all_model_rmses.extend(stage_model_rmses)
            stage_breaks.append(len(all_total_losses) - 1)
            stage_labels.append(stage.key)

        stage_image = apply_model_weights(case, image)
        save_model_snapshot(
            stage_image,
            case.output_dir / f"depsilon_stage_{stage.key}.png",
            f"{stage.desc} result",
            vmax=vmax_stage,
        )
        print(
            f"Stage {stage.key} PDE solves: born_forward {pde_counts['born_forward'] - forward_start:.2f}, "
            f"born_adjoint {pde_counts['born_adjoint'] - adjoint_start:.2f}"
        )

    total_time = time.time() - time_start
    report_pde_totals("Total ")
    summarize_image(
        "Final LSRTM summary",
        case,
        image,
        observed_sets[stage_specs[-1].key],
        stage_specs[-1],
        args.reg_lambda,
    )
    save_summary(case, rtm_image, image, all_total_losses, all_model_rmses, stage_breaks, stage_labels)
    np.save(case.output_dir / "depsilon_inverted.npy", apply_model_weights(case, image).detach().cpu().numpy())

    initial_rmse = case.depsilon_true.square().mean().sqrt().item()
    final_rmse = (apply_model_weights(case, image) - case.depsilon_true).square().mean().sqrt().item()
    print(f"Initial model RMSE (zero image): {initial_rmse:.6e}")
    print(f"Final model RMSE: {final_rmse:.6e}")
    if initial_rmse > 0.0:
        print(f"Improvement: {(1.0 - final_rmse / initial_rmse) * 100.0:.2f}%")
    print(f"Total elapsed time: {total_time:.2f}s")
    print(f"Outputs saved to {case.output_dir}")


if __name__ == "__main__":
    main()
