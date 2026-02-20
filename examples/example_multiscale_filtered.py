import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import tide

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dx = 0.02
dt = 4e-11
nt = 1500
pml_width = 10
air_layer = 3

n_shots = 100
d_source = 4
first_source = 0
# Shots per batch (batch size).
batch_size = 16
model_gradient_sampling_interval = 5


model_path = "examples/data/OverThrust.npy"
epsilon_true_raw = np.load(model_path)
print(f"Loaded model shape: {epsilon_true_raw.shape}")
print(
    f"Permittivity range: {epsilon_true_raw.min():.2f} - {epsilon_true_raw.max():.2f}"
)

ny, nx = epsilon_true_raw.shape
epsilon_true_np = epsilon_true_raw.copy()
epsilon_true_np[:air_layer, :] = 1.0

sigma_true_np = np.ones_like(epsilon_true_np) * 1e-3
sigma_true_np[:air_layer, :] = 0.0

epsilon_true = torch.tensor(epsilon_true_np, dtype=torch.float32, device=device)
sigma_true = torch.tensor(sigma_true_np, dtype=torch.float32, device=device)
mu_true = torch.ones_like(epsilon_true)

source_depth = air_layer - 1
source_x = torch.arange(n_shots, device=device) * d_source + first_source

source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
source_locations[:, 0, 0] = source_depth
source_locations[:, 0, 1] = source_x

receiver_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
receiver_locations[:, 0, 0] = source_depth
receiver_locations[:, 0, 1] = source_x + 1

n_shots_per_batch = batch_size

base_forward_freq = 600e6
filter_specs = {
    "lp250": {"lowpass_mhz": 200, "desc": "600 MHz forward result low-pass to 200 MHz"},
    "lp500": {"lowpass_mhz": 400, "desc": "600 MHz forward result low-pass to 400 MHz"},
    "lp700": {"lowpass_mhz": 600, "desc": "600 MHz forward result low-pass to 600 MHz"},
}
inversion_schedule = [
    {"data_key": "lp250", "adamw_epochs": 40, "lbfgs_epochs": 6},
    {"data_key": "lp500", "adamw_epochs": 30, "lbfgs_epochs": 6},
    {"data_key": "lp700", "adamw_epochs": 10, "lbfgs_epochs": 6},
]

print(f"Base forward frequency: {base_forward_freq / 1e6:.0f} MHz")
print("FIR low-pass schedule on observed data:")
for key, spec in filter_specs.items():
    print(f"  {key}: {spec['desc']} (cutoff {spec['lowpass_mhz']} MHz)")
print("Inversion schedule:")
for item in inversion_schedule:
    print(
        f"  {item['data_key']}: AdamW {item['adamw_epochs']}e  "
        f"LBFGS {item['lbfgs_epochs']}e"
    )

lowpass_tag = "-".join(str(spec["lowpass_mhz"]) for spec in filter_specs.values())
output_dir = Path("outputs") / (
    f"multiscale_fir_base{int(base_forward_freq / 1e6)}MHz_lp{lowpass_tag}_shots{n_shots}_bs{batch_size}_nt{nt}"
)
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving figures to: {output_dir}")


pde_counts = {"forward": 0.0, "adjoint": 0.0}


def add_pde_counts(
    batch_size: int, forward: bool = False, adjoint: bool = False
) -> None:
    if batch_size <= 0:
        return
    frac = batch_size / n_shots
    if forward:
        pde_counts["forward"] += frac
    if adjoint:
        pde_counts["adjoint"] += frac


def format_pde_counts(forward: float, adjoint: float) -> str:
    total = forward + adjoint
    return f"forward {forward:.2f}, adjoint {adjoint:.2f}, total {total:.2f}"


def report_pde_totals(prefix: str) -> None:
    print(
        f"{prefix}PDE solves (100 shots = 1): {format_pde_counts(pde_counts['forward'], pde_counts['adjoint'])}"
    )


def report_pde_delta(prefix: str, forward_start: float, adjoint_start: float) -> None:
    forward = pde_counts["forward"] - forward_start
    adjoint = pde_counts["adjoint"] - adjoint_start
    print(f"{prefix}PDE solves: {format_pde_counts(forward, adjoint)}")


def make_shot_batches() -> list[torch.Tensor]:
    perm = torch.arange(n_shots, device=device)
    return [
        perm[i : i + n_shots_per_batch] for i in range(0, n_shots, n_shots_per_batch)
    ]


@dataclass
class _DistanceGeom:
    dfield: torch.Tensor
    irays: torch.Tensor
    lrays: torch.Tensor
    xrays: torch.Tensor
    points: torch.Tensor
    seg_start: torch.Tensor
    seg_delta: torch.Tensor
    seg_lsq: torch.Tensor
    u_span: torch.Tensor
    t_axis_norm: torch.Tensor
    u_axis_norm: torch.Tensor


@dataclass
class _DistanceGeomBatch:
    dfield: torch.Tensor  # [B, nugrid, ntgrid]
    irays: torch.Tensor  # [B, nump]
    lrays: torch.Tensor  # [B, nump]
    xrays: torch.Tensor  # [B, nump, 2]
    points: torch.Tensor  # [nump, 2]
    seg_start: torch.Tensor  # [B, nt-1, 2]
    seg_delta: torch.Tensor  # [B, nt-1, 2]
    seg_lsq: torch.Tensor  # [B, nt-1]
    u_span: torch.Tensor  # [B]
    t_axis_norm: torch.Tensor  # [ntgrid]
    u_axis_norm: torch.Tensor  # [nugrid]


def _build_distance_geom(
    trace: torch.Tensor,
    dt_local: float,
    nugrid: int,
    ntgrid: int,
    eps: float,
    t0: torch.Tensor | None = None,
    t1: torch.Tensor | None = None,
    u0: torch.Tensor | None = None,
    u1: torch.Tensor | None = None,
    chunk_size: int = 2048,
) -> _DistanceGeom:
    device_local = trace.device
    dtype_local = trace.dtype
    nt_local = int(trace.numel())
    if nt_local < 2:
        raise ValueError("Each trace must have at least 2 samples.")
    if nugrid < 2 or ntgrid < 2:
        raise ValueError("nugrid and ntgrid must both be >= 2.")

    t = torch.arange(nt_local, device=device_local, dtype=dtype_local) * torch.as_tensor(
        dt_local, device=device_local, dtype=dtype_local
    )
    if t0 is None:
        t0 = t[0]
    if t1 is None:
        t1 = t[-1]
    t_span = (t1 - t0).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )

    if u0 is None:
        u0 = 2.0 * torch.min(trace)
    if u1 is None:
        u1 = 2.0 * torch.max(trace)
    u_span = (u1 - u0).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )

    t_norm = (t - t0) / t_span
    u_norm = (trace - u0) / u_span
    pn = torch.stack((t_norm, u_norm), dim=1)
    seg_start = pn[:-1]
    seg_delta = pn[1:] - pn[:-1]
    seg_lsq = (seg_delta * seg_delta).sum(dim=1).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )

    t_axis_norm = torch.linspace(
        t_norm[0], t_norm[-1], ntgrid, device=device_local, dtype=dtype_local
    )
    u_axis_norm = torch.linspace(0.0, 1.0, nugrid, device=device_local, dtype=dtype_local)
    tt = t_axis_norm.unsqueeze(0).expand(nugrid, ntgrid)
    uu = u_axis_norm.unsqueeze(1).expand(nugrid, ntgrid)
    points = torch.stack((tt.reshape(-1), uu.reshape(-1)), dim=1)
    nump = points.shape[0]

    irays = torch.empty(nump, device=device_local, dtype=torch.long)
    lrays = torch.empty(nump, device=device_local, dtype=dtype_local)
    xrays = torch.empty(nump, 2, device=device_local, dtype=dtype_local)
    dvals = torch.empty(nump, device=device_local, dtype=dtype_local)

    for beg in range(0, nump, chunk_size):
        end = min(beg + chunk_size, nump)
        p = points[beg:end]
        b = p[:, None, :] - seg_start[None, :, :]
        lam = (b * seg_delta[None, :, :]).sum(dim=2) / seg_lsq[None, :]
        lam = torch.clamp(lam, 0.0, 1.0)
        ds = b - lam[:, :, None] * seg_delta[None, :, :]
        dsq = (ds * ds).sum(dim=2)

        iclose = torch.argmin(dsq, dim=1)
        row = torch.arange(end - beg, device=device_local)
        l = lam[row, iclose]
        xclose = seg_start[iclose] + l[:, None] * seg_delta[iclose]
        d = torch.sqrt(
            dsq[row, iclose].clamp_min(
                torch.as_tensor(eps, device=device_local, dtype=dtype_local)
            )
        )

        irays[beg:end] = iclose
        lrays[beg:end] = l
        xrays[beg:end] = xclose
        dvals[beg:end] = d

    return _DistanceGeom(
        dfield=dvals.reshape(nugrid, ntgrid),
        irays=irays,
        lrays=lrays,
        xrays=xrays,
        points=points,
        seg_start=seg_start,
        seg_delta=seg_delta,
        seg_lsq=seg_lsq,
        u_span=u_span,
        t_axis_norm=t_axis_norm,
        u_axis_norm=u_axis_norm,
    )


def _build_distance_geom_batch(
    traces: torch.Tensor,
    dt_local: float,
    nugrid: int,
    ntgrid: int,
    eps: float,
    u0: torch.Tensor | None = None,
    u1: torch.Tensor | None = None,
    point_chunk_size: int = 512,
) -> _DistanceGeomBatch:
    """Build distance geometry for a batch of traces.

    traces: [B, nt]
    """
    device_local = traces.device
    dtype_local = traces.dtype
    if traces.ndim != 2:
        raise ValueError("traces must have shape [B, nt].")
    B, nt_local = traces.shape
    if nt_local < 2:
        raise ValueError("Each trace must have at least 2 samples.")
    if nugrid < 2 or ntgrid < 2:
        raise ValueError("nugrid and ntgrid must both be >= 2.")

    t = torch.arange(nt_local, device=device_local, dtype=dtype_local) * torch.as_tensor(
        dt_local, device=device_local, dtype=dtype_local
    )
    t0 = t[0]
    t1 = t[-1]
    t_span = (t1 - t0).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )

    if u0 is None:
        u0 = 2.0 * torch.min(traces, dim=1).values
    if u1 is None:
        u1 = 2.0 * torch.max(traces, dim=1).values
    u_span = (u1 - u0).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )

    t_norm = (t - t0) / t_span  # [nt]
    u_norm = (traces - u0[:, None]) / u_span[:, None]  # [B, nt]
    pn = torch.stack((t_norm.unsqueeze(0).expand(B, -1), u_norm), dim=2)  # [B, nt, 2]
    seg_start = pn[:, :-1, :]  # [B, S, 2]
    seg_delta = pn[:, 1:, :] - pn[:, :-1, :]  # [B, S, 2]
    seg_lsq = (seg_delta * seg_delta).sum(dim=2).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )  # [B, S]

    t_axis_norm = torch.linspace(
        t_norm[0], t_norm[-1], ntgrid, device=device_local, dtype=dtype_local
    )
    u_axis_norm = torch.linspace(0.0, 1.0, nugrid, device=device_local, dtype=dtype_local)
    tt = t_axis_norm.unsqueeze(0).expand(nugrid, ntgrid)
    uu = u_axis_norm.unsqueeze(1).expand(nugrid, ntgrid)
    points = torch.stack((tt.reshape(-1), uu.reshape(-1)), dim=1)  # [P, 2]
    nump = points.shape[0]

    irays = torch.empty((B, nump), device=device_local, dtype=torch.long)
    lrays = torch.empty((B, nump), device=device_local, dtype=dtype_local)
    xrays = torch.empty((B, nump, 2), device=device_local, dtype=dtype_local)
    dvals = torch.empty((B, nump), device=device_local, dtype=dtype_local)

    for p0 in range(0, nump, point_chunk_size):
        p1 = min(p0 + point_chunk_size, nump)
        p = points[p0:p1]  # [Pc,2]

        # [B, Pc, S, 2]
        b = p.unsqueeze(0).unsqueeze(2) - seg_start.unsqueeze(1)
        lam = (b * seg_delta.unsqueeze(1)).sum(dim=3) / seg_lsq.unsqueeze(1)  # [B, Pc, S]
        lam = torch.clamp(lam, 0.0, 1.0)
        ds = b - lam.unsqueeze(3) * seg_delta.unsqueeze(1)  # [B, Pc, S, 2]
        dsq = (ds * ds).sum(dim=3)  # [B, Pc, S]

        iclose = torch.argmin(dsq, dim=2)  # [B, Pc]
        row_b = torch.arange(B, device=device_local)[:, None]  # [B,1]
        row_p = torch.arange(p1 - p0, device=device_local)[None, :]  # [1,Pc]
        l = lam[row_b, row_p, iclose]  # [B, Pc]
        xclose = seg_start[row_b, iclose, :] + l.unsqueeze(2) * seg_delta[row_b, iclose, :]
        d = torch.sqrt(
            dsq[row_b, row_p, iclose].clamp_min(
                torch.as_tensor(eps, device=device_local, dtype=dtype_local)
            )
        )

        irays[:, p0:p1] = iclose
        lrays[:, p0:p1] = l
        xrays[:, p0:p1, :] = xclose
        dvals[:, p0:p1] = d

    return _DistanceGeomBatch(
        dfield=dvals.reshape(B, nugrid, ntgrid),
        irays=irays,
        lrays=lrays,
        xrays=xrays,
        points=points,
        seg_start=seg_start,
        seg_delta=seg_delta,
        seg_lsq=seg_lsq,
        u_span=u_span,
        t_axis_norm=t_axis_norm,
        u_axis_norm=u_axis_norm,
    )


def _wasser_w2_and_grad(
    source_cdf: torch.Tensor,
    target_cdf: torch.Tensor,
    source_x: torch.Tensor,
    target_x: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device_local = source_cdf.device
    dtype_local = source_cdf.dtype
    n = int(source_cdf.numel())
    m = int(target_cdf.numel())
    if n < 2 or m < 2:
        raise ValueError("Marginal CDFs must have at least 2 samples.")

    jitter_s = torch.arange(n, device=device_local, dtype=dtype_local) * (eps / n)
    jitter_t = torch.arange(m, device=device_local, dtype=dtype_local) * (eps / m)
    cf = torch.clamp(source_cdf + jitter_s, max=1.0)
    cg = torch.clamp(target_cdf + jitter_t, max=1.0)

    a = torch.cat((cf[:-1], cg), dim=0)
    tk, tkarg = torch.sort(a)

    indf = torch.searchsorted(cf, tk, right=False).clamp(0, n - 1)
    indg = torch.searchsorted(cg, tk, right=False).clamp(0, m - 1)

    dtk = torch.cat((tk[:1], tk[1:] - tk[:-1]), dim=0)
    xft = source_x[indf]
    xgt = target_x[indg]
    dsqx = (xft - xgt) ** 2
    w2 = torch.sum(dsqx * dtk)

    B = torch.triu(torch.ones((n, m), device=device_local, dtype=dtype_local))
    C = B - cf.view(1, -1)
    D = torch.cat(
        (C[:, :-1], torch.zeros((n, m), device=device_local, dtype=dtype_local)), dim=1
    )
    Difftk = D[:, tkarg]
    Diffdtk = torch.cat((Difftk[:, :1], Difftk[:, 1:] - Difftk[:, :-1]), dim=1)
    dW2 = torch.sum(Diffdtk * dsqx.view(1, -1), dim=1)

    dW2dt = torch.sum(2.0 * (xft - xgt) * dtk)
    return w2, dW2, dW2dt


def _pdf_deriv_marg(
    chain_matrix: torch.Tensor,
    pdf_raw: torch.Tensor,
    geom: _DistanceGeom,
    lambda_d: float,
    eps: float,
    nt_local: int,
) -> torch.Tensor:
    device_local = chain_matrix.device
    dtype_local = chain_matrix.dtype

    dis = geom.dfield.reshape(-1).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )
    p = geom.points
    dddx = (geom.xrays - p) / dis[:, None]

    x0 = geom.seg_start[geom.irays]
    c = geom.seg_delta[geom.irays]
    lsq = geom.seg_lsq[geom.irays]
    lr = geom.lrays

    dx0dy0 = torch.tensor([0.0, 1.0], device=device_local, dtype=dtype_local)
    dx1dy1 = torch.tensor([0.0, 1.0], device=device_local, dtype=dtype_local)

    dlamdy0 = (
        2.0 * c[:, 1] * lr
        + torch.sum((p - dx0dy0) * c - (p - x0) * dx0dy0, dim=1)
    ) / lsq
    clip_mask = (lr == 0.0) | (lr == 1.0)
    dlamdy0[clip_mask] = 0.0
    dxdy0 = dx0dy0 + dlamdy0[:, None] * c - lr[:, None] * dx0dy0

    dlamdy1 = (
        -2.0 * c[:, 1] * lr + torch.sum(p * c + (p - x0) * dx1dy1, dim=1)
    ) / lsq
    dlamdy1[clip_mask] = 0.0
    dxdy1 = dlamdy1[:, None] * c + lr[:, None] * dx1dy1

    dddy0 = torch.sum(dddx * dxdy0, dim=1) / geom.u_span
    dddy1 = torch.sum(dddx * dxdy1, dim=1) / geom.u_span

    pdfrow = pdf_raw.reshape(-1) * chain_matrix.reshape(-1)

    s = torch.zeros(nt_local, device=device_local, dtype=dtype_local)
    s.scatter_add_(0, geom.irays, dddy0 * pdfrow)

    idx_next = geom.irays + 1
    valid = idx_next < nt_local
    s.scatter_add_(0, idx_next[valid], dddy1[valid] * pdfrow[valid])

    return -s / torch.as_tensor(lambda_d, device=device_local, dtype=dtype_local)


def _trace_ot_loss_grad_batch(
    pred_batch: torch.Tensor,  # [B, nt]
    obs_batch: torch.Tensor,  # [B, nt]
    dt_local: float,
    lambda_d: float,
    nugrid: int,
    ntgrid: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch version of OT loss+grad where the heavy distance-grid stage is vectorized."""
    device_local = pred_batch.device
    dtype_local = pred_batch.dtype
    B, nt_local = pred_batch.shape

    grid_u0 = 2.0 * torch.min(pred_batch, dim=1).values
    grid_u1 = 2.0 * torch.max(pred_batch, dim=1).values

    geom_pred_b = _build_distance_geom_batch(
        pred_batch,
        dt_local=dt_local,
        nugrid=nugrid,
        ntgrid=ntgrid,
        eps=eps,
        u0=grid_u0,
        u1=grid_u1,
    )
    geom_obs_b = _build_distance_geom_batch(
        obs_batch,
        dt_local=dt_local,
        nugrid=nugrid,
        ntgrid=ntgrid,
        eps=eps,
        u0=grid_u0,
        u1=grid_u1,
    )

    lam = torch.as_tensor(lambda_d, device=device_local, dtype=dtype_local)
    pdf_obs_raw_b = torch.exp(-torch.abs(geom_obs_b.dfield) / lam)  # [B,U,T]
    pdf_pred_raw_b = torch.exp(-torch.abs(geom_pred_b.dfield) / lam)

    amp_obs_b = pdf_obs_raw_b.sum(dim=(1, 2)).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )
    amp_pred_b = pdf_pred_raw_b.sum(dim=(1, 2)).clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )
    pdf_obs_b = pdf_obs_raw_b / amp_obs_b[:, None, None]
    pdf_pred_b = pdf_pred_raw_b / amp_pred_b[:, None, None]

    f0_obs_b = torch.sum(pdf_obs_b, dim=1)  # [B,T]
    f1_obs_b = torch.sum(pdf_obs_b, dim=2)  # [B,U]
    f0_pred_b = torch.sum(pdf_pred_b, dim=1)
    f1_pred_b = torch.sum(pdf_pred_b, dim=2)

    f0_cdf_obs_b = torch.cumsum(f0_obs_b, dim=1)
    f1_cdf_obs_b = torch.cumsum(f1_obs_b, dim=1)
    f0_cdf_pred_b = torch.cumsum(f0_pred_b, dim=1)
    f1_cdf_pred_b = torch.cumsum(f1_pred_b, dim=1)

    losses = []
    grads = []
    for i in range(B):
        w0, dw0, _ = _wasser_w2_and_grad(
            f0_cdf_pred_b[i],
            f0_cdf_obs_b[i],
            geom_pred_b.t_axis_norm,
            geom_obs_b.t_axis_norm,
            eps,
        )
        w1, dw1, _ = _wasser_w2_and_grad(
            f1_cdf_pred_b[i],
            f1_cdf_obs_b[i],
            geom_pred_b.u_axis_norm,
            geom_obs_b.u_axis_norm,
            eps,
        )

        dwpmargX = dw0.view(1, -1).repeat(nugrid, 1)
        dwpmargY = dw1.view(-1, 1).repeat(1, ntgrid)

        flat_pdf_pred = pdf_pred_b[i].reshape(-1)
        dwpmargX = dwpmargX - torch.sum(dwpmargX.reshape(-1) * flat_pdf_pred)
        dwpmargY = dwpmargY - torch.sum(dwpmargY.reshape(-1) * flat_pdf_pred)
        dwpmargX = dwpmargX / amp_pred_b[i]
        dwpmargY = dwpmargY / amp_pred_b[i]

        geom_i = _DistanceGeom(
            dfield=geom_pred_b.dfield[i],
            irays=geom_pred_b.irays[i],
            lrays=geom_pred_b.lrays[i],
            xrays=geom_pred_b.xrays[i],
            points=geom_pred_b.points,
            seg_start=geom_pred_b.seg_start[i],
            seg_delta=geom_pred_b.seg_delta[i],
            seg_lsq=geom_pred_b.seg_lsq[i],
            u_span=geom_pred_b.u_span[i],
            t_axis_norm=geom_pred_b.t_axis_norm,
            u_axis_norm=geom_pred_b.u_axis_norm,
        )

        grad_t = _pdf_deriv_marg(
            dwpmargX, pdf_pred_raw_b[i], geom_i, lambda_d, eps, nt_local
        )
        grad_u = _pdf_deriv_marg(
            dwpmargY, pdf_pred_raw_b[i], geom_i, lambda_d, eps, nt_local
        )

        losses.append(0.5 * (w0 + w1))
        grads.append(0.5 * (grad_t + grad_u))

    return torch.stack(losses, dim=0), torch.stack(grads, dim=0)


def _trace_ot_loss_grad(
    pred_trace: torch.Tensor,
    obs_trace: torch.Tensor,
    dt_local: float,
    lambda_d: float,
    nugrid: int,
    ntgrid: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    device_local = pred_trace.device
    dtype_local = pred_trace.dtype
    nt_local = int(pred_trace.numel())

    # Keep observed/predicted fingerprints on the same window, following OT_LS_FWI.
    t = torch.arange(nt_local, device=device_local, dtype=dtype_local) * torch.as_tensor(
        dt_local, device=device_local, dtype=dtype_local
    )
    grid_t0 = t[0]
    grid_t1 = t[-1]
    grid_u0 = 2.0 * torch.min(pred_trace)
    grid_u1 = 2.0 * torch.max(pred_trace)

    geom_pred = _build_distance_geom(
        pred_trace,
        dt_local,
        nugrid,
        ntgrid,
        eps,
        t0=grid_t0,
        t1=grid_t1,
        u0=grid_u0,
        u1=grid_u1,
    )
    geom_obs = _build_distance_geom(
        obs_trace,
        dt_local,
        nugrid,
        ntgrid,
        eps,
        t0=grid_t0,
        t1=grid_t1,
        u0=grid_u0,
        u1=grid_u1,
    )

    lam = torch.as_tensor(lambda_d, device=device_local, dtype=dtype_local)
    pdf_obs_raw = torch.exp(-torch.abs(geom_obs.dfield) / lam)
    pdf_pred_raw = torch.exp(-torch.abs(geom_pred.dfield) / lam)

    amp_obs = pdf_obs_raw.sum().clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )
    amp_pred = pdf_pred_raw.sum().clamp_min(
        torch.as_tensor(eps, device=device_local, dtype=dtype_local)
    )
    pdf_obs = pdf_obs_raw / amp_obs
    pdf_pred = pdf_pred_raw / amp_pred

    f0_obs = torch.sum(pdf_obs, dim=0)
    f1_obs = torch.sum(pdf_obs, dim=1)
    f0_pred = torch.sum(pdf_pred, dim=0)
    f1_pred = torch.sum(pdf_pred, dim=1)

    f0_cdf_obs = torch.cumsum(f0_obs, dim=0)
    f1_cdf_obs = torch.cumsum(f1_obs, dim=0)
    f0_cdf_pred = torch.cumsum(f0_pred, dim=0)
    f1_cdf_pred = torch.cumsum(f1_pred, dim=0)

    w0, dw0, dw0dt = _wasser_w2_and_grad(
        f0_cdf_pred, f0_cdf_obs, geom_pred.t_axis_norm, geom_obs.t_axis_norm, eps
    )
    w1, dw1, dw1dt = _wasser_w2_and_grad(
        f1_cdf_pred, f1_cdf_obs, geom_pred.u_axis_norm, geom_obs.u_axis_norm, eps
    )

    dwpmargX = dw0.view(1, -1).repeat(nugrid, 1)
    dwpmargY = dw1.view(-1, 1).repeat(1, ntgrid)

    flat_pdf_pred = pdf_pred.reshape(-1)
    dwpmargX = dwpmargX - torch.sum(dwpmargX.reshape(-1) * flat_pdf_pred)
    dwpmargY = dwpmargY - torch.sum(dwpmargY.reshape(-1) * flat_pdf_pred)
    dwpmargX = dwpmargX / amp_pred
    dwpmargY = dwpmargY / amp_pred

    grad_t = _pdf_deriv_marg(dwpmargX, pdf_pred_raw, geom_pred, lambda_d, eps, nt_local)
    grad_u = _pdf_deriv_marg(dwpmargY, pdf_pred_raw, geom_pred, lambda_d, eps, nt_local)

    loss = 0.5 * (w0 + w1)
    grad = 0.5 * (grad_t + grad_u)
    components = {
        "w_time": w0,
        "w_amp": w1,
        "dw_dt_time_marginal": dw0dt,
        "dw_dt_amp_marginal": dw1dt,
    }
    return loss, grad, components


def ot_marginal_w2_loss_and_grad(
    d_pred: torch.Tensor,
    d_obs: torch.Tensor,
    *,
    dt: float,
    lambda_d: float = 0.04,
    nugrid: int = 40,
    ntgrid: int | None = None,
    reduction: str = "mean",
    eps: float = 1e-12,
    return_components: bool = False,
    trace_batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if d_pred.shape != d_obs.shape:
        raise ValueError(
            f"d_pred and d_obs must have the same shape, got {d_pred.shape} vs {d_obs.shape}."
        )
    if d_pred.ndim != 3:
        raise ValueError(
            f"Expected shape [nt, n_shots, n_receivers], got ndim={d_pred.ndim}."
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be either 'mean' or 'sum'.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if lambda_d <= 0.0:
        raise ValueError("lambda_d must be positive.")
    if nugrid < 2:
        raise ValueError("nugrid must be >= 2.")
    if not torch.isfinite(d_pred).all() or not torch.isfinite(d_obs).all():
        raise ValueError("d_pred and d_obs must contain only finite values.")

    if ntgrid is None:
        ntgrid = int(d_pred.shape[0])
    if ntgrid < 2:
        raise ValueError("ntgrid must be >= 2.")
    if trace_batch_size <= 0:
        raise ValueError("trace_batch_size must be positive.")

    nt_local, n_shots_local, n_receivers_local = d_pred.shape
    pred_flat = d_pred.detach().permute(1, 2, 0).reshape(-1, nt_local)
    obs_flat = d_obs.detach().permute(1, 2, 0).reshape(-1, nt_local)

    losses: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    comp_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        if return_components:
            for i in range(pred_flat.shape[0]):
                w, g, c = _trace_ot_loss_grad(
                    pred_flat[i],
                    obs_flat[i],
                    dt_local=dt,
                    lambda_d=lambda_d,
                    nugrid=nugrid,
                    ntgrid=ntgrid,
                    eps=eps,
                )
                losses.append(w)
                grads.append(g)
                comp_rows.append(c)
        else:
            for i0 in range(0, pred_flat.shape[0], trace_batch_size):
                i1 = min(i0 + trace_batch_size, pred_flat.shape[0])
                w_b, g_b = _trace_ot_loss_grad_batch(
                    pred_flat[i0:i1],
                    obs_flat[i0:i1],
                    dt_local=dt,
                    lambda_d=lambda_d,
                    nugrid=nugrid,
                    ntgrid=ntgrid,
                    eps=eps,
                )
                losses.extend([w_b[j] for j in range(w_b.shape[0])])
                grads.extend([g_b[j] for j in range(g_b.shape[0])])

    loss_vec = torch.stack(losses, dim=0)
    grad_flat = torch.stack(grads, dim=0)

    if reduction == "mean":
        scale = 1.0 / float(loss_vec.numel())
        loss = loss_vec.mean()
        grad_flat = grad_flat * grad_flat.new_tensor(scale)
    else:
        loss = loss_vec.sum()

    grad_pred = (
        grad_flat.reshape(n_shots_local, n_receivers_local, nt_local)
        .permute(2, 0, 1)
        .contiguous()
    )

    if return_components:
        return loss, grad_pred, {"per_trace": comp_rows}
    return loss, grad_pred


def apply_external_trace_grad(d_pred: torch.Tensor, grad_pred: torch.Tensor) -> torch.Tensor:
    if d_pred.shape != grad_pred.shape:
        raise ValueError(
            f"d_pred and grad_pred must have identical shapes, got {d_pred.shape} vs {grad_pred.shape}."
        )
    if not torch.isfinite(grad_pred).all():
        raise ValueError("grad_pred must contain only finite values.")
    return torch.sum(d_pred * grad_pred.detach())


def design_fir_filter(cutoff_hz: float, fs: float, numtaps: int) -> torch.Tensor:
    """Design a Hamming-windowed low-pass FIR filter."""
    n = torch.arange(numtaps, dtype=torch.float32)
    window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (numtaps - 1))
    sinc = torch.sin(2 * torch.pi * (cutoff_hz / fs) * (n - (numtaps - 1) / 2)) / (
        torch.pi * (n - (numtaps - 1) / 2)
    )
    center = (numtaps - 1) // 2
    sinc[center] = 2 * cutoff_hz / fs
    h = window * sinc
    return h / h.sum()


def apply_fir_lowpass(data: torch.Tensor, dt: float, cutoff_hz: float) -> torch.Tensor:
    """Apply FIR low-pass filter along the time axis to observed/synthetic data."""
    if cutoff_hz <= 0:
        return data

    fs = 1.0 / dt
    numtaps = max(3, int(fs / cutoff_hz))
    if numtaps % 2 == 0:
        numtaps += 1
    fir_coeff = design_fir_filter(cutoff_hz, fs, numtaps).to(
        device=data.device, dtype=data.dtype
    )

    if data.ndim == 1:
        data_2d = data.view(1, 1, -1)
        padded = F.pad(data_2d, (numtaps - 1, 0), mode="reflect")
        filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
        return filtered.view(-1)

    if data.ndim == 3:
        nt_local, n_shots_local, n_rx_local = data.shape
        reshaped = data.permute(1, 2, 0).reshape(-1, 1, nt_local)
        padded = F.pad(reshaped, (numtaps - 1, 0), mode="reflect")
        filtered = F.conv1d(padded, fir_coeff.view(1, 1, -1), padding=0)
        return filtered.view(n_shots_local, n_rx_local, nt_local).permute(2, 0, 1)

    raise ValueError(
        f"Unsupported data dimension: {data.ndim}. Expected 1D or 3D tensor."
    )


def save_filter_comparison(
    observed_base: torch.Tensor, observed_sets: dict, output_dir: Path
) -> None:
    """Save base vs filtered data comparison figure."""
    base_np = observed_base.detach().cpu().numpy()[:, :, 0]
    filtered_arrays = []
    for key in filter_specs:
        data_np = observed_sets[key]["data"].detach().cpu().numpy()[:, :, 0]
        filtered_arrays.append((key, data_np, observed_sets[key]["desc"]))

    absmax = max(
        np.abs(base_np).max(), *(np.abs(arr).max() for _, arr, _ in filtered_arrays)
    )
    vlim = (-absmax, absmax)

    n_cols = 1 + len(filtered_arrays)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(5 * n_cols, 5), sharex=True, sharey=True
    )
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(base_np, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
    axes[0].set_title(f"{base_forward_freq / 1e6:.0f} MHz base")
    axes[0].set_xlabel("Shots")
    axes[0].set_ylabel("Time samples")

    for idx, (_, arr, desc) in enumerate(filtered_arrays, start=1):
        axes[idx].imshow(arr, aspect="auto", cmap="seismic", vmin=vlim[0], vmax=vlim[1])
        axes[idx].set_title(desc)
        axes[idx].set_xlabel("Shots")

    plt.tight_layout()
    filename = (
        output_dir
        / f"data_filter_comparison_base{int(base_forward_freq / 1e6)}_lp{lowpass_tag}.jpg"
    )
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved data filter comparison to '{filename}'")


def save_model_snapshot(
    eps_array: np.ndarray, title: str, filename: Path, vmin: float, vmax: float
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(eps_array, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X (grid points)")
    ax.set_ylabel("Y (grid points)")
    plt.colorbar(im, ax=ax, label="εr")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved model snapshot to '{filename}'")


def forward_shots(
    epsilon, sigma, mu, shot_indices, source_amplitude_full, requires_grad=True
):
    src_amp = source_amplitude_full[shot_indices]
    src_loc = source_locations[shot_indices]
    rec_loc = receiver_locations[shot_indices]

    out = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dx,
        dt=dt,
        source_amplitude=src_amp,
        source_location=src_loc,
        receiver_location=rec_loc,
        pml_width=pml_width,
        save_snapshots=requires_grad,
        model_gradient_sampling_interval=model_gradient_sampling_interval
        if requires_grad
        else 1,
    )
    return out[-1]  # [nt, shots_in_batch, 1]


def generate_base_and_filtered_observed():
    with torch.no_grad():
        wavelet = tide.ricker(
            base_forward_freq, nt, dt, peak_time=1.0 / base_forward_freq
        ).to(device)
        src_amp_full = wavelet.view(1, 1, nt).repeat(n_shots, 1, 1)

        obs_list = []
        for shot_indices in make_shot_batches():
            obs_list.append(
                forward_shots(
                    epsilon_true,
                    sigma_true,
                    mu_true,
                    shot_indices,
                    src_amp_full,
                    requires_grad=False,
                )
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
        observed_base = torch.cat(obs_list, dim=1)

        observed_sets = {}
        for key, spec in filter_specs.items():
            lowpass_hz = float(spec["lowpass_mhz"]) * 1e6
            data_filtered = (
                apply_fir_lowpass(observed_base, dt=dt, cutoff_hz=lowpass_hz)
                if lowpass_hz > 0
                else observed_base
            )
            observed_sets[key] = {
                "data": data_filtered,
                "lowpass_hz": lowpass_hz,
                "desc": spec["desc"],
            }

    return observed_base, observed_sets, src_amp_full


sigma_smooth = 8
epsilon_init_raw = gaussian_filter(epsilon_true_raw, sigma=sigma_smooth)
epsilon_init_np = epsilon_init_raw.copy()
epsilon_init_np[:air_layer, :] = 1.0

sigma_init_np = np.ones_like(epsilon_init_np) * 0
sigma_init_np[:air_layer, :] = 0.0

epsilon_init = torch.tensor(epsilon_init_np, dtype=torch.float32, device=device)
sigma_init = torch.tensor(sigma_init_np, dtype=torch.float32, device=device)

epsilon_inv = epsilon_init.clone().detach()
epsilon_inv.requires_grad_(True)

sigma_fixed = sigma_init.clone().detach()
mu_fixed = torch.ones_like(epsilon_inv)

air_mask = torch.zeros_like(epsilon_inv, dtype=torch.bool)
air_mask[:air_layer, :] = True

loss_mode = "ot_manual"  # Options: "ot_manual", "mse"
ot_lambda_d = 0.04
ot_nugrid = 40
ot_ntgrid = nt
ot_trace_batch_size = 16
ot_reduction = "sum"  # Use trace-sum reduction to amplify update magnitude.
all_losses = []
stage_breaks = []

print("Starting multiscale filtered inversion")
time_start_all = time.time()

print("Generating base observed data once, then FIR filtering...")
observed_raw, observed_sets, src_amp_full = generate_base_and_filtered_observed()
print(f"Base forward modeled at {base_forward_freq / 1e6:.0f} MHz.")
print(f"Loss mode: {loss_mode}")
print(f"OT trace batch size: {ot_trace_batch_size}")
print(f"OT reduction: {ot_reduction}")
report_pde_totals("After observed generation: ")
save_filter_comparison(observed_raw, observed_sets, output_dir)

vmin_stage = epsilon_true_np.min()
vmax_stage = epsilon_true_np.max()

for stage_idx, cfg in enumerate(inversion_schedule, 1):
    data_key = cfg["data_key"]
    obs_cfg = observed_sets[data_key]
    n_epochs_adamw = int(cfg["adamw_epochs"])
    n_epochs_lbfgs = int(cfg["lbfgs_epochs"])
    lowpass_hz = obs_cfg["lowpass_hz"]

    print(f"\n==== Stage {stage_idx}: {obs_cfg['desc']} ====")
    observed_filtered = obs_cfg["data"]
    stage_forward_start = pde_counts["forward"]
    stage_adjoint_start = pde_counts["adjoint"]

    # Stage 1: AdamW
    optimizer_adamw = torch.optim.AdamW(
        [epsilon_inv], lr=0.01, betas=(0.9, 0.99), weight_decay=1e-3
    )
    for epoch in range(n_epochs_adamw):
        optimizer_adamw.zero_grad()
        epoch_loss = 0.0

        for shot_indices in make_shot_batches():
            syn = forward_shots(
                epsilon_inv,
                sigma_fixed,
                mu_fixed,
                shot_indices,
                src_amp_full,
                requires_grad=True,
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
            syn_filtered = apply_fir_lowpass(syn, dt=dt, cutoff_hz=lowpass_hz)
            obs_batch = observed_filtered[:, shot_indices, :]
            if loss_mode == "ot_manual":
                loss, grad_syn = ot_marginal_w2_loss_and_grad(
                    syn_filtered,
                    obs_batch,
                    dt=dt,
                    lambda_d=ot_lambda_d,
                    nugrid=ot_nugrid,
                    ntgrid=ot_ntgrid,
                    reduction=ot_reduction,
                    trace_batch_size=ot_trace_batch_size,
                )
                surrogate = apply_external_trace_grad(syn_filtered, grad_syn)
                surrogate.backward()
            else:
                loss = torch.nn.functional.mse_loss(syn_filtered, obs_batch)
                loss.backward()
            add_pde_counts(int(shot_indices.numel()), adjoint=True)
            epoch_loss += loss.item()

        if epsilon_inv.grad is not None:
            epsilon_inv.grad[air_mask] = 0.0
            valid_grads = epsilon_inv.grad[~air_mask].abs()
            if valid_grads.numel() > 0:
                clip_val = torch.quantile(valid_grads, 0.98)
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val.item())

        optimizer_adamw.step()

        with torch.no_grad():
            epsilon_inv.clamp_(1.0, 9.0)
            epsilon_inv[air_mask] = 1.0

        all_losses.append(epoch_loss)
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"  AdamW epoch {epoch + 1}/{n_epochs_adamw}  Loss={epoch_loss:.6e}")

    # Stage 2: L-BFGS
    optimizer_lbfgs = torch.optim.LBFGS(
        [epsilon_inv],
        lr=1.0,
        history_size=5,
        max_iter=5,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss = torch.zeros((), device=device)
        for shot_indices in make_shot_batches():
            syn = forward_shots(
                epsilon_inv,
                sigma_fixed,
                mu_fixed,
                shot_indices,
                src_amp_full,
                requires_grad=True,
            )
            add_pde_counts(int(shot_indices.numel()), forward=True)
            syn_filtered = apply_fir_lowpass(syn, dt=dt, cutoff_hz=lowpass_hz)
            obs_batch = observed_filtered[:, shot_indices, :]
            if loss_mode == "ot_manual":
                loss, grad_syn = ot_marginal_w2_loss_and_grad(
                    syn_filtered,
                    obs_batch,
                    dt=dt,
                    lambda_d=ot_lambda_d,
                    nugrid=ot_nugrid,
                    ntgrid=ot_ntgrid,
                    reduction=ot_reduction,
                    trace_batch_size=ot_trace_batch_size,
                )
                surrogate = apply_external_trace_grad(syn_filtered, grad_syn)
                surrogate.backward()
            else:
                loss = torch.nn.functional.mse_loss(syn_filtered, obs_batch)
                loss.backward()
            add_pde_counts(int(shot_indices.numel()), adjoint=True)
            total_loss = total_loss + loss

        if epsilon_inv.grad is not None:
            epsilon_inv.grad[air_mask] = 0.0
            valid_grads = epsilon_inv.grad[~air_mask].abs()
            if valid_grads.numel() > 0:
                clip_val = torch.quantile(valid_grads, 0.98)
                torch.nn.utils.clip_grad_value_([epsilon_inv], clip_val.item())
        return total_loss

    for epoch in range(n_epochs_lbfgs):
        loss = optimizer_lbfgs.step(closure)
        with torch.no_grad():
            epsilon_inv.clamp_(1.0, 9.0)
            epsilon_inv[air_mask] = 1.0
        loss_value = loss.item()
        all_losses.append(loss_value)
        print(f"  LBFGS epoch {epoch + 1}/{n_epochs_lbfgs}  Loss={loss_value:.6e}")

    stage_breaks.append(len(all_losses) - 1)
    report_pde_delta(f"Stage {stage_idx} ", stage_forward_start, stage_adjoint_start)
    eps_stage = epsilon_inv.detach().cpu().numpy()
    stage_title = f"{obs_cfg['desc']} inversion result"
    stage_fname = output_dir / f"epsilon_stage_{data_key}.jpg"
    save_model_snapshot(eps_stage, stage_title, stage_fname, vmin_stage, vmax_stage)

time_all = time.time() - time_start_all
print(f"\nTotal inversion time: {time_all:.2f}s")
report_pde_totals("Total ")

eps_true = epsilon_true.cpu().numpy()
eps_init = epsilon_init.cpu().numpy()
eps_result = epsilon_inv.detach().cpu().numpy()

vmin = eps_true.min()
vmax = eps_true.max()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
im = ax.imshow(eps_true, aspect="auto", vmin=vmin, vmax=vmax)
ax.set_title("True Model")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[0, 1]
im = ax.imshow(eps_init, aspect="auto", vmin=vmin, vmax=vmax)
ax.set_title("Initial Model (Smoothed)")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[1, 0]
im = ax.imshow(eps_result, aspect="auto", vmin=vmin, vmax=vmax)
ax.set_title("Multiscale Filtered Result")
ax.set_xlabel("X (grid points)")
ax.set_ylabel("Y (grid points)")
plt.colorbar(im, ax=ax, label="εr")

ax = axes[1, 1]
ax.semilogy(all_losses, label="Loss")
for idx in stage_breaks:
    ax.axvline(idx, color="r", linestyle="--", alpha=0.5)
ax.set_title("Loss Curve (AdamW -> LBFGS stages)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
ax.legend()

plt.tight_layout()
final_plot = output_dir / "multiscale_filtered_summary.jpg"
plt.savefig(final_plot, dpi=150)
print(f"\nResults saved to '{final_plot}'")

# Save inverted model for metrics computation
np.save(output_dir / "epsilon_inverted.npy", eps_result)
print(f"Saved inverted model to '{output_dir / 'epsilon_inverted.npy'}'")

mask = ~(air_mask.cpu().numpy())
rms_init = np.sqrt(np.mean((eps_init[mask] - eps_true[mask]) ** 2))
rms_result = np.sqrt(np.mean((eps_result[mask] - eps_true[mask]) ** 2))

print(f"RMS Error (Initial):  {rms_init:.4f}")
print(f"RMS Error (Inverted): {rms_result:.4f}")
print(f"Improvement: {(1 - rms_result / rms_init) * 100:.1f}%")

print("\n=== Timing Summary ===")
print(f"Total inversion time: {time_all:.2f}s")
