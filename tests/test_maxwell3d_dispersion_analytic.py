import math

import pytest
import torch

import tide
from tide import backend_utils


def _require_cuda_backend() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for 3D dispersive analytic comparison.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for faster 3D dispersive comparison.")


def _analytic_dispersive_dipole_z(
    wavelet: torch.Tensor,
    dt: float,
    x: float,
    y: float,
    z: float,
    epsr: float,
    delta: float,
    tau: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frequency-domain closed form for a z-directed dipole in Debye medium."""
    nt = int(wavelet.numel())
    wavelet = wavelet.to(torch.float64)
    n_fft = 1 << ((max(4 * nt, 512) - 1).bit_length())
    wavelet_padded = torch.zeros(n_fft, dtype=torch.float64)
    wavelet_padded[:nt] = wavelet

    eps0 = 8.854e-12
    mu0 = 4.0 * math.pi * 1e-7

    r = math.sqrt(x * x + y * y + z * z) + 1e-12
    theta = math.atan2(math.sqrt(x * x + y * y), z)
    phi = math.atan2(y, x)

    spectrum = torch.fft.rfft(wavelet_padded)
    freqs = torch.fft.rfftfreq(n_fft, d=dt)
    omega = 2.0 * math.pi * freqs

    efx = torch.zeros_like(spectrum, dtype=torch.complex128)
    efy = torch.zeros_like(spectrum, dtype=torch.complex128)
    efz = torch.zeros_like(spectrum, dtype=torch.complex128)

    idx = torch.nonzero(omega > 0.0, as_tuple=False).flatten()
    if idx.numel() == 0:
        zt = torch.zeros(nt, dtype=torch.float64)
        return zt.clone(), zt.clone(), zt.clone()

    om = omega[idx].to(torch.complex128)
    ep = epsr + delta / (1.0 + 1j * om * tau)
    k = torch.sqrt(om * om * eps0 * mu0 * (ep - 1j * sigma / (om * eps0)))
    eta = torch.sqrt(mu0 / (eps0 * (ep - 1j * sigma / (om * eps0))))

    # Io*l is set to 1 to match the reference scripts.
    er = (
        (eta / (2.0 * math.pi * r * r))
        * (1.0 + 1.0 / (1j * k * r))
        * math.cos(theta)
        * torch.exp(-1j * k * r)
    )
    etheta = (
        (1j * eta * k / (4.0 * math.pi * r))
        * (1.0 + 1.0 / (1j * k * r) - 1.0 / (k * r) ** 2)
        * math.sin(theta)
        * torch.exp(-1j * k * r)
    )

    ex = er * math.sin(theta) * math.cos(phi) + etheta * math.cos(theta) * math.cos(phi)
    ey = er * math.sin(theta) * math.sin(phi) + etheta * math.cos(theta) * math.sin(phi)
    ez = er * math.cos(theta) - etheta * math.sin(theta)

    efx[idx] = ex * spectrum[idx]
    efy[idx] = ey * spectrum[idx]
    efz[idx] = ez * spectrum[idx]

    tx = torch.fft.irfft(efx, n=n_fft).real[:nt]
    ty = torch.fft.irfft(efy, n=n_fft).real[:nt]
    tz = torch.fft.irfft(efz, n=n_fft).real[:nt]
    return tx, ty, tz


def _analytic_dispersive_dipole_y(
    wavelet: torch.Tensor,
    dt: float,
    x: float,
    y: float,
    z: float,
    epsr: float,
    delta: float,
    tau: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse z-directed solution with y<->z axis swap (same as ref/analytical_y.m)."""
    x_swap = x
    y_swap = z
    z_swap = y

    ex_swap, ey_swap, ez_swap = _analytic_dispersive_dipole_z(
        wavelet=wavelet,
        dt=dt,
        x=x_swap,
        y=y_swap,
        z=z_swap,
        epsr=epsr,
        delta=delta,
        tau=tau,
        sigma=sigma,
    )

    ex = ex_swap
    ey = ez_swap
    ez = ey_swap
    return ex, ey, ez


def _run_numeric_trace(
    source_component: str,
    receiver_component: str,
    *,
    epsr: float,
    sigma: float,
    delta: float,
    tau: float,
    dt: float,
    nt: int,
    ds: float,
) -> tuple[torch.Tensor, tuple[float, float, float], torch.Tensor]:
    device = torch.device("cuda")
    dtype = torch.float32

    nz = ny = nx = 96
    src = (32, 32, 32)  # z, y, x
    rec = (63, 63, 63)  # z, y, x  -> 0.155 m offset on each axis when ds=0.005

    epsilon = torch.full((nz, ny, nx), epsr, device=device, dtype=dtype)
    cond = torch.full_like(epsilon, sigma)
    mu = torch.ones_like(epsilon)

    wavelet = tide.ricker(
        9e8,
        nt,
        dt,
        peak_time=1.0 / 9e8,
        dtype=dtype,
        device=device,
    )

    out = tide.maxwell3d(
        epsilon=epsilon,
        sigma=cond,
        mu=mu,
        grid_spacing=[ds, ds, ds],
        dt=dt,
        source_amplitude=wavelet.view(1, 1, nt),
        source_location=torch.tensor([[list(src)]], dtype=torch.long, device=device),
        receiver_location=torch.tensor([[list(rec)]], dtype=torch.long, device=device),
        pml_width=12,
        stencil=4,
        source_component=source_component,
        receiver_component=receiver_component,
        python_backend=False,
        dispersion=tide.DebyeDispersion(delta_epsilon=delta, tau=tau),
    )

    simulated = out[-1][:, 0, 0].detach().cpu().to(torch.float64)

    dx = (rec[2] - src[2]) * ds
    dy = (rec[1] - src[1]) * ds
    dz = (rec[0] - src[0]) * ds
    return simulated, (dx, dy, dz), wavelet.detach().cpu().to(torch.float64)


def _assert_scaled_trace_match(simulated: torch.Tensor, analytic: torch.Tensor) -> None:
    scale = torch.dot(simulated, analytic) / (torch.dot(analytic, analytic) + 1e-24)
    residual = simulated - scale * analytic
    misfit = torch.linalg.norm(residual) / (torch.linalg.norm(analytic) + 1e-24)
    peak_shift = abs(
        int(simulated.abs().argmax().item()) - int(analytic.abs().argmax().item())
    )

    # Dispersive + finite-grid + finite-domain setting: keep tolerance practical.
    assert misfit < 0.20, f"misfit too large: {float(misfit):.4f}"
    assert peak_shift <= 5, f"peak shift too large: {peak_shift}"


def test_maxwell3d_dispersive_matches_analytic_z_source_ez() -> None:
    _require_cuda_backend()

    epsr = 4.0
    delta = 2.0
    tau = 2e-10
    sigma = 0.005
    dt = 1e-11
    nt = 360
    ds = 0.005

    simulated, (x, y, z), wavelet = _run_numeric_trace(
        source_component="ez",
        receiver_component="ez",
        epsr=epsr,
        sigma=sigma,
        delta=delta,
        tau=tau,
        dt=dt,
        nt=nt,
        ds=ds,
    )

    _, _, analytic_ez = _analytic_dispersive_dipole_z(
        wavelet=wavelet,
        dt=dt,
        x=x,
        y=y,
        z=z,
        epsr=epsr,
        delta=delta,
        tau=tau,
        sigma=sigma,
    )

    _assert_scaled_trace_match(simulated, analytic_ez)


def test_maxwell3d_dispersive_matches_analytic_y_source_ey() -> None:
    _require_cuda_backend()

    epsr = 4.0
    delta = 2.0
    tau = 2e-10
    sigma = 0.005
    dt = 1e-11
    nt = 360
    ds = 0.005

    simulated, (x, y, z), wavelet = _run_numeric_trace(
        source_component="ey",
        receiver_component="ey",
        epsr=epsr,
        sigma=sigma,
        delta=delta,
        tau=tau,
        dt=dt,
        nt=nt,
        ds=ds,
    )

    _, analytic_ey, _ = _analytic_dispersive_dipole_y(
        wavelet=wavelet,
        dt=dt,
        x=x,
        y=y,
        z=z,
        epsr=epsr,
        delta=delta,
        tau=tau,
        sigma=sigma,
    )

    _assert_scaled_trace_match(simulated, analytic_ey)
