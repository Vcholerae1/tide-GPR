import math

import pytest
import torch

import tide


def _devices() -> list[torch.device]:
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def _analytic_trace_const_medium_point_source_3d(
    wavelet: torch.Tensor,
    dt: float,
    src_pos_m: tuple[float, float, float],
    rec_pos_m: tuple[float, float, float],
    eps_r: float,
    sigma: float,
    mu_r: float = 1.0,
    source_component: str = "ey",
    receiver_component: str = "ey",
) -> torch.Tensor:
    """Analytical trace for arbitrary source/receiver component in 3D homogeneous medium."""
    component_to_idx = {"ex": 0, "ey": 1, "ez": 2}
    src_i = component_to_idx[source_component]
    rec_i = component_to_idx[receiver_component]

    device = wavelet.device
    dtype = torch.float64
    nt = wavelet.numel()

    eps0 = 1.0 / (36.0 * math.pi) * 1e-9
    mu0 = 4.0 * math.pi * 1e-7

    r_zyx = (
        torch.tensor(rec_pos_m, device=device, dtype=dtype)
        - torch.tensor(src_pos_m, device=device, dtype=dtype)
    )
    r_xyz = torch.stack((r_zyx[2], r_zyx[1], r_zyx[0]))
    R = torch.linalg.norm(r_xyz) + 1e-12
    r_hat = r_xyz / R

    spectrum = torch.fft.rfft(wavelet.to(dtype))
    freqs = torch.fft.rfftfreq(nt, d=dt).to(device)
    omega = 2.0 * math.pi * freqs
    omega_c = omega.to(torch.complex128)
    omega_safe = omega_c.clone()
    if omega_safe.numel() > 1:
        omega_safe[0] = omega_safe[1]
    else:
        omega_safe[0] = 1.0 + 0.0j

    eps_complex = (
        eps0 * torch.tensor(eps_r, device=device, dtype=torch.complex128)
        - 1j * torch.tensor(sigma, device=device, dtype=torch.float64) / omega_safe
    )
    k = omega_safe * torch.sqrt(
        mu0 * torch.tensor(mu_r, device=device, dtype=torch.complex128) * eps_complex
    )
    green_scalar = torch.exp(-1j * k * R) / (4.0 * math.pi * R)

    # Dyadic Green tensor component:
    # G_ij = A*delta_ij + B*rhat_i*rhat_j.
    kr = k * R
    a_term = 1.0 - 1j / kr - 1.0 / (kr * kr)
    b_term = -1.0 + 3j / kr + 3.0 / (kr * kr)
    delta = 1.0 if src_i == rec_i else 0.0
    dyadic_component = a_term * delta + b_term * (r_hat[rec_i] * r_hat[src_i])
    transfer = 1j * omega_safe * mu0 * green_scalar * dyadic_component
    transfer[0] = 0.0 + 0.0j

    return torch.fft.irfft(spectrum * transfer, n=nt).real


@pytest.mark.parametrize("device", _devices())
def test_maxwell3d_uniform_medium_plane_wave_travel_time(device: torch.device):
    """Uniform 3D medium should match analytic plane-wave travel-time lag.

    We initialize a right-going 1D plane wave in a 3D homogeneous, lossless
    medium and record Ey at two receivers along x. Analytically:

        delta_t = delta_x / v,  v = c0 / sqrt(epsilon_r * mu_r)

    This checks a core physical law (wave speed in homogeneous medium) while
    staying robust to discretization and source-modeling choices.
    """

    dtype = torch.float32
    nz, ny, nx = 10, 10, 180
    nt = 400

    epsilon_r = 4.0
    mu_r = 1.0
    epsilon = torch.full((nz, ny, nx), epsilon_r, device=device, dtype=dtype)
    sigma = torch.zeros_like(epsilon)
    mu = torch.full_like(epsilon, mu_r)

    dz = dy = dx = 0.02
    dt = 4e-11

    # Initial plane-wave packet Ey(x, t=0), uniform in y/z.
    x = torch.arange(nx, device=device, dtype=dtype) * dx
    x0 = 0.5
    width = 0.08
    ey_line = torch.exp(-((x - x0) / width) ** 2)
    ey_0 = ey_line.view(1, 1, nx).expand(nz, ny, nx).contiguous()

    # For a right-going wave in homogeneous medium: Hz = -Ey / eta.
    eta0 = 376.730313668
    eta = eta0 * math.sqrt(mu_r / epsilon_r)
    hz_0 = -ey_0 / eta

    rx_x = [90, 120]
    receiver_location = torch.tensor(
        [[[nz // 2, ny // 2, rx] for rx in rx_x]],
        device=device,
        dtype=torch.long,
    )

    out = tide.maxwell3d(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=[dz, dy, dx],
        dt=dt,
        source_amplitude=None,
        source_location=None,
        receiver_location=receiver_location,
        nt=nt,
        pml_width=0,
        stencil=4,
        Ey_0=ey_0,
        Hz_0=hz_0,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )
    rec = out[-1][:, 0, :]  # [nt, n_receivers]

    # Use absolute-peak time as robust arrival marker.
    i1 = int(torch.argmax(rec[:, 0].abs()).item())
    i2 = int(torch.argmax(rec[:, 1].abs()).item())
    observed_lag = i2 - i1

    v = tide.utils.C0 / math.sqrt(epsilon_r * mu_r)
    predicted_lag = (rx_x[1] - rx_x[0]) * dx / (v * dt)

    # 3D FDTD has numerical dispersion; allow a small sample-level tolerance.
    assert abs(observed_lag - predicted_lag) <= 3.0


@pytest.mark.parametrize("device", _devices())
def test_maxwell3d_matches_uniform_medium_point_source_green_polarizations(
    device: torch.device,
):
    """Uniform 3D medium should match point-source Green traces for ex/ey/ez polarizations."""
    dtype = torch.float32

    freq0 = 120e6  # Hz
    dt = 8e-11  # s
    nt = 260
    spacing = 0.02  # m
    eps_r = 9.0
    conductivity = 0.0
    mu_r = 1.0

    nz = ny = nx = 22
    src_idx = (nz // 2, ny // 2, nx // 2)
    rec_idx = (src_idx[0] + 3, src_idx[1] + 2, src_idx[2] + 5)

    epsilon = torch.full((nz, ny, nx), eps_r, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, conductivity)
    mu = torch.full_like(epsilon, mu_r)

    wavelet = tide.ricker(
        freq0,
        nt,
        dt,
        peak_time=1.2 / freq0,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, nt)
    source_location = torch.tensor([[list(src_idx)]], device=device, dtype=torch.long)
    receiver_location = torch.tensor([[list(rec_idx)]], device=device, dtype=torch.long)

    wavelet_cpu = wavelet.cpu()
    src_pos_m = tuple(idx * spacing for idx in src_idx)
    rec_pos_m = tuple(idx * spacing for idx in rec_idx)

    for source_component in ("ex", "ey", "ez"):
        for receiver_component in ("ex", "ey", "ez"):
            out = tide.maxwell3d(
                epsilon=epsilon,
                sigma=sigma,
                mu=mu,
                grid_spacing=[spacing, spacing, spacing],
                dt=dt,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                pml_width=7,
                stencil=4,
                source_component=source_component,
                receiver_component=receiver_component,
                python_backend=True,
            )
            simulated = out[-1][:, 0, 0].cpu().to(torch.float64)

            analytic = _analytic_trace_const_medium_point_source_3d(
                wavelet=wavelet_cpu,
                dt=dt,
                src_pos_m=src_pos_m,
                rec_pos_m=rec_pos_m,
                eps_r=eps_r,
                sigma=conductivity,
                mu_r=mu_r,
                source_component=source_component,
                receiver_component=receiver_component,
            )

            scale = torch.dot(simulated, analytic) / torch.dot(analytic, analytic)
            misfit = (
                torch.linalg.norm(simulated - scale * analytic)
                / torch.linalg.norm(analytic)
            )
            peak_shift = abs(
                int(simulated.abs().argmax().item())
                - int(analytic.abs().argmax().item())
            )

            assert misfit < 0.08, (
                f"misfit too large for {source_component}->{receiver_component}: "
                f"{float(misfit):.4f}"
            )
            assert peak_shift <= 2, (
                f"peak shift too large for {source_component}->{receiver_component}: "
                f"{peak_shift}"
            )


@pytest.mark.parametrize("device", _devices())
def test_maxwell3d_matches_uniform_medium_point_source_green_long_nt(
    device: torch.device,
):
    """Long-time trace should still match homogeneous-medium 3D Green response."""
    dtype = torch.float32

    freq0 = 120e6  # Hz
    dt = 8e-11  # s
    nt = 1200
    spacing = 0.02  # m
    eps_r = 9.0
    conductivity = 0.0
    mu_r = 1.0

    nz = ny = nx = 22
    src_idx = (nz // 2, ny // 2, nx // 2)
    rec_idx = (src_idx[0] + 3, src_idx[1] + 2, src_idx[2] + 5)

    epsilon = torch.full((nz, ny, nx), eps_r, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, conductivity)
    mu = torch.full_like(epsilon, mu_r)

    wavelet = tide.ricker(
        freq0,
        nt,
        dt,
        peak_time=1.2 / freq0,
        dtype=dtype,
        device=device,
    )
    source_amplitude = wavelet.view(1, 1, nt)
    source_location = torch.tensor([[list(src_idx)]], device=device, dtype=torch.long)
    receiver_location = torch.tensor([[list(rec_idx)]], device=device, dtype=torch.long)

    out = tide.maxwell3d(
        epsilon=epsilon,
        sigma=sigma,
        mu=mu,
        grid_spacing=[spacing, spacing, spacing],
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=7,
        stencil=4,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )
    simulated = out[-1][:, 0, 0].cpu().to(torch.float64)

    analytic = _analytic_trace_const_medium_point_source_3d(
        wavelet=wavelet.cpu(),
        dt=dt,
        src_pos_m=tuple(idx * spacing for idx in src_idx),
        rec_pos_m=tuple(idx * spacing for idx in rec_idx),
        eps_r=eps_r,
        sigma=conductivity,
        mu_r=mu_r,
        source_component="ey",
        receiver_component="ey",
    )

    scale = torch.dot(simulated, analytic) / torch.dot(analytic, analytic)
    misfit = (
        torch.linalg.norm(simulated - scale * analytic) / torch.linalg.norm(analytic)
    )
    peak_shift = abs(
        int(simulated.abs().argmax().item()) - int(analytic.abs().argmax().item())
    )

    assert misfit < 0.08
    assert peak_shift <= 2
