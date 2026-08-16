from __future__ import annotations

import math
import pytest
import scipy.special
import tide
import torch
import warnings
from numerical_utils import make_maxwell3d_example, make_tm2d_example
from tide import backend_utils

# --- test_maxwell_analytic.py ---


def analytic_trace_const_medium(
    wavelet: torch.Tensor,
    dt: float,
    src_pos_m: tuple[float, float],
    rec_pos_m: tuple[float, float],
    eps_r: float,
    sigma: float,
    current: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical 2D TM solution in a homogeneous medium."""
    device = wavelet.device
    dtype = torch.float64
    nt = wavelet.numel()

    eps0 = 1.0 / (36.0 * math.pi) * 1e-9
    mu0 = 4.0 * math.pi * 1e-7

    t = torch.arange(nt, device=device, dtype=dtype) * dt
    r = torch.tensor(rec_pos_m, device=device, dtype=dtype) - torch.tensor(
        src_pos_m, device=device, dtype=dtype
    )
    R = torch.linalg.norm(r) + 1e-12

    ricker_real = wavelet.to(dtype)
    spectrum = torch.fft.rfft(ricker_real)

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
    k = omega_safe * torch.sqrt(mu0 * eps_complex)
    hankel0 = torch.from_numpy(scipy.special.hankel2(0, (k * R).cpu().numpy())).to(
        device=device, dtype=torch.complex128
    )

    green = -current * omega_safe * mu0 * hankel0 / 4.0
    green[0] = 0.0 + 0.0j

    u_freq = spectrum * green
    u_time = torch.fft.irfft(u_freq, n=nt).real
    return t, u_time


def test_maxwelltm_matches_constant_medium_analytic_waveform():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Maxwell analytic test.")
    device = torch.device("cuda")
    dtype = torch.float64

    freq0 = 9e8  # Hz
    dt = 1e-11  # s
    nt = 800

    dx = dy = 0.005  # m
    eps_r = 10.0
    conductivity = 1e-3  # S/m

    ny, nx = 96, 128
    src_idx = (ny // 2, nx // 2)
    rec_idx = (ny // 2, nx // 2 + 20)  # ~0.1 m offset

    epsilon = torch.full((ny, nx), eps_r, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, conductivity)
    mu = torch.ones_like(epsilon)

    wavelet = tide.ricker(
        freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device
    )
    source_amplitude = wavelet.view(1, 1, nt)

    source_location = torch.tensor([[src_idx]], device=device)
    receiver_location = torch.tensor([[rec_idx]], device=device)

    _, _, _, _, _, _, _, receivers = tide.maxwell._kernel_api.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=dy,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=2,
        pml_width=10,
        save_snapshots=False,
    )

    simulated = receivers[:, 0, 0].cpu()

    src_pos_m = (src_idx[0] * dy, src_idx[1] * dx)
    rec_pos_m = (rec_idx[0] * dy, rec_idx[1] * dx)

    _, analytic = analytic_trace_const_medium(
        wavelet=wavelet.cpu(),
        dt=dt,
        src_pos_m=src_pos_m,
        rec_pos_m=rec_pos_m,
        eps_r=eps_r,
        sigma=conductivity,
    )

    alignment = torch.sign(torch.dot(simulated, analytic))
    misfit = torch.linalg.norm(
        simulated / torch.linalg.norm(simulated)
        - alignment * analytic / torch.linalg.norm(analytic)
    )
    peak_shift = abs(int(simulated.abs().argmax()) - int(analytic.abs().argmax()))

    assert misfit < 0.05
    assert peak_shift <= 3


@pytest.mark.slow
@pytest.mark.numerical
def test_maxwelltm_waveform_converges_under_fixed_domain_refinement():
    eps_r = 4.0
    frequency = 250e6
    physical_time = 6.0e-9
    traces = []
    for spacing in (0.02, 0.01, 0.005):
        dt = 2.0e-9 * spacing
        nt = round(physical_time / dt)
        size = round(0.6 / spacing) + 1
        source_index = (size // 2, size // 2)
        receiver_index = (size // 2, source_index[1] + round(0.1 / spacing))
        epsilon = torch.full((size, size), eps_r, dtype=torch.float64)
        sigma = torch.zeros_like(epsilon)
        source = tide.ricker(
            frequency,
            nt,
            dt,
            peak_time=1.0 / frequency,
            dtype=torch.float64,
        )
        receiver = tide.maxwell._kernel_api.maxwelltm(
            epsilon,
            sigma,
            torch.ones_like(epsilon),
            spacing,
            dt,
            source.view(1, 1, nt),
            torch.tensor([[source_index]]),
            torch.tensor([[receiver_index]]),
            stencil=2,
            pml_width=round(0.1 / spacing),
            python_backend=True,
        )[-1][:, 0, 0]
        traces.append(receiver)
    errors = [
        float(
            torch.linalg.vector_norm(
                coarse / torch.linalg.vector_norm(coarse)
                - fine[1::2] / torch.linalg.vector_norm(fine[1::2])
            )
        )
        for coarse, fine in zip(traces[:-1], traces[1:], strict=True)
    ]

    observed_order = math.log2(errors[0] / errors[1])
    assert errors[1] < errors[0], errors
    # ponytail: point injection is first-order; use spatial interpolation for 1.7+.
    assert observed_order >= 0.9, (errors, observed_order)
    assert errors[-1] <= 0.03, errors


# --- test_maxwell3d_analytic.py ---


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

    r_zyx = torch.tensor(rec_pos_m, device=device, dtype=dtype) - torch.tensor(
        src_pos_m, device=device, dtype=dtype
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
    ey_line = torch.exp(-(((x - x0) / width) ** 2))
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

    out = tide.maxwell._kernel_api.maxwell3d(
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
def test_maxwell3d_matches_uniform_medium_green_waveform_polarizations(
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
            out = tide.maxwell._kernel_api.maxwell3d(
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

            alignment = torch.sign(torch.dot(simulated, analytic))
            misfit = torch.linalg.norm(
                simulated / torch.linalg.norm(simulated)
                - alignment * analytic / torch.linalg.norm(analytic)
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
def test_maxwell3d_matches_uniform_medium_green_waveform_long_nt(
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

    out = tide.maxwell._kernel_api.maxwell3d(
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

    alignment = torch.sign(torch.dot(simulated, analytic))
    misfit = torch.linalg.norm(
        simulated / torch.linalg.norm(simulated)
        - alignment * analytic / torch.linalg.norm(analytic)
    )
    peak_shift = abs(
        int(simulated.abs().argmax().item()) - int(analytic.abs().argmax().item())
    )

    assert misfit < 0.08
    assert peak_shift <= 2


# --- test_maxwell3d_dispersion_analytic.py ---


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

    out = tide.maxwell._kernel_api.maxwell3d(
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


def _assert_waveform_match(simulated: torch.Tensor, analytic: torch.Tensor) -> None:
    simulated_norm = torch.linalg.vector_norm(simulated)
    analytic_norm = torch.linalg.vector_norm(analytic)
    assert simulated_norm > 0.0 and analytic_norm > 0.0
    alignment = torch.sign(torch.dot(simulated, analytic))
    misfit = torch.linalg.vector_norm(
        simulated / simulated_norm - alignment * analytic / analytic_norm
    )
    peak_shift = abs(
        int(simulated.abs().argmax().item()) - int(analytic.abs().argmax().item())
    )
    assert misfit < 0.20, f"misfit too large: {float(misfit):.4f}"
    assert peak_shift <= 5, f"peak shift too large: {peak_shift}"


@pytest.mark.parametrize(
    ("component", "analytic_solver", "analytic_index"),
    [
        ("ez", _analytic_dispersive_dipole_z, 2),
        ("ey", _analytic_dispersive_dipole_y, 1),
    ],
)
def test_maxwell3d_dispersive_matches_analytic(
    component, analytic_solver, analytic_index
) -> None:
    _require_cuda_backend()
    parameters = {
        "epsr": 4.0,
        "delta": 2.0,
        "tau": 2e-10,
        "sigma": 0.005,
        "dt": 1e-11,
        "nt": 360,
        "ds": 0.005,
    }
    simulated, (x, y, z), wavelet = _run_numeric_trace(
        source_component=component,
        receiver_component=component,
        **parameters,
    )
    analytic = analytic_solver(
        wavelet=wavelet,
        x=x,
        y=y,
        z=z,
        **{name: parameters[name] for name in ("dt", "epsr", "delta", "tau", "sigma")},
    )[analytic_index]

    _assert_waveform_match(simulated, analytic)


# --- test_dispersion.py ---


def _tm_example():
    return make_tm2d_example(
        shape=(8, 9),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=80e6,
        source_location=(4, 4),
        receiver_locations=((4, 5),),
        pml_width=1,
        python_backend=True,
    )


def _maxwell3d_example(device: torch.device | str = "cpu"):
    return make_maxwell3d_example(
        shape=(5, 6, 7),
        nt=8,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=70e6,
        device=device,
        source_location=(2, 3, 2),
        receiver_locations=((2, 3, 4),),
        pml_width=1,
    )


def test_debye_tm_operator_matches_reference_kernel():
    example = _tm_example()
    dispersion = tide.DebyeDispersion(delta_epsilon=2.0, tau=5e-10)
    operator = tide.MaxwellTM(
        tide.Discretization(
            example.grid_spacing,
            example.dt,
            boundary=tide.CPML(example.pml_width),
        ),
        tide.Experiment(
            tide.Acquisition(example.source_location, example.receiver_location),
            example.source_amplitude,
        ),
        execution=tide.ExecutionOptions(backend=tide.BackendPreference.REFERENCE),
    )
    actual = operator(
        tide.EMModel(
            example.epsilon,
            example.sigma,
            example.mu,
            dispersion=dispersion,
        )
    )
    expected = example.run(dispersion=dispersion)
    torch.testing.assert_close(actual.receiver_data, expected[-1])


def test_debye_zero_delta_matches_nondispersive():
    example = _tm_example()
    reference = example.run()
    actual = example.run(dispersion=tide.DebyeDispersion(delta_epsilon=0.0, tau=5e-10))
    for reference_output, actual_output in zip(reference, actual, strict=True):
        torch.testing.assert_close(reference_output, actual_output)


def test_debye_single_pole_matches_explicit_pole_axis():
    example = _tm_example()
    ny, nx = example.epsilon.shape
    scalar = example.run(dispersion=tide.DebyeDispersion(delta_epsilon=1.5, tau=5e-10))
    explicit = example.run(
        dispersion=tide.DebyeDispersion(
            delta_epsilon=torch.full(
                (1, ny, nx),
                1.5,
                dtype=example.epsilon.dtype,
            ),
            tau=torch.full(
                (1, ny, nx),
                5e-10,
                dtype=example.epsilon.dtype,
            ),
        )
    )
    for scalar_output, explicit_output in zip(scalar, explicit, strict=True):
        torch.testing.assert_close(scalar_output, explicit_output)


def test_debye_requires_dt_smaller_than_tau():
    example = _tm_example()
    with pytest.raises(ValueError, match="dt < min\\(tau\\)"):
        example.run(
            dt=5e-10,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )


def test_debye_tm_native_forward_matches_python():
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")
    example = _tm_example()
    dispersion = tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10)
    reference = example.run(python_backend=True, dispersion=dispersion)
    actual = example.run(python_backend=False, dispersion=dispersion)
    for reference_output, actual_output in zip(reference, actual, strict=True):
        torch.testing.assert_close(
            reference_output,
            actual_output,
            rtol=1e-4,
            atol=1e-5,
        )


def test_debye_em3d_native_forward_matches_python():
    if not backend_utils.is_backend_available():
        pytest.skip("native backend not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for 3D native Debye parity test")
    example = _maxwell3d_example("cuda")
    dispersion = tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10)
    reference = example.run(python_backend=True, dispersion=dispersion)
    actual = example.run(python_backend=False, dispersion=dispersion)
    for reference_output, actual_output in zip(reference, actual, strict=True):
        torch.testing.assert_close(
            reference_output,
            actual_output,
            rtol=1e-4,
            atol=1e-5,
        )


def test_debye_em3d_cpu_backend_falls_back_to_python():
    example = _maxwell3d_example()
    dispersion = tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10)
    reference = example.run(python_backend=True, dispersion=dispersion)
    with pytest.warns(RuntimeWarning, match="3D Debye CPU backend is not enabled yet"):
        actual = example.run(python_backend=False, dispersion=dispersion)

    for actual_output, reference_output in zip(actual, reference, strict=True):
        torch.testing.assert_close(actual_output, reference_output)


def test_debye_gradient_fallback_routes_through_policy():
    example = _tm_example()
    epsilon = example.epsilon.clone().requires_grad_(True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = example.run(
            epsilon=epsilon,
            python_backend=False,
            dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        )
    (gradient,) = torch.autograd.grad(output[-1].square().sum(), epsilon)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert not any(
        "Debye native backend currently supports forward inference only"
        in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize(
    ("example_factory", "expected_rank", "run_options"),
    [
        (_tm_example, 3, {}),
        (_maxwell3d_example, 5, {"python_backend": True}),
    ],
)
def test_debye_callback_exposes_dispersion_and_polarization(
    example_factory, expected_rank, run_options
) -> None:
    seen = {}

    def callback(state: tide.CallbackState) -> None:
        if seen:
            return
        seen["model_names"] = state.model_names
        seen["wavefield_names"] = state.wavefield_names
        seen["dispersion"] = state.get_model("dispersion")
        seen["polarization_shape"] = tuple(
            state.get_wavefield("polarization", view="inner").shape
        )

    example_factory().run(
        forward_callback=callback,
        dispersion=tide.DebyeDispersion(delta_epsilon=1.0, tau=5e-10),
        **run_options,
    )
    assert "dispersion" in seen["model_names"]
    assert "polarization" in seen["wavefield_names"]
    assert isinstance(seen["dispersion"], tide.DebyeDispersion)
    assert len(seen["polarization_shape"]) == expected_rank
