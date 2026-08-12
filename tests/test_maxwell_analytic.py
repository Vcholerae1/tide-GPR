import math

import pytest
import scipy.special
import torch

import tide


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
    r = (
        torch.tensor(rec_pos_m, device=device, dtype=dtype)
        - torch.tensor(src_pos_m, device=device, dtype=dtype)
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
    hankel0 = torch.from_numpy(
        scipy.special.hankel2(0, (k * R).cpu().numpy())
    ).to(device=device, dtype=torch.complex128)

    green = -current * omega_safe * mu0 * hankel0 / 4.0
    green[0] = 0.0 + 0.0j

    u_freq = spectrum * green
    u_time = torch.fft.irfft(u_freq, n=nt).real
    return t, u_time


def test_maxwelltm_matches_constant_medium_analytic():
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

    wavelet = tide.ricker(freq0, nt, dt, peak_time=1.0 / freq0, dtype=dtype, device=device)
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

    scale = torch.dot(simulated, analytic) / torch.dot(analytic, analytic)
    misfit = torch.linalg.norm(simulated - scale * analytic) / torch.linalg.norm(analytic)
    peak_shift = abs(int(simulated.abs().argmax()) - int(analytic.abs().argmax()))

    assert misfit < 0.05
    assert peak_shift <= 3
