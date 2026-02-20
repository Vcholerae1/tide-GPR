"""Gradient validation for epsilon and sigma in MaxwellTM.

This script runs:
1. Dot-product / Taylor remainder tests for epsilon-only, sigma-only, and joint perturbations.
2. Directional finite-difference checks against adjoint gradients.
3. Pointwise finite-difference checks at top-|gradient| locations.

Run from repo root:
    PYTHONPATH=src .venv/bin/python examples/example_gradient_dot_fd_validation.py

Optional backend selection:
    --backend eager      # Python backend reference
    --backend c          # Native backend (python_backend=False)
    --backend both       # Run both eager and c

Optional extra CUDA run:
    --also-cuda          # Keep primary runs and append CUDA run(s)
    --cuda-backend c     # Backend for the extra CUDA run (default: c)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import torch

import tide

Tensor = torch.Tensor


@dataclass(frozen=True)
class ProblemConfig:
    ny: int = 24
    nx: int = 28
    nt: int = 180
    air_layer: int = 2
    n_shots: int = 3
    n_receivers: int = 8
    dx: float = 0.02
    dt: float = 2e-11
    pml_width: int = 6
    stencil: int = 2
    freq: float = 220e6
    max_vel: float = 3e8
    model_gradient_sampling_interval: int = 1
    seed: int = 1234


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_backend_tokens(token: str) -> list[bool | str]:
    norm = token.strip().lower()
    if norm == "both":
        return ["eager", False]
    if norm in ("c", "native", "false"):
        return [False]
    if norm in ("eager", "jit"):
        return [norm]
    raise ValueError(f"Unsupported backend '{token}'. Use one of: both, c, eager, jit.")


def backend_label(backend: bool | str) -> str:
    if backend is False:
        return "c"
    return str(backend)


def append_unique_target(
    targets: list[tuple[torch.device, bool | str]],
    device: torch.device,
    backend: bool | str,
) -> None:
    backend_name = backend_label(backend)
    for dev_i, backend_i in targets:
        if dev_i.type == device.type and backend_label(backend_i) == backend_name:
            return
    targets.append((device, backend))


def build_problem(
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(cfg.seed)

    y = torch.linspace(0.0, 1.0, cfg.ny, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, cfg.nx, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    epsilon_true = 4.0 + 1.0 * torch.exp(-((xx - 0.42) ** 2 + (yy - 0.58) ** 2) / 0.03)
    sigma_true = 4e-4 + 1.5e-3 * torch.exp(-((xx - 0.68) ** 2 + (yy - 0.40) ** 2) / 0.04)

    epsilon_init = torch.full_like(epsilon_true, 4.0)
    sigma_init = torch.full_like(sigma_true, 7e-4)

    epsilon_true[: cfg.air_layer, :] = 1.0
    sigma_true[: cfg.air_layer, :] = 0.0
    epsilon_init[: cfg.air_layer, :] = 1.0
    sigma_init[: cfg.air_layer, :] = 0.0

    mu = torch.ones_like(epsilon_true)

    source_location = torch.zeros(cfg.n_shots, 1, 2, dtype=torch.long, device=device)
    receiver_location = torch.zeros(
        cfg.n_shots, cfg.n_receivers, 2, dtype=torch.long, device=device
    )

    source_x = torch.linspace(4, cfg.nx - 8, cfg.n_shots, device=device).round().long()
    receiver_x = torch.linspace(
        2, cfg.nx - 3, cfg.n_receivers, device=device
    ).round().long()
    for i in range(cfg.n_shots):
        source_location[i, 0, 0] = cfg.air_layer
        source_location[i, 0, 1] = source_x[i]
        receiver_location[i, :, 0] = cfg.air_layer
        receiver_location[i, :, 1] = receiver_x

    wavelet = tide.ricker(
        cfg.freq, cfg.nt, cfg.dt, peak_time=1.0 / cfg.freq, dtype=dtype, device=device
    )
    source_amplitude = wavelet.view(1, 1, cfg.nt).repeat(cfg.n_shots, 1, 1)

    return epsilon_true, sigma_true, epsilon_init, sigma_init, mu, source_location, receiver_location, source_amplitude


def make_forward(
    cfg: ProblemConfig,
    mu: Tensor,
    source_location: Tensor,
    receiver_location: Tensor,
    source_amplitude: Tensor,
    backend: bool | str,
) -> Callable[[Tensor, Tensor], Tensor]:
    def forward(epsilon: Tensor, sigma: Tensor) -> Tensor:
        return tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=cfg.dx,
            dt=cfg.dt,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=cfg.pml_width,
            stencil=cfg.stencil,
            max_vel=cfg.max_vel,
            model_gradient_sampling_interval=cfg.model_gradient_sampling_interval,
            python_backend=backend,
        )[-1]

    return forward


def make_objective(
    forward: Callable[[Tensor, Tensor], Tensor],
    epsilon_true: Tensor,
    sigma_true: Tensor,
) -> Callable[[Tensor, Tensor], Tensor]:
    with torch.no_grad():
        observed = forward(epsilon_true, sigma_true)

    def objective(epsilon: Tensor, sigma: Tensor) -> Tensor:
        residual = forward(epsilon, sigma) - observed
        return 0.5 * residual.square().mean()

    return objective


def make_normalized_direction(grad: Tensor, air_layer: int) -> Tensor:
    direction = torch.randn_like(grad)
    direction[:air_layer, :] = 0.0
    direction = direction / (direction.norm() + 1e-12)
    return direction


def directional_derivative(grad_eps: Tensor, grad_sigma: Tensor, d_eps: Tensor, d_sigma: Tensor) -> float:
    return float((grad_eps * d_eps).sum().item() + (grad_sigma * d_sigma).sum().item())


def run_taylor_test(
    objective: Callable[[Tensor, Tensor], Tensor],
    epsilon0: Tensor,
    sigma0: Tensor,
    loss0: float,
    grad_eps: Tensor,
    grad_sigma: Tensor,
    d_eps: Tensor,
    d_sigma: Tensor,
    hs: list[float],
) -> None:
    cases = [
        ("epsilon-only", d_eps, torch.zeros_like(d_sigma)),
        ("sigma-only", torch.zeros_like(d_eps), d_sigma),
        ("joint", d_eps, d_sigma),
    ]

    print("Dot Product / Taylor Test")
    for name, case_deps, case_dsigma in cases:
        adj = directional_derivative(grad_eps, grad_sigma, case_deps, case_dsigma)
        print(f"\n  [{name}]  <g,dm> = {adj:.6e}")
        print("    h          |J(m+h dm)-J(m)-h<g,dm>|     ratio(prev/curr)   order(log2)")
        prev = None
        for h in hs:
            eps_h = (epsilon0 + h * case_deps).detach()
            sig_h = (sigma0 + h * case_dsigma).detach()
            loss_h = float(objective(eps_h, sig_h).detach().item())
            remainder2 = abs((loss_h - loss0) - h * adj)
            ratio = float("nan") if prev is None else prev / (remainder2 + 1e-30)
            order = float("nan") if prev is None else math.log2(ratio)
            print(f"    {h:8.2e}   {remainder2:24.6e}   {ratio:16.6f}   {order:11.6f}")
            prev = remainder2


def run_directional_fd_test(
    objective: Callable[[Tensor, Tensor], Tensor],
    epsilon0: Tensor,
    sigma0: Tensor,
    grad_eps: Tensor,
    grad_sigma: Tensor,
    d_eps: Tensor,
    d_sigma: Tensor,
    h_eps: float,
    h_sigma: float,
    h_joint: float,
) -> None:
    cases = [
        ("epsilon-only", d_eps, torch.zeros_like(d_sigma), h_eps),
        ("sigma-only", torch.zeros_like(d_eps), d_sigma, h_sigma),
        ("joint", d_eps, d_sigma, h_joint),
    ]

    print("\nDirectional Finite Difference Check")
    print("  case          h          FD(centered)        <g,dm>            rel_error")
    for name, case_deps, case_dsigma, h in cases:
        eps_p = (epsilon0 + h * case_deps).detach()
        sig_p = (sigma0 + h * case_dsigma).detach()
        eps_m = (epsilon0 - h * case_deps).detach()
        sig_m = (sigma0 - h * case_dsigma).detach()

        fd = float(((objective(eps_p, sig_p) - objective(eps_m, sig_m)) / (2.0 * h)).item())
        adj = directional_derivative(grad_eps, grad_sigma, case_deps, case_dsigma)
        rel_err = abs(fd - adj) / (abs(fd) + 1e-12)
        print(f"  {name:12s}  {h:8.2e}  {fd:16.6e}  {adj:16.6e}  {rel_err:12.6e}")


def topk_points_by_grad(grad: Tensor, air_layer: int, k: int) -> list[tuple[int, int]]:
    mag = grad.abs().clone()
    mag[:air_layer, :] = 0.0
    k = min(k, mag.numel())
    _, idx = torch.topk(mag.reshape(-1), k=k)
    nx = grad.shape[1]
    points = []
    for flat_idx in idx.tolist():
        iy = flat_idx // nx
        ix = flat_idx % nx
        points.append((iy, ix))
    return points


def run_pointwise_fd_test(
    objective: Callable[[Tensor, Tensor], Tensor],
    epsilon0: Tensor,
    sigma0: Tensor,
    grad_eps: Tensor,
    grad_sigma: Tensor,
    air_layer: int,
    k_points: int,
    h_eps: float,
    h_sigma: float,
) -> None:
    print("\nPointwise Finite Difference Check (top-|grad| points)")

    eps_points = topk_points_by_grad(grad_eps, air_layer=air_layer, k=k_points)
    sig_points = topk_points_by_grad(grad_sigma, air_layer=air_layer, k=k_points)

    print("  epsilon:")
    print("    (y,x)        FD(centered)        grad_autograd      rel_error")
    for iy, ix in eps_points:
        eps_p = epsilon0.clone()
        eps_m = epsilon0.clone()
        eps_p[iy, ix] += h_eps
        eps_m[iy, ix] -= h_eps
        fd = float(((objective(eps_p, sigma0) - objective(eps_m, sigma0)) / (2.0 * h_eps)).item())
        ga = float(grad_eps[iy, ix].item())
        rel_err = abs(fd - ga) / (abs(fd) + 1e-12)
        print(f"    ({iy:2d},{ix:2d})   {fd:16.6e}   {ga:16.6e}   {rel_err:12.6e}")

    print("  sigma:")
    print("    (y,x)        FD(centered)        grad_autograd      rel_error")
    for iy, ix in sig_points:
        sig_p = sigma0.clone()
        sig_m = sigma0.clone()
        sig_p[iy, ix] += h_sigma
        sig_m[iy, ix] -= h_sigma
        fd = float(((objective(epsilon0, sig_p) - objective(epsilon0, sig_m)) / (2.0 * h_sigma)).item())
        ga = float(grad_sigma[iy, ix].item())
        rel_err = abs(fd - ga) / (abs(fd) + 1e-12)
        print(f"    ({iy:2d},{ix:2d})   {fd:16.6e}   {ga:16.6e}   {rel_err:12.6e}")


def run_for_backend(
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
    backend: bool | str,
    dot_steps: list[float],
    fd_h_eps: float,
    fd_h_sigma: float,
    fd_h_joint: float,
    point_h_eps: float,
    point_h_sigma: float,
    point_k: int,
) -> None:
    (
        epsilon_true,
        sigma_true,
        epsilon0,
        sigma0,
        mu,
        source_location,
        receiver_location,
        source_amplitude,
    ) = build_problem(cfg, device=device, dtype=dtype)

    forward = make_forward(
        cfg,
        mu=mu,
        source_location=source_location,
        receiver_location=receiver_location,
        source_amplitude=source_amplitude,
        backend=backend,
    )
    objective = make_objective(forward, epsilon_true=epsilon_true, sigma_true=sigma_true)

    epsilon = epsilon0.clone().detach().requires_grad_(True)
    sigma = sigma0.clone().detach().requires_grad_(True)
    loss0_t = objective(epsilon, sigma)
    loss0_t.backward()
    assert epsilon.grad is not None
    assert sigma.grad is not None
    grad_eps = epsilon.grad.detach().clone()
    grad_sigma = sigma.grad.detach().clone()
    loss0 = float(loss0_t.detach().item())

    d_eps = make_normalized_direction(grad_eps, cfg.air_layer)
    d_sigma = make_normalized_direction(grad_sigma, cfg.air_layer)

    print("\n" + "=" * 84)
    print(f"Backend: {backend_label(backend)}   Device: {device.type}   dtype: {dtype}")
    print(f"Loss(m0): {loss0:.6e}")
    print(
        f"||grad_eps||2={grad_eps.norm().item():.6e}   "
        f"||grad_sigma||2={grad_sigma.norm().item():.6e}"
    )

    run_taylor_test(
        objective=objective,
        epsilon0=epsilon0,
        sigma0=sigma0,
        loss0=loss0,
        grad_eps=grad_eps,
        grad_sigma=grad_sigma,
        d_eps=d_eps,
        d_sigma=d_sigma,
        hs=dot_steps,
    )
    run_directional_fd_test(
        objective=objective,
        epsilon0=epsilon0,
        sigma0=sigma0,
        grad_eps=grad_eps,
        grad_sigma=grad_sigma,
        d_eps=d_eps,
        d_sigma=d_sigma,
        h_eps=fd_h_eps,
        h_sigma=fd_h_sigma,
        h_joint=fd_h_joint,
    )
    run_pointwise_fd_test(
        objective=objective,
        epsilon0=epsilon0,
        sigma0=sigma0,
        grad_eps=grad_eps,
        grad_sigma=grad_sigma,
        air_layer=cfg.air_layer,
        k_points=point_k,
        h_eps=point_h_eps,
        h_sigma=point_h_sigma,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dot-product and finite-difference gradient validation for epsilon and sigma."
    )
    parser.add_argument("--backend", type=str, default="both", help="both | c | eager | jit")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--also-cuda",
        action="store_true",
        help="Append an extra CUDA run in addition to the primary --device run(s).",
    )
    parser.add_argument(
        "--cuda-backend",
        type=str,
        default="c",
        help="Backend for the extra CUDA run: c | eager | jit | both (default: c).",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="float32 | float64")
    parser.add_argument("--dot-steps", type=str, default="1e-1,5e-2,2.5e-2,1.25e-2,6.25e-3")
    parser.add_argument("--fd-h-eps", type=float, default=1e-3)
    parser.add_argument("--fd-h-sigma", type=float, default=1e-4)
    parser.add_argument("--fd-h-joint", type=float, default=1e-4)
    parser.add_argument("--point-h-eps", type=float, default=5e-3)
    parser.add_argument("--point-h-sigma", type=float, default=1e-4)
    parser.add_argument("--point-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    dot_steps = [float(x) for x in args.dot_steps.split(",") if x.strip()]
    primary_backends = parse_backend_tokens(args.backend)

    cfg = ProblemConfig(seed=args.seed)
    run_targets: list[tuple[torch.device, bool | str]] = []
    for backend in primary_backends:
        append_unique_target(run_targets, primary_device, backend)

    if args.also_cuda:
        if torch.cuda.is_available():
            cuda_backends = parse_backend_tokens(args.cuda_backend)
            cuda_device = torch.device("cuda")
            for backend in cuda_backends:
                append_unique_target(run_targets, cuda_device, backend)
        else:
            print("CUDA not available; skipping extra CUDA run requested by --also-cuda.")

    for device, backend in run_targets:
        run_for_backend(
            cfg=cfg,
            device=device,
            dtype=dtype,
            backend=backend,
            dot_steps=dot_steps,
            fd_h_eps=args.fd_h_eps,
            fd_h_sigma=args.fd_h_sigma,
            fd_h_joint=args.fd_h_joint,
            point_h_eps=args.point_h_eps,
            point_h_sigma=args.point_h_sigma,
            point_k=args.point_k,
        )


if __name__ == "__main__":
    main()
