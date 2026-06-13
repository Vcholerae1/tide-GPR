from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import tide

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regularization.tv_l1 import project_tv_l1_ball, tv_l1_value

try:
    from example_ifwi_gpr_eps import (
        build_geometry,
        generate_observed_data,
        make_shot_batches,
        make_true_epsilon_model,
        relative_l2,
        resolve_device,
        save_gather_panel,
        save_loss_curve,
    )
except ImportError:
    from examples.example_ifwi_gpr_eps import (
        build_geometry,
        generate_observed_data,
        make_shot_batches,
        make_true_epsilon_model,
        relative_l2,
        resolve_device,
        save_gather_panel,
        save_loss_curve,
    )


# Script configuration
MODEL_PATH = "examples/OverThrust.npy"
NX: int | None = None
NY: int | None = None

DX = 0.02
DT = 4e-11
NT = 2500
PML_WIDTH = 16
AIR_LAYERS = 3

BASE_FREQ = 1e8
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 0.0
GRAD_PRECONDITIONER = True
GRAD_PRECOND_UPDATE_INTERVAL = 0
GRAD_PRECOND_SMOOTH_SIGMA = 3.0
GRAD_PRECOND_DAMPING = 5e-2
GRAD_PRECOND_POWER = 0.5
GRAD_PRECOND_CLIP_LO = 0.3
GRAD_PRECOND_CLIP_HI = 3.0
GRAD_PRECOND_BLEND = 0.7

N_SHOTS = 8
SOURCE_X_MIN = 0
SOURCE_X_MAX: int | None = None
SOURCE_DEPTH: int | None = None

RECEIVER_DEPTH: int | None = None
BATCH_SIZE = 8
MODEL_GRADIENT_SAMPLING_INTERVAL = 16

EPS_MIN = 1.0
EPS_MAX = 8.0
SIGMA_MIN = 0.0
SIGMA_MAX = 5e-3
SIGMA_K = 1e-4
SIGMA_P = 2.0

INIT_TOP_EPS = 3.5
INIT_BOTTOM_EPS = 6.5
EPS_DELTA_SCALE = 1.5
SIGMA_DELTA_SCALE = 1.5e-3
FOURIER_MAPPING_SIZE = 128
FOURIER_SCALE = 3.0
FOURIER_INCLUDE_INPUT = True
HIDDEN_FEATURES = 256
HIDDEN_LAYERS = 4
ACTIVATION = "relu"
FINAL_LAYER_SCALE = 10
OUTPUT_SMOOTHING_KERNEL = 1
OUTPUT_SMOOTHING_PASSES = 0
COORD_CHUNK_SIZE: int | None = 65536
COMPILE_NETWORK = True

TV_L1_PROJECTION = True
TV_L1_EPSILON_RADIUS: float | None = None
TV_L1_SIGMA_RADIUS: float | None = None
TV_L1_EPSILON_RADIUS_FRACTION = 4.0
TV_L1_SIGMA_RADIUS_FRACTION = 0.0
TV_L1_PROJECTION_RHO = 10.0
TV_L1_PROJECTION_MAX_ITER = 60
TV_L1_PROJECTION_TOL = 5e-4
TV_L1_PROJECTION_CG_MAX_ITER = 80
TV_L1_PROJECTION_CG_TOL = 1e-5

LOG_INTERVAL = 10
SEED = 2026
DEVICE_NAME = "auto"
OUTPUT_DIR_NAME: str | None = None


class SineActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "sine":
        return SineActivation()
    raise ValueError(f"Unsupported activation: {name}")


def make_linear_initial_epsilon(
    *,
    ny: int,
    nx: int,
    air_layers: int,
    top_eps: float,
    bottom_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if air_layers < 0 or air_layers > ny:
        raise ValueError(f"air_layers must be in [0, {ny}], got {air_layers}.")
    if top_eps <= 0.0 or bottom_eps <= 0.0:
        raise ValueError("top_eps and bottom_eps must be positive.")

    epsilon = torch.ones((ny, nx), device=device, dtype=dtype)
    n_subsurface = ny - air_layers
    if n_subsurface > 0:
        depth = torch.linspace(0.0, 1.0, n_subsurface, device=device, dtype=dtype).view(-1, 1)
        trend = float(top_eps) + (float(bottom_eps) - float(top_eps)) * depth
        epsilon[air_layers:, :] = trend.expand(n_subsurface, nx)
    return epsilon


def sigma_from_epsilon_np(epsilon: np.ndarray) -> np.ndarray:
    eps_eff = np.maximum(epsilon - 1.0, 0.0)
    sigma = SIGMA_MIN + SIGMA_K * np.power(eps_eff, SIGMA_P)
    return np.clip(sigma, SIGMA_MIN, SIGMA_MAX).astype(np.float32, copy=False)


def sigma_from_epsilon(epsilon: torch.Tensor) -> torch.Tensor:
    eps_eff = torch.clamp(epsilon - 1.0, min=0.0)
    sigma = SIGMA_MIN + SIGMA_K * torch.pow(eps_eff, SIGMA_P)
    return torch.clamp(sigma, min=SIGMA_MIN, max=SIGMA_MAX)


def save_model_panel_labeled(
    *,
    panels: list[tuple[str, np.ndarray]],
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str = "turbo",
) -> None:
    vmin = float(min(float(data.min()) for _, data in panels))
    vmax = float(max(float(data.max()) for _, data in panels))
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5 * len(panels), 4.5),
        sharex=True,
        sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, panels, strict=True):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


class FourierFeatureEncoding(nn.Module):
    """Fixed random Fourier feature mapping from Tancik et al."""

    def __init__(
        self,
        *,
        in_features: int,
        mapping_size: int,
        scale: float,
        include_input: bool,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if mapping_size <= 0:
            raise ValueError("mapping_size must be positive.")
        if scale <= 0.0:
            raise ValueError("scale must be positive.")

        self.in_features = int(in_features)
        self.mapping_size = int(mapping_size)
        self.include_input = bool(include_input)
        b_matrix = torch.randn(mapping_size, in_features, dtype=torch.float32) * float(scale)
        self.register_buffer("b_matrix", b_matrix)

    @property
    def out_features(self) -> int:
        encoded = 2 * self.mapping_size
        if self.include_input:
            encoded += self.in_features
        return encoded

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projected = (2.0 * math.pi) * (coords @ self.b_matrix.T)
        encoded = torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)
        if self.include_input:
            encoded = torch.cat((coords, encoded), dim=-1)
        return encoded


class FourierFeatureMLP(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 2,
        out_features: int = 1,
        mapping_size: int = 256,
        scale: float = 8.0,
        include_input: bool = True,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        activation: str = "relu",
        final_layer_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if hidden_features <= 0:
            raise ValueError("hidden_features must be positive.")
        if hidden_layers < 0:
            raise ValueError("hidden_layers must be non-negative.")
        if final_layer_scale < 0.0:
            raise ValueError("final_layer_scale must be non-negative.")

        self.encoding = FourierFeatureEncoding(
            in_features=in_features,
            mapping_size=mapping_size,
            scale=scale,
            include_input=include_input,
        )

        layers: list[nn.Module] = []
        prev_features = self.encoding.out_features
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_features, hidden_features))
            layers.append(make_activation(activation))
            prev_features = hidden_features
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev_features, out_features)
        self._scale_final_layer(final_layer_scale)

    def _scale_final_layer(self, scale: float) -> None:
        with torch.no_grad():
            self.out.weight.mul_(float(scale))
            if self.out.bias is not None:
                self.out.bias.zero_()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.encoding(coords)
        x = self.hidden(x)
        return self.out(x)


class MaxwellTMFourierFeatureJointFWI(nn.Module):
    def __init__(
        self,
        *,
        ny: int,
        nx: int,
        dx: float,
        dt: float,
        nt: int,
        pml_width: int,
        eps_min: float,
        eps_max: float,
        sigma_min: float,
        sigma_max: float,
        sigma_init: torch.Tensor,
        mu: torch.Tensor,
        source_locations: torch.Tensor,
        receiver_locations: torch.Tensor,
        air_layers: int,
        fourier_mapping_size: int,
        fourier_scale: float,
        fourier_include_input: bool,
        hidden_features: int,
        hidden_layers: int,
        activation: str,
        final_layer_scale: float,
        epsilon_init: torch.Tensor,
        eps_delta_scale: float,
        sigma_delta_scale: float,
        model_gradient_sampling_interval: int,
        output_smoothing_kernel: int = 1,
        output_smoothing_passes: int = 0,
        coord_chunk_size: int | None = None,
        compile_network: bool = False,
    ) -> None:
        super().__init__()
        self.ny = int(ny)
        self.nx = int(nx)
        self.dx = float(dx)
        self.dt = float(dt)
        self.nt = int(nt)
        self.pml_width = int(pml_width)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps_delta_scale = float(eps_delta_scale)
        self.sigma_delta_scale = float(sigma_delta_scale)
        self.model_gradient_sampling_interval = int(model_gradient_sampling_interval)
        self.air_layers = int(air_layers)
        self.output_smoothing_kernel = int(output_smoothing_kernel)
        self.output_smoothing_passes = int(output_smoothing_passes)
        self.coord_chunk_size = None if coord_chunk_size is None else int(coord_chunk_size)
        if epsilon_init.shape != (self.ny, self.nx):
            raise ValueError(
                f"epsilon_init shape must be ({self.ny}, {self.nx}), got {tuple(epsilon_init.shape)}."
            )
        if sigma_init.shape != (self.ny, self.nx):
            raise ValueError(
                f"sigma_init shape must be ({self.ny}, {self.nx}), got {tuple(sigma_init.shape)}."
            )
        if self.eps_delta_scale <= 0.0:
            raise ValueError("eps_delta_scale must be positive.")
        if self.sigma_delta_scale <= 0.0:
            raise ValueError("sigma_delta_scale must be positive.")
        if self.sigma_min < 0.0 or self.sigma_max <= self.sigma_min:
            raise ValueError("sigma bounds must satisfy 0 <= sigma_min < sigma_max.")
        if self.output_smoothing_kernel < 1:
            raise ValueError("output_smoothing_kernel must be >= 1.")
        if self.output_smoothing_kernel % 2 == 0:
            raise ValueError("output_smoothing_kernel must be odd.")
        if self.output_smoothing_passes < 0:
            raise ValueError("output_smoothing_passes must be >= 0.")
        if self.coord_chunk_size is not None and self.coord_chunk_size <= 0:
            raise ValueError("coord_chunk_size must be positive or None.")

        y = torch.linspace(-1.0, 1.0, self.ny, device=sigma_init.device, dtype=sigma_init.dtype)
        x = torch.linspace(-1.0, 1.0, self.nx, device=sigma_init.device, dtype=sigma_init.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        air_mask = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=sigma_init.device)
        air_mask[:air_layers, :] = True

        self.register_buffer("coords", coords)
        self.register_buffer("epsilon_init", epsilon_init)
        self.register_buffer("sigma_init", sigma_init)
        self.register_buffer("mu", mu)
        self.register_buffer("source_locations", source_locations)
        self.register_buffer("receiver_locations", receiver_locations)
        self.register_buffer("air_mask", air_mask)
        self.register_buffer("epsilon_grad_preconditioner", torch.ones_like(epsilon_init))
        self.register_buffer("sigma_grad_preconditioner", torch.ones_like(sigma_init))
        self.output_gradient_preconditioning = False

        self.net = FourierFeatureMLP(
            in_features=2,
            out_features=2,
            mapping_size=fourier_mapping_size,
            scale=fourier_scale,
            include_input=fourier_include_input,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            activation=activation,
            final_layer_scale=final_layer_scale,
        )
        self.compile_network_requested = bool(compile_network)
        self._compiled_net = None

    def maybe_compile_network(self) -> bool:
        if not self.compile_network_requested or not hasattr(torch, "compile"):
            return False
        if self._compiled_net is not None:
            return True
        self._compiled_net = torch.compile(self.net)
        return True

    @property
    def n_shots(self) -> int:
        return int(self.source_locations.shape[0])

    def _run_network(self) -> torch.Tensor:
        net = self.net if self._compiled_net is None else self._compiled_net
        if self.coord_chunk_size is None or self.coords.shape[0] <= self.coord_chunk_size:
            return net(self.coords).reshape(self.ny, self.nx, 2)

        chunks = [net(chunk) for chunk in self.coords.split(self.coord_chunk_size, dim=0)]
        return torch.cat(chunks, dim=0).reshape(self.ny, self.nx, 2)

    def _smooth_field(self, field: torch.Tensor) -> torch.Tensor:
        if self.output_smoothing_kernel == 1 or self.output_smoothing_passes == 0:
            return field
        if self.air_layers >= self.ny:
            return field

        kernel = self.output_smoothing_kernel
        pad = kernel // 2
        subsurface = field[self.air_layers :, :].unsqueeze(0).unsqueeze(0)
        for _ in range(self.output_smoothing_passes):
            subsurface = F.avg_pool2d(
                F.pad(subsurface, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=kernel,
                stride=1,
            )
        return torch.cat((field[: self.air_layers, :], subsurface[0, 0]), dim=0)

    def set_output_gradient_preconditioner(
        self,
        *,
        epsilon_diag: torch.Tensor,
        sigma_diag: torch.Tensor,
    ) -> None:
        if epsilon_diag.shape != (self.ny, self.nx):
            raise ValueError(
                f"epsilon_diag shape must be ({self.ny}, {self.nx}), got {tuple(epsilon_diag.shape)}."
            )
        if sigma_diag.shape != (self.ny, self.nx):
            raise ValueError(
                f"sigma_diag shape must be ({self.ny}, {self.nx}), got {tuple(sigma_diag.shape)}."
            )

        with torch.no_grad():
            self.epsilon_grad_preconditioner.copy_(
                epsilon_diag.to(
                    device=self.epsilon_grad_preconditioner.device,
                    dtype=self.epsilon_grad_preconditioner.dtype,
                )
            )
            self.sigma_grad_preconditioner.copy_(
                sigma_diag.to(
                    device=self.sigma_grad_preconditioner.device,
                    dtype=self.sigma_grad_preconditioner.dtype,
                )
            )
        self.output_gradient_preconditioning = True

    def clear_output_gradient_preconditioner(self) -> None:
        with torch.no_grad():
            self.epsilon_grad_preconditioner.fill_(1.0)
            self.sigma_grad_preconditioner.fill_(1.0)
        self.output_gradient_preconditioning = False

    @staticmethod
    def _apply_output_preconditioner(
        grad: torch.Tensor,
        diag: torch.Tensor,
    ) -> torch.Tensor:
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return grad * diag

    def predict_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._run_network()
        eps_delta = self.eps_delta_scale * torch.tanh(raw[..., 0])
        sigma_delta = self.sigma_delta_scale * torch.tanh(raw[..., 1])
        eps_delta = torch.where(self.air_mask, torch.zeros_like(eps_delta), eps_delta)
        sigma_delta = torch.where(self.air_mask, torch.zeros_like(sigma_delta), sigma_delta)
        return eps_delta, sigma_delta

    def predict_models(self) -> tuple[torch.Tensor, torch.Tensor]:
        eps_delta, sigma_delta = self.predict_deltas()
        epsilon = self.epsilon_init + eps_delta
        sigma = self.sigma_init + sigma_delta

        epsilon = self._smooth_field(epsilon)
        sigma = self._smooth_field(sigma)

        epsilon = torch.clamp(epsilon, min=self.eps_min, max=self.eps_max)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        epsilon = torch.where(self.air_mask, torch.ones_like(epsilon), epsilon)
        sigma = torch.where(self.air_mask, torch.zeros_like(sigma), sigma)
        if self.output_gradient_preconditioning and epsilon.requires_grad:
            epsilon.register_hook(
                lambda grad: self._apply_output_preconditioner(
                    grad, self.epsilon_grad_preconditioner
                )
            )
            sigma.register_hook(
                lambda grad: self._apply_output_preconditioner(
                    grad, self.sigma_grad_preconditioner
                )
            )
        return epsilon, sigma

    def predict_epsilon(self) -> torch.Tensor:
        epsilon, _ = self.predict_models()
        return epsilon

    def predict_sigma(self) -> torch.Tensor:
        _, sigma = self.predict_models()
        return sigma

    def forward_shots(
        self,
        *,
        wavelet: torch.Tensor,
        shot_indices: torch.Tensor,
        requires_grad: bool,
    ) -> torch.Tensor:
        epsilon, sigma = self.predict_models()
        return self.forward_shots_with_models(
            epsilon=epsilon,
            sigma=sigma,
            wavelet=wavelet,
            shot_indices=shot_indices,
            requires_grad=requires_grad,
        )

    def forward_shots_with_models(
        self,
        *,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        wavelet: torch.Tensor,
        shot_indices: torch.Tensor,
        requires_grad: bool,
    ) -> torch.Tensor:
        batch_size = int(shot_indices.numel())
        src_amp = wavelet.view(1, 1, self.nt).expand(batch_size, 1, self.nt).contiguous()
        src_loc = self.source_locations[shot_indices]
        rec_loc = self.receiver_locations[shot_indices]

        out = tide.maxwelltm(
            epsilon=epsilon,
            sigma=sigma,
            mu=self.mu,
            grid_spacing=self.dx,
            dt=self.dt,
            source_amplitude=src_amp,
            source_location=src_loc,
            receiver_location=rec_loc,
            pml_width=self.pml_width,
            save_snapshots=requires_grad,
            model_gradient_sampling_interval=(
                self.model_gradient_sampling_interval if requires_grad else 1
            ),
            storage_mode="auto",
            storage_compression="bf16" if epsilon.device.type == "cuda" else False,
        )
        return out[-1]


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _make_preconditioner_diag(
    *,
    p_diag: torch.Tensor,
    air_mask: torch.Tensor,
    smooth_sigma: float,
    damping: float,
    power: float,
    clip_lo: float,
    clip_hi: float,
    blend: float,
) -> torch.Tensor:
    p_np = p_diag.detach().cpu().numpy().astype(np.float32, copy=False)
    if smooth_sigma > 0.0:
        p_np = gaussian_filter(p_np, sigma=float(smooth_sigma))
    p_np = np.nan_to_num(p_np, nan=0.0, posinf=0.0, neginf=0.0)

    air_mask_np = air_mask.detach().cpu().numpy()
    valid = ~air_mask_np
    p_valid = p_np[valid]
    if p_valid.size == 0:
        b0 = np.ones_like(p_np, dtype=np.float32)
        b0[air_mask_np] = 0.0
        return torch.from_numpy(b0).to(device=p_diag.device, dtype=p_diag.dtype)

    scale = float(np.quantile(p_valid, 0.95))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    p_np = p_np / scale

    b0 = np.power(p_np + float(damping), -float(power)).astype(np.float32, copy=False)
    b0 = np.nan_to_num(b0, nan=0.0, posinf=0.0, neginf=0.0)
    b_valid = b0[valid]
    if b_valid.size > 0:
        med = float(np.median(b_valid))
        if np.isfinite(med) and med > 0.0:
            b0 = b0 / med
        np.clip(b0, float(clip_lo), float(clip_hi), out=b0)
        b0 = (1.0 - float(blend)) + float(blend) * b0

    b0[air_mask_np] = 0.0
    return torch.from_numpy(b0.astype(np.float32, copy=False)).to(
        device=p_diag.device,
        dtype=p_diag.dtype,
    )


def build_output_gradient_preconditioner(
    *,
    model: MaxwellTMFourierFeatureJointFWI,
    wavelet: torch.Tensor,
    observed: torch.Tensor,
    batch_size: int,
    smooth_sigma: float,
    damping: float,
    power: float,
    clip_lo: float,
    clip_hi: float,
    blend: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_enabled = model.output_gradient_preconditioning
    model.output_gradient_preconditioning = False
    for param in model.net.parameters():
        param.grad = None

    p_eps = torch.zeros((model.ny, model.nx), device=model.coords.device, dtype=model.coords.dtype)
    p_sig = torch.zeros_like(p_eps)
    shot_batches = make_shot_batches(model.n_shots, batch_size, model.coords.device)

    for shot_indices in shot_batches:
        for param in model.net.parameters():
            param.grad = None

        epsilon, sigma = model.predict_models()
        epsilon.retain_grad()
        sigma.retain_grad()
        syn = model.forward_shots_with_models(
            epsilon=epsilon,
            sigma=sigma,
            wavelet=wavelet,
            shot_indices=shot_indices,
            requires_grad=True,
        )
        obs_batch = observed[:, shot_indices, :]
        weight = float(shot_indices.numel()) / float(model.n_shots)
        loss = F.mse_loss(syn, obs_batch)
        (loss * weight).backward()

        if epsilon.grad is not None:
            g_eps = torch.nan_to_num(
                epsilon.grad.detach(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            g_eps = torch.where(model.air_mask, torch.zeros_like(g_eps), g_eps)
            p_eps += g_eps * g_eps
        if sigma.grad is not None:
            g_sig = torch.nan_to_num(
                sigma.grad.detach(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            g_sig = torch.where(model.air_mask, torch.zeros_like(g_sig), g_sig)
            p_sig += g_sig * g_sig

    for param in model.net.parameters():
        param.grad = None

    eps_diag = _make_preconditioner_diag(
        p_diag=p_eps,
        air_mask=model.air_mask,
        smooth_sigma=smooth_sigma,
        damping=damping,
        power=power,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        blend=blend,
    )
    sig_diag = _make_preconditioner_diag(
        p_diag=p_sig,
        air_mask=model.air_mask,
        smooth_sigma=smooth_sigma,
        damping=damping,
        power=power,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        blend=blend,
    )
    model.output_gradient_preconditioning = was_enabled
    return eps_diag, sig_diag


def install_output_gradient_preconditioner(
    *,
    model: MaxwellTMFourierFeatureJointFWI,
    wavelet: torch.Tensor,
    observed: torch.Tensor,
    batch_size: int,
) -> None:
    eps_diag, sig_diag = build_output_gradient_preconditioner(
        model=model,
        wavelet=wavelet,
        observed=observed,
        batch_size=batch_size,
        smooth_sigma=GRAD_PRECOND_SMOOTH_SIGMA,
        damping=GRAD_PRECOND_DAMPING,
        power=GRAD_PRECOND_POWER,
        clip_lo=GRAD_PRECOND_CLIP_LO,
        clip_hi=GRAD_PRECOND_CLIP_HI,
        blend=GRAD_PRECOND_BLEND,
    )
    model.set_output_gradient_preconditioner(epsilon_diag=eps_diag, sigma_diag=sig_diag)

    valid = ~model.air_mask
    eps_valid = eps_diag[valid].detach().cpu().numpy()
    sig_valid = sig_diag[valid].detach().cpu().numpy()
    print(
        "Output-gradient preconditioner: "
        f"smooth_sigma={GRAD_PRECOND_SMOOTH_SIGMA:g}, "
        f"damping={GRAD_PRECOND_DAMPING:.1e}, "
        f"power={GRAD_PRECOND_POWER:.2f}, "
        f"clip=[{GRAD_PRECOND_CLIP_LO:.2f}, {GRAD_PRECOND_CLIP_HI:.2f}], "
        f"blend={GRAD_PRECOND_BLEND:.2f}"
    )
    print(
        "  epsilon B0 stats: "
        f"min={eps_valid.min():.3e}, median={np.median(eps_valid):.3e}, "
        f"max={eps_valid.max():.3e}"
    )
    print(
        "  sigma B0 stats: "
        f"min={sig_valid.min():.3e}, median={np.median(sig_valid):.3e}, "
        f"max={sig_valid.max():.3e}"
    )


def _resolve_tv_l1_projection_radius(
    *,
    model_tensor: torch.Tensor,
    active_mask: torch.Tensor,
    absolute_radius: float | None,
    radius_fraction: float,
    spacing: tuple[float, float],
    label: str,
) -> float | None:
    if absolute_radius is not None:
        if absolute_radius < 0.0:
            raise ValueError(f"{label} TV-L1 radius must be non-negative.")
        return float(absolute_radius)
    if radius_fraction <= 0.0:
        return None
    initial_tv = tv_l1_value(
        model_tensor,
        spacing=spacing,
        active_mask=active_mask,
        reduction="sum",
        spatial_ndim=2,
    )
    return float(radius_fraction * initial_tv.detach().item())


def _project_models_tv_l1(
    *,
    model: MaxwellTMFourierFeatureJointFWI,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    epsilon_radius: float | None,
    sigma_radius: float | None,
    straight_through: bool,
) -> tuple[torch.Tensor, torch.Tensor, object | None, object | None]:
    active_mask = ~model.air_mask
    spacing = (model.dx, model.dx)
    eps_info = None
    sig_info = None

    if epsilon_radius is not None:
        epsilon_projected, eps_info = project_tv_l1_ball(
            epsilon.detach(),
            radius=epsilon_radius,
            spacing=spacing,
            active_mask=active_mask,
            rho=TV_L1_PROJECTION_RHO,
            max_iter=TV_L1_PROJECTION_MAX_ITER,
            tol=TV_L1_PROJECTION_TOL,
            cg_max_iter=TV_L1_PROJECTION_CG_MAX_ITER,
            cg_tol=TV_L1_PROJECTION_CG_TOL,
            spatial_ndim=2,
        )
        if straight_through:
            epsilon = epsilon + (epsilon_projected - epsilon).detach()
        else:
            epsilon = epsilon_projected

    if sigma_radius is not None:
        sigma_projected, sig_info = project_tv_l1_ball(
            sigma.detach(),
            radius=sigma_radius,
            spacing=spacing,
            active_mask=active_mask,
            rho=TV_L1_PROJECTION_RHO,
            max_iter=TV_L1_PROJECTION_MAX_ITER,
            tol=TV_L1_PROJECTION_TOL,
            cg_max_iter=TV_L1_PROJECTION_CG_MAX_ITER,
            cg_tol=TV_L1_PROJECTION_CG_TOL,
            spatial_ndim=2,
        )
        if straight_through:
            sigma = sigma + (sigma_projected - sigma).detach()
        else:
            sigma = sigma_projected

    return epsilon, sigma, eps_info, sig_info


def _format_tv_projection_info(label: str, info: object | None) -> str:
    if info is None:
        return f"{label}=off"
    converged = "ok" if info.converged else "maxit"
    return (
        f"{label}={info.tv_l1:.3e}/{info.radius:.3e} "
        f"{converged} it={info.iterations}"
    )


def run_waveform_inversion(
    *,
    model: MaxwellTMFourierFeatureJointFWI,
    wavelet: torch.Tensor,
    observed: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    grad_clip: float,
    log_interval: int,
    grad_preconditioner: bool = GRAD_PRECONDITIONER,
    grad_precond_update_interval: int = GRAD_PRECOND_UPDATE_INTERVAL,
    tv_l1_projection: bool = TV_L1_PROJECTION,
    tv_l1_epsilon_radius: float | None = None,
    tv_l1_sigma_radius: float | None = None,
) -> list[float]:
    history: list[float] = []
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr, weight_decay=weight_decay)
    shot_batches = make_shot_batches(model.n_shots, batch_size, model.coords.device)
    tv_enabled = tv_l1_projection and (
        tv_l1_epsilon_radius is not None or tv_l1_sigma_radius is not None
    )

    if tv_enabled:
        print(
            "TV-L1 model-space projection: "
            f"epsilon_radius={tv_l1_epsilon_radius}, sigma_radius={tv_l1_sigma_radius}, "
            f"rho={TV_L1_PROJECTION_RHO:g}, max_iter={TV_L1_PROJECTION_MAX_ITER}, "
            "active=subsurface, gradient=straight-through"
        )

    if grad_preconditioner:
        print("Building output-gradient preconditioner...")
        install_output_gradient_preconditioner(
            model=model,
            wavelet=wavelet,
            observed=observed,
            batch_size=batch_size,
        )
    else:
        model.clear_output_gradient_preconditioner()

    for epoch in range(epochs):
        if (
            grad_preconditioner
            and grad_precond_update_interval > 0
            and epoch > 0
            and epoch % grad_precond_update_interval == 0
        ):
            print(f"Refreshing output-gradient preconditioner at epoch {epoch + 1}...")
            install_output_gradient_preconditioner(
                model=model,
                wavelet=wavelet,
                observed=observed,
                batch_size=batch_size,
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        last_eps_info = None
        last_sig_info = None

        for shot_indices in shot_batches:
            if tv_enabled:
                epsilon, sigma = model.predict_models()
                epsilon, sigma, last_eps_info, last_sig_info = _project_models_tv_l1(
                    model=model,
                    epsilon=epsilon,
                    sigma=sigma,
                    epsilon_radius=tv_l1_epsilon_radius,
                    sigma_radius=tv_l1_sigma_radius,
                    straight_through=True,
                )
                syn = model.forward_shots_with_models(
                    epsilon=epsilon,
                    sigma=sigma,
                    wavelet=wavelet,
                    shot_indices=shot_indices,
                    requires_grad=True,
                )
            else:
                syn = model.forward_shots(
                    wavelet=wavelet,
                    shot_indices=shot_indices,
                    requires_grad=True,
                )
            obs_batch = observed[:, shot_indices, :]
            weight = float(shot_indices.numel()) / float(model.n_shots)
            batch_loss = F.mse_loss(syn, obs_batch)
            (batch_loss * weight).backward()
            total_loss += weight * float(batch_loss.item())

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), grad_clip)
        optimizer.step()
        history.append(total_loss)

        if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
            if tv_enabled:
                print(
                    f"Joint Fourier-feature IFWI epoch {epoch + 1:4d}/{epochs}  "
                    f"loss={total_loss:.6e} "
                    f"{_format_tv_projection_info('eps_tv', last_eps_info)} "
                    f"{_format_tv_projection_info('sig_tv', last_sig_info)}"
                )
            else:
                print(
                    f"Joint Fourier-feature IFWI epoch {epoch + 1:4d}/{epochs}  "
                    f"loss={total_loss:.6e}"
                )
    return history


def main() -> None:
    device = resolve_device(DEVICE_NAME)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    output_dir_name = OUTPUT_DIR_NAME
    if output_dir_name is None:
        output_dir_name = (
            f"implicit_gpr_eps_sigma_fourier_features_shots{N_SHOTS}_nt{NT}"
            f"_m{FOURIER_MAPPING_SIZE}_s{FOURIER_SCALE:g}"
            f"_linear{INIT_TOP_EPS:g}-{INIT_BOTTOM_EPS:g}"
            f"_de{EPS_DELTA_SCALE:g}_ds{SIGMA_DELTA_SCALE:g}"
            f"_gpc{int(GRAD_PRECONDITIONER)}"
        )
        if TV_L1_PROJECTION:
            output_dir_name += "_tvproj"
    output_dir = Path("outputs") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Saving outputs to: {output_dir}")

    epsilon_true_np = make_true_epsilon_model(
        model_path=MODEL_PATH,
        air_layers=AIR_LAYERS,
        ny=NY,
        nx=NX,
    )
    ny, nx = epsilon_true_np.shape
    sigma_true_np = sigma_from_epsilon_np(epsilon_true_np)
    sigma_true_np[:AIR_LAYERS, :] = 0.0

    print(
        "Model: "
        f"path={MODEL_PATH}, shape=({ny}, {nx}), "
        f"epsilon_range=({epsilon_true_np.min():.3f}, {epsilon_true_np.max():.3f})"
    )
    print(
        "Sigma true model: "
        f"sigma=clamp({SIGMA_K:.1e} * max(eps-1,0)^{SIGMA_P:g}, "
        f"{SIGMA_MIN:.1e}, {SIGMA_MAX:.1e}), "
        f"range=({sigma_true_np.min():.3e}, {sigma_true_np.max():.3e})"
    )

    epsilon_true = torch.tensor(epsilon_true_np, device=device, dtype=torch.float32)
    sigma_true = torch.tensor(sigma_true_np, device=device, dtype=torch.float32)
    mu = torch.ones_like(epsilon_true)
    epsilon_init = make_linear_initial_epsilon(
        ny=ny,
        nx=nx,
        air_layers=AIR_LAYERS,
        top_eps=INIT_TOP_EPS,
        bottom_eps=INIT_BOTTOM_EPS,
        device=device,
        dtype=torch.float32,
    )
    sigma_init = sigma_from_epsilon(epsilon_init)
    sigma_init = torch.where(
        torch.arange(ny, device=device).view(-1, 1) < AIR_LAYERS,
        torch.zeros_like(sigma_init),
        sigma_init,
    )

    source_locations, receiver_locations = build_geometry(
        nx=nx,
        air_layers=AIR_LAYERS,
        n_shots=N_SHOTS,
        source_x_min=SOURCE_X_MIN,
        source_x_max=SOURCE_X_MAX,
        source_depth=SOURCE_DEPTH,
        receiver_depth=RECEIVER_DEPTH,
        device=device,
    )

    actual_n_shots = int(source_locations.shape[0])
    actual_n_receivers = int(receiver_locations.shape[1])
    source_x_list = source_locations[:, 0, 1].detach().cpu().tolist()
    receiver_x_range = (
        int(receiver_locations[0, 0, 1].item()),
        int(receiver_locations[0, -1, 1].item()),
    )
    print(
        "Acquisition: "
        f"shots={actual_n_shots}, receivers={actual_n_receivers}, "
        f"batch_size={min(BATCH_SIZE, actual_n_shots)}, "
        f"source_x={source_x_list}, receiver_x=[{receiver_x_range[0]}..{receiver_x_range[1]}]"
    )

    wavelet = tide.ricker(
        BASE_FREQ,
        NT,
        DT,
        peak_time=1.5 / BASE_FREQ,
        device=device,
        dtype=torch.float32,
    )

    time_start = time.time()
    observed = generate_observed_data(
        epsilon_true=epsilon_true,
        sigma_fixed=sigma_true,
        mu=mu,
        wavelet=wavelet,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        dx=DX,
        dt=DT,
        pml_width=PML_WIDTH,
        batch_size=BATCH_SIZE,
        model_gradient_sampling_interval=MODEL_GRADIENT_SAMPLING_INTERVAL,
    )
    print(f"Observed data generated in {time.time() - time_start:.2f}s")

    model = MaxwellTMFourierFeatureJointFWI(
        ny=ny,
        nx=nx,
        dx=DX,
        dt=DT,
        nt=NT,
        pml_width=PML_WIDTH,
        eps_min=EPS_MIN,
        eps_max=EPS_MAX,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        sigma_init=sigma_init,
        mu=mu,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        air_layers=AIR_LAYERS,
        fourier_mapping_size=FOURIER_MAPPING_SIZE,
        fourier_scale=FOURIER_SCALE,
        fourier_include_input=FOURIER_INCLUDE_INPUT,
        hidden_features=HIDDEN_FEATURES,
        hidden_layers=HIDDEN_LAYERS,
        activation=ACTIVATION,
        final_layer_scale=FINAL_LAYER_SCALE,
        epsilon_init=epsilon_init,
        eps_delta_scale=EPS_DELTA_SCALE,
        sigma_delta_scale=SIGMA_DELTA_SCALE,
        model_gradient_sampling_interval=MODEL_GRADIENT_SAMPLING_INTERVAL,
        output_smoothing_kernel=OUTPUT_SMOOTHING_KERNEL,
        output_smoothing_passes=OUTPUT_SMOOTHING_PASSES,
        coord_chunk_size=COORD_CHUNK_SIZE,
        compile_network=COMPILE_NETWORK,
    ).to(device)
    compiled_network = model.maybe_compile_network()

    with torch.no_grad():
        epsilon_init_np = model.epsilon_init.detach().cpu().numpy()
        sigma_init_np = model.sigma_init.detach().cpu().numpy()
        epsilon_start, sigma_start = model.predict_models()
        epsilon_start_np = epsilon_start.detach().cpu().numpy()
        sigma_start_np = sigma_start.detach().cpu().numpy()

    print(
        "Fourier-feature network: "
        f"mapping_size={FOURIER_MAPPING_SIZE}, scale={FOURIER_SCALE:.3g}, "
        f"include_input={FOURIER_INCLUDE_INPUT}, "
        f"hidden={HIDDEN_FEATURES}x{HIDDEN_LAYERS}, activation={ACTIVATION}, "
        f"eps_delta_scale={EPS_DELTA_SCALE:.3g}, "
        f"sigma_delta_scale={SIGMA_DELTA_SCALE:.3e}, "
        f"params={count_parameters(model.net):,}, "
        f"torch_compile={'on' if compiled_network else 'off'}"
    )
    print(
        "Linear initial epsilon: "
        f"top={INIT_TOP_EPS:.3f}, bottom={INIT_BOTTOM_EPS:.3f}, "
        f"std={epsilon_init_np[AIR_LAYERS:, :].std():.3f}, "
        f"range=({epsilon_init_np[AIR_LAYERS:, :].min():.3f}, "
        f"{epsilon_init_np[AIR_LAYERS:, :].max():.3f})"
    )
    print(
        "Empirical initial sigma: "
        f"std={sigma_init_np[AIR_LAYERS:, :].std():.3e}, "
        f"range=({sigma_init_np[AIR_LAYERS:, :].min():.3e}, "
        f"{sigma_init_np[AIR_LAYERS:, :].max():.3e})"
    )
    print(
        "Network start models: "
        f"final_layer_scale={FINAL_LAYER_SCALE:.3g}, "
        f"eps_range=({epsilon_start_np[AIR_LAYERS:, :].min():.3f}, "
        f"{epsilon_start_np[AIR_LAYERS:, :].max():.3f}), "
        f"sigma_range=({sigma_start_np[AIR_LAYERS:, :].min():.3e}, "
        f"{sigma_start_np[AIR_LAYERS:, :].max():.3e})"
    )

    tv_l1_epsilon_radius = None
    tv_l1_sigma_radius = None
    if TV_L1_PROJECTION:
        tv_active_mask = ~model.air_mask
        tv_spacing = (DX, DX)
        tv_l1_epsilon_radius = _resolve_tv_l1_projection_radius(
            model_tensor=epsilon_start,
            active_mask=tv_active_mask,
            absolute_radius=TV_L1_EPSILON_RADIUS,
            radius_fraction=TV_L1_EPSILON_RADIUS_FRACTION,
            spacing=tv_spacing,
            label="epsilon",
        )
        tv_l1_sigma_radius = _resolve_tv_l1_projection_radius(
            model_tensor=sigma_start,
            active_mask=tv_active_mask,
            absolute_radius=TV_L1_SIGMA_RADIUS,
            radius_fraction=TV_L1_SIGMA_RADIUS_FRACTION,
            spacing=tv_spacing,
            label="sigma",
        )
        print(
            "TV-L1 projection radii: "
            f"epsilon={tv_l1_epsilon_radius}, sigma={tv_l1_sigma_radius}, "
            f"fraction=({TV_L1_EPSILON_RADIUS_FRACTION:g}, {TV_L1_SIGMA_RADIUS_FRACTION:g})"
        )
    print(
        "Single-stage joint Fourier-feature IFWI: "
        f"epochs={EPOCHS}, lr={LR:.2e}, weight_decay={WEIGHT_DECAY:.1e}, "
        f"base_freq={BASE_FREQ/1e6:.1f}MHz"
    )

    waveform_losses = run_waveform_inversion(
        model=model,
        wavelet=wavelet,
        observed=observed,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        grad_clip=GRAD_CLIP,
        log_interval=LOG_INTERVAL,
        grad_preconditioner=GRAD_PRECONDITIONER,
        grad_precond_update_interval=GRAD_PRECOND_UPDATE_INTERVAL,
        tv_l1_projection=TV_L1_PROJECTION,
        tv_l1_epsilon_radius=tv_l1_epsilon_radius,
        tv_l1_sigma_radius=tv_l1_sigma_radius,
    )

    with torch.no_grad():
        epsilon_pred, sigma_pred = model.predict_models()
        if TV_L1_PROJECTION:
            epsilon_pred, sigma_pred, eps_proj_info, sig_proj_info = _project_models_tv_l1(
                model=model,
                epsilon=epsilon_pred,
                sigma=sigma_pred,
                epsilon_radius=tv_l1_epsilon_radius,
                sigma_radius=tv_l1_sigma_radius,
                straight_through=False,
            )
            print(
                "Final TV-L1 projection: "
                f"{_format_tv_projection_info('eps_tv', eps_proj_info)} "
                f"{_format_tv_projection_info('sig_tv', sig_proj_info)}"
            )
        predicted_ifwi = generate_observed_data(
            epsilon_true=epsilon_pred,
            sigma_fixed=sigma_pred,
            mu=mu,
            wavelet=wavelet,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            dx=DX,
            dt=DT,
            pml_width=PML_WIDTH,
            batch_size=BATCH_SIZE,
            model_gradient_sampling_interval=MODEL_GRADIENT_SAMPLING_INTERVAL,
        )

    epsilon_pred_np = epsilon_pred.detach().cpu().numpy()
    sigma_pred_np = sigma_pred.detach().cpu().numpy()
    valid_mask = np.ones_like(epsilon_true_np, dtype=bool)
    valid_mask[:AIR_LAYERS, :] = False
    rel_l2_ifwi = relative_l2(epsilon_pred_np, epsilon_true_np, valid_mask)
    rel_l2_sigma = relative_l2(sigma_pred_np, sigma_true_np, valid_mask)
    print(f"\nJoint Fourier-feature IFWI epsilon relative L2 (subsurface only): {rel_l2_ifwi:.4f}")
    print(f"Joint Fourier-feature IFWI sigma relative L2 (subsurface only): {rel_l2_sigma:.4f}")

    save_model_panel_labeled(
        panels=[
            ("True", epsilon_true_np),
            ("Linear init", epsilon_init_np),
            ("Fourier IFWI", epsilon_pred_np),
        ],
        output_path=output_dir / "epsilon_summary.jpg",
        title="2D GPR Fourier-feature implicit joint epsilon FWI",
        colorbar_label="epsilon_r",
    )
    save_model_panel_labeled(
        panels=[
            ("True", sigma_true_np),
            ("Empirical init", sigma_init_np),
            ("Fourier IFWI", sigma_pred_np),
        ],
        output_path=output_dir / "sigma_summary.jpg",
        title="2D GPR Fourier-feature implicit joint sigma FWI",
        colorbar_label="sigma",
    )

    shot_index = actual_n_shots // 2
    observed_plot = observed[:, shot_index, :].detach().cpu().numpy()
    predicted_plot = predicted_ifwi[:, shot_index, :].detach().cpu().numpy()
    save_gather_panel(
        observed=observed_plot,
        synthetic=predicted_plot,
        output_path=output_dir / "gather_summary.jpg",
        title=f"Shot {shot_index} gather",
    )

    save_loss_curve(
        losses=waveform_losses,
        stage_breaks=[],
        output_path=output_dir / "waveform_loss.jpg",
    )

    elapsed = time.time() - time_start
    np.save(output_dir / "epsilon_inverted.npy", epsilon_pred_np)
    np.save(output_dir / "sigma_inverted.npy", sigma_pred_np)
    print(f"Total elapsed time: {elapsed:.2f}s")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
