from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regularization.tv_l1 import TVL1Regularizer

try:
    import example_ifwi_gpr_eps_fourier_features as base
except ImportError:
    from examples import example_ifwi_gpr_eps_fourier_features as base


# Script configuration
MODEL_PATH = base.MODEL_PATH
NX = base.NX
NY = base.NY

DX = base.DX
DT = base.DT
NT = base.NT
PML_WIDTH = base.PML_WIDTH
AIR_LAYERS = base.AIR_LAYERS

BASE_FREQ = base.BASE_FREQ
EPOCHS = base.EPOCHS
LR = base.LR
WEIGHT_DECAY = base.WEIGHT_DECAY
GRAD_CLIP = base.GRAD_CLIP
GRAD_PRECONDITIONER = True
GRAD_PRECOND_UPDATE_INTERVAL = 0
GRAD_PRECOND_SMOOTH_SIGMA = 3.0
GRAD_PRECOND_DAMPING = 5e-2
GRAD_PRECOND_POWER = 0.5
GRAD_PRECOND_CLIP_LO = 0.3
GRAD_PRECOND_CLIP_HI = 3.0
GRAD_PRECOND_BLEND = 0.7

N_SHOTS = base.N_SHOTS
SOURCE_X_MIN = base.SOURCE_X_MIN
SOURCE_X_MAX = base.SOURCE_X_MAX
SOURCE_DEPTH = base.SOURCE_DEPTH

RECEIVER_DEPTH = base.RECEIVER_DEPTH
BATCH_SIZE = base.BATCH_SIZE
MODEL_GRADIENT_SAMPLING_INTERVAL = base.MODEL_GRADIENT_SAMPLING_INTERVAL

EPS_MIN = base.EPS_MIN
EPS_MAX = base.EPS_MAX
SIGMA_MIN = base.SIGMA_MIN
SIGMA_MAX = base.SIGMA_MAX
SIGMA_K = base.SIGMA_K
SIGMA_P = base.SIGMA_P

FOURIER_MAPPING_SIZE = base.FOURIER_MAPPING_SIZE
FOURIER_SCALE = base.FOURIER_SCALE
FOURIER_INCLUDE_INPUT = base.FOURIER_INCLUDE_INPUT
HIDDEN_FEATURES = base.HIDDEN_FEATURES
HIDDEN_LAYERS = base.HIDDEN_LAYERS
ACTIVATION = base.ACTIVATION
FINAL_LAYER_SCALE = 50.0
OUTPUT_SMOOTHING_KERNEL = base.OUTPUT_SMOOTHING_KERNEL
OUTPUT_SMOOTHING_PASSES = base.OUTPUT_SMOOTHING_PASSES
COORD_CHUNK_SIZE = base.COORD_CHUNK_SIZE
COMPILE_NETWORK = base.COMPILE_NETWORK

TV_L1_EPSILON_WEIGHT = 0.0
TV_L1_SIGMA_WEIGHT = 0.0
TV_L1_REDUCTION = "mean"
TV_L1_SMOOTH_EPSILON = 0.0

LOG_INTERVAL = base.LOG_INTERVAL
SEED = base.SEED
DEVICE_NAME = base.DEVICE_NAME
OUTPUT_DIR_NAME: str | None = None


class MaxwellTMFourierFeatureDirectFWI(nn.Module):
    """Fourier-feature model that directly predicts epsilon and sigma."""

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
        self.model_gradient_sampling_interval = int(model_gradient_sampling_interval)
        self.air_layers = int(air_layers)
        self.output_smoothing_kernel = int(output_smoothing_kernel)
        self.output_smoothing_passes = int(output_smoothing_passes)
        self.coord_chunk_size = None if coord_chunk_size is None else int(coord_chunk_size)

        if self.eps_min <= 0.0 or self.eps_max <= self.eps_min:
            raise ValueError("epsilon bounds must satisfy 0 < eps_min < eps_max.")
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

        y = torch.linspace(-1.0, 1.0, self.ny, device=mu.device, dtype=mu.dtype)
        x = torch.linspace(-1.0, 1.0, self.nx, device=mu.device, dtype=mu.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        air_mask = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=mu.device)
        air_mask[:air_layers, :] = True

        self.register_buffer("coords", coords)
        self.register_buffer("mu", mu)
        self.register_buffer("source_locations", source_locations)
        self.register_buffer("receiver_locations", receiver_locations)
        self.register_buffer("air_mask", air_mask)
        self.register_buffer("epsilon_grad_preconditioner", torch.ones_like(mu))
        self.register_buffer("sigma_grad_preconditioner", torch.ones_like(mu))
        self.output_gradient_preconditioning = False

        self.net = base.FourierFeatureMLP(
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

    def predict_models(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._run_network()
        epsilon = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(raw[..., 0])
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw[..., 1])

        epsilon = self._smooth_field(epsilon)
        sigma = self._smooth_field(sigma)

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

        out = base.tide.maxwelltm(
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
    model: MaxwellTMFourierFeatureDirectFWI,
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
    shot_batches = base.make_shot_batches(model.n_shots, batch_size, model.coords.device)

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
    model: MaxwellTMFourierFeatureDirectFWI,
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


def _predict_models_without_output_preconditioner(
    model: MaxwellTMFourierFeatureDirectFWI,
) -> tuple[torch.Tensor, torch.Tensor]:
    was_enabled = model.output_gradient_preconditioning
    model.output_gradient_preconditioning = False
    try:
        return model.predict_models()
    finally:
        model.output_gradient_preconditioning = was_enabled


def run_waveform_inversion(
    *,
    model: MaxwellTMFourierFeatureDirectFWI,
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
    tv_l1_epsilon_weight: float = TV_L1_EPSILON_WEIGHT,
    tv_l1_sigma_weight: float = TV_L1_SIGMA_WEIGHT,
    tv_l1_reduction: str = TV_L1_REDUCTION,
    tv_l1_smooth_epsilon: float = TV_L1_SMOOTH_EPSILON,
) -> list[float]:
    history: list[float] = []
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=lr, weight_decay=weight_decay)
    shot_batches = base.make_shot_batches(model.n_shots, batch_size, model.coords.device)
    tv_enabled = tv_l1_epsilon_weight > 0.0 or tv_l1_sigma_weight > 0.0
    tv_spacing = (model.dx, model.dx)
    tv_active_mask = ~model.air_mask
    tv_epsilon = TVL1Regularizer(
        weight=tv_l1_epsilon_weight,
        spacing=tv_spacing,
        reduction=tv_l1_reduction,  # type: ignore[arg-type]
        epsilon=tv_l1_smooth_epsilon,
        spatial_ndim=2,
    )
    tv_sigma = TVL1Regularizer(
        weight=tv_l1_sigma_weight,
        spacing=tv_spacing,
        reduction=tv_l1_reduction,  # type: ignore[arg-type]
        epsilon=tv_l1_smooth_epsilon,
        spatial_ndim=2,
    )

    if tv_enabled:
        print(
            "TV-L1 regularization: "
            f"epsilon_weight={tv_l1_epsilon_weight:.3e}, "
            f"sigma_weight={tv_l1_sigma_weight:.3e}, "
            f"reduction={tv_l1_reduction}, smooth_epsilon={tv_l1_smooth_epsilon:.1e}, "
            "active=subsurface"
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
        total_data_loss = 0.0
        total_reg_loss = 0.0

        for shot_indices in shot_batches:
            if tv_enabled:
                epsilon, sigma = model.predict_models()
                syn = model.forward_shots_with_models(
                    epsilon=epsilon,
                    sigma=sigma,
                    wavelet=wavelet,
                    shot_indices=shot_indices,
                    requires_grad=True,
                )
                epsilon_reg, sigma_reg = _predict_models_without_output_preconditioner(model)
                reg_loss = tv_epsilon(epsilon_reg, active_mask=tv_active_mask) + tv_sigma(
                    sigma_reg,
                    active_mask=tv_active_mask,
                )
            else:
                syn = model.forward_shots(
                    wavelet=wavelet,
                    shot_indices=shot_indices,
                    requires_grad=True,
                )
                reg_loss = model.coords.new_zeros(())
            obs_batch = observed[:, shot_indices, :]
            weight = float(shot_indices.numel()) / float(model.n_shots)
            batch_loss = F.mse_loss(syn, obs_batch)
            ((batch_loss + reg_loss) * weight).backward()
            total_data_loss += weight * float(batch_loss.item())
            total_reg_loss += weight * float(reg_loss.detach().item())

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), grad_clip)
        optimizer.step()
        total_loss = total_data_loss + total_reg_loss
        history.append(total_loss)

        if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
            if tv_enabled:
                print(
                    f"Direct Fourier-feature IFWI epoch {epoch + 1:4d}/{epochs}  "
                    f"loss={total_loss:.6e} data={total_data_loss:.6e} tv={total_reg_loss:.6e}"
                )
            else:
                print(
                    f"Direct Fourier-feature IFWI epoch {epoch + 1:4d}/{epochs}  "
                    f"loss={total_loss:.6e}"
                )
    return history


def main() -> None:
    device = base.resolve_device(DEVICE_NAME)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    output_dir_name = OUTPUT_DIR_NAME
    if output_dir_name is None:
        output_dir_name = (
            f"implicit_gpr_eps_sigma_fourier_features_direct_shots{N_SHOTS}_nt{NT}"
            f"_m{FOURIER_MAPPING_SIZE}_s{FOURIER_SCALE:g}"
            f"_fls{FINAL_LAYER_SCALE:g}"
            f"_gpc{int(GRAD_PRECONDITIONER)}"
        )
    output_dir = Path("outputs") / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Saving outputs to: {output_dir}")

    epsilon_true_np = base.make_true_epsilon_model(
        model_path=MODEL_PATH,
        air_layers=AIR_LAYERS,
        ny=NY,
        nx=NX,
    )
    ny, nx = epsilon_true_np.shape
    sigma_true_np = base.sigma_from_epsilon_np(epsilon_true_np)
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

    source_locations, receiver_locations = base.build_geometry(
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

    wavelet = base.tide.ricker(
        BASE_FREQ,
        NT,
        DT,
        peak_time=1.5 / BASE_FREQ,
        device=device,
        dtype=torch.float32,
    )

    time_start = time.time()
    observed = base.generate_observed_data(
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

    model = MaxwellTMFourierFeatureDirectFWI(
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
        model_gradient_sampling_interval=MODEL_GRADIENT_SAMPLING_INTERVAL,
        output_smoothing_kernel=OUTPUT_SMOOTHING_KERNEL,
        output_smoothing_passes=OUTPUT_SMOOTHING_PASSES,
        coord_chunk_size=COORD_CHUNK_SIZE,
        compile_network=COMPILE_NETWORK,
    ).to(device)
    compiled_network = model.maybe_compile_network()

    with torch.no_grad():
        epsilon_start, sigma_start = model.predict_models()
        epsilon_start_np = epsilon_start.detach().cpu().numpy()
        sigma_start_np = sigma_start.detach().cpu().numpy()

    print(
        "Direct Fourier-feature network: "
        f"mapping_size={FOURIER_MAPPING_SIZE}, scale={FOURIER_SCALE:.3g}, "
        f"include_input={FOURIER_INCLUDE_INPUT}, "
        f"hidden={HIDDEN_FEATURES}x{HIDDEN_LAYERS}, activation={ACTIVATION}, "
        f"final_layer_scale={FINAL_LAYER_SCALE:.3g}, "
        f"params={base.count_parameters(model.net):,}, "
        f"torch_compile={'on' if compiled_network else 'off'}"
    )
    print(
        "Network start epsilon: "
        f"std={epsilon_start_np[AIR_LAYERS:, :].std():.3f}, "
        f"range=({epsilon_start_np[AIR_LAYERS:, :].min():.3f}, "
        f"{epsilon_start_np[AIR_LAYERS:, :].max():.3f})"
    )
    print(
        "Network start sigma: "
        f"std={sigma_start_np[AIR_LAYERS:, :].std():.3e}, "
        f"range=({sigma_start_np[AIR_LAYERS:, :].min():.3e}, "
        f"{sigma_start_np[AIR_LAYERS:, :].max():.3e})"
    )
    print(
        "Single-stage direct Fourier-feature IFWI: "
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
        tv_l1_epsilon_weight=TV_L1_EPSILON_WEIGHT,
        tv_l1_sigma_weight=TV_L1_SIGMA_WEIGHT,
        tv_l1_reduction=TV_L1_REDUCTION,
        tv_l1_smooth_epsilon=TV_L1_SMOOTH_EPSILON,
    )

    with torch.no_grad():
        epsilon_pred, sigma_pred = model.predict_models()
        predicted_ifwi = base.generate_observed_data(
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
    rel_l2_ifwi = base.relative_l2(epsilon_pred_np, epsilon_true_np, valid_mask)
    rel_l2_sigma = base.relative_l2(sigma_pred_np, sigma_true_np, valid_mask)
    print(f"\nDirect Fourier-feature IFWI epsilon relative L2 (subsurface only): {rel_l2_ifwi:.4f}")
    print(f"Direct Fourier-feature IFWI sigma relative L2 (subsurface only): {rel_l2_sigma:.4f}")

    base.save_model_panel_labeled(
        panels=[
            ("True", epsilon_true_np),
            ("Network start", epsilon_start_np),
            ("Direct Fourier IFWI", epsilon_pred_np),
        ],
        output_path=output_dir / "epsilon_summary.jpg",
        title="2D GPR Fourier-feature direct joint epsilon FWI",
        colorbar_label="epsilon_r",
    )
    base.save_model_panel_labeled(
        panels=[
            ("True", sigma_true_np),
            ("Network start", sigma_start_np),
            ("Direct Fourier IFWI", sigma_pred_np),
        ],
        output_path=output_dir / "sigma_summary.jpg",
        title="2D GPR Fourier-feature direct joint sigma FWI",
        colorbar_label="sigma",
    )

    shot_index = actual_n_shots // 2
    observed_plot = observed[:, shot_index, :].detach().cpu().numpy()
    predicted_plot = predicted_ifwi[:, shot_index, :].detach().cpu().numpy()
    base.save_gather_panel(
        observed=observed_plot,
        synthetic=predicted_plot,
        output_path=output_dir / "gather_summary.jpg",
        title=f"Shot {shot_index} gather",
    )

    base.save_loss_curve(
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
