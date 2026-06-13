from __future__ import annotations

import time
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tide

try:
    import tinycudann as tcnn
except ImportError:  # tiny-cuda-nn is an optional CUDA extension.
    tcnn = None

# Script configuration
MODEL_PATH = "examples/OverThrust.npy"
NX: int | None = None
NY: int | None = None

DX = 0.02
DT = 4e-11
NT = 2000
PML_WIDTH = 16
AIR_LAYERS = 3

BASE_FREQ = 1e8
EPOCHS = 5000
LR = 2e-5
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 0.0

N_SHOTS = 8
SOURCE_X_MIN = 0
SOURCE_X_MAX: int | None = None
SOURCE_DEPTH: int | None = None

RECEIVER_DEPTH: int | None = None
BATCH_SIZE = 8
MODEL_GRADIENT_SAMPLING_INTERVAL = 8

EPS_MIN = 1.0
EPS_MAX = 8.0
SIGMA_GROUND = 8e-4

HIDDEN_FEATURES = 512
HIDDEN_LAYERS = 5
OMEGA0 = 15.0
ACTIVATION = "sine"
NETWORK_ARCH = "tcnn"
COMPILE_IRN = True
RANDOM_OUTPUT_INIT_SCALE = 8.0
OUTPUT_SMOOTHING_KERNEL = 1
OUTPUT_SMOOTHING_PASSES = 0

FOURIER_NUM_FREQUENCIES = 32
FOURIER_NUM_PHASES = 16
FOURIER_MAX_FREQUENCY = 32.0

TCNN_ENCODING = "frequency"
TCNN_ACTIVATION = "relu"
TCNN_NUM_FREQUENCIES = 16
TCNN_HASH_LEVELS = 16
TCNN_HASH_FEATURES_PER_LEVEL = 2
TCNN_HASH_LOG2_SIZE = 19
TCNN_HASH_BASE_RESOLUTION = 16
TCNN_HASH_PER_LEVEL_SCALE = 1.5
TCNN_NETWORK_TYPE = "CutlassMLP"
TCNN_JIT_FUSION = False

RUN_NETWORK_SPEED_BENCHMARK = False
NETWORK_SPEED_BENCHMARK_WARMUP = 10
NETWORK_SPEED_BENCHMARK_ITERS = 50
NETWORK_SPEED_BENCHMARK_BACKWARD = True

LOG_INTERVAL = 10
SEED = 2026
DEVICE_NAME = "auto"
OUTPUT_DIR_NAME: str | None = None


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _center_crop_or_pad(model: np.ndarray, target_ny: int, target_nx: int) -> np.ndarray:
    src_ny, src_nx = model.shape

    if src_ny >= target_ny:
        y0 = (src_ny - target_ny) // 2
        cropped = model[y0 : y0 + target_ny, :]
    else:
        pad_top = (target_ny - src_ny) // 2
        pad_bottom = target_ny - src_ny - pad_top
        cropped = np.pad(model, ((pad_top, pad_bottom), (0, 0)), mode="edge")

    if src_nx >= target_nx:
        x0 = (src_nx - target_nx) // 2
        cropped = cropped[:, x0 : x0 + target_nx]
    else:
        pad_left = (target_nx - src_nx) // 2
        pad_right = target_nx - src_nx - pad_left
        cropped = np.pad(cropped, ((0, 0), (pad_left, pad_right)), mode="edge")

    return cropped.astype(np.float32, copy=False)


def make_true_epsilon_model(
    model_path: str,
    air_layers: int,
    ny: int | None = None,
    nx: int | None = None,
) -> np.ndarray:
    epsilon_raw = np.load(model_path).astype(np.float32, copy=False)
    target_ny = epsilon_raw.shape[0] if ny is None else int(ny)
    target_nx = epsilon_raw.shape[1] if nx is None else int(nx)
    epsilon = _center_crop_or_pad(epsilon_raw, target_ny, target_nx)
    epsilon[:air_layers, :] = 1.0
    return epsilon


def make_fixed_sigma_model(
    epsilon: np.ndarray,
    air_layers: int,
    sigma_ground: float,
) -> np.ndarray:
    sigma = np.full_like(epsilon, sigma_ground, dtype=np.float32)
    sigma[:air_layers, :] = 0.0
    return sigma


def build_geometry(
    *,
    nx: int,
    air_layers: int,
    n_shots: int,
    source_x_min: int,
    source_x_max: int | None,
    source_depth: int | None,
    receiver_depth: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_depth = air_layers - 1 if source_depth is None else source_depth
    rec_depth = air_layers - 1 if receiver_depth is None else receiver_depth

    left = int(source_x_min)
    right = nx - 1 if source_x_max is None else int(source_x_max)
    if left < 0 or left >= nx:
        raise ValueError(f"source_x_min must be in [0, {nx - 1}], got {left}.")
    if right < left or right >= nx:
        raise ValueError(
            f"source_x_max must be in [{left}, {nx - 1}], got {right}."
        )
    if n_shots <= 0:
        raise ValueError("N_SHOTS must be positive.")

    if n_shots == 1:
        source_x = torch.tensor([(left + right) // 2], dtype=torch.long, device=device)
    else:
        source_x = torch.linspace(
            left,
            right,
            steps=n_shots,
            device=device,
            dtype=torch.float32,
        ).round().to(torch.long)
    if int(source_x[-1]) >= nx:
        raise ValueError(
            f"Shot geometry exceeds model width: last source x={int(source_x[-1])}, nx={nx}."
        )
    n_receivers = nx - 1
    receiver_x = torch.arange(n_receivers, device=device, dtype=torch.long)

    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = int(src_depth)
    source_locations[:, 0, 1] = source_x

    receiver_locations = torch.zeros(n_shots, n_receivers, 2, dtype=torch.long, device=device)
    receiver_locations[:, :, 0] = int(rec_depth)
    receiver_locations[:, :, 1] = receiver_x.view(1, -1)
    return source_locations, receiver_locations


def make_shot_batches(n_shots: int, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("BATCH_SIZE must be positive.")
    batch_size = min(batch_size, n_shots)
    indices = torch.arange(n_shots, device=device)
    return [indices[i : i + batch_size] for i in range(0, n_shots, batch_size)]


def build_fourier_basis(
    *,
    in_features: int,
    num_frequencies: int,
    num_phases: int,
    max_frequency: float,
) -> torch.Tensor:
    if in_features <= 0:
        raise ValueError("in_features must be positive.")
    if num_frequencies <= 0:
        raise ValueError("num_frequencies must be positive.")
    if num_phases <= 0:
        raise ValueError("num_phases must be positive.")
    if max_frequency <= 0:
        raise ValueError("max_frequency must be positive.")

    positions = torch.linspace(-1.0, 1.0, steps=in_features, dtype=torch.float32)
    frequencies = torch.linspace(1.0, max_frequency, steps=num_frequencies, dtype=torch.float32)
    phase_step = 2.0 * math.pi / float(num_phases)
    phases = torch.arange(num_phases, dtype=torch.float32) * phase_step

    rows = [
        torch.cos(freq * positions + phase)
        for freq in frequencies
        for phase in phases
    ]
    basis = torch.stack(rows, dim=0)
    return basis


class FourierReparamLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_frequencies: int,
        num_phases: int,
        max_frequency: float,
        bias: bool = True,
        activation: str = "sine",
        omega_0: float = 1.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.omega_0 = float(omega_0)
        self.activation = activation
        self.is_first = bool(is_first)

        basis = build_fourier_basis(
            in_features=self.in_features,
            num_frequencies=num_frequencies,
            num_phases=num_phases,
            max_frequency=max_frequency,
        )
        self.register_buffer("basis", basis)
        self.coeff = nn.Parameter(torch.empty(self.out_features, self.basis.shape[0]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.activation == "sine":
            if self.is_first:
                target_bound = 1.0 / max(self.in_features, 1)
            else:
                target_bound = np.sqrt(6.0 / max(self.in_features, 1)) / max(self.omega_0, 1e-6)
        else:
            target_bound = np.sqrt(1.0 / max(self.in_features, 1))

        coeff_bound = target_bound / max(np.sqrt(float(self.basis.shape[0])), 1.0)
        with torch.no_grad():
            self.coeff.uniform_(-coeff_bound, coeff_bound)
            if self.bias is not None:
                self.bias.zero_()

    def effective_weight(self) -> torch.Tensor:
        return self.coeff @ self.basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


class IRN(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        out_features: int = 1,
        omega_0: float = 30.0,
        activation: str = "sine",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)
        self.activation_name = activation
        self.hidden_features = int(hidden_features)
        self.in_features = int(in_features)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features, bias=bias))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features, bias=bias))
        self.out = nn.Linear(hidden_features, out_features, bias=bias)

        if activation == "relu":
            self.activation = nn.ReLU()
            self.omega_0 = 1.0
        elif activation == "tanh":
            self.activation = nn.Tanh()
            self.omega_0 = 1.0
        elif activation == "sine":
            self.activation = torch.sin
            self._init_siren_weights()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _init_siren_weights(self) -> None:
        with torch.no_grad():
            self.layers[0].weight.uniform_(
                -1.0 / self.in_features,
                1.0 / self.in_features,
            )
            for layer in self.layers[1:]:
                bound = np.sqrt(6.0 / layer.in_features) / self.omega_0
                layer.weight.uniform_(-bound, bound)
            out_bound = np.sqrt(6.0 / self.out.in_features) / self.omega_0
            self.out.weight.uniform_(-out_bound, out_bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        if self.activation_name == "sine":
            x = self.activation(self.omega_0 * self.layers[0](x))
            for layer in self.layers[1:]:
                x = self.activation(self.omega_0 * layer(x))
        else:
            x = self.activation(self.layers[0](x))
            for layer in self.layers[1:]:
                x = self.activation(layer(x))
        return self.out(x)


class FourierReparamIRN(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        out_features: int = 1,
        omega_0: float = 30.0,
        activation: str = "sine",
        bias: bool = True,
        num_frequencies: int = 32,
        num_phases: int = 16,
        max_frequency: float = 32.0,
    ) -> None:
        super().__init__()
        self.omega_0 = float(omega_0)
        self.activation_name = activation

        self.layers = nn.ModuleList()
        self.layers.append(
            FourierReparamLinear(
                in_features,
                hidden_features,
                num_frequencies=num_frequencies,
                num_phases=num_phases,
                max_frequency=max_frequency,
                bias=bias,
                activation=activation,
                omega_0=self.omega_0,
                is_first=True,
            )
        )
        for _ in range(hidden_layers - 1):
            self.layers.append(
                FourierReparamLinear(
                    hidden_features,
                    hidden_features,
                    num_frequencies=num_frequencies,
                    num_phases=num_phases,
                    max_frequency=max_frequency,
                    bias=bias,
                    activation=activation,
                    omega_0=self.omega_0,
                )
            )
        self.out = FourierReparamLinear(
            hidden_features,
            out_features,
            num_frequencies=num_frequencies,
            num_phases=num_phases,
            max_frequency=max_frequency,
            bias=bias,
            activation=activation,
            omega_0=self.omega_0,
        )

        if activation == "relu":
            self.activation = nn.ReLU()
            self.omega_0 = 1.0
        elif activation == "tanh":
            self.activation = nn.Tanh()
            self.omega_0 = 1.0
        elif activation == "sine":
            self.activation = torch.sin
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        if self.activation_name == "sine":
            x = self.activation(self.omega_0 * self.layers[0](x))
            for layer in self.layers[1:]:
                x = self.activation(self.omega_0 * layer(x))
        else:
            x = self.activation(self.layers[0](x))
            for layer in self.layers[1:]:
                x = self.activation(layer(x))
        return self.out(x)


def _tcnn_activation_name(activation: str) -> str:
    name = activation.lower()
    if name == "relu":
        return "ReLU"
    if name == "gelu":
        return "ReLU"
    if name == "silu":
        return "SiLU"
    if name == "tanh":
        return "Tanh"
    if name == "sine":
        raise ValueError(
            "Sine activation is not supported by this tinycudann build. "
            "Use TCNN_ACTIVATION='relu', 'silu', or 'tanh'."
        )
    raise ValueError(f"Unsupported tcnn activation: {activation}")


def _make_tcnn_encoding_config(encoding: str) -> dict[str, int | float | str]:
    name = encoding.lower()
    if name in {"identity", "none"}:
        return {
            "otype": "Identity",
        }
    if name in {"frequency", "fourier"}:
        return {
            "otype": "Frequency",
            "n_frequencies": TCNN_NUM_FREQUENCIES,
        }
    if name in {"hashgrid", "hash", "grid"}:
        return {
            "otype": "HashGrid",
            "n_levels": TCNN_HASH_LEVELS,
            "n_features_per_level": TCNN_HASH_FEATURES_PER_LEVEL,
            "log2_hashmap_size": TCNN_HASH_LOG2_SIZE,
            "base_resolution": TCNN_HASH_BASE_RESOLUTION,
            "per_level_scale": TCNN_HASH_PER_LEVEL_SCALE,
        }
    raise ValueError(f"Unsupported tcnn encoding: {encoding}")


class TCNNIRN(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 2,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        out_features: int = 1,
        activation: str = "sine",
        encoding: str = "frequency",
        network_type: str = "FullyFusedMLP",
        jit_fusion: bool = True,
    ) -> None:
        super().__init__()
        if tcnn is None:
            raise ImportError(
                "tinycudann is required for network_arch='tcnn'. Install the "
                "tiny-cuda-nn PyTorch bindings first."
            )
        if torch.cuda.is_available() is False:
            raise RuntimeError("tinycudann requires a CUDA-capable PyTorch runtime.")

        network_type = str(network_type)
        if network_type == "FullyFusedMLP" and hidden_features not in {16, 32, 64, 128}:
            raise ValueError(
                "TCNN FullyFusedMLP requires hidden_features in {16, 32, 64, 128}; "
                f"got {hidden_features}. Use TCNN_NETWORK_TYPE='CutlassMLP' for "
                "larger hidden layers."
            )

        self.output_scale = 1.0
        encoding_config = _make_tcnn_encoding_config(encoding)
        network_config = {
            "otype": network_type,
            "activation": _tcnn_activation_name(activation),
            "output_activation": "None",
            "n_neurons": int(hidden_features),
            "n_hidden_layers": int(hidden_layers),
        }
        self.model = tcnn.NetworkWithInputEncoding(
            int(in_features),
            int(out_features),
            encoding_config,
            network_config,
        )
        if bool(jit_fusion) and hasattr(tcnn, "supports_jit_fusion"):
            self.model.jit_fusion = bool(tcnn.supports_jit_fusion())

    def set_output_scale(self, scale: float) -> None:
        self.output_scale = float(scale)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.model(coords).float() * self.output_scale


class MaxwellTMImplicitEpsilonFWI(nn.Module):
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
        sigma_fixed: torch.Tensor,
        mu: torch.Tensor,
        source_locations: torch.Tensor,
        receiver_locations: torch.Tensor,
        air_layers: int,
        hidden_features: int,
        hidden_layers: int,
        omega_0: float,
        activation: str,
        network_arch: str,
        fourier_num_frequencies: int,
        fourier_num_phases: int,
        fourier_max_frequency: float,
        tcnn_encoding: str,
        tcnn_activation: str,
        tcnn_network_type: str,
        tcnn_jit_fusion: bool,
        model_gradient_sampling_interval: int,
        random_output_init_scale: float = 1.0,
        output_smoothing_kernel: int = 1,
        output_smoothing_passes: int = 0,
        compile_irn: bool = False,
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
        self.model_gradient_sampling_interval = int(model_gradient_sampling_interval)
        self.air_layers = int(air_layers)
        self.output_smoothing_kernel = int(output_smoothing_kernel)
        self.output_smoothing_passes = int(output_smoothing_passes)
        if self.output_smoothing_kernel < 1:
            raise ValueError("output_smoothing_kernel must be >= 1.")
        if self.output_smoothing_kernel % 2 == 0:
            raise ValueError("output_smoothing_kernel must be odd.")
        if self.output_smoothing_passes < 0:
            raise ValueError("output_smoothing_passes must be >= 0.")

        x = torch.arange(self.nx, device=sigma_fixed.device, dtype=sigma_fixed.dtype) * self.dx
        y = torch.arange(self.ny, device=sigma_fixed.device, dtype=sigma_fixed.dtype) * self.dx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        x_unit = torch.linspace(0.0, 1.0, self.nx, device=sigma_fixed.device, dtype=sigma_fixed.dtype)
        y_unit = torch.linspace(0.0, 1.0, self.ny, device=sigma_fixed.device, dtype=sigma_fixed.dtype)
        yy_unit, xx_unit = torch.meshgrid(y_unit, x_unit, indexing="ij")
        coords_unit = torch.stack([xx_unit, yy_unit], dim=-1).reshape(-1, 2)
        air_mask = torch.zeros(self.ny, self.nx, dtype=torch.bool, device=sigma_fixed.device)
        air_mask[:air_layers, :] = True

        self.register_buffer("coords", coords)
        self.register_buffer("coords_unit", coords_unit)
        self.register_buffer("sigma_fixed", sigma_fixed)
        self.register_buffer("mu", mu)
        self.register_buffer("source_locations", source_locations)
        self.register_buffer("receiver_locations", receiver_locations)
        self.register_buffer("air_mask", air_mask)

        network_arch = network_arch.lower()
        if network_arch == "ifwi":
            self.irn = IRN(
                in_features=2,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                out_features=1,
                omega_0=omega_0,
                activation=activation,
            )
        elif network_arch in {"fr_ifwi", "fr-ifwi", "fourier"}:
            self.irn = FourierReparamIRN(
                in_features=2,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                out_features=1,
                omega_0=omega_0,
                activation=activation,
                num_frequencies=fourier_num_frequencies,
                num_phases=fourier_num_phases,
                max_frequency=fourier_max_frequency,
            )
        elif network_arch in {"tcnn", "tcnn_frequency", "tcnn_fourier", "tcnn_hashgrid", "tcnn_identity"}:
            if network_arch in {"tcnn_frequency", "tcnn_fourier"}:
                tcnn_encoding = "frequency"
            elif network_arch == "tcnn_hashgrid":
                tcnn_encoding = "hashgrid"
            elif network_arch == "tcnn_identity":
                tcnn_encoding = "identity"
            self.irn = TCNNIRN(
                in_features=2,
                hidden_features=hidden_features,
                hidden_layers=hidden_layers,
                out_features=1,
                activation=tcnn_activation,
                encoding=tcnn_encoding,
                network_type=tcnn_network_type,
                jit_fusion=tcnn_jit_fusion,
            )
        else:
            raise ValueError(f"Unsupported network_arch: {network_arch}")
        self.network_arch = network_arch
        self.tcnn_encoding = tcnn_encoding
        self.tcnn_activation = tcnn_activation
        self.tcnn_network_type = tcnn_network_type
        self.random_output_init_scale = float(random_output_init_scale)
        self._scale_output_layer_init(self.random_output_init_scale)
        self.compile_irn_requested = bool(compile_irn)
        self._compiled_irn = None

    def _scale_output_layer_init(self, scale: float) -> None:
        if scale <= 0.0:
            raise ValueError("random_output_init_scale must be positive.")
        if scale == 1.0:
            return

        with torch.no_grad():
            out_layer = getattr(self.irn, "out", None)
            if out_layer is not None and hasattr(out_layer, "weight"):
                out_layer.weight.mul_(scale)
            elif out_layer is not None and hasattr(out_layer, "coeff"):
                out_layer.coeff.mul_(scale)
            elif hasattr(self.irn, "set_output_scale"):
                self.irn.set_output_scale(scale)
            else:
                raise TypeError(f"Unsupported IRN type: {type(self.irn)!r}")

    def _smooth_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        if self.output_smoothing_kernel == 1 or self.output_smoothing_passes == 0:
            return epsilon
        if self.air_layers >= self.ny:
            return epsilon

        kernel = self.output_smoothing_kernel
        pad = kernel // 2
        subsurface = epsilon[self.air_layers :, :].unsqueeze(0).unsqueeze(0)
        for _ in range(self.output_smoothing_passes):
            subsurface = F.avg_pool2d(
                F.pad(subsurface, (pad, pad, pad, pad), mode="replicate"),
                kernel_size=kernel,
                stride=1,
            )
        return torch.cat([epsilon[: self.air_layers, :], subsurface[0, 0]], dim=0)

    def maybe_compile_irn(self) -> bool:
        if self.network_arch.startswith("tcnn"):
            return False
        if not self.compile_irn_requested or not hasattr(torch, "compile"):
            return False
        if self._compiled_irn is not None:
            return True
        self._compiled_irn = torch.compile(self.irn)
        return True

    @property
    def n_shots(self) -> int:
        return int(self.source_locations.shape[0])

    def _run_irn(self) -> torch.Tensor:
        irn = self.irn if self._compiled_irn is None else self._compiled_irn
        coords = self.coords_unit if self.network_arch.startswith("tcnn") else self.coords
        raw = irn(coords).reshape(self.ny, self.nx)
        return raw

    def predict_epsilon(self) -> torch.Tensor:
        raw = self._run_irn()
        epsilon = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(raw)
        epsilon = self._smooth_epsilon(epsilon)
        epsilon = torch.where(self.air_mask, torch.ones_like(epsilon), epsilon)
        return epsilon

    def forward_shots(
        self,
        *,
        wavelet: torch.Tensor,
        shot_indices: torch.Tensor,
        requires_grad: bool,
    ) -> torch.Tensor:
        epsilon = self.predict_epsilon()
        batch_size = int(shot_indices.numel())
        src_amp = wavelet.view(1, 1, self.nt).expand(batch_size, 1, self.nt).contiguous()
        src_loc = self.source_locations[shot_indices]
        rec_loc = self.receiver_locations[shot_indices]

        out = tide.maxwelltm(
            epsilon=epsilon,
            sigma=self.sigma_fixed,
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


def forward_fixed_model(
    *,
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    wavelet: torch.Tensor,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    shot_indices: torch.Tensor,
    dx: float,
    dt: float,
    pml_width: int,
    model_gradient_sampling_interval: int,
    requires_grad: bool,
) -> torch.Tensor:
    batch_size = int(shot_indices.numel())
    nt = int(wavelet.shape[0])
    src_amp = wavelet.view(1, 1, nt).expand(batch_size, 1, nt).contiguous()
    src_loc = source_locations[shot_indices]
    rec_loc = receiver_locations[shot_indices]

    out = tide.maxwelltm(
        epsilon=epsilon if requires_grad else epsilon.detach(),
        sigma=sigma if requires_grad else sigma.detach(),
        mu=mu if requires_grad else mu.detach(),
        grid_spacing=dx,
        dt=dt,
        source_amplitude=src_amp,
        source_location=src_loc,
        receiver_location=rec_loc,
        pml_width=pml_width,
        save_snapshots=requires_grad,
        model_gradient_sampling_interval=model_gradient_sampling_interval if requires_grad else 1,
        storage_mode="auto",
        storage_compression="bf16" if epsilon.device.type == "cuda" else False,
    )
    return out[-1]


@torch.no_grad()
def generate_observed_data(
    *,
    epsilon_true: torch.Tensor,
    sigma_fixed: torch.Tensor,
    mu: torch.Tensor,
    wavelet: torch.Tensor,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    dx: float,
    dt: float,
    pml_width: int,
    batch_size: int,
    model_gradient_sampling_interval: int,
) -> torch.Tensor:
    shot_batches = make_shot_batches(source_locations.shape[0], batch_size, epsilon_true.device)
    gathers = []
    for shot_indices in shot_batches:
        gathers.append(
            forward_fixed_model(
                epsilon=epsilon_true,
                sigma=sigma_fixed,
                mu=mu,
                wavelet=wavelet,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                shot_indices=shot_indices,
                dx=dx,
                dt=dt,
                pml_width=pml_width,
                model_gradient_sampling_interval=model_gradient_sampling_interval,
                requires_grad=False,
            )
        )
    return torch.cat(gathers, dim=1)


def run_waveform_inversion(
    *,
    model: MaxwellTMImplicitEpsilonFWI,
    wavelet: torch.Tensor,
    observed: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    grad_clip: float,
    log_interval: int,
) -> list[float]:
    history: list[float] = []
    optimizer = torch.optim.Adam(model.irn.parameters(), lr=lr)
    shot_batches = make_shot_batches(model.n_shots, batch_size, model.coords.device)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for shot_indices in shot_batches:
            syn = model.forward_shots(wavelet=wavelet, shot_indices=shot_indices, requires_grad=True)
            obs_batch = observed[:, shot_indices, :]
            weight = float(shot_indices.numel()) / float(model.n_shots)
            batch_loss = F.mse_loss(syn, obs_batch)
            (batch_loss * weight).backward()
            total_loss += weight * float(batch_loss.item())

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.irn.parameters(), grad_clip)
        optimizer.step()
        history.append(total_loss)

        if (epoch + 1) == 1 or (epoch + 1) % log_interval == 0 or (epoch + 1) == epochs:
            print(
                f"IFWI epoch {epoch + 1:4d}/{epochs}  loss={total_loss:.6e}"
            )
    return history


def relative_l2(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    num = np.linalg.norm((pred - true)[mask].ravel())
    den = np.linalg.norm(true[mask].ravel())
    return float(num / max(den, 1e-12))


def save_model_panel(
    *,
    panels: list[tuple[str, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    vmin = float(min(float(data.min()) for _, data in panels))
    vmax = float(max(float(data.max()) for _, data in panels))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, data) in zip(axes, panels, strict=True):
        im = ax.imshow(data, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="epsilon_r")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_gather_panel(
    *,
    observed: np.ndarray,
    synthetic: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    amp = max(
        np.percentile(np.abs(observed), 99.0),
        np.percentile(np.abs(synthetic), 99.0),
        1e-12,
    )
    residual = synthetic - observed
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for ax, data, name in zip(
        axes,
        [observed, synthetic, residual],
        ["Observed", "Predicted", "Residual"],
        strict=True,
    ):
        im = ax.imshow(data, aspect="auto", cmap="seismic", vmin=-amp, vmax=amp)
        ax.set_title(name)
        ax.set_xlabel("receiver")
        ax.set_ylabel("time sample")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_loss_curve(losses: list[float], stage_breaks: list[int], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(losses, lw=1.5)
    for idx in stage_breaks:
        ax.axvline(idx, color="gray", ls="--", lw=0.8)
    ax.set_title("Waveform Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_predict_epsilon(
    model: MaxwellTMImplicitEpsilonFWI,
    *,
    warmup: int,
    iterations: int,
    backward: bool,
) -> float:
    device = model.coords.device
    model.train(backward)

    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        epsilon = model.predict_epsilon()
        if backward:
            epsilon.mean().backward()
    _synchronize(device)

    start = time.perf_counter()
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        epsilon = model.predict_epsilon()
        if backward:
            epsilon.mean().backward()
    _synchronize(device)
    return (time.perf_counter() - start) / float(iterations)


def benchmark_network_predict_epsilon(
    *,
    ny: int,
    nx: int,
    device: torch.device,
    hidden_features: int,
    hidden_layers: int,
    activation: str,
    omega_0: float,
    warmup: int = NETWORK_SPEED_BENCHMARK_WARMUP,
    iterations: int = NETWORK_SPEED_BENCHMARK_ITERS,
    backward: bool = NETWORK_SPEED_BENCHMARK_BACKWARD,
) -> None:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if warmup < 0:
        raise ValueError("warmup must be non-negative.")

    sigma_fixed = torch.zeros((ny, nx), device=device, dtype=torch.float32)
    mu = torch.ones_like(sigma_fixed)
    source_locations = torch.zeros((1, 1, 2), dtype=torch.long, device=device)
    receiver_locations = torch.zeros((1, 1, 2), dtype=torch.long, device=device)

    archs = ["ifwi"]
    if tcnn is None:
        print("Network benchmark: tinycudann is not installed; skipping tcnn timing.")
    elif device.type != "cuda":
        print("Network benchmark: tcnn requires CUDA; skipping tcnn timing.")
    else:
        archs.append("tcnn")

    timings: dict[str, float] = {}
    for arch in archs:
        model = MaxwellTMImplicitEpsilonFWI(
            ny=ny,
            nx=nx,
            dx=DX,
            dt=DT,
            nt=NT,
            pml_width=PML_WIDTH,
            eps_min=EPS_MIN,
            eps_max=EPS_MAX,
            sigma_fixed=sigma_fixed,
            mu=mu,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            air_layers=AIR_LAYERS,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=omega_0,
            activation=activation,
            network_arch=arch,
            fourier_num_frequencies=FOURIER_NUM_FREQUENCIES,
            fourier_num_phases=FOURIER_NUM_PHASES,
            fourier_max_frequency=FOURIER_MAX_FREQUENCY,
            tcnn_encoding=TCNN_ENCODING,
            tcnn_activation=TCNN_ACTIVATION,
            tcnn_network_type=TCNN_NETWORK_TYPE,
            tcnn_jit_fusion=TCNN_JIT_FUSION,
            model_gradient_sampling_interval=MODEL_GRADIENT_SAMPLING_INTERVAL,
            random_output_init_scale=RANDOM_OUTPUT_INIT_SCALE,
            output_smoothing_kernel=OUTPUT_SMOOTHING_KERNEL,
            output_smoothing_passes=OUTPUT_SMOOTHING_PASSES,
            compile_irn=COMPILE_IRN and arch == "ifwi",
        ).to(device)
        model.maybe_compile_irn()
        timings[arch] = _time_predict_epsilon(
            model,
            warmup=warmup,
            iterations=iterations,
            backward=backward,
        )

    print(
        "Network predict_epsilon benchmark "
        f"({ny}x{nx}, hidden={hidden_features}x{hidden_layers}, "
        f"{'forward+backward' if backward else 'forward only'}):"
    )
    for arch, seconds in timings.items():
        print(f"  {arch:>4}: {seconds * 1000.0:.3f} ms/iter")
    if "ifwi" in timings and "tcnn" in timings:
        speedup = timings["ifwi"] / timings["tcnn"]
        print(f"  speedup: {speedup:.2f}x vs ifwi")


def main() -> None:
    device = resolve_device(DEVICE_NAME)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    output_dir_name = OUTPUT_DIR_NAME
    if output_dir_name is None:
        output_dir_name = (
            f"implicit_gpr_eps_{NETWORK_ARCH}_noinit_random_shots{N_SHOTS}_nt{NT}"
        )
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
    sigma_fixed_np = make_fixed_sigma_model(
        epsilon_true_np,
        air_layers=AIR_LAYERS,
        sigma_ground=SIGMA_GROUND,
    )

    print(
        "Model: "
        f"path={MODEL_PATH}, shape=({ny}, {nx}), "
        f"epsilon_range=({epsilon_true_np.min():.3f}, {epsilon_true_np.max():.3f})"
    )

    epsilon_true = torch.tensor(epsilon_true_np, device=device, dtype=torch.float32)
    sigma_fixed = torch.tensor(sigma_fixed_np, device=device, dtype=torch.float32)
    mu = torch.ones_like(epsilon_true)

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

    if RUN_NETWORK_SPEED_BENCHMARK:
        benchmark_network_predict_epsilon(
            ny=ny,
            nx=nx,
            device=device,
            hidden_features=HIDDEN_FEATURES,
            hidden_layers=HIDDEN_LAYERS,
            activation=ACTIVATION,
            omega_0=OMEGA0,
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
    observed_raw = generate_observed_data(
        epsilon_true=epsilon_true,
        sigma_fixed=sigma_fixed,
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
    observed = observed_raw
    print(f"Observed data generated in {time.time() - time_start:.2f}s")

    model = MaxwellTMImplicitEpsilonFWI(
        ny=ny,
        nx=nx,
        dx=DX,
        dt=DT,
        nt=NT,
        pml_width=PML_WIDTH,
        eps_min=EPS_MIN,
        eps_max=EPS_MAX,
        sigma_fixed=sigma_fixed,
        mu=mu,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        air_layers=AIR_LAYERS,
        hidden_features=HIDDEN_FEATURES,
        hidden_layers=HIDDEN_LAYERS,
        omega_0=OMEGA0,
        activation=ACTIVATION,
        network_arch=NETWORK_ARCH,
        fourier_num_frequencies=FOURIER_NUM_FREQUENCIES,
        fourier_num_phases=FOURIER_NUM_PHASES,
        fourier_max_frequency=FOURIER_MAX_FREQUENCY,
        tcnn_encoding=TCNN_ENCODING,
        tcnn_activation=TCNN_ACTIVATION,
        tcnn_network_type=TCNN_NETWORK_TYPE,
        tcnn_jit_fusion=TCNN_JIT_FUSION,
        model_gradient_sampling_interval=MODEL_GRADIENT_SAMPLING_INTERVAL,
        random_output_init_scale=RANDOM_OUTPUT_INIT_SCALE,
        output_smoothing_kernel=OUTPUT_SMOOTHING_KERNEL,
        output_smoothing_passes=OUTPUT_SMOOTHING_PASSES,
        compile_irn=COMPILE_IRN,
    ).to(device)
    compiled_irn = model.maybe_compile_irn()

    with torch.no_grad():
        epsilon_init_np = model.predict_epsilon().detach().cpu().numpy()

    print(
        "Random no-init prior: "
        f"output_init_scale={RANDOM_OUTPUT_INIT_SCALE:.2f}, "
        f"initial_std={epsilon_init_np[AIR_LAYERS:, :].std():.3f}, "
        f"initial_range=({epsilon_init_np[AIR_LAYERS:, :].min():.3f}, "
        f"{epsilon_init_np[AIR_LAYERS:, :].max():.3f})"
    )
    print(
        "Single-stage implicit inversion (random network initialization): "
        f"epochs={EPOCHS}, lr={LR:.2e}, "
        f"arch={model.network_arch}, "
        f"hidden={HIDDEN_FEATURES}x{HIDDEN_LAYERS}, "
        f"output_smoothing={OUTPUT_SMOOTHING_KERNEL}x{OUTPUT_SMOOTHING_PASSES}, "
        f"base_freq={BASE_FREQ/1e6:.1f}MHz, "
        f"torch_compile={'on' if compiled_irn else 'off'}"
    )
    if model.network_arch != "ifwi":
        if model.network_arch.startswith("tcnn"):
            print(
                "TCNN network: "
                f"encoding={model.tcnn_encoding}, "
                f"activation={model.tcnn_activation}, "
                f"network_type={model.tcnn_network_type}, "
                f"jit_fusion={'on' if TCNN_JIT_FUSION else 'off'}"
            )
        else:
            print(
                "Fourier reparametrization: "
                f"num_frequencies={FOURIER_NUM_FREQUENCIES}, "
                f"num_phases={FOURIER_NUM_PHASES}, "
                f"max_frequency={FOURIER_MAX_FREQUENCY:.1f}"
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
    )

    with torch.no_grad():
        epsilon_pred = model.predict_epsilon()
        predicted_ifwi = generate_observed_data(
            epsilon_true=epsilon_pred,
            sigma_fixed=sigma_fixed,
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
    valid_mask = np.ones_like(epsilon_true_np, dtype=bool)
    valid_mask[:AIR_LAYERS, :] = False
    rel_l2_ifwi = relative_l2(epsilon_pred_np, epsilon_true_np, valid_mask)
    print(f"\nIFWI epsilon relative L2 (subsurface only): {rel_l2_ifwi:.4f}")

    predicted_final = predicted_ifwi
    panel_title = "2D GPR implicit epsilon FWI (no-init random)"
    panels: list[tuple[str, np.ndarray]] = [
        ("True", epsilon_true_np),
        ("Random init", epsilon_init_np),
        ("IFWI", epsilon_pred_np),
    ]

    save_model_panel(
        panels=panels,
        output_path=output_dir / "epsilon_summary.jpg",
        title=panel_title,
    )

    shot_index = actual_n_shots // 2
    observed_plot = observed[:, shot_index, :].detach().cpu().numpy()
    predicted_plot = predicted_final[:, shot_index, :].detach().cpu().numpy()
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
    print(f"Total elapsed time: {elapsed:.2f}s")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
