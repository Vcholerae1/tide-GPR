"""Reusable TM2D linearization contexts for multiple Hessian directions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

from .tm2d import _default_receiver_misfit, maxwelltm_hvp
from .tm2d_born_autograd import tm2d_receiver_hvp_native

ReceiverMisfit = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _tensor_fingerprint(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(tensor),
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
    )


class TM2DLinearizationContext:
    """Cache a TM2D background linearization for repeated HVP applications.

    On native CUDA with device snapshots and sampling interval 1, the first HVP
    stores the background ``Ey`` and ``curl(H)`` history. Later directions
    propagate only the tangent field and reuse that history. Other
    configurations retain the same API and fall back to independent HVP calls.
    """

    def __init__(
        self,
        epsilon: torch.Tensor,
        sigma: torch.Tensor,
        mu: torch.Tensor,
        *,
        grid_spacing: float | Sequence[float],
        dt: float,
        source_amplitude: torch.Tensor | None,
        source_location: torch.Tensor | None,
        receiver_location: torch.Tensor | None,
        observed_data: torch.Tensor,
        misfit: ReceiverMisfit | None = None,
        stencil: int = 2,
        pml_width: int | Sequence[int] = 20,
        max_vel: float | None = None,
        nt: int | None = None,
        model_gradient_sampling_interval: int = 1,
        linearize_source: bool = True,
        hessian_mode: Literal["full", "gauss_newton"] = "full",
        python_backend: bool = False,
        storage_mode: Literal["device", "cpu", "disk"] = "device",
        storage_compression: bool | str | None = None,
    ) -> None:
        if epsilon.ndim != 2:
            raise ValueError("TM2DLinearizationContext requires a 2D model.")
        if sigma.shape != epsilon.shape or mu.shape != epsilon.shape:
            raise ValueError("epsilon, sigma, and mu must have the same shape.")
        if hessian_mode not in {"full", "gauss_newton"}:
            raise ValueError("hessian_mode must be 'full' or 'gauss_newton'.")

        self.epsilon = epsilon
        self.sigma = sigma
        self.mu = mu
        self.grid_spacing = grid_spacing
        self.dt = dt
        self.source_amplitude = source_amplitude
        self.source_location = source_location
        self.receiver_location = receiver_location
        self.observed_data = observed_data
        self.misfit = _default_receiver_misfit if misfit is None else misfit
        self.stencil = stencil
        self.pml_width = pml_width
        self.max_vel = max_vel
        self.nt = nt
        self.model_gradient_sampling_interval = model_gradient_sampling_interval
        self.linearize_source = linearize_source
        self.hessian_mode = hessian_mode
        self.python_backend = python_backend
        self.storage_mode = storage_mode
        self.storage_compression = storage_compression
        self._background_cache: dict[str, Any] | None = None
        self._closed = False
        self._fingerprints = {
            "epsilon": _tensor_fingerprint(epsilon),
            "sigma": _tensor_fingerprint(sigma),
            "mu": _tensor_fingerprint(mu),
            "source_amplitude": (
                _tensor_fingerprint(source_amplitude)
                if source_amplitude is not None
                else None
            ),
            "observed_data": _tensor_fingerprint(observed_data),
        }
        self.background_builds = 0
        self.reused_directions = 0

    @property
    def can_reuse_background(self) -> bool:
        """Whether this configuration supports native snapshot reuse."""
        return (
            not self.python_backend
            and self.epsilon.device.type == "cuda"
            and self.storage_mode == "device"
            and self.model_gradient_sampling_interval in {0, 1}
        )

    @property
    def predicted_data(self) -> torch.Tensor | None:
        """Cached baseline receiver data after the first HVP."""
        if self._background_cache is None:
            return None
        return self._background_cache["predicted_data"]

    def _validate(self) -> None:
        if self._closed:
            raise RuntimeError("TM2DLinearizationContext is closed.")
        tensors = {
            "epsilon": self.epsilon,
            "sigma": self.sigma,
            "mu": self.mu,
            "source_amplitude": self.source_amplitude,
            "observed_data": self.observed_data,
        }
        for name, tensor in tensors.items():
            expected = self._fingerprints[name]
            actual = _tensor_fingerprint(tensor) if tensor is not None else None
            if actual != expected:
                raise RuntimeError(
                    f"{name} changed after linearization; create a new context."
                )

    def hvp(
        self,
        *,
        vepsilon: torch.Tensor | None = None,
        vsigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the cached linearization to one model direction."""
        self._validate()
        if vepsilon is None and vsigma is None:
            raise ValueError("At least one direction tensor must be provided.")
        for name, direction, model in (
            ("vepsilon", vepsilon, self.epsilon),
            ("vsigma", vsigma, self.sigma),
        ):
            if direction is not None and direction.shape != model.shape:
                raise ValueError(f"{name} must have shape {tuple(model.shape)}.")

        if self.can_reuse_background:
            capture = self._background_cache is None
            result = tm2d_receiver_hvp_native(
                self.epsilon,
                self.sigma,
                self.mu,
                vepsilon=vepsilon,
                vsigma=vsigma,
                grid_spacing=self.grid_spacing,
                dt=self.dt,
                source_amplitude=self.source_amplitude,
                source_location=self.source_location,
                receiver_location=self.receiver_location,
                observed_data=self.observed_data,
                misfit_fn=self.misfit,
                stencil=self.stencil,
                pml_width=self.pml_width,
                max_vel=self.max_vel,
                nt=self.nt,
                model_gradient_sampling_interval=(
                    self.model_gradient_sampling_interval
                ),
                linearize_source=self.linearize_source,
                hessian_mode=self.hessian_mode,
                storage_mode=self.storage_mode,
                storage_compression=self.storage_compression,
                background_cache=self._background_cache,
                capture_background_cache=capture,
            )
            if capture:
                hvp_epsilon, hvp_sigma, self._background_cache = result
                self.background_builds += 1
                return hvp_epsilon, hvp_sigma
            self.reused_directions += 1
            return result

        return maxwelltm_hvp(
            self.epsilon,
            self.sigma,
            self.mu,
            grid_spacing=self.grid_spacing,
            dt=self.dt,
            source_amplitude=self.source_amplitude,
            source_location=self.source_location,
            receiver_location=self.receiver_location,
            observed_data=self.observed_data,
            vepsilon=vepsilon,
            vsigma=vsigma,
            misfit=self.misfit,
            stencil=self.stencil,
            pml_width=self.pml_width,
            max_vel=self.max_vel,
            nt=self.nt,
            model_gradient_sampling_interval=self.model_gradient_sampling_interval,
            linearize_source=self.linearize_source,
            hessian_mode=self.hessian_mode,
            python_backend=self.python_backend,
            storage_mode=self.storage_mode,
            storage_compression=self.storage_compression,
        )

    def hvp_batch(
        self,
        *,
        vepsilon: torch.Tensor | None = None,
        vsigma: torch.Tensor | None = None,
        block_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply HVPs to ``K`` directions, processing them in bounded blocks."""
        self._validate()
        directions = [value for value in (vepsilon, vsigma) if value is not None]
        if not directions:
            raise ValueError("At least one direction batch must be provided.")
        if any(direction.ndim != 3 for direction in directions):
            raise ValueError("Direction batches must have shape (K, ny, nx).")
        k = int(directions[0].shape[0])
        if any(int(direction.shape[0]) != k for direction in directions):
            raise ValueError("Direction batches must have the same leading size.")
        if k < 1:
            raise ValueError("Direction batches must be non-empty.")
        if block_size is None:
            block_size = k
        if block_size < 1:
            raise ValueError("block_size must be positive.")

        epsilon_parts: list[torch.Tensor] = []
        sigma_parts: list[torch.Tensor] = []
        for block_start in range(0, k, block_size):
            block_end = min(block_start + block_size, k)
            for index in range(block_start, block_end):
                hvp_epsilon, hvp_sigma = self.hvp(
                    vepsilon=None if vepsilon is None else vepsilon[index],
                    vsigma=None if vsigma is None else vsigma[index],
                )
                epsilon_parts.append(hvp_epsilon)
                sigma_parts.append(hvp_sigma)
        return torch.stack(epsilon_parts), torch.stack(sigma_parts)

    def close(self) -> None:
        """Release cached snapshots."""
        self._background_cache = None
        self._closed = True

    def __enter__(self) -> TM2DLinearizationContext:
        self._validate()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def linearize_maxwelltm(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    **kwargs: Any,
) -> TM2DLinearizationContext:
    """Create a reusable TM2D Hessian linearization context."""
    return TM2DLinearizationContext(epsilon, sigma, mu, **kwargs)


__all__ = ["TM2DLinearizationContext", "linearize_maxwelltm"]
