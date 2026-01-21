"""Callback state and helpers for TIDE propagators."""

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from types import EllipsisType

import torch


class CallbackState:
    """State provided to user callbacks during wave propagation.

    This class encapsulates the simulation state at a given time step,
    providing convenient access to wavefields, model parameters, and
    gradients with different views (full, pml, inner).

    The three views correspond to different regions of the computational domain:
    - 'full': The entire padded domain including FD padding
    - 'pml': The model region plus PML absorbing layers
    - 'inner': Only the physical model region (excluding PML)

    Example:
        >>> def my_callback(state: CallbackState):
        ...     # Get the Ey field in the inner (physical) region
        ...     ey = state.get_wavefield("Ey", view="inner")
        ...     print(f"Step {state.step}, max |Ey| = {ey.abs().max():.6e}")
        ...
        ...     # Get the permittivity model
        ...     eps = state.get_model("epsilon", view="inner")
        ...
        ...     # During backward pass, get gradients
        ...     if state.is_backward:
        ...         grad_eps = state.get_gradient("epsilon", view="inner")
        >>>
        >>> # Use with maxwell propagator
        >>> result = maxwell.maxwelltm(..., forward_callback=my_callback)

    Attributes:
        dt: The time step size in seconds.
        step: The current time step number (0-indexed).
        nt: Total number of time steps.
        is_backward: Whether this is during backward (adjoint) propagation.
    """

    def __init__(
        self,
        dt: float,
        step: int,
        nt: int,
        wavefields: Dict[str, torch.Tensor],
        models: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        fd_pad: Optional[list[int]] = None,
        pml_width: Optional[list[int]] = None,
        is_backward: bool = False,
        grid_spacing: Optional[list[float]] = None,
    ) -> None:
        """Initialize the callback state.

        Args:
            dt: The time step size in seconds.
            step: The current time step number.
            nt: Total number of time steps.
            wavefields: A dictionary mapping wavefield names to tensors.
                For Maxwell TM: {"Ey", "Hx", "Hz", "m_Ey_x", "m_Ey_z", ...}
            models: A dictionary mapping model names to tensors.
                For Maxwell TM: {"epsilon", "sigma", "mu", "ca", "cb", "cq"}
            gradients: A dictionary mapping gradient names to tensors.
                Only available during backward pass.
            fd_pad: Padding for finite difference stencil [y0, y1, x0, x1].
                If None, assumes no padding.
            pml_width: Width of PML layers [top, bottom, left, right].
                If None, assumes no PML.
            is_backward: Whether this is during backward propagation.
            grid_spacing: Grid spacing [dy, dx] in meters.
        """
        self.dt = dt
        self.step = step
        self.nt = nt
        self.is_backward = is_backward
        self._wavefields = wavefields
        self._models = models
        self._gradients = gradients if gradients is not None else {}
        self._fd_pad = fd_pad if fd_pad is not None else [0, 0, 0, 0]
        self._pml_width = pml_width if pml_width is not None else [0, 0, 0, 0]
        self._grid_spacing = grid_spacing

        # Determine spatial ndim from padding (preferred) or model tensors.
        # Padding lists are in [d0_low, d0_high, d1_low, d1_high, ...] format.
        if fd_pad is not None and len(fd_pad) in {4, 6}:
            self._ndim = len(fd_pad) // 2
        elif pml_width is not None and len(pml_width) in {4, 6}:
            self._ndim = len(pml_width) // 2
        elif models:
            first_model = next(iter(models.values()))
            # Heuristic:
            # - 2D unbatched: [ny, nx] -> 2
            # - 2D batched:   [n_shots, ny, nx] -> 2
            # - 3D unbatched: [nz, ny, nx] -> ambiguous with 2D batched; callers
            #   should pass fd_pad/pml_width to disambiguate.
            # - 3D batched:   [n_shots, nz, ny, nx] -> 3
            if first_model.ndim == 2:
                self._ndim = 2
            elif first_model.ndim == 4:
                self._ndim = 3
            else:
                # Preserve existing behavior (Maxwell TM callbacks) as default.
                self._ndim = 2
        else:
            # Default to 2D when no other information is available.
            self._ndim = 2

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self.step * self.dt

    @property
    def progress(self) -> float:
        """Simulation progress as a fraction [0, 1]."""
        return self.step / max(self.nt - 1, 1)

    @property
    def wavefield_names(self) -> list[str]:
        """list of available wavefield names."""
        return list(self._wavefields.keys())

    @property
    def model_names(self) -> list[str]:
        """list of available model names."""
        return list(self._models.keys())

    @property
    def gradient_names(self) -> list[str]:
        """list of available gradient names."""
        return list(self._gradients.keys())

    def get_wavefield(self, name: str, view: str = "inner") -> torch.Tensor:
        """Get a wavefield tensor.

        Args:
            name: The name of the wavefield. For Maxwell TM mode:
                - "Ey": Electric field (y-component)
                - "Hx": Magnetic field (x-component)
                - "Hz": Magnetic field (z-component)
                - "m_Ey_x", "m_Ey_z", "m_Hx_z", "m_Hz_x": CPML auxiliary fields
                - During backward: "lambda_Ey", "lambda_Hx", "lambda_Hz"
            view: The part of the wavefield to return:
                - 'inner': The physical model region (default)
                - 'pml': Model region plus PML layers
                - 'full': Entire domain including FD padding

        Returns:
            The specified part of the wavefield tensor.
            Shape depends on view and whether batched: [n_shots, ny, nx] or [ny, nx]

        Raises:
            KeyError: If the wavefield name is not found.
            ValueError: If view is not valid.
        """
        if name not in self._wavefields:
            available = ", ".join(self._wavefields.keys())
            raise KeyError(f"Wavefield '{name}' not found. Available: {available}")
        return self._get_view(self._wavefields[name], view)

    def get_model(self, name: str, view: str = "inner") -> torch.Tensor:
        """Get a model parameter tensor.

        Args:
            name: The name of the model parameter. For Maxwell TM:
                - "epsilon": Relative permittivity
                - "sigma": Electrical conductivity (S/m)
                - "mu": Relative permeability
                - "ca", "cb", "cq": Update coefficients
            view: The part of the model to return:
                - 'inner': The physical model region (default)
                - 'pml': Model region plus PML layers
                - 'full': Entire domain including FD padding

        Returns:
            The specified part of the model tensor.

        Raises:
            KeyError: If the model name is not found.
            ValueError: If view is not valid.
        """
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._get_view(self._models[name], view)

    def get_gradient(self, name: str, view: str = "inner") -> torch.Tensor:
        """Get a gradient tensor (only available during backward pass).

        Args:
            name: The name of the gradient. For Maxwell TM:
                - "epsilon" or "ca": Gradient w.r.t. permittivity/Ca
                - "sigma" or "cb": Gradient w.r.t. conductivity/Cb
            view: The part of the gradient to return:
                - 'inner': The physical model region (default)
                - 'pml': Model region plus PML layers
                - 'full': Entire domain including FD padding

        Returns:
            The specified part of the gradient tensor.

        Raises:
            KeyError: If the gradient name is not found.
            ValueError: If view is not valid.
            RuntimeError: If called during forward pass (no gradients available).
        """
        if not self._gradients:
            raise RuntimeError(
                "Gradients are only available during backward propagation. "
                "Use backward_callback instead of forward_callback."
            )
        if name not in self._gradients:
            available = ", ".join(self._gradients.keys())
            raise KeyError(f"Gradient '{name}' not found. Available: {available}")
        return self._get_view(self._gradients[name], view)

    def _get_view(self, x: torch.Tensor, view: str) -> torch.Tensor:
        """Extract a view of a tensor based on the specified region.

        Args:
            x: The tensor to extract a view from.
            view: One of 'full', 'pml', or 'inner'.

        Returns:
            A view of the tensor corresponding to the specified region.
        """
        if view == "full":
            return x

        if view not in {"pml", "inner"}:
            raise ValueError(
                f"view must be 'full', 'pml', or 'inner', but got '{view}'"
            )

        spatial_ndim = self._ndim
        if spatial_ndim not in {2, 3}:
            raise ValueError(f"Unsupported spatial ndim {spatial_ndim}.")

        if view == "pml":
            starts = [self._fd_pad[2 * i] for i in range(spatial_ndim)]
            ends = [self._fd_pad[2 * i + 1] for i in range(spatial_ndim)]
        else:
            starts = [
                self._fd_pad[2 * i] + self._pml_width[2 * i]
                for i in range(spatial_ndim)
            ]
            ends = [
                self._fd_pad[2 * i + 1] + self._pml_width[2 * i + 1]
                for i in range(spatial_ndim)
            ]

        def _slice(dim_size: int, start: int, end: int) -> slice:
            stop = dim_size - end if end > 0 else None
            return slice(start, stop)

        if x.ndim == spatial_ndim:
            # Non-batched: [ny, nx] or [nz, ny, nx]
            idx = tuple(
                _slice(x.shape[i], starts[i], ends[i]) for i in range(spatial_ndim)
            )
            return x[idx]

        # Batched: [..., ny, nx] or [..., nz, ny, nx]
        idx_batched: tuple[Union["EllipsisType", slice], ...] = (
            ...,
            *(
                _slice(
                    x.shape[-spatial_ndim + i],
                    starts[i],
                    ends[i],
                )
                for i in range(spatial_ndim)
            ),
        )
        return x[idx_batched]

    def __repr__(self) -> str:
        """Return a string representation of the callback state."""
        return (
            f"CallbackState(step={self.step}/{self.nt}, "
            f"time={self.time:.2e}s, "
            f"is_backward={self.is_backward}, "
            f"wavefields={self.wavefield_names}, "
            f"models={self.model_names})"
        )


# Type alias for callback functions
Callback = Callable[[CallbackState], None]


def create_callback_state(
    dt: float,
    step: int,
    nt: int,
    wavefields: Dict[str, torch.Tensor],
    models: Dict[str, torch.Tensor],
    gradients: Optional[Dict[str, torch.Tensor]] = None,
    fd_pad: Optional[list[int]] = None,
    pml_width: Optional[list[int]] = None,
    is_backward: bool = False,
    grid_spacing: Optional[list[float]] = None,
) -> CallbackState:
    """Factory function to create a CallbackState.

    This is a convenience function that creates a CallbackState with
    the given parameters. It's equivalent to calling the CallbackState
    constructor directly.

    Args:
        dt: The time step size in seconds.
        step: The current time step number.
        nt: Total number of time steps.
        wavefields: A dictionary mapping wavefield names to tensors.
        models: A dictionary mapping model names to tensors.
        gradients: A dictionary mapping gradient names to tensors (backward only).
        fd_pad: Padding for finite difference stencil [y0, y1, x0, x1].
        pml_width: Width of PML layers [top, bottom, left, right].
        is_backward: Whether this is during backward propagation.
        grid_spacing: Grid spacing [dy, dx] in meters.

    Returns:
        A new CallbackState instance.
    """
    return CallbackState(
        dt=dt,
        step=step,
        nt=nt,
        wavefields=wavefields,
        models=models,
        gradients=gradients,
        fd_pad=fd_pad,
        pml_width=pml_width,
        is_backward=is_backward,
        grid_spacing=grid_spacing,
    )
