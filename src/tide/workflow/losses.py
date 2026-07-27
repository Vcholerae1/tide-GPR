"""Receiver-data loss helpers for shot-batched workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .distributed import local_shot_positions
from .shots import infer_receiver_shot_dim, index_shots

LossNormalization = Literal["batch", "all", "sum"]
_ShotLossFn = Callable[[torch.Tensor], torch.Tensor]
_ZeroGradFn = Callable[[], None]
_AfterBackwardFn = Callable[[torch.Tensor, torch.Tensor], None]


def take_receiver_batch(
    receiver: torch.Tensor,
    shot_indices: torch.Tensor,
    *,
    shot_dim: int | None = None,
) -> torch.Tensor:
    """Select receiver data for one shot mini-batch."""

    dim = infer_receiver_shot_dim(receiver) if shot_dim is None else shot_dim
    return index_shots(receiver, shot_indices, shot_dim=dim)


def take_receiver_shard_batch(
    receiver_shard: torch.Tensor,
    global_shot_indices: torch.Tensor,
    local_shot_indices: torch.Tensor,
    *,
    shot_dim: int | None = None,
) -> torch.Tensor:
    """Select global shot ids from a receiver tensor that stores one local shard."""

    positions = local_shot_positions(global_shot_indices, local_shot_indices)
    dim = infer_receiver_shot_dim(receiver_shard) if shot_dim is None else shot_dim
    return index_shots(receiver_shard, positions, shot_dim=dim)


def receiver_mse_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    shot_indices: torch.Tensor,
    *,
    shot_dim: int | None = None,
    normalization: LossNormalization = "batch",
) -> torch.Tensor:
    """Return MSE between predicted receivers and the matching observed shots."""

    target = take_receiver_batch(observed, shot_indices, shot_dim=shot_dim)
    if normalization == "batch":
        return F.mse_loss(predicted, target)
    residual = predicted - target
    if normalization == "all":
        return residual.square().sum() / observed.numel()
    if normalization == "sum":
        return residual.square().sum()
    raise ValueError("normalization must be 'batch', 'all', or 'sum'.")


def receiver_sinkhorn_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    shot_indices: torch.Tensor,
    *,
    dt: float = 1.0,
    sparse_sampling: int = 1,
    p: int = 1,
    blur: float = 1e-2,
    scaling: float = 0.5,
    shot_dim: int | None = None,
    time_dim: int = 0,
) -> torch.Tensor:
    """Trace-wise Sinkhorn loss on ``(time, amplitude)`` point clouds.

    This follows ADFWI's Wasserstein-Sinkhorn misfit: each receiver trace is
    represented by points whose coordinates are sample time and amplitude.
    ``geomloss`` is imported lazily and is available through Tide's
    ``experiments`` optional dependency.
    """

    target = take_receiver_batch(observed, shot_indices, shot_dim=shot_dim)
    return _receiver_sinkhorn_loss(
        predicted,
        target,
        dt=dt,
        sparse_sampling=sparse_sampling,
        p=p,
        blur=blur,
        scaling=scaling,
        time_dim=time_dim,
    )


def receiver_sinkhorn_loss_shard(
    predicted: torch.Tensor,
    observed_shard: torch.Tensor,
    global_shot_indices: torch.Tensor,
    local_shot_indices: torch.Tensor,
    *,
    dt: float = 1.0,
    sparse_sampling: int = 1,
    p: int = 1,
    blur: float = 1e-2,
    scaling: float = 0.5,
    shot_dim: int | None = None,
    time_dim: int = 0,
) -> torch.Tensor:
    """Shard-aware form of :func:`receiver_sinkhorn_loss`."""

    target = take_receiver_shard_batch(
        observed_shard,
        global_shot_indices,
        local_shot_indices,
        shot_dim=shot_dim,
    )
    return _receiver_sinkhorn_loss(
        predicted,
        target,
        dt=dt,
        sparse_sampling=sparse_sampling,
        p=p,
        blur=blur,
        scaling=scaling,
        time_dim=time_dim,
    )


def receiver_gsot_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    shot_indices: torch.Tensor,
    *,
    dt: float = 1.0,
    sparse_sampling: int = 1,
    p: int = 2,
    max_time_shift: float = 1.0,
    observed_energy_weighting: bool = True,
    shot_dim: int | None = None,
    time_dim: int = 0,
) -> torch.Tensor:
    """Return a trace-wise graph-space optimal transport loss.

    Each trace is represented by ``(time, amplitude)`` points. A detached CPU
    copy is used to solve the exact linear-sum assignment problem, then the
    selected permutation is applied on the original tensors so gradients flow
    through the waveform amplitudes. This is a reference implementation, not
    a high-performance GPU assignment solver.
    """

    target = take_receiver_batch(observed, shot_indices, shot_dim=shot_dim)
    return _receiver_gsot_loss(
        predicted,
        target,
        dt=dt,
        sparse_sampling=sparse_sampling,
        p=p,
        max_time_shift=max_time_shift,
        observed_energy_weighting=observed_energy_weighting,
        time_dim=time_dim,
    )


def receiver_gsot_loss_shard(
    predicted: torch.Tensor,
    observed_shard: torch.Tensor,
    global_shot_indices: torch.Tensor,
    local_shot_indices: torch.Tensor,
    *,
    dt: float = 1.0,
    sparse_sampling: int = 1,
    p: int = 2,
    max_time_shift: float = 1.0,
    observed_energy_weighting: bool = True,
    shot_dim: int | None = None,
    time_dim: int = 0,
) -> torch.Tensor:
    """Shard-aware form of :func:`receiver_gsot_loss`."""

    target = take_receiver_shard_batch(
        observed_shard,
        global_shot_indices,
        local_shot_indices,
        shot_dim=shot_dim,
    )
    return _receiver_gsot_loss(
        predicted,
        target,
        dt=dt,
        sparse_sampling=sparse_sampling,
        p=p,
        max_time_shift=max_time_shift,
        observed_energy_weighting=observed_energy_weighting,
        time_dim=time_dim,
    )


def _receiver_gsot_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    dt: float,
    sparse_sampling: int,
    p: int,
    max_time_shift: float,
    observed_energy_weighting: bool,
    time_dim: int,
) -> torch.Tensor:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if sparse_sampling <= 0:
        raise ValueError("sparse_sampling must be positive.")
    if p not in (1, 2):
        raise ValueError("p must be 1 or 2.")
    if max_time_shift <= 0:
        raise ValueError("max_time_shift must be positive.")
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and selected observed data must match, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    if predicted.device != target.device:
        raise ValueError("predicted and selected observed data must share a device.")

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise ImportError("receiver_gsot_loss requires scipy.") from exc

    time_dim = time_dim % predicted.ndim
    predicted_traces = predicted.movedim(time_dim, -1)[..., ::sparse_sampling]
    target_traces = target.movedim(time_dim, -1)[..., ::sparse_sampling]
    nt = predicted_traces.shape[-1]
    predicted_traces = predicted_traces.reshape(-1, nt)
    target_traces = target_traces.reshape(-1, nt)
    if nt == 0:
        raise ValueError("GSOT requires at least one time sample.")

    # Transfer all traces once. Building each assignment cost on CPU avoids an
    # O(n_traces * nt**2) CUDA allocation and, importantly, avoids one device
    # synchronization per trace.
    predicted_numpy = predicted_traces.detach().float().cpu().numpy()
    target_numpy = target_traces.detach().float().cpu().numpy()
    active = (np.abs(predicted_numpy).sum(axis=1) != 0) | (
        np.abs(target_numpy).sum(axis=1) != 0
    )
    assignments = np.broadcast_to(
        np.arange(nt, dtype=np.int64), predicted_numpy.shape
    ).copy()
    coordinate_times = np.arange(nt, dtype=np.float64) * dt * sparse_sampling
    time_cost = np.abs(coordinate_times[:, None] - coordinate_times[None, :]) ** p

    # Métivier et al. (2019), equations (88)--(92): calibrate every trace with
    # eta = DeltaT / A, where A is the joint calculated/observed amplitude
    # range. DeltaT is therefore the maximum time shift the assignment is
    # expected to recover, expressed in the same units as ``dt``.
    joint_max = np.maximum(predicted_numpy.max(axis=1), target_numpy.max(axis=1))
    joint_min = np.minimum(predicted_numpy.min(axis=1), target_numpy.min(axis=1))
    amplitude_range = joint_max - joint_min
    eta = np.divide(
        max_time_shift,
        amplitude_range,
        out=np.zeros_like(amplitude_range, dtype=np.float64),
        where=amplitude_range > 0,
    )

    for trace_index in np.flatnonzero(active):
        amplitude_cost = (
            np.abs(
                eta[trace_index]
                * (
                    predicted_numpy[trace_index, :, None]
                    - target_numpy[trace_index, None, :]
                )
            )
            ** p
        )
        row_indices, column_indices = linear_sum_assignment(time_cost + amplitude_cost)
        assignments[trace_index, row_indices] = column_indices

    assignment = torch.as_tensor(
        assignments,
        dtype=torch.long,
        device=predicted.device,
    )
    matched_target = torch.gather(target_traces, 1, assignment)
    times = (
        torch.arange(nt, dtype=predicted.dtype, device=predicted.device)
        * dt
        * sparse_sampling
    ).expand_as(predicted_traces)
    matched_times = torch.gather(times, 1, assignment)
    graph_cost = (times - matched_times).abs().pow(p)
    eta_tensor = torch.as_tensor(
        eta, dtype=predicted.dtype, device=predicted.device
    ).unsqueeze(1)
    graph_cost = graph_cost + (
        eta_tensor * (predicted_traces - matched_target)
    ).abs().pow(p)
    if observed_energy_weighting:
        # Equation (99): preserve relative trace amplitudes (AVO) after the
        # trace-wise eta normalization. Dividing by the mean energy retains
        # those relative weights while avoiding dependence on waveform units.
        trace_energy = target_traces.square().mean(dim=1, keepdim=True)
        active_tensor = torch.as_tensor(active, device=predicted.device)
        mean_energy = (
            trace_energy[active_tensor]
            .mean()
            .clamp_min(torch.finfo(predicted.dtype).tiny)
        )
        graph_cost = graph_cost * (trace_energy / mean_energy)
    else:
        active_tensor = torch.as_tensor(active, device=predicted.device)
    # This global nondimensionalization leaves both the assignment and the
    # minimizer unchanged, but prevents nanosecond-scale problems from falling
    # below an optimizer's absolute stopping tolerance.
    return graph_cost[active_tensor].sum() / max_time_shift**p


def _receiver_sinkhorn_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    dt: float,
    sparse_sampling: int,
    p: int,
    blur: float,
    scaling: float,
    time_dim: int,
) -> torch.Tensor:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if sparse_sampling <= 0:
        raise ValueError("sparse_sampling must be positive.")
    try:
        from geomloss import SamplesLoss
    except ImportError as exc:
        raise ImportError(
            "receiver_sinkhorn_loss requires geomloss; "
            "install Tide with the 'experiments' extra."
        ) from exc

    time_dim = time_dim % predicted.ndim
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and selected observed data must match, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )

    predicted_traces = predicted.movedim(time_dim, -1)[..., ::sparse_sampling]
    target_traces = target.movedim(time_dim, -1)[..., ::sparse_sampling]
    nt = predicted_traces.shape[-1]
    predicted_traces = predicted_traces.reshape(-1, nt)
    target_traces = target_traces.reshape(-1, nt)

    active = (predicted_traces.abs().sum(dim=-1) != 0) | (
        target_traces.abs().sum(dim=-1) != 0
    )
    if not bool(active.any()):
        return predicted.sum() * 0
    predicted_traces = predicted_traces[active]
    target_traces = target_traces[active]

    times = (
        torch.arange(nt, dtype=predicted.dtype, device=predicted.device)
        * dt
        * sparse_sampling
    )
    times = times.expand_as(predicted_traces)
    predicted_points = torch.stack((times, predicted_traces), dim=-1)
    target_points = torch.stack((times, target_traces), dim=-1)
    distances = SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur,
        scaling=scaling,
    )(predicted_points, target_points)
    return distances.square().sum() * dt


def receiver_mse_loss_shard(
    predicted: torch.Tensor,
    observed_shard: torch.Tensor,
    global_shot_indices: torch.Tensor,
    local_shot_indices: torch.Tensor,
    *,
    global_observed_numel: int,
    shot_dim: int | None = None,
    normalization: LossNormalization = "all",
) -> torch.Tensor:
    """Return MSE for one local observed shard using global normalization."""

    target = take_receiver_shard_batch(
        observed_shard,
        global_shot_indices,
        local_shot_indices,
        shot_dim=shot_dim,
    )
    if normalization == "batch":
        return F.mse_loss(predicted, target)
    residual = predicted - target
    if normalization == "all":
        if global_observed_numel <= 0:
            raise ValueError("global_observed_numel must be positive.")
        return residual.square().sum() / int(global_observed_numel)
    if normalization == "sum":
        return residual.square().sum()
    raise ValueError("normalization must be 'batch', 'all', or 'sum'.")


def backward_shot_batches(
    loss_fn: _ShotLossFn,
    shot_batches: Iterable[torch.Tensor],
    *,
    zero_grad: _ZeroGradFn | None = None,
    zero_each_batch: bool = False,
    after_backward: _AfterBackwardFn | None = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> float:
    """Backpropagate scalar losses over shot mini-batches.

    By default, ``zero_grad`` is called once before the loop so gradients
    accumulate across all batches. Set ``zero_each_batch=True`` when the caller
    wants to inspect per-batch gradients in ``after_backward``.
    """

    if zero_grad is not None and not zero_each_batch:
        zero_grad()

    # Keep the running loss on its device. Converting each batch loss to a
    # Python float would synchronize CUDA once per batch; defer that transfer
    # until every backward pass has been launched.
    total_loss: torch.Tensor | None = None
    n_batches = 0
    for shot_indices in shot_batches:
        if zero_grad is not None and zero_each_batch:
            zero_grad()

        loss = loss_fn(shot_indices)
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss_fn must return a torch.Tensor.")
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar loss tensor.")

        loss.backward(retain_graph=retain_graph, create_graph=create_graph)
        if after_backward is not None:
            after_backward(shot_indices, loss)

        detached_loss = loss.detach()
        total_loss = detached_loss if total_loss is None else total_loss + detached_loss
        n_batches += 1

    if n_batches == 0 or total_loss is None:
        raise ValueError("shot_batches must contain at least one batch.")
    return float(total_loss)


__all__ = [
    "LossNormalization",
    "backward_shot_batches",
    "receiver_gsot_loss",
    "receiver_gsot_loss_shard",
    "receiver_mse_loss",
    "receiver_mse_loss_shard",
    "receiver_sinkhorn_loss",
    "receiver_sinkhorn_loss_shard",
    "take_receiver_batch",
    "take_receiver_shard_batch",
]
