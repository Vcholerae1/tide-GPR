import math
from typing import Any

import torch

from ..callbacks import CallbackState
from ..storage import (
    _CPU_STORAGE_BUFFERS,
    STORAGE_CPU,
    STORAGE_DEVICE,
    STORAGE_DISK,
    STORAGE_NONE,
    TemporaryStorage,
    _resolve_storage_compression,
    storage_mode_to_int,
)
from .common import _make_storage_streams


_MAXWELL3D_NUM_WAVEFIELDS = 18


def _call_forward_with_storage(
    backend_utils: Any,
    forward_func: Any,
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cq: torch.Tensor,
    source_amplitudes_scaled: torch.Tensor,
    wavefields: tuple[torch.Tensor, ...],
    receiver_amplitudes: torch.Tensor,
    stores: tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
    ],
    profiles: tuple[torch.Tensor, ...],
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
    rdz: float,
    rdy: float,
    rdx: float,
    dt: float,
    step_nt: int,
    n_shots: int,
    nz: int,
    ny: int,
    nx: int,
    n_sources: int,
    n_receivers: int,
    step_ratio: int,
    storage_mode: int,
    storage_format: int,
    shot_bytes_uncomp: int,
    ca_requires_grad: bool,
    cb_requires_grad: bool,
    ca_batched: bool,
    cb_batched: bool,
    cq_batched: bool,
    start_t: int,
    pml_z0: int,
    pml_y0: int,
    pml_x0: int,
    pml_z1: int,
    pml_y1: int,
    pml_x1: int,
    source_component_idx: int,
    receiver_component_idx: int,
    n_threads: int,
    device_idx: int,
    execution_backend_id: int,
    compute_stream_handle: int,
    storage_stream_handle: int,
) -> None:
    (
        az,
        bz,
        az_h,
        bz_h,
        ay,
        by,
        ay_h,
        by_h,
        ax,
        bx,
        ax_h,
        bx_h,
        kz,
        kz_h,
        ky,
        ky_h,
        kx,
        kx_h,
    ) = profiles
    (
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        m_hz_y,
        m_hy_z,
        m_hx_z,
        m_hz_x,
        m_hy_x,
        m_hx_y,
        m_ey_z,
        m_ez_y,
        m_ez_x,
        m_ex_z,
        m_ex_y,
        m_ey_x,
    ) = wavefields
    (
        store_ex,
        store_ex_host,
        store_ex_filenames_ptr,
        store_ey,
        store_ey_host,
        store_ey_filenames_ptr,
        store_ez,
        store_ez_host,
        store_ez_filenames_ptr,
        store_curl_x,
        store_curl_x_host,
        store_curl_x_filenames_ptr,
        store_curl_y,
        store_curl_y_host,
        store_curl_y_filenames_ptr,
        store_curl_z,
        store_curl_z_host,
        store_curl_z_filenames_ptr,
    ) = stores

    forward_func(
        backend_utils.tensor_to_ptr(ca),
        backend_utils.tensor_to_ptr(cb),
        backend_utils.tensor_to_ptr(cq),
        backend_utils.tensor_to_ptr(source_amplitudes_scaled),
        backend_utils.tensor_to_ptr(Ex),
        backend_utils.tensor_to_ptr(Ey),
        backend_utils.tensor_to_ptr(Ez),
        backend_utils.tensor_to_ptr(Hx),
        backend_utils.tensor_to_ptr(Hy),
        backend_utils.tensor_to_ptr(Hz),
        backend_utils.tensor_to_ptr(m_hz_y),
        backend_utils.tensor_to_ptr(m_hy_z),
        backend_utils.tensor_to_ptr(m_hx_z),
        backend_utils.tensor_to_ptr(m_hz_x),
        backend_utils.tensor_to_ptr(m_hy_x),
        backend_utils.tensor_to_ptr(m_hx_y),
        backend_utils.tensor_to_ptr(m_ey_z),
        backend_utils.tensor_to_ptr(m_ez_y),
        backend_utils.tensor_to_ptr(m_ez_x),
        backend_utils.tensor_to_ptr(m_ex_z),
        backend_utils.tensor_to_ptr(m_ex_y),
        backend_utils.tensor_to_ptr(m_ey_x),
        backend_utils.tensor_to_ptr(receiver_amplitudes),
        backend_utils.tensor_to_ptr(store_ex),
        backend_utils.tensor_to_ptr(store_ex_host),
        store_ex_filenames_ptr,
        backend_utils.tensor_to_ptr(store_ey),
        backend_utils.tensor_to_ptr(store_ey_host),
        store_ey_filenames_ptr,
        backend_utils.tensor_to_ptr(store_ez),
        backend_utils.tensor_to_ptr(store_ez_host),
        store_ez_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_x),
        backend_utils.tensor_to_ptr(store_curl_x_host),
        store_curl_x_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_y),
        backend_utils.tensor_to_ptr(store_curl_y_host),
        store_curl_y_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_z),
        backend_utils.tensor_to_ptr(store_curl_z_host),
        store_curl_z_filenames_ptr,
        backend_utils.tensor_to_ptr(az),
        backend_utils.tensor_to_ptr(bz),
        backend_utils.tensor_to_ptr(az_h),
        backend_utils.tensor_to_ptr(bz_h),
        backend_utils.tensor_to_ptr(ay),
        backend_utils.tensor_to_ptr(by),
        backend_utils.tensor_to_ptr(ay_h),
        backend_utils.tensor_to_ptr(by_h),
        backend_utils.tensor_to_ptr(ax),
        backend_utils.tensor_to_ptr(bx),
        backend_utils.tensor_to_ptr(ax_h),
        backend_utils.tensor_to_ptr(bx_h),
        backend_utils.tensor_to_ptr(kz),
        backend_utils.tensor_to_ptr(kz_h),
        backend_utils.tensor_to_ptr(ky),
        backend_utils.tensor_to_ptr(ky_h),
        backend_utils.tensor_to_ptr(kx),
        backend_utils.tensor_to_ptr(kx_h),
        backend_utils.tensor_to_ptr(sources_i),
        backend_utils.tensor_to_ptr(receivers_i),
        rdz,
        rdy,
        rdx,
        dt,
        step_nt,
        n_shots,
        nz,
        ny,
        nx,
        n_sources,
        n_receivers,
        step_ratio,
        storage_mode,
        storage_format,
        shot_bytes_uncomp,
        ca_requires_grad,
        cb_requires_grad,
        ca_batched,
        cb_batched,
        cq_batched,
        start_t,
        pml_z0,
        pml_y0,
        pml_x0,
        pml_z1,
        pml_y1,
        pml_x1,
        source_component_idx,
        receiver_component_idx,
        n_threads,
        device_idx,
        execution_backend_id,
        compute_stream_handle,
        storage_stream_handle,
    )


def _call_backward(
    backend_utils: Any,
    backward_func: Any,
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cq: torch.Tensor,
    grad_r: torch.Tensor,
    lambdas: tuple[torch.Tensor, ...],
    stores: tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
    ],
    grad_f: torch.Tensor,
    grad_ca: torch.Tensor,
    grad_cb: torch.Tensor,
    grad_eps: torch.Tensor,
    grad_sigma: torch.Tensor,
    grad_ca_shot: torch.Tensor,
    grad_cb_shot: torch.Tensor,
    zero_grad_on_entry: bool,
    profiles: tuple[torch.Tensor, ...],
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
    rdz: float,
    rdy: float,
    rdx: float,
    dt: float,
    step_nt: int,
    n_shots: int,
    nz: int,
    ny: int,
    nx: int,
    n_sources: int,
    n_receivers: int,
    step_ratio: int,
    storage_mode: int,
    storage_format: int,
    shot_bytes_uncomp: int,
    ca_requires_grad: bool,
    cb_requires_grad: bool,
    ca_batched: bool,
    cb_batched: bool,
    cq_batched: bool,
    start_t: int,
    pml_z0: int,
    pml_y0: int,
    pml_x0: int,
    pml_z1: int,
    pml_y1: int,
    pml_x1: int,
    source_component_idx: int,
    receiver_component_idx: int,
    n_threads: int,
    device_idx: int,
    execution_backend_id: int,
    compute_stream_handle: int,
    storage_stream_handle: int,
) -> None:
    (
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_ey_z,
        m_lambda_ez_y,
        m_lambda_ez_x,
        m_lambda_ex_z,
        m_lambda_ex_y,
        m_lambda_ey_x,
        m_lambda_hz_y,
        m_lambda_hy_z,
        m_lambda_hx_z,
        m_lambda_hz_x,
        m_lambda_hy_x,
        m_lambda_hx_y,
    ) = lambdas
    (
        az,
        bz,
        az_h,
        bz_h,
        ay,
        by,
        ay_h,
        by_h,
        ax,
        bx,
        ax_h,
        bx_h,
        kz,
        kz_h,
        ky,
        ky_h,
        kx,
        kx_h,
    ) = profiles
    (
        store_ex,
        store_ex_host,
        store_ex_filenames_ptr,
        store_ey,
        store_ey_host,
        store_ey_filenames_ptr,
        store_ez,
        store_ez_host,
        store_ez_filenames_ptr,
        store_curl_x,
        store_curl_x_host,
        store_curl_x_filenames_ptr,
        store_curl_y,
        store_curl_y_host,
        store_curl_y_filenames_ptr,
        store_curl_z,
        store_curl_z_host,
        store_curl_z_filenames_ptr,
    ) = stores

    backward_func(
        backend_utils.tensor_to_ptr(ca),
        backend_utils.tensor_to_ptr(cb),
        backend_utils.tensor_to_ptr(cq),
        backend_utils.tensor_to_ptr(grad_r),
        backend_utils.tensor_to_ptr(lambda_ex),
        backend_utils.tensor_to_ptr(lambda_ey),
        backend_utils.tensor_to_ptr(lambda_ez),
        backend_utils.tensor_to_ptr(lambda_hx),
        backend_utils.tensor_to_ptr(lambda_hy),
        backend_utils.tensor_to_ptr(lambda_hz),
        backend_utils.tensor_to_ptr(m_lambda_ey_z),
        backend_utils.tensor_to_ptr(m_lambda_ez_y),
        backend_utils.tensor_to_ptr(m_lambda_ez_x),
        backend_utils.tensor_to_ptr(m_lambda_ex_z),
        backend_utils.tensor_to_ptr(m_lambda_ex_y),
        backend_utils.tensor_to_ptr(m_lambda_ey_x),
        backend_utils.tensor_to_ptr(m_lambda_hz_y),
        backend_utils.tensor_to_ptr(m_lambda_hy_z),
        backend_utils.tensor_to_ptr(m_lambda_hx_z),
        backend_utils.tensor_to_ptr(m_lambda_hz_x),
        backend_utils.tensor_to_ptr(m_lambda_hy_x),
        backend_utils.tensor_to_ptr(m_lambda_hx_y),
        backend_utils.tensor_to_ptr(store_ex),
        backend_utils.tensor_to_ptr(store_ex_host),
        store_ex_filenames_ptr,
        backend_utils.tensor_to_ptr(store_ey),
        backend_utils.tensor_to_ptr(store_ey_host),
        store_ey_filenames_ptr,
        backend_utils.tensor_to_ptr(store_ez),
        backend_utils.tensor_to_ptr(store_ez_host),
        store_ez_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_x),
        backend_utils.tensor_to_ptr(store_curl_x_host),
        store_curl_x_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_y),
        backend_utils.tensor_to_ptr(store_curl_y_host),
        store_curl_y_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_z),
        backend_utils.tensor_to_ptr(store_curl_z_host),
        store_curl_z_filenames_ptr,
        backend_utils.tensor_to_ptr(grad_f),
        backend_utils.tensor_to_ptr(grad_ca),
        backend_utils.tensor_to_ptr(grad_cb),
        backend_utils.tensor_to_ptr(grad_eps),
        backend_utils.tensor_to_ptr(grad_sigma),
        backend_utils.tensor_to_ptr(grad_ca_shot),
        backend_utils.tensor_to_ptr(grad_cb_shot),
        zero_grad_on_entry,
        backend_utils.tensor_to_ptr(az),
        backend_utils.tensor_to_ptr(bz),
        backend_utils.tensor_to_ptr(az_h),
        backend_utils.tensor_to_ptr(bz_h),
        backend_utils.tensor_to_ptr(ay),
        backend_utils.tensor_to_ptr(by),
        backend_utils.tensor_to_ptr(ay_h),
        backend_utils.tensor_to_ptr(by_h),
        backend_utils.tensor_to_ptr(ax),
        backend_utils.tensor_to_ptr(bx),
        backend_utils.tensor_to_ptr(ax_h),
        backend_utils.tensor_to_ptr(bx_h),
        backend_utils.tensor_to_ptr(kz),
        backend_utils.tensor_to_ptr(kz_h),
        backend_utils.tensor_to_ptr(ky),
        backend_utils.tensor_to_ptr(ky_h),
        backend_utils.tensor_to_ptr(kx),
        backend_utils.tensor_to_ptr(kx_h),
        backend_utils.tensor_to_ptr(sources_i),
        backend_utils.tensor_to_ptr(receivers_i),
        rdz,
        rdy,
        rdx,
        dt,
        step_nt,
        n_shots,
        nz,
        ny,
        nx,
        n_sources,
        n_receivers,
        step_ratio,
        storage_mode,
        storage_format,
        shot_bytes_uncomp,
        ca_requires_grad,
        cb_requires_grad,
        ca_batched,
        cb_batched,
        cq_batched,
        start_t,
        pml_z0,
        pml_y0,
        pml_x0,
        pml_z1,
        pml_y1,
        pml_x1,
        source_component_idx,
        receiver_component_idx,
        n_threads,
        device_idx,
        execution_backend_id,
        compute_stream_handle,
        storage_stream_handle,
    )


def _call_checkpoint_revolve_backward(
    backend_utils: Any,
    checkpoint_func: Any,
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cq: torch.Tensor,
    source_amplitudes_scaled: torch.Tensor,
    grad_r: torch.Tensor,
    checkpoint_pool: torch.Tensor,
    scratch_wavefields: torch.Tensor,
    lambdas: tuple[torch.Tensor, ...],
    stores: tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        int,
    ],
    grad_f: torch.Tensor,
    grad_ca: torch.Tensor,
    grad_cb: torch.Tensor,
    grad_eps: torch.Tensor,
    grad_sigma: torch.Tensor,
    grad_ca_shot: torch.Tensor,
    grad_cb_shot: torch.Tensor,
    profiles: tuple[torch.Tensor, ...],
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
    rdz: float,
    rdy: float,
    rdx: float,
    dt: float,
    nt: int,
    n_shots: int,
    nz: int,
    ny: int,
    nx: int,
    n_sources: int,
    n_receivers: int,
    storage_mode: int,
    storage_format: int,
    shot_bytes_uncomp: int,
    ca_requires_grad: bool,
    cb_requires_grad: bool,
    ca_batched: bool,
    cb_batched: bool,
    cq_batched: bool,
    pml_z0: int,
    pml_y0: int,
    pml_x0: int,
    pml_z1: int,
    pml_y1: int,
    pml_x1: int,
    source_component_idx: int,
    receiver_component_idx: int,
    n_threads: int,
    device_idx: int,
    execution_backend_id: int,
    segment_steps: int,
    num_segments: int,
    pool_slots: int,
    compute_stream_handle: int,
    storage_stream_handle: int,
) -> None:
    (
        lambda_ex,
        lambda_ey,
        lambda_ez,
        lambda_hx,
        lambda_hy,
        lambda_hz,
        m_lambda_ey_z,
        m_lambda_ez_y,
        m_lambda_ez_x,
        m_lambda_ex_z,
        m_lambda_ex_y,
        m_lambda_ey_x,
        m_lambda_hz_y,
        m_lambda_hy_z,
        m_lambda_hx_z,
        m_lambda_hz_x,
        m_lambda_hy_x,
        m_lambda_hx_y,
    ) = lambdas
    (
        az,
        bz,
        az_h,
        bz_h,
        ay,
        by,
        ay_h,
        by_h,
        ax,
        bx,
        ax_h,
        bx_h,
        kz,
        kz_h,
        ky,
        ky_h,
        kx,
        kx_h,
    ) = profiles
    (
        store_ex,
        store_ex_host,
        store_ex_filenames_ptr,
        store_ey,
        store_ey_host,
        store_ey_filenames_ptr,
        store_ez,
        store_ez_host,
        store_ez_filenames_ptr,
        store_curl_x,
        store_curl_x_host,
        store_curl_x_filenames_ptr,
        store_curl_y,
        store_curl_y_host,
        store_curl_y_filenames_ptr,
        store_curl_z,
        store_curl_z_host,
        store_curl_z_filenames_ptr,
    ) = stores

    checkpoint_func(
        backend_utils.tensor_to_ptr(ca),
        backend_utils.tensor_to_ptr(cb),
        backend_utils.tensor_to_ptr(cq),
        backend_utils.tensor_to_ptr(source_amplitudes_scaled),
        backend_utils.tensor_to_ptr(grad_r),
        backend_utils.tensor_to_ptr(checkpoint_pool),
        backend_utils.tensor_to_ptr(scratch_wavefields),
        backend_utils.tensor_to_ptr(lambda_ex),
        backend_utils.tensor_to_ptr(lambda_ey),
        backend_utils.tensor_to_ptr(lambda_ez),
        backend_utils.tensor_to_ptr(lambda_hx),
        backend_utils.tensor_to_ptr(lambda_hy),
        backend_utils.tensor_to_ptr(lambda_hz),
        backend_utils.tensor_to_ptr(m_lambda_ey_z),
        backend_utils.tensor_to_ptr(m_lambda_ez_y),
        backend_utils.tensor_to_ptr(m_lambda_ez_x),
        backend_utils.tensor_to_ptr(m_lambda_ex_z),
        backend_utils.tensor_to_ptr(m_lambda_ex_y),
        backend_utils.tensor_to_ptr(m_lambda_ey_x),
        backend_utils.tensor_to_ptr(m_lambda_hz_y),
        backend_utils.tensor_to_ptr(m_lambda_hy_z),
        backend_utils.tensor_to_ptr(m_lambda_hx_z),
        backend_utils.tensor_to_ptr(m_lambda_hz_x),
        backend_utils.tensor_to_ptr(m_lambda_hy_x),
        backend_utils.tensor_to_ptr(m_lambda_hx_y),
        backend_utils.tensor_to_ptr(store_ex),
        backend_utils.tensor_to_ptr(store_ex_host),
        store_ex_filenames_ptr,
        backend_utils.tensor_to_ptr(store_ey),
        backend_utils.tensor_to_ptr(store_ey_host),
        store_ey_filenames_ptr,
        backend_utils.tensor_to_ptr(store_ez),
        backend_utils.tensor_to_ptr(store_ez_host),
        store_ez_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_x),
        backend_utils.tensor_to_ptr(store_curl_x_host),
        store_curl_x_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_y),
        backend_utils.tensor_to_ptr(store_curl_y_host),
        store_curl_y_filenames_ptr,
        backend_utils.tensor_to_ptr(store_curl_z),
        backend_utils.tensor_to_ptr(store_curl_z_host),
        store_curl_z_filenames_ptr,
        backend_utils.tensor_to_ptr(grad_f),
        backend_utils.tensor_to_ptr(grad_ca),
        backend_utils.tensor_to_ptr(grad_cb),
        backend_utils.tensor_to_ptr(grad_eps),
        backend_utils.tensor_to_ptr(grad_sigma),
        backend_utils.tensor_to_ptr(grad_ca_shot),
        backend_utils.tensor_to_ptr(grad_cb_shot),
        backend_utils.tensor_to_ptr(az),
        backend_utils.tensor_to_ptr(bz),
        backend_utils.tensor_to_ptr(az_h),
        backend_utils.tensor_to_ptr(bz_h),
        backend_utils.tensor_to_ptr(ay),
        backend_utils.tensor_to_ptr(by),
        backend_utils.tensor_to_ptr(ay_h),
        backend_utils.tensor_to_ptr(by_h),
        backend_utils.tensor_to_ptr(ax),
        backend_utils.tensor_to_ptr(bx),
        backend_utils.tensor_to_ptr(ax_h),
        backend_utils.tensor_to_ptr(bx_h),
        backend_utils.tensor_to_ptr(kz),
        backend_utils.tensor_to_ptr(kz_h),
        backend_utils.tensor_to_ptr(ky),
        backend_utils.tensor_to_ptr(ky_h),
        backend_utils.tensor_to_ptr(kx),
        backend_utils.tensor_to_ptr(kx_h),
        backend_utils.tensor_to_ptr(sources_i),
        backend_utils.tensor_to_ptr(receivers_i),
        rdz,
        rdy,
        rdx,
        dt,
        nt,
        n_shots,
        nz,
        ny,
        nx,
        n_sources,
        n_receivers,
        1,
        storage_mode,
        storage_format,
        shot_bytes_uncomp,
        ca_requires_grad,
        cb_requires_grad,
        ca_batched,
        cb_batched,
        cq_batched,
        0,
        pml_z0,
        pml_y0,
        pml_x0,
        pml_z1,
        pml_y1,
        pml_x1,
        source_component_idx,
        receiver_component_idx,
        n_threads,
        device_idx,
        execution_backend_id,
        segment_steps,
        num_segments,
        pool_slots,
        compute_stream_handle,
        storage_stream_handle,
    )


class Maxwell3DForwardFunc(torch.autograd.Function):
    """Autograd function for 3D C/CUDA backend propagation."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        epsilon_padded: torch.Tensor,
        sigma_padded: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        profiles: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, torch.Tensor],
        wavefields: tuple[torch.Tensor, ...],
        meta: dict[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        from .. import backend_utils
        import ctypes

        (
            az,
            bz,
            az_h,
            bz_h,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            kz,
            kz_h,
            ky,
            ky_h,
            kx,
            kx_h,
        ) = profiles
        sources_i, receivers_i = indices
        (
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            m_hz_y,
            m_hy_z,
            m_hx_z,
            m_hz_x,
            m_hy_x,
            m_hx_y,
            m_ey_z,
            m_ez_y,
            m_ez_x,
            m_ex_z,
            m_ex_y,
            m_ey_x,
        ) = wavefields

        device = Ex.device
        dtype = Ex.dtype

        nt = int(meta["nt"])
        n_shots = int(meta["n_shots"])
        nz = int(meta["nz"])
        ny = int(meta["ny"])
        nx = int(meta["nx"])
        n_sources = int(meta["n_sources"])
        n_receivers = int(meta["n_receivers"])
        step_ratio = int(meta["step_ratio"])
        accuracy = int(meta["accuracy"])
        pml_z0 = int(meta["pml_z0"])
        pml_y0 = int(meta["pml_y0"])
        pml_x0 = int(meta["pml_x0"])
        pml_z1 = int(meta["pml_z1"])
        pml_y1 = int(meta["pml_y1"])
        pml_x1 = int(meta["pml_x1"])
        source_component_idx = int(meta["source_component_idx"])
        receiver_component_idx = int(meta["receiver_component_idx"])
        execution_backend_id = int(meta.get("execution_backend_id", 0))
        del epsilon_padded, sigma_padded
        n_threads = int(meta["n_threads"])
        callback_frequency = int(meta["callback_frequency"])
        forward_callback = meta["forward_callback"]
        models = meta["models"]
        fd_pad = meta["fd_pad"]
        pml_width = meta["pml_width"]
        grid_spacing = meta["grid_spacing"]
        dt = float(meta["dt"])
        ca_batched = bool(meta.get("ca_batched", False))
        cb_batched = bool(meta.get("cb_batched", False))
        cq_batched = bool(meta.get("cq_batched", False))

        ca_requires_grad = bool(ca.requires_grad)
        cb_requires_grad = bool(cb.requires_grad)
        requires_grad = ca_requires_grad or cb_requires_grad
        if not requires_grad:
            raise RuntimeError(
                "Maxwell3DForwardFunc should only be used when gradients are required."
            )

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        step_ratio = max(1, step_ratio)
        num_steps_stored = (nt + step_ratio - 1) // step_ratio
        _, store_dtype, _, storage_format = _resolve_storage_compression(
            meta["storage_compression"],
            dtype,
            device,
            context="storage_compression",
        )
        storage_mode_str = str(meta["storage_mode_str"]).lower()
        storage_path = str(meta["storage_path"])
        if device.type == "cpu" and storage_mode_str in {"cpu", "disk"}:
            storage_mode_str = "device"
        storage_mode = storage_mode_to_int(storage_mode_str)
        eonly_snapshots = (
            bool(meta.get("experimental_eonly_snapshots", False))
            and storage_mode == STORAGE_DEVICE
            and step_ratio == 1
            and ca_requires_grad
            and cb_requires_grad
        )
        if not eonly_snapshots and execution_backend_id == 1:
            execution_backend_id = 0
        if not eonly_snapshots and execution_backend_id == 2:
            execution_backend_id = 0
        physical_snapshot_storage = (
            bool(meta.get("physical_snapshot_storage", False))
            and storage_mode == STORAGE_DEVICE
            and not eonly_snapshots
        )
        physical_nz = max(0, pml_z1 - pml_z0)
        physical_ny = max(0, pml_y1 - pml_y0)
        physical_nx = max(0, pml_x1 - pml_x0)
        if physical_nz == 0 or physical_ny == 0 or physical_nx == 0:
            physical_snapshot_storage = False
        store_nz = physical_nz if physical_snapshot_storage else nz
        store_ny = physical_ny if physical_snapshot_storage else ny
        store_nx = physical_nx if physical_snapshot_storage else nx
        shot_numel_store = store_nz * store_ny * store_nx
        shot_bytes_uncomp = shot_numel_store * store_dtype.itemsize
        compute_stream_handle, storage_stream_handle, stream_keepalive = (
            _make_storage_streams(device, storage_mode)
        )
        backward_storage_objects: list[Any] = []
        backward_storage_filename_arrays: list[Any] = []
        char_ptr_type = ctypes.c_char_p
        is_cuda = device.type == "cuda"
        empty_store = torch.empty(0, device=device, dtype=store_dtype)

        def alloc_storage(requires_grad_cond: bool):
            store_1 = empty_store
            store_3 = empty_store
            filenames_arr = (char_ptr_type * 0)()
            if not requires_grad_cond:
                backward_storage_filename_arrays.append(filenames_arr)
                return store_1, store_3, 0

            if storage_mode == STORAGE_DEVICE:
                store_1 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    store_nz,
                    store_ny,
                    store_nx,
                    device=device,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_CPU:
                store_1 = torch.empty(
                    _CPU_STORAGE_BUFFERS,
                    n_shots,
                    nz,
                    ny,
                    nx,
                    device=device,
                    dtype=store_dtype,
                )
                store_3 = torch.empty(
                    num_steps_stored,
                    n_shots,
                    nz,
                    ny,
                    nx,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )
            elif storage_mode == STORAGE_DISK:
                storage_obj = TemporaryStorage(storage_path, 1 if is_cuda else n_shots)
                backward_storage_objects.append(storage_obj)
                filenames_list = [
                    f.encode("utf-8") for f in storage_obj.get_filenames()
                ]
                filenames_arr = (char_ptr_type * len(filenames_list))()
                for i_file, f_name in enumerate(filenames_list):
                    filenames_arr[i_file] = ctypes.cast(
                        ctypes.create_string_buffer(f_name), char_ptr_type
                    )
                if is_cuda:
                    store_1 = torch.empty(
                        _CPU_STORAGE_BUFFERS,
                        n_shots,
                        nz,
                        ny,
                        nx,
                        device=device,
                        dtype=store_dtype,
                    )
                    store_3 = torch.empty(
                        _CPU_STORAGE_BUFFERS,
                        n_shots,
                        nz,
                        ny,
                        nx,
                        device="cpu",
                        pin_memory=True,
                        dtype=store_dtype,
                    )
                else:
                    store_1 = torch.empty(
                        n_shots, nz, ny, nx, device=device, dtype=store_dtype
                    )

            backward_storage_filename_arrays.append(filenames_arr)
            filenames_ptr = (
                ctypes.cast(filenames_arr, ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0
            )
            return store_1, store_3, filenames_ptr

        def alloc_final_e_storage(requires_grad_cond: bool):
            filenames_arr = (char_ptr_type * 0)()
            backward_storage_filename_arrays.append(filenames_arr)
            if not requires_grad_cond:
                return empty_store, empty_store, 0
            if storage_mode != STORAGE_DEVICE:
                raise RuntimeError(
                    "E-only 3D snapshot storage is only supported on device storage."
                )
            store_1 = torch.empty(
                n_shots,
                nz,
                ny,
                nx,
                device=device,
                dtype=store_dtype,
            )
            return store_1, empty_store, 0

        store_ex, store_ex_host, store_ex_filenames_ptr = alloc_storage(
            ca_requires_grad
        )
        store_ey, store_ey_host, store_ey_filenames_ptr = alloc_storage(
            ca_requires_grad
        )
        store_ez, store_ez_host, store_ez_filenames_ptr = alloc_storage(
            ca_requires_grad
        )
        if eonly_snapshots:
            store_curl_x, store_curl_x_host, store_curl_x_filenames_ptr = (
                alloc_final_e_storage(True)
            )
            store_curl_y, store_curl_y_host, store_curl_y_filenames_ptr = (
                alloc_final_e_storage(True)
            )
            store_curl_z, store_curl_z_host, store_curl_z_filenames_ptr = (
                alloc_final_e_storage(True)
            )
        else:
            store_curl_x, store_curl_x_host, store_curl_x_filenames_ptr = alloc_storage(
                cb_requires_grad
            )
            store_curl_y, store_curl_y_host, store_curl_y_filenames_ptr = alloc_storage(
                cb_requires_grad
            )
            store_curl_z, store_curl_z_host, store_curl_z_filenames_ptr = alloc_storage(
                cb_requires_grad
            )

        forward_func = backend_utils.get_backend_function(
            "maxwell_3d", "forward_with_storage", accuracy, dtype, device
        )
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        effective_callback_freq = nt if forward_callback is None else callback_frequency
        if effective_callback_freq <= 0:
            effective_callback_freq = nt if nt > 0 else 1

        for step in range(0, nt, effective_callback_freq):
            if forward_callback is not None:
                forward_callback(
                    CallbackState(
                        dt=dt,
                        step=step,
                        nt=nt,
                        wavefields={
                            "Ex": Ex,
                            "Ey": Ey,
                            "Ez": Ez,
                            "Hx": Hx,
                            "Hy": Hy,
                            "Hz": Hz,
                            "m_hz_y": m_hz_y,
                            "m_hy_z": m_hy_z,
                            "m_hx_z": m_hx_z,
                            "m_hz_x": m_hz_x,
                            "m_hy_x": m_hy_x,
                            "m_hx_y": m_hx_y,
                            "m_ey_z": m_ey_z,
                            "m_ez_y": m_ez_y,
                            "m_ez_x": m_ez_x,
                            "m_ex_z": m_ex_z,
                            "m_ex_y": m_ex_y,
                            "m_ey_x": m_ey_x,
                        },
                        models=models,
                        gradients={},
                        fd_pad=list(fd_pad),
                        pml_width=list(pml_width),
                        is_backward=False,
                        grid_spacing=list(grid_spacing),
                    )
                )

            step_nt = min(nt - step, effective_callback_freq)
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(source_amplitudes_scaled),
                backend_utils.tensor_to_ptr(Ex),
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Ez),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hy),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_hz_y),
                backend_utils.tensor_to_ptr(m_hy_z),
                backend_utils.tensor_to_ptr(m_hx_z),
                backend_utils.tensor_to_ptr(m_hz_x),
                backend_utils.tensor_to_ptr(m_hy_x),
                backend_utils.tensor_to_ptr(m_hx_y),
                backend_utils.tensor_to_ptr(m_ey_z),
                backend_utils.tensor_to_ptr(m_ez_y),
                backend_utils.tensor_to_ptr(m_ez_x),
                backend_utils.tensor_to_ptr(m_ex_z),
                backend_utils.tensor_to_ptr(m_ex_y),
                backend_utils.tensor_to_ptr(m_ey_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(store_ex),
                backend_utils.tensor_to_ptr(store_ex_host),
                store_ex_filenames_ptr,
                backend_utils.tensor_to_ptr(store_ey),
                backend_utils.tensor_to_ptr(store_ey_host),
                store_ey_filenames_ptr,
                backend_utils.tensor_to_ptr(store_ez),
                backend_utils.tensor_to_ptr(store_ez_host),
                store_ez_filenames_ptr,
                backend_utils.tensor_to_ptr(store_curl_x),
                backend_utils.tensor_to_ptr(store_curl_x_host),
                store_curl_x_filenames_ptr,
                backend_utils.tensor_to_ptr(store_curl_y),
                backend_utils.tensor_to_ptr(store_curl_y_host),
                store_curl_y_filenames_ptr,
                backend_utils.tensor_to_ptr(store_curl_z),
                backend_utils.tensor_to_ptr(store_curl_z_host),
                store_curl_z_filenames_ptr,
                backend_utils.tensor_to_ptr(az),
                backend_utils.tensor_to_ptr(bz),
                backend_utils.tensor_to_ptr(az_h),
                backend_utils.tensor_to_ptr(bz_h),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(kz),
                backend_utils.tensor_to_ptr(kz_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                float(meta["rdz"]),
                float(meta["rdy"]),
                float(meta["rdx"]),
                dt,
                step_nt,
                n_shots,
                nz,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                storage_mode,
                storage_format,
                shot_bytes_uncomp,
                ca_requires_grad,
                cb_requires_grad,
                ca_batched,
                cb_batched,
                cq_batched,
                step,
                pml_z0,
                pml_y0,
                pml_x0,
                pml_z1,
                pml_y1,
                pml_x1,
                source_component_idx,
                receiver_component_idx,
                n_threads,
                device_idx,
                execution_backend_id,
                compute_stream_handle,
                storage_stream_handle,
            )
            if forward_callback is not None:
                callback_wavefields = {
                    "Ex": Ex,
                    "Ey": Ey,
                    "Ez": Ez,
                    "Hx": Hx,
                    "Hy": Hy,
                    "Hz": Hz,
                    "m_hz_y": m_hz_y,
                    "m_hy_z": m_hy_z,
                    "m_hx_z": m_hx_z,
                    "m_hz_x": m_hz_x,
                    "m_hy_x": m_hy_x,
                    "m_hx_y": m_hx_y,
                    "m_ey_z": m_ey_z,
                    "m_ez_y": m_ez_y,
                    "m_ez_x": m_ez_x,
                    "m_ex_z": m_ex_z,
                    "m_ex_y": m_ex_y,
                    "m_ey_x": m_ey_x,
                }
                forward_callback(
                    CallbackState(
                        dt=dt,
                        step=nt // step_ratio,
                        nt=nt // step_ratio,
                        wavefields=callback_wavefields,
                        models=models,
                        gradients={},
                        fd_pad=list(fd_pad),
                        pml_width=list(pml_width),
                        is_backward=False,
                        grid_spacing=list(grid_spacing),
                    )
                )

        ctx.save_for_backward(
            ca,
            cb,
            cq,
            az,
            bz,
            az_h,
            bz_h,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            kz,
            kz_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            store_ex,
            store_ex_host,
            store_ey,
            store_ey_host,
            store_ez,
            store_ez_host,
            store_curl_x,
            store_curl_x_host,
            store_curl_y,
            store_curl_y_host,
            store_curl_z,
            store_curl_z_host,
            source_amplitudes_scaled,
        )
        ctx.meta = {
            "dt": dt,
            "nt": nt,
            "n_shots": n_shots,
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "n_sources": n_sources,
            "n_receivers": n_receivers,
            "step_ratio": step_ratio,
            "accuracy": accuracy,
            "pml_z0": pml_z0,
            "pml_y0": pml_y0,
            "pml_x0": pml_x0,
            "pml_z1": pml_z1,
            "pml_y1": pml_y1,
            "pml_x1": pml_x1,
            "source_component_idx": source_component_idx,
            "receiver_component_idx": receiver_component_idx,
            "execution_backend_id": execution_backend_id,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
            "models": models,
            "fd_pad": fd_pad,
            "pml_width": pml_width,
            "backward_callback": meta["backward_callback"],
            "callback_frequency": callback_frequency,
            "n_threads": n_threads,
            "rdz": float(meta["rdz"]),
            "rdy": float(meta["rdy"]),
            "rdx": float(meta["rdx"]),
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "storage_mode": storage_mode,
            "storage_format": storage_format,
            "stream_keepalive": stream_keepalive,
            "experimental_eonly_snapshots": eonly_snapshots,
            "physical_snapshot_storage": physical_snapshot_storage,
            "ca_batched": ca_batched,
            "cb_batched": cb_batched,
            "cq_batched": cq_batched,
        }
        ctx.backward_storage_objects = backward_storage_objects
        ctx.backward_storage_filename_arrays = backward_storage_filename_arrays

        return (
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            m_hz_y,
            m_hy_z,
            m_hx_z,
            m_hz_x,
            m_hy_x,
            m_hx_y,
            m_ey_z,
            m_ez_y,
            m_ez_x,
            m_ex_z,
            m_ex_y,
            m_ey_x,
            receiver_amplitudes,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        from .. import backend_utils
        import ctypes

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        az, bz, az_h, bz_h = saved[3], saved[4], saved[5], saved[6]
        ay, by, ay_h, by_h = saved[7], saved[8], saved[9], saved[10]
        ax, bx, ax_h, bx_h = saved[11], saved[12], saved[13], saved[14]
        kz, kz_h, ky, ky_h, kx, kx_h = (
            saved[15],
            saved[16],
            saved[17],
            saved[18],
            saved[19],
            saved[20],
        )
        sources_i, receivers_i = saved[21], saved[22]
        store_ex, store_ex_host = saved[23], saved[24]
        store_ey, store_ey_host = saved[25], saved[26]
        store_ez, store_ez_host = saved[27], saved[28]
        store_curl_x, store_curl_x_host = saved[29], saved[30]
        store_curl_y, store_curl_y_host = saved[31], saved[32]
        store_curl_z, store_curl_z_host = saved[33], saved[34]
        source_amplitudes_scaled = saved[35]

        meta = ctx.meta
        device = ca.device
        dtype = ca.dtype

        nt = int(meta["nt"])
        n_shots = int(meta["n_shots"])
        nz = int(meta["nz"])
        ny = int(meta["ny"])
        nx = int(meta["nx"])
        n_sources = int(meta["n_sources"])
        n_receivers = int(meta["n_receivers"])
        step_ratio = int(meta["step_ratio"])
        accuracy = int(meta["accuracy"])
        pml_z0 = int(meta["pml_z0"])
        pml_y0 = int(meta["pml_y0"])
        pml_x0 = int(meta["pml_x0"])
        pml_z1 = int(meta["pml_z1"])
        pml_y1 = int(meta["pml_y1"])
        pml_x1 = int(meta["pml_x1"])
        source_component_idx = int(meta["source_component_idx"])
        receiver_component_idx = int(meta["receiver_component_idx"])
        execution_backend_id = int(meta.get("execution_backend_id", 0))
        ca_requires_grad = bool(meta["ca_requires_grad"])
        cb_requires_grad = bool(meta["cb_requires_grad"])
        backward_callback = meta["backward_callback"]
        callback_frequency = int(meta["callback_frequency"])
        models = meta["models"]
        fd_pad = meta["fd_pad"]
        pml_width = meta["pml_width"]
        n_threads = int(meta["n_threads"])
        shot_bytes_uncomp = int(meta["shot_bytes_uncomp"])
        dt = float(meta["dt"])
        storage_mode = int(meta["storage_mode"])
        storage_format = int(meta["storage_format"])
        ca_batched = bool(meta.get("ca_batched", False))
        cb_batched = bool(meta.get("cb_batched", False))
        cq_batched = bool(meta.get("cq_batched", False))
        eonly_snapshots = bool(meta.get("experimental_eonly_snapshots", False))
        direct_material_grad = execution_backend_id == 2 and eonly_snapshots
        direct_eps_requires_grad = direct_material_grad and ctx.needs_input_grad[3]
        direct_sigma_requires_grad = direct_material_grad and ctx.needs_input_grad[4]
        coeff_ca_requires_grad = ca_requires_grad and not direct_material_grad
        coeff_cb_requires_grad = cb_requires_grad and not direct_material_grad

        grad_r = grad_outputs[-1]
        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            grad_r = grad_r.contiguous()

        lambda_ex = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_ey = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_ez = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_hx = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_hy = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_hz = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)

        m_lambda_ey_z = torch.zeros_like(lambda_ex)
        m_lambda_ez_y = torch.zeros_like(lambda_ex)
        m_lambda_ez_x = torch.zeros_like(lambda_ex)
        m_lambda_ex_z = torch.zeros_like(lambda_ex)
        m_lambda_ex_y = torch.zeros_like(lambda_ex)
        m_lambda_ey_x = torch.zeros_like(lambda_ex)
        m_lambda_hz_y = torch.zeros_like(lambda_ex)
        m_lambda_hy_z = torch.zeros_like(lambda_ex)
        m_lambda_hx_z = torch.zeros_like(lambda_ex)
        m_lambda_hz_x = torch.zeros_like(lambda_ex)
        m_lambda_hy_x = torch.zeros_like(lambda_ex)
        m_lambda_hx_y = torch.zeros_like(lambda_ex)

        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if coeff_ca_requires_grad:
            grad_ca = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if ca_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
            grad_ca_shot = (
                torch.empty(0, device=device, dtype=dtype)
                if ca_batched
                else torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            )
        elif direct_eps_requires_grad:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = (
                torch.empty(0, device=device, dtype=dtype)
                if ca_batched
                else torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            )
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if coeff_cb_requires_grad:
            grad_cb = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if cb_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
            grad_cb_shot = (
                torch.empty(0, device=device, dtype=dtype)
                if cb_batched
                else torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            )
        elif direct_sigma_requires_grad:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = (
                torch.empty(0, device=device, dtype=dtype)
                if cb_batched
                else torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            )
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if (
            (ca_requires_grad or cb_requires_grad) and not direct_material_grad
        ) or direct_eps_requires_grad:
            grad_eps = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if ca_batched or cb_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)

        if (
            (ca_requires_grad or cb_requires_grad) and not direct_material_grad
        ) or direct_sigma_requires_grad:
            grad_sigma = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if ca_batched or cb_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
        else:
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        backward_func = backend_utils.get_backend_function(
            "maxwell_3d", "backward", accuracy, dtype, device
        )
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        compute_stream_handle, storage_stream_handle, stream_keepalive = (
            _make_storage_streams(device, storage_mode)
        )
        ctx.stream_keepalive = stream_keepalive
        effective_callback_freq = (
            nt if backward_callback is None else callback_frequency
        )
        if effective_callback_freq <= 0:
            effective_callback_freq = nt if nt > 0 else 1

        for chunk_idx, step in enumerate(range(nt, 0, -effective_callback_freq)):
            step_nt = min(step, effective_callback_freq)
            backward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(grad_r),
                backend_utils.tensor_to_ptr(lambda_ex),
                backend_utils.tensor_to_ptr(lambda_ey),
                backend_utils.tensor_to_ptr(lambda_ez),
                backend_utils.tensor_to_ptr(lambda_hx),
                backend_utils.tensor_to_ptr(lambda_hy),
                backend_utils.tensor_to_ptr(lambda_hz),
                backend_utils.tensor_to_ptr(m_lambda_ey_z),
                backend_utils.tensor_to_ptr(m_lambda_ez_y),
                backend_utils.tensor_to_ptr(m_lambda_ez_x),
                backend_utils.tensor_to_ptr(m_lambda_ex_z),
                backend_utils.tensor_to_ptr(m_lambda_ex_y),
                backend_utils.tensor_to_ptr(m_lambda_ey_x),
                backend_utils.tensor_to_ptr(m_lambda_hz_y),
                backend_utils.tensor_to_ptr(m_lambda_hy_z),
                backend_utils.tensor_to_ptr(m_lambda_hx_z),
                backend_utils.tensor_to_ptr(m_lambda_hz_x),
                backend_utils.tensor_to_ptr(m_lambda_hy_x),
                backend_utils.tensor_to_ptr(m_lambda_hx_y),
                backend_utils.tensor_to_ptr(store_ex),
                backend_utils.tensor_to_ptr(store_ex_host),
                ctypes.cast(ctx.backward_storage_filename_arrays[0], ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_ey),
                backend_utils.tensor_to_ptr(store_ey_host),
                ctypes.cast(ctx.backward_storage_filename_arrays[1], ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_ez),
                backend_utils.tensor_to_ptr(store_ez_host),
                ctypes.cast(ctx.backward_storage_filename_arrays[2], ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_curl_x),
                backend_utils.tensor_to_ptr(store_curl_x_host),
                ctypes.cast(ctx.backward_storage_filename_arrays[3], ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_curl_y),
                backend_utils.tensor_to_ptr(store_curl_y_host),
                ctypes.cast(ctx.backward_storage_filename_arrays[4], ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_curl_z),
                backend_utils.tensor_to_ptr(store_curl_z_host),
                ctypes.cast(ctx.backward_storage_filename_arrays[5], ctypes.c_void_p)
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(grad_f),
                backend_utils.tensor_to_ptr(grad_ca),
                backend_utils.tensor_to_ptr(grad_cb),
                backend_utils.tensor_to_ptr(grad_eps),
                backend_utils.tensor_to_ptr(grad_sigma),
                backend_utils.tensor_to_ptr(grad_ca_shot),
                backend_utils.tensor_to_ptr(grad_cb_shot),
                chunk_idx == 0,
                backend_utils.tensor_to_ptr(az),
                backend_utils.tensor_to_ptr(bz),
                backend_utils.tensor_to_ptr(az_h),
                backend_utils.tensor_to_ptr(bz_h),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(kz),
                backend_utils.tensor_to_ptr(kz_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                float(meta["rdz"]),
                float(meta["rdy"]),
                float(meta["rdx"]),
                dt,
                step_nt,
                n_shots,
                nz,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                storage_mode,
                storage_format,
                shot_bytes_uncomp,
                ca_requires_grad,
                cb_requires_grad,
                ca_batched,
                cb_batched,
                cq_batched,
                step,
                pml_z0,
                pml_y0,
                pml_x0,
                pml_z1,
                pml_y1,
                pml_x1,
                source_component_idx,
                receiver_component_idx,
                n_threads,
                device_idx,
                execution_backend_id,
                compute_stream_handle,
                storage_stream_handle,
            )

            if backward_callback is not None:
                callback_gradients = {}
                if ca_requires_grad:
                    callback_gradients["ca"] = grad_ca
                if cb_requires_grad:
                    callback_gradients["cb"] = grad_cb
                if ca_requires_grad or cb_requires_grad:
                    callback_gradients["epsilon"] = grad_eps
                    callback_gradients["sigma"] = grad_sigma
                backward_callback(
                    CallbackState(
                        dt=dt,
                        step=step - 1,
                        nt=nt,
                        wavefields={
                            "lambda_Ex": lambda_ex,
                            "lambda_Ey": lambda_ey,
                            "lambda_Ez": lambda_ez,
                            "lambda_Hx": lambda_hx,
                            "lambda_Hy": lambda_hy,
                            "lambda_Hz": lambda_hz,
                        },
                        models=models,
                        gradients=callback_gradients,
                        fd_pad=list(fd_pad),
                        pml_width=list(pml_width),
                        is_backward=True,
                    )
                )

        if direct_material_grad and n_sources > 0:
            from ..utils import EP0

            grad_f_view = grad_f.reshape(nt, n_shots, n_sources)
            source_view = source_amplitudes_scaled.reshape(nt, n_shots, n_sources)
            cb_flat = cb.reshape(n_shots if cb_batched else 1, nz * ny * nx)
            if not cb_batched:
                cb_flat = cb_flat.expand(n_shots, -1)
            valid_sources = sources_i >= 0
            source_idx = sources_i.clamp_min(0)
            cb_at_src = cb_flat.gather(1, source_idx)
            source_factor = grad_f_view * source_view * cb_at_src[None, :, :]
            if direct_eps_requires_grad:
                correction_eps = (source_factor * (EP0 / dt)).sum(dim=0)
                correction_eps = correction_eps.masked_fill(~valid_sources, 0)
                if ca_batched:
                    grad_eps.reshape(n_shots, -1).scatter_add_(
                        1, source_idx, correction_eps
                    )
                else:
                    grad_eps.reshape(-1).scatter_add_(
                        0,
                        source_idx.reshape(-1),
                        correction_eps.reshape(-1),
                    )
            if direct_sigma_requires_grad:
                correction_sigma = (source_factor * 0.5).sum(dim=0)
                correction_sigma = correction_sigma.masked_fill(~valid_sources, 0)
                if ca_batched:
                    grad_sigma.reshape(n_shots, -1).scatter_add_(
                        1, source_idx, correction_sigma
                    )
                else:
                    grad_sigma.reshape(-1).scatter_add_(
                        0,
                        source_idx.reshape(-1),
                        correction_sigma.reshape(-1),
                    )

        if (
            eonly_snapshots
            and not direct_material_grad
            and cb_requires_grad
            and n_sources > 0
        ):
            grad_f_view = grad_f.reshape(nt, n_shots, n_sources)
            source_view = source_amplitudes_scaled.reshape(nt, n_shots, n_sources)
            cb_flat = cb.reshape(n_shots if cb_batched else 1, nz * ny * nx)
            if not cb_batched:
                cb_flat = cb_flat.expand(n_shots, -1)
            valid_sources = sources_i >= 0
            source_idx = sources_i.clamp_min(0)
            cb_at_src = cb_flat.gather(1, source_idx)
            correction = -(grad_f_view * source_view / cb_at_src[None, :, :])
            correction = correction.sum(dim=0).masked_fill(~valid_sources, 0)
            if cb_batched:
                grad_cb.reshape(n_shots, -1).scatter_add_(1, source_idx, correction)
            else:
                grad_cb.reshape(-1).scatter_add_(
                    0,
                    source_idx.reshape(-1),
                    correction.reshape(-1),
                )

        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None

        grad_ca_out = (
            (
                grad_ca.unsqueeze(0)
                if coeff_ca_requires_grad and not ca_batched
                else grad_ca
            )
            if coeff_ca_requires_grad
            else None
        )
        grad_cb_out = (
            (
                grad_cb.unsqueeze(0)
                if coeff_cb_requires_grad and not cb_batched
                else grad_cb
            )
            if coeff_cb_requires_grad
            else None
        )
        grad_eps_out = grad_eps if direct_eps_requires_grad else None
        grad_sigma_out = grad_sigma if direct_sigma_requires_grad else None
        return (
            grad_ca_out,
            grad_cb_out,
            None,
            grad_eps_out,
            grad_sigma_out,
            grad_f_flat,
            None,
            None,
            None,
            None,
        )


class Maxwell3DCheckpointForwardFunc(torch.autograd.Function):
    """Checkpoint/recompute prototypes for 3D CUDA gradients.

    The segmented scheduler stores full wavefield/CPML states at segment
    boundaries. The Revolve-style scheduler stores a bounded checkpoint pool and
    lets the CUDA backend regenerate segment starts during backward. Both
    schedulers replay leaf segments with ordinary native snapshot storage, then
    feed those snapshots into the existing native adjoint.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        epsilon_padded: torch.Tensor,
        sigma_padded: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
        profiles: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, torch.Tensor],
        wavefields: tuple[torch.Tensor, ...],
        meta: dict[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        from .. import backend_utils

        del epsilon_padded, sigma_padded

        sources_i, receivers_i = indices
        if len(wavefields) != _MAXWELL3D_NUM_WAVEFIELDS:
            raise RuntimeError("checkpointed 3D Maxwell received invalid wavefields.")

        Ex = wavefields[0]
        device = Ex.device
        dtype = Ex.dtype

        nt = int(meta["nt"])
        n_shots = int(meta["n_shots"])
        nz = int(meta["nz"])
        ny = int(meta["ny"])
        nx = int(meta["nx"])
        n_sources = int(meta["n_sources"])
        n_receivers = int(meta["n_receivers"])
        accuracy = int(meta["accuracy"])
        pml_z0 = int(meta["pml_z0"])
        pml_y0 = int(meta["pml_y0"])
        pml_x0 = int(meta["pml_x0"])
        pml_z1 = int(meta["pml_z1"])
        pml_y1 = int(meta["pml_y1"])
        pml_x1 = int(meta["pml_x1"])
        source_component_idx = int(meta["source_component_idx"])
        receiver_component_idx = int(meta["receiver_component_idx"])
        n_threads = int(meta["n_threads"])
        ca_batched = bool(meta.get("ca_batched", False))
        cb_batched = bool(meta.get("cb_batched", False))
        cq_batched = bool(meta.get("cq_batched", False))
        ca_requires_grad = bool(ca.requires_grad)
        cb_requires_grad = bool(cb.requires_grad)
        if not (ca_requires_grad or cb_requires_grad):
            raise RuntimeError(
                "Maxwell3DCheckpointForwardFunc requires material gradients."
            )

        segment_steps = int(meta["checkpoint_segment_steps"])
        segment_steps = max(1, min(segment_steps, nt if nt > 0 else 1))
        checkpoint_scheduler = str(meta.get("checkpoint_scheduler", "segmented"))
        checkpoint_scheduler = checkpoint_scheduler.lower()
        if checkpoint_scheduler not in {"segmented", "revolve"}:
            raise RuntimeError(
                f"Unsupported 3D checkpoint scheduler {checkpoint_scheduler!r}."
            )
        segment_starts = list(range(0, nt, segment_steps))
        num_segments = len(segment_starts)
        checkpoint_tensors: tuple[torch.Tensor, ...] = ()
        checkpoint_pool_slots = 0
        checkpoint_pool: torch.Tensor | None = None
        if checkpoint_scheduler == "revolve":
            checkpoint_pool_slots = (
                1 if num_segments <= 1 else int(math.ceil(math.log2(num_segments))) + 1
            )
            checkpoint_pool = torch.empty(
                (
                    checkpoint_pool_slots,
                    _MAXWELL3D_NUM_WAVEFIELDS,
                    n_shots,
                    nz,
                    ny,
                    nx,
                ),
                device=device,
                dtype=dtype,
            )
            for field_idx, field in enumerate(wavefields):
                checkpoint_pool[0, field_idx].copy_(field)
        else:
            checkpoint_tensors = tuple(
                torch.empty(
                    (num_segments, *field.shape),
                    device=device,
                    dtype=field.dtype,
                )
                for field in wavefields
            )

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)

        _, store_dtype, _, storage_format = _resolve_storage_compression(
            meta["storage_compression"],
            dtype,
            device,
            context="storage_compression",
        )
        shot_numel = nz * ny * nx
        shot_bytes_uncomp = shot_numel * store_dtype.itemsize
        empty_store = torch.empty(0, device=device, dtype=store_dtype)
        no_stores = (
            empty_store,
            empty_store,
            0,
            empty_store,
            empty_store,
            0,
            empty_store,
            empty_store,
            0,
            empty_store,
            empty_store,
            0,
            empty_store,
            empty_store,
            0,
            empty_store,
            empty_store,
            0,
        )

        forward_func = backend_utils.get_backend_function(
            "maxwell_3d", "forward_with_storage", accuracy, dtype, device
        )
        compute_stream_handle, storage_stream_handle, stream_keepalive = (
            _make_storage_streams(device, STORAGE_NONE)
        )
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        if checkpoint_scheduler == "revolve":
            _call_forward_with_storage(
                backend_utils,
                forward_func,
                ca=ca,
                cb=cb,
                cq=cq,
                source_amplitudes_scaled=source_amplitudes_scaled,
                wavefields=wavefields,
                receiver_amplitudes=receiver_amplitudes,
                stores=no_stores,
                profiles=profiles,
                sources_i=sources_i,
                receivers_i=receivers_i,
                rdz=float(meta["rdz"]),
                rdy=float(meta["rdy"]),
                rdx=float(meta["rdx"]),
                dt=float(meta["dt"]),
                step_nt=nt,
                n_shots=n_shots,
                nz=nz,
                ny=ny,
                nx=nx,
                n_sources=n_sources,
                n_receivers=n_receivers,
                step_ratio=1,
                storage_mode=STORAGE_NONE,
                storage_format=storage_format,
                shot_bytes_uncomp=shot_bytes_uncomp,
                ca_requires_grad=False,
                cb_requires_grad=False,
                ca_batched=ca_batched,
                cb_batched=cb_batched,
                cq_batched=cq_batched,
                start_t=0,
                pml_z0=pml_z0,
                pml_y0=pml_y0,
                pml_x0=pml_x0,
                pml_z1=pml_z1,
                pml_y1=pml_y1,
                pml_x1=pml_x1,
                source_component_idx=source_component_idx,
                receiver_component_idx=receiver_component_idx,
                n_threads=n_threads,
                device_idx=device_idx,
                execution_backend_id=0,
                compute_stream_handle=compute_stream_handle,
                storage_stream_handle=storage_stream_handle,
            )
        else:
            for segment_idx, start_t in enumerate(segment_starts):
                for checkpoint, field in zip(checkpoint_tensors, wavefields):
                    checkpoint[segment_idx].copy_(field)
                step_nt = min(segment_steps, nt - start_t)
                _call_forward_with_storage(
                    backend_utils,
                    forward_func,
                    ca=ca,
                    cb=cb,
                    cq=cq,
                    source_amplitudes_scaled=source_amplitudes_scaled,
                    wavefields=wavefields,
                    receiver_amplitudes=receiver_amplitudes,
                    stores=no_stores,
                    profiles=profiles,
                    sources_i=sources_i,
                    receivers_i=receivers_i,
                    rdz=float(meta["rdz"]),
                    rdy=float(meta["rdy"]),
                    rdx=float(meta["rdx"]),
                    dt=float(meta["dt"]),
                    step_nt=step_nt,
                    n_shots=n_shots,
                    nz=nz,
                    ny=ny,
                    nx=nx,
                    n_sources=n_sources,
                    n_receivers=n_receivers,
                    step_ratio=1,
                    storage_mode=STORAGE_NONE,
                    storage_format=storage_format,
                    shot_bytes_uncomp=shot_bytes_uncomp,
                    ca_requires_grad=False,
                    cb_requires_grad=False,
                    ca_batched=ca_batched,
                    cb_batched=cb_batched,
                    cq_batched=cq_batched,
                    start_t=start_t,
                    pml_z0=pml_z0,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_z1=pml_z1,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    source_component_idx=source_component_idx,
                    receiver_component_idx=receiver_component_idx,
                    n_threads=n_threads,
                    device_idx=device_idx,
                    execution_backend_id=0,
                    compute_stream_handle=compute_stream_handle,
                    storage_stream_handle=storage_stream_handle,
                )

        checkpoint_payload = (
            (checkpoint_pool,) if checkpoint_pool is not None else checkpoint_tensors
        )
        ctx.save_for_backward(
            ca,
            cb,
            cq,
            *profiles,
            sources_i,
            receivers_i,
            source_amplitudes_scaled,
            *checkpoint_payload,
        )
        ctx.meta = {
            "dt": float(meta["dt"]),
            "nt": nt,
            "n_shots": n_shots,
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "n_sources": n_sources,
            "n_receivers": n_receivers,
            "accuracy": accuracy,
            "pml_z0": pml_z0,
            "pml_y0": pml_y0,
            "pml_x0": pml_x0,
            "pml_z1": pml_z1,
            "pml_y1": pml_y1,
            "pml_x1": pml_x1,
            "source_component_idx": source_component_idx,
            "receiver_component_idx": receiver_component_idx,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
            "n_threads": n_threads,
            "rdz": float(meta["rdz"]),
            "rdy": float(meta["rdy"]),
            "rdx": float(meta["rdx"]),
            "storage_format": storage_format,
            "store_dtype": store_dtype,
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "checkpoint_segment_steps": segment_steps,
            "num_checkpoint_segments": num_segments,
            "checkpoint_scheduler": checkpoint_scheduler,
            "checkpoint_pool_slots": checkpoint_pool_slots,
            "checkpoint_eonly_snapshots": bool(ca_requires_grad and cb_requires_grad),
            "ca_batched": ca_batched,
            "cb_batched": cb_batched,
            "cq_batched": cq_batched,
        }
        ctx.stream_keepalive = stream_keepalive

        return (*wavefields, receiver_amplitudes)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        from .. import backend_utils

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        profiles = tuple(saved[3:21])
        sources_i, receivers_i = saved[21], saved[22]
        source_amplitudes_scaled = saved[23]

        meta = ctx.meta
        checkpoint_scheduler = str(meta.get("checkpoint_scheduler", "segmented"))
        checkpoint_scheduler = checkpoint_scheduler.lower()
        if checkpoint_scheduler == "revolve":
            checkpoint_pool = saved[24]
            checkpoint_tensors: tuple[torch.Tensor, ...] = ()
        else:
            checkpoint_pool = None
            checkpoint_tensors = tuple(saved[24 : 24 + _MAXWELL3D_NUM_WAVEFIELDS])
        device = ca.device
        dtype = ca.dtype

        nt = int(meta["nt"])
        n_shots = int(meta["n_shots"])
        nz = int(meta["nz"])
        ny = int(meta["ny"])
        nx = int(meta["nx"])
        n_sources = int(meta["n_sources"])
        n_receivers = int(meta["n_receivers"])
        accuracy = int(meta["accuracy"])
        pml_z0 = int(meta["pml_z0"])
        pml_y0 = int(meta["pml_y0"])
        pml_x0 = int(meta["pml_x0"])
        pml_z1 = int(meta["pml_z1"])
        pml_y1 = int(meta["pml_y1"])
        pml_x1 = int(meta["pml_x1"])
        source_component_idx = int(meta["source_component_idx"])
        receiver_component_idx = int(meta["receiver_component_idx"])
        ca_requires_grad = bool(meta["ca_requires_grad"])
        cb_requires_grad = bool(meta["cb_requires_grad"])
        n_threads = int(meta["n_threads"])
        dt = float(meta["dt"])
        storage_format = int(meta["storage_format"])
        shot_bytes_uncomp = int(meta["shot_bytes_uncomp"])
        segment_steps = int(meta["checkpoint_segment_steps"])
        num_segments = int(meta["num_checkpoint_segments"])
        ca_batched = bool(meta.get("ca_batched", False))
        cb_batched = bool(meta.get("cb_batched", False))
        cq_batched = bool(meta.get("cq_batched", False))
        eonly_snapshots = bool(meta.get("checkpoint_eonly_snapshots", False))
        execution_backend_id = 1 if eonly_snapshots else 0

        grad_r = grad_outputs[-1]
        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(nt, n_shots, n_receivers, device=device, dtype=dtype)
        else:
            grad_r = grad_r.contiguous()

        lambda_ex = torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
        lambda_ey = torch.zeros_like(lambda_ex)
        lambda_ez = torch.zeros_like(lambda_ex)
        lambda_hx = torch.zeros_like(lambda_ex)
        lambda_hy = torch.zeros_like(lambda_ex)
        lambda_hz = torch.zeros_like(lambda_ex)
        lambdas = (
            lambda_ex,
            lambda_ey,
            lambda_ez,
            lambda_hx,
            lambda_hy,
            lambda_hz,
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
            torch.zeros_like(lambda_ex),
        )

        if n_sources > 0:
            grad_f = torch.empty(nt, n_shots, n_sources, device=device, dtype=dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad:
            grad_ca = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if ca_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
            grad_ca_shot = (
                torch.empty(0, device=device, dtype=dtype)
                if ca_batched
                else torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            )
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if cb_requires_grad:
            grad_cb = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if cb_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
            grad_cb_shot = (
                torch.empty(0, device=device, dtype=dtype)
                if cb_batched
                else torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
            )
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        if ca_requires_grad or cb_requires_grad:
            grad_eps = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if ca_batched or cb_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
            grad_sigma = torch.zeros_like(grad_eps)
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
            grad_sigma = torch.empty(0, device=device, dtype=dtype)

        store_dtype = (
            checkpoint_pool.dtype
            if checkpoint_pool is not None
            else checkpoint_tensors[0].dtype
        )
        if storage_format != 0:
            store_dtype = torch.bfloat16
        empty_store = torch.empty(0, device=device, dtype=store_dtype)
        max_segment_steps = min(segment_steps, nt if nt > 0 else segment_steps)

        def alloc_store(requires_grad_cond: bool) -> torch.Tensor:
            if not requires_grad_cond:
                return empty_store
            return torch.empty(
                max_segment_steps,
                n_shots,
                nz,
                ny,
                nx,
                device=device,
                dtype=store_dtype,
            )

        def alloc_final_e(requires_grad_cond: bool) -> torch.Tensor:
            if not requires_grad_cond:
                return empty_store
            return torch.empty(n_shots, nz, ny, nx, device=device, dtype=store_dtype)

        store_ex = alloc_store(ca_requires_grad)
        store_ey = alloc_store(ca_requires_grad)
        store_ez = alloc_store(ca_requires_grad)
        if eonly_snapshots:
            store_curl_x = alloc_final_e(True)
            store_curl_y = alloc_final_e(True)
            store_curl_z = alloc_final_e(True)
        else:
            store_curl_x = alloc_store(cb_requires_grad)
            store_curl_y = alloc_store(cb_requires_grad)
            store_curl_z = alloc_store(cb_requires_grad)
        stores = (
            store_ex,
            empty_store,
            0,
            store_ey,
            empty_store,
            0,
            store_ez,
            empty_store,
            0,
            store_curl_x,
            empty_store,
            0,
            store_curl_y,
            empty_store,
            0,
            store_curl_z,
            empty_store,
            0,
        )

        compute_stream_handle, storage_stream_handle, stream_keepalive = (
            _make_storage_streams(device, STORAGE_DEVICE)
        )
        ctx.stream_keepalive = stream_keepalive
        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        if checkpoint_scheduler == "revolve":
            if checkpoint_pool is None:
                raise RuntimeError("checkpoint_revolve missing checkpoint pool.")
            checkpoint_func = backend_utils.get_backend_function(
                "maxwell_3d",
                "checkpoint_revolve_backward",
                accuracy,
                dtype,
                device,
            )
            scratch_wavefields = torch.empty(
                (2, _MAXWELL3D_NUM_WAVEFIELDS, n_shots, nz, ny, nx),
                device=device,
                dtype=dtype,
            )
            if num_segments > 0:
                _call_checkpoint_revolve_backward(
                    backend_utils,
                    checkpoint_func,
                    ca=ca,
                    cb=cb,
                    cq=cq,
                    source_amplitudes_scaled=source_amplitudes_scaled,
                    grad_r=grad_r,
                    checkpoint_pool=checkpoint_pool,
                    scratch_wavefields=scratch_wavefields,
                    lambdas=lambdas,
                    stores=stores,
                    grad_f=grad_f,
                    grad_ca=grad_ca,
                    grad_cb=grad_cb,
                    grad_eps=grad_eps,
                    grad_sigma=grad_sigma,
                    grad_ca_shot=grad_ca_shot,
                    grad_cb_shot=grad_cb_shot,
                    profiles=profiles,
                    sources_i=sources_i,
                    receivers_i=receivers_i,
                    rdz=float(meta["rdz"]),
                    rdy=float(meta["rdy"]),
                    rdx=float(meta["rdx"]),
                    dt=dt,
                    nt=nt,
                    n_shots=n_shots,
                    nz=nz,
                    ny=ny,
                    nx=nx,
                    n_sources=n_sources,
                    n_receivers=n_receivers,
                    storage_mode=STORAGE_DEVICE,
                    storage_format=storage_format,
                    shot_bytes_uncomp=shot_bytes_uncomp,
                    ca_requires_grad=ca_requires_grad,
                    cb_requires_grad=cb_requires_grad,
                    ca_batched=ca_batched,
                    cb_batched=cb_batched,
                    cq_batched=cq_batched,
                    pml_z0=pml_z0,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_z1=pml_z1,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    source_component_idx=source_component_idx,
                    receiver_component_idx=receiver_component_idx,
                    n_threads=n_threads,
                    device_idx=device_idx,
                    execution_backend_id=execution_backend_id,
                    segment_steps=segment_steps,
                    num_segments=num_segments,
                    pool_slots=int(meta["checkpoint_pool_slots"]),
                    compute_stream_handle=compute_stream_handle,
                    storage_stream_handle=storage_stream_handle,
                )
        else:
            forward_func = backend_utils.get_backend_function(
                "maxwell_3d", "forward_with_storage", accuracy, dtype, device
            )
            backward_func = backend_utils.get_backend_function(
                "maxwell_3d", "backward", accuracy, dtype, device
            )
            if n_receivers > 0:
                receiver_scratch = torch.empty(
                    max_segment_steps,
                    n_shots,
                    n_receivers,
                    device=device,
                    dtype=dtype,
                )
            else:
                receiver_scratch = torch.empty(0, device=device, dtype=dtype)
            source_view = (
                source_amplitudes_scaled.reshape(nt, n_shots, n_sources)
                if n_sources > 0
                else torch.empty(0, device=device, dtype=dtype)
            )

            for reverse_idx, segment_idx in enumerate(range(num_segments - 1, -1, -1)):
                segment_start = segment_idx * segment_steps
                step_nt = min(segment_steps, nt - segment_start)
                replay_wavefields = tuple(
                    checkpoint[segment_idx].clone() for checkpoint in checkpoint_tensors
                )
                source_segment = (
                    source_view[segment_start : segment_start + step_nt]
                    .contiguous()
                    .reshape(-1)
                    if n_sources > 0
                    else source_amplitudes_scaled
                )
                receiver_segment = (
                    receiver_scratch[:step_nt] if n_receivers > 0 else receiver_scratch
                )
                _call_forward_with_storage(
                    backend_utils,
                    forward_func,
                    ca=ca,
                    cb=cb,
                    cq=cq,
                    source_amplitudes_scaled=source_segment,
                    wavefields=replay_wavefields,
                    receiver_amplitudes=receiver_segment,
                    stores=stores,
                    profiles=profiles,
                    sources_i=sources_i,
                    receivers_i=receivers_i,
                    rdz=float(meta["rdz"]),
                    rdy=float(meta["rdy"]),
                    rdx=float(meta["rdx"]),
                    dt=dt,
                    step_nt=step_nt,
                    n_shots=n_shots,
                    nz=nz,
                    ny=ny,
                    nx=nx,
                    n_sources=n_sources,
                    n_receivers=n_receivers,
                    step_ratio=1,
                    storage_mode=STORAGE_DEVICE,
                    storage_format=storage_format,
                    shot_bytes_uncomp=shot_bytes_uncomp,
                    ca_requires_grad=ca_requires_grad,
                    cb_requires_grad=cb_requires_grad,
                    ca_batched=ca_batched,
                    cb_batched=cb_batched,
                    cq_batched=cq_batched,
                    start_t=0,
                    pml_z0=pml_z0,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_z1=pml_z1,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    source_component_idx=source_component_idx,
                    receiver_component_idx=receiver_component_idx,
                    n_threads=n_threads,
                    device_idx=device_idx,
                    execution_backend_id=execution_backend_id,
                    compute_stream_handle=compute_stream_handle,
                    storage_stream_handle=storage_stream_handle,
                )

                grad_r_segment = grad_r[
                    segment_start : segment_start + step_nt
                ].contiguous()
                grad_f_segment = (
                    torch.empty(step_nt, n_shots, n_sources, device=device, dtype=dtype)
                    if n_sources > 0
                    else torch.empty(0, device=device, dtype=dtype)
                )
                _call_backward(
                    backend_utils,
                    backward_func,
                    ca=ca,
                    cb=cb,
                    cq=cq,
                    grad_r=grad_r_segment,
                    lambdas=lambdas,
                    stores=stores,
                    grad_f=grad_f_segment,
                    grad_ca=grad_ca,
                    grad_cb=grad_cb,
                    grad_eps=grad_eps,
                    grad_sigma=grad_sigma,
                    grad_ca_shot=grad_ca_shot,
                    grad_cb_shot=grad_cb_shot,
                    zero_grad_on_entry=reverse_idx == 0,
                    profiles=profiles,
                    sources_i=sources_i,
                    receivers_i=receivers_i,
                    rdz=float(meta["rdz"]),
                    rdy=float(meta["rdy"]),
                    rdx=float(meta["rdx"]),
                    dt=dt,
                    step_nt=step_nt,
                    n_shots=n_shots,
                    nz=nz,
                    ny=ny,
                    nx=nx,
                    n_sources=n_sources,
                    n_receivers=n_receivers,
                    step_ratio=1,
                    storage_mode=STORAGE_DEVICE,
                    storage_format=storage_format,
                    shot_bytes_uncomp=shot_bytes_uncomp,
                    ca_requires_grad=ca_requires_grad,
                    cb_requires_grad=cb_requires_grad,
                    ca_batched=ca_batched,
                    cb_batched=cb_batched,
                    cq_batched=cq_batched,
                    start_t=step_nt,
                    pml_z0=pml_z0,
                    pml_y0=pml_y0,
                    pml_x0=pml_x0,
                    pml_z1=pml_z1,
                    pml_y1=pml_y1,
                    pml_x1=pml_x1,
                    source_component_idx=source_component_idx,
                    receiver_component_idx=receiver_component_idx,
                    n_threads=n_threads,
                    device_idx=device_idx,
                    execution_backend_id=execution_backend_id,
                    compute_stream_handle=compute_stream_handle,
                    storage_stream_handle=storage_stream_handle,
                )
                if n_sources > 0:
                    grad_f[segment_start : segment_start + step_nt].copy_(
                        grad_f_segment
                    )

        if eonly_snapshots and cb_requires_grad and n_sources > 0:
            grad_f_view = grad_f.reshape(nt, n_shots, n_sources)
            source_view_full = source_amplitudes_scaled.reshape(nt, n_shots, n_sources)
            cb_flat = cb.reshape(n_shots if cb_batched else 1, nz * ny * nx)
            if not cb_batched:
                cb_flat = cb_flat.expand(n_shots, -1)
            valid_sources = sources_i >= 0
            source_idx = sources_i.clamp_min(0)
            cb_at_src = cb_flat.gather(1, source_idx)
            correction = -(grad_f_view * source_view_full / cb_at_src[None, :, :])
            correction = correction.sum(dim=0).masked_fill(~valid_sources, 0)
            if cb_batched:
                grad_cb.reshape(n_shots, -1).scatter_add_(1, source_idx, correction)
            else:
                grad_cb.reshape(-1).scatter_add_(
                    0,
                    source_idx.reshape(-1),
                    correction.reshape(-1),
                )

        grad_f_flat = (
            grad_f.reshape(nt * n_shots * n_sources) if n_sources > 0 else None
        )
        grad_ca_out = (
            (grad_ca.unsqueeze(0) if not ca_batched else grad_ca)
            if ca_requires_grad
            else None
        )
        grad_cb_out = (
            (grad_cb.unsqueeze(0) if not cb_batched else grad_cb)
            if cb_requires_grad
            else None
        )
        return (
            grad_ca_out,
            grad_cb_out,
            None,
            None,
            None,
            grad_f_flat,
            None,
            None,
            None,
            None,
        )


__all__ = ["Maxwell3DForwardFunc", "Maxwell3DCheckpointForwardFunc"]
