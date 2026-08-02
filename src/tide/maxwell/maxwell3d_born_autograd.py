from collections.abc import Sequence
from typing import Any

import torch

from ..storage import SnapshotAllocator, resolve_snapshot_storage
from ..validation import validate_model_gradient_sampling_interval
from .common import (
    ReceiverMisfit,
    _clone_param,
    _directional_receiver_hvp,
    _directional_receiver_hvp_from_born,
    _make_storage_streams,
)


class Born3DForwardFunc(torch.autograd.Function):
    """Autograd function for the native 3D Born operator."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        dca: torch.Tensor,
        dcb: torch.Tensor,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        f0: torch.Tensor,
        df: torch.Tensor,
        profiles: tuple[torch.Tensor, ...],
        indices: tuple[torch.Tensor, torch.Tensor],
        background_wavefields: tuple[torch.Tensor, ...],
        scattered_wavefields: tuple[torch.Tensor, ...],
        meta: dict[str, Any],
    ) -> tuple[torch.Tensor, ...]:
        from .. import backend_utils

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
        ) = background_wavefields
        (
            dEx,
            dEy,
            dEz,
            dHx,
            dHy,
            dHz,
            dm_hz_y,
            dm_hy_z,
            dm_hx_z,
            dm_hz_x,
            dm_hy_x,
            dm_hx_y,
            dm_ey_z,
            dm_ez_y,
            dm_ez_x,
            dm_ex_z,
            dm_ex_y,
            dm_ey_x,
        ) = scattered_wavefields

        device = dca.device
        dtype = dca.dtype

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
        step_ratio = int(meta.get("step_ratio", 1))
        if step_ratio < 1:
            step_ratio = 1
        if nt > 0:
            step_ratio = min(step_ratio, nt)
        n_threads = int(meta["n_threads"])
        rdz = float(meta["rdz"])
        rdy = float(meta["rdy"])
        rdx = float(meta["rdx"])
        dt = float(meta["dt"])
        backend_device = meta["backend_device"]

        dca_requires_grad = bool(dca.requires_grad)
        dcb_requires_grad = bool(dcb.requires_grad)
        ca_requires_grad = bool(ca.requires_grad)
        cb_requires_grad = bool(cb.requires_grad)
        f0_requires_grad = bool(f0.requires_grad)
        df_requires_grad = bool(df.requires_grad)
        background_grad_required = (
            ca_requires_grad or cb_requires_grad or f0_requires_grad
        )
        store_e_needed = dca_requires_grad or ca_requires_grad
        store_curl_needed = dcb_requires_grad or cb_requires_grad
        needs_storage = store_e_needed or store_curl_needed or background_grad_required

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=dtype
            )
            background_receiver_amplitudes = (
                torch.zeros_like(receiver_amplitudes)
                if meta.get("return_background_receiver_amplitudes", False)
                else torch.empty(0, device=device, dtype=dtype)
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=dtype)
            background_receiver_amplitudes = torch.empty(
                0, device=device, dtype=dtype
            )

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        storage_spec = resolve_snapshot_storage(
            storage_mode="device",
            storage_compression=(
                meta.get("storage_compression", False) if needs_storage else False
            ),
            dtype=dtype,
            device=device,
            nt=nt,
            step_ratio=step_ratio,
            shot_shape=(n_shots, nz, ny, nx),
            enabled=needs_storage,
        )
        snapshot_allocator = SnapshotAllocator(storage_spec, device)
        storage_mode = storage_spec.mode
        storage_format = storage_spec.format
        shot_bytes_uncomp = storage_spec.shot_bytes if needs_storage else 0

        if needs_storage:
            if storage_format != 0 and device.type != "cuda":
                raise NotImplementedError(
                    "3D Born BF16 snapshot storage is currently supported only on CUDA."
                )
            compute_stream_handle, storage_stream_handle, stream_keepalive = (
                _make_storage_streams(device, storage_mode)
            )
            store_ex, store_ey, store_ez = snapshot_allocator.group(
                3, store_e_needed
            )
            store_curl_x, store_curl_y, store_curl_z = snapshot_allocator.group(
                3, store_curl_needed
            )
            store_dex, store_dey, store_dez = snapshot_allocator.group(
                3, ca_requires_grad
            )
            store_dcurl_x, store_dcurl_y, store_dcurl_z = snapshot_allocator.group(
                3, cb_requires_grad
            )
            empty_host = snapshot_allocator.empty()

            forward_func = backend_utils.get_backend_function(
                "maxwell_3d",
                "born_forward_with_storage",
                accuracy,
                dtype,
                backend_device,
            )
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(dca),
                backend_utils.tensor_to_ptr(dcb),
                backend_utils.tensor_to_ptr(f0),
                backend_utils.tensor_to_ptr(df),
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
                backend_utils.tensor_to_ptr(dEx),
                backend_utils.tensor_to_ptr(dEy),
                backend_utils.tensor_to_ptr(dEz),
                backend_utils.tensor_to_ptr(dHx),
                backend_utils.tensor_to_ptr(dHy),
                backend_utils.tensor_to_ptr(dHz),
                backend_utils.tensor_to_ptr(dm_hz_y),
                backend_utils.tensor_to_ptr(dm_hy_z),
                backend_utils.tensor_to_ptr(dm_hx_z),
                backend_utils.tensor_to_ptr(dm_hz_x),
                backend_utils.tensor_to_ptr(dm_hy_x),
                backend_utils.tensor_to_ptr(dm_hx_y),
                backend_utils.tensor_to_ptr(dm_ey_z),
                backend_utils.tensor_to_ptr(dm_ez_y),
                backend_utils.tensor_to_ptr(dm_ez_x),
                backend_utils.tensor_to_ptr(dm_ex_z),
                backend_utils.tensor_to_ptr(dm_ex_y),
                backend_utils.tensor_to_ptr(dm_ey_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(background_receiver_amplitudes),
                backend_utils.tensor_to_ptr(store_ex),
                backend_utils.tensor_to_ptr(empty_host),
                0,
                backend_utils.tensor_to_ptr(store_ey),
                backend_utils.tensor_to_ptr(empty_host),
                0,
                backend_utils.tensor_to_ptr(store_ez),
                backend_utils.tensor_to_ptr(empty_host),
                0,
                backend_utils.tensor_to_ptr(store_curl_x),
                backend_utils.tensor_to_ptr(empty_host),
                0,
                backend_utils.tensor_to_ptr(store_curl_y),
                backend_utils.tensor_to_ptr(empty_host),
                0,
                backend_utils.tensor_to_ptr(store_curl_z),
                backend_utils.tensor_to_ptr(empty_host),
                0,
                backend_utils.tensor_to_ptr(store_dex),
                backend_utils.tensor_to_ptr(store_dey),
                backend_utils.tensor_to_ptr(store_dez),
                backend_utils.tensor_to_ptr(store_dcurl_x),
                backend_utils.tensor_to_ptr(store_dcurl_y),
                backend_utils.tensor_to_ptr(store_dcurl_z),
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
                step_ratio,
                storage_mode,
                storage_format,
                shot_bytes_uncomp,
                store_e_needed,
                store_curl_needed,
                False,
                False,
                False,
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
                0,
                compute_stream_handle,
                storage_stream_handle,
            )
        else:
            compute_stream_handle, _, stream_keepalive = _make_storage_streams(
                device, storage_mode
            )
            (
                store_ex,
                store_ey,
                store_ez,
                store_curl_x,
                store_curl_y,
                store_curl_z,
                store_dex,
                store_dey,
                store_dez,
                store_dcurl_x,
                store_dcurl_y,
                store_dcurl_z,
            ) = snapshot_allocator.group(12, False)

            forward_func = backend_utils.get_backend_function(
                "maxwell_3d",
                "born_forward",
                accuracy,
                dtype,
                backend_device,
            )
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(dca),
                backend_utils.tensor_to_ptr(dcb),
                backend_utils.tensor_to_ptr(f0),
                backend_utils.tensor_to_ptr(df),
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
                backend_utils.tensor_to_ptr(dEx),
                backend_utils.tensor_to_ptr(dEy),
                backend_utils.tensor_to_ptr(dEz),
                backend_utils.tensor_to_ptr(dHx),
                backend_utils.tensor_to_ptr(dHy),
                backend_utils.tensor_to_ptr(dHz),
                backend_utils.tensor_to_ptr(dm_hz_y),
                backend_utils.tensor_to_ptr(dm_hy_z),
                backend_utils.tensor_to_ptr(dm_hx_z),
                backend_utils.tensor_to_ptr(dm_hz_x),
                backend_utils.tensor_to_ptr(dm_hy_x),
                backend_utils.tensor_to_ptr(dm_hx_y),
                backend_utils.tensor_to_ptr(dm_ey_z),
                backend_utils.tensor_to_ptr(dm_ez_y),
                backend_utils.tensor_to_ptr(dm_ez_x),
                backend_utils.tensor_to_ptr(dm_ex_z),
                backend_utils.tensor_to_ptr(dm_ex_y),
                backend_utils.tensor_to_ptr(dm_ey_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(background_receiver_amplitudes),
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
                step_ratio,
                False,
                False,
                False,
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
                0,
                compute_stream_handle,
            )

        ctx.save_for_backward(
            dca,
            dcb,
            ca,
            cb,
            cq,
            f0,
            df,
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
            store_ey,
            store_ez,
            store_curl_x,
            store_curl_y,
            store_curl_z,
            store_dex,
            store_dey,
            store_dez,
            store_dcurl_x,
            store_dcurl_y,
            store_dcurl_z,
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
            "accuracy": accuracy,
            "pml_z0": pml_z0,
            "pml_y0": pml_y0,
            "pml_x0": pml_x0,
            "pml_z1": pml_z1,
            "pml_y1": pml_y1,
            "pml_x1": pml_x1,
            "source_component_idx": source_component_idx,
            "receiver_component_idx": receiver_component_idx,
            "step_ratio": step_ratio,
            "n_threads": n_threads,
            "rdz": rdz,
            "rdy": rdy,
            "rdx": rdx,
            "storage_mode": storage_mode,
            "storage_format": storage_format,
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "dca_requires_grad": dca_requires_grad,
            "dcb_requires_grad": dcb_requires_grad,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
            "f0_requires_grad": f0_requires_grad,
            "df_requires_grad": df_requires_grad,
            "background_grad_required": background_grad_required,
            "backend_device": backend_device,
        }
        ctx.stream_keepalive = stream_keepalive
        ctx.snapshot_allocator = snapshot_allocator
        # Tests may call ``forward`` with a lightweight context rather than an
        # autograd FunctionCtx.  Real autograd contexts expose this method.
        if hasattr(ctx, "mark_non_differentiable"):
            ctx.mark_non_differentiable(background_receiver_amplitudes)

        return (
            dEx,
            dEy,
            dEz,
            dHx,
            dHy,
            dHz,
            dm_hz_y,
            dm_hy_z,
            dm_hx_z,
            dm_hz_x,
            dm_hy_x,
            dm_hx_y,
            dm_ey_z,
            dm_ez_y,
            dm_ez_x,
            dm_ex_z,
            dm_ex_y,
            dm_ey_x,
            receiver_amplitudes,
            background_receiver_amplitudes,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        from .. import backend_utils

        (
            dca,
            dcb,
            ca,
            cb,
            cq,
            f0,
            df,
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
            store_ey,
            store_ez,
            store_curl_x,
            store_curl_y,
            store_curl_z,
            store_dex,
            store_dey,
            store_dez,
            store_dcurl_x,
            store_dcurl_y,
            store_dcurl_z,
        ) = ctx.saved_tensors

        meta = ctx.meta
        device = ca.device
        dtype = ca.dtype

        grad_r = grad_outputs[-2]
        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(
                meta["nt"],
                meta["n_shots"],
                meta["n_receivers"],
                device=device,
                dtype=dtype,
            )
        else:
            grad_r = grad_r.contiguous()

        lambda_ex = torch.zeros(
            meta["n_shots"],
            meta["nz"],
            meta["ny"],
            meta["nx"],
            device=device,
            dtype=dtype,
        )
        lambda_ey = torch.zeros_like(lambda_ex)
        lambda_ez = torch.zeros_like(lambda_ex)
        lambda_hx = torch.zeros_like(lambda_ex)
        lambda_hy = torch.zeros_like(lambda_ex)
        lambda_hz = torch.zeros_like(lambda_ex)

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

        if meta["n_sources"] > 0:
            grad_f = torch.zeros(
                meta["nt"],
                meta["n_shots"],
                meta["n_sources"],
                device=device,
                dtype=dtype,
            )
        else:
            grad_f = torch.empty(0, device=device, dtype=dtype)

        if meta["dca_requires_grad"]:
            grad_ca = torch.zeros(
                meta["nz"], meta["ny"], meta["nx"], device=device, dtype=dtype
            )
            grad_ca_shot = torch.zeros(
                meta["n_shots"],
                meta["nz"],
                meta["ny"],
                meta["nx"],
                device=device,
                dtype=dtype,
            )
        else:
            grad_ca = torch.empty(0, device=device, dtype=dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=dtype)

        if meta["dcb_requires_grad"]:
            grad_cb = torch.zeros(
                meta["nz"], meta["ny"], meta["nx"], device=device, dtype=dtype
            )
            grad_cb_shot = torch.zeros(
                meta["n_shots"],
                meta["nz"],
                meta["ny"],
                meta["nx"],
                device=device,
                dtype=dtype,
            )
        else:
            grad_cb = torch.empty(0, device=device, dtype=dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=dtype)

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        compute_stream_handle, storage_stream_handle, stream_keepalive = (
            _make_storage_streams(device, meta["storage_mode"])
        )
        ctx.stream_keepalive = stream_keepalive

        if meta["background_grad_required"]:
            eta_ex = torch.zeros_like(lambda_ex)
            eta_ey = torch.zeros_like(lambda_ex)
            eta_ez = torch.zeros_like(lambda_ex)
            eta_hx = torch.zeros_like(lambda_ex)
            eta_hy = torch.zeros_like(lambda_ex)
            eta_hz = torch.zeros_like(lambda_ex)

            m_eta_ey_z = torch.zeros_like(lambda_ex)
            m_eta_ez_y = torch.zeros_like(lambda_ex)
            m_eta_ez_x = torch.zeros_like(lambda_ex)
            m_eta_ex_z = torch.zeros_like(lambda_ex)
            m_eta_ex_y = torch.zeros_like(lambda_ex)
            m_eta_ey_x = torch.zeros_like(lambda_ex)
            m_eta_hz_y = torch.zeros_like(lambda_ex)
            m_eta_hy_z = torch.zeros_like(lambda_ex)
            m_eta_hx_z = torch.zeros_like(lambda_ex)
            m_eta_hz_x = torch.zeros_like(lambda_ex)
            m_eta_hy_x = torch.zeros_like(lambda_ex)
            m_eta_hx_y = torch.zeros_like(lambda_ex)

            grad_f0 = (
                torch.zeros(
                    meta["nt"],
                    meta["n_shots"],
                    meta["n_sources"],
                    device=device,
                    dtype=dtype,
                )
                if meta["f0_requires_grad"] and meta["n_sources"] > 0
                else None
            )
            grad_df = (
                torch.zeros(
                    meta["nt"],
                    meta["n_shots"],
                    meta["n_sources"],
                    device=device,
                    dtype=dtype,
                )
                if meta["df_requires_grad"] and meta["n_sources"] > 0
                else None
            )

            model_shape = (meta["nz"], meta["ny"], meta["nx"])
            shot_model_shape = (meta["n_shots"], *model_shape)
            grad_ca_bg = (
                torch.zeros(model_shape, device=device, dtype=dtype)
                if meta["ca_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            grad_cb_bg = (
                torch.zeros(model_shape, device=device, dtype=dtype)
                if meta["cb_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            grad_dca = (
                torch.zeros(model_shape, device=device, dtype=dtype)
                if meta["dca_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            grad_dcb = (
                torch.zeros(model_shape, device=device, dtype=dtype)
                if meta["dcb_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            grad_ca_bg_shot = (
                torch.zeros(shot_model_shape, device=device, dtype=dtype)
                if meta["ca_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            grad_cb_bg_shot = (
                torch.zeros(shot_model_shape, device=device, dtype=dtype)
                if meta["cb_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            grad_dca_shot = (
                torch.zeros(shot_model_shape, device=device, dtype=dtype)
                if meta["dca_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            grad_dcb_shot = (
                torch.zeros(shot_model_shape, device=device, dtype=dtype)
                if meta["dcb_requires_grad"]
                else torch.empty(0, device=device, dtype=dtype)
            )
            eta_source_ex = torch.zeros_like(lambda_ex)
            eta_source_ey = torch.zeros_like(lambda_ex)
            eta_source_ez = torch.zeros_like(lambda_ex)

            backward_func = backend_utils.get_backend_function(
                "maxwell_3d",
                "born_backward_bggrad",
                meta["accuracy"],
                dtype,
                meta["backend_device"],
            )
            backward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(dca),
                backend_utils.tensor_to_ptr(dcb),
                backend_utils.tensor_to_ptr(f0),
                backend_utils.tensor_to_ptr(df),
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
                backend_utils.tensor_to_ptr(eta_ex),
                backend_utils.tensor_to_ptr(eta_ey),
                backend_utils.tensor_to_ptr(eta_ez),
                backend_utils.tensor_to_ptr(eta_hx),
                backend_utils.tensor_to_ptr(eta_hy),
                backend_utils.tensor_to_ptr(eta_hz),
                backend_utils.tensor_to_ptr(m_eta_ey_z),
                backend_utils.tensor_to_ptr(m_eta_ez_y),
                backend_utils.tensor_to_ptr(m_eta_ez_x),
                backend_utils.tensor_to_ptr(m_eta_ex_z),
                backend_utils.tensor_to_ptr(m_eta_ex_y),
                backend_utils.tensor_to_ptr(m_eta_ey_x),
                backend_utils.tensor_to_ptr(m_eta_hz_y),
                backend_utils.tensor_to_ptr(m_eta_hy_z),
                backend_utils.tensor_to_ptr(m_eta_hx_z),
                backend_utils.tensor_to_ptr(m_eta_hz_x),
                backend_utils.tensor_to_ptr(m_eta_hy_x),
                backend_utils.tensor_to_ptr(m_eta_hx_y),
                backend_utils.tensor_to_ptr(store_ex),
                0,
                0,
                backend_utils.tensor_to_ptr(store_ey),
                0,
                0,
                backend_utils.tensor_to_ptr(store_ez),
                0,
                0,
                backend_utils.tensor_to_ptr(store_curl_x),
                0,
                0,
                backend_utils.tensor_to_ptr(store_curl_y),
                0,
                0,
                backend_utils.tensor_to_ptr(store_curl_z),
                0,
                0,
                backend_utils.tensor_to_ptr(store_dex),
                backend_utils.tensor_to_ptr(store_dey),
                backend_utils.tensor_to_ptr(store_dez),
                backend_utils.tensor_to_ptr(store_dcurl_x),
                backend_utils.tensor_to_ptr(store_dcurl_y),
                backend_utils.tensor_to_ptr(store_dcurl_z),
                backend_utils.tensor_to_ptr(grad_f0),
                backend_utils.tensor_to_ptr(grad_df),
                backend_utils.tensor_to_ptr(grad_ca_bg),
                backend_utils.tensor_to_ptr(grad_cb_bg),
                backend_utils.tensor_to_ptr(grad_dca),
                backend_utils.tensor_to_ptr(grad_dcb),
                backend_utils.tensor_to_ptr(grad_ca_bg_shot),
                backend_utils.tensor_to_ptr(grad_cb_bg_shot),
                backend_utils.tensor_to_ptr(grad_dca_shot),
                backend_utils.tensor_to_ptr(grad_dcb_shot),
                backend_utils.tensor_to_ptr(eta_source_ex),
                backend_utils.tensor_to_ptr(eta_source_ey),
                backend_utils.tensor_to_ptr(eta_source_ez),
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
                meta["rdz"],
                meta["rdy"],
                meta["rdx"],
                meta["dt"],
                meta["nt"],
                meta["n_shots"],
                meta["nz"],
                meta["ny"],
                meta["nx"],
                meta["n_sources"],
                meta["n_receivers"],
                meta["step_ratio"],
                meta["storage_mode"],
                meta["storage_format"],
                meta["shot_bytes_uncomp"],
                meta["ca_requires_grad"],
                meta["cb_requires_grad"],
                meta["dca_requires_grad"],
                meta["dcb_requires_grad"],
                False,
                False,
                False,
                meta["nt"],
                meta["pml_z0"],
                meta["pml_y0"],
                meta["pml_x0"],
                meta["pml_z1"],
                meta["pml_y1"],
                meta["pml_x1"],
                meta["source_component_idx"],
                meta["receiver_component_idx"],
                meta["n_threads"],
                device_idx,
                0,
                compute_stream_handle,
                storage_stream_handle,
            )

            grad_f0_flat = (
                grad_f0.reshape(meta["nt"] * meta["n_shots"] * meta["n_sources"])
                if grad_f0 is not None
                else None
            )
            grad_df_flat = (
                grad_df.reshape(meta["nt"] * meta["n_shots"] * meta["n_sources"])
                if grad_df is not None
                else None
            )
            return (
                grad_dca.unsqueeze(0) if meta["dca_requires_grad"] else None,
                grad_dcb.unsqueeze(0) if meta["dcb_requires_grad"] else None,
                grad_ca_bg.unsqueeze(0) if meta["ca_requires_grad"] else None,
                grad_cb_bg.unsqueeze(0) if meta["cb_requires_grad"] else None,
                None,
                grad_f0_flat,
                grad_df_flat,
                None,
                None,
                None,
                None,
                None,
            )

        backward_func = backend_utils.get_backend_function(
            "maxwell_3d",
            "born_backward",
            meta["accuracy"],
            dtype,
            meta["backend_device"],
        )
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
            0,
            0,
            backend_utils.tensor_to_ptr(store_ey),
            0,
            0,
            backend_utils.tensor_to_ptr(store_ez),
            0,
            0,
            backend_utils.tensor_to_ptr(store_curl_x),
            0,
            0,
            backend_utils.tensor_to_ptr(store_curl_y),
            0,
            0,
            backend_utils.tensor_to_ptr(store_curl_z),
            0,
            0,
            backend_utils.tensor_to_ptr(grad_f),
            backend_utils.tensor_to_ptr(grad_ca),
            backend_utils.tensor_to_ptr(grad_cb),
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
            meta["rdz"],
            meta["rdy"],
            meta["rdx"],
            meta["dt"],
            meta["nt"],
            meta["n_shots"],
            meta["nz"],
            meta["ny"],
            meta["nx"],
            meta["n_sources"],
            meta["n_receivers"],
            meta["step_ratio"],
            meta["storage_mode"],
            meta["storage_format"],
            meta["shot_bytes_uncomp"],
            meta["dca_requires_grad"],
            meta["dcb_requires_grad"],
            False,
            False,
            False,
            meta["nt"],
            meta["pml_z0"],
            meta["pml_y0"],
            meta["pml_x0"],
            meta["pml_z1"],
            meta["pml_y1"],
            meta["pml_x1"],
            meta["source_component_idx"],
            meta["receiver_component_idx"],
            meta["n_threads"],
            device_idx,
            0,
            compute_stream_handle,
            storage_stream_handle,
        )

        grad_f_flat = (
            grad_f.reshape(meta["nt"] * meta["n_shots"] * meta["n_sources"])
            if meta["n_sources"] > 0 and meta["df_requires_grad"]
            else None
        )

        grad_ca_out = grad_ca.unsqueeze(0) if meta["dca_requires_grad"] else None
        grad_cb_out = grad_cb.unsqueeze(0) if meta["dcb_requires_grad"] else None

        return (
            grad_ca_out,
            grad_cb_out,
            None,
            None,
            None,
            None,
            grad_f_flat,
            None,
            None,
            None,
            None,
            None,
        )


def maxwell3d_receiver_hvp_naive(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    *,
    vepsilon: torch.Tensor | None = None,
    vsigma: torch.Tensor | None = None,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    observed_data: torch.Tensor,
    misfit_fn: ReceiverMisfit,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    nt: int | None = None,
    model_gradient_sampling_interval: int = 1,
    linearize_source: bool = True,
    source_component: str = "ey",
    receiver_component: str = "ey",
    hessian_mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference 3D receiver-space HVP on the Python Maxwell/Born path."""
    if vepsilon is None and vsigma is None:
        raise ValueError("At least one of vepsilon or vsigma must be provided.")
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    if model_gradient_sampling_interval not in {0, 1}:
        raise NotImplementedError(
            "Python 3D HVP currently requires model_gradient_sampling_interval in {0, 1}."
        )

    from .maxwell3d_born import born3d

    epsilon_req = _clone_param(epsilon)
    sigma_req = _clone_param(sigma)
    mu_fixed = mu.detach()
    if vepsilon is None:
        vepsilon = torch.zeros_like(epsilon_req)
    if vsigma is None:
        vsigma = torch.zeros_like(sigma_req)

    born_outputs = born3d(
        epsilon_req,
        sigma_req,
        mu_fixed,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        bg_receiver_location=receiver_location,
        depsilon=vepsilon,
        dsigma=vsigma,
        stencil=stencil,
        pml_width=pml_width,
        max_vel=max_vel,
        nt=nt,
        linearize_source=linearize_source,
        source_component=source_component,
        receiver_component=receiver_component,
        python_backend=True,
    )
    predicted_data = born_outputs[-2]
    delta_predicted_data = born_outputs[-1]
    hvp_epsilon, hvp_sigma = _directional_receiver_hvp(
        params=(epsilon_req, sigma_req),
        observed_data=observed_data,
        misfit_fn=misfit_fn,
        predicted_data=predicted_data,
        delta_predicted_data=delta_predicted_data,
        hessian_mode=hessian_mode,
    )
    return hvp_epsilon, hvp_sigma


def maxwell3d_receiver_hvp_native(
    epsilon: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    *,
    vepsilon: torch.Tensor | None = None,
    vsigma: torch.Tensor | None = None,
    grid_spacing: float | Sequence[float],
    dt: float,
    source_amplitude: torch.Tensor | None,
    source_location: torch.Tensor | None,
    receiver_location: torch.Tensor | None,
    observed_data: torch.Tensor,
    misfit_fn: ReceiverMisfit,
    stencil: int = 2,
    pml_width: int | Sequence[int] = 20,
    max_vel: float | None = None,
    nt: int | None = None,
    model_gradient_sampling_interval: int = 1,
    linearize_source: bool = True,
    source_component: str = "ey",
    receiver_component: str = "ey",
    hessian_mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D receiver-space HVP through the native Maxwell/Born forward paths."""
    if vepsilon is None and vsigma is None:
        raise ValueError("At least one of vepsilon or vsigma must be provided.")
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    if epsilon.device.type not in {"cpu", "cuda"}:
        raise NotImplementedError("Native 3D HVP currently supports CPU and CUDA only.")

    from .maxwell3d_born import born3d

    epsilon_req = _clone_param(epsilon)
    sigma_req = _clone_param(sigma)
    mu_fixed = mu.detach()
    vepsilon_req = _clone_param(
        torch.zeros_like(epsilon_req) if vepsilon is None else vepsilon
    )
    vsigma_req = _clone_param(
        torch.zeros_like(sigma_req) if vsigma is None else vsigma
    )
    storage_compression = (
        "bf16"
        if epsilon_req.device.type == "cuda" and epsilon_req.dtype == torch.float32
        else False
    )

    born_outputs = born3d(
        epsilon_req,
        sigma_req,
        mu_fixed,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        bg_receiver_location=receiver_location,
        depsilon=vepsilon_req,
        dsigma=vsigma_req,
        stencil=stencil,
        pml_width=pml_width,
        max_vel=max_vel,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        linearize_source=linearize_source,
        source_component=source_component,
        receiver_component=receiver_component,
        python_backend=False,
        storage_mode="device",
        storage_compression=storage_compression,
    )
    predicted_data = born_outputs[-2]
    delta_predicted_data = born_outputs[-1]
    hvp_epsilon, hvp_sigma = _directional_receiver_hvp_from_born(
        params=(epsilon_req, sigma_req),
        direction_params=(vepsilon_req, vsigma_req),
        observed_data=observed_data,
        misfit_fn=misfit_fn,
        predicted_data=predicted_data,
        delta_predicted_data=delta_predicted_data,
        hessian_mode=hessian_mode,
    )
    return hvp_epsilon, hvp_sigma


__all__ = [
    "Born3DForwardFunc",
    "maxwell3d_receiver_hvp_naive",
    "maxwell3d_receiver_hvp_native",
]
