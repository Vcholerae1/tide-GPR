from collections.abc import Sequence
from typing import Any

import torch

from ..storage import (
    _CPU_STORAGE_BUFFERS,
    STORAGE_CPU,
    STORAGE_DEVICE,
    STORAGE_DISK,
    STORAGE_NONE,
    TemporaryStorage,
    storage_mode_to_int,
)
from ..validation import validate_model_gradient_sampling_interval
from .common import _get_ctx_handle, _register_ctx_handle, _release_ctx_handle
from .common import _clone_param, _directional_receiver_hvp, ReceiverMisfit
from .tm2d_helpers import _make_tm_storage_streams, _resolve_tm2d_storage_spec


def _alloc_tm2d_field(
    *,
    ctx: Any,
    device: torch.device,
    dtype: torch.dtype,
    zeros: bool = True,
) -> torch.Tensor:
    factory = torch.zeros if zeros else torch.empty
    return factory(ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=dtype)


def _zero_tensors_(
    *tensors: torch.Tensor,
) -> None:
    for tensor in tensors:
        if tensor.numel() > 0:
            tensor.zero_()


class BornTMForwardFunc(torch.autograd.Function):
    """Autograd function for the native 2D TM Born operator."""

    @staticmethod
    def forward(
        dca: torch.Tensor,
        dcb: torch.Tensor,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        f0: torch.Tensor,
        df: torch.Tensor,
        ay: torch.Tensor,
        by: torch.Tensor,
        ay_h: torch.Tensor,
        by_h: torch.Tensor,
        ax: torch.Tensor,
        bx: torch.Tensor,
        ax_h: torch.Tensor,
        bx_h: torch.Tensor,
        ky: torch.Tensor,
        ky_h: torch.Tensor,
        kx: torch.Tensor,
        kx_h: torch.Tensor,
        sources_i: torch.Tensor,
        receivers_i: torch.Tensor,
        rdy: float,
        rdx: float,
        dt: float,
        nt: int,
        n_shots: int,
        ny: int,
        nx: int,
        n_sources: int,
        n_receivers: int,
        step_ratio: int,
        accuracy: int,
        ca_batched: bool,
        cb_batched: bool,
        cq_batched: bool,
        pml_y0: int,
        pml_x0: int,
        pml_y1: int,
        pml_x1: int,
        storage_mode_str: str,
        storage_format: int,
        storage_path: str,
        storage_compression: bool | str,
        Ey: torch.Tensor,
        Hx: torch.Tensor,
        Hz: torch.Tensor,
        m_Ey_x: torch.Tensor,
        m_Ey_z: torch.Tensor,
        m_Hx_z: torch.Tensor,
        m_Hz_x: torch.Tensor,
        dEy: torch.Tensor,
        dHx: torch.Tensor,
        dHz: torch.Tensor,
        dm_Ey_x: torch.Tensor,
        dm_Ey_z: torch.Tensor,
        dm_Hx_z: torch.Tensor,
        dm_Hz_x: torch.Tensor,
        n_threads: int,
        backend_device: torch.device,
    ) -> tuple[Any, ...]:
        from .. import backend_utils

        device = dEy.device
        coeff_dtype = ca.dtype

        dca_requires_grad = dca.requires_grad
        dcb_requires_grad = dcb.requires_grad
        df_requires_grad = df.requires_grad
        background_grad_possible = (
            ca.requires_grad or cb.requires_grad or f0.requires_grad
        )
        store_ey_needed = dca_requires_grad or background_grad_possible
        store_curl_needed = dcb_requires_grad or background_grad_possible
        needs_storage = store_ey_needed or store_curl_needed

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=coeff_dtype
            )
            background_receiver_amplitudes = torch.zeros_like(receiver_amplitudes)
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=coeff_dtype)
            background_receiver_amplitudes = torch.empty(
                0, device=device, dtype=coeff_dtype
            )

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        backward_storage_tensors: list[torch.Tensor] = []
        backward_storage_objects: list[Any] = []
        backward_storage_filename_arrays: list[Any] = []
        storage_mode = STORAGE_NONE
        shot_bytes_uncomp = 0
        store_dtype = coeff_dtype
        stream_keepalive: tuple[Any, ...] = ()
        direct_snapshot_tensors = (
            torch.empty(0, device=device, dtype=store_dtype),
            torch.empty(0, device=device, dtype=store_dtype),
        )

        if needs_storage:
            import ctypes

            if str(device) == "cpu" and storage_mode_str == "cpu":
                storage_mode_str = "device"
            storage_mode = storage_mode_to_int(storage_mode_str)
            compute_stream_handle, storage_stream_handle, stream_keepalive = (
                _make_tm_storage_streams(device, storage_mode)
            )

            num_elements_per_shot = ny * nx
            _, store_dtype, _, resolved_storage_format = _resolve_tm2d_storage_spec(
                storage_compression=storage_compression,
                dtype=coeff_dtype,
                device=device,
                context="storage_compression",
            )
            if resolved_storage_format != storage_format:
                raise RuntimeError("Mismatched TM2D Born storage format resolution.")

            shot_bytes_uncomp = num_elements_per_shot * store_dtype.itemsize
            num_steps_stored = (nt + step_ratio - 1) // step_ratio
            if background_grad_possible:
                direct_snapshot_tensors = (
                    torch.empty(
                        num_steps_stored,
                        n_shots,
                        ny,
                        nx,
                        device=device,
                        dtype=store_dtype,
                    ),
                    torch.empty(
                        num_steps_stored,
                        n_shots,
                        ny,
                        nx,
                        device=device,
                        dtype=store_dtype,
                    ),
                )

            char_ptr_type = ctypes.c_char_p
            is_cuda = device.type == "cuda"

            def alloc_storage(requires_grad_cond: bool):
                store_1 = torch.empty(0, device=device, dtype=store_dtype)
                store_3 = torch.empty(0, device=device, dtype=store_dtype)
                filenames_arr = (char_ptr_type * 0)()

                if requires_grad_cond and storage_mode != STORAGE_NONE:
                    if storage_mode == STORAGE_DEVICE:
                        store_1 = torch.empty(
                            num_steps_stored,
                            n_shots,
                            ny,
                            nx,
                            device=device,
                            dtype=store_dtype,
                        )
                    elif storage_mode == STORAGE_CPU:
                        store_1 = torch.empty(
                            _CPU_STORAGE_BUFFERS,
                            n_shots,
                            ny,
                            nx,
                            device=device,
                            dtype=store_dtype,
                        )
                        store_3 = torch.empty(
                            num_steps_stored,
                            n_shots,
                            shot_bytes_uncomp // store_dtype.itemsize,
                            device="cpu",
                            pin_memory=True,
                            dtype=store_dtype,
                        )
                    elif storage_mode == STORAGE_DISK:
                        storage_obj = TemporaryStorage(
                            storage_path, 1 if is_cuda else n_shots
                        )
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
                                ny,
                                nx,
                                device=device,
                                dtype=store_dtype,
                            )
                            store_3 = torch.empty(
                                _CPU_STORAGE_BUFFERS,
                                n_shots,
                                shot_bytes_uncomp // store_dtype.itemsize,
                                device="cpu",
                                pin_memory=True,
                                dtype=store_dtype,
                            )
                        else:
                            store_1 = torch.empty(
                                n_shots, ny, nx, device=device, dtype=store_dtype
                            )

                backward_storage_tensors.extend([store_1, store_3])
                backward_storage_filename_arrays.append(filenames_arr)

                filenames_ptr = (
                    ctypes.cast(filenames_arr, ctypes.c_void_p)
                    if storage_mode == STORAGE_DISK
                    else 0
                )
                return store_1, store_3, filenames_ptr

            ey_store_1, ey_store_3, ey_filenames_ptr = alloc_storage(store_ey_needed)
            curl_store_1, curl_store_3, curl_filenames_ptr = alloc_storage(
                store_curl_needed
            )

            forward_func = backend_utils.get_backend_function(
                "maxwell_tm",
                "born_forward_with_storage",
                accuracy,
                coeff_dtype,
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
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_Ey_x),
                backend_utils.tensor_to_ptr(m_Ey_z),
                backend_utils.tensor_to_ptr(m_Hx_z),
                backend_utils.tensor_to_ptr(m_Hz_x),
                backend_utils.tensor_to_ptr(dEy),
                backend_utils.tensor_to_ptr(dHx),
                backend_utils.tensor_to_ptr(dHz),
                backend_utils.tensor_to_ptr(dm_Ey_x),
                backend_utils.tensor_to_ptr(dm_Ey_z),
                backend_utils.tensor_to_ptr(dm_Hx_z),
                backend_utils.tensor_to_ptr(dm_Hz_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(background_receiver_amplitudes),
                backend_utils.tensor_to_ptr(ey_store_1),
                backend_utils.tensor_to_ptr(ey_store_3),
                ey_filenames_ptr,
                backend_utils.tensor_to_ptr(curl_store_1),
                backend_utils.tensor_to_ptr(curl_store_3),
                curl_filenames_ptr,
                backend_utils.tensor_to_ptr(direct_snapshot_tensors[0]),
                backend_utils.tensor_to_ptr(direct_snapshot_tensors[1]),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                rdy,
                rdx,
                dt,
                nt,
                n_shots,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                storage_mode,
                storage_format,
                shot_bytes_uncomp,
                store_ey_needed,
                store_curl_needed,
                ca_batched,
                cb_batched,
                cq_batched,
                0,
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                n_threads,
                device_idx,
                compute_stream_handle,
                storage_stream_handle,
            )
        else:
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm",
                "born_forward",
                accuracy,
                coeff_dtype,
                backend_device,
            )
            compute_stream_handle, _, stream_keepalive = _make_tm_storage_streams(
                device, STORAGE_NONE
            )
            forward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(dca),
                backend_utils.tensor_to_ptr(dcb),
                backend_utils.tensor_to_ptr(f0),
                backend_utils.tensor_to_ptr(df),
                backend_utils.tensor_to_ptr(Ey),
                backend_utils.tensor_to_ptr(Hx),
                backend_utils.tensor_to_ptr(Hz),
                backend_utils.tensor_to_ptr(m_Ey_x),
                backend_utils.tensor_to_ptr(m_Ey_z),
                backend_utils.tensor_to_ptr(m_Hx_z),
                backend_utils.tensor_to_ptr(m_Hz_x),
                backend_utils.tensor_to_ptr(dEy),
                backend_utils.tensor_to_ptr(dHx),
                backend_utils.tensor_to_ptr(dHz),
                backend_utils.tensor_to_ptr(dm_Ey_x),
                backend_utils.tensor_to_ptr(dm_Ey_z),
                backend_utils.tensor_to_ptr(dm_Hx_z),
                backend_utils.tensor_to_ptr(dm_Hz_x),
                backend_utils.tensor_to_ptr(receiver_amplitudes),
                backend_utils.tensor_to_ptr(background_receiver_amplitudes),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                rdy,
                rdx,
                dt,
                nt,
                n_shots,
                ny,
                nx,
                n_sources,
                n_receivers,
                step_ratio,
                ca_batched,
                cb_batched,
                cq_batched,
                0,
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                n_threads,
                device_idx,
                compute_stream_handle,
            )

        if not backward_storage_tensors:
            backward_storage_tensors = [
                torch.empty(0, device=device, dtype=store_dtype),
                torch.empty(0, device=device, dtype=store_dtype),
                torch.empty(0, device=device, dtype=store_dtype),
                torch.empty(0, device=device, dtype=store_dtype),
            ]
        if not backward_storage_filename_arrays:
            backward_storage_filename_arrays = [None, None]

        ctx_data = {
            "backward_storage_tensors": backward_storage_tensors,
            "backward_storage_objects": backward_storage_objects,
            "backward_storage_filename_arrays": backward_storage_filename_arrays,
            "storage_mode": storage_mode,
            "storage_format": storage_format,
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "dca_requires_grad": dca_requires_grad,
            "dcb_requires_grad": dcb_requires_grad,
            "df_requires_grad": df_requires_grad,
            "direct_snapshot_tensors": direct_snapshot_tensors,
            "stream_keepalive": stream_keepalive,
        }
        ctx_handle = _register_ctx_handle(ctx_data)
        return (
            dEy,
            dHx,
            dHz,
            dm_Ey_x,
            dm_Ey_z,
            dm_Hx_z,
            dm_Hz_x,
            receiver_amplitudes,
            background_receiver_amplitudes,
            ctx_handle,
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        outputs = output if isinstance(output, tuple) else (output,)
        ctx_handle = outputs[-1]
        if not isinstance(ctx_handle, torch.Tensor):
            raise RuntimeError("BornTMForwardFunc context handle must be a Tensor.")

        ctx_handle_id = int(ctx_handle.item())
        ctx_data = _get_ctx_handle(ctx_handle_id)
        ctx._ctx_handle_id = ctx_handle_id
        backward_storage_tensors = ctx_data["backward_storage_tensors"]
        direct_snapshot_tensors = ctx_data["direct_snapshot_tensors"]
        ctx.backward_storage_filename_arrays = ctx_data[
            "backward_storage_filename_arrays"
        ]

        ctx.save_for_backward(
            inputs[0],  # dca
            inputs[1],  # dcb
            inputs[2],  # ca
            inputs[3],  # cb
            inputs[4],  # cq
            inputs[5],  # f0
            inputs[6],  # df
            inputs[7],  # ay
            inputs[8],  # by
            inputs[9],  # ay_h
            inputs[10],  # by_h
            inputs[11],  # ax
            inputs[12],  # bx
            inputs[13],  # ax_h
            inputs[14],  # bx_h
            inputs[15],  # ky
            inputs[16],  # ky_h
            inputs[17],  # kx
            inputs[18],  # kx_h
            inputs[19],  # sources_i
            inputs[20],  # receivers_i
            *backward_storage_tensors,
            *direct_snapshot_tensors,
        )
        ctx.stream_keepalive = ctx_data["stream_keepalive"]
        ctx.rdy = inputs[21]
        ctx.rdx = inputs[22]
        ctx.dt = inputs[23]
        ctx.nt = inputs[24]
        ctx.n_shots = inputs[25]
        ctx.ny = inputs[26]
        ctx.nx = inputs[27]
        ctx.n_sources = inputs[28]
        ctx.n_receivers = inputs[29]
        ctx.step_ratio = inputs[30]
        ctx.accuracy = inputs[31]
        ctx.ca_batched = inputs[32]
        ctx.cb_batched = inputs[33]
        ctx.cq_batched = inputs[34]
        ctx.pml_y0 = inputs[35]
        ctx.pml_x0 = inputs[36]
        ctx.pml_y1 = inputs[37]
        ctx.pml_x1 = inputs[38]
        ctx.storage_mode = ctx_data["storage_mode"]
        ctx.storage_format = ctx_data["storage_format"]
        ctx.shot_bytes_uncomp = ctx_data["shot_bytes_uncomp"]
        ctx.dca_requires_grad = ctx_data["dca_requires_grad"]
        ctx.dcb_requires_grad = ctx_data["dcb_requires_grad"]
        ctx.df_requires_grad = ctx_data["df_requires_grad"]
        ctx.background_grad_required = any(ctx.needs_input_grad[i] for i in (2, 3, 5))
        ctx.n_threads = inputs[57]
        ctx.backend_device = inputs[58]
        ctx.n_inputs = len(inputs)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        from .. import backend_utils

        grad_outputs_list = list(grad_outputs)
        if len(grad_outputs_list) == 10:
            grad_outputs_list.pop()

        (
            grad_dEy,
            grad_dHx,
            grad_dHz,
            grad_dm_Ey_x,
            grad_dm_Ey_z,
            grad_dm_Hx_z,
            grad_dm_Hz_x,
            grad_r,
            grad_background_r,
        ) = grad_outputs_list
        del (
            grad_dEy,
            grad_dHx,
            grad_dHz,
            grad_dm_Ey_x,
            grad_dm_Ey_z,
            grad_dm_Hx_z,
            grad_dm_Hz_x,
        )

        (
            dca,
            dcb,
            ca,
            cb,
            cq,
            f0,
            df,
            ay,
            by,
            ay_h,
            by_h,
            ax,
            bx,
            ax_h,
            bx_h,
            ky,
            ky_h,
            kx,
            kx_h,
            sources_i,
            receivers_i,
            ey_store_1,
            ey_store_3,
            curl_store_1,
            curl_store_3,
            dey_store,
            dcurl_store,
        ) = ctx.saved_tensors

        device = ca.device
        coeff_dtype = ca.dtype

        import ctypes

        if ctx.storage_mode == STORAGE_DISK:
            ey_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[0], ctypes.c_void_p
            )
            curl_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[1], ctypes.c_void_p
            )
        else:
            ey_filenames_ptr = 0
            curl_filenames_ptr = 0

        bg_ca_requires_grad = ctx.needs_input_grad[2]
        bg_cb_requires_grad = ctx.needs_input_grad[3]
        f0_requires_grad = ctx.needs_input_grad[5]
        model_grad_requested = (
            bg_ca_requires_grad or bg_cb_requires_grad or f0_requires_grad
        )

        receiver_grad_needed = bool(
            grad_r is not None
            and grad_r.numel() > 0
            and torch.count_nonzero(grad_r).item() > 0
        )
        if receiver_grad_needed:
            grad_r = grad_r.contiguous()
        elif ctx.n_receivers > 0:
            grad_r = torch.zeros(
                ctx.nt,
                ctx.n_shots,
                ctx.n_receivers,
                device=device,
                dtype=coeff_dtype,
            )
        else:
            grad_r = torch.empty(0, device=device, dtype=coeff_dtype)

        background_receiver_grad_needed = bool(
            grad_background_r is not None
            and grad_background_r.numel() > 0
            and torch.count_nonzero(grad_background_r).item() > 0
        )
        if background_receiver_grad_needed:
            grad_background_r = grad_background_r.contiguous()
        elif ctx.n_receivers > 0:
            grad_background_r = torch.zeros(
                ctx.nt,
                ctx.n_shots,
                ctx.n_receivers,
                device=device,
                dtype=coeff_dtype,
            )
        else:
            grad_background_r = torch.empty(0, device=device, dtype=coeff_dtype)

        needs_bggrad = receiver_grad_needed and model_grad_requested
        needs_born_backward = (
            receiver_grad_needed
            and not needs_bggrad
            and (ctx.dca_requires_grad or ctx.dcb_requires_grad or ctx.df_requires_grad)
        )
        needs_background_backward = (
            background_receiver_grad_needed and model_grad_requested
        )
        needs_lambda_workspace = (
            needs_bggrad or needs_born_backward or needs_background_backward
        )

        if needs_lambda_workspace:
            lambda_ey = _alloc_tm2d_field(ctx=ctx, device=device, dtype=coeff_dtype)
            lambda_hx = torch.zeros_like(lambda_ey)
            lambda_hz = torch.zeros_like(lambda_ey)
            m_lambda_ey_x = torch.zeros_like(lambda_ey)
            m_lambda_ey_z = torch.zeros_like(lambda_ey)
            m_lambda_hx_z = torch.zeros_like(lambda_ey)
            m_lambda_hz_x = torch.zeros_like(lambda_ey)
            work_x = torch.zeros_like(lambda_ey)
            work_z = torch.zeros_like(lambda_ey)
        else:
            lambda_ey = torch.empty(0, device=device, dtype=coeff_dtype)
            lambda_hx = torch.empty(0, device=device, dtype=coeff_dtype)
            lambda_hz = torch.empty(0, device=device, dtype=coeff_dtype)
            m_lambda_ey_x = torch.empty(0, device=device, dtype=coeff_dtype)
            m_lambda_ey_z = torch.empty(0, device=device, dtype=coeff_dtype)
            m_lambda_hx_z = torch.empty(0, device=device, dtype=coeff_dtype)
            m_lambda_hz_x = torch.empty(0, device=device, dtype=coeff_dtype)
            work_x = torch.empty(0, device=device, dtype=coeff_dtype)
            work_z = torch.empty(0, device=device, dtype=coeff_dtype)

        if receiver_grad_needed and ctx.n_sources > 0:
            grad_f = torch.zeros(
                ctx.nt, ctx.n_shots, ctx.n_sources, device=device, dtype=coeff_dtype
            )
        else:
            grad_f = torch.empty(0, device=device, dtype=coeff_dtype)

        if needs_bggrad or ctx.dca_requires_grad:
            grad_dca = (
                torch.zeros(
                    ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
                )
                if ctx.ca_batched
                else torch.zeros(ctx.ny, ctx.nx, device=device, dtype=coeff_dtype)
            )
            grad_dca_shot = torch.zeros(
                ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
            )
        else:
            grad_dca = torch.empty(0, device=device, dtype=coeff_dtype)
            grad_dca_shot = torch.empty(0, device=device, dtype=coeff_dtype)

        if needs_bggrad or ctx.dcb_requires_grad:
            grad_dcb = (
                torch.zeros(
                    ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
                )
                if ctx.cb_batched
                else torch.zeros(ctx.ny, ctx.nx, device=device, dtype=coeff_dtype)
            )
            grad_dcb_shot = torch.zeros(
                ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
            )
        else:
            grad_dcb = torch.empty(0, device=device, dtype=coeff_dtype)
            grad_dcb_shot = torch.empty(0, device=device, dtype=coeff_dtype)

        if model_grad_requested and (needs_bggrad or needs_background_backward):
            grad_ca = (
                torch.zeros(
                    ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
                )
                if ctx.ca_batched
                else torch.zeros(ctx.ny, ctx.nx, device=device, dtype=coeff_dtype)
            )
            grad_cb = (
                torch.zeros(
                    ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
                )
                if ctx.cb_batched
                else torch.zeros(ctx.ny, ctx.nx, device=device, dtype=coeff_dtype)
            )
        else:
            grad_ca = torch.empty(0, device=device, dtype=coeff_dtype)
            grad_cb = torch.empty(0, device=device, dtype=coeff_dtype)

        if bg_ca_requires_grad and (needs_bggrad or needs_background_backward):
            grad_ca_shot = torch.zeros(
                ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
            )
        else:
            grad_ca_shot = torch.empty(0, device=device, dtype=coeff_dtype)

        if bg_cb_requires_grad and (needs_bggrad or needs_background_backward):
            grad_cb_shot = torch.zeros(
                ctx.n_shots, ctx.ny, ctx.nx, device=device, dtype=coeff_dtype
            )
        else:
            grad_cb_shot = torch.empty(0, device=device, dtype=coeff_dtype)

        if (
            f0_requires_grad
            and (needs_bggrad or needs_background_backward)
            and ctx.n_sources > 0
        ):
            grad_f0 = torch.zeros(
                ctx.nt, ctx.n_shots, ctx.n_sources, device=device, dtype=coeff_dtype
            )
        else:
            grad_f0 = torch.empty(0, device=device, dtype=coeff_dtype)

        store_ey_needed = ctx.dca_requires_grad or model_grad_requested
        store_curl_needed = ctx.dcb_requires_grad or model_grad_requested

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        compute_stream_handle, storage_stream_handle, stream_keepalive = (
            _make_tm_storage_streams(device, ctx.storage_mode)
        )
        ctx.stream_keepalive = stream_keepalive
        grad_ca_shot_ptr = grad_ca if ctx.ca_batched else grad_ca_shot
        grad_cb_shot_ptr = grad_cb if ctx.cb_batched else grad_cb_shot
        grad_dca_shot_ptr = grad_dca if ctx.ca_batched else grad_dca_shot
        grad_dcb_shot_ptr = grad_dcb if ctx.cb_batched else grad_dcb_shot

        if needs_background_backward:
            _zero_tensors_(
                lambda_ey,
                lambda_hx,
                lambda_hz,
                m_lambda_ey_x,
                m_lambda_ey_z,
                m_lambda_hx_z,
                m_lambda_hz_x,
                work_x,
                work_z,
            )
            if not ctx.ca_batched:
                _zero_tensors_(grad_ca_shot)
            if not ctx.cb_batched:
                _zero_tensors_(grad_cb_shot)
            background_backward_func = backend_utils.get_backend_function(
                "maxwell_tm",
                "backward",
                ctx.accuracy,
                coeff_dtype,
                ctx.backend_device,
            )
            background_backward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(grad_background_r),
                backend_utils.tensor_to_ptr(lambda_ey),
                backend_utils.tensor_to_ptr(lambda_hx),
                backend_utils.tensor_to_ptr(lambda_hz),
                backend_utils.tensor_to_ptr(m_lambda_ey_x),
                backend_utils.tensor_to_ptr(m_lambda_ey_z),
                backend_utils.tensor_to_ptr(m_lambda_hx_z),
                backend_utils.tensor_to_ptr(m_lambda_hz_x),
                backend_utils.tensor_to_ptr(ey_store_1),
                backend_utils.tensor_to_ptr(ey_store_3),
                ey_filenames_ptr,
                backend_utils.tensor_to_ptr(curl_store_1),
                backend_utils.tensor_to_ptr(curl_store_3),
                curl_filenames_ptr,
                backend_utils.tensor_to_ptr(grad_f0),
                backend_utils.tensor_to_ptr(grad_ca),
                backend_utils.tensor_to_ptr(grad_cb),
                backend_utils.tensor_to_ptr(grad_ca_shot_ptr),
                backend_utils.tensor_to_ptr(grad_cb_shot_ptr),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                ctx.rdy,
                ctx.rdx,
                ctx.dt,
                ctx.nt,
                ctx.n_shots,
                ctx.ny,
                ctx.nx,
                ctx.n_sources,
                ctx.n_receivers,
                ctx.step_ratio,
                ctx.storage_mode,
                ctx.storage_format,
                ctx.shot_bytes_uncomp,
                bg_ca_requires_grad,
                bg_cb_requires_grad,
                ctx.ca_batched,
                ctx.cb_batched,
                ctx.cq_batched,
                ctx.nt,
                ctx.pml_y0,
                ctx.pml_x0,
                ctx.pml_y1,
                ctx.pml_x1,
                ctx.n_threads,
                device_idx,
                compute_stream_handle,
                storage_stream_handle,
            )

        if needs_bggrad:
            bg_eta_ey = torch.zeros_like(lambda_ey)
            bg_eta_hx = torch.zeros_like(lambda_hx)
            bg_eta_hz = torch.zeros_like(lambda_hz)
            m_eta_ey_x = torch.zeros_like(lambda_ey)
            m_eta_ey_z = torch.zeros_like(lambda_ey)
            m_eta_hx_z = torch.zeros_like(lambda_ey)
            m_eta_hz_x = torch.zeros_like(lambda_ey)
            eta_source_old = torch.zeros_like(lambda_ey)
            work_eta_x = torch.zeros_like(lambda_ey)
            work_eta_z = torch.zeros_like(lambda_ey)
            bggrad_grad_f0 = grad_f0
            bggrad_grad_ca = grad_ca
            bggrad_grad_cb = grad_cb
            if needs_background_backward:
                bggrad_grad_f0 = (
                    torch.zeros_like(grad_f0) if grad_f0.numel() > 0 else grad_f0
                )
                bggrad_grad_ca = (
                    torch.zeros_like(grad_ca) if grad_ca.numel() > 0 else grad_ca
                )
                bggrad_grad_cb = (
                    torch.zeros_like(grad_cb) if grad_cb.numel() > 0 else grad_cb
                )
            bggrad_grad_ca_shot_ptr = (
                bggrad_grad_ca if ctx.ca_batched else grad_ca_shot_ptr
            )
            bggrad_grad_cb_shot_ptr = (
                bggrad_grad_cb if ctx.cb_batched else grad_cb_shot_ptr
            )
            _zero_tensors_(
                lambda_ey,
                lambda_hx,
                lambda_hz,
                m_lambda_ey_x,
                m_lambda_ey_z,
                m_lambda_hx_z,
                m_lambda_hz_x,
                bg_eta_ey,
                bg_eta_hx,
                bg_eta_hz,
                m_eta_ey_x,
                m_eta_ey_z,
                m_eta_hx_z,
                m_eta_hz_x,
                eta_source_old,
                work_eta_x,
                work_eta_z,
            )
            if not ctx.ca_batched:
                _zero_tensors_(grad_ca_shot, grad_dca_shot)
            if not ctx.cb_batched:
                _zero_tensors_(grad_cb_shot, grad_dcb_shot)
            bggrad_func = backend_utils.get_backend_function(
                "maxwell_tm",
                "born_backward_bggrad",
                ctx.accuracy,
                coeff_dtype,
                ctx.backend_device,
            )
            bggrad_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(dca),
                backend_utils.tensor_to_ptr(dcb),
                backend_utils.tensor_to_ptr(f0),
                backend_utils.tensor_to_ptr(df),
                backend_utils.tensor_to_ptr(grad_r),
                backend_utils.tensor_to_ptr(ey_store_1),
                backend_utils.tensor_to_ptr(ey_store_3),
                ey_filenames_ptr,
                backend_utils.tensor_to_ptr(curl_store_1),
                backend_utils.tensor_to_ptr(curl_store_3),
                curl_filenames_ptr,
                backend_utils.tensor_to_ptr(dey_store),
                backend_utils.tensor_to_ptr(dcurl_store),
                backend_utils.tensor_to_ptr(lambda_ey),
                backend_utils.tensor_to_ptr(lambda_hx),
                backend_utils.tensor_to_ptr(lambda_hz),
                backend_utils.tensor_to_ptr(bg_eta_ey),
                backend_utils.tensor_to_ptr(bg_eta_hx),
                backend_utils.tensor_to_ptr(bg_eta_hz),
                backend_utils.tensor_to_ptr(bggrad_grad_f0),
                backend_utils.tensor_to_ptr(grad_f),
                backend_utils.tensor_to_ptr(bggrad_grad_ca),
                backend_utils.tensor_to_ptr(bggrad_grad_cb),
                backend_utils.tensor_to_ptr(grad_dca),
                backend_utils.tensor_to_ptr(grad_dcb),
                backend_utils.tensor_to_ptr(m_lambda_ey_x),
                backend_utils.tensor_to_ptr(m_lambda_ey_z),
                backend_utils.tensor_to_ptr(m_lambda_hx_z),
                backend_utils.tensor_to_ptr(m_lambda_hz_x),
                backend_utils.tensor_to_ptr(m_eta_ey_x),
                backend_utils.tensor_to_ptr(m_eta_ey_z),
                backend_utils.tensor_to_ptr(m_eta_hx_z),
                backend_utils.tensor_to_ptr(m_eta_hz_x),
                backend_utils.tensor_to_ptr(eta_source_old),
                backend_utils.tensor_to_ptr(work_eta_x),
                backend_utils.tensor_to_ptr(work_eta_z),
                backend_utils.tensor_to_ptr(bggrad_grad_ca_shot_ptr),
                backend_utils.tensor_to_ptr(bggrad_grad_cb_shot_ptr),
                backend_utils.tensor_to_ptr(grad_dca_shot_ptr),
                backend_utils.tensor_to_ptr(grad_dcb_shot_ptr),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                ctx.rdy,
                ctx.rdx,
                ctx.dt,
                ctx.nt,
                ctx.n_shots,
                ctx.ny,
                ctx.nx,
                ctx.n_sources,
                ctx.n_receivers,
                ctx.step_ratio,
                ctx.storage_mode,
                ctx.storage_format,
                ctx.shot_bytes_uncomp,
                store_ey_needed,
                store_curl_needed,
                ctx.ca_batched,
                ctx.cb_batched,
                ctx.cq_batched,
                ctx.nt,
                ctx.pml_y0,
                ctx.pml_x0,
                ctx.pml_y1,
                ctx.pml_x1,
                ctx.n_threads,
                device_idx,
                compute_stream_handle,
                storage_stream_handle,
            )
            if needs_background_backward:
                if bggrad_grad_f0.numel() > 0:
                    grad_f0.add_(bggrad_grad_f0)
                if bggrad_grad_ca.numel() > 0:
                    grad_ca.add_(bggrad_grad_ca)
                if bggrad_grad_cb.numel() > 0:
                    grad_cb.add_(bggrad_grad_cb)
        elif needs_born_backward:
            _zero_tensors_(
                lambda_ey,
                lambda_hx,
                lambda_hz,
                m_lambda_ey_x,
                m_lambda_ey_z,
                m_lambda_hx_z,
                m_lambda_hz_x,
                work_x,
                work_z,
                grad_dca_shot,
                grad_dcb_shot,
            )
            backward_func = backend_utils.get_backend_function(
                "maxwell_tm",
                "born_backward",
                ctx.accuracy,
                coeff_dtype,
                ctx.backend_device,
            )
            backward_func(
                backend_utils.tensor_to_ptr(ca),
                backend_utils.tensor_to_ptr(cb),
                backend_utils.tensor_to_ptr(cq),
                backend_utils.tensor_to_ptr(grad_r),
                backend_utils.tensor_to_ptr(lambda_ey),
                backend_utils.tensor_to_ptr(lambda_hx),
                backend_utils.tensor_to_ptr(lambda_hz),
                backend_utils.tensor_to_ptr(m_lambda_ey_x),
                backend_utils.tensor_to_ptr(m_lambda_ey_z),
                backend_utils.tensor_to_ptr(m_lambda_hx_z),
                backend_utils.tensor_to_ptr(m_lambda_hz_x),
                backend_utils.tensor_to_ptr(ey_store_1),
                backend_utils.tensor_to_ptr(ey_store_3),
                ey_filenames_ptr,
                backend_utils.tensor_to_ptr(curl_store_1),
                backend_utils.tensor_to_ptr(curl_store_3),
                curl_filenames_ptr,
                backend_utils.tensor_to_ptr(grad_f),
                backend_utils.tensor_to_ptr(grad_dca),
                backend_utils.tensor_to_ptr(grad_dcb),
                backend_utils.tensor_to_ptr(grad_dca_shot_ptr),
                backend_utils.tensor_to_ptr(grad_dcb_shot_ptr),
                backend_utils.tensor_to_ptr(work_x),
                backend_utils.tensor_to_ptr(work_z),
                backend_utils.tensor_to_ptr(ay),
                backend_utils.tensor_to_ptr(by),
                backend_utils.tensor_to_ptr(ay_h),
                backend_utils.tensor_to_ptr(by_h),
                backend_utils.tensor_to_ptr(ax),
                backend_utils.tensor_to_ptr(bx),
                backend_utils.tensor_to_ptr(ax_h),
                backend_utils.tensor_to_ptr(bx_h),
                backend_utils.tensor_to_ptr(ky),
                backend_utils.tensor_to_ptr(ky_h),
                backend_utils.tensor_to_ptr(kx),
                backend_utils.tensor_to_ptr(kx_h),
                backend_utils.tensor_to_ptr(sources_i),
                backend_utils.tensor_to_ptr(receivers_i),
                ctx.rdy,
                ctx.rdx,
                ctx.dt,
                ctx.nt,
                ctx.n_shots,
                ctx.ny,
                ctx.nx,
                ctx.n_sources,
                ctx.n_receivers,
                ctx.step_ratio,
                ctx.storage_mode,
                ctx.storage_format,
                ctx.shot_bytes_uncomp,
                ctx.dca_requires_grad,
                ctx.dcb_requires_grad,
                ctx.ca_batched,
                ctx.cb_batched,
                ctx.cq_batched,
                ctx.nt,
                ctx.pml_y0,
                ctx.pml_x0,
                ctx.pml_y1,
                ctx.pml_x1,
                ctx.n_threads,
                device_idx,
                compute_stream_handle,
                storage_stream_handle,
            )

        if ctx.dca_requires_grad and not ctx.ca_batched:
            grad_dca = grad_dca.unsqueeze(0)
        if ctx.dcb_requires_grad and not ctx.cb_batched:
            grad_dcb = grad_dcb.unsqueeze(0)
        if bg_ca_requires_grad and not ctx.ca_batched:
            grad_ca = grad_ca.unsqueeze(0)
        if bg_cb_requires_grad and not ctx.cb_batched:
            grad_cb = grad_cb.unsqueeze(0)

        grads: list[torch.Tensor | None] = [None] * ctx.n_inputs
        grads[0] = grad_dca if ctx.dca_requires_grad else None
        grads[1] = grad_dcb if ctx.dcb_requires_grad else None
        grads[2] = grad_ca if bg_ca_requires_grad else None
        grads[3] = grad_cb if bg_cb_requires_grad else None
        grads[5] = (
            grad_f0.reshape(ctx.nt * ctx.n_shots * ctx.n_sources)
            if f0_requires_grad and ctx.n_sources > 0
            else None
        )
        grads[6] = (
            grad_f.reshape(ctx.nt * ctx.n_shots * ctx.n_sources)
            if ctx.df_requires_grad and ctx.n_sources > 0 and grad_f.numel() > 0
            else None
        )

        _release_ctx_handle(getattr(ctx, "_ctx_handle_id", None))
        return tuple(grads)


def tm2d_receiver_hvp_naive(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference TM2D receiver-space HVP on the Python Maxwell/Born path."""
    if vepsilon is None and vsigma is None:
        raise ValueError("At least one of vepsilon or vsigma must be provided.")
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    if model_gradient_sampling_interval > 1:
        raise NotImplementedError(
            "Python TM2D HVP currently requires "
            "model_gradient_sampling_interval in {0, 1}."
        )

    from .tm2d import maxwelltm
    from .tm2d_born import borntm

    epsilon_req = _clone_param(epsilon)
    sigma_req = _clone_param(sigma)
    mu_fixed = mu.detach()
    if vepsilon is None:
        vepsilon = torch.zeros_like(epsilon_req)
    if vsigma is None:
        vsigma = torch.zeros_like(sigma_req)

    predicted_data = maxwelltm(
        epsilon_req,
        sigma_req,
        mu_fixed,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        stencil=stencil,
        pml_width=pml_width,
        max_vel=max_vel,
        nt=nt,
        model_gradient_sampling_interval=model_gradient_sampling_interval,
        python_backend=True,
    )[-1]
    delta_predicted_data = borntm(
        epsilon_req,
        sigma_req,
        mu_fixed,
        grid_spacing=grid_spacing,
        dt=dt,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        depsilon=vepsilon,
        dsigma=vsigma,
        stencil=stencil,
        pml_width=pml_width,
        max_vel=max_vel,
        nt=nt,
        linearize_source=linearize_source,
        python_backend=True,
    )[-1]
    hvp_epsilon, hvp_sigma = _directional_receiver_hvp(
        params=(epsilon_req, sigma_req),
        observed_data=observed_data,
        misfit_fn=misfit_fn,
        predicted_data=predicted_data,
        delta_predicted_data=delta_predicted_data,
    )
    return hvp_epsilon, hvp_sigma


def tm2d_receiver_hvp_native(
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
    pml_width: int | Sequence[int] = 0,
    max_vel: float | None = None,
    nt: int | None = None,
    model_gradient_sampling_interval: int = 1,
    linearize_source: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TM2D native receiver-space HVP via the native Born bggrad path."""
    from .. import backend_utils
    from ..grid_utils import _normalize_pml_width_2d
    from .tm2d_born_cuda import borntm_c_cuda

    if vepsilon is None and vsigma is None:
        raise ValueError("At least one of vepsilon or vsigma must be provided.")
    model_gradient_sampling_interval = validate_model_gradient_sampling_interval(
        model_gradient_sampling_interval
    )
    if not backend_utils.is_backend_available():
        raise RuntimeError("Native TM2D HVP requires the compiled backend.")
    if epsilon.device.type not in {"cpu", "cuda"}:
        raise NotImplementedError(
            "Native TM2D HVP currently supports cpu and cuda devices only."
        )
    if epsilon.device.type == "cpu" and model_gradient_sampling_interval > 1:
        raise NotImplementedError(
            "Native TM2D HVP on CPU currently requires "
            "model_gradient_sampling_interval in {0, 1}."
        )
    _normalize_pml_width_2d(pml_width)

    epsilon_req = _clone_param(epsilon)
    sigma_req = _clone_param(sigma)
    mu_fixed = mu.detach()
    if vepsilon is None:
        vepsilon = torch.zeros_like(epsilon_req)
    if vsigma is None:
        vsigma = torch.zeros_like(sigma_req)
    storage_compression = (
        "bf16"
        if epsilon_req.device.type == "cuda" and epsilon_req.dtype == torch.float32
        else False
    )

    born_outputs = borntm_c_cuda(
        epsilon_req,
        sigma_req,
        mu_fixed,
        vepsilon,
        vsigma,
        None,
        None,
        grid_spacing,
        dt,
        source_amplitude,
        source_location,
        receiver_location,
        stencil,
        pml_width,
        max_vel,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        nt,
        "epsilon_sigma",
        model_gradient_sampling_interval,
        linearize_source,
        storage_mode="device",
        storage_compression=storage_compression,
        return_background_receiver_amplitudes=True,
    )
    delta_predicted_data = born_outputs[-2]
    predicted_data = born_outputs[-1]
    hvp_epsilon, hvp_sigma = _directional_receiver_hvp(
        params=(epsilon_req, sigma_req),
        observed_data=observed_data,
        misfit_fn=misfit_fn,
        predicted_data=predicted_data,
        delta_predicted_data=delta_predicted_data,
    )
    return hvp_epsilon, hvp_sigma


__all__ = [
    "BornTMForwardFunc",
    "tm2d_receiver_hvp_naive",
    "tm2d_receiver_hvp_native",
]
