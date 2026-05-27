from typing import Any

import torch

from ..callbacks import Callback, CallbackState
from ..storage import (
    _CPU_STORAGE_BUFFERS,
    STORAGE_CPU,
    STORAGE_DEVICE,
    STORAGE_DISK,
    STORAGE_NONE,
    TemporaryStorage,
    storage_mode_to_int,
)
from ..utils import prepare_parameters
from .common import _get_ctx_handle, _make_compute_stream, _register_ctx_handle, _release_ctx_handle
from .tm2d_helpers import (
    _make_tm_storage_streams,
    _physical_tm2d_adjoint_callback_wavefields,
    _physical_tm2d_callback_wavefields,
    _resolve_tm2d_storage_spec,
)


class MaxwellTMForwardFunc(torch.autograd.Function):
    """Autograd function for the 2D TM Maxwell native backend."""

    @staticmethod
    def forward(
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
        source_amplitudes_scaled: torch.Tensor,
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
        fd_pad: tuple[int, int, int, int],
        pml_width: tuple[int, int, int, int],
        models: dict,
        forward_callback: Callback | None,
        backward_callback: Callback | None,
        callback_frequency: int,
        scale_ctx: dict[str, Any] | None,
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
        n_threads: int,
        backend_device: torch.device,
    ) -> tuple[Any, ...]:
        from .. import backend_utils

        device = Ey.device
        coeff_dtype = ca.dtype
        receiver_dtype = coeff_dtype
        variant = ""

        ca_requires_grad = ca.requires_grad
        cb_requires_grad = cb.requires_grad
        needs_grad = ca_requires_grad or cb_requires_grad

        if n_receivers > 0:
            receiver_amplitudes = torch.zeros(
                nt, n_shots, n_receivers, device=device, dtype=receiver_dtype
            )
        else:
            receiver_amplitudes = torch.empty(0, device=device, dtype=receiver_dtype)

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )

        backward_storage_tensors: list[torch.Tensor] = []
        backward_storage_objects: list[Any] = []
        backward_storage_filename_arrays: list[Any] = []
        storage_mode = STORAGE_NONE
        shot_bytes_uncomp = 0
        stream_keepalive: tuple[Any, ...] = ()

        if needs_grad:
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
                raise RuntimeError("Mismatched TM2D storage format resolution.")

            shot_bytes_uncomp = num_elements_per_shot * store_dtype.itemsize
            num_steps_stored = (nt + step_ratio - 1) // step_ratio

            char_ptr_type = ctypes.c_char_p
            is_cuda = device.type == "cuda"

            def alloc_storage(requires_grad_cond: bool):
                store_1 = torch.empty(0)
                store_3 = torch.empty(0)
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

            ey_store_1, ey_store_3, ey_filenames_ptr = alloc_storage(ca_requires_grad)
            curl_store_1, curl_store_3, curl_filenames_ptr = alloc_storage(
                cb_requires_grad
            )

            forward_func = backend_utils.get_backend_function(
                "maxwell_tm",
                "forward_with_storage",
                accuracy,
                coeff_dtype,
                backend_device,
                variant=variant,
            )
            effective_callback_freq = (
                nt // step_ratio if forward_callback is None else callback_frequency
            )

            for step in range(0, nt // step_ratio, effective_callback_freq):
                step_nt = (
                    min(effective_callback_freq, nt // step_ratio - step) * step_ratio
                )
                start_t = step * step_ratio
                forward_func(
                    backend_utils.tensor_to_ptr(ca),
                    backend_utils.tensor_to_ptr(cb),
                    backend_utils.tensor_to_ptr(cq),
                    backend_utils.tensor_to_ptr(source_amplitudes_scaled),
                    backend_utils.tensor_to_ptr(Ey),
                    backend_utils.tensor_to_ptr(Hx),
                    backend_utils.tensor_to_ptr(Hz),
                    backend_utils.tensor_to_ptr(m_Ey_x),
                    backend_utils.tensor_to_ptr(m_Ey_z),
                    backend_utils.tensor_to_ptr(m_Hx_z),
                    backend_utils.tensor_to_ptr(m_Hz_x),
                    backend_utils.tensor_to_ptr(receiver_amplitudes),
                    backend_utils.tensor_to_ptr(ey_store_1),
                    backend_utils.tensor_to_ptr(ey_store_3),
                    ey_filenames_ptr,
                    backend_utils.tensor_to_ptr(curl_store_1),
                    backend_utils.tensor_to_ptr(curl_store_3),
                    curl_filenames_ptr,
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
                    step_nt,
                    n_shots,
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
                    pml_y0,
                    pml_x0,
                    pml_y1,
                    pml_x1,
                    n_threads,
                    device_idx,
                    compute_stream_handle,
                    storage_stream_handle,
                )

                if forward_callback is not None:
                    callback_wavefields = _physical_tm2d_callback_wavefields(
                        {
                            "Ey": Ey,
                            "Hx": Hx,
                            "Hz": Hz,
                            "m_Ey_x": m_Ey_x,
                            "m_Ey_z": m_Ey_z,
                            "m_Hx_z": m_Hx_z,
                            "m_Hz_x": m_Hz_x,
                        },
                        scale_ctx=scale_ctx,
                    )
                    forward_callback(
                        CallbackState(
                            dt=dt,
                            step=step + step_nt // step_ratio,
                            nt=nt // step_ratio,
                            wavefields=callback_wavefields,
                            models=models,
                            gradients={},
                            fd_pad=list(fd_pad),
                            pml_width=list(pml_width),
                            is_backward=False,
                        )
                    )
        else:
            forward_func = backend_utils.get_backend_function(
                "maxwell_tm",
                "forward",
                accuracy,
                coeff_dtype,
                backend_device,
                variant=variant,
            )
            effective_callback_freq = (
                nt // step_ratio if forward_callback is None else callback_frequency
            )
            compute_stream_handle, _keepalive = _make_compute_stream(backend_device)

            for step in range(0, nt // step_ratio, effective_callback_freq):
                step_nt = (
                    min(effective_callback_freq, nt // step_ratio - step) * step_ratio
                )
                start_t = step * step_ratio
                forward_func(
                    backend_utils.tensor_to_ptr(ca),
                    backend_utils.tensor_to_ptr(cb),
                    backend_utils.tensor_to_ptr(cq),
                    backend_utils.tensor_to_ptr(source_amplitudes_scaled),
                    backend_utils.tensor_to_ptr(Ey),
                    backend_utils.tensor_to_ptr(Hx),
                    backend_utils.tensor_to_ptr(Hz),
                    backend_utils.tensor_to_ptr(m_Ey_x),
                    backend_utils.tensor_to_ptr(m_Ey_z),
                    backend_utils.tensor_to_ptr(m_Hx_z),
                    backend_utils.tensor_to_ptr(m_Hz_x),
                    backend_utils.tensor_to_ptr(receiver_amplitudes),
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
                    step_nt,
                    n_shots,
                    ny,
                    nx,
                    n_sources,
                    n_receivers,
                    step_ratio,
                    ca_batched,
                    cb_batched,
                    cq_batched,
                    start_t,
                    pml_y0,
                    pml_x0,
                    pml_y1,
                    pml_x1,
                    n_threads,
                    device_idx,
                    compute_stream_handle,
                )

                if forward_callback is not None:
                    callback_wavefields = _physical_tm2d_callback_wavefields(
                        {
                            "Ey": Ey,
                            "Hx": Hx,
                            "Hz": Hz,
                            "m_Ey_x": m_Ey_x,
                            "m_Ey_z": m_Ey_z,
                            "m_Hx_z": m_Hx_z,
                            "m_Hz_x": m_Hz_x,
                        },
                        scale_ctx=scale_ctx,
                    )
                    forward_callback(
                        CallbackState(
                            dt=dt,
                            step=step + step_nt // step_ratio,
                            nt=nt // step_ratio,
                            wavefields=callback_wavefields,
                            models=models,
                            gradients={},
                            fd_pad=list(fd_pad),
                            pml_width=list(pml_width),
                            is_backward=False,
                        )
                    )

        ctx_data = {
            "backward_storage_tensors": backward_storage_tensors,
            "backward_storage_objects": backward_storage_objects,
            "backward_storage_filename_arrays": backward_storage_filename_arrays,
            "storage_mode": storage_mode,
            "storage_format": storage_format,
            "shot_bytes_uncomp": shot_bytes_uncomp,
            "source_amplitudes_scaled": source_amplitudes_scaled,
            "ca_requires_grad": ca_requires_grad,
            "cb_requires_grad": cb_requires_grad,
            "scale_ctx": scale_ctx,
            "stream_keepalive": stream_keepalive,
        }
        ctx_handle = _register_ctx_handle(ctx_data)
        return (
            Ey,
            Hx,
            Hz,
            m_Ey_x,
            m_Ey_z,
            m_Hx_z,
            m_Hz_x,
            receiver_amplitudes,
            ctx_handle,
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
        (
            ca,
            cb,
            cq,
            _source_amplitudes_scaled,
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
            accuracy,
            ca_batched,
            cb_batched,
            cq_batched,
            pml_y0,
            pml_x0,
            pml_y1,
            pml_x1,
            fd_pad,
            pml_width,
            models,
            _forward_callback,
            backward_callback,
            callback_frequency,
            _scale_ctx,
            _storage_mode_str,
            _storage_format,
            _storage_path,
            _storage_compression,
            _Ey,
            _Hx,
            _Hz,
            _m_Ey_x,
            _m_Ey_z,
            _m_Hx_z,
            _m_Hz_x,
            n_threads,
            _backend_device,
        ) = inputs

        outputs = output if isinstance(output, tuple) else (output,)
        if len(outputs) != 9:
            raise RuntimeError(
                "MaxwellTMForwardFunc expected a context handle output for setup_context."
            )
        ctx_handle = outputs[-1]
        if not isinstance(ctx_handle, torch.Tensor):
            raise RuntimeError("MaxwellTMForwardFunc context handle must be a Tensor.")

        ctx_handle_id = int(ctx_handle.item())
        ctx_data = _get_ctx_handle(ctx_handle_id)
        ctx._ctx_handle_id = ctx_handle_id
        backward_storage_tensors = ctx_data["backward_storage_tensors"]

        ctx.save_for_backward(
            ca,
            cb,
            cq,
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
            *backward_storage_tensors,
        )
        ctx.backward_storage_objects = ctx_data["backward_storage_objects"]
        ctx.backward_storage_filename_arrays = ctx_data[
            "backward_storage_filename_arrays"
        ]
        ctx.stream_keepalive = ctx_data["stream_keepalive"]
        ctx.rdy = rdy
        ctx.rdx = rdx
        ctx.dt = dt
        ctx.nt = nt
        ctx.n_shots = n_shots
        ctx.ny = ny
        ctx.nx = nx
        ctx.n_sources = n_sources
        ctx.n_receivers = n_receivers
        ctx.step_ratio = step_ratio
        ctx.accuracy = accuracy
        ctx.ca_batched = ca_batched
        ctx.cb_batched = cb_batched
        ctx.cq_batched = cq_batched
        ctx.pml_y0 = pml_y0
        ctx.pml_x0 = pml_x0
        ctx.pml_y1 = pml_y1
        ctx.pml_x1 = pml_x1
        ctx.ca_requires_grad = ctx_data["ca_requires_grad"]
        ctx.cb_requires_grad = ctx_data["cb_requires_grad"]
        ctx.storage_mode = ctx_data["storage_mode"]
        ctx.storage_format = ctx_data["storage_format"]
        ctx.shot_bytes_uncomp = ctx_data["shot_bytes_uncomp"]
        ctx.fd_pad = fd_pad
        ctx.pml_width = pml_width
        ctx.models = models
        ctx.backward_callback = backward_callback
        ctx.callback_frequency = callback_frequency
        ctx.source_amplitudes_scaled = ctx_data["source_amplitudes_scaled"]
        ctx.n_threads = n_threads
        ctx.backend_device = _backend_device
        ctx.scale_ctx = ctx_data["scale_ctx"]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        from .. import backend_utils

        grad_outputs_list = list(grad_outputs)
        if len(grad_outputs_list) == 9:
            grad_outputs_list.pop()

        (
            grad_Ey,
            grad_Hx,
            grad_Hz,
            grad_m_Ey_x,
            grad_m_Ey_z,
            grad_m_Hx_z,
            grad_m_Hz_x,
            grad_r,
        ) = grad_outputs_list
        del grad_Ey, grad_Hx, grad_Hz, grad_m_Ey_x, grad_m_Ey_z, grad_m_Hx_z, grad_m_Hz_x

        saved = ctx.saved_tensors
        ca, cb, cq = saved[0], saved[1], saved[2]
        ay, by, ay_h, by_h = saved[3], saved[4], saved[5], saved[6]
        ax, bx, ax_h, bx_h = saved[7], saved[8], saved[9], saved[10]
        ky, ky_h, kx, kx_h = saved[11], saved[12], saved[13], saved[14]
        sources_i, receivers_i = saved[15], saved[16]
        ey_store_1, ey_store_3 = saved[17], saved[18]
        curl_store_1, curl_store_3 = saved[19], saved[20]

        device = ca.device
        coeff_dtype = ca.dtype
        scale_ctx = ctx.scale_ctx

        rdy = ctx.rdy
        rdx = ctx.rdx
        dt = ctx.dt
        nt = ctx.nt
        n_shots = ctx.n_shots
        ny = ctx.ny
        nx = ctx.nx
        n_sources = ctx.n_sources
        n_receivers = ctx.n_receivers
        step_ratio = ctx.step_ratio
        accuracy = ctx.accuracy
        ca_batched = ctx.ca_batched
        cb_batched = ctx.cb_batched
        cq_batched = ctx.cq_batched
        pml_y0 = ctx.pml_y0
        pml_x0 = ctx.pml_x0
        pml_y1 = ctx.pml_y1
        pml_x1 = ctx.pml_x1
        ca_requires_grad = ctx.ca_requires_grad
        cb_requires_grad = ctx.cb_requires_grad
        pml_width = ctx.pml_width
        storage_mode = ctx.storage_mode
        storage_format = ctx.storage_format
        shot_bytes_uncomp = ctx.shot_bytes_uncomp

        import ctypes

        if storage_mode == STORAGE_DISK:
            ey_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[0], ctypes.c_void_p
            )
            curl_filenames_ptr = ctypes.cast(
                ctx.backward_storage_filename_arrays[1], ctypes.c_void_p
            )
        else:
            ey_filenames_ptr = 0
            curl_filenames_ptr = 0

        if grad_r is None or grad_r.numel() == 0:
            grad_r = torch.zeros(
                nt,
                n_shots,
                n_receivers,
                device=device,
                dtype=coeff_dtype,
            )
        else:
            grad_r = grad_r.contiguous()

        lambda_ey = torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
        lambda_hx = torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
        lambda_hz = torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
        m_lambda_ey_x = torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
        m_lambda_ey_z = torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
        m_lambda_hx_z = torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
        m_lambda_hz_x = torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)

        if n_sources > 0:
            grad_f = torch.zeros(nt, n_shots, n_sources, device=device, dtype=coeff_dtype)
        else:
            grad_f = torch.empty(0, device=device, dtype=coeff_dtype)

        if ca_requires_grad:
            grad_ca = (
                torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
                if ca_batched
                else torch.zeros(ny, nx, device=device, dtype=coeff_dtype)
            )
            grad_ca_shot = torch.zeros(
                n_shots, ny, nx, device=device, dtype=coeff_dtype
            )
        else:
            grad_ca = torch.empty(0, device=device, dtype=coeff_dtype)
            grad_ca_shot = torch.empty(0, device=device, dtype=coeff_dtype)

        if cb_requires_grad:
            grad_cb = (
                torch.zeros(n_shots, ny, nx, device=device, dtype=coeff_dtype)
                if cb_batched
                else torch.zeros(ny, nx, device=device, dtype=coeff_dtype)
            )
            grad_cb_shot = torch.zeros(
                n_shots, ny, nx, device=device, dtype=coeff_dtype
            )
        else:
            grad_cb = torch.empty(0, device=device, dtype=coeff_dtype)
            grad_cb_shot = torch.empty(0, device=device, dtype=coeff_dtype)

        device_idx = (
            device.index if device.type == "cuda" and device.index is not None else 0
        )
        compute_stream_handle, storage_stream_handle, stream_keepalive = (
            _make_tm_storage_streams(device, storage_mode)
        )
        ctx.stream_keepalive = stream_keepalive

        backward_callback = ctx.backward_callback
        callback_frequency = ctx.callback_frequency
        fd_pad_ctx = ctx.fd_pad
        models = ctx.models
        n_threads = ctx.n_threads

        backward_func = backend_utils.get_backend_function(
            "maxwell_tm",
            "backward",
            accuracy,
            coeff_dtype,
            ctx.backend_device,
        )
        effective_callback_freq = (
            nt // step_ratio if backward_callback is None else callback_frequency
        )

        for step in range(nt // step_ratio, 0, -effective_callback_freq):
            step_nt = min(step, effective_callback_freq) * step_ratio
            start_t = step * step_ratio
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
                backend_utils.tensor_to_ptr(grad_ca),
                backend_utils.tensor_to_ptr(grad_cb),
                backend_utils.tensor_to_ptr(grad_ca_shot),
                backend_utils.tensor_to_ptr(grad_cb_shot),
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
                step_nt,
                n_shots,
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
                pml_y0,
                pml_x0,
                pml_y1,
                pml_x1,
                n_threads,
                device_idx,
                compute_stream_handle,
                storage_stream_handle,
            )

            if backward_callback is not None:
                callback_wavefields = _physical_tm2d_adjoint_callback_wavefields(
                    {
                        "lambda_Ey": lambda_ey,
                        "lambda_Hx": lambda_hx,
                        "lambda_Hz": lambda_hz,
                        "m_lambda_Ey_x": m_lambda_ey_x,
                        "m_lambda_Ey_z": m_lambda_ey_z,
                        "m_lambda_Hx_z": m_lambda_hx_z,
                        "m_lambda_Hz_x": m_lambda_hz_x,
                    },
                    scale_ctx=scale_ctx,
                )
                callback_gradients = {}
                if ca_requires_grad:
                    callback_gradients["ca"] = grad_ca
                if cb_requires_grad:
                    callback_gradients["cb"] = grad_cb
                if ca_requires_grad or cb_requires_grad:
                    with torch.enable_grad():
                        eps_req = models["epsilon"].detach().requires_grad_(True)
                        sig_req = models["sigma"].detach().requires_grad_(True)
                        mu_req = models["mu"]

                        ca_v, cb_v, _ = prepare_parameters(eps_req, sig_req, mu_req, dt)

                        vjp_tensors = []
                        vjp_grads = []
                        if ca_requires_grad:
                            vjp_tensors.append(ca_v)
                            vjp_grads.append(grad_ca)
                        if cb_requires_grad:
                            vjp_tensors.append(cb_v)
                            vjp_grads.append(callback_gradients["cb"])

                        torch.autograd.backward(vjp_tensors, vjp_grads)
                        callback_gradients["epsilon"] = eps_req.grad
                        callback_gradients["sigma"] = sig_req.grad

                backward_callback(
                    CallbackState(
                        dt=dt,
                        step=step - 1,
                        nt=nt // step_ratio,
                        wavefields=callback_wavefields,
                        models=models,
                        gradients=callback_gradients,
                        fd_pad=list(fd_pad_ctx),
                        pml_width=list(pml_width),
                        is_backward=True,
                    )
                )

        grad_f_flat = grad_f.reshape(nt * n_shots * n_sources) if n_sources > 0 else None
        if ca_requires_grad and not ca_batched:
            grad_ca = grad_ca.unsqueeze(0)
        if cb_requires_grad and not cb_batched:
            grad_cb = grad_cb.unsqueeze(0)

        _release_ctx_handle(getattr(ctx, "_ctx_handle_id", None))
        return (
            grad_ca if ca_requires_grad else None,
            grad_cb if cb_requires_grad else None,
            None,
            grad_f_flat,
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
        )


__all__ = ["MaxwellTMForwardFunc"]
