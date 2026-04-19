from typing import Any

import torch

from ..callbacks import CallbackState
from ..storage import (
    _CPU_STORAGE_BUFFERS,
    STORAGE_CPU,
    STORAGE_DEVICE,
    STORAGE_DISK,
    TemporaryStorage,
    _resolve_storage_compression,
    storage_mode_to_int,
)
from .common import _make_storage_streams

class Maxwell3DForwardFunc(torch.autograd.Function):
    """Autograd function for 3D C/CUDA backend propagation."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        ca: torch.Tensor,
        cb: torch.Tensor,
        cq: torch.Tensor,
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
        shot_numel = nz * ny * nx
        _, store_dtype, _, storage_format = _resolve_storage_compression(
            meta["storage_compression"],
            dtype,
            device,
            context="storage_compression",
        )
        shot_bytes_uncomp = shot_numel * store_dtype.itemsize
        storage_mode_str = str(meta["storage_mode_str"]).lower()
        storage_path = str(meta["storage_path"])
        if device.type == "cpu" and storage_mode_str in {"cpu", "disk"}:
            storage_mode_str = "device"
        storage_mode = storage_mode_to_int(storage_mode_str)
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
                    nz,
                    ny,
                    nx,
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
                filenames_list = [f.encode("utf-8") for f in storage_obj.get_filenames()]
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

        store_ex, store_ex_host, store_ex_filenames_ptr = alloc_storage(ca_requires_grad)
        store_ey, store_ey_host, store_ey_filenames_ptr = alloc_storage(ca_requires_grad)
        store_ez, store_ez_host, store_ez_filenames_ptr = alloc_storage(ca_requires_grad)
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
            grad_sigma = (
                torch.zeros(n_shots, nz, ny, nx, device=device, dtype=dtype)
                if ca_batched or cb_batched
                else torch.zeros(nz, ny, nx, device=device, dtype=dtype)
            )
        else:
            grad_eps = torch.empty(0, device=device, dtype=dtype)
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
                ctypes.cast(
                    ctx.backward_storage_filename_arrays[0], ctypes.c_void_p
                )
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_ey),
                backend_utils.tensor_to_ptr(store_ey_host),
                ctypes.cast(
                    ctx.backward_storage_filename_arrays[1], ctypes.c_void_p
                )
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_ez),
                backend_utils.tensor_to_ptr(store_ez_host),
                ctypes.cast(
                    ctx.backward_storage_filename_arrays[2], ctypes.c_void_p
                )
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_curl_x),
                backend_utils.tensor_to_ptr(store_curl_x_host),
                ctypes.cast(
                    ctx.backward_storage_filename_arrays[3], ctypes.c_void_p
                )
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_curl_y),
                backend_utils.tensor_to_ptr(store_curl_y_host),
                ctypes.cast(
                    ctx.backward_storage_filename_arrays[4], ctypes.c_void_p
                )
                if storage_mode == STORAGE_DISK
                else 0,
                backend_utils.tensor_to_ptr(store_curl_z),
                backend_utils.tensor_to_ptr(store_curl_z_host),
                ctypes.cast(
                    ctx.backward_storage_filename_arrays[5], ctypes.c_void_p
                )
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

        if n_sources > 0:
            grad_f_flat = grad_f.reshape(nt * n_shots * n_sources)
        else:
            grad_f_flat = None

        grad_ca_out = (
            grad_ca.unsqueeze(0) if ca_requires_grad and not ca_batched else grad_ca
        ) if ca_requires_grad else None
        grad_cb_out = (
            grad_cb.unsqueeze(0) if cb_requires_grad and not cb_batched else grad_cb
        ) if cb_requires_grad else None
        return (
            grad_ca_out,
            grad_cb_out,
            None,
            grad_f_flat,
            None,
            None,
            None,
            None,
        )

__all__ = ["Maxwell3DForwardFunc"]
