from typing import Any

import torch

from ..padding import create_or_pad
from ..storage import _resolve_storage_compression
from .common import _make_storage_streams


def _make_tm_storage_streams(
    device: torch.device, storage_mode: int
) -> tuple[int, int, tuple[Any, ...]]:
    return _make_storage_streams(device, storage_mode)


def _init_tm_wavefield(
    field_0: torch.Tensor | None,
    *,
    n_shots: int,
    size_with_batch: tuple[int, int, int],
    fd_pad_list: list[int],
    device: torch.device,
    dtype: torch.dtype,
    contiguous: bool = False,
) -> torch.Tensor:
    if field_0 is not None:
        if field_0.ndim == 2:
            field_0 = field_0[None, :, :].expand(n_shots, -1, -1)
        wavefield = create_or_pad(
            field_0,
            fd_pad_list,
            device,
            dtype,
            size_with_batch,
            mode="constant",
        )
    else:
        wavefield = torch.zeros(size_with_batch, device=device, dtype=dtype)
    return wavefield.contiguous() if contiguous else wavefield


def _resolve_tm2d_storage_spec(
    *,
    storage_compression: bool | str | None,
    dtype: torch.dtype,
    device: torch.device,
    context: str,
) -> tuple[str, torch.dtype, int, int]:
    storage_kind, store_dtype, itemsize, storage_format = _resolve_storage_compression(
        storage_compression,
        dtype,
        device,
        context=context,
    )
    return storage_kind, store_dtype, itemsize, storage_format


def _prepare_tm2d_source_injection(
    *,
    source_amplitude: torch.Tensor | None,
    cb_at_src: torch.Tensor | None,
    source_coeff: float,
    dtype: torch.dtype,
    n_shots: int,
    n_sources: int,
    nt_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = (
        source_amplitude.device
        if source_amplitude is not None
        else cb_at_src.device if cb_at_src is not None else torch.device("cpu")
    )
    if (
        source_amplitude is None
        or source_amplitude.numel() == 0
        or cb_at_src is None
        or cb_at_src.numel() == 0
        or n_sources == 0
    ):
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, torch.zeros(n_shots, device=device, dtype=torch.float32)

    source_abs_max = source_amplitude.detach().abs().amax(dim=2).to(torch.float32)
    cb_abs = cb_at_src.detach().abs().to(torch.float32)
    f_shot = (source_abs_max * cb_abs * abs(source_coeff)).amax(dim=1)

    source = source_amplitude.permute(2, 0, 1).contiguous().to(dtype)
    source = source * cb_at_src.to(dtype).unsqueeze(0)
    source.mul_(source_coeff)
    return source.reshape(nt_steps * n_shots * n_sources).contiguous(), f_shot


def _unscale_tm2d_outputs(
    *,
    scale_ctx: dict[str, Any] | None,
    Ey: torch.Tensor,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    m_Ey_x: torch.Tensor,
    m_Ey_z: torch.Tensor,
    m_Hx_z: torch.Tensor,
    m_Hz_x: torch.Tensor,
    receiver_amplitudes: torch.Tensor,
    inplace_float_outputs: bool = False,
) -> tuple[torch.Tensor, ...]:
    del scale_ctx, inplace_float_outputs
    return Ey, Hx, Hz, m_Ey_x, m_Ey_z, m_Hx_z, m_Hz_x, receiver_amplitudes


def _physical_tm2d_callback_wavefields(
    wavefields: dict[str, torch.Tensor],
    *,
    scale_ctx: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    del scale_ctx
    return wavefields


def _physical_tm2d_adjoint_callback_wavefields(
    wavefields: dict[str, torch.Tensor],
    *,
    scale_ctx: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    del scale_ctx
    return {name: tensor.float() for name, tensor in wavefields.items()}


__all__ = [
    "_init_tm_wavefield",
    "_make_tm_storage_streams",
    "_physical_tm2d_adjoint_callback_wavefields",
    "_physical_tm2d_callback_wavefields",
    "_prepare_tm2d_source_injection",
    "_resolve_tm2d_storage_spec",
    "_unscale_tm2d_outputs",
]
