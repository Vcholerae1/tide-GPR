"""Storage helpers for wavefield snapshots.

This mirrors Deepwave's snapshot storage abstraction for use in the Maxwell
propagator. Stage 1 supports snapshot storage on device/CPU/disk.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory

import torch

# Snapshot storage modes: prefer DEVICE, fall back to CPU or DISK; NONE disables snapshotting
STORAGE_DEVICE = 0  # Keep snapshots on the accelerator (fastest, uses device memory)
STORAGE_CPU = 1  # Stage snapshots in host memory (slower, avoids GPU OOM)
STORAGE_DISK = 2  # Spill snapshots to disk (slowest, preserves host/GPU memory)
STORAGE_NONE = 3  # Do not store snapshots

# Snapshot payload formats. These are passed to the native TM2D storage path so
# it can distinguish full-precision and bf16-compressed payloads explicitly.
STORAGE_FORMAT_FULL = 0
STORAGE_FORMAT_BF16 = 1

# Number of ring buffers for host-staged snapshot storage. CUDA CPU- and
# disk-backed storage use the same ring size and must match csrc NUM_BUFFERS.
_CPU_STORAGE_BUFFERS = 3


def _normalize_storage_compression(storage_compression: bool | str | None) -> str:
    """Normalize the storage compression setting to a standard string.

    Args:
        storage_compression: The input storage compression setting, which can be
            a boolean, a string, or None.

    Returns:
        A normalized string representing the storage compression mode:
        - "none" for no compression
        - "bf16" for bfloat16 compression

    Raises:
        ValueError: If the input value is not recognized.
    """
    if storage_compression is True:
        return "bf16"
    if storage_compression is False or storage_compression is None:
        return "none"
    if isinstance(storage_compression, str):
        value = storage_compression.strip().lower()
        if value in {"none", "false", "off", "0"}:
            return "none"
        if value in {"bf16", "bfloat16"}:
            return "bf16"
    raise ValueError(
        "storage_compression must be False/True or one of 'none' or 'bf16'."
    )


def _resolve_storage_compression(
    storage_compression: bool | str | None,
    dtype: torch.dtype,
    device: torch.device,
    *,
    context: str,
) -> tuple[str, torch.dtype, int, int]:
    storage_kind = _normalize_storage_compression(storage_compression)
    if storage_kind == "none":
        return storage_kind, dtype, dtype.itemsize, STORAGE_FORMAT_FULL
    if storage_kind == "bf16":
        if dtype != torch.float32:
            raise NotImplementedError(
                f"{context} (BF16 storage) is only supported for float32."
            )
        return storage_kind, torch.bfloat16, 2, STORAGE_FORMAT_BF16
    raise RuntimeError(f"Unsupported storage compression mode: {storage_kind}")


def storage_mode_to_int(storage_mode_str: str) -> int:
    mode = storage_mode_str.lower()
    if mode == "device":
        return STORAGE_DEVICE
    if mode == "cpu":
        return STORAGE_CPU
    if mode == "disk":
        return STORAGE_DISK
    if mode == "none":
        return STORAGE_NONE
    raise ValueError(
        "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
        f"but got {storage_mode_str!r}"
    )


@dataclass(frozen=True, slots=True)
class SnapshotStorageSpec:
    """Resolved snapshot representation shared by Maxwell operators."""

    mode_name: str
    mode: int
    compression: str
    dtype: torch.dtype
    format: int
    num_steps: int
    shot_shape: tuple[int, ...]

    @property
    def shot_numel(self) -> int:
        return prod(self.shot_shape)

    @property
    def shot_bytes(self) -> int:
        if not self.shot_shape:
            return 0
        return self.shot_numel // self.shot_shape[0] * self.dtype.itemsize

    @property
    def history_shape(self) -> tuple[int, ...]:
        return (self.num_steps, *self.shot_shape)


def resolve_snapshot_storage(
    *,
    storage_mode: str,
    storage_compression: bool | str | None,
    dtype: torch.dtype,
    device: torch.device,
    nt: int,
    step_ratio: int,
    shot_shape: tuple[int, ...],
    context: str = "storage_compression",
    enabled: bool = True,
    cpu_alias_modes: tuple[str, ...] = ("cpu",),
) -> SnapshotStorageSpec:
    """Normalize mode, compression, temporal sampling, and snapshot shape."""

    mode_name = str(storage_mode).lower()
    if not enabled:
        mode_name = "none"
    elif device.type == "cpu" and mode_name in cpu_alias_modes:
        mode_name = "device"
    mode = storage_mode_to_int(mode_name)
    compression, store_dtype, _, storage_format = _resolve_storage_compression(
        storage_compression,
        dtype,
        device,
        context=context,
    )
    resolved_step_ratio = max(1, int(step_ratio))
    num_steps = (int(nt) + resolved_step_ratio - 1) // resolved_step_ratio
    return SnapshotStorageSpec(
        mode_name=mode_name,
        mode=mode,
        compression=compression,
        dtype=store_dtype,
        format=storage_format,
        num_steps=num_steps,
        shot_shape=tuple(int(size) for size in shot_shape),
    )


@dataclass(frozen=True, slots=True)
class SnapshotAllocation:
    device: torch.Tensor
    host: torch.Tensor
    filenames_ptr: Any = 0


@dataclass(slots=True)
class SnapshotAllocator:
    """Allocate and retain snapshot buffers for one resolved storage policy."""

    spec: SnapshotStorageSpec
    device: torch.device
    storage_path: str = "."
    host_flatten_spatial: bool = False
    tensors: list[torch.Tensor] = field(default_factory=list)
    storage_objects: list[TemporaryDirectory[str]] = field(default_factory=list)
    filename_arrays: list[Any] = field(default_factory=list)
    filename_buffers: list[Any] = field(default_factory=list)

    def empty(self) -> torch.Tensor:
        return torch.empty(0, device=self.device, dtype=self.spec.dtype)

    def direct(self, enabled: bool, *, final_only: bool = False) -> torch.Tensor:
        """Allocate device-resident history not managed by staged storage."""

        if not enabled:
            return self.empty()
        shape = self.spec.shot_shape if final_only else self.spec.history_shape
        tensor = torch.empty(shape, device=self.device, dtype=self.spec.dtype)
        self.tensors.append(tensor)
        return tensor

    def group(
        self,
        count: int,
        enabled: bool,
        *,
        final_only: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(self.direct(enabled, final_only=final_only) for _ in range(count))

    def allocate(self, enabled: bool) -> SnapshotAllocation:
        empty = self.empty()
        filenames_array = (ctypes.c_char_p * 0)()
        self.filename_arrays.append(filenames_array)
        if not enabled or self.spec.mode == STORAGE_NONE:
            self.tensors.extend((empty, empty))
            return SnapshotAllocation(empty, empty)

        device_store = empty
        host_store = empty
        if self.spec.mode == STORAGE_DEVICE:
            device_store = torch.empty(
                self.spec.history_shape,
                device=self.device,
                dtype=self.spec.dtype,
            )
        elif self.spec.mode == STORAGE_CPU:
            device_store = torch.empty(
                _CPU_STORAGE_BUFFERS,
                *self.spec.shot_shape,
                device=self.device,
                dtype=self.spec.dtype,
            )
            host_shape = self._host_history_shape()
            host_store = torch.empty(
                host_shape,
                device="cpu",
                pin_memory=True,
                dtype=self.spec.dtype,
            )
        elif self.spec.mode == STORAGE_DISK:
            is_cuda = self.device.type == "cuda"
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            storage = TemporaryDirectory(
                prefix="tide_tmp_",
                dir=self.storage_path,
                ignore_cleanup_errors=True,
            )
            self.storage_objects.append(storage)
            filenames = [
                str(Path(storage.name) / f"shot_{index}.bin")
                for index in range(1 if is_cuda else self.spec.shot_shape[0])
            ]
            buffers = [
                ctypes.create_string_buffer(name.encode("utf-8")) for name in filenames
            ]
            self.filename_buffers.extend(buffers)
            filenames_array = (ctypes.c_char_p * len(buffers))()
            for index, buffer in enumerate(buffers):
                filenames_array[index] = ctypes.cast(buffer, ctypes.c_char_p)
            self.filename_arrays[-1] = filenames_array
            if is_cuda:
                ring_shape = (_CPU_STORAGE_BUFFERS, *self.spec.shot_shape)
                device_store = torch.empty(
                    ring_shape, device=self.device, dtype=self.spec.dtype
                )
                host_store = torch.empty(
                    self._host_ring_shape(),
                    device="cpu",
                    pin_memory=True,
                    dtype=self.spec.dtype,
                )
            else:
                device_store = torch.empty(
                    self.spec.shot_shape,
                    device=self.device,
                    dtype=self.spec.dtype,
                )

        self.tensors.extend((device_store, host_store))
        filenames_ptr = (
            ctypes.cast(filenames_array, ctypes.c_void_p)
            if self.spec.mode == STORAGE_DISK
            else 0
        )
        return SnapshotAllocation(device_store, host_store, filenames_ptr)

    def _host_history_shape(self) -> tuple[int, ...]:
        if not self.host_flatten_spatial or len(self.spec.shot_shape) < 2:
            return self.spec.history_shape
        n_shots = self.spec.shot_shape[0]
        spatial_numel = self.spec.shot_numel // n_shots
        return (self.spec.num_steps, n_shots, spatial_numel)

    def _host_ring_shape(self) -> tuple[int, ...]:
        if not self.host_flatten_spatial or len(self.spec.shot_shape) < 2:
            return (_CPU_STORAGE_BUFFERS, *self.spec.shot_shape)
        n_shots = self.spec.shot_shape[0]
        spatial_numel = self.spec.shot_numel // n_shots
        return (_CPU_STORAGE_BUFFERS, n_shots, spatial_numel)
