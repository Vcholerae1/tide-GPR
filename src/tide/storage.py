"""Storage helpers for wavefield snapshots.

This mirrors Deepwave's snapshot storage abstraction for use in the Maxwell
propagator. Stage 1 supports snapshot storage on device/CPU/disk.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
from enum import IntEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch


class StorageMode(IntEnum):
    """Snapshot storage location for native forward/backward state."""

    DEVICE = 0
    CPU = 1
    DISK = 2
    NONE = 3


# Snapshot storage modes: prefer DEVICE, fall back to CPU or DISK; NONE disables snapshotting
STORAGE_DEVICE = StorageMode.DEVICE  # Keep snapshots on the accelerator (fastest)
STORAGE_CPU = StorageMode.CPU  # Stage snapshots in host memory
STORAGE_DISK = StorageMode.DISK  # Spill snapshots to disk
STORAGE_NONE = StorageMode.NONE  # Do not store snapshots

# Snapshot payload formats. These are passed to the native TM2D storage path so
# it can distinguish full-precision, bf16-compressed, and fp16 payloads without
# guessing from element size alone.
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
    compute_precision: str = "default",
) -> tuple[str, torch.dtype, int, int]:
    if compute_precision != "default":
        raise ValueError(
            f"{context} only supports compute_precision='default', got {compute_precision!r}."
        )

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


def get_storage_mode(storage_mode_str: str, device: torch.device) -> StorageMode:
    """Convert a storage mode string to a StorageMode enum."""
    if str(device) == "cpu" and storage_mode_str == "cpu":
        storage_mode_str = "device"

    mode = storage_mode_str.lower()
    if mode == "device":
        return StorageMode.DEVICE
    if mode == "cpu":
        return StorageMode.CPU
    if mode == "disk":
        return StorageMode.DISK
    if mode == "none":
        return StorageMode.NONE
    raise ValueError(
        f"storage_mode must be 'device', 'cpu', 'disk', or 'none', but got {storage_mode_str!r}"
    )


def storage_mode_to_int(storage_mode_str: str) -> int:
    return int(get_storage_mode(storage_mode_str, torch.device("cuda")))


class TemporaryStorage:
    """Manages temporary files for disk storage.

    Creates a unique subdirectory for each instantiation to prevent collisions.
    """

    def __init__(self, base_path: str, num_files: int) -> None:
        self.base_dir = Path(base_path) / f"tide_tmp_{os.getpid()}_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.filenames: list[bytes] = []
        for i in range(num_files):
            self.filenames.append(str(self.base_dir / f"shot_{i}.bin").encode("utf-8"))
        self.filenames_ptr = (ctypes.c_char_p * len(self.filenames))(*self.filenames)

    def get_filenames_ptr(self) -> ctypes.Array[ctypes.c_char_p]:
        return self.filenames_ptr

    def close(self) -> None:
        if self.base_dir.exists():
            with contextlib.suppress(OSError):
                shutil.rmtree(self.base_dir)

    def __del__(self) -> None:
        self.close()


class IntermediateStorage:
    """Owns storage tensors and the optional disk-backed file list."""

    def __init__(
        self,
        store_device: torch.Tensor,
        store_host: torch.Tensor,
        temporary_storage: TemporaryStorage | None,
    ) -> None:
        self.store_device = store_device
        self.store_host = store_host
        self.temporary_storage = temporary_storage

    def get_filenames_ptr(self) -> ctypes.Array[ctypes.c_char_p]:
        if self.temporary_storage is None:
            return (ctypes.c_char_p * 0)()
        return self.temporary_storage.get_filenames_ptr()


class StorageManager:
    """Centralized allocation helper for native snapshot storage."""

    def __init__(
        self,
        *,
        shot_shape: tuple[int, ...],
        dtype: torch.dtype,
        n_shots: int,
        nt: int,
        step_ratio: int,
        storage_mode: StorageMode,
        storage_path: str,
        device: torch.device,
        num_device_buffers: int = _CPU_STORAGE_BUFFERS,
    ) -> None:
        self.shot_shape = shot_shape
        self.dtype = dtype
        self.n_shots = n_shots
        self.nt = nt
        self.step_ratio = step_ratio
        self.storage_mode = storage_mode
        self.storage_path = storage_path
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.num_steps_stored = (nt + step_ratio - 1) // step_ratio
        self.num_device_buffers = num_device_buffers
        self.allocations: list[IntermediateStorage] = []

    def allocate(
        self,
        *,
        requires_grad: bool,
        dtype: torch.dtype | None = None,
        host_linear: bool = False,
    ) -> IntermediateStorage:
        store_dtype = self.dtype if dtype is None else dtype
        store_device = torch.empty(0, device=self.device, dtype=store_dtype)
        store_host = torch.empty(0)
        temporary_storage: TemporaryStorage | None = None

        if requires_grad and self.storage_mode != StorageMode.NONE:
            flat_numel = int(torch.prod(torch.tensor(self.shot_shape)).item())
            host_shape = (
                (self.num_steps_stored, self.n_shots, flat_numel)
                if host_linear
                else (self.num_steps_stored, self.n_shots, *self.shot_shape)
            )
            ring_host_shape = (
                (self.num_device_buffers, self.n_shots, flat_numel)
                if host_linear
                else (self.num_device_buffers, self.n_shots, *self.shot_shape)
            )
            device_ring_shape = (
                self.num_device_buffers,
                self.n_shots,
                *self.shot_shape,
            )

            if self.storage_mode == StorageMode.DEVICE:
                store_device = torch.empty(
                    self.num_steps_stored,
                    self.n_shots,
                    *self.shot_shape,
                    device=self.device,
                    dtype=store_dtype,
                )
            elif self.storage_mode == StorageMode.CPU:
                store_device = torch.empty(
                    *device_ring_shape,
                    device=self.device,
                    dtype=store_dtype,
                )
                store_host = torch.empty(
                    *host_shape,
                    device="cpu",
                    pin_memory=True,
                    dtype=store_dtype,
                )
            elif self.storage_mode == StorageMode.DISK:
                temporary_storage = TemporaryStorage(
                    self.storage_path, 1 if self.is_cuda else self.n_shots
                )
                if self.is_cuda:
                    store_device = torch.empty(
                        *device_ring_shape,
                        device=self.device,
                        dtype=store_dtype,
                    )
                    store_host = torch.empty(
                        *ring_host_shape,
                        device="cpu",
                        pin_memory=True,
                        dtype=store_dtype,
                    )
                else:
                    store_device = torch.empty(
                        self.n_shots,
                        *self.shot_shape,
                        device=self.device,
                        dtype=store_dtype,
                    )

        allocation = IntermediateStorage(store_device, store_host, temporary_storage)
        self.allocations.append(allocation)
        return allocation


def setup_storage(
    *,
    shot_shape: tuple[int, ...],
    dtype: torch.dtype,
    n_shots: int,
    nt: int,
    step_ratio: int,
    storage_mode: StorageMode,
    storage_path: str,
    device: torch.device,
    requires_grad_list: list[bool],
    allocation_kwargs_list: list[dict[str, Any]] | None = None,
) -> StorageManager:
    """Create a StorageManager and allocate per-component storage."""
    storage_manager = StorageManager(
        shot_shape=shot_shape,
        dtype=dtype,
        n_shots=n_shots,
        nt=nt,
        step_ratio=step_ratio,
        storage_mode=storage_mode,
        storage_path=storage_path,
        device=device,
    )
    if allocation_kwargs_list is None:
        allocation_kwargs_list = [{} for _ in requires_grad_list]
    for requires_grad, allocation_kwargs in zip(
        requires_grad_list, allocation_kwargs_list, strict=True
    ):
        storage_manager.allocate(
            requires_grad=requires_grad,
            **allocation_kwargs,
        )
    return storage_manager
