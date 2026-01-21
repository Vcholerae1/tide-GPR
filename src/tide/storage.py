"""Storage helpers for wavefield snapshots.

This mirrors Deepwave's snapshot storage abstraction for use in the Maxwell
propagator. Stage 1 supports snapshot storage on device/CPU/disk.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from typing import Union
from uuid import uuid4

import torch

# Snapshot storage modes: prefer DEVICE, fall back to CPU or DISK; NONE disables snapshotting
STORAGE_DEVICE = 0  # Keep snapshots on the accelerator (fastest, uses device memory)
STORAGE_CPU    = 1  # Stage snapshots in host memory (slower, avoids GPU OOM)
STORAGE_DISK   = 2  # Spill snapshots to disk (slowest, preserves host/GPU memory)
STORAGE_NONE   = 3  # Do not store snapshots

# Number of ring buffers for CPU-stage ping-pong: allows overlapping reads/writes
# (write to one, read from another, keep one ready). MUST match csrc NUM_BUFFERS.
_CPU_STORAGE_BUFFERS = 3


def _normalize_storage_compression(storage_compression: Union[bool, str, None]) -> str:
    """Normalize the storage compression setting to a standard string.

    Args:
        storage_compression: The input storage compression setting, which can be
            a boolean, a string, or None.

    Returns:
        A normalized string representing the storage compression mode:
        - "none" for no compression
        - "bf16" for bfloat16 compression
        - "fp8" for float8 compression

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
        if value in {"fp8", "float8", "e4m3", "e4m3fn", "fp8_e4m3"}:
            return "fp8"
    raise ValueError(
        "storage_compression must be False/True or one of 'none', 'bf16', or 'fp8'."
    )


def _resolve_storage_compression(
    storage_compression: Union[bool, str, None],
    dtype: torch.dtype,
    device: torch.device,
    *,
    context: str,
    allow_fp8: bool = True,
) -> tuple[str, torch.dtype, int]:
    storage_kind = _normalize_storage_compression(storage_compression)
    if storage_kind == "none":
        return storage_kind, dtype, dtype.itemsize
    if storage_kind == "bf16":
        if dtype != torch.float32:
            raise NotImplementedError(
                f"{context} (BF16 storage) is only supported for float32."
            )
        return storage_kind, torch.bfloat16, 2
    if storage_kind == "fp8":
        if not allow_fp8:
            raise NotImplementedError(
                f"{context} (FP8 storage) is not supported in this path."
            )
        # FP8 now supported on both CUDA and CPU
        if dtype != torch.float32:
            raise NotImplementedError(
                f"{context} (FP8 storage) is only supported for float32."
            )
        return storage_kind, torch.uint8, 1
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


class TemporaryStorage:
    """Manages temporary files for disk storage.

    Creates a unique subdirectory for each instantiation to prevent collisions.
    """

    def __init__(self, base_path: str, num_files: int) -> None:
        self.base_dir = Path(base_path) / f"tide_tmp_{os.getpid()}_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.filenames: list[str] = [
            str(self.base_dir / f"shot_{i}.bin") for i in range(num_files)
        ]

    def get_filenames(self) -> list[str]:
        return self.filenames

    def close(self) -> None:
        if self.base_dir.exists():
            with contextlib.suppress(OSError):
                shutil.rmtree(self.base_dir)

    def __del__(self) -> None:
        self.close()
