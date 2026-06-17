"""Source amplitude helpers for TIDE workflows."""

from __future__ import annotations

import torch


def expand_source_amplitude(
    wavelet: torch.Tensor,
    n_shots: int,
    *,
    n_sources: int = 1,
) -> torch.Tensor:
    """Expand a wavelet to solver source-amplitude shape ``[S, Ns, nt]``."""

    n_shots = int(n_shots)
    n_sources = int(n_sources)
    if n_shots <= 0:
        raise ValueError("n_shots must be positive.")
    if n_sources <= 0:
        raise ValueError("n_sources must be positive.")
    if wavelet.ndim == 1:
        amplitude = wavelet.reshape(1, 1, -1).expand(n_shots, n_sources, -1)
    elif wavelet.ndim == 2:
        if int(wavelet.shape[0]) != n_sources:
            raise ValueError("2D wavelet input must be shaped [n_sources, nt].")
        amplitude = wavelet.unsqueeze(0).expand(n_shots, -1, -1)
    else:
        raise ValueError("wavelet must be shaped [nt] or [n_sources, nt].")
    return amplitude.contiguous()


__all__ = ["expand_source_amplitude"]
