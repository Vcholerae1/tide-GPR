from __future__ import annotations

import argparse
import tide
import time
import torch
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    dtype = torch.float64
    ny, nx, nt = 26, 30, 100
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype, requires_grad=True)
    sigma = torch.full((ny, nx), 2.0e-4, dtype=dtype, requires_grad=True)
    mu = torch.ones_like(epsilon)
    wavelet = tide.ricker(300e6, nt, 2.5e-11, peak_time=1.0e-9, dtype=dtype)
    source_amplitude = wavelet.view(1, 1, nt).expand(3, -1, -1).contiguous()
    source_location = torch.tensor([[[10, 7]], [[13, 7]], [[16, 7]]], dtype=torch.long)
    receiver_location = torch.tensor(
        [
            [[10, 15], [10, 20]],
            [[13, 15], [13, 20]],
            [[16, 15], [16, 20]],
        ],
        dtype=torch.long,
    )

    started = time.perf_counter()
    receiver = tide.maxwell._kernel_api.maxwelltm(
        epsilon,
        sigma,
        mu,
        [0.018, 0.022],
        2.5e-11,
        source_amplitude,
        source_location,
        receiver_location,
        stencil=4,
        pml_width=4,
        python_backend=False,
        storage_compression=False,
    )[-1]
    receiver.square().mean().backward()
    elapsed_seconds = time.perf_counter() - started
    assert epsilon.grad is not None and sigma.grad is not None
    torch.save(
        {
            "receiver": receiver.detach(),
            "epsilon_gradient": epsilon.grad.detach(),
            "sigma_gradient": sigma.grad.detach(),
            "elapsed_seconds": elapsed_seconds,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
