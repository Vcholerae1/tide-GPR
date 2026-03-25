import argparse
import statistics
import time
from importlib import import_module

import torch

import tide

maxwell_module = import_module("tide.maxwell")


def _build_case(
    *,
    nz: int,
    ny: int,
    nx: int,
    nt: int,
    shots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | float | list[float]]:
    epsilon = torch.full((nz, ny, nx), 4.0, device=device, dtype=dtype)
    sigma = torch.full_like(epsilon, 2e-4)
    mu = torch.ones_like(epsilon)
    source_location = torch.tensor(
        [[[nz // 2, ny // 2, max(1, nx // 3)]]],
        dtype=torch.long,
        device=device,
    ).repeat(shots, 1, 1)
    receiver_location = torch.tensor(
        [[[nz // 2, ny // 2, min(nx - 2, max(2, (2 * nx) // 3))]]],
        dtype=torch.long,
        device=device,
    ).repeat(shots, 1, 1)
    source_amplitude = tide.ricker(
        80e6,
        nt,
        4e-11,
        peak_time=1.0 / 80e6,
        dtype=dtype,
        device=device,
    ).view(1, 1, nt).repeat(shots, 1, 1)
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "grid_spacing": [0.03, 0.02, 0.02],

        "dt": 4e-11,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
    }


def _run_forward(
    case: dict[str, torch.Tensor | float | list[float]],
    *,
    pml_width: int,
    experimental_cuda_graph: bool,
    callback_frequency: int,
    touch_callback_state: bool,
) -> torch.Tensor:
    kwargs = {
        **case,
        "pml_width": pml_width,
        "python_backend": False,
        "storage_mode": "none",
        "experimental_cuda_graph": experimental_cuda_graph,
    }
    if callback_frequency > 0:

        def _callback(state):
            if touch_callback_state:
                _ = state.get_wavefield("Ey", view="inner").abs().amax()

        kwargs["forward_callback"] = _callback
        kwargs["callback_frequency"] = callback_frequency

    return tide.maxwell3d(**kwargs)[-1]


def _benchmark(
    case: dict[str, torch.Tensor | float | list[float]],
    *,
    pml_width: int,
    experimental_cuda_graph: bool,
    callback_frequency: int,
    touch_callback_state: bool,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        _run_forward(
            case,
            pml_width=pml_width,
            experimental_cuda_graph=experimental_cuda_graph,
            callback_frequency=callback_frequency,
            touch_callback_state=touch_callback_state,
        )
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _run_forward(
            case,
            pml_width=pml_width,
            experimental_cuda_graph=experimental_cuda_graph,
            callback_frequency=callback_frequency,
            touch_callback_state=touch_callback_state,
        )
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(samples_ms)
    std_ms = statistics.pstdev(samples_ms)
    return mean_ms, std_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark experimental CUDA Graph mode for maxwell3d forward."
    )
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nt", type=int, default=96)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--pml-width", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--callback-frequency",
        type=int,
        default=0,
        help="0 disables callbacks. Positive values benchmark chunked callback mode.",
    )
    parser.add_argument(
        "--touch-callback-state",
        action="store_true",
        help="Make the callback read Ey(inner) to include callback state access cost.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify graph/non-graph receiver traces match before timing.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    case = _build_case(
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        nt=args.nt,
        shots=args.shots,
        device=device,
        dtype=dtype,
    )

    if args.verify:
        maxwell_module._clear_maxwell3d_cuda_graph_cache()
        out_plain = _run_forward(
            case,
            pml_width=args.pml_width,
            experimental_cuda_graph=False,
            callback_frequency=args.callback_frequency,
            touch_callback_state=args.touch_callback_state,
        )
        out_graph = _run_forward(
            case,
            pml_width=args.pml_width,
            experimental_cuda_graph=True,
            callback_frequency=args.callback_frequency,
            touch_callback_state=args.touch_callback_state,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out_graph, out_plain, rtol=1e-5, atol=1e-6)

    maxwell_module._clear_maxwell3d_cuda_graph_cache()
    plain_ms, plain_std = _benchmark(
        case,
        pml_width=args.pml_width,
        experimental_cuda_graph=False,
        callback_frequency=args.callback_frequency,
        touch_callback_state=args.touch_callback_state,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    maxwell_module._clear_maxwell3d_cuda_graph_cache()
    graph_ms, graph_std = _benchmark(
        case,
        pml_width=args.pml_width,
        experimental_cuda_graph=True,
        callback_frequency=args.callback_frequency,
        touch_callback_state=args.touch_callback_state,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    speedup = plain_ms / graph_ms
    print(
        f"shape=({args.nz},{args.ny},{args.nx}) nt={args.nt} "
        f"shots={args.shots} "
        f"callback_frequency={args.callback_frequency} "
        f"touch_callback_state={args.touch_callback_state}"
    )
    print(f"baseline: {plain_ms:.4f} ms +- {plain_std:.4f}")
    print(f"graph:    {graph_ms:.4f} ms +- {graph_std:.4f}")
    print(f"speedup:  {speedup:.4f}x")


if __name__ == "__main__":
    main()
