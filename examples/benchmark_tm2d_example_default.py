import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

import tide


EXAMPLE_CONFIGS = [
    ("baseline", 0, 64, 16, 1),
    ("s3_40x16", 3, 40, 16, 1),
    ("s3_48x16", 3, 48, 16, 1),
    ("s3_56x16", 3, 56, 16, 1),
    ("s3_64x16", 3, 64, 16, 1),
    ("s2_48x16", 2, 48, 16, 1),
]


def build_example_inputs():
    device = torch.device("cuda")
    dtype = torch.float32

    dx = 0.02
    dt = 4e-11
    nt = 1500
    pml_width = 10
    air_layer = 3
    n_shots = 100
    d_source = 4
    first_source = 0
    batch_size = 8
    base_forward_freq = 600e6

    epsilon_true_raw = np.load("examples/data/OverThrust.npy")
    epsilon_true_np = epsilon_true_raw.copy()
    epsilon_true_np[:air_layer, :] = 1.0
    sigma_true_np = np.clip(
        1e-4 * np.power(np.maximum(epsilon_true_np - 1.0, 0.0), 2.0),
        0.0,
        0.005,
    )
    sigma_true_np[:air_layer, :] = 0.0

    epsilon = torch.tensor(epsilon_true_np, dtype=dtype, device=device)
    sigma = torch.tensor(sigma_true_np, dtype=dtype, device=device)
    mu = torch.ones_like(epsilon)

    source_depth = air_layer - 1
    source_x = torch.arange(n_shots, device=device) * d_source + first_source
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = source_depth
    source_locations[:, 0, 1] = source_x

    receiver_locations = torch.zeros(
        n_shots, 1, 2, dtype=torch.long, device=device
    )
    receiver_locations[:, 0, 0] = source_depth
    receiver_locations[:, 0, 1] = source_x + 1

    wavelet = tide.ricker(
        base_forward_freq, nt, dt, peak_time=1.0 / base_forward_freq
    ).to(device=device, dtype=dtype)
    source_amplitude_full = wavelet.view(1, 1, nt).repeat(n_shots, 1, 1).contiguous()
    shot_indices = torch.arange(batch_size, device=device)

    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "dx": dx,
        "dt": dt,
        "pml_width": pml_width,
        "source_amplitude": source_amplitude_full[shot_indices].contiguous(),
        "source_location": source_locations[shot_indices].contiguous(),
        "receiver_location": receiver_locations[shot_indices].contiguous(),
    }


def run_forward(example_inputs):
    with torch.no_grad():
        tide.maxwelltm(
            example_inputs["epsilon"],
            example_inputs["sigma"],
            example_inputs["mu"],
            grid_spacing=example_inputs["dx"],
            dt=example_inputs["dt"],
            source_amplitude=example_inputs["source_amplitude"],
            source_location=example_inputs["source_location"],
            receiver_location=example_inputs["receiver_location"],
            pml_width=example_inputs["pml_width"],
            save_snapshots=False,
            model_gradient_sampling_interval=1,
            compute_precision="default",
        )


def worker():
    steps = int(os.environ["TIDE_TM_EBISU_STEPS"])
    tile_x = int(os.environ["TIDE_TM_EBISU_TILE_X"])
    tile_y = int(os.environ["TIDE_TM_EBISU_TILE_Y"])
    ilp = int(os.environ["TIDE_TM_EBISU_ILP"])
    repeats = int(os.environ.get("TIDE_TM_BENCH_REPEATS", "8"))
    warmups = int(os.environ.get("TIDE_TM_BENCH_WARMUPS", "2"))

    example_inputs = build_example_inputs()

    os.environ["TIDE_TM_FUSED_STEPS"] = "0"
    os.environ["TIDE_TM_EBISU_STEPS"] = str(steps)
    os.environ["TIDE_TM_EBISU_TILE_X"] = str(tile_x)
    os.environ["TIDE_TM_EBISU_TILE_Y"] = str(tile_y)
    os.environ["TIDE_TM_EBISU_ILP"] = str(ilp)

    for _ in range(warmups):
        run_forward(example_inputs)
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_forward(example_inputs)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    result = {
        "steps": steps,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "ilp": ilp,
        "times_ms": times_ms,
        "mean_ms": sum(times_ms) / len(times_ms),
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
    }
    print(json.dumps(result))


def main():
    if "--worker" in sys.argv:
        worker()
        return

    results = []
    for name, steps, tile_x, tile_y, ilp in EXAMPLE_CONFIGS:
        env = os.environ.copy()
        env["TIDE_TM_EBISU_STEPS"] = str(steps)
        env["TIDE_TM_EBISU_TILE_X"] = str(tile_x)
        env["TIDE_TM_EBISU_TILE_Y"] = str(tile_y)
        env["TIDE_TM_EBISU_ILP"] = str(ilp)
        proc = subprocess.run(
            [sys.executable, __file__, "--worker"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        result["name"] = name
        results.append(result)

    baseline = next(item for item in results if item["steps"] == 0)
    for item in results:
        speedup = baseline["mean_ms"] / item["mean_ms"]
        print(
            f'{item["name"]:>12}  mean={item["mean_ms"]:.3f} ms  '
            f'median={item["median_ms"]:.3f} ms  speedup={speedup:.3f}x'
        )


if __name__ == "__main__":
    main()
