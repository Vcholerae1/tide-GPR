import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

import tide


TRAINING_CONFIGS = [
    ("baseline", 0, 40, 16, 1),
    ("ebisu_s3_40x16", 3, 40, 16, 1),
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
    model_gradient_sampling_interval = 10

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
        "epsilon_base": epsilon,
        "sigma_base": sigma,
        "mu": mu,
        "dx": dx,
        "dt": dt,
        "pml_width": pml_width,
        "source_amplitude": source_amplitude_full[shot_indices].contiguous(),
        "source_location": source_locations[shot_indices].contiguous(),
        "receiver_location": receiver_locations[shot_indices].contiguous(),
        "model_gradient_sampling_interval": model_gradient_sampling_interval,
    }


def run_training_batch(example_inputs):
    epsilon = example_inputs["epsilon_base"].clone().detach().requires_grad_(True)
    sigma = example_inputs["sigma_base"].clone().detach().requires_grad_(True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = tide.maxwelltm(
        epsilon,
        sigma,
        example_inputs["mu"],
        grid_spacing=example_inputs["dx"],
        dt=example_inputs["dt"],
        source_amplitude=example_inputs["source_amplitude"],
        source_location=example_inputs["source_location"],
        receiver_location=example_inputs["receiver_location"],
        pml_width=example_inputs["pml_width"],
        save_snapshots=True,
        model_gradient_sampling_interval=example_inputs[
            "model_gradient_sampling_interval"
        ],
        compute_precision="default",
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    receivers = outputs[-1]
    loss = receivers.square().mean()

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    return {
        "forward_ms": (t1 - t0) * 1000.0,
        "backward_ms": (t3 - t2) * 1000.0,
        "end_to_end_ms": (t3 - t0) * 1000.0,
        "loss": float(loss.detach().cpu()),
    }


def worker():
    steps = int(os.environ["TIDE_TM_EBISU_STEPS"])
    tile_x = int(os.environ["TIDE_TM_EBISU_TILE_X"])
    tile_y = int(os.environ["TIDE_TM_EBISU_TILE_Y"])
    ilp = int(os.environ["TIDE_TM_EBISU_ILP"])
    repeats = int(os.environ.get("TIDE_TM_BENCH_REPEATS", "5"))
    warmups = int(os.environ.get("TIDE_TM_BENCH_WARMUPS", "1"))

    example_inputs = build_example_inputs()

    os.environ["TIDE_TM_FUSED_STEPS"] = "0"
    os.environ["TIDE_TM_EBISU_STEPS"] = str(steps)
    os.environ["TIDE_TM_EBISU_TILE_X"] = str(tile_x)
    os.environ["TIDE_TM_EBISU_TILE_Y"] = str(tile_y)
    os.environ["TIDE_TM_EBISU_ILP"] = str(ilp)

    for _ in range(warmups):
        run_training_batch(example_inputs)

    samples = []
    for _ in range(repeats):
        samples.append(run_training_batch(example_inputs))

    def summarize(key):
        values = [sample[key] for sample in samples]
        values_sorted = sorted(values)
        return {
            "mean_ms": sum(values) / len(values),
            "median_ms": values_sorted[len(values_sorted) // 2],
            "samples_ms": values,
        }

    result = {
        "steps": steps,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "ilp": ilp,
        "forward": summarize("forward_ms"),
        "backward": summarize("backward_ms"),
        "end_to_end": summarize("end_to_end_ms"),
        "loss": samples[-1]["loss"],
    }
    print(json.dumps(result))


def main():
    if "--worker" in sys.argv:
        worker()
        return

    results = []
    for name, steps, tile_x, tile_y, ilp in TRAINING_CONFIGS:
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
        forward_speedup = baseline["forward"]["mean_ms"] / item["forward"]["mean_ms"]
        backward_speedup = baseline["backward"]["mean_ms"] / item["backward"]["mean_ms"]
        end_to_end_speedup = (
            baseline["end_to_end"]["mean_ms"] / item["end_to_end"]["mean_ms"]
        )
        print(
            f'{item["name"]:>16}  '
            f'forward={item["forward"]["mean_ms"]:.3f} ms ({forward_speedup:.3f}x)  '
            f'backward={item["backward"]["mean_ms"]:.3f} ms ({backward_speedup:.3f}x)  '
            f'end_to_end={item["end_to_end"]["mean_ms"]:.3f} ms ({end_to_end_speedup:.3f}x)'
        )


if __name__ == "__main__":
    main()
