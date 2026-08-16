from __future__ import annotations

import os
import pytest
import queue
import socket
import subprocess
import sys
import tide
import torch
import torch.multiprocessing as mp
from numerical_utils import cosine_similarity, relative_l2, require_native_backend
from pathlib import Path

# --- test_openmp_parallelism.py ---


@pytest.mark.slow
@pytest.mark.openmp
@pytest.mark.numerical
def test_openmp_thread_counts_preserve_traces_and_gradients(tmp_path: Path) -> None:
    require_native_backend()
    worker = Path(__file__).parent.parent / "helpers" / "openmp_worker.py"
    results: dict[int, dict[str, object]] = {}
    for thread_count in (1, 4):
        output = tmp_path / f"openmp-{thread_count}.pt"
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": str(thread_count),
                "OMP_DYNAMIC": "FALSE",
            }
        )
        subprocess.run(
            [sys.executable, str(worker), "--output", str(output)],
            check=True,
            env=environment,
            cwd=Path(__file__).parents[2],
            capture_output=True,
            text=True,
        )
        results[thread_count] = torch.load(output, weights_only=True)

    reference = results[1]
    actual = results[4]
    assert relative_l2(actual["receiver"], reference["receiver"]) < 1.0e-7
    for name in ("epsilon_gradient", "sigma_gradient"):
        assert relative_l2(actual[name], reference[name]) < 2.0e-6
        assert cosine_similarity(actual[name], reference[name]) > 0.999999
    assert float(actual["elapsed_seconds"]) > 0.0


# --- test_workflow_distributed.py ---


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: mp.Queue,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    context = tide.workflow.init_distributed(
        enabled=True,
        requested_device="cpu",
        backend="gloo",
    )
    try:
        local_indices = tide.workflow.rank_shot_indices(5, context=context)
        local_receiver = (
            local_indices.to(dtype=torch.float32).view(1, -1, 1).expand(2, -1, 1)
        )
        gathered = tide.workflow.gather_receiver_shards(
            local_receiver,
            local_indices,
            5,
            context=context,
        )

        value = torch.tensor(float(rank + 1), requires_grad=True)
        (value * float(rank + 1)).backward()
        tide.workflow.all_reduce_gradients([value], context=context)
        reduced_loss = tide.workflow.all_reduce_float(
            float(rank + 1),
            device=context.device,
            context=context,
        )

        result: dict[str, object] = {
            "rank": rank,
            "indices": local_indices.tolist(),
            "grad": float(value.grad.item()),
            "loss": reduced_loss,
            "gathered": None,
        }
        if rank == 0:
            if gathered is None:
                raise RuntimeError("rank 0 did not receive gathered receivers")
            result["gathered"] = gathered.squeeze(-1).tolist()
        result_queue.put(result)
    except BaseException as exc:
        result_queue.put({"rank": rank, "error": repr(exc)})
        raise
    finally:
        tide.workflow.destroy_distributed(context)


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="torch.distributed is unavailable",
)
def test_distributed_helpers_reduce_gradients_and_gather_receiver_shards() -> None:
    world_size = 2
    port = _free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_distributed_worker,
            args=(rank, world_size, port, result_queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            pytest.fail("distributed helper worker timed out")

    results = []
    for _ in range(world_size):
        try:
            results.append(result_queue.get(timeout=5))
        except queue.Empty as exc:
            raise AssertionError(
                "distributed helper worker did not report a result"
            ) from exc

    errors = [result for result in results if "error" in result]
    assert errors == []
    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(result["indices"] for result in results) == [[0, 2, 4], [1, 3]]
    assert {result["grad"] for result in results} == {3.0}
    assert {result["loss"] for result in results} == {3.0}
    root = next(result for result in results if result["rank"] == 0)
    assert root["gathered"] == [[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]]
