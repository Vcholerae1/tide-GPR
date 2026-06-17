import pytest
import torch

import tide
from tide.workflow import (
    backward_shot_batches,
    block_preconditioner,
    curvature_preconditioner_block,
    curvature_preconditioner_diagonal,
    diagonal_preconditioner,
    expand_source_amplitude,
    index_shots,
    line_acquisition_2d,
    merge_receiver_batches,
    point_acquisition,
    receiver_mse_loss,
    run_shot_batches,
    split_shots,
    take_receiver_batch,
    take_shot_batch,
)


def test_split_shots_uses_long_indices_on_requested_device() -> None:
    batches = split_shots(5, 2, device=torch.device("cpu"))

    assert [batch.tolist() for batch in batches] == [[0, 1], [2, 3], [4]]
    assert all(batch.dtype == torch.long for batch in batches)
    assert all(batch.device.type == "cpu" for batch in batches)


def test_split_shots_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_shots"):
        split_shots(-1, 2)
    with pytest.raises(ValueError, match="batch_size"):
        split_shots(2, 0)


def test_index_shots_handles_shared_and_per_model_shot_axes() -> None:
    shared = torch.arange(4 * 2).reshape(4, 2)
    per_model = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    indices = torch.tensor([1, 3])

    torch.testing.assert_close(index_shots(shared, indices), shared[indices])
    torch.testing.assert_close(
        index_shots(per_model, indices, shot_dim=1),
        per_model[:, indices],
    )


def test_take_shot_batch_preserves_none_and_selects_all_locations() -> None:
    source_location = torch.arange(4 * 1 * 2).reshape(4, 1, 2)
    receiver_location = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
    indices = torch.tensor([0, 2])

    batch = take_shot_batch(
        source_amplitude=None,
        source_location=source_location,
        receiver_location=receiver_location,
        shot_indices=indices,
    )

    assert batch.source_amplitude is None
    torch.testing.assert_close(batch.source_location, source_location[indices])
    torch.testing.assert_close(batch.receiver_location, receiver_location[indices])


def test_point_acquisition_builds_shared_and_paired_receivers() -> None:
    source_points = torch.tensor([[1, 2], [1, 4], [1, 6]])
    shared_receivers = torch.tensor([[2, 3], [2, 5]])
    paired_receivers = torch.tensor([[2, 2], [2, 4], [2, 6]])

    shared = point_acquisition(source_points, shared_receivers, receiver_mode="shared")
    paired = point_acquisition(source_points, paired_receivers, receiver_mode="paired")

    assert shared.source_location.shape == (3, 1, 2)
    assert shared.receiver_location.shape == (3, 2, 2)
    assert paired.receiver_location.shape == (3, 1, 2)
    assert shared.n_shots == 3
    assert shared.n_receivers == 2
    assert shared.spatial_ndim == 2
    torch.testing.assert_close(shared.receiver_location[1], shared_receivers)
    torch.testing.assert_close(paired.receiver_location[:, 0], paired_receivers)


def test_line_acquisition_2d_builds_solver_locations() -> None:
    acquisition = line_acquisition_2d(
        torch.tensor([2, 4, 6]),
        torch.tensor([3, 5, 7]),
        source_depth=1,
        receiver_mode="paired",
    )

    expected_source = torch.tensor([[[1, 2]], [[1, 4]], [[1, 6]]])
    expected_receiver = torch.tensor([[[1, 3]], [[1, 5]], [[1, 7]]])
    torch.testing.assert_close(acquisition.source_location, expected_source)
    torch.testing.assert_close(acquisition.receiver_location, expected_receiver)


def test_expand_source_amplitude_handles_single_and_multi_source_wavelets() -> None:
    wavelet = torch.arange(4, dtype=torch.float32)
    multi_source = torch.stack([wavelet, wavelet + 1])

    single = expand_source_amplitude(wavelet, 3)
    multi = expand_source_amplitude(multi_source, 3, n_sources=2)

    assert single.shape == (3, 1, 4)
    assert multi.shape == (3, 2, 4)
    torch.testing.assert_close(single[2, 0], wavelet)
    torch.testing.assert_close(multi[1], multi_source)


def test_merge_receiver_batches_infers_tide_receiver_shot_axis() -> None:
    shared_chunks = [
        torch.full((3, 2, 1), 1.0),
        torch.full((3, 1, 1), 2.0),
    ]
    batched_model_chunks = [
        torch.full((3, 2, 2, 1), 1.0),
        torch.full((3, 2, 1, 1), 2.0),
    ]

    shared = merge_receiver_batches(shared_chunks)
    batched_model = merge_receiver_batches(batched_model_chunks)

    assert shared.shape == (3, 3, 1)
    assert batched_model.shape == (3, 2, 3, 1)
    torch.testing.assert_close(shared[:, :2], shared_chunks[0])
    torch.testing.assert_close(batched_model[:, :, :2], batched_model_chunks[0])


def test_receiver_batch_helpers_select_and_normalize_loss() -> None:
    observed = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    indices = torch.tensor([0, 2])
    predicted = observed[:, indices, :] + 1.0

    selected = take_receiver_batch(observed, indices)
    batch_loss = receiver_mse_loss(predicted, observed, indices)
    full_loss = receiver_mse_loss(
        predicted,
        observed,
        indices,
        normalization="all",
    )

    torch.testing.assert_close(selected, observed[:, indices, :])
    torch.testing.assert_close(batch_loss, torch.tensor(1.0))
    torch.testing.assert_close(full_loss, torch.tensor(predicted.numel() / observed.numel()))


def test_backward_shot_batches_accumulates_full_gradient() -> None:
    x = torch.arange(1, 6, dtype=torch.float32)
    observed = 2.5 * x
    shot_batches = split_shots(x.numel(), 2)
    scale = torch.tensor(0.25, requires_grad=True)

    def clear_grad() -> None:
        scale.grad = None

    def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
        residual = scale * x[shot_indices] - observed[shot_indices]
        return residual.square().sum() / observed.numel()

    total_loss = backward_shot_batches(
        batch_loss,
        shot_batches,
        zero_grad=clear_grad,
    )

    expected_scale = torch.tensor(0.25, requires_grad=True)
    expected_loss = ((expected_scale * x - observed).square()).sum() / observed.numel()
    expected_loss.backward()

    torch.testing.assert_close(torch.tensor(total_loss), expected_loss.detach())
    assert scale.grad is not None
    assert expected_scale.grad is not None
    torch.testing.assert_close(scale.grad, expected_scale.grad)


def test_backward_shot_batches_can_inspect_per_batch_gradients() -> None:
    x = torch.arange(1, 5, dtype=torch.float32)
    observed = 3.0 * x
    shot_batches = split_shots(x.numel(), 2)
    scale = torch.tensor(1.0, requires_grad=True)
    per_batch_grads: list[torch.Tensor] = []

    def clear_grad() -> None:
        scale.grad = None

    def batch_loss(shot_indices: torch.Tensor) -> torch.Tensor:
        residual = scale * x[shot_indices] - observed[shot_indices]
        return residual.square().sum()

    def record_grad(_shot_indices: torch.Tensor, _loss: torch.Tensor) -> None:
        assert scale.grad is not None
        per_batch_grads.append(scale.grad.detach().clone())

    total_loss = backward_shot_batches(
        batch_loss,
        shot_batches,
        zero_grad=clear_grad,
        zero_each_batch=True,
        after_backward=record_grad,
    )

    expected_grads = []
    expected_loss = 0.0
    for shot_indices in shot_batches:
        residual = scale.detach() * x[shot_indices] - observed[shot_indices]
        expected_loss += float(residual.square().sum())
        expected_grads.append((2.0 * residual * x[shot_indices]).sum())

    assert len(per_batch_grads) == len(expected_grads)
    torch.testing.assert_close(torch.stack(per_batch_grads), torch.stack(expected_grads))
    torch.testing.assert_close(torch.tensor(total_loss), torch.tensor(expected_loss))


def test_curvature_preconditioner_diagonal_normalizes_clips_and_masks() -> None:
    curvature = torch.tensor(
        [
            [0.0, 1.0, 4.0],
            [float("nan"), float("inf"), 9.0],
        ],
        dtype=torch.float32,
    )
    inactive = torch.tensor(
        [
            [False, False, False],
            [True, True, False],
        ]
    )

    diagonal = curvature_preconditioner_diagonal(
        curvature,
        inactive_mask=inactive,
        damping=0.1,
        power=0.5,
        clip_min=0.5,
        clip_max=2.0,
        blend=0.75,
    )

    assert diagonal.shape == curvature.shape
    assert diagonal.dtype == curvature.dtype
    assert torch.all(torch.isfinite(diagonal))
    assert torch.all(diagonal[~inactive] >= 0.5)
    assert torch.all(diagonal[~inactive] <= 2.0)
    torch.testing.assert_close(diagonal[inactive], torch.zeros_like(diagonal[inactive]))


def test_diagonal_preconditioner_matches_tide_optim_callback_contract() -> None:
    diagonal = torch.tensor([2.0, 0.5], dtype=torch.float32)
    preconditioner = diagonal_preconditioner(diagonal)
    x = torch.zeros(2).numpy()
    vector = torch.tensor([3.0, 4.0], dtype=torch.float32).numpy()
    out = torch.empty(2, dtype=torch.float32).numpy()

    preconditioner(x, vector, out)

    torch.testing.assert_close(torch.from_numpy(out), torch.tensor([6.0, 2.0]))


def test_curvature_preconditioner_block_normalizes_clips_and_masks() -> None:
    curvature_11 = torch.tensor(
        [[[1.0, 4.0], [float("nan"), 9.0]]],
        dtype=torch.float32,
    )
    curvature_22 = torch.tensor([[[2.0, 8.0], [3.0, float("inf")]]])
    curvature_12 = torch.tensor([[[0.25, -0.5], [1.0, 2.0]]])
    inactive = torch.tensor([[[False, False], [True, False]]])

    block = curvature_preconditioner_block(
        curvature_11,
        curvature_22,
        curvature_12,
        inactive_mask=inactive,
        damping=0.1,
        power=0.5,
        clip_min=0.25,
        clip_max=4.0,
        blend=0.75,
    )

    assert block.diag11.shape == curvature_11.shape
    assert block.offdiag12.shape == curvature_11.shape
    assert block.diag22.shape == curvature_11.shape
    assert torch.all(torch.isfinite(block.diag11))
    assert torch.all(torch.isfinite(block.offdiag12))
    assert torch.all(torch.isfinite(block.diag22))
    assert torch.all(block.diag11[~inactive] >= 0.25)
    assert torch.all(block.diag22[~inactive] >= 0.25)
    assert torch.all(block.diag11[~inactive] <= 4.0)
    assert torch.all(block.diag22[~inactive] <= 4.0)
    torch.testing.assert_close(block.diag11[inactive], torch.zeros(1))
    torch.testing.assert_close(block.offdiag12[inactive], torch.zeros(1))
    torch.testing.assert_close(block.diag22[inactive], torch.zeros(1))


def test_block_preconditioner_matches_tide_optim_callback_contract() -> None:
    block = tide.workflow.BlockPreconditioner(
        diag11=torch.tensor([2.0, 3.0]),
        offdiag12=torch.tensor([0.5, -1.0]),
        diag22=torch.tensor([4.0, 5.0]),
    )
    preconditioner = block_preconditioner(block)
    x = torch.zeros(4).numpy()
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    out = torch.empty(4, dtype=torch.float32).numpy()

    preconditioner(x, vector, out)

    torch.testing.assert_close(
        torch.from_numpy(out),
        torch.tensor([3.5, 2.0, 12.5, 18.0]),
    )


def test_run_shot_batches_preserves_autograd() -> None:
    source_amplitude = torch.arange(4 * 1 * 3, dtype=torch.float32).reshape(4, 1, 3)
    source_location = torch.zeros(4, 1, 2, dtype=torch.long)
    receiver_location = torch.zeros(4, 1, 2, dtype=torch.long)
    weight = torch.tensor(2.0, requires_grad=True)

    def solver(
        *,
        source_amplitude: torch.Tensor,
        source_location: torch.Tensor,
        receiver_location: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        assert source_location.shape[0] == receiver_location.shape[0]
        return source_amplitude[:, 0, :].transpose(0, 1).unsqueeze(-1) * weight

    receiver = run_shot_batches(
        solver,
        n_shots=4,
        batch_size=2,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        weight=weight,
    )
    loss = receiver.square().sum()
    loss.backward()

    assert receiver.shape == (3, 4, 1)
    assert weight.grad is not None
    assert float(weight.grad) > 0.0


def _tm_case() -> dict[str, torch.Tensor | float | int | bool]:
    dtype = torch.float32
    ny, nx = 7, 8
    nt = 6
    dt = 4e-11
    epsilon = torch.full((ny, nx), 4.0, dtype=dtype)
    sigma = torch.full_like(epsilon, 1e-3)
    mu = torch.ones_like(epsilon)
    source_amplitude = tide.ricker(80e6, nt, dt, dtype=dtype).view(1, 1, nt).repeat(3, 1, 1)
    source_location = torch.tensor(
        [[[3, 2]], [[3, 3]], [[3, 4]]],
        dtype=torch.long,
    )
    receiver_location = torch.tensor(
        [[[3, 5]], [[3, 5]], [[3, 5]]],
        dtype=torch.long,
    )
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "grid_spacing": 0.02,
        "dt": dt,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "pml_width": 1,
        "stencil": 2,
        "python_backend": True,
    }


def test_run_shot_batches_matches_full_maxwelltm_python_call() -> None:
    case = _tm_case()

    full = tide.maxwelltm(**case)[-1]
    batched = run_shot_batches(
        tide.maxwelltm,
        n_shots=3,
        batch_size=2,
        **case,
    )

    torch.testing.assert_close(batched, full)


def test_run_shot_batches_preserves_batched_model_receiver_shape() -> None:
    case = _tm_case()
    epsilon = case["epsilon"]
    sigma = case["sigma"]
    mu = case["mu"]
    assert isinstance(epsilon, torch.Tensor)
    assert isinstance(sigma, torch.Tensor)
    assert isinstance(mu, torch.Tensor)
    case["epsilon"] = torch.stack([epsilon, epsilon * 1.1])
    case["sigma"] = torch.stack([sigma, sigma * 1.2])
    case["mu"] = torch.stack([mu, mu])

    full = tide.maxwelltm(**case)[-1]
    batched = run_shot_batches(
        tide.maxwelltm,
        n_shots=3,
        batch_size=1,
        **case,
    )

    assert batched.shape == (6, 2, 3, 1)
    torch.testing.assert_close(batched, full)


def _em3d_case() -> dict[str, torch.Tensor | float | int | str | bool]:
    dtype = torch.float32
    nz, ny, nx = 5, 6, 7
    nt = 5
    dt = 4e-11
    epsilon = torch.full((nz, ny, nx), 4.0, dtype=dtype)
    sigma = torch.full_like(epsilon, 1e-3)
    mu = torch.ones_like(epsilon)
    source_amplitude = tide.ricker(70e6, nt, dt, dtype=dtype).view(1, 1, nt).repeat(3, 1, 1)
    source_location = torch.tensor(
        [[[2, 2, 2]], [[2, 3, 2]], [[2, 4, 2]]],
        dtype=torch.long,
    )
    receiver_location = torch.tensor(
        [[[2, 2, 4]], [[2, 3, 4]], [[2, 4, 4]]],
        dtype=torch.long,
    )
    return {
        "epsilon": epsilon,
        "sigma": sigma,
        "mu": mu,
        "grid_spacing": 0.02,
        "dt": dt,
        "source_amplitude": source_amplitude,
        "source_location": source_location,
        "receiver_location": receiver_location,
        "pml_width": 1,
        "source_component": "ey",
        "receiver_component": "ey",
        "python_backend": True,
    }


def test_run_shot_batches_matches_full_maxwell3d_python_call() -> None:
    case = _em3d_case()

    full = tide.maxwell3d(**case)[-1]
    batched = run_shot_batches(
        tide.maxwell3d,
        n_shots=3,
        batch_size=2,
        **case,
    )

    torch.testing.assert_close(batched, full)
