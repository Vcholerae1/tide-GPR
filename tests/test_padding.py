import torch

from tide.padding import create_or_pad, reverse_pad, zero_interior


def test_reverse_pad_2d():
    assert reverse_pad([1, 2, 3, 4]) == [3, 4, 1, 2]


def test_create_or_pad_empty_and_constant():
    device = torch.device("cpu")
    dtype = torch.float32
    result = create_or_pad(torch.empty(0), 2, device, dtype, (2, 5, 6))
    assert result.shape == (2, 5, 6)
    assert torch.allclose(result, torch.zeros_like(result))

    base = torch.ones((2, 2), dtype=dtype, device=device)
    padded = create_or_pad(base, [1, 1, 1, 1], device, dtype, (4, 4))
    assert padded.shape == (4, 4)
    assert torch.allclose(padded[1:3, 1:3], base)
    assert padded[0, 0].item() == 0.0


def test_create_or_pad_replicate():
    device = torch.device("cpu")
    dtype = torch.float32
    base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, dtype=dtype)
    padded = create_or_pad(base, [1, 1, 1, 1], device, dtype, (4, 4), mode="replicate")
    expected = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(padded, expected)


def test_zero_interior_y_and_x():
    tensor = torch.ones((1, 6, 6), dtype=torch.float32)
    fd_pad = [1, 1, 1, 1]
    pml_width = [1, 1, 1, 1]

    y_zeroed = zero_interior(tensor.clone(), fd_pad, pml_width, dim=0)
    assert torch.allclose(y_zeroed[:, 2:4, :], torch.zeros((1, 2, 6)))
    assert torch.all(y_zeroed[:, :2, :] == 1)
    assert torch.all(y_zeroed[:, 4:, :] == 1)

    x_zeroed = zero_interior(tensor.clone(), fd_pad, pml_width, dim=1)
    assert torch.allclose(x_zeroed[:, :, 2:4], torch.zeros((1, 6, 2)))
    assert torch.all(x_zeroed[:, :, :2] == 1)
    assert torch.all(x_zeroed[:, :, 4:] == 1)
