import pytest
import torch

import tide
from tide import backend_utils


def _skip_if_no_backend() -> None:
    if not backend_utils.is_backend_available():
        pytest.skip("native backend unavailable")


def _tm_case(device: torch.device) -> tuple[torch.Tensor, ...]:
    dtype = torch.float32
    B, S = 2, 2
    ny, nx = 8, 9
    nt = 10
    epsilon = torch.stack(
        [
            torch.full((ny, nx), 4.0 + 0.5 * i, device=device, dtype=dtype)
            for i in range(B)
        ],
        dim=0,
    )
    sigma = torch.stack(
        [
            torch.full((ny, nx), 1e-3 * (i + 1), device=device, dtype=dtype)
            for i in range(B)
        ],
        dim=0,
    )
    mu = torch.ones_like(epsilon)
    wavelet = tide.ricker(80e6, nt, 4e-11, dtype=dtype, device=device)
    source_amplitude = wavelet.view(1, 1, nt).repeat(S, 1, 1)
    source_location = torch.tensor(
        [[[ny // 2, nx // 2]], [[ny // 2, nx // 2 + 1]]],
        dtype=torch.long,
        device=device,
    )
    receiver_location = torch.tensor(
        [[[ny // 2, nx // 2 + 2]], [[ny // 2, nx // 2 + 3]]],
        dtype=torch.long,
        device=device,
    )
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def _em3d_case(device: torch.device) -> tuple[torch.Tensor, ...]:
    dtype = torch.float32
    B, S = 2, 2
    nz, ny, nx = 5, 6, 7
    nt = 8
    epsilon = torch.stack(
        [
            torch.full((nz, ny, nx), 4.0 + 0.5 * i, device=device, dtype=dtype)
            for i in range(B)
        ],
        dim=0,
    )
    sigma = torch.stack(
        [
            torch.full((nz, ny, nx), 1e-3 * (i + 1), device=device, dtype=dtype)
            for i in range(B)
        ],
        dim=0,
    )
    mu = torch.ones_like(epsilon)
    wavelet = tide.ricker(70e6, nt, 4e-11, dtype=dtype, device=device)
    source_amplitude = wavelet.view(1, 1, nt).repeat(S, 1, 1)
    source_location = torch.tensor(
        [[[2, 2, 2]], [[2, 3, 2]]],
        dtype=torch.long,
        device=device,
    )
    receiver_location = torch.tensor(
        [[[2, 2, 4]], [[2, 3, 4]]],
        dtype=torch.long,
        device=device,
    )
    return epsilon, sigma, mu, source_amplitude, source_location, receiver_location


def test_maxwelltm_batched_models_shared_shots_forward_matches_loop():
    _skip_if_no_backend()
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case(
        device
    )

    out = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        stencil=2,
        python_backend=False,
    )

    expected = []
    for i in range(epsilon.shape[0]):
        expected.append(
            tide.maxwelltm(
                epsilon[i],
                sigma[i],
                mu[i],
                grid_spacing=0.02,
                dt=4e-11,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                pml_width=2,
                stencil=2,
                python_backend=False,
            )[-1]
        )
    expected_receiver = torch.stack(expected, dim=1)

    assert out[-1].shape == (
        source_amplitude.shape[-1],
        epsilon.shape[0],
        source_amplitude.shape[0],
        receiver_location.shape[1],
    )
    assert out[0].shape[:2] == (epsilon.shape[0], source_amplitude.shape[0])
    torch.testing.assert_close(out[-1], expected_receiver)


def test_maxwelltm_batched_models_per_model_shots_backward_matches_loop():
    _skip_if_no_backend()
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case(
        device
    )
    epsilon = epsilon.clone().requires_grad_(True)
    sigma = sigma.clone().requires_grad_(True)
    source_amplitude = source_amplitude.unsqueeze(0).expand(2, -1, -1, -1).clone()
    source_amplitude[1] *= 0.75
    source_location = source_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    receiver_location = receiver_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    receiver_location[1, :, 0, 1] -= 1

    receivers = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        stencil=2,
        python_backend=False,
    )[-1]
    receivers.square().sum().backward()
    assert epsilon.grad is not None

    grads_loop = []
    for i in range(epsilon.shape[0]):
        eps_i = epsilon.detach()[i].clone().requires_grad_(True)
        sig_i = sigma.detach()[i].clone().requires_grad_(True)
        rec_i = tide.maxwelltm(
            eps_i,
            sig_i,
            mu[i],
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude[i],
            source_location=source_location[i],
            receiver_location=receiver_location[i],
            pml_width=2,
            stencil=2,
            python_backend=False,
        )[-1]
        rec_i.square().sum().backward()
        assert eps_i.grad is not None
        grads_loop.append(eps_i.grad)

    expected_grad = torch.stack(grads_loop, dim=0)
    torch.testing.assert_close(
        epsilon.grad,
        expected_grad,
        atol=2e-5,
        rtol=1e-5,
        equal_nan=True,
    )


def test_maxwelltm_batched_model_callbacks_expose_structured_shapes():
    _skip_if_no_backend()
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case(
        device
    )
    epsilon = epsilon.clone().requires_grad_(True)
    sigma = sigma.clone().requires_grad_(True)
    seen: dict[str, tuple[int, ...]] = {}

    def forward_cb(state: tide.CallbackState) -> None:
        if "forward_wavefield" not in seen:
            seen["forward_wavefield"] = tuple(state.get_wavefield("Ey").shape)
            seen["forward_model"] = tuple(state.get_model("epsilon").shape)

    def backward_cb(state: tide.CallbackState) -> None:
        if "backward_gradient" not in seen:
            seen["backward_gradient"] = tuple(state.get_gradient("epsilon").shape)

    receivers = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        stencil=2,
        python_backend=False,
        forward_callback=forward_cb,
        backward_callback=backward_cb,
    )[-1]
    receivers.square().sum().backward()

    assert seen["forward_wavefield"] == (2, 2, 8, 9)
    assert seen["forward_model"] == (2, 8, 9)
    assert seen["backward_gradient"] == (2, 8, 9)


def test_maxwell3d_batched_models_shared_shots_forward_matches_loop():
    _skip_if_no_backend()
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _em3d_case(
        device
    )

    out = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        source_component="ey",
        receiver_component="ey",
        python_backend=False,
    )

    expected = []
    for i in range(epsilon.shape[0]):
        expected.append(
            tide.maxwell3d(
                epsilon[i],
                sigma[i],
                mu[i],
                grid_spacing=0.02,
                dt=4e-11,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                pml_width=1,
                source_component="ey",
                receiver_component="ey",
                python_backend=False,
            )[-1]
        )
    expected_receiver = torch.stack(expected, dim=1)

    assert out[-1].shape == (
        source_amplitude.shape[-1],
        epsilon.shape[0],
        source_amplitude.shape[0],
        receiver_location.shape[1],
    )
    assert out[0].shape[:2] == (epsilon.shape[0], source_amplitude.shape[0])
    torch.testing.assert_close(out[-1], expected_receiver)


def test_maxwell3d_batched_models_shared_shots_backward_matches_loop():
    _skip_if_no_backend()
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _em3d_case(
        device
    )
    epsilon = epsilon.clone().requires_grad_(True)
    sigma = sigma.clone().requires_grad_(True)

    receivers = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        source_component="ey",
        receiver_component="ey",
        python_backend=False,
    )[-1]
    receivers.square().sum().backward()
    assert epsilon.grad is not None

    grads_loop = []
    for i in range(epsilon.shape[0]):
        eps_i = epsilon.detach()[i].clone().requires_grad_(True)
        sig_i = sigma.detach()[i].clone().requires_grad_(True)
        rec_i = tide.maxwell3d(
            eps_i,
            sig_i,
            mu[i],
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=1,
            source_component="ey",
            receiver_component="ey",
            python_backend=False,
        )[-1]
        rec_i.square().sum().backward()
        assert eps_i.grad is not None
        grads_loop.append(eps_i.grad)

    expected_grad = torch.stack(grads_loop, dim=0)
    torch.testing.assert_close(epsilon.grad, expected_grad)


def test_maxwelltm_batched_models_python_backend_forward_matches_loop():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case(
        device
    )

    out = tide.maxwelltm(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=2,
        stencil=2,
        python_backend=True,
    )

    expected = []
    for i in range(epsilon.shape[0]):
        expected.append(
            tide.maxwelltm(
                epsilon[i],
                sigma[i],
                mu[i],
                grid_spacing=0.02,
                dt=4e-11,
                source_amplitude=source_amplitude,
                source_location=source_location,
                receiver_location=receiver_location,
                pml_width=2,
                stencil=2,
                python_backend=True,
            )[-1]
        )
    expected_receiver = torch.stack(expected, dim=1)

    assert out[-1].shape == (
        source_amplitude.shape[-1],
        epsilon.shape[0],
        source_amplitude.shape[0],
        receiver_location.shape[1],
    )
    assert out[0].shape[:2] == (epsilon.shape[0], source_amplitude.shape[0])
    torch.testing.assert_close(out[-1], expected_receiver)


def test_maxwell3d_batched_models_python_backend_backward_matches_loop():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _em3d_case(
        device
    )
    epsilon = epsilon.clone().requires_grad_(True)
    sigma = sigma.clone().requires_grad_(True)

    receivers = tide.maxwell3d(
        epsilon,
        sigma,
        mu,
        grid_spacing=0.02,
        dt=4e-11,
        source_amplitude=source_amplitude,
        source_location=source_location,
        receiver_location=receiver_location,
        pml_width=1,
        source_component="ey",
        receiver_component="ey",
        python_backend=True,
    )[-1]
    receivers.square().sum().backward()
    assert epsilon.grad is not None

    grads_loop = []
    for i in range(epsilon.shape[0]):
        eps_i = epsilon.detach()[i].clone().requires_grad_(True)
        sig_i = sigma.detach()[i].clone().requires_grad_(True)
        rec_i = tide.maxwell3d(
            eps_i,
            sig_i,
            mu[i],
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=1,
            source_component="ey",
            receiver_component="ey",
            python_backend=True,
        )[-1]
        rec_i.square().sum().backward()
        assert eps_i.grad is not None
        grads_loop.append(eps_i.grad)

    expected_grad = torch.stack(grads_loop, dim=0)
    torch.testing.assert_close(epsilon.grad, expected_grad)


def test_batched_models_python_backend_callbacks_rejected():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case(
        device
    )

    with pytest.raises(NotImplementedError):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=2,
            stencil=2,
            python_backend=True,
            forward_callback=lambda state: None,
        )

    epsilon3, sigma3, mu3, source_amplitude3, source_location3, receiver_location3 = (
        _em3d_case(device)
    )
    with pytest.raises(NotImplementedError):
        tide.maxwell3d(
            epsilon3,
            sigma3,
            mu3,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude3,
            source_location=source_location3,
            receiver_location=receiver_location3,
            pml_width=1,
            source_component="ey",
            receiver_component="ey",
            python_backend=True,
            forward_callback=lambda state: None,
        )


def test_batched_models_validate_B_and_S_mismatch():
    device = torch.device("cpu")
    epsilon, sigma, mu, source_amplitude, source_location, receiver_location = _tm_case(
        device
    )
    bad_source = source_amplitude.unsqueeze(0).expand(3, -1, -1, -1).clone()

    with pytest.raises(RuntimeError):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=bad_source,
            source_location=source_location,
            receiver_location=receiver_location,
            pml_width=2,
            stencil=2,
            python_backend=False,
        )

    bad_receiver = receiver_location.unsqueeze(0).expand(2, -1, -1, -1).clone()
    bad_receiver = bad_receiver[:, :1]
    with pytest.raises(RuntimeError):
        tide.maxwelltm(
            epsilon,
            sigma,
            mu,
            grid_spacing=0.02,
            dt=4e-11,
            source_amplitude=source_amplitude,
            source_location=source_location,
            receiver_location=bad_receiver,
            pml_width=2,
            stencil=2,
            python_backend=False,
        )
