import torch

from tide.callbacks import CallbackState, create_callback_state


def test_callback_state_views():
    ey = torch.arange(36, dtype=torch.float32).reshape(6, 6)
    models = {"epsilon": torch.ones_like(ey)}
    state = CallbackState(
        dt=0.1,
        step=2,
        nt=5,
        wavefields={"Ey": ey},
        models=models,
        fd_pad=[1, 1, 1, 1],
        pml_width=[1, 1, 1, 1],
    )

    torch.testing.assert_close(state.get_wavefield("Ey", view="full"), ey)
    torch.testing.assert_close(state.get_wavefield("Ey", view="pml"), ey[1:-1, 1:-1])
    torch.testing.assert_close(state.get_wavefield("Ey", view="inner"), ey[2:-2, 2:-2])
    torch.testing.assert_close(
        state.get_model("epsilon", view="inner"),
        models["epsilon"][2:-2, 2:-2],
    )


def test_create_callback_state_factory():
    ey = torch.zeros((2, 3), dtype=torch.float32)
    models = {"epsilon": torch.ones_like(ey)}
    gradients = {"epsilon": torch.full_like(ey, 2.0)}
    state = create_callback_state(
        dt=0.2,
        step=3,
        nt=10,
        wavefields={"Ey": ey},
        models=models,
        gradients=gradients,
        fd_pad=[1, 1, 1, 1],
        pml_width=[2, 2, 2, 2],
        is_backward=True,
        grid_spacing=[0.1, 0.1],
    )

    assert state.dt == 0.2
    assert state.step == 3
    assert state.nt == 10
    assert state.is_backward is True
    assert state.wavefield_names == ["Ey"]
    assert state.model_names == ["epsilon"]
    assert state.gradient_names == ["epsilon"]
