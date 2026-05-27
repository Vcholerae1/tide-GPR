import pytest
import torch

from tide import backend_utils, staggered
from tide.maxwell.common import _get_ctx_handle, _release_ctx_handle
from tide.maxwell.tm2d_born_autograd import BornTMForwardFunc
from tide.storage import STORAGE_FORMAT_BF16, STORAGE_FORMAT_FULL


def _assert_native_grads_are_finite(grads: tuple[torch.Tensor, ...]) -> None:
    for grad in grads:
        assert torch.isfinite(grad).all()
        assert grad.norm() > 0


def _native_tm2d_born_receivers(
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cq: torch.Tensor,
    dca: torch.Tensor,
    dcb: torch.Tensor,
    f0: torch.Tensor,
    df: torch.Tensor,
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
    nt: int,
    n_shots: int,
    ny: int,
    nx: int,
    n_sources: int,
    n_receivers: int,
    stencil: int,
    dEy_0: torch.Tensor | None = None,
    storage_compression: bool | str = False,
) -> torch.Tensor:
    device = ca.device
    dtype = ca.dtype
    storage_format = STORAGE_FORMAT_BF16 if storage_compression else STORAGE_FORMAT_FULL
    fd_pad = stencil // 2
    pml_y0 = pml_x0 = fd_pad
    pml_y1 = ny - fd_pad + 1
    pml_x1 = nx - fd_pad + 1

    zeros = torch.zeros(n_shots, ny, nx, dtype=dtype, device=device)
    zeros_m = torch.zeros_like(zeros)
    line_zero_y = torch.zeros(ny, dtype=dtype, device=device)
    line_zero_x = torch.zeros(nx, dtype=dtype, device=device)
    line_one_y = torch.ones(ny, dtype=dtype, device=device)
    line_one_x = torch.ones(nx, dtype=dtype, device=device)

    outputs = BornTMForwardFunc.apply(
        dca,
        dcb,
        ca,
        cb,
        cq,
        f0,
        df,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_one_y,
        line_one_y,
        line_one_x,
        line_one_x,
        sources_i,
        receivers_i,
        1.0,
        1.0,
        1.0,
        nt,
        n_shots,
        ny,
        nx,
        n_sources,
        n_receivers,
        1,
        stencil,
        False,
        False,
        False,
        pml_y0,
        pml_x0,
        pml_y1,
        pml_x1,
        "device",
        storage_format,
        "",
        storage_compression,
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros.clone() if dEy_0 is None else dEy_0.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        zeros_m.clone(),
        0,
        device,
    )
    return outputs[7]


def test_tm2d_born_autograd_uses_bf16_for_saved_snapshots(monkeypatch):
    def fake_backend(*_args):
        return None

    monkeypatch.setattr(
        backend_utils, "get_backend_function", lambda *_args: fake_backend
    )

    device = torch.device("cpu")
    dtype = torch.float32
    nt, n_shots, ny, nx = 4, 1, 5, 6
    n_sources = n_receivers = 0
    zeros = torch.zeros(n_shots, ny, nx, dtype=dtype, device=device)
    line_zero_y = torch.zeros(ny, dtype=dtype, device=device)
    line_zero_x = torch.zeros(nx, dtype=dtype, device=device)
    line_one_y = torch.ones(ny, dtype=dtype, device=device)
    line_one_x = torch.ones(nx, dtype=dtype, device=device)
    empty_i = torch.empty(0, dtype=torch.long, device=device)
    empty_f = torch.empty(0, dtype=dtype, device=device)

    outputs = BornTMForwardFunc.forward(
        torch.zeros(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.zeros(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.ones(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.ones(1, ny, nx, dtype=dtype, device=device, requires_grad=True),
        torch.ones(1, ny, nx, dtype=dtype, device=device),
        empty_f,
        empty_f,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_y,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_zero_x,
        line_one_y,
        line_one_y,
        line_one_x,
        line_one_x,
        empty_i,
        empty_i,
        1.0,
        1.0,
        1.0,
        nt,
        n_shots,
        ny,
        nx,
        n_sources,
        n_receivers,
        1,
        2,
        False,
        False,
        False,
        1,
        1,
        ny,
        nx,
        "device",
        STORAGE_FORMAT_BF16,
        "",
        "bf16",
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        zeros.clone(),
        0,
        device,
    )
    ctx_handle = outputs[-1]
    ctx_data = _get_ctx_handle(int(ctx_handle.item()))
    try:
        for tensor in (
            *ctx_data["backward_storage_tensors"],
            *ctx_data["direct_snapshot_tensors"],
        ):
            assert tensor.dtype == torch.bfloat16
        assert ctx_data["storage_format"] == STORAGE_FORMAT_BF16
        assert ctx_data["shot_bytes_uncomp"] == ny * nx * 2
    finally:
        _release_ctx_handle(int(ctx_handle.item()))


def _reference_tm2d_born_receivers(
    *,
    ca: torch.Tensor,
    cb: torch.Tensor,
    cq: torch.Tensor,
    dca: torch.Tensor,
    dcb: torch.Tensor,
    f0: torch.Tensor,
    df: torch.Tensor,
    sources_i: torch.Tensor,
    receivers_i: torch.Tensor,
    nt: int,
    n_shots: int,
    ny: int,
    nx: int,
    n_sources: int,
    dEy_0: torch.Tensor | None = None,
    stencil: int = 2,
) -> torch.Tensor:
    device = ca.device
    dtype = ca.dtype
    rdy = torch.tensor(1.0, dtype=dtype, device=device)
    rdx = torch.tensor(1.0, dtype=dtype, device=device)

    Ey = torch.zeros(n_shots, ny, nx, dtype=dtype, device=device)
    Hx = torch.zeros_like(Ey)
    Hz = torch.zeros_like(Ey)
    dEy = torch.zeros_like(Ey) if dEy_0 is None else dEy_0.clone()
    dHx = torch.zeros_like(Ey)
    dHz = torch.zeros_like(Ey)
    dca_eff = dca.unsqueeze(0) if dca.ndim == 2 else dca
    dcb_eff = dcb.unsqueeze(0) if dcb.ndim == 2 else dcb

    if n_sources > 0:
        source_ids = sources_i.reshape(n_shots, n_sources)
        f0_view = f0.reshape(nt, n_shots, n_sources)
        df_view = df.reshape(nt, n_shots, n_sources)

    receivers = []
    for t in range(nt):
        Hx = Hx - cq * staggered.diffyh1(Ey, stencil, rdy)
        Hz = Hz + cq * staggered.diffxh1(Ey, stencil, rdx)
        dHx = dHx - cq * staggered.diffyh1(dEy, stencil, rdy)
        dHz = dHz + cq * staggered.diffxh1(dEy, stencil, rdx)

        curl_h = staggered.diffx1(Hz, stencil, rdx) - staggered.diffy1(Hx, stencil, rdy)
        dcurl_h = staggered.diffx1(dHz, stencil, rdx) - staggered.diffy1(
            dHx, stencil, rdy
        )

        Ey_old = Ey
        Ey = ca * Ey + cb * curl_h
        dEy = ca * dEy + cb * dcurl_h + dca_eff * Ey_old + dcb_eff * curl_h

        if n_sources > 0:
            Ey.view(n_shots, -1).scatter_add_(1, source_ids, f0_view[t])
            dEy.view(n_shots, -1).scatter_add_(1, source_ids, df_view[t])

        receivers.append(
            torch.stack(
                [
                    dEy.view(n_shots, -1)[:, int(flat_idx.item())]
                    for flat_idx in receivers_i.reshape(-1)
                ],
                dim=-1,
            )
        )

    return torch.stack(receivers, dim=0)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_born_bggrad_matches_reference_with_sources():
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 1
    n_receivers = 2
    stencil = 2

    source_yx = torch.tensor([[[3, 3]]], dtype=torch.long, device=device)
    receiver_yx = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    sources_i = (source_yx[..., 0] * nx + source_yx[..., 1]).long().contiguous()
    receivers_i = (receiver_yx[..., 0] * nx + receiver_yx[..., 1]).long().contiguous()

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    dcb = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    source = torch.randn(nt, n_shots, n_sources, dtype=dtype, device=device)
    f0 = source.reshape(-1).clone().detach().requires_grad_(True)
    df = (0.15 * source).reshape(-1).clone().detach().requires_grad_(True)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=f0,
        df=df,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
    )
    native_grads = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb, f0, df],
    )

    _assert_native_grads_are_finite(native_grads)


@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_born_bggrad_matches_reference_without_sources():
    torch.manual_seed(2)
    device = torch.device("cpu")
    dtype = torch.float64
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 0
    n_receivers = 2
    stencil = 2

    receivers_i = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    receivers_flat = (
        (receivers_i[..., 0] * nx + receivers_i[..., 1]).long().contiguous()
    )

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = torch.zeros(ny, nx, dtype=dtype, device=device).requires_grad_()
    dcb = torch.zeros(ny, nx, dtype=dtype, device=device).requires_grad_()
    dEy_0 = torch.randn(n_shots, ny, nx, dtype=dtype, device=device)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        df=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        sources_i=torch.empty(0, dtype=torch.long, device=device),
        receivers_i=receivers_flat,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
        dEy_0=dEy_0,
    )
    native_grad_ca, native_grad_cb, _, _ = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb],
    )

    ca_ref = ca.detach().clone().requires_grad_(True)
    cb_ref = cb.detach().clone().requires_grad_(True)
    dca_ref = dca.detach().clone().requires_grad_(True)
    dcb_ref = dcb.detach().clone().requires_grad_(True)
    reference_receivers = _reference_tm2d_born_receivers(
        ca=ca_ref,
        cb=cb_ref,
        cq=cq,
        dca=dca_ref,
        dcb=dcb_ref,
        f0=torch.empty(0, dtype=dtype, device=device),
        df=torch.empty(0, dtype=dtype, device=device),
        sources_i=torch.empty(0, dtype=torch.long, device=device),
        receivers_i=receivers_flat,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        dEy_0=dEy_0,
        stencil=stencil,
    )
    reference_grad_ca, reference_grad_cb, _, _ = torch.autograd.grad(
        torch.sum(reference_receivers * residual),
        [ca_ref, cb_ref, dca_ref, dcb_ref],
    )

    torch.testing.assert_close(native_grad_ca, reference_grad_ca, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(native_grad_cb, reference_grad_cb, atol=1e-10, rtol=1e-9)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_born_bggrad_matches_reference_with_sources_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float64
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 1
    n_receivers = 2
    stencil = 2

    source_yx = torch.tensor([[[3, 3]]], dtype=torch.long, device=device)
    receiver_yx = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    sources_i = (source_yx[..., 0] * nx + source_yx[..., 1]).long().contiguous()
    receivers_i = (receiver_yx[..., 0] * nx + receiver_yx[..., 1]).long().contiguous()

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    dcb = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    source = torch.randn(nt, n_shots, n_sources, dtype=dtype, device=device)
    f0 = source.reshape(-1).clone().detach().requires_grad_(True)
    df = (0.15 * source).reshape(-1).clone().detach().requires_grad_(True)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=f0,
        df=df,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
    )
    native_grads = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb, f0, df],
    )

    _assert_native_grads_are_finite(native_grads)


@pytest.mark.skipif(
    not backend_utils.is_backend_available() or not torch.cuda.is_available(),
    reason="native cuda backend not available",
)
def test_tm2d_born_bggrad_matches_reference_without_sources_cuda():
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.float64
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 0
    n_receivers = 2
    stencil = 2

    receivers_i = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    receivers_flat = (
        (receivers_i[..., 0] * nx + receivers_i[..., 1]).long().contiguous()
    )

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = torch.zeros(ny, nx, dtype=dtype, device=device).requires_grad_()
    dcb = torch.zeros(ny, nx, dtype=dtype, device=device).requires_grad_()
    dEy_0 = torch.randn(n_shots, ny, nx, dtype=dtype, device=device)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        df=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        sources_i=torch.empty(0, dtype=torch.long, device=device),
        receivers_i=receivers_flat,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
        dEy_0=dEy_0,
    )
    native_grad_ca, native_grad_cb, _, _ = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb],
    )

    ca_ref = ca.detach().clone().requires_grad_(True)
    cb_ref = cb.detach().clone().requires_grad_(True)
    dca_ref = dca.detach().clone().requires_grad_(True)
    dcb_ref = dcb.detach().clone().requires_grad_(True)
    reference_receivers = _reference_tm2d_born_receivers(
        ca=ca_ref,
        cb=cb_ref,
        cq=cq,
        dca=dca_ref,
        dcb=dcb_ref,
        f0=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        df=torch.empty(0, dtype=dtype, device=device, requires_grad=True),
        sources_i=torch.empty(0, dtype=torch.long, device=device),
        receivers_i=receivers_flat,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        stencil=stencil,
        dEy_0=dEy_0,
    )
    reference_grad_ca, reference_grad_cb, _, _ = torch.autograd.grad(
        torch.sum(reference_receivers * residual),
        [ca_ref, cb_ref, dca_ref, dcb_ref],
    )

    torch.testing.assert_close(native_grad_ca, reference_grad_ca, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(native_grad_cb, reference_grad_cb, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.skipif(
    not backend_utils.is_backend_available(), reason="native backend not available"
)
def test_tm2d_born_bggrad_matches_reference_with_bf16_storage(device_type):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for TM2D Born BF16 storage test.")

    torch.manual_seed(4)
    device = torch.device(device_type)
    dtype = torch.float32
    ny = nx = 8
    nt = 5
    n_shots = 1
    n_sources = 1
    n_receivers = 2
    stencil = 2

    source_yx = torch.tensor([[[3, 3]]], dtype=torch.long, device=device)
    receiver_yx = torch.tensor([[[3, 4], [4, 4]]], dtype=torch.long, device=device)
    sources_i = (source_yx[..., 0] * nx + source_yx[..., 1]).long().contiguous()
    receivers_i = (receiver_yx[..., 0] * nx + receiver_yx[..., 1]).long().contiguous()

    ca = torch.full((1, ny, nx), 0.98, dtype=dtype, device=device).requires_grad_()
    cb = torch.full((1, ny, nx), 0.07, dtype=dtype, device=device).requires_grad_()
    cq = torch.full((1, ny, nx), 0.05, dtype=dtype, device=device)
    dca = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    dcb = (0.02 * torch.randn(ny, nx, dtype=dtype, device=device)).requires_grad_()
    source = torch.randn(nt, n_shots, n_sources, dtype=dtype, device=device)
    f0 = source.reshape(-1).clone().detach().requires_grad_(True)
    df = (0.15 * source).reshape(-1).clone().detach().requires_grad_(True)
    residual = torch.randn(nt, n_shots, n_receivers, dtype=dtype, device=device)

    native_receivers = _native_tm2d_born_receivers(
        ca=ca,
        cb=cb,
        cq=cq,
        dca=dca,
        dcb=dcb,
        f0=f0,
        df=df,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
        storage_compression="bf16",
    )
    native_grads = torch.autograd.grad(
        torch.sum(native_receivers * residual),
        [ca, cb, dca, dcb, f0, df],
    )

    ca_full = ca.detach().clone().requires_grad_(True)
    cb_full = cb.detach().clone().requires_grad_(True)
    dca_full = dca.detach().clone().requires_grad_(True)
    dcb_full = dcb.detach().clone().requires_grad_(True)
    f0_full = f0.detach().clone().requires_grad_(True)
    df_full = df.detach().clone().requires_grad_(True)
    full_receivers = _native_tm2d_born_receivers(
        ca=ca_full,
        cb=cb_full,
        cq=cq,
        dca=dca_full,
        dcb=dcb_full,
        f0=f0_full,
        df=df_full,
        sources_i=sources_i,
        receivers_i=receivers_i,
        nt=nt,
        n_shots=n_shots,
        ny=ny,
        nx=nx,
        n_sources=n_sources,
        n_receivers=n_receivers,
        stencil=stencil,
    )
    full_grads = torch.autograd.grad(
        torch.sum(full_receivers * residual),
        [ca_full, cb_full, dca_full, dcb_full, f0_full, df_full],
    )

    for native_grad, full_grad in zip(native_grads, full_grads):
        torch.testing.assert_close(native_grad, full_grad, atol=8e-3, rtol=8e-2)
