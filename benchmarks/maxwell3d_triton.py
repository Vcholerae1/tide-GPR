"""Compare flat fourth-order Triton Maxwell-3D kernels with native CUDA.

The public CUDA path is invoked once (outside timing) while its prepared low-level
arguments are captured.  Native CUDA and Triton then run from independent clones
of exactly those tensors.  JIT compilation and CUDA graph capture are excluded.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch
import triton
import triton.language as tl

import tide
from tide import backend_utils


@triton.jit
def _diff_bwd(ptr, i, stride: tl.constexpr, scale: tl.constexpr):
    return (
        1.125 * (tl.load(ptr + i) - tl.load(ptr + i - stride))
        - (1.0 / 24.0) * (tl.load(ptr + i + stride) - tl.load(ptr + i - 2 * stride))
    ) * scale


@triton.jit
def _diff_fwd(ptr, i, stride: tl.constexpr, scale: tl.constexpr):
    return (
        1.125 * (tl.load(ptr + i + stride) - tl.load(ptr + i))
        - (1.0 / 24.0) * (tl.load(ptr + i + 2 * stride) - tl.load(ptr + i - stride))
    ) * scale


@triton.jit
def _h_kernel(
    cq,
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    m_ey_z,
    m_ez_y,
    m_ez_x,
    m_ex_z,
    m_ex_y,
    m_ey_x,
    azh,
    bzh,
    ayh,
    byh,
    axh,
    bxh,
    kzh,
    kyh,
    kxh,
    n_cells: tl.constexpr,
    nz: tl.constexpr,
    ny: tl.constexpr,
    nx: tl.constexpr,
    pml_z0: tl.constexpr,
    pml_y0: tl.constexpr,
    pml_x0: tl.constexpr,
    pml_z1: tl.constexpr,
    pml_y1: tl.constexpr,
    pml_x1: tl.constexpr,
    rdz: tl.constexpr,
    rdy: tl.constexpr,
    rdx: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = i < n_cells
    j = i % n_cells
    x = j % nx
    y = (j // nx) % ny
    z = j // (ny * nx)
    active = (
        in_range
        & (z >= 2)
        & (z < nz - 1)
        & (y >= 2)
        & (y < ny - 1)
        & (x >= 2)
        & (x < nx - 1)
    )
    # Give inactive lanes a safe central address; stores remain masked.
    q = tl.where(active, i, 2 * ny * nx + 2 * nx + 2)
    zs = tl.where(active, z, 2)
    ys = tl.where(active, y, 2)
    xs = tl.where(active, x, 2)
    vz = active & (z < nz - 2)
    vy = active & (y < ny - 2)
    vx = active & (x < nx - 2)
    dez_y = tl.where(vy, _diff_fwd(ez, q, nx, rdy), 0.0)
    dey_z = tl.where(vz, _diff_fwd(ey, q, ny * nx, rdz), 0.0)
    dez_x = tl.where(vx, _diff_fwd(ez, q, 1, rdx), 0.0)
    dex_z = tl.where(vz, _diff_fwd(ex, q, ny * nx, rdz), 0.0)
    dex_y = tl.where(vy, _diff_fwd(ex, q, nx, rdy), 0.0)
    dey_x = tl.where(vx, _diff_fwd(ey, q, 1, rdx), 0.0)

    pz = (z < pml_z0) | (z >= pml_z1 - 1)
    py = (y < pml_y0) | (y >= pml_y1 - 1)
    px = (x < pml_x0) | (x >= pml_x1 - 1)
    mz0 = tl.load(m_ey_z + q)
    mz1 = tl.load(m_ex_z + q)
    my0 = tl.load(m_ez_y + q)
    my1 = tl.load(m_ex_y + q)
    mx0 = tl.load(m_ez_x + q)
    mx1 = tl.load(m_ey_x + q)
    nmz0 = tl.load(bzh + zs) * mz0 + tl.load(azh + zs) * dey_z
    nmz1 = tl.load(bzh + zs) * mz1 + tl.load(azh + zs) * dex_z
    nmy0 = tl.load(byh + ys) * my0 + tl.load(ayh + ys) * dez_y
    nmy1 = tl.load(byh + ys) * my1 + tl.load(ayh + ys) * dex_y
    nmx0 = tl.load(bxh + xs) * mx0 + tl.load(axh + xs) * dez_x
    nmx1 = tl.load(bxh + xs) * mx1 + tl.load(axh + xs) * dey_x
    tl.store(m_ey_z + i, nmz0, mask=vz & pz)
    tl.store(m_ex_z + i, nmz1, mask=vz & pz)
    tl.store(m_ez_y + i, nmy0, mask=vy & py)
    tl.store(m_ex_y + i, nmy1, mask=vy & py)
    tl.store(m_ez_x + i, nmx0, mask=vx & px)
    tl.store(m_ey_x + i, nmx1, mask=vx & px)
    dey_z = tl.where(vz & pz, dey_z / tl.load(kzh + zs) + nmz0, dey_z)
    dex_z = tl.where(vz & pz, dex_z / tl.load(kzh + zs) + nmz1, dex_z)
    dez_y = tl.where(vy & py, dez_y / tl.load(kyh + ys) + nmy0, dez_y)
    dex_y = tl.where(vy & py, dex_y / tl.load(kyh + ys) + nmy1, dex_y)
    dez_x = tl.where(vx & px, dez_x / tl.load(kxh + xs) + nmx0, dez_x)
    dey_x = tl.where(vx & px, dey_x / tl.load(kxh + xs) + nmx1, dey_x)
    cv = tl.load(cq + j)
    tl.store(hx + i, tl.load(hx + q) - cv * (dey_z - dez_y), mask=active)
    tl.store(hy + i, tl.load(hy + q) - cv * (dez_x - dex_z), mask=active)
    tl.store(hz + i, tl.load(hz + q) - cv * (dex_y - dey_x), mask=active)


@triton.jit
def _e_kernel(
    ca,
    cb,
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    m_hy_z,
    m_hz_y,
    m_hz_x,
    m_hx_z,
    m_hx_y,
    m_hy_x,
    az,
    bz,
    ay,
    by,
    ax,
    bx,
    kz,
    ky,
    kx,
    n_cells: tl.constexpr,
    nz: tl.constexpr,
    ny: tl.constexpr,
    nx: tl.constexpr,
    pml_z0: tl.constexpr,
    pml_y0: tl.constexpr,
    pml_x0: tl.constexpr,
    pml_z1: tl.constexpr,
    pml_y1: tl.constexpr,
    pml_x1: tl.constexpr,
    rdz: tl.constexpr,
    rdy: tl.constexpr,
    rdx: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = i < n_cells
    x = i % nx
    y = (i // nx) % ny
    z = i // (ny * nx)
    active = (
        in_range
        & (z >= 2)
        & (z < nz - 1)
        & (y >= 2)
        & (y < ny - 1)
        & (x >= 2)
        & (x < nx - 1)
    )
    q = tl.where(active, i, 2 * ny * nx + 2 * nx + 2)
    zs = tl.where(active, z, 2)
    ys = tl.where(active, y, 2)
    xs = tl.where(active, x, 2)
    dhy_z = _diff_bwd(hy, q, ny * nx, rdz)
    dhz_y = _diff_bwd(hz, q, nx, rdy)
    dhz_x = _diff_bwd(hz, q, 1, rdx)
    dhx_z = _diff_bwd(hx, q, ny * nx, rdz)
    dhx_y = _diff_bwd(hx, q, nx, rdy)
    dhy_x = _diff_bwd(hy, q, 1, rdx)
    pz = (z < pml_z0) | (z >= pml_z1)
    py = (y < pml_y0) | (y >= pml_y1)
    px = (x < pml_x0) | (x >= pml_x1)
    mz0 = tl.load(m_hy_z + q)
    mz1 = tl.load(m_hx_z + q)
    my0 = tl.load(m_hz_y + q)
    my1 = tl.load(m_hx_y + q)
    mx0 = tl.load(m_hz_x + q)
    mx1 = tl.load(m_hy_x + q)
    nmz0 = tl.load(bz + zs) * mz0 + tl.load(az + zs) * dhy_z
    nmz1 = tl.load(bz + zs) * mz1 + tl.load(az + zs) * dhx_z
    nmy0 = tl.load(by + ys) * my0 + tl.load(ay + ys) * dhz_y
    nmy1 = tl.load(by + ys) * my1 + tl.load(ay + ys) * dhx_y
    nmx0 = tl.load(bx + xs) * mx0 + tl.load(ax + xs) * dhz_x
    nmx1 = tl.load(bx + xs) * mx1 + tl.load(ax + xs) * dhy_x
    tl.store(m_hy_z + i, nmz0, mask=active & pz)
    tl.store(m_hx_z + i, nmz1, mask=active & pz)
    tl.store(m_hz_y + i, nmy0, mask=active & py)
    tl.store(m_hx_y + i, nmy1, mask=active & py)
    tl.store(m_hz_x + i, nmx0, mask=active & px)
    tl.store(m_hy_x + i, nmx1, mask=active & px)
    dhy_z = tl.where(pz, dhy_z / tl.load(kz + zs) + nmz0, dhy_z)
    dhx_z = tl.where(pz, dhx_z / tl.load(kz + zs) + nmz1, dhx_z)
    dhz_y = tl.where(py, dhz_y / tl.load(ky + ys) + nmy0, dhz_y)
    dhx_y = tl.where(py, dhx_y / tl.load(ky + ys) + nmy1, dhx_y)
    dhz_x = tl.where(px, dhz_x / tl.load(kx + xs) + nmx0, dhz_x)
    dhy_x = tl.where(px, dhy_x / tl.load(kx + xs) + nmx1, dhy_x)
    av = tl.load(ca + q)
    bv = tl.load(cb + q)
    tl.store(ex + i, av * tl.load(ex + q) + bv * (dhy_z - dhz_y), mask=active)
    tl.store(ey + i, av * tl.load(ey + q) + bv * (dhz_x - dhx_z), mask=active)
    tl.store(ez + i, av * tl.load(ez + q) + bv * (dhx_y - dhy_x), mask=active)


@triton.jit
def _io_kernel(field, f, receiver, sources_i, receivers_i, t):
    src = tl.load(sources_i)
    tl.store(field + src, tl.load(field + src) + tl.load(f + t))
    rec = tl.load(receivers_i)
    tl.store(receiver + t, tl.load(field + rec))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="40,40,40")
    p.add_argument("--nt", type=int, default=120)
    p.add_argument("--pml-width", type=int, default=10)
    p.add_argument(
        "--block-size", type=int, default=256, choices=(64, 128, 256, 512, 1024)
    )
    p.add_argument("--repeat", type=int, default=9)
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def _capture_prepared(shape: tuple[int, int, int], nt: int, pml: int):
    captured: dict[str, object] = {}
    original_get = backend_utils.get_backend_function
    original_ptr = backend_utils.tensor_to_ptr

    def get_wrapped(*args, **kwargs):
        real = original_get(*args, **kwargs)

        def invoke(*call_args):
            captured["real"] = real
            captured["args"] = call_args
            return real(
                *(
                    original_ptr(x) if isinstance(x, torch.Tensor) else x
                    for x in call_args
                )
            )

        return invoke

    backend_utils.get_backend_function = get_wrapped
    backend_utils.tensor_to_ptr = lambda x: x
    try:
        device = torch.device("cuda")
        dt = 1.6e-11
        epsilon = torch.full(shape, 4.0, device=device, dtype=torch.float32)
        sigma = torch.zeros_like(epsilon)
        mu = torch.ones_like(epsilon)
        source = torch.tensor([[[5, shape[1] // 2, shape[2] // 2]]], device=device)
        receiver = source.clone()
        receiver[..., 2] += min(10, max(0, shape[2] // 2 - 6))
        amplitude = tide.ricker(
            160e6, nt, dt, peak_time=1.2 / 160e6, device=device
        ).view(1, 1, nt)
        tide.maxwell3d(
            epsilon,
            sigma,
            mu,
            0.02,
            dt,
            amplitude,
            source,
            receiver,
            pml_width=pml,
            stencil=4,
            source_component="ey",
            receiver_component="ey",
            n_threads=256,
            python_backend=False,
            save_snapshots=False,
        )
        torch.cuda.synchronize()
    finally:
        backend_utils.get_backend_function = original_get
        backend_utils.tensor_to_ptr = original_ptr
    names = backend_utils.backend_signature("maxwell_3d", "forward")
    return dict(zip(names, captured["args"])), captured["real"], original_ptr


def _clone_prepared(prepared: dict[str, object]) -> dict[str, object]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in prepared.items()
    }


STATE_NAMES = (
    "ex",
    "ey",
    "ez",
    "hx",
    "hy",
    "hz",
    "m_hz_y",
    "m_hy_z",
    "m_hx_z",
    "m_hz_x",
    "m_hy_x",
    "m_hx_y",
    "m_ey_z",
    "m_ez_y",
    "m_ez_x",
    "m_ex_z",
    "m_ex_y",
    "m_ey_x",
    "r",
)


def _reset(p: dict[str, object]) -> None:
    for name in STATE_NAMES:
        tensor = p[name]
        assert isinstance(tensor, torch.Tensor)
        tensor.zero_()


def _triton_run(p: dict[str, object], block: int) -> None:
    if int(p["n_shots"]) != 1:
        raise ValueError("The experimental Triton path supports exactly one shot")
    if int(p["n_sources_per_shot"]) != 1 or int(p["n_receivers_per_shot"]) != 1:
        raise ValueError(
            "The experimental Triton path supports one source and receiver per shot"
        )
    nz, ny, nx = int(p["nz"]), int(p["ny"]), int(p["nx"])
    cells = nz * ny * nx
    grid = (triton.cdiv(cells, block),)
    common = dict(
        n_cells=cells,
        nz=nz,
        ny=ny,
        nx=nx,
        pml_z0=int(p["pml_z0"]),
        pml_y0=int(p["pml_y0"]),
        pml_x0=int(p["pml_x0"]),
        pml_z1=int(p["pml_z1"]),
        pml_y1=int(p["pml_y1"]),
        pml_x1=int(p["pml_x1"]),
        rdz=float(p["rdz"]),
        rdy=float(p["rdy"]),
        rdx=float(p["rdx"]),
        BLOCK=block,
        num_warps=8 if block >= 256 else 4,
    )
    for t in range(int(p["nt"])):
        _h_kernel[grid](
            p["cq"],
            p["ex"],
            p["ey"],
            p["ez"],
            p["hx"],
            p["hy"],
            p["hz"],
            p["m_ey_z"],
            p["m_ez_y"],
            p["m_ez_x"],
            p["m_ex_z"],
            p["m_ex_y"],
            p["m_ey_x"],
            p["azh"],
            p["bzh"],
            p["ayh"],
            p["byh"],
            p["axh"],
            p["bxh"],
            p["kzh"],
            p["kyh"],
            p["kxh"],
            **common,
        )
        _e_kernel[grid](
            p["ca"],
            p["cb"],
            p["ex"],
            p["ey"],
            p["ez"],
            p["hx"],
            p["hy"],
            p["hz"],
            p["m_hy_z"],
            p["m_hz_y"],
            p["m_hz_x"],
            p["m_hx_z"],
            p["m_hx_y"],
            p["m_hy_x"],
            p["az"],
            p["bz"],
            p["ay"],
            p["by"],
            p["ax"],
            p["bx"],
            p["kz"],
            p["ky"],
            p["kx"],
            **common,
        )
        _io_kernel[(1,)](p["ey"], p["f"], p["r"], p["sources_i"], p["receivers_i"], t=t)


def _measure_once(run: Callable[[], None], reset: Callable[[], None]) -> float:
    reset()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _elapsed_pair(native_run, native_reset, triton_run, triton_reset, repeat):
    native_samples, triton_samples = [], []
    # Alternate order to reduce bias from clocks or concurrent machine load.
    for idx in range(repeat):
        if idx % 2 == 0:
            native_samples.append(_measure_once(native_run, native_reset))
            triton_samples.append(_measure_once(triton_run, triton_reset))
        else:
            triton_samples.append(_measure_once(triton_run, triton_reset))
            native_samples.append(_measure_once(native_run, native_reset))
    return native_samples, triton_samples


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = tuple(int(x) for x in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError("--shape must have three dimensions")
    prepared, native_func, to_ptr = _capture_prepared(shape, args.nt, args.pml_width)
    native = _clone_prepared(prepared)
    triton_p = _clone_prepared(prepared)
    names = backend_utils.backend_signature("maxwell_3d", "forward")
    native_args = [native[name] for name in names]

    def native_run():
        native_func(
            *(to_ptr(x) if isinstance(x, torch.Tensor) else x for x in native_args)
        )

    # Compile both implementations and establish correctness from clean state.
    _reset(native)
    native_run()
    torch.cuda.synchronize()
    native_r = native["r"].clone()
    native_fields = [native[n].clone() for n in ("ex", "ey", "ez", "hx", "hy", "hz")]
    _reset(triton_p)
    _triton_run(triton_p, args.block_size)
    torch.cuda.synchronize()
    triton_r = triton_p["r"].clone()
    triton_fields = [triton_p[n].clone() for n in ("ex", "ey", "ez", "hx", "hy", "hz")]
    receiver_rel_l2 = float(
        torch.linalg.vector_norm(triton_r - native_r)
        / torch.linalg.vector_norm(native_r).clamp_min(1e-30)
    )
    field_num = sum(
        torch.sum((a - b) ** 2) for a, b in zip(triton_fields, native_fields)
    )
    field_den = sum(torch.sum(a**2) for a in native_fields)
    field_rel_l2 = float(torch.sqrt(field_num / field_den.clamp_min(1e-30)))

    if args.cuda_graph:
        # Replay is the propagation analog of one low-level native call; state
        # reset remains outside both timings.
        _reset(triton_p)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _triton_run(triton_p, args.block_size)
        torch.cuda.synchronize()
        triton_run = graph.replay
    else:
        def triton_run():
            _triton_run(triton_p, args.block_size)

    native_samples, triton_samples = _elapsed_pair(
        native_run,
        lambda: _reset(native),
        triton_run,
        lambda: _reset(triton_p),
        args.repeat,
    )
    native_ms = statistics.median(native_samples)
    triton_ms = statistics.median(triton_samples)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "shape": shape,
        "prepared_shape": (
            int(prepared["nz"]),
            int(prepared["ny"]),
            int(prepared["nx"]),
        ),
        "nt": args.nt,
        "pml_width": args.pml_width,
        "stencil": 4,
        "block_size": args.block_size,
        "native_cuda_median_ms": native_ms,
        "triton_median_ms": triton_ms,
        "ratio": triton_ms / native_ms,
        "receiver_relative_l2": receiver_rel_l2,
        "final_fields_relative_l2": field_rel_l2,
        "native_samples_ms": native_samples,
        "triton_samples_ms": triton_samples,
        "triton_cuda_graph": args.cuda_graph,
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
