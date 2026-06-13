from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "image_domain_lsrtm_pcgnr_over_new.py"
    )
    spec = importlib.util.spec_from_file_location("image_domain_lsrtm_pcgnr_over_new", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


example = _load_example_module()


def test_psf_hessian3d_adjoint_with_row_weights():
    generator = torch.Generator().manual_seed(20260602)
    psf = torch.rand((4, 5, 6), generator=generator, dtype=torch.float64) + 0.1
    active = torch.ones_like(psf)
    active[0, :, :] = 0.0
    row_weights = torch.rand(psf.shape, generator=generator, dtype=torch.float64) + 0.2
    operator = example.PsfHessian3D(
        psf,
        active_mask=active,
        cw_z=2,
        cw_y=3,
        cw_x=2,
        origins=(1, 1, 2),
        symmetrize=False,
        row_weights=row_weights,
    )
    x = torch.randn(psf.shape, generator=generator, dtype=torch.float64) * active
    y = torch.randn(psf.shape, generator=generator, dtype=torch.float64) * active

    lhs = example.dot_product(operator.matvec(x), y)
    rhs = example.dot_product(x, operator.adjoint(y))
    denom = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(torch.finfo(torch.float64).eps)

    assert ((lhs - rhs).abs() / denom).item() < 1e-10


def test_psf_hessian3d_couples_crossline_x_direction():
    psf = torch.ones((5, 5, 5), dtype=torch.float64)
    active = torch.ones_like(psf)
    operator = example.PsfHessian3D(
        psf,
        active_mask=active,
        cw_z=1,
        cw_y=1,
        cw_x=3,
        origins=(0, 0, 0),
        symmetrize=False,
    )
    image = torch.zeros_like(psf)
    image[2, 2, 3] = 1.0

    out = operator.matvec(image)

    assert out[2, 2, 2].item() > 0.0


def test_balanced_probe_origin_centers_lateral_comb():
    assert example.balanced_probe_origin(200, 10) == 5
    assert list(example.regular_probe_positions(200, 5, 10))[:3] == [5, 15, 25]
    assert list(example.regular_probe_positions(200, 5, 10))[-3:] == [175, 185, 195]


def test_center_value_interpolation_normalizes_edge_corners():
    centers = torch.tensor([[0, 5, 0]], dtype=torch.long)
    center_values = torch.tensor([2.0], dtype=torch.float64)

    out = example.interpolate_center_value_volume_3d(
        center_values,
        centers=centers,
        shape=(1, 10, 1),
        origins=(0, 5, 0),
        cw_z=1,
        cw_y=10,
        cw_x=1,
    )

    torch.testing.assert_close(out[0, 0, 0], center_values[0])
    torch.testing.assert_close(out[0, 9, 0], center_values[0])


def test_lateral_cosine_taper3_is_symmetric():
    mask = torch.ones((1, 9, 1), dtype=torch.float64)

    taper = example.lateral_cosine_taper_3d(mask, 3)[0, :, 0]

    expected = torch.tensor(
        [0.0, 0.25, 0.75, 1.0, 1.0, 1.0, 0.75, 0.25, 0.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(taper, expected, atol=1e-12, rtol=1e-12)


def test_psf_hessian3d_uses_zero_outside_boundary():
    psf = torch.ones((1, 3, 1), dtype=torch.float64)
    active = torch.ones_like(psf)
    operator = example.PsfHessian3D(
        psf,
        active_mask=active,
        cw_z=1,
        cw_y=3,
        cw_x=1,
        origins=(0, 0, 0),
        symmetrize=False,
    )
    image = torch.zeros_like(psf)
    image[0, 0, 0] = 1.0

    out = operator.matvec(image)

    assert out[0, 0, 0].item() == 1.0
    assert out[0, 1, 0].item() == 1.0
    assert out[0, 2, 0].item() == 0.0


def test_illumination_compensation_normalizes_active_region():
    illumination = torch.tensor([[[1.0, 10.0, 0.0]]], dtype=torch.float64)
    image_mask = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64)

    _, compensation = example.build_illumination_compensation(
        illumination,
        image_mask,
        enabled=True,
        floor_ratio=0.0,
        power=1.0,
        max_gain=100.0,
    )
    _, identity = example.build_illumination_compensation(
        illumination,
        image_mask,
        enabled=False,
        floor_ratio=0.0,
        power=1.0,
        max_gain=100.0,
    )

    assert compensation[0, 0, 0] > compensation[0, 0, 1]
    assert compensation[0, 0, 2] == 0.0
    torch.testing.assert_close(identity, image_mask)


def test_estimate_deblur_filter3d_delta_psf_returns_identity():
    psf_patch = torch.zeros((5, 5, 5), dtype=torch.float64)
    psf_patch[2, 2, 2] = 1.0
    reference_patch = psf_patch.clone()

    filt = example.estimate_deblur_filter_3d(
        psf_patch,
        reference_patch,
        filter_shape=(3, 3, 3),
        damping=1e-12,
    )

    expected = torch.zeros_like(filt)
    expected[1, 1, 1] = 1.0
    torch.testing.assert_close(filt, expected, atol=1e-8, rtol=1e-8)


def test_estimate_deblur_filter3d_ignores_invalid_patch_rows():
    psf_patch = torch.ones((3, 3, 3), dtype=torch.float64)
    reference_patch = torch.full_like(psf_patch, 100.0)
    reference_patch[1, 1, 1] = 1.0
    valid_mask = torch.zeros_like(psf_patch, dtype=torch.bool)
    valid_mask[1, 1, 1] = True

    filt = example.estimate_deblur_filter_3d(
        psf_patch,
        reference_patch,
        filter_shape=(1, 1, 1),
        damping=0.0,
        valid_mask=valid_mask,
    )

    torch.testing.assert_close(filt, torch.ones_like(filt))


def test_apply_deblur_filter_bank3d_identity_is_noop():
    generator = torch.Generator().manual_seed(20260603)
    image = torch.randn((4, 5, 6), generator=generator, dtype=torch.float64)
    active = torch.ones_like(image)
    centers = torch.nonzero(active > 0, as_tuple=False)
    filters = torch.zeros((centers.shape[0], 3, 3, 3), dtype=torch.float64)
    filters[:, 1, 1, 1] = 1.0
    filter_bank = example.DeblurFilterBank3D(
        centers=centers,
        filters=filters,
        filter_shape=(3, 3, 3),
        patch_shape=(3, 3, 3),
        damping=0.0,
    )

    out = example.apply_deblur_filter_bank_3d(
        image,
        filter_bank,
        active_mask=active,
        origins=(0, 0, 0),
        cw_z=1,
        cw_y=1,
        cw_x=1,
    )

    torch.testing.assert_close(out, image)
