import torch

from tide.maxwell.compact_pml import (
    aux_direction_3d,
    build_compact_pml_layout_3d,
    pack_aux_field_3d,
    unpack_aux_field_3d,
)


def test_compact_pml_layout_shapes_and_memory_ratio():
    layout = build_compact_pml_layout_3d(
        n_shots=2,
        nz=21,
        ny=22,
        nx=23,
        fd_pad=(1, 0, 1, 0, 1, 0),
        pml_width=(4, 4, 3, 3, 2, 2),
    )

    assert layout.direction_shape("z") == (2, 9, 22, 23)
    assert layout.direction_shape("y") == (2, 21, 7, 23)
    assert layout.direction_shape("x") == (2, 21, 22, 5)
    assert layout.aux_shape("m_hy_z") == layout.direction_shape("z")
    assert layout.aux_shape("m_hz_y") == layout.direction_shape("y")
    assert layout.aux_shape("m_ey_x") == layout.direction_shape("x")
    assert layout.compact_aux_elements < layout.full_aux_elements
    assert layout.compact_to_full_ratio < 0.5


def test_compact_pml_aux_direction_map_covers_edges_by_direction_slab():
    assert aux_direction_3d("m_hy_z") == "z"
    assert aux_direction_3d("m_hx_z") == "z"
    assert aux_direction_3d("m_hz_y") == "y"
    assert aux_direction_3d("m_ex_y") == "y"
    assert aux_direction_3d("m_hz_x") == "x"
    assert aux_direction_3d("m_ez_x") == "x"


def test_pack_unpack_directional_aux_preserves_zeroed_full_domain():
    layout = build_compact_pml_layout_3d(
        n_shots=1,
        nz=10,
        ny=11,
        nx=12,
        fd_pad=(1, 0, 1, 0, 1, 0),
        pml_width=(2, 3, 2, 2, 1, 3),
    )
    full = torch.zeros(1, 10, 11, 12)
    full[:, layout.z_indices, :, :] = torch.randn(1, len(layout.z_indices), 11, 12)
    compact = pack_aux_field_3d("m_ey_z", full, layout=layout)
    unpacked = unpack_aux_field_3d("m_ey_z", compact, layout=layout)
    torch.testing.assert_close(unpacked, full)

    edge_value = torch.tensor(7.0)
    full[:, layout.z_indices[0], layout.y_indices[0], layout.x_indices[0]] = edge_value
    compact = pack_aux_field_3d("m_ex_z", full, layout=layout)
    unpacked = unpack_aux_field_3d("m_ex_z", compact, layout=layout)
    torch.testing.assert_close(unpacked, full)


def test_pack_unpack_rejects_wrong_direction_shape():
    layout = build_compact_pml_layout_3d(
        n_shots=1,
        nz=8,
        ny=9,
        nx=10,
        fd_pad=(1, 0, 1, 0, 1, 0),
        pml_width=(1, 1, 1, 1, 1, 1),
    )
    full = torch.zeros(1, 8, 9, 10)
    compact = pack_aux_field_3d("m_hz_y", full, layout=layout)
    try:
        unpack_aux_field_3d("m_hz_x", compact, layout=layout)
    except ValueError as exc:
        assert "compact aux shape" in str(exc)
    else:
        raise AssertionError("unpack should reject a compact tensor from another direction")
