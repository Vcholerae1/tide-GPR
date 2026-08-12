import pytest
import torch

from numerical_utils import make_maxwell3d_example
from tide import backend_utils


def _example(device: torch.device):
    return make_maxwell3d_example(
        shape=(6, 6, 7),
        nt=10,
        grid_spacing=0.02,
        dt=4e-11,
        frequency=90e6,
        device=device,
        source_location=(2, 2, 2),
        receiver_locations=((2, 2, 4),),
        pml_width=2,
    )


def test_maxwell3d_backend_parity_via_fallback():
    example = _example(torch.device("cpu"))
    out_python = example.run(python_backend=True)
    out_backend = example.run(python_backend=False)
    for actual, reference in zip(out_backend, out_python, strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("n_threads", [0, 128, 256])
def test_maxwell3d_native_cuda_matches_python_without_callback(n_threads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for native 3D CUDA parity test.")
    if not backend_utils.is_backend_available():
        pytest.skip("Native backend is required for native 3D CUDA parity test.")

    example = _example(torch.device("cuda"))
    out_python = example.run(python_backend=True)
    out_backend = example.run(python_backend=False, n_threads=n_threads)
    torch.testing.assert_close(
        out_backend[-1],
        out_python[-1],
        atol=1e-4,
        rtol=1e-4,
    )
