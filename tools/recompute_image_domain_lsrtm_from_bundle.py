from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "image_domain_lsrtm_pcgnr_over_new.py"
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs"
    / "over_image_domain_lsrtm_1lines_sigma10"
    / "image_domain_lsrtm_3d_results.h5"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "over_image_domain_lsrtm_1lines_sigma10_interp_norm_taper3_bundle"
)


def load_example_module():
    spec = importlib.util.spec_from_file_location(
        "image_domain_lsrtm_pcgnr_over_new",
        EXAMPLE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_from_h5(h5: h5py.File, name: str, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(h5[name][...]).to(device=device)


def attr_string(h5: h5py.File, name: str, default: str) -> str:
    value = h5.attrs.get(name, default)
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def attr_int_tuple(h5: h5py.File, name: str) -> tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(h5.attrs[name]).reshape(-1))


def load_bundle(
    module,
    input_path: Path,
    *,
    device: torch.device,
):
    with h5py.File(input_path, "r") as h5:
        epsilon_true = tensor_from_h5(h5, "models/epsilon_true", device=device)
        epsilon_background = tensor_from_h5(
            h5,
            "models/epsilon_background",
            device=device,
        )
        depsilon_true = tensor_from_h5(h5, "models/depsilon_true", device=device)
        model_weights = tensor_from_h5(h5, "models/model_weights", device=device)
        line_mask = tensor_from_h5(h5, "models/line_mask", device=device)
        source_locations = tensor_from_h5(
            h5,
            "geometry/source_locations",
            device=device,
        )
        receiver_locations = tensor_from_h5(
            h5,
            "geometry/receiver_locations",
            device=device,
        )
        line_xs = tuple(int(v) for v in np.asarray(h5["geometry/line_xs"][...]))
        observed_data = tensor_from_h5(h5, "data/observed_data", device=device)
        background_data = tensor_from_h5(h5, "data/background_data", device=device)
        observed_scattered = tensor_from_h5(
            h5,
            "data/observed_scattered",
            device=device,
        )
        standard_rtm_image = tensor_from_h5(
            h5,
            "models/standard_rtm_image",
            device=device,
        )
        standard_psf_image = tensor_from_h5(
            h5,
            "models/standard_psf_image",
            device=device,
        )
        psf_probe = tensor_from_h5(h5, "models/psf_probe", device=device)
        illumination = tensor_from_h5(
            h5,
            "models/source_illumination",
            device=device,
        )
        illumination_compensation = tensor_from_h5(
            h5,
            "models/illumination_compensation",
            device=device,
        )
        centers = torch.from_numpy(h5["deblur/centers"][...]).to(
            device=device,
            dtype=torch.long,
        )
        filters = torch.from_numpy(h5["deblur/filters"][...]).to(
            device=device,
            dtype=standard_rtm_image.dtype,
        )
        filter_bank = module.DeblurFilterBank3D(
            centers=centers,
            filters=filters,
            filter_shape=attr_int_tuple(h5, "deblur_filter_shape"),
            patch_shape=attr_int_tuple(h5, "deblur_patch_shape"),
            damping=float(h5.attrs["deblur_damping"]),
        )
        case = SimpleNamespace(
            epsilon_true=epsilon_true,
            epsilon_background=epsilon_background,
            sigma=torch.zeros_like(epsilon_background),
            mu=torch.ones_like(epsilon_background),
            depsilon_true=depsilon_true,
            source_amplitude=torch.empty(0, device=device),
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            active_mask=model_weights,
            model_weights=model_weights,
            line_mask=line_mask,
            data_weights=torch.ones_like(observed_data),
            model_gradient_sampling_interval=int(
                h5.attrs["model_gradient_sampling_interval"]
            ),
            illumination_wavefield_sampling_interval=int(
                h5.attrs["source_illumination_sampling_interval"]
            ),
            batch_size=1,
            dx=float(h5.attrs["dx"]),
            dt=float(h5.attrs["dt"]),
            freq=float(h5.attrs["freq"]),
            pml_width=int(h5.attrs["pml_width"]),
            stencil=int(h5.attrs["stencil"]),
            max_vel=float("nan"),
            python_backend=False,
            storage_mode=attr_string(h5, "storage_mode", module.STORAGE_MODE),
            storage_compression=attr_string(
                h5,
                "storage_compression",
                module.STORAGE_COMPRESSION,
            ),
            model_label=attr_string(h5, "model_label", "bundle"),
            line_xs=line_xs,
        )
        attrs = {
            "psf_cw_z": int(h5.attrs["psf_cw_z"]),
            "psf_cw_y": int(h5.attrs["psf_cw_y"]),
            "psf_cw_x": int(h5.attrs["psf_cw_x"]),
            "image_domain_lateral_taper_cells": int(
                h5.attrs.get(
                    "image_domain_lateral_taper_cells",
                    module.IMAGE_DOMAIN_LATERAL_TAPER_CELLS,
                )
            ),
            "illumination_compensation_enabled": bool(
                h5.attrs["illumination_compensation_enabled"]
            ),
            "illumination_compensation_floor": float(
                h5.attrs["illumination_compensation_floor"]
            ),
            "illumination_compensation_power": float(
                h5.attrs["illumination_compensation_power"]
            ),
            "illumination_compensation_max_gain": float(
                h5.attrs["illumination_compensation_max_gain"]
            ),
        }
    return SimpleNamespace(
        case=case,
        observed_data=observed_data,
        background_data=background_data,
        observed_scattered=observed_scattered,
        standard_rtm_image=standard_rtm_image,
        standard_psf_image=standard_psf_image,
        psf_probe=psf_probe,
        illumination=illumination,
        illumination_compensation=illumination_compensation,
        filter_bank=filter_bank,
        attrs=attrs,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, name: str, fn):
    synchronize(device)
    start = perf_counter()
    value = fn()
    synchronize(device)
    seconds = perf_counter() - start
    print(f"{name}: {seconds:.3f}s")
    return value, seconds


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute image-domain LSRTM post-processing from an HDF5 bundle.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lateral-taper-cells", type=int, default=None)
    args = parser.parse_args()

    module = load_example_module()
    device = choose_device(args.device)
    print(f"Using device: {device}")
    bundle = load_bundle(module, args.input, device=device)
    taper_cells = (
        bundle.attrs["image_domain_lateral_taper_cells"]
        if args.lateral_taper_cells is None
        else args.lateral_taper_cells
    )
    image_domain_mask = bundle.case.model_weights
    illumination_compensation = bundle.illumination_compensation
    if taper_cells > 0:
        taper = module.lateral_cosine_taper_3d(
            image_domain_mask,
            taper_cells,
        )
        image_domain_mask = image_domain_mask * taper
        illumination_compensation = illumination_compensation * taper
        print(f"Applied lateral cosine taper: cells={taper_cells}")
    centers = bundle.filter_bank.centers
    if centers.numel() == 0:
        raise ValueError("Bundle deblur filter bank has no centers.")
    psf_origins = tuple(
        int(v) for v in centers.min(dim=0).values.detach().cpu().tolist()
    )
    print(
        "Using bundle PSF grid "
        f"origins={psf_origins}, "
        f"cw=({bundle.attrs['psf_cw_z']}, "
        f"{bundle.attrs['psf_cw_y']}, {bundle.attrs['psf_cw_x']}), "
        f"centers={centers.shape[0]}"
    )

    timings: dict[str, float] = {}
    (rtm_target_image, psf_operator_image), seconds = timed(
        device,
        "Apply stored deblur filters with current interpolation",
        lambda: (
            module.apply_deblur_filter_bank_3d(
                bundle.standard_rtm_image,
                bundle.filter_bank,
                active_mask=image_domain_mask,
                origins=psf_origins,
                cw_z=bundle.attrs["psf_cw_z"],
                cw_y=bundle.attrs["psf_cw_y"],
                cw_x=bundle.attrs["psf_cw_x"],
            ).detach(),
            module.apply_deblur_filter_bank_3d(
                bundle.standard_psf_image,
                bundle.filter_bank,
                active_mask=image_domain_mask,
                origins=psf_origins,
                cw_z=bundle.attrs["psf_cw_z"],
                cw_y=bundle.attrs["psf_cw_y"],
                cw_x=bundle.attrs["psf_cw_x"],
            ).detach(),
        ),
    )
    timings["deblur_apply"] = seconds

    operator, seconds = timed(
        device,
        "Build current PSF Hessian",
        lambda: module.PsfHessian3D(
            psf_operator_image,
            active_mask=image_domain_mask,
            cw_z=bundle.attrs["psf_cw_z"],
            cw_y=bundle.attrs["psf_cw_y"],
            cw_x=bundle.attrs["psf_cw_x"],
            origins=psf_origins,
            symmetrize=module.SYMMETRIZE_PSF,
            row_weights=illumination_compensation,
        ),
    )
    timings["psf_operator"] = seconds

    iterations = module.IMAGE_DOMAIN_ITERS if args.iterations is None else args.iterations
    result, seconds = timed(
        device,
        "Solve image-domain PCGNR",
        lambda: module.solve_image_domain_pcgnr(
            operator,
            rtm_target_image,
            image_mask=image_domain_mask,
            iterations=iterations,
            precondition_damping=module.PRECONDITION_DAMPING,
        ),
    )
    timings["image_domain_pcgnr"] = seconds
    image_domain_lsrtm = module.apply_mask(result.image, image_domain_mask)

    module.plot_results(
        bundle.case,
        observed_data=bundle.observed_data,
        background_data=bundle.background_data,
        observed_scattered=bundle.observed_scattered,
        rtm_image=bundle.standard_rtm_image,
        psf_probe=bundle.psf_probe,
        psf_image=psf_operator_image,
        illumination=bundle.illumination,
        illumination_compensation=illumination_compensation,
        image_domain_lsrtm=image_domain_lsrtm,
        result=result,
        deblurred=True,
        output_dir=args.output_dir,
    )
    module.save_h5(
        bundle.case,
        observed_data=bundle.observed_data,
        background_data=bundle.background_data,
        observed_scattered=bundle.observed_scattered,
        rtm_image=rtm_target_image,
        psf_probe=bundle.psf_probe,
        psf_image=psf_operator_image,
        illumination=bundle.illumination,
        illumination_compensation=illumination_compensation,
        image_domain_mask=image_domain_mask,
        image_domain_lsrtm=image_domain_lsrtm,
        result=result,
        standard_rtm_image=bundle.standard_rtm_image,
        standard_psf_image=bundle.standard_psf_image,
        deblur_filter_bank=bundle.filter_bank,
        timings=timings,
        psf_cw_z=bundle.attrs["psf_cw_z"],
        psf_cw_y=bundle.attrs["psf_cw_y"],
        psf_cw_x=bundle.attrs["psf_cw_x"],
        image_domain_lateral_taper_cells=taper_cells,
        deblurred=True,
        illumination_compensation_enabled=bundle.attrs[
            "illumination_compensation_enabled"
        ],
        illumination_compensation_floor=bundle.attrs[
            "illumination_compensation_floor"
        ],
        illumination_compensation_power=bundle.attrs[
            "illumination_compensation_power"
        ],
        illumination_compensation_max_gain=bundle.attrs[
            "illumination_compensation_max_gain"
        ],
        output_dir=args.output_dir,
    )
    print(f"Wrote bundle recompute to {args.output_dir}")


if __name__ == "__main__":
    main()
