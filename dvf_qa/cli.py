from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dynamic_xray import (
    analyze_dynamic_xray,
    read_frame_sequence,
    resolve_lateral_mask_method,
    run_lateral_smoothing_demo,
    run_public_cxr_demo,
    write_combined_phase,
    write_dynamic_xray_outputs,
)
from .drr import DrrGeometry, make_diffdrr_subject, render_diffdrr_subject, subject_volume_numpy
from .image_io import read_image, read_volume, write_png, write_volume
from .metrics import image_similarity, summarize_dvf_qa
from .report import write_metrics, write_overlay_report
from .warp import warp_volume_moving_to_fixed


def main() -> None:
    parser = argparse.ArgumentParser(prog="dvf-qa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run DVF/CT/DRR QA")
    run.add_argument("--dvf", help="Predicted displacement vector field, shape (z,y,x,3)")
    run.add_argument("--fixed", help="Fixed/target CT for warped CT similarity")
    run.add_argument("--moving", help="Moving/source CT to warp with DVF")
    run.add_argument("--generated-ct", help="Generated CT used for DRR reprojection")
    run.add_argument("--lung-mask", help="Optional lung mask on DVF/fixed grid")
    run.add_argument("--fluoro", help="Input fluoroscopy image to compare with DRR")
    run.add_argument("--geometry", help="DRR geometry JSON")
    run.add_argument("--device", default=None, help="DiffDRR device, e.g. cuda, mps, or cpu")
    run.add_argument("--out", required=True, help="Output directory")
    dynamic = subparsers.add_parser(
        "analyze-dynamic-xray",
        help="Estimate lung masks and respiratory phase from AP/frontal and lateral dynamic X-ray frames",
    )
    dynamic.add_argument("--ap", help="AP/frontal frame sequence: directory, .npy/.npz, or SimpleITK-readable image")
    dynamic.add_argument("--lateral", help="Lateral frame sequence: directory, .npy/.npz, or SimpleITK-readable image")
    dynamic.add_argument(
        "--mask-method",
        choices=("unsupervised", "torchxrayvision"),
        default="unsupervised",
        help="Lung mask method. torchxrayvision uses ChestX-Det PSPNet left/right lung channels.",
    )
    dynamic.add_argument(
        "--lateral-mask-method",
        choices=("auto", "unsupervised", "torchxrayvision"),
        default="auto",
        help="Mask method for lateral images. auto uses unsupervised when --mask-method is torchxrayvision.",
    )
    dynamic.add_argument("--mask-threshold", type=float, default=0.5, help="Probability threshold for TorchXRayVision masks")
    dynamic.add_argument(
        "--signal-method",
        choices=("motion", "intensity-roi", "combined"),
        default="motion",
        help="Respiratory signal source. intensity-roi uses mean pixel intensity in a fixed central lung ROI.",
    )
    dynamic.add_argument("--roi-fraction", type=float, default=0.5, help="Central fraction of each lung component used as ROI")
    dynamic.add_argument(
        "--roi-min-mask-frequency",
        type=float,
        default=0.6,
        help="Minimum fraction of frames where a pixel must be inside the lung mask to enter the fixed ROI",
    )
    dynamic.add_argument("--model-cache-dir", help="Optional TorchXRayVision model cache directory")
    dynamic.add_argument("--device", default=None, help="Torch device for TorchXRayVision, e.g. cuda, mps, or cpu")
    dynamic.add_argument("--out", required=True, help="Output directory")
    demo = subparsers.add_parser(
        "demo-public-cxr",
        help="Download public CC0 frontal/lateral CXR images and write lung mask visualizations",
    )
    demo.add_argument(
        "--mask-method",
        choices=("unsupervised", "torchxrayvision"),
        default="torchxrayvision",
        help="Mask method for the frontal/AP public demo image",
    )
    demo.add_argument(
        "--lateral-mask-method",
        choices=("auto", "unsupervised", "torchxrayvision"),
        default="auto",
        help="Mask method for the lateral public demo image",
    )
    demo.add_argument("--mask-threshold", type=float, default=0.5, help="Probability threshold for TorchXRayVision masks")
    demo.add_argument(
        "--signal-method",
        choices=("motion", "intensity-roi", "combined"),
        default="intensity-roi",
        help="Respiratory signal source for the demo",
    )
    demo.add_argument("--roi-fraction", type=float, default=0.5, help="Central fraction of each lung component used as ROI")
    demo.add_argument("--roi-min-mask-frequency", type=float, default=0.6, help="Minimum mask frequency for fixed ROI")
    demo.add_argument("--model-cache-dir", help="Optional TorchXRayVision model cache directory")
    demo.add_argument("--device", default=None, help="Torch device for TorchXRayVision, e.g. cuda, mps, or cpu")
    demo.add_argument("--out", required=True, help="Output directory")
    smooth_demo = subparsers.add_parser(
        "demo-lateral-smoothing",
        help="Download public lateral CXR images and compare mask smoothing coefficients",
    )
    smooth_demo.add_argument("--cases", type=int, default=10, help="Number of public lateral cases to visualize")
    smooth_demo.add_argument(
        "--sigmas",
        default="0,2,4,6,8",
        help="Comma-separated Gaussian smoothing sigma values in pixels",
    )
    smooth_demo.add_argument("--roi-fraction", type=float, default=0.5, help="Central fraction used for the lateral ROI")
    smooth_demo.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    if args.command == "run":
        run_qa(args)
    elif args.command == "analyze-dynamic-xray":
        run_dynamic_xray(args)
    elif args.command == "demo-public-cxr":
        run_demo_public_cxr(args)
    elif args.command == "demo-lateral-smoothing":
        run_demo_lateral_smoothing(args)


def run_qa(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float | str] = {}
    jacobian = None
    ct_for_report = None

    mask = read_volume(args.lung_mask).data > 0 if args.lung_mask else None

    if args.dvf:
        dvf_vol = read_volume(args.dvf)
        dvf = dvf_vol.data.astype(np.float32)
        dvf_metrics, jacobian = summarize_dvf_qa(dvf, dvf_vol.spacing_xyz, mask)
        metrics.update(dvf_metrics)
        write_volume(out / "jacobian.nii.gz", jacobian.astype(np.float32), dvf_vol)
        write_volume(out / "folding_mask.nii.gz", (jacobian <= 0).astype(np.uint8), dvf_vol)

        if args.fixed:
            fixed_vol = read_volume(args.fixed)
            ct_for_report = fixed_vol.data
        if args.fixed and args.moving:
            fixed_vol = read_volume(args.fixed)
            moving_vol = read_volume(args.moving)
            warped = warp_volume_moving_to_fixed(moving_vol.data.astype(np.float32), dvf, dvf_vol.spacing_xyz)
            write_volume(out / "warped_moving.nii.gz", warped.astype(np.float32), fixed_vol)
            metrics.update(image_similarity(warped, fixed_vol.data, mask, prefix="warped_ct_"))

    drr = None
    fluoro = None
    if args.generated_ct and args.geometry:
        geometry = DrrGeometry.from_dict(json.loads(Path(args.geometry).read_text(encoding="utf-8")))
        subject = make_diffdrr_subject(
            args.generated_ct,
            orientation=geometry.orientation,
            center_volume=geometry.center_volume,
            bone_attenuation_multiplier=geometry.bone_attenuation_multiplier,
            resample_target=geometry.resample_target,
            ct_value_mode=geometry.ct_value_mode,
        )
        ct_for_report = subject_volume_numpy(subject)
        drr = render_diffdrr_subject(subject, geometry, device=args.device).detach().cpu().squeeze().numpy().astype(np.float32)
        write_png(out / "generated_ct_drr.png", drr)
        if args.fluoro:
            fluoro = read_image(args.fluoro)
            metrics.update(image_similarity(_resize_if_needed(drr, fluoro.shape), fluoro, prefix="drr_fluoro_"))

    if "qa_status" not in metrics:
        metrics["qa_status"] = "PASS"
    write_metrics(out / "metrics.json", metrics)
    write_overlay_report(out / "qa_report.png", ct_for_report, jacobian, drr, fluoro, metrics)


def run_dynamic_xray(args: argparse.Namespace) -> None:
    if not args.ap and not args.lateral:
        raise SystemExit("At least one of --ap or --lateral is required.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    ap_result = None
    lateral_result = None
    if args.ap:
        ap_result = analyze_dynamic_xray(
            read_frame_sequence(args.ap),
            view="ap",
            mask_method=args.mask_method,
            device=args.device,
            model_cache_dir=args.model_cache_dir,
            mask_threshold=args.mask_threshold,
            signal_method=args.signal_method,
            roi_fraction=args.roi_fraction,
            roi_min_mask_frequency=args.roi_min_mask_frequency,
        )
        summaries["ap"] = write_dynamic_xray_outputs(ap_result, out)
    if args.lateral:
        lateral_method = resolve_lateral_mask_method(args.mask_method, args.lateral_mask_method)
        lateral_result = analyze_dynamic_xray(
            read_frame_sequence(args.lateral),
            view="lateral",
            mask_method=lateral_method,
            device=args.device,
            model_cache_dir=args.model_cache_dir,
            mask_threshold=args.mask_threshold,
            signal_method=args.signal_method,
            roi_fraction=args.roi_fraction,
            roi_min_mask_frequency=args.roi_min_mask_frequency,
        )
        summaries["lateral"] = write_dynamic_xray_outputs(lateral_result, out)
    combined = write_combined_phase(ap_result, lateral_result, out)
    if combined is not None:
        summaries["combined"] = combined
    write_metrics(out / "dynamic_xray_summary.json", summaries)


def run_demo_public_cxr(args: argparse.Namespace) -> None:
    run_public_cxr_demo(
        args.out,
        mask_method=args.mask_method,
        lateral_mask_method=args.lateral_mask_method,
        device=args.device,
        model_cache_dir=args.model_cache_dir,
        mask_threshold=args.mask_threshold,
        signal_method=args.signal_method,
        roi_fraction=args.roi_fraction,
        roi_min_mask_frequency=args.roi_min_mask_frequency,
    )


def run_demo_lateral_smoothing(args: argparse.Namespace) -> None:
    sigmas = tuple(float(v) for v in args.sigmas.split(",") if v.strip())
    run_lateral_smoothing_demo(
        args.out,
        cases=args.cases,
        sigmas=sigmas,
        roi_fraction=args.roi_fraction,
    )


def _resize_if_needed(image: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if image.shape == shape:
        return image
    from scipy.ndimage import zoom

    factors = (shape[0] / image.shape[0], shape[1] / image.shape[1])
    return zoom(image, factors, order=1)


if __name__ == "__main__":
    main()
