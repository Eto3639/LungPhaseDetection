"""TCIA 4D-Lung demo: load per-phase 4DCT and run the full QA pipeline.

This script wires the pieces together:
  4DCT phases (on disk)
    -> simulate_from_4dct  (per-phase AP + lateral DRR, then dynamic sequence)
    -> respiratory_signal  (Amsterdam Shroud, per view)
    -> pair_best_correlated_cycles
    -> write_html_report

Network access is not required at runtime: the script reads volumes from a
local directory you populate yourself.

Downloading 4D-Lung from TCIA (one-time, manual)
------------------------------------------------
1. Open the 4D-Lung collection page:
     https://www.cancerimagingarchive.net/collection/4d-lung/
2. Use the NBIA Data Retriever (GUI) or the NBIA REST API to download
   one patient's 4DCT study. A 4DCT study contains 10 phase series
   (usually labeled 0%, 10%, ..., 90%), each a full 3D CT.
3. Lay the phases out in a parent directory, one phase per subdirectory:

     /path/to/4dct_phases/
       phase_00/   <- DICOM files for 0% phase
       phase_10/   <- DICOM files for 10% phase
       ...
       phase_90/

   Subdirectory names are sorted alphabetically and that ordering becomes
   the phase index. NIfTI/MHA/.npy/.npz files are also supported via the
   project's ``read_volume``.

4. Run this script:

     python scripts/tcia_4d_lung_demo.py \\
         --phases-dir /path/to/4dct_phases \\
         --out-dir outputs/tcia_demo \\
         --fps 15 --n-cycles 3 --device cpu

   The output directory will contain ``report.html`` plus QC PNGs.

Notes
-----
- Run inside the project venv or Docker image so DiffDRR/TorchIO/SimpleITK
  resolve correctly. Do not pip-install onto the system Python.
- For interactive validation on multiple cases, just call this script per
  case and diff the resulting reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dvf_qa.amsterdam_shroud import respiratory_signal
from dvf_qa.cycle_pairing import pair_best_correlated_cycles
from dvf_qa.image_io import read_volume
from dvf_qa.pipeline_report import ReportInputs, write_html_report
from dvf_qa.synthetic_dynamic import (
    DynamicSimulationConfig,
    default_ap_geometry,
    default_lateral_geometry,
    simulate_from_4dct,
)


IMAGE_SUFFIXES = {".nii", ".gz", ".mha", ".mhd", ".nrrd", ".npy", ".npz"}


def load_phases(phases_dir: Path) -> tuple[list[np.ndarray], tuple[float, float, float]]:
    """Load per-phase volumes from a directory.

    Each immediate child of ``phases_dir`` becomes one phase. Children can
    be either a DICOM directory or a single readable volume file. Children
    are sorted alphabetically; that order becomes the phase index.
    """
    if not phases_dir.is_dir():
        raise SystemExit(f"--phases-dir is not a directory: {phases_dir}")
    children = sorted(p for p in phases_dir.iterdir() if not p.name.startswith("."))
    if not children:
        raise SystemExit(f"No phase entries found under {phases_dir}")

    volumes: list[np.ndarray] = []
    spacing: tuple[float, float, float] | None = None
    for child in children:
        target = _resolve_volume_target(child)
        vol = read_volume(target)
        volumes.append(vol.data.astype(np.float32))
        if spacing is None:
            spacing = tuple(float(v) for v in vol.spacing_xyz)
    assert spacing is not None
    if len(volumes) < 2:
        raise SystemExit(f"Need at least 2 phases, got {len(volumes)}")
    print(f"Loaded {len(volumes)} phases from {phases_dir}; spacing={spacing}; shape={volumes[0].shape}")
    return volumes, spacing


def _resolve_volume_target(entry: Path) -> Path:
    """Pick the path read_volume should consume for a given phase entry.

    A directory is returned as-is so read_volume can detect a DICOM series
    via SimpleITK; a single image file (NIfTI/MHA/.npy/.npz) is returned
    directly.
    """
    if entry.is_file():
        return entry
    files = [p for p in entry.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not files:
        raise SystemExit(f"Phase entry has no files: {entry}")
    image_files = [p for p in files if "".join(p.suffixes).lower().endswith(tuple(IMAGE_SUFFIXES))]
    if image_files:
        return sorted(image_files)[0]
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="TCIA 4D-Lung demo: 4DCT -> dynamic AP/lateral -> respiratory QA")
    parser.add_argument("--phases-dir", required=True, type=Path, help="Directory containing one subdir/file per phase")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for the report and artifacts")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--frames-per-cycle", type=int, default=60)
    parser.add_argument("--cycle-jitter-fraction", type=float, default=0.05)
    parser.add_argument("--intensity-jitter", type=float, default=0.01)
    parser.add_argument("--device", default="cpu", help="DiffDRR device: cpu, cuda, or mps")
    parser.add_argument("--detector-size", type=int, default=256, help="DRR detector size (square)")
    parser.add_argument("--pixel-spacing-mm", type=float, default=1.0)
    parser.add_argument("--ct-value-mode", default="hu", choices=("hu", "normalized_minus1_1_to_hu", "normalized_0_1_to_hu"))
    parser.add_argument("--min-period-s", type=float, default=2.0)
    parser.add_argument("--case-id", default=None, help="Case label for the report header")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes, spacing = load_phases(args.phases_dir)

    cfg = DynamicSimulationConfig(
        fps=args.fps,
        n_cycles=args.n_cycles,
        frames_per_cycle=args.frames_per_cycle,
        cycle_jitter_fraction=args.cycle_jitter_fraction,
        intensity_jitter=args.intensity_jitter,
        seed=args.seed,
    )
    detector_shape = (args.detector_size, args.detector_size)
    pixel_spacing_mm = (args.pixel_spacing_mm, args.pixel_spacing_mm)
    ap_geom = default_ap_geometry(detector_shape=detector_shape, pixel_spacing_mm=pixel_spacing_mm)
    lat_geom = default_lateral_geometry(detector_shape=detector_shape, pixel_spacing_mm=pixel_spacing_mm)

    print(f"Rendering DRRs and assembling dynamic sequence (fps={cfg.fps}, n_cycles={cfg.n_cycles})...")
    sim = simulate_from_4dct(
        volumes,
        spacing,
        config=cfg,
        ap_geometry=ap_geom,
        lateral_geometry=lat_geom,
        device=args.device,
        ct_value_mode=args.ct_value_mode,
    )
    print(f"  AP frames: {sim.ap_frames.shape}, Lateral frames: {sim.lateral_frames.shape}")

    print("Extracting respiratory signals (Amsterdam Shroud)...")
    ap_res = respiratory_signal(sim.ap_frames, fps=args.fps)
    lat_res = respiratory_signal(sim.lateral_frames, fps=args.fps)
    print(f"  AP  : f={ap_res.dominant_frequency_hz:.3f} Hz, period={ap_res.period_frames:.1f} frames")
    print(f"  Lat : f={lat_res.dominant_frequency_hz:.3f} Hz, period={lat_res.period_frames:.1f} frames")

    print("Pairing AP/lateral cycles by NCC...")
    pair = pair_best_correlated_cycles(
        ap_res.signal,
        lat_res.signal,
        ap_fps=args.fps,
        lateral_fps=args.fps,
        min_period_s=args.min_period_s,
    )
    print(
        f"  Best NCC = {pair.correlation:.3f}; "
        f"AP frames {pair.ap_start}-{pair.ap_end}, "
        f"Lateral frames {pair.lateral_start}-{pair.lateral_end}"
    )

    metrics = {
        "case_id": args.case_id or args.phases_dir.name,
        "fps": float(args.fps),
        "n_cycles": int(args.n_cycles),
        "frames_per_cycle": int(args.frames_per_cycle),
        "n_phases": int(sim.n_phases),
        "ap_dominant_frequency_hz": float(ap_res.dominant_frequency_hz),
        "lateral_dominant_frequency_hz": float(lat_res.dominant_frequency_hz),
        "best_pair_ncc": float(pair.correlation),
        "ap_cycle_frames": [int(pair.ap_start), int(pair.ap_end)],
        "lateral_cycle_frames": [int(pair.lateral_start), int(pair.lateral_end)],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    np.save(out_dir / "ap_cycle_frames.npy", sim.ap_frames[pair.ap_start : pair.ap_end])
    np.save(out_dir / "lateral_cycle_frames.npy", sim.lateral_frames[pair.lateral_start : pair.lateral_end])

    report_path = out_dir / "report.html"
    inputs = ReportInputs(
        case_id=metrics["case_id"],
        config_summary={
            "phases_dir": str(args.phases_dir),
            "fps": cfg.fps,
            "n_cycles": cfg.n_cycles,
            "frames_per_cycle": cfg.frames_per_cycle,
            "cycle_jitter_fraction": cfg.cycle_jitter_fraction,
            "intensity_jitter": cfg.intensity_jitter,
            "n_phases": sim.n_phases,
            "ct_shape": volumes[0].shape,
            "detector_shape": detector_shape,
            "pixel_spacing_mm": pixel_spacing_mm,
            "device": args.device,
            "ct_value_mode": args.ct_value_mode,
        },
        phase_volumes=volumes,
        spacing_xyz=spacing,
        ap_phase_drrs=sim.ap_phase_drrs,
        lateral_phase_drrs=sim.lateral_phase_drrs,
        ap_frames=sim.ap_frames,
        lateral_frames=sim.lateral_frames,
        phase_curve=sim.phase_indices,
        fps=args.fps,
        ap_shroud_result=ap_res,
        lateral_shroud_result=lat_res,
        pair_result=pair,
        extra_metrics={"min_period_s": args.min_period_s},
    )
    write_html_report(inputs, report_path)
    print(f"Wrote report: {report_path}")
    print(f"Wrote metrics: {out_dir / 'metrics.json'}")
    print(f"Wrote paired cycle arrays: ap_cycle_frames.npy, lateral_cycle_frames.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
