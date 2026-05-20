"""DIR-Lab 4DCT demo: loop over CaseNPack directories, run the full pipeline.

For each case directory it:
  1. loads the 10-phase 4DCT (DIR-Lab raw int16 .img format),
  2. renders AP + lateral DRRs and assembles a dynamic sequence via
     :mod:`dvf_qa.synthetic_dynamic`,
  3. extracts respiratory signals (Amsterdam Shroud) for both views,
  4. picks the best correlated AP/lateral one-breath pair,
  5. writes per-case ``report.html`` + ``report.pdf`` + ``metrics.json``
     + selected-cycle ``.npy`` arrays,
  6. emits a top-level ``index.html`` + ``summary.json`` aggregating cases.

Usage
-----
::

    python scripts/dirlab_demo.py \\
        --root outputs/dirlab_data \\
        --out-dir outputs/dirlab_demo \\
        --fps 15 --n-cycles 4 --device cpu

Each case is independent; failures on one case do not stop the others.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from dvf_qa.amsterdam_shroud import respiratory_signal
from dvf_qa.cycle_pairing import pair_best_correlated_cycles
from dvf_qa.dirlab import detect_case_id, load_case_phases
from dvf_qa.pipeline_report import ReportInputs, write_html_report, write_pdf_report
from dvf_qa.synthetic_dynamic import (
    DynamicSimulationConfig,
    default_ap_geometry,
    default_lateral_geometry,
    simulate_from_4dct,
)


def find_case_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("case"))


def run_case(case_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict:
    case_id = detect_case_id(case_dir)
    label = f"DIR-Lab-{case_id}"
    print(f"\n=== {label}  ({case_dir.name}) ===")
    volumes, spacing, spec = load_case_phases(case_dir, case_id=case_id)
    print(f"  loaded {len(volumes)} phases, shape={volumes[0].shape}, spacing={spacing}")

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

    sim = simulate_from_4dct(
        volumes, spacing,
        config=cfg, ap_geometry=ap_geom, lateral_geometry=lat_geom,
        device=args.device, ct_value_mode="hu",
    )
    ap_res = respiratory_signal(sim.ap_frames, fps=args.fps)
    lat_res = respiratory_signal(sim.lateral_frames, fps=args.fps)
    print(f"  AP  f={ap_res.dominant_frequency_hz:.3f} Hz  Lat f={lat_res.dominant_frequency_hz:.3f} Hz")
    pair = pair_best_correlated_cycles(
        ap_res.signal, lat_res.signal,
        ap_fps=args.fps, lateral_fps=args.fps,
        min_period_s=args.min_period_s,
    )
    print(f"  NCC = {pair.correlation:.3f}  AP[{pair.ap_start}:{pair.ap_end}] Lat[{pair.lateral_start}:{pair.lateral_end}]")

    case_out = out_dir / spec.case_id
    case_out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "case_id": label,
        "source": str(case_dir),
        "fps": float(args.fps),
        "n_cycles": int(args.n_cycles),
        "frames_per_cycle": int(args.frames_per_cycle),
        "n_phases": int(sim.n_phases),
        "ct_shape": list(volumes[0].shape),
        "spacing_xyz_mm": list(spacing),
        "ap_dominant_frequency_hz": float(ap_res.dominant_frequency_hz),
        "lateral_dominant_frequency_hz": float(lat_res.dominant_frequency_hz),
        "best_pair_ncc": float(pair.correlation),
        "ap_cycle_frames": [int(pair.ap_start), int(pair.ap_end)],
        "lateral_cycle_frames": [int(pair.lateral_start), int(pair.lateral_end)],
    }
    (case_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.save(case_out / "ap_cycle_frames.npy", sim.ap_frames[pair.ap_start : pair.ap_end])
    np.save(case_out / "lateral_cycle_frames.npy", sim.lateral_frames[pair.lateral_start : pair.lateral_end])

    inputs = ReportInputs(
        case_id=label,
        config_summary={
            "source": case_dir.name,
            "fps": cfg.fps,
            "n_cycles": cfg.n_cycles,
            "frames_per_cycle": cfg.frames_per_cycle,
            "n_phases": sim.n_phases,
            "ct_shape": volumes[0].shape,
            "spacing_xyz_mm": spacing,
            "detector_shape": detector_shape,
            "pixel_spacing_mm": pixel_spacing_mm,
            "device": args.device,
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
    write_html_report(inputs, case_out / "report.html")
    write_pdf_report(inputs, case_out / "report.pdf")
    print(f"  wrote {case_out / 'report.pdf'}")
    return metrics


def write_index(out_dir: Path, results: list[dict]) -> None:
    rows_html = "\n".join(
        f"<tr><td><a href='{r['case_id'].split('-')[-1]}/report.html'>{r['case_id']}</a></td>"
        f"<td><a href='{r['case_id'].split('-')[-1]}/report.pdf'>PDF</a></td>"
        f"<td>{r['ap_dominant_frequency_hz']:.3f}</td>"
        f"<td>{r['lateral_dominant_frequency_hz']:.3f}</td>"
        f"<td>{r['best_pair_ncc']:.3f}</td>"
        f"<td>{r['ap_cycle_frames'][0]}-{r['ap_cycle_frames'][1]}</td>"
        f"<td>{r['lateral_cycle_frames'][0]}-{r['lateral_cycle_frames'][1]}</td></tr>"
        for r in results
    )
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>DIR-Lab pipeline runs</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #eee; }}
</style></head>
<body>
<h1>DIR-Lab pipeline runs</h1>
<p>Cases processed: {len(results)}</p>
<table>
<tr><th>Case</th><th>PDF</th><th>AP f (Hz)</th><th>Lat f (Hz)</th><th>NCC</th><th>AP cycle</th><th>Lat cycle</th></tr>
{rows_html}
</table>
</body></html>"""
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DVF QA pipeline on DIR-Lab 4DCT cases")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing CaseNPack/ subfolders")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="Optional subset of CaseNPack dirnames to run (default: all)")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--n-cycles", type=int, default=4)
    parser.add_argument("--frames-per-cycle", type=int, default=30)
    parser.add_argument("--cycle-jitter-fraction", type=float, default=0.05)
    parser.add_argument("--intensity-jitter", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector-size", type=int, default=256)
    parser.add_argument("--pixel-spacing-mm", type=float, default=1.0)
    parser.add_argument("--min-period-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    case_dirs = find_case_dirs(args.root)
    if args.cases:
        chosen = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in chosen]
    if not case_dirs:
        print(f"No CaseNPack directories under {args.root}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for case_dir in case_dirs:
        try:
            results.append(run_case(case_dir, args.out_dir, args))
        except Exception:
            print(f"FAILED {case_dir.name}:")
            traceback.print_exc()
    if results:
        write_index(args.out_dir, results)
        print(f"\nIndex: {args.out_dir / 'index.html'}")
        print(f"Summary: {args.out_dir / 'summary.json'}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
