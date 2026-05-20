"""Compare lateral-signal extraction strategies across DIR-Lab cases.

Builds the synthetic dynamic sequence once per case, then re-runs the
respiratory-signal step on the lateral frames with several strategies. For
each (case, strategy) it pairs the lateral signal with the same AP signal
and records dominant frequency, NCC, and selected cycle frame ranges, plus
writes a per-strategy PDF/HTML report. A top-level ``index.html`` and
``comparison.json`` aggregate the matrix so it is easy to see which
strategy worked on which case.

Strategies
----------
- ``shroud_default``: Amsterdam Shroud, ``search_band=(0.3, 1.0)`` (baseline)
- ``shroud_narrow`` : Amsterdam Shroud, ``search_band=(0.5, 0.95)``
- ``shroud_lower``  : Amsterdam Shroud, ``search_band=(0.6, 0.9)``
- ``intensity_roi`` : mean intensity in a central lower ROI
- ``frame_diff``    : sum of absolute inter-frame differences in lower band
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from dvf_qa.amsterdam_shroud import (
    ShroudResult,
    respiratory_signal,
    respiratory_signal_frame_diff,
    respiratory_signal_intensity_roi,
)
from dvf_qa.cycle_pairing import pair_best_correlated_cycles
from dvf_qa.dirlab import detect_case_id, load_case_phases
from dvf_qa.pipeline_report import ReportInputs, write_html_report, write_pdf_report
from dvf_qa.synthetic_dynamic import (
    DynamicSimulationConfig,
    default_ap_geometry,
    default_lateral_geometry,
    simulate_from_4dct,
)


LATERAL_STRATEGIES: dict[str, Callable[[np.ndarray, float], ShroudResult]] = {
    "shroud_default": lambda f, fps: respiratory_signal(f, fps=fps, search_band=(0.3, 1.0)),
    "shroud_narrow":  lambda f, fps: respiratory_signal(f, fps=fps, search_band=(0.5, 0.95)),
    "shroud_lower":   lambda f, fps: respiratory_signal(f, fps=fps, search_band=(0.6, 0.9)),
    "intensity_roi":  lambda f, fps: respiratory_signal_intensity_roi(f, fps=fps),
    "frame_diff":     lambda f, fps: respiratory_signal_frame_diff(f, fps=fps),
}


def run_case_variants(case_dir: Path, out_dir: Path, args: argparse.Namespace) -> list[dict]:
    case_id = detect_case_id(case_dir)
    label = f"DIR-Lab-{case_id}"
    print(f"\n=== {label} ===")
    volumes, spacing, spec = load_case_phases(case_dir, case_id=case_id)

    cfg = DynamicSimulationConfig(
        fps=args.fps, n_cycles=args.n_cycles, frames_per_cycle=args.frames_per_cycle,
        cycle_jitter_fraction=args.cycle_jitter_fraction, intensity_jitter=args.intensity_jitter,
        seed=args.seed,
    )
    detector_shape = (args.detector_size, args.detector_size)
    pixel_spacing_mm = (args.pixel_spacing_mm, args.pixel_spacing_mm)
    sim = simulate_from_4dct(
        volumes, spacing, config=cfg,
        ap_geometry=default_ap_geometry(detector_shape, pixel_spacing_mm),
        lateral_geometry=default_lateral_geometry(detector_shape, pixel_spacing_mm),
        device=args.device, ct_value_mode="hu",
    )

    ap_res = respiratory_signal(sim.ap_frames, fps=args.fps)
    print(f"  AP  f={ap_res.dominant_frequency_hz:.3f} Hz")

    case_rows: list[dict] = []
    case_out = out_dir / spec.case_id
    case_out.mkdir(parents=True, exist_ok=True)
    for strategy_name in args.strategies:
        builder = LATERAL_STRATEGIES[strategy_name]
        try:
            lat_res = builder(sim.lateral_frames, args.fps)
        except Exception as exc:
            print(f"  [{strategy_name}] lateral extraction failed: {exc}")
            case_rows.append({
                "case_id": label, "strategy": strategy_name,
                "lat_freq_hz": None, "ncc": None,
                "ap_cycle_frames": None, "lateral_cycle_frames": None,
                "error": f"signal: {exc}",
            })
            continue
        try:
            pair = pair_best_correlated_cycles(
                ap_res.signal, lat_res.signal,
                ap_fps=args.fps, lateral_fps=args.fps,
                min_period_s=args.min_period_s,
            )
        except Exception as exc:
            print(f"  [{strategy_name}] pair failed: {exc}")
            case_rows.append({
                "case_id": label, "strategy": strategy_name,
                "lat_freq_hz": float(lat_res.dominant_frequency_hz),
                "ncc": None,
                "ap_cycle_frames": None, "lateral_cycle_frames": None,
                "error": f"pair: {exc}",
            })
            continue

        row = {
            "case_id": label, "strategy": strategy_name,
            "lat_freq_hz": float(lat_res.dominant_frequency_hz),
            "ncc": float(pair.correlation),
            "ap_cycle_frames": [int(pair.ap_start), int(pair.ap_end)],
            "lateral_cycle_frames": [int(pair.lateral_start), int(pair.lateral_end)],
        }
        print(f"  [{strategy_name}] Lat f={row['lat_freq_hz']:.3f} Hz  NCC={row['ncc']:.3f}")

        strategy_dir = case_out / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        (strategy_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        inputs = ReportInputs(
            case_id=f"{label} [{strategy_name}]",
            config_summary={
                "source": case_dir.name, "lateral_strategy": strategy_name,
                "fps": cfg.fps, "n_cycles": cfg.n_cycles,
                "frames_per_cycle": cfg.frames_per_cycle, "n_phases": sim.n_phases,
                "ct_shape": volumes[0].shape, "spacing_xyz_mm": spacing,
                "detector_shape": detector_shape, "pixel_spacing_mm": pixel_spacing_mm,
            },
            phase_volumes=volumes, spacing_xyz=spacing,
            ap_phase_drrs=sim.ap_phase_drrs, lateral_phase_drrs=sim.lateral_phase_drrs,
            ap_frames=sim.ap_frames, lateral_frames=sim.lateral_frames,
            phase_curve=sim.phase_indices, fps=args.fps,
            ap_shroud_result=ap_res, lateral_shroud_result=lat_res, pair_result=pair,
        )
        write_pdf_report(inputs, strategy_dir / "report.pdf")
        write_html_report(inputs, strategy_dir / "report.html")
        case_rows.append(row)
    return case_rows


def write_comparison_index(out_dir: Path, rows: list[dict], strategies: list[str]) -> None:
    by_case: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["strategy"]] = r

    head = "<th>Case</th>" + "".join(
        f"<th colspan='2'>{s}<br/><span class='small'>Lat Hz / NCC</span></th>" for s in strategies
    )
    body_rows = []
    for case_id, strat_map in sorted(by_case.items()):
        cells = [f"<td><strong>{case_id}</strong></td>"]
        for s in strategies:
            r = strat_map.get(s)
            if r is None or r.get("error"):
                err = (r.get("error") if r else "missing") or "missing"
                cells.append(f"<td colspan='2' class='fail'>{err}</td>")
                continue
            ncc = r["ncc"]
            cls = "good" if ncc and ncc >= 0.9 else ("warn" if ncc and ncc >= 0.5 else "fail")
            short = r["case_id"].split("-")[-1]
            link = f"{short}/{s}/report.pdf"
            cells.append(f"<td class='{cls}'>{r['lat_freq_hz']:.3f}</td>")
            cells.append(f"<td class='{cls}'><a href='{link}'>{ncc:.3f}</a></td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>DIR-Lab lateral-strategy comparison</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #eee; }}
.good {{ background: #d4f5d4; }}
.warn {{ background: #fff3cd; }}
.fail {{ background: #fbd1d1; }}
.small {{ font-weight: normal; font-size: 11px; color: #666; }}
</style></head>
<body>
<h1>DIR-Lab lateral-signal strategy comparison</h1>
<p>Each cell shows the dominant lateral frequency (Hz) and the AP–lateral best-pair NCC.
Click an NCC value to open the per-strategy PDF report.</p>
<table>
<thead><tr>{head}</tr></thead>
<tbody>
{chr(10).join(body_rows)}
</tbody></table>
</body></html>"""
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare lateral signal strategies on DIR-Lab cases")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--strategies", nargs="*",
                        default=list(LATERAL_STRATEGIES.keys()),
                        choices=list(LATERAL_STRATEGIES.keys()))
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

    case_dirs = sorted(p for p in args.root.iterdir() if p.is_dir() and p.name.lower().startswith("case"))
    if args.cases:
        chosen = set(args.cases)
        case_dirs = [d for d in case_dirs if d.name in chosen]
    if not case_dirs:
        print(f"No CaseNPack dirs under {args.root}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for case_dir in case_dirs:
        try:
            all_rows.extend(run_case_variants(case_dir, args.out_dir, args))
        except Exception:
            print(f"FAILED case {case_dir.name}:")
            traceback.print_exc()
    if all_rows:
        write_comparison_index(args.out_dir, all_rows, args.strategies)
        print(f"\nIndex: {args.out_dir / 'index.html'}")
        print(f"Comparison: {args.out_dir / 'comparison.json'}")
    return 0 if all_rows else 1


if __name__ == "__main__":
    sys.exit(main())
