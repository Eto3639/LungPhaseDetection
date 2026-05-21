"""Respiratory phase QA on anonymized dynamic chest X-ray DICOMs.

Reads each multi-frame DICOM (one recording per case from the Konica DDR
system), extracts the frame stack, runs the Amsterdam Shroud respiratory
signal extractor, detects per-cycle boundaries, and emits a per-recording
PDF + HTML report plus an aggregated index.

Assumes the DICOMs have already been anonymized (PHI stripped, new UIDs).
Use ``scripts/anonymize_dicom.py`` first.

Usage::

    python scripts/dynamic_xray_phase_qa.py \\
        --root DynamicChestXray_anon \\
        --out-dir outputs/dynamic_xray_phase_qa
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html as html_mod
import io
import json
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.backends.backend_pdf import PdfPages

from dvf_qa.amsterdam_shroud import respiratory_signal
from dvf_qa.cycle_pairing import detect_cycle_boundaries


def load_frames_from_dicom(path: Path) -> tuple[np.ndarray, float, dict]:
    """Return ``(frames_TxHxW, fps, metadata)`` from a multi-frame DICOM."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    # Apply rescale if present
    slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
    intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    # MONOCHROME1: invert so air = low (dark) for our analysis convention
    photometric = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    frame_time_ms = float(ds.get("FrameTime", 0.0) or 0.0)
    fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 15.0
    metadata = {
        "n_frames": int(arr.shape[0]),
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "fps": float(fps),
        "frame_time_ms": float(frame_time_ms),
        "photometric_interpretation": photometric,
        "series_description": str(ds.get("SeriesDescription", "")),
        "anon_id": str(ds.get("PatientID", "")),
    }
    return arr, fps, metadata


# ---------- figures ----------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_mean_frame(frames: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(frames.mean(axis=0), cmap="gray")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return fig


def fig_shroud_signal(result, fps: float, boundaries: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].imshow(result.shroud, aspect="auto", cmap="gray", origin="upper")
    axes[0].plot(np.arange(result.diaphragm_row.size), result.diaphragm_row,
                 color="tab:red", lw=1.0)
    axes[0].set_title("Shroud + tracked diaphragm")
    axes[0].set_xlabel("Frame"); axes[0].set_ylabel("CC position (px)")

    t = np.arange(result.signal.size) / fps
    axes[1].plot(t, result.signal, color="tab:blue", lw=1.0)
    for b in boundaries:
        if 0 <= b < t.size:
            axes[1].axvline(t[b], color="tab:red", alpha=0.5, lw=0.7,
                            linestyle="--")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Bandpassed signal")
    axes[1].set_title(f"Respiratory signal | f={result.dominant_frequency_hz:.3f} Hz, "
                       f"{len(boundaries)} cycle boundaries")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def fig_cycle_frames(frames: np.ndarray, result, boundaries: np.ndarray):
    """Show key frames: end-expiration valleys + end-inspiration peak (max signal)."""
    if boundaries.size < 1:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No cycle boundaries detected", ha="center", va="center",
                fontsize=14, color="orange")
        return fig
    end_insp = int(np.argmax(result.signal))
    sample_frames = sorted(set([int(boundaries[0]), end_insp,
                                 int(boundaries[-1])]))
    sample_frames = sample_frames[:4]
    n = len(sample_frames)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, sample_frames):
        ax.imshow(frames[t], cmap="gray")
        ax.set_title(f"frame {t} ({'end-exp' if t in boundaries else 'end-insp'})")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return fig


# ---------- per-recording ----------

def run_recording(path: Path, out_dir: Path) -> dict:
    print(f"\n--- {path.name} ---")
    frames, fps, meta = load_frames_from_dicom(path)
    print(f"  {meta['n_frames']} frames, {meta['height']}x{meta['width']}, fps={fps:.2f}")

    result = respiratory_signal(frames, fps=fps)
    boundaries = detect_cycle_boundaries(result.signal, fps, min_period_s=2.0,
                                         prominence_factor=0.2)
    print(f"  dominant freq: {result.dominant_frequency_hz:.3f} Hz, "
          f"period: {result.period_frames:.1f} frames")
    print(f"  cycle boundaries: {len(boundaries)} valleys")
    metrics = {
        "anon_id": meta["anon_id"],
        "n_frames": meta["n_frames"],
        "fps": fps,
        "series_description": meta["series_description"],
        "dominant_frequency_hz": float(result.dominant_frequency_hz),
        "period_frames": float(result.period_frames),
        "n_cycles_detected": int(len(boundaries)),
        "cycle_boundary_frames": [int(b) for b in boundaries],
        "end_inspiration_frame": int(np.argmax(result.signal)),
    }

    rec_out = out_dir / meta["anon_id"]
    rec_out.mkdir(parents=True, exist_ok=True)
    (rec_out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figs = {
        "Mean frame":   fig_mean_frame(frames, meta["anon_id"]),
        "Shroud + signal": fig_shroud_signal(result, fps, boundaries),
        "Cycle key frames": fig_cycle_frames(frames, result, boundaries),
    }

    write_pdf(rec_out / "report.pdf", meta, metrics, figs)
    figs_for_html = {
        "Mean frame":   fig_mean_frame(frames, meta["anon_id"]),
        "Shroud + signal": fig_shroud_signal(result, fps, boundaries),
        "Cycle key frames": fig_cycle_frames(frames, result, boundaries),
    }
    write_html(rec_out / "report.html", meta, metrics, figs_for_html)
    return metrics


def write_pdf(path: Path, meta: dict, metrics: dict, figs: dict) -> None:
    with PdfPages(path) as pdf:
        # Cover with metrics
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(0.5, 0.97, f"Respiratory phase QA — {meta['anon_id']}",
                ha="center", va="top", fontsize=18, fontweight="bold")
        ax.text(0.5, 0.92, meta["series_description"], ha="center", va="top",
                fontsize=12, color="#555")
        rows = [
            ("Frames", str(meta["n_frames"])),
            ("FPS", f"{meta['fps']:.2f}"),
            ("Duration (s)", f"{meta['n_frames'] / meta['fps']:.2f}"),
            ("Dominant freq (Hz)", f"{metrics['dominant_frequency_hz']:.4f}"),
            ("Period (frames)", f"{metrics['period_frames']:.1f}"),
            ("Period (s)", f"{metrics['period_frames'] / meta['fps']:.2f}"),
            ("Detected cycle boundaries", str(metrics["n_cycles_detected"])),
            ("End-inspiration frame", str(metrics["end_inspiration_frame"])),
        ]
        table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                         cellLoc="left", colLoc="left", loc="center",
                         colWidths=[0.55, 0.4])
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 1.6)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for title, fig in figs.items():
            fig.suptitle(title, fontsize=12, y=1.02)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def write_html(path: Path, meta: dict, metrics: dict, figs: dict) -> None:
    sections = [
        f"<header><h1>Respiratory phase QA — {html_mod.escape(meta['anon_id'])}</h1>"
        f"<p>{html_mod.escape(meta['series_description'])}</p>"
        f"<p><b>{meta['n_frames']} frames</b> @ {meta['fps']:.2f} fps "
        f"({meta['n_frames']/meta['fps']:.2f} s)</p>"
        f"<p>Dominant freq: <b>{metrics['dominant_frequency_hz']:.4f} Hz</b>, "
        f"period: <b>{metrics['period_frames']:.1f} frames</b> "
        f"({metrics['period_frames']/meta['fps']:.2f} s), "
        f"boundaries detected: <b>{metrics['n_cycles_detected']}</b></p></header>"
    ]
    for title, fig in figs.items():
        b64 = _fig_to_b64(fig)
        sections.append(f"<section><h2>{html_mod.escape(title)}</h2>"
                        f"<img src='data:image/png;base64,{b64}'/></section>")
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="robots" content="noindex,nofollow">
<title>Respiratory phase QA — {html_mod.escape(meta['anon_id'])}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-top: 16px; }} h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
img {{ max-width: 100%; height: auto; }}
section {{ margin-bottom: 16px; }}
</style></head><body>
{chr(10).join(sections)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def write_top_index(out_dir: Path, results: list[dict]) -> None:
    rows = []
    for r in sorted(results, key=lambda x: x["anon_id"]):
        rid = r["anon_id"]
        rows.append(
            f"<tr><td><a href='{rid}/report.html'>{html_mod.escape(rid)}</a></td>"
            f"<td><a href='{rid}/report.pdf'>PDF</a></td>"
            f"<td>{html_mod.escape(r['series_description'])}</td>"
            f"<td>{r['n_frames']}</td>"
            f"<td>{r['fps']:.2f}</td>"
            f"<td>{r['dominant_frequency_hz']:.4f}</td>"
            f"<td>{r['period_frames']/r['fps']:.2f}</td>"
            f"<td>{r['n_cycles_detected']}</td></tr>"
        )
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>Dynamic chest X-ray phase QA</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #eee; }}
</style></head>
<body>
<h1>Dynamic chest X-ray — respiratory phase QA</h1>
<p>Anonymized recordings analyzed with Amsterdam Shroud + valley-based cycle
detection. Each row links to the per-recording PDF / HTML.</p>
<table>
<tr><th>Recording</th><th>PDF</th><th>SeriesDescription</th>
<th>Frames</th><th>FPS</th><th>f (Hz)</th><th>Period (s)</th><th>#cycles</th></tr>
{chr(10).join(rows)}
</table>
</body></html>"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory of anonymized DICOMs (CaseNN/*.dcm)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Optional cap (smoke testing)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dicoms = sorted(p for p in args.root.rglob("*.dcm"))
    if args.limit:
        dicoms = dicoms[: args.limit]
    if not dicoms:
        print(f"No .dcm files under {args.root}", file=sys.stderr); return 2

    results: list[dict] = []
    for path in dicoms:
        try:
            results.append(run_recording(path, args.out_dir))
        except Exception:
            print(f"FAILED {path.name}:")
            traceback.print_exc()

    if results:
        write_top_index(args.out_dir, results)
        print(f"\nIndex: {args.out_dir / 'index.html'}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
