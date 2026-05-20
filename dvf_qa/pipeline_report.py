"""Self-contained HTML report for the 4DCT -> dynamic -> phase pipeline.

Bundles 4DCT preview, per-phase DRRs, dynamic sample frames, Amsterdam Shroud
output per view, cycle-pairing diagnostics, and a metrics table into a single
HTML file with base64-embedded matplotlib figures. No external assets.

Designed to be invoked at the end of a test run or a TCIA demo so reviewers
can inspect the full chain without launching a notebook.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReportInputs:
    case_id: str
    config_summary: dict[str, object]
    phase_volumes: list[np.ndarray] | None
    spacing_xyz: tuple[float, float, float] | None
    ap_phase_drrs: np.ndarray
    lateral_phase_drrs: np.ndarray
    ap_frames: np.ndarray
    lateral_frames: np.ndarray
    phase_curve: np.ndarray
    fps: float
    ap_shroud_result: object
    lateral_shroud_result: object
    pair_result: object
    extra_metrics: dict[str, float | str] | None = None


def write_html_report(inputs: ReportInputs, path: str | Path) -> Path:
    """Render the full pipeline report as a self-contained HTML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    sections.append(_header(inputs))
    sections.append(_config_section(inputs))

    for title, fig in _build_figure_sections(inputs):
        sections.append(_section(title, fig))
    sections.append(_metrics_section(inputs))

    html_text = _wrap_html(inputs.case_id, "\n".join(sections))
    path.write_text(html_text, encoding="utf-8")
    return path


def write_pdf_report(inputs: ReportInputs, path: str | Path) -> Path:
    """Render the full pipeline report as a multi-page PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(path) as pdf:
        pdf.savefig(_cover_fig(inputs), bbox_inches="tight")
        plt.close("all")
        for title, fig in _build_figure_sections(inputs):
            fig.suptitle(title, fontsize=14, y=1.02)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf.savefig(_metrics_fig(inputs), bbox_inches="tight")
        plt.close("all")
        meta = pdf.infodict()
        meta["Title"] = f"DVF QA pipeline report — {inputs.case_id}"
        meta["Author"] = "dvf-qa"
        meta["Subject"] = "4DCT -> dynamic AP/lateral -> respiratory phase QA"
        meta["CreationDate"] = datetime.now()
    return path


def _build_figure_sections(inputs: ReportInputs) -> list[tuple[str, "object"]]:
    sections: list[tuple[str, object]] = []
    if inputs.phase_volumes is not None:
        sections.append(("4DCT phase preview", _phase_slices_fig(inputs.phase_volumes)))
    sections.append(("Per-phase DRRs", _per_phase_drr_fig(inputs.ap_phase_drrs, inputs.lateral_phase_drrs)))
    sections.append(("Dynamic sequence samples", _dynamic_samples_fig(inputs.ap_frames, inputs.lateral_frames, inputs.fps)))
    sections.append(("Driving phase curve", _phase_curve_fig(inputs.phase_curve, inputs.fps)))
    sections.append(("Amsterdam Shroud — AP", _shroud_fig(inputs.ap_shroud_result, "AP", inputs.fps)))
    sections.append(("Amsterdam Shroud — Lateral", _shroud_fig(inputs.lateral_shroud_result, "Lateral", inputs.fps)))
    sections.append(("Cycle pairing", _pair_fig(inputs)))
    return sections


def _cover_fig(inputs: ReportInputs):
    import matplotlib.pyplot as plt

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        ("Case", inputs.case_id),
        ("Generated", now),
        *[(str(k), str(v)) for k, v in inputs.config_summary.items()],
    ]
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.95, "DVF QA pipeline report", ha="center", va="top", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.90, inputs.case_id, ha="center", va="top", fontsize=14)
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Field", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.3, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    return fig


def _metrics_fig(inputs: ReportInputs):
    import matplotlib.pyplot as plt

    metrics = _metrics_dict(inputs)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.5, 0.95, "Metrics", ha="center", va="top", fontsize=20, fontweight="bold")
    table = ax.table(
        cellText=[[k, v] for k, v in metrics.items()],
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.55, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    return fig


def _metrics_dict(inputs: ReportInputs) -> dict[str, str]:
    ap_res = inputs.ap_shroud_result
    lat_res = inputs.lateral_shroud_result
    pair = inputs.pair_result
    metrics: dict[str, str] = {
        "AP dominant frequency (Hz)": f"{getattr(ap_res, 'dominant_frequency_hz', float('nan')):.4f}",
        "AP period (frames)": f"{getattr(ap_res, 'period_frames', float('nan')):.2f}",
        "Lateral dominant frequency (Hz)": f"{getattr(lat_res, 'dominant_frequency_hz', float('nan')):.4f}",
        "Lateral period (frames)": f"{getattr(lat_res, 'period_frames', float('nan')):.2f}",
        "Best pair NCC": f"{getattr(pair, 'correlation', float('nan')):.4f}",
        "AP cycle frames": f"{getattr(pair, 'ap_start', '?')} - {getattr(pair, 'ap_end', '?')}",
        "Lateral cycle frames": f"{getattr(pair, 'lateral_start', '?')} - {getattr(pair, 'lateral_end', '?')}",
    }
    if inputs.extra_metrics:
        for k, v in inputs.extra_metrics.items():
            metrics[str(k)] = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    return metrics


def _header(inputs: ReportInputs) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"<header><h1>DVF QA pipeline report</h1>"
        f"<p><b>Case:</b> {html.escape(inputs.case_id)}</p>"
        f"<p><b>Generated:</b> {html.escape(now)}</p></header>"
    )


def _config_section(inputs: ReportInputs) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in inputs.config_summary.items()
    )
    return f"<section><h2>Configuration</h2><table>{rows}</table></section>"


def _metrics_section(inputs: ReportInputs) -> str:
    metrics = _metrics_dict(inputs)
    rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
        for k, v in metrics.items()
    )
    return f"<section><h2>Metrics</h2><table>{rows}</table></section>"


def _phase_slices_fig(phase_volumes):
    import matplotlib.pyplot as plt

    n = len(phase_volumes)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            vol = phase_volumes[i]
            ax.imshow(vol[vol.shape[0] // 2], cmap="gray")
            ax.set_title(f"φ{i}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def _per_phase_drr_fig(ap_phase_drrs, lateral_phase_drrs):
    import matplotlib.pyplot as plt

    n = ap_phase_drrs.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.5), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(ap_phase_drrs[i], cmap="gray")
        axes[0, i].set_title(f"AP φ{i}")
        axes[1, i].imshow(lateral_phase_drrs[i], cmap="gray")
        axes[1, i].set_title(f"Lat φ{i}")
        for r in range(2):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    fig.tight_layout()
    return fig


def _dynamic_samples_fig(ap_frames, lateral_frames, fps, n_samples=6):
    import matplotlib.pyplot as plt

    T = ap_frames.shape[0]
    sample_t = np.linspace(0, T - 1, min(n_samples, T)).astype(int)
    n = sample_t.size
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.5), squeeze=False)
    for i, t in enumerate(sample_t):
        axes[0, i].imshow(ap_frames[t], cmap="gray")
        axes[0, i].set_title(f"AP t={t} ({t/fps:.2f}s)")
        axes[1, i].imshow(lateral_frames[t], cmap="gray")
        axes[1, i].set_title(f"Lat t={t}")
        for r in range(2):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    fig.tight_layout()
    return fig


def _phase_curve_fig(phase_curve, fps):
    import matplotlib.pyplot as plt

    t = np.arange(phase_curve.size) / fps
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, phase_curve, lw=1.0, color="tab:purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase index")
    ax.set_title("Driving phase curve (ground-truth respiratory motion)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _shroud_fig(result, view, fps):
    import matplotlib.pyplot as plt

    shroud = getattr(result, "shroud")
    diaphragm = getattr(result, "diaphragm_row")
    signal = getattr(result, "signal")
    freq = getattr(result, "dominant_frequency_hz", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].imshow(shroud, aspect="auto", cmap="gray", origin="upper")
    axes[0].plot(np.arange(diaphragm.size), diaphragm, color="tab:red", lw=1.0)
    axes[0].set_title(f"{view} shroud + tracked diaphragm")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("CC position (px)")

    t = np.arange(signal.size) / fps
    axes[1].plot(t, signal, color="tab:blue", lw=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Bandpassed signal")
    axes[1].set_title(f"{view} respiratory signal  |  f={freq:.3f} Hz")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _pair_fig(inputs):
    import matplotlib.pyplot as plt

    pair = inputs.pair_result
    ap_sig = getattr(inputs.ap_shroud_result, "signal")
    lat_sig = getattr(inputs.lateral_shroud_result, "signal")

    all_pairs = getattr(pair, "all_pairs", [])
    ap_starts = sorted({p["ap_start"] for p in all_pairs})
    lat_starts = sorted({p["lateral_start"] for p in all_pairs})
    if ap_starts and lat_starts:
        matrix = np.full((len(ap_starts), len(lat_starts)), np.nan)
        for p in all_pairs:
            i = ap_starts.index(p["ap_start"])
            j = lat_starts.index(p["lateral_start"])
            matrix[i, j] = p["correlation"]
    else:
        matrix = np.zeros((1, 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(np.arange(ap_sig.size) / inputs.fps, ap_sig, color="tab:gray", lw=0.8)
    axes[0, 0].axvspan(pair.ap_start / inputs.fps, pair.ap_end / inputs.fps, color="tab:blue", alpha=0.2)
    axes[0, 0].set_title("AP signal (selected cycle highlighted)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(np.arange(lat_sig.size) / inputs.fps, lat_sig, color="tab:gray", lw=0.8)
    axes[0, 1].axvspan(pair.lateral_start / inputs.fps, pair.lateral_end / inputs.fps, color="tab:orange", alpha=0.2)
    axes[0, 1].set_title("Lateral signal (selected cycle highlighted)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].grid(True, alpha=0.3)

    im = axes[1, 0].imshow(matrix, cmap="viridis", aspect="auto", origin="upper")
    axes[1, 0].set_xlabel("Lateral cycle index")
    axes[1, 0].set_ylabel("AP cycle index")
    axes[1, 0].set_title(f"NCC matrix (best={pair.correlation:.3f})")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    x = np.linspace(0.0, 1.0, pair.resample_length)
    axes[1, 1].plot(x, pair.ap_resampled, label="AP", color="tab:blue")
    axes[1, 1].plot(x, pair.lateral_resampled, label="Lateral", color="tab:orange")
    axes[1, 1].set_xlabel("Cycle fraction")
    axes[1, 1].set_ylabel("z-scored signal")
    axes[1, 1].set_title("Selected cycle overlay")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    return fig


def _section(title: str, fig) -> str:
    return f"<section><h2>{html.escape(title)}</h2>{_fig_to_img_tag(fig)}</section>"


def _fig_to_img_tag(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" />'


def _wrap_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>DVF QA report - {html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 24px; color: #222; }}
  h1 {{ margin: 0 0 8px 0; }}
  h2 {{ margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin: 8px 0 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
  th {{ background: #f6f6f6; }}
  img {{ max-width: 100%; height: auto; }}
  section {{ margin-bottom: 16px; }}
  header p {{ margin: 2px 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
