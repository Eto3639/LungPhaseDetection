from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def write_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def write_overlay_report(
    path: str | Path,
    ct: np.ndarray | None,
    jacobian: np.ndarray | None,
    drr: np.ndarray | None,
    fluoro: np.ndarray | None,
    metrics: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels = []
    if ct is not None:
        panels.append(("CT mid-slice", _mid_slice(ct), "gray", None, None))
    if jacobian is not None:
        panels.append(("Jacobian mid-slice", _mid_slice(jacobian), "coolwarm", 0.0, 2.0))
    if drr is not None:
        panels.append(("Generated CT DRR", drr, "gray", None, None))
    if fluoro is not None:
        panels.append(("Input fluoroscopy", fluoro, "gray", None, None))
    if drr is not None and fluoro is not None:
        panels.append(("DRR - fluoroscopy", _normalize(drr) - _normalize(fluoro), "coolwarm", -0.5, 0.5))

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for ax, (title, image, cmap, vmin, vmax) in zip(axes.ravel(), panels, strict=True):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"DVF QA: {metrics.get('qa_status', 'N/A')}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _mid_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]


def _normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)

