from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from .jacobian import bending_energy, gradient_norm, jacobian_determinant


@dataclass(frozen=True)
class Thresholds:
    jac_fail_fraction_le_0: float = 0.0
    jac_warn_fraction_lt_0_2: float = 1e-4
    jac_warn_fraction_gt_5: float = 1e-4
    largest_folding_component_ml_fail: float = 0.01


def summarize_dvf_qa(
    dvf: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    mask: np.ndarray | None = None,
    thresholds: Thresholds = Thresholds(),
) -> tuple[dict[str, float | str], np.ndarray]:
    jac = jacobian_determinant(dvf, spacing_xyz)
    roi = _roi(mask, jac.shape)
    voxel_ml = float(np.prod(spacing_xyz) / 1000.0)

    selected_jac = jac[roi]
    mag = np.linalg.norm(dvf, axis=-1)[roi]
    grad = gradient_norm(dvf, spacing_xyz)[roi]
    bend = bending_energy(dvf, spacing_xyz)[roi]

    folding = (jac <= 0) & roi
    labeled, n_components = label(folding)
    largest_component_voxels = 0
    if n_components:
        counts = np.bincount(labeled.ravel())
        largest_component_voxels = int(counts[1:].max())

    metrics: dict[str, float | str] = {
        "jac_min": float(np.min(selected_jac)),
        "jac_p01": float(np.percentile(selected_jac, 1)),
        "jac_p05": float(np.percentile(selected_jac, 5)),
        "jac_median": float(np.median(selected_jac)),
        "jac_mean": float(np.mean(selected_jac)),
        "jac_p95": float(np.percentile(selected_jac, 95)),
        "jac_p99": float(np.percentile(selected_jac, 99)),
        "jac_max": float(np.max(selected_jac)),
        "fraction_jac_le_0": float(np.mean(selected_jac <= 0)),
        "fraction_jac_lt_0_2": float(np.mean(selected_jac < 0.2)),
        "fraction_jac_gt_5": float(np.mean(selected_jac > 5.0)),
        "folding_component_count": float(n_components),
        "largest_folding_component_ml": float(largest_component_voxels * voxel_ml),
        "dvf_magnitude_p95_mm": float(np.percentile(mag, 95)),
        "dvf_magnitude_max_mm": float(np.max(mag)),
        "gradient_norm_mean": float(np.mean(grad)),
        "bending_energy_mean": float(np.mean(bend)),
    }
    metrics["qa_status"] = _status(metrics, thresholds)
    return metrics, jac


def image_similarity(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, prefix: str = "") -> dict[str, float | None]:
    roi = _roi(mask, a.shape)
    aa = a[roi].astype(np.float64)
    bb = b[roi].astype(np.float64)
    diff = aa - bb
    aa0 = aa - aa.mean()
    bb0 = bb - bb.mean()
    denom = np.sqrt(np.sum(aa0 * aa0) * np.sum(bb0 * bb0))
    ncc = float(np.sum(aa0 * bb0) / denom) if denom > 0 else None
    return {
        f"{prefix}mse": float(np.mean(diff * diff)),
        f"{prefix}mae": float(np.mean(np.abs(diff))),
        f"{prefix}ncc": ncc,
    }


def dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    denom = aa.sum() + bb.sum()
    return float(2 * np.logical_and(aa, bb).sum() / denom) if denom else 1.0


def _status(metrics: dict[str, float | str], thresholds: Thresholds) -> str:
    if (
        metrics["fraction_jac_le_0"] > thresholds.jac_fail_fraction_le_0
        or metrics["largest_folding_component_ml"] >= thresholds.largest_folding_component_ml_fail
    ):
        return "FAIL"
    if metrics["fraction_jac_lt_0_2"] > thresholds.jac_warn_fraction_lt_0_2:
        return "WARNING"
    if metrics["fraction_jac_gt_5"] > thresholds.jac_warn_fraction_gt_5:
        return "WARNING"
    return "PASS"


def _roi(mask: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match expected {shape}")
    return mask.astype(bool)
