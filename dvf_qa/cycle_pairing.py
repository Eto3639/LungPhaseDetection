"""Pair the most correlated single respiratory cycles between AP and lateral.

Consumes 1D respiratory signals (e.g. from :mod:`dvf_qa.amsterdam_shroud`),
detects end-of-expiration boundaries in each view, enumerates candidate
single-cycle slices, and returns the AP/lateral pair with the highest
normalized cross-correlation after length normalization.

Intended downstream use: produce paired (AP, lateral) single-breath clips for a
2D-4D reconstruction model that expects time-aligned biplane inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class CycleBoundaries:
    indices: np.ndarray
    fps: float


@dataclass(frozen=True)
class CyclePairResult:
    ap_start: int
    ap_end: int
    lateral_start: int
    lateral_end: int
    correlation: float
    ap_signal: np.ndarray
    lateral_signal: np.ndarray
    ap_resampled: np.ndarray
    lateral_resampled: np.ndarray
    resample_length: int
    ap_boundaries: np.ndarray
    lateral_boundaries: np.ndarray
    all_pairs: list[dict]


def detect_cycle_boundaries(
    signal: np.ndarray,
    fps: float,
    *,
    min_period_s: float = 2.0,
    prominence_factor: float = 0.2,
) -> np.ndarray:
    """Return end-of-expiration frame indices in ``signal``.

    End-of-expiration is taken as a local minimum (valley) of the signal,
    matching the existing DVF-QA convention where ``argmin(signal)`` is the
    end-expiration frame. ``min_period_s`` enforces a minimum spacing between
    valleys to suppress sub-respiratory ripples.

    To stay robust on low-amplitude signals (e.g. lateral projections where
    the diaphragm gradient is weak), the prominence threshold is relaxed
    progressively if the initial threshold leaves fewer than two valleys.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.size < 4:
        return np.empty(0, dtype=np.int64)
    distance = max(1, int(round(min_period_s * fps)))
    std = float(np.std(sig))
    if std <= 0:
        return np.empty(0, dtype=np.int64)
    candidate_factors = [prominence_factor, prominence_factor * 0.25, 0.0]
    last_valleys: np.ndarray = np.empty(0, dtype=np.int64)
    for factor in candidate_factors:
        prom = factor * std if factor > 0 else None
        valleys, _ = find_peaks(-sig, distance=distance, prominence=prom)
        last_valleys = valleys.astype(np.int64)
        if last_valleys.size >= 2:
            return last_valleys
    return last_valleys


def pair_best_correlated_cycles(
    ap_signal: np.ndarray,
    lateral_signal: np.ndarray,
    *,
    ap_fps: float,
    lateral_fps: float,
    min_period_s: float = 2.0,
    prominence_factor: float = 0.2,
    resample_length: int = 64,
    min_cycle_length_factor: float = 0.5,
    max_cycle_length_factor: float = 2.0,
) -> CyclePairResult:
    """Return the AP/lateral cycle pair with maximum normalized correlation.

    Each candidate cycle runs from one end-of-expiration boundary to the next.
    Cycles whose length falls outside ``[min_period_s * min_factor,
    min_period_s * max_factor]`` (scaled by the respective fps) are skipped,
    which discards spurious very-short or very-long detections.
    """
    ap_b = detect_cycle_boundaries(
        ap_signal, ap_fps, min_period_s=min_period_s, prominence_factor=prominence_factor
    )
    lat_b = detect_cycle_boundaries(
        lateral_signal, lateral_fps, min_period_s=min_period_s, prominence_factor=prominence_factor
    )
    if ap_b.size < 2 and lat_b.size < 2:
        raise ValueError(
            f"Both signals lack cycle boundaries (AP={ap_b.size}, Lateral={lat_b.size}); "
            "Amsterdam Shroud likely failed on this case"
        )
    # If one view has no detectable boundaries, fall back to using the other's
    # boundaries scaled to match the failing-view frame count. The pairing
    # still computes NCC over the same-length resampled cycles.
    if ap_b.size < 2:
        ap_b = _project_boundaries(lat_b, target_n_frames=len(ap_signal), lateral_fps=lateral_fps, ap_fps=ap_fps)
    if lat_b.size < 2:
        lat_b = _project_boundaries(ap_b, target_n_frames=len(lateral_signal), lateral_fps=ap_fps, ap_fps=lateral_fps)

    ap_min_len = int(min_period_s * ap_fps * min_cycle_length_factor)
    ap_max_len = int(min_period_s * ap_fps * max_cycle_length_factor / min_cycle_length_factor)
    lat_min_len = int(min_period_s * lateral_fps * min_cycle_length_factor)
    lat_max_len = int(min_period_s * lateral_fps * max_cycle_length_factor / min_cycle_length_factor)

    ap_cycles = _viable_cycles(ap_signal, ap_b, ap_min_len, ap_max_len)
    lat_cycles = _viable_cycles(lateral_signal, lat_b, lat_min_len, lat_max_len)
    if not ap_cycles:
        raise ValueError("No AP cycles passed length filtering")
    if not lat_cycles:
        raise ValueError("No lateral cycles passed length filtering")

    pairs: list[dict] = []
    best_idx = -1
    best_corr = -np.inf
    ap_resampled_cache = [_zscore(_resample_linear(c[2], resample_length)) for c in ap_cycles]
    lat_resampled_cache = [_zscore(_resample_linear(c[2], resample_length)) for c in lat_cycles]

    for i, (ap_start, ap_end, _ap_c) in enumerate(ap_cycles):
        ap_r = ap_resampled_cache[i]
        for j, (lat_start, lat_end, _lat_c) in enumerate(lat_cycles):
            lat_r = lat_resampled_cache[j]
            corr = float(np.mean(ap_r * lat_r))
            pairs.append(
                {
                    "ap_start": ap_start,
                    "ap_end": ap_end,
                    "lateral_start": lat_start,
                    "lateral_end": lat_end,
                    "correlation": corr,
                }
            )
            if corr > best_corr:
                best_corr = corr
                best_idx = len(pairs) - 1

    best = pairs[best_idx]
    ap_start, ap_end = best["ap_start"], best["ap_end"]
    lat_start, lat_end = best["lateral_start"], best["lateral_end"]
    ap_cycle = np.asarray(ap_signal[ap_start:ap_end], dtype=np.float64)
    lat_cycle = np.asarray(lateral_signal[lat_start:lat_end], dtype=np.float64)

    return CyclePairResult(
        ap_start=ap_start,
        ap_end=ap_end,
        lateral_start=lat_start,
        lateral_end=lat_end,
        correlation=best_corr,
        ap_signal=ap_cycle,
        lateral_signal=lat_cycle,
        ap_resampled=_zscore(_resample_linear(ap_cycle, resample_length)),
        lateral_resampled=_zscore(_resample_linear(lat_cycle, resample_length)),
        resample_length=resample_length,
        ap_boundaries=ap_b,
        lateral_boundaries=lat_b,
        all_pairs=pairs,
    )


def save_pair_quicklook(
    ap_signal: np.ndarray,
    lateral_signal: np.ndarray,
    result: CyclePairResult,
    path: str | Path,
    *,
    ap_fps: float | None = None,
    lateral_fps: float | None = None,
) -> None:
    """QC PNG with both full signals (selected cycle shaded) + resampled overlay."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    _plot_signal_with_cycle(
        axes[0], ap_signal, result.ap_start, result.ap_end, result.ap_boundaries, ap_fps, "AP"
    )
    _plot_signal_with_cycle(
        axes[1],
        lateral_signal,
        result.lateral_start,
        result.lateral_end,
        result.lateral_boundaries,
        lateral_fps,
        "Lateral",
    )
    fig.suptitle(
        f"Best paired cycle  |  NCC = {result.correlation:.3f}  "
        f"|  AP frames {result.ap_start}-{result.ap_end}  "
        f"|  Lateral frames {result.lateral_start}-{result.lateral_end}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    overlay_path = path.with_name(path.stem + "_overlay.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0.0, 1.0, result.resample_length)
    ax.plot(x, result.ap_resampled, label="AP (z-scored)", color="tab:blue")
    ax.plot(x, result.lateral_resampled, label="Lateral (z-scored)", color="tab:orange")
    ax.set_xlabel("Cycle fraction")
    ax.set_ylabel("Normalized signal")
    ax.set_title(f"Resampled cycle overlay  |  NCC = {result.correlation:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)


def _project_boundaries(
    source_boundaries: np.ndarray,
    target_n_frames: int,
    lateral_fps: float,
    ap_fps: float,
) -> np.ndarray:
    """Map cycle boundaries from one view's frame index to the other's.

    Used as a fallback when one view's signal is too degenerate to yield its
    own boundaries. Boundaries are scaled by the fps ratio and clipped to the
    target frame count.
    """
    if source_boundaries.size < 2 or lateral_fps <= 0 or ap_fps <= 0:
        return np.empty(0, dtype=np.int64)
    scale = ap_fps / lateral_fps
    projected = (source_boundaries.astype(np.float64) * scale).round().astype(np.int64)
    projected = projected[(projected >= 0) & (projected < target_n_frames)]
    return np.unique(projected)


def _viable_cycles(
    signal: np.ndarray,
    boundaries: np.ndarray,
    min_len: int,
    max_len: int,
) -> list[tuple[int, int, np.ndarray]]:
    out: list[tuple[int, int, np.ndarray]] = []
    sig = np.asarray(signal)
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        length = int(b1 - b0)
        if length < max(2, min_len) or length > max_len:
            continue
        out.append((int(b0), int(b1), sig[b0:b1].astype(np.float64)))
    return out


def _resample_linear(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == n:
        return x.copy()
    if x.size < 2:
        return np.full(n, float(x[0]) if x.size else 0.0)
    src = np.linspace(0.0, 1.0, x.size)
    dst = np.linspace(0.0, 1.0, n)
    return np.interp(dst, src, x)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd <= 0:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / sd


def _plot_signal_with_cycle(ax, signal, start, end, boundaries, fps, title):
    sig = np.asarray(signal)
    if fps is not None and fps > 0:
        t = np.arange(sig.size) / fps
        xlabel = "Time (s)"
        cycle_t0 = start / fps
        cycle_t1 = end / fps
    else:
        t = np.arange(sig.size)
        xlabel = "Frame"
        cycle_t0 = float(start)
        cycle_t1 = float(end)
    ax.plot(t, sig, color="tab:gray", lw=0.8)
    ax.axvspan(cycle_t0, cycle_t1, color="tab:blue", alpha=0.2, label="selected cycle")
    for b in boundaries:
        ax.axvline(t[b] if b < t.size else t[-1], color="tab:red", alpha=0.4, lw=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Signal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
