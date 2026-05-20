import numpy as np

from dvf_qa.amsterdam_shroud import (
    build_shroud,
    respiratory_signal,
    track_diaphragm,
)


def _synthetic_diaphragm_frames(
    n_frames: int = 120,
    height: int = 200,
    width: int = 160,
    breathing_hz: float = 0.25,
    fps: float = 15.0,
    amplitude_px: float = 25.0,
    base_row: float = 150.0,
    noise: float = 0.01,
    seed: int = 0,
):
    t = np.arange(n_frames) / fps
    diaphragm_y = base_row + amplitude_px * np.sin(2 * np.pi * breathing_hz * t)
    frames = np.zeros((n_frames, height, width), dtype=np.float32)
    y = np.arange(height).reshape(-1, 1)
    for i, dy in enumerate(diaphragm_y):
        frames[i] = 1.0 / (1.0 + np.exp(-(y - dy) / 1.5))
    rng = np.random.default_rng(seed)
    frames += rng.normal(0, noise, size=frames.shape).astype(np.float32)
    return frames, diaphragm_y


def test_build_shroud_shape():
    frames = np.zeros((30, 100, 80), dtype=np.float32)
    shroud = build_shroud(frames)
    assert shroud.shape == (99, 30)


def test_track_diaphragm_constant_peak():
    shroud = np.zeros((50, 20), dtype=np.float32)
    shroud[25, :] = 1.0
    rows = track_diaphragm(shroud, smoothing_sigma=0.0, search_band=(0.0, 1.0))
    assert rows.shape == (20,)
    assert np.allclose(rows, 25.0)


def test_recovers_breathing_frequency():
    fps = 15.0
    breathing_hz = 0.25
    frames, _ = _synthetic_diaphragm_frames(fps=fps, breathing_hz=breathing_hz)
    result = respiratory_signal(frames, fps=fps)
    assert abs(result.dominant_frequency_hz - breathing_hz) < 0.05


def test_tracked_position_correlates_with_truth():
    fps = 15.0
    frames, truth = _synthetic_diaphragm_frames(fps=fps, breathing_hz=0.25)
    result = respiratory_signal(frames, fps=fps)
    tracked = result.diaphragm_row - np.mean(result.diaphragm_row)
    truth_centered = truth - np.mean(truth)
    corr = np.corrcoef(tracked, truth_centered)[0, 1]
    assert abs(corr) > 0.95


def test_bandpass_suppresses_cardiac_component():
    fps = 15.0
    breathing_hz = 0.25
    cardiac_hz = 1.1
    t = np.arange(150) / fps
    diaphragm_y = (
        150.0
        + 25.0 * np.sin(2 * np.pi * breathing_hz * t)
        + 4.0 * np.sin(2 * np.pi * cardiac_hz * t)
    )
    frames = np.zeros((150, 200, 160), dtype=np.float32)
    y = np.arange(200).reshape(-1, 1)
    for i, dy in enumerate(diaphragm_y):
        frames[i] = 1.0 / (1.0 + np.exp(-(y - dy) / 1.5))
    result = respiratory_signal(frames, fps=fps, respiratory_band=(0.1, 0.6))
    assert abs(result.dominant_frequency_hz - breathing_hz) < 0.05
