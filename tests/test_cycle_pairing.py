import numpy as np
import pytest

from dvf_qa.cycle_pairing import (
    detect_cycle_boundaries,
    pair_best_correlated_cycles,
)


def _sin_signal(n_frames=200, breathing_hz=0.25, fps=15.0, phase=0.0, noise=0.0, seed=0):
    t = np.arange(n_frames) / fps
    sig = np.sin(2 * np.pi * breathing_hz * t + phase)
    if noise > 0:
        rng = np.random.default_rng(seed)
        sig = sig + rng.normal(0.0, noise, size=sig.size)
    return sig


def test_detect_boundaries_on_clean_sine():
    fps = 15.0
    breathing_hz = 0.25
    n_frames = 200
    signal = _sin_signal(n_frames=n_frames, breathing_hz=breathing_hz, fps=fps)
    boundaries = detect_cycle_boundaries(signal, fps)
    expected_cycles = breathing_hz * (n_frames / fps)
    assert len(boundaries) >= int(expected_cycles) - 1
    assert len(boundaries) <= int(expected_cycles) + 1


def test_pair_simultaneous_signals_returns_high_correlation():
    fps = 15.0
    ap = _sin_signal(fps=fps)
    lateral = _sin_signal(fps=fps)
    result = pair_best_correlated_cycles(ap, lateral, ap_fps=fps, lateral_fps=fps)
    assert result.correlation > 0.95
    assert result.ap_end > result.ap_start
    assert result.lateral_end > result.lateral_start


def test_pair_phase_shifted_signals_still_align():
    fps = 15.0
    ap = _sin_signal(fps=fps, phase=0.0)
    lateral = _sin_signal(fps=fps, phase=1.7)
    result = pair_best_correlated_cycles(ap, lateral, ap_fps=fps, lateral_fps=fps)
    assert result.correlation > 0.9


def test_pair_handles_different_fps():
    ap_fps = 15.0
    lateral_fps = 7.5
    ap = _sin_signal(n_frames=200, fps=ap_fps)
    lateral = _sin_signal(n_frames=100, fps=lateral_fps)
    result = pair_best_correlated_cycles(ap, lateral, ap_fps=ap_fps, lateral_fps=lateral_fps)
    assert result.correlation > 0.9
    ap_cycle_s = (result.ap_end - result.ap_start) / ap_fps
    lat_cycle_s = (result.lateral_end - result.lateral_start) / lateral_fps
    assert abs(ap_cycle_s - lat_cycle_s) < 0.5


def test_pair_picks_clean_cycle_over_noisy_one():
    fps = 15.0
    t = np.arange(300) / fps
    ap = np.sin(2 * np.pi * 0.25 * t)
    lateral = np.sin(2 * np.pi * 0.25 * t)
    rng = np.random.default_rng(0)
    noise_window = slice(0, 100)
    ap[noise_window] += rng.normal(0.0, 1.5, size=100)
    lateral[noise_window] += rng.normal(0.0, 1.5, size=100)
    result = pair_best_correlated_cycles(ap, lateral, ap_fps=fps, lateral_fps=fps)
    assert result.ap_start >= 100 or result.ap_end > 200


def test_pair_raises_when_too_few_cycles():
    short = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        pair_best_correlated_cycles(short, short, ap_fps=15.0, lateral_fps=15.0)


def test_resampled_outputs_have_requested_length():
    fps = 15.0
    ap = _sin_signal(fps=fps)
    lateral = _sin_signal(fps=fps)
    result = pair_best_correlated_cycles(
        ap, lateral, ap_fps=fps, lateral_fps=fps, resample_length=128
    )
    assert result.ap_resampled.shape == (128,)
    assert result.lateral_resampled.shape == (128,)
