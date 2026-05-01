import pytest
import numpy as np
from core.waveform import process_waveform, find_all_cycles

def test_process_waveform():
    """Test that the low-pass filter smooths the waveform."""
    frame_rate = 30.0
    t = np.linspace(0, 10, 300)
    # Sine wave (0.2 Hz) + high frequency noise (5 Hz)
    clean_wave = np.sin(2 * np.pi * 0.2 * t)
    noise = 0.5 * np.sin(2 * np.pi * 5.0 * t)
    raw_wave = clean_wave + noise

    filtered_wave = process_waveform(raw_wave, frame_rate, cutoff_hz=0.5)

    # The filtered wave should be closer to the clean wave than the raw wave is
    raw_error = np.mean((raw_wave - clean_wave)**2)
    filtered_error = np.mean((filtered_wave - clean_wave)**2)

    assert filtered_error < raw_error

def test_find_all_cycles():
    """Test that cycles (peaks and troughs) are correctly identified."""
    t = np.linspace(0, 10, 300)
    # Sine wave, peaks at ~1.25, 6.25, troughs at ~3.75, 8.75
    wave = np.sin(2 * np.pi * 0.2 * t)

    cycles = find_all_cycles(wave)
    assert len(cycles) > 0

    # Check types
    types = [c[3] for c in cycles]
    assert 'trough-to-trough' in types
    assert 'peak-to-peak' in types
