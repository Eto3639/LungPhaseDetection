from pathlib import Path

import numpy as np

from dvf_qa.amsterdam_shroud import respiratory_signal
from dvf_qa.cycle_pairing import pair_best_correlated_cycles
from dvf_qa.pipeline_report import ReportInputs, write_html_report
from dvf_qa.synthetic_dynamic import (
    DynamicSimulationConfig,
    make_phantom_4dct,
    simulate_from_4dct,
)


def test_pipeline_recovers_simulated_period(tmp_path: Path):
    fps = 15.0
    n_phases = 8
    frames_per_cycle = 30
    n_cycles = 3
    expected_period_s = frames_per_cycle / fps
    expected_freq_hz = 1.0 / expected_period_s

    volumes, spacing = make_phantom_4dct(
        n_phases=n_phases,
        shape=(20, 40, 40),
        motion_amplitude=4.0,
    )
    cfg = DynamicSimulationConfig(
        fps=fps,
        n_cycles=n_cycles,
        frames_per_cycle=frames_per_cycle,
        cycle_jitter_fraction=0.0,
        intensity_jitter=0.0,
    )
    sim = simulate_from_4dct(volumes, spacing, config=cfg, device="cpu")

    ap_signal = respiratory_signal(sim.ap_frames, fps=fps)
    lat_signal = respiratory_signal(sim.lateral_frames, fps=fps)

    assert abs(ap_signal.dominant_frequency_hz - expected_freq_hz) < 0.1
    assert abs(lat_signal.dominant_frequency_hz - expected_freq_hz) < 0.1

    pair = pair_best_correlated_cycles(
        ap_signal.signal,
        lat_signal.signal,
        ap_fps=fps,
        lateral_fps=fps,
        min_period_s=expected_period_s * 0.5,
    )
    assert pair.correlation > 0.9
    recovered_cycle_frames = pair.ap_end - pair.ap_start
    assert abs(recovered_cycle_frames - frames_per_cycle) <= 4


def test_pipeline_writes_html_report(tmp_path: Path):
    fps = 15.0
    volumes, spacing = make_phantom_4dct(n_phases=6, shape=(16, 32, 32), motion_amplitude=3.0)
    cfg = DynamicSimulationConfig(fps=fps, n_cycles=3, frames_per_cycle=24, cycle_jitter_fraction=0.0)
    sim = simulate_from_4dct(volumes, spacing, config=cfg, device="cpu")
    ap_res = respiratory_signal(sim.ap_frames, fps=fps)
    lat_res = respiratory_signal(sim.lateral_frames, fps=fps)
    pair = pair_best_correlated_cycles(
        ap_res.signal, lat_res.signal, ap_fps=fps, lateral_fps=fps, min_period_s=1.0
    )

    report_path = tmp_path / "report.html"
    inputs = ReportInputs(
        case_id="phantom-end-to-end",
        config_summary={
            "fps": cfg.fps,
            "n_cycles": cfg.n_cycles,
            "frames_per_cycle": cfg.frames_per_cycle,
            "n_phases": sim.n_phases,
            "ct_shape": volumes[0].shape,
        },
        phase_volumes=volumes,
        spacing_xyz=spacing,
        ap_phase_drrs=sim.ap_phase_drrs,
        lateral_phase_drrs=sim.lateral_phase_drrs,
        ap_frames=sim.ap_frames,
        lateral_frames=sim.lateral_frames,
        phase_curve=sim.phase_indices,
        fps=fps,
        ap_shroud_result=ap_res,
        lateral_shroud_result=lat_res,
        pair_result=pair,
        extra_metrics={"expected_freq_hz": 1.0 / (cfg.frames_per_cycle / fps)},
    )
    out = write_html_report(inputs, report_path)
    assert out.exists()
    assert out.stat().st_size > 50_000
    text = out.read_text(encoding="utf-8")
    assert "DVF QA pipeline report" in text
    assert "Amsterdam Shroud" in text
    assert "Cycle pairing" in text
    assert "data:image/png;base64" in text
