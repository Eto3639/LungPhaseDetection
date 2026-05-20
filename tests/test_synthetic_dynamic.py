import numpy as np
import pytest

from dvf_qa.synthetic_dynamic import (
    DynamicSimulationConfig,
    build_dynamic_sequence,
    make_phantom_4dct,
    render_phase_drrs,
    simulate_from_4dct,
)


def test_make_phantom_4dct_shapes():
    volumes, spacing = make_phantom_4dct(n_phases=4, shape=(16, 32, 32))
    assert len(volumes) == 4
    assert all(v.shape == (16, 32, 32) for v in volumes)
    assert spacing == (1.0, 1.0, 1.0)


def test_phantom_4dct_sphere_moves_across_phases():
    volumes, _ = make_phantom_4dct(n_phases=8, shape=(24, 48, 48), motion_amplitude=4.0)
    centroids_z = []
    for v in volumes:
        mask = v > 0.0
        zz = np.indices(v.shape)[0][mask]
        centroids_z.append(float(zz.mean()) if zz.size > 0 else float("nan"))
    centroids_z = np.array(centroids_z)
    assert (centroids_z.max() - centroids_z.min()) > 2.0


def test_render_phase_drrs_returns_expected_shapes():
    volumes, spacing = make_phantom_4dct(n_phases=3, shape=(12, 24, 24))
    ap, lat = render_phase_drrs(volumes, spacing, device="cpu")
    assert ap.shape[0] == 3
    assert lat.shape[0] == 3
    assert ap.ndim == 3 and lat.ndim == 3
    assert np.isfinite(ap).all() and np.isfinite(lat).all()


def test_build_dynamic_sequence_length_matches_config():
    rng = np.random.default_rng(0)
    ap_phase = rng.uniform(0.0, 1.0, size=(5, 16, 16)).astype(np.float32)
    lat_phase = rng.uniform(0.0, 1.0, size=(5, 16, 16)).astype(np.float32)
    cfg = DynamicSimulationConfig(n_cycles=2, frames_per_cycle=20, cycle_jitter_fraction=0.0)
    result = build_dynamic_sequence(ap_phase, lat_phase, cfg)
    assert result.ap_frames.shape == (40, 16, 16)
    assert result.lateral_frames.shape == (40, 16, 16)
    assert result.phase_indices.shape == (40,)


def test_dynamic_sequence_phase_curve_cycles():
    rng = np.random.default_rng(0)
    ap_phase = rng.uniform(0.0, 1.0, size=(5, 8, 8)).astype(np.float32)
    cfg = DynamicSimulationConfig(n_cycles=2, frames_per_cycle=20, cycle_jitter_fraction=0.0)
    result = build_dynamic_sequence(ap_phase, ap_phase, cfg)
    curve = result.phase_indices
    assert curve.min() == pytest.approx(0.0, abs=0.5)
    assert curve.max() < 5.0
    assert curve.max() > 4.0
    cycle_a = curve[:20]
    cycle_b = curve[20:40]
    np.testing.assert_allclose(cycle_a, cycle_b, atol=1e-6)


def test_simulate_from_4dct_end_to_end():
    volumes, spacing = make_phantom_4dct(n_phases=4, shape=(12, 24, 24))
    cfg = DynamicSimulationConfig(n_cycles=2, frames_per_cycle=16, cycle_jitter_fraction=0.0)
    result = simulate_from_4dct(volumes, spacing, config=cfg, device="cpu")
    assert result.ap_frames.shape[0] == 32
    assert result.lateral_frames.shape[0] == 32
    assert result.fps == cfg.fps
    assert result.n_cycles == cfg.n_cycles
