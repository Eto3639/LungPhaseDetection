"""Build simulated AP + lateral dynamic projection sequences from a 4DCT.

Given a 4DCT — a list of 3D CT volumes representing different respiratory
phases — this module renders an AP DRR and a lateral DRR for each phase via
DiffDRR, then assembles the per-phase projections into time-resolved
sequences by cycling the phase index over a configurable number of breaths.

The output is two arrays with shape ``(T, H, W)``: one AP, one lateral. They
can be fed directly into :mod:`dvf_qa.amsterdam_shroud` for respiratory signal
extraction and into :mod:`dvf_qa.cycle_pairing` for AP/lateral cycle pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .drr import DiffDRRGeometry, project_drr


@dataclass(frozen=True)
class SyntheticDynamicResult:
    ap_frames: np.ndarray
    lateral_frames: np.ndarray
    phase_indices: np.ndarray
    fps: float
    n_cycles: int
    n_phases: int
    ap_phase_drrs: np.ndarray
    lateral_phase_drrs: np.ndarray


@dataclass
class DynamicSimulationConfig:
    fps: float = 15.0
    n_cycles: int = 3
    frames_per_cycle: int = 60
    cycle_jitter_fraction: float = 0.05
    intensity_jitter: float = 0.0
    interpolation: str = "linear"
    seed: int = 0


def default_ap_geometry(detector_shape=(256, 256), pixel_spacing_mm=(1.0, 1.0)) -> DiffDRRGeometry:
    """AP projection (beam along +y in DiffDRR's world frame)."""
    return DiffDRRGeometry(
        sdd=1020.0,
        detector_shape=detector_shape,
        pixel_spacing_mm=pixel_spacing_mm,
        rotation=(0.0, 0.0, 0.0),
        translation=(0.0, 850.0, 0.0),
        parameterization="euler_angles",
        convention="ZXY",
        orientation="AP",
        center_volume=True,
    )


def default_lateral_geometry(detector_shape=(256, 256), pixel_spacing_mm=(1.0, 1.0)) -> DiffDRRGeometry:
    """Lateral projection (90 degree rotation around z from the AP pose)."""
    return DiffDRRGeometry(
        sdd=1020.0,
        detector_shape=detector_shape,
        pixel_spacing_mm=pixel_spacing_mm,
        rotation=(np.pi / 2.0, 0.0, 0.0),
        translation=(0.0, 850.0, 0.0),
        parameterization="euler_angles",
        convention="ZXY",
        orientation="AP",
        center_volume=True,
    )


def render_phase_drrs(
    phase_volumes: list[np.ndarray],
    spacing_xyz: tuple[float, float, float],
    *,
    ap_geometry: DiffDRRGeometry | None = None,
    lateral_geometry: DiffDRRGeometry | None = None,
    device: str | None = None,
    ct_value_mode: str = "hu",
    invert_vertical: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Render AP and lateral DRR for each phase volume.

    ``invert_vertical`` (default True) flips the detector vertical axis so the
    head appears at the top of the image, matching the conventional clinical
    display. Disable if your downstream pipeline expects DiffDRR's raw axis
    order.

    Returns
    -------
    ap_drrs:
        Array with shape ``(n_phases, H, W)``.
    lateral_drrs:
        Array with shape ``(n_phases, H, W)``.
    """
    ap_geom = ap_geometry or default_ap_geometry()
    lat_geom = lateral_geometry or default_lateral_geometry()
    ap_list: list[np.ndarray] = []
    lat_list: list[np.ndarray] = []
    for vol in phase_volumes:
        ap_geom_local = _with_value_mode(ap_geom, ct_value_mode)
        lat_geom_local = _with_value_mode(lat_geom, ct_value_mode)
        ap_img = project_drr(vol, spacing_xyz, (0.0, 0.0, 0.0), ap_geom_local, device=device)
        lat_img = project_drr(vol, spacing_xyz, (0.0, 0.0, 0.0), lat_geom_local, device=device)
        if invert_vertical:
            ap_img = np.flipud(ap_img).copy()
            lat_img = np.flipud(lat_img).copy()
        ap_list.append(ap_img)
        lat_list.append(lat_img)
    return np.stack(ap_list).astype(np.float32), np.stack(lat_list).astype(np.float32)


def build_dynamic_sequence(
    ap_phase_drrs: np.ndarray,
    lateral_phase_drrs: np.ndarray,
    config: DynamicSimulationConfig | None = None,
) -> SyntheticDynamicResult:
    """Build (T, H, W) dynamic AP/lateral sequences from per-phase DRRs.

    The phase axis is traversed forward then backward (0,1,...,N-1,N-2,...,1)
    per breath. Optional Gaussian timing jitter and intensity jitter inject
    mild irregularity so the result is non-trivial input for downstream
    respiratory analysis.
    """
    cfg = config or DynamicSimulationConfig()
    if ap_phase_drrs.shape != lateral_phase_drrs.shape:
        raise ValueError(
            "ap_phase_drrs and lateral_phase_drrs must have the same shape; "
            f"got {ap_phase_drrs.shape} vs {lateral_phase_drrs.shape}"
        )
    n_phases = int(ap_phase_drrs.shape[0])
    if n_phases < 2:
        raise ValueError(f"Need at least 2 phases, got {n_phases}")

    rng = np.random.default_rng(cfg.seed)
    phase_curve = _phase_curve(
        n_phases=n_phases,
        n_cycles=cfg.n_cycles,
        frames_per_cycle=cfg.frames_per_cycle,
        jitter_fraction=cfg.cycle_jitter_fraction,
        rng=rng,
    )

    ap_frames = _interpolate_phase_stack(ap_phase_drrs, phase_curve, cfg.interpolation)
    lat_frames = _interpolate_phase_stack(lateral_phase_drrs, phase_curve, cfg.interpolation)

    if cfg.intensity_jitter > 0:
        ap_frames = ap_frames * (1.0 + rng.normal(0.0, cfg.intensity_jitter, size=(ap_frames.shape[0], 1, 1)).astype(np.float32))
        lat_frames = lat_frames * (1.0 + rng.normal(0.0, cfg.intensity_jitter, size=(lat_frames.shape[0], 1, 1)).astype(np.float32))

    return SyntheticDynamicResult(
        ap_frames=ap_frames.astype(np.float32),
        lateral_frames=lat_frames.astype(np.float32),
        phase_indices=phase_curve.astype(np.float32),
        fps=cfg.fps,
        n_cycles=cfg.n_cycles,
        n_phases=n_phases,
        ap_phase_drrs=ap_phase_drrs.astype(np.float32),
        lateral_phase_drrs=lateral_phase_drrs.astype(np.float32),
    )


def simulate_from_4dct(
    phase_volumes: list[np.ndarray],
    spacing_xyz: tuple[float, float, float],
    *,
    config: DynamicSimulationConfig | None = None,
    ap_geometry: DiffDRRGeometry | None = None,
    lateral_geometry: DiffDRRGeometry | None = None,
    device: str | None = None,
    ct_value_mode: str = "hu",
    invert_vertical: bool = True,
) -> SyntheticDynamicResult:
    """Convenience: render per-phase DRRs then build the dynamic sequence."""
    ap_drrs, lat_drrs = render_phase_drrs(
        phase_volumes,
        spacing_xyz,
        ap_geometry=ap_geometry,
        lateral_geometry=lateral_geometry,
        device=device,
        ct_value_mode=ct_value_mode,
        invert_vertical=invert_vertical,
    )
    return build_dynamic_sequence(ap_drrs, lat_drrs, config=config)


def make_phantom_4dct(
    n_phases: int = 10,
    shape: tuple[int, int, int] = (32, 64, 64),
    air_hu: float = -800.0,
    tissue_hu: float = 200.0,
    diaphragm_z_fraction: float = 0.55,
    motion_amplitude: float = 6.0,
    soft_transition_px: float = 1.5,
    seed: int = 0,
) -> tuple[list[np.ndarray], tuple[float, float, float]]:
    """Tiny phantom 4DCT with a moving diaphragm-like HU step in z.

    Each phase has air (low HU) above and tissue (high HU) below a sigmoidal
    boundary whose z position oscillates sinusoidally across phases. Matches
    the use case Amsterdam Shroud is designed for (a single CC-direction
    gradient ridge moving with respiration).
    """
    z, _, _ = shape
    rng = np.random.default_rng(seed)
    base_z = z * float(diaphragm_z_fraction)
    phase_offsets = motion_amplitude * np.sin(np.linspace(0.0, 2.0 * np.pi, n_phases, endpoint=False))
    zz = np.arange(z, dtype=np.float32).reshape(-1, 1, 1)
    volumes: list[np.ndarray] = []
    for offset in phase_offsets:
        diaphragm_z = base_z + float(offset)
        weight = 1.0 / (1.0 + np.exp(-(zz - diaphragm_z) / max(soft_transition_px, 1e-3)))
        vol = (air_hu + (tissue_hu - air_hu) * weight).astype(np.float32)
        vol = np.broadcast_to(vol, shape).copy()
        vol += rng.normal(0.0, 5.0, size=shape).astype(np.float32)
        volumes.append(vol)
    spacing = (1.0, 1.0, 1.0)
    return volumes, spacing


def save_quicklook(
    result: SyntheticDynamicResult,
    path: str | Path,
    *,
    n_samples: int = 6,
) -> None:
    """PNG preview of phase DRRs and dynamic frames."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_phases = result.n_phases
    n_samples = min(n_samples, result.ap_frames.shape[0])
    sample_t = np.linspace(0, result.ap_frames.shape[0] - 1, n_samples).astype(int)

    fig, axes = plt.subplots(4, max(n_phases, n_samples), figsize=(2 * max(n_phases, n_samples), 8))
    for i in range(max(n_phases, n_samples)):
        if i < n_phases:
            axes[0, i].imshow(result.ap_phase_drrs[i], cmap="gray")
            axes[0, i].set_title(f"AP φ{i}")
            axes[1, i].imshow(result.lateral_phase_drrs[i], cmap="gray")
            axes[1, i].set_title(f"Lat φ{i}")
        else:
            axes[0, i].axis("off")
            axes[1, i].axis("off")
        if i < n_samples:
            t = sample_t[i]
            axes[2, i].imshow(result.ap_frames[t], cmap="gray")
            axes[2, i].set_title(f"AP t={t}")
            axes[3, i].imshow(result.lateral_frames[t], cmap="gray")
            axes[3, i].set_title(f"Lat t={t}")
        else:
            axes[2, i].axis("off")
            axes[3, i].axis("off")
        for r in range(4):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _phase_curve(
    *,
    n_phases: int,
    n_cycles: int,
    frames_per_cycle: int,
    jitter_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a continuous, cyclic phase index over time.

    One breath = one full traversal of [0, n_phases), matching the clinical
    4DCT convention where phase indices cover one respiratory period.
    Phase indices are emitted modulo n_phases and downstream interpolation
    wraps cyclically (phase n_phases-1 connects back to phase 0).
    """
    if n_phases < 2 or n_cycles < 1 or frames_per_cycle < 2:
        raise ValueError("Need n_phases>=2, n_cycles>=1, frames_per_cycle>=2")
    phase_span = float(n_phases)
    curves: list[np.ndarray] = []
    for _ in range(n_cycles):
        samples_per_cycle = max(2, int(round(frames_per_cycle * (1.0 + rng.normal(0.0, jitter_fraction)))))
        curve = np.linspace(0.0, phase_span, samples_per_cycle, endpoint=False)
        curves.append(curve)
    return np.concatenate(curves)


def _interpolate_phase_stack(
    phase_drrs: np.ndarray,
    phase_curve: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    n_phases = phase_drrs.shape[0]
    wrapped = np.mod(phase_curve, n_phases)
    if interpolation == "nearest":
        idx = np.mod(np.round(wrapped).astype(int), n_phases)
        return phase_drrs[idx]
    if interpolation != "linear":
        raise ValueError(f"Unsupported interpolation: {interpolation}")
    floor = np.mod(np.floor(wrapped).astype(int), n_phases)
    nxt = np.mod(floor + 1, n_phases)
    frac = (wrapped - np.floor(wrapped)).reshape(-1, 1, 1)
    return ((1.0 - frac) * phase_drrs[floor] + frac * phase_drrs[nxt]).astype(np.float32)


def _with_value_mode(geom: DiffDRRGeometry, ct_value_mode: str) -> DiffDRRGeometry:
    if geom.ct_value_mode == ct_value_mode:
        return geom
    return DiffDRRGeometry(
        sdd=geom.sdd,
        detector_shape=geom.detector_shape,
        pixel_spacing_mm=geom.pixel_spacing_mm,
        rotation=geom.rotation,
        translation=geom.translation,
        parameterization=geom.parameterization,
        convention=geom.convention,
        degrees=geom.degrees,
        orientation=geom.orientation,
        center_volume=geom.center_volume,
        renderer=geom.renderer,
        reverse_x_axis=geom.reverse_x_axis,
        x0=geom.x0,
        y0=geom.y0,
        patch_size=geom.patch_size,
        bone_attenuation_multiplier=geom.bone_attenuation_multiplier,
        resample_target=geom.resample_target,
        ct_value_mode=ct_value_mode,
        renderer_kwargs=geom.renderer_kwargs,
    )
