import json
import subprocess
import sys
import types

import numpy as np
import torch

from dvf_qa.dynamic_xray import (
    PUBLIC_DEMO_IMAGES,
    analyze_dynamic_xray,
    read_frame_sequence,
    resolve_lateral_mask_method,
    run_public_cxr_demo,
    segment_lungs_torchxrayvision,
    write_dynamic_xray_outputs,
)


def _synthetic_dynamic_xray(frames=16, height=96, width=96):
    yy, xx = np.mgrid[:height, :width]
    stack = []
    for i in range(frames):
        amp = np.sin(2 * np.pi * i / frames)
        y_radius = 24 + 5 * amp
        y_center = 44 + 4 * amp
        left = ((xx - 35) / 15) ** 2 + ((yy - y_center) / y_radius) ** 2 <= 1
        right = ((xx - 61) / 15) ** 2 + ((yy - y_center) / y_radius) ** 2 <= 1
        image = np.full((height, width), 0.75, dtype=np.float32)
        image[left | right] = 0.2
        image += 0.02 * np.random.default_rng(i).normal(size=image.shape).astype(np.float32)
        stack.append(image)
    return np.stack(stack)


def _synthetic_intensity_dynamic_xray(frames=16, height=96, width=96):
    yy, xx = np.mgrid[:height, :width]
    left = ((xx - 35) / 15) ** 2 + ((yy - 44) / 24) ** 2 <= 1
    right = ((xx - 61) / 15) ** 2 + ((yy - 44) / 24) ** 2 <= 1
    stack = []
    for i in range(frames):
        image = np.full((height, width), 0.75, dtype=np.float32)
        lung_intensity = 0.26 + 0.08 * np.cos(2 * np.pi * i / frames)
        image[left | right] = lung_intensity
        stack.append(image)
    return np.stack(stack)


def test_analyze_dynamic_xray_estimates_masks_and_phase():
    frames = _synthetic_dynamic_xray()
    result = analyze_dynamic_xray(frames, view="ap")
    assert result.masks.shape == frames.shape
    assert result.phase_rad.shape == (frames.shape[0],)
    assert result.area_px.max() > result.area_px.min()
    assert result.polarity == "dark"
    assert result.mask_method == "unsupervised"
    assert set(result.state) <= {"inspiration", "expiration"}


def test_intensity_roi_signal_selects_high_density_expiration_without_diaphragm_motion():
    frames = _synthetic_intensity_dynamic_xray()
    result = analyze_dynamic_xray(frames, view="ap", signal_method="intensity-roi")
    assert result.signal_method == "intensity-roi"
    assert result.roi_mask.sum() > 0
    assert result.roi_mean_intensity.max() > result.roi_mean_intensity.min()
    assert int(np.argmax(result.roi_mean_intensity)) == 0
    assert int(np.argmin(result.signal)) == 0


def test_torchxrayvision_lung_segmentation_uses_left_right_lung_channels(monkeypatch):
    class FakePSPNet:
        targets = [
            "Left Clavicle",
            "Right Clavicle",
            "Left Scapula",
            "Right Scapula",
            "Left Lung",
            "Right Lung",
            "Left Hilus Pulmonis",
            "Right Hilus Pulmonis",
            "Heart",
            "Aorta",
            "Facies Diaphragmatica",
            "Mediastinum",
            "Weasand",
            "Spine",
        ]

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, image):
            batch, _, height, width = image.shape
            output = torch.zeros((batch, len(self.targets), height, width), device=image.device)
            output[:, self.targets.index("Left Lung"), 20:70, 18:42] = 0.9
            output[:, self.targets.index("Right Lung"), 20:70, 54:78] = 0.9
            return output

    fake_xrv = types.SimpleNamespace(
        baseline_models=types.SimpleNamespace(
            chestx_det=types.SimpleNamespace(PSPNet=lambda cache_dir=None: FakePSPNet())
        )
    )
    monkeypatch.setitem(sys.modules, "torchxrayvision", fake_xrv)

    frames = _synthetic_dynamic_xray(frames=2)
    masks = segment_lungs_torchxrayvision(frames, device="cpu")
    assert masks.shape == frames.shape
    assert masks[:, 20:70, 18:42].mean() > 0.8
    assert masks[:, 20:70, 54:78].mean() > 0.8


def test_torchxrayvision_lung_segmentation_accepts_rectangular_images(monkeypatch):
    class FakePSPNet:
        targets = ["Left Lung", "Right Lung"]

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, image):
            batch, _, height, width = image.shape
            assert height == width
            output = torch.zeros((batch, len(self.targets), height, width), device=image.device)
            output[:, 0, height // 4 : height // 2, width // 4 : width // 2] = 0.9
            output[:, 1, height // 4 : height // 2, width // 2 : 3 * width // 4] = 0.9
            return output

    fake_xrv = types.SimpleNamespace(
        baseline_models=types.SimpleNamespace(
            chestx_det=types.SimpleNamespace(PSPNet=lambda cache_dir=None: FakePSPNet())
        )
    )
    monkeypatch.setitem(sys.modules, "torchxrayvision", fake_xrv)

    frames = _synthetic_dynamic_xray(frames=2, height=80, width=120)
    masks = segment_lungs_torchxrayvision(frames, device="cpu")
    assert masks.shape == frames.shape
    assert masks.any()


def test_dynamic_xray_outputs(tmp_path):
    result = analyze_dynamic_xray(_synthetic_dynamic_xray(frames=8), view="ap", signal_method="intensity-roi")
    summary = write_dynamic_xray_outputs(result, tmp_path)
    assert summary["frames"] == 8
    assert summary["signal_method"] == "intensity-roi"
    assert (tmp_path / "ap_lung_masks.npy").exists()
    assert (tmp_path / "ap_roi_mask.npy").exists()
    assert (tmp_path / "ap_roi_quicklook.png").exists()
    assert (tmp_path / "ap_end_expiration_frame.png").exists()
    assert (tmp_path / "ap_phase.csv").exists()
    assert (tmp_path / "ap_summary.json").exists()


def test_read_frame_sequence_from_npy(tmp_path):
    frames = _synthetic_dynamic_xray(frames=4)
    path = tmp_path / "frames.npy"
    np.save(path, frames)
    loaded = read_frame_sequence(path)
    assert loaded.shape == frames.shape


def test_lateral_auto_falls_back_from_torchxrayvision_to_unsupervised():
    assert resolve_lateral_mask_method("torchxrayvision", "auto") == "unsupervised"
    assert resolve_lateral_mask_method("unsupervised", "auto") == "unsupervised"
    assert resolve_lateral_mask_method("torchxrayvision", "torchxrayvision") == "torchxrayvision"


def test_public_cxr_demo_writes_visual_outputs_without_network(monkeypatch, tmp_path):
    frames = _synthetic_dynamic_xray(frames=1)

    def fake_download(view, out_dir):
        path = out_dir / PUBLIC_DEMO_IMAGES[view]["filename"]
        np.save(path.with_suffix(".npy"), frames[0])
        return path.with_suffix(".npy")

    monkeypatch.setattr("dvf_qa.dynamic_xray._download_public_demo_image", fake_download)
    summary = run_public_cxr_demo(tmp_path, mask_method="unsupervised")
    assert summary["ap"]["frames"] == 1
    assert summary["lateral"]["mask_method"] == "unsupervised"
    assert (tmp_path / "public_cxr_mask_demo.png").exists()
    assert (tmp_path / "public_cxr_demo_summary.json").exists()


def test_dynamic_xray_cli(tmp_path):
    frames = _synthetic_dynamic_xray(frames=8)
    ap = tmp_path / "ap.npy"
    lateral = tmp_path / "lateral.npy"
    out = tmp_path / "out"
    np.save(ap, frames)
    np.save(lateral, frames)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dvf_qa.cli",
            "analyze-dynamic-xray",
            "--ap",
            str(ap),
            "--lateral",
            str(lateral),
            "--out",
            str(out),
        ],
        check=True,
    )
    summary = json.loads((out / "dynamic_xray_summary.json").read_text())
    assert summary["ap"]["frames"] == 8
    assert (out / "combined_phase.csv").exists()
