from __future__ import annotations

import csv
import json
import time
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.signal import hilbert, savgol_filter

from .image_io import write_png


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PUBLIC_DEMO_IMAGES = {
    "ap": {
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Chest%20Xray%20PA%203-8-2010.png",
        "filename": "public_cc0_chest_xray_pa.png",
        "source": "https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png",
        "license": "CC0 1.0",
    },
    "lateral": {
        "url": "https://commons.wikimedia.org/wiki/Special:Redirect/file/Chest%20Xray%20Lateral%203-8-2010.png",
        "filename": "public_cc0_chest_xray_lateral.png",
        "source": "https://commons.wikimedia.org/wiki/File:Chest_Xray_Lateral_3-8-2010.png",
        "license": "CC0 1.0",
    },
}
WIKIMEDIA_LATERAL_CATEGORY_API = (
    "https://commons.wikimedia.org/w/api.php?"
    "action=query&generator=categorymembers&gcmtitle=Category:Lateral_x-rays_of_the_chest"
    "&gcmtype=file&gcmlimit=50&prop=imageinfo&iiprop=url|size&iiurlwidth=1200&format=json"
)


@dataclass(frozen=True)
class DynamicXrayResult:
    view: str
    frames: np.ndarray
    masks: np.ndarray
    signal: np.ndarray
    phase_rad: np.ndarray
    phase_fraction: np.ndarray
    state: list[str]
    area_px: np.ndarray
    inferior_boundary_px: np.ndarray
    polarity: str
    mask_method: str
    signal_method: str
    roi_mask: np.ndarray
    roi_mean_intensity: np.ndarray
    qc: dict[str, float | str]


def read_frame_sequence(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"No image files found in {path}")
        frames = [_as_gray(iio.imread(p)) for p in files]
        return np.stack(frames).astype(np.float32)

    if path.suffix == ".npy":
        return _ensure_frame_stack(np.load(path))
    if path.suffix == ".npz":
        npz = np.load(path)
        key = "data" if "data" in npz else npz.files[0]
        return _ensure_frame_stack(npz[key])
    if _is_sitk_path(path):
        image = sitk.ReadImage(str(path))
        return _ensure_frame_stack(sitk.GetArrayFromImage(image))

    return _ensure_frame_stack(iio.imread(path))


def analyze_dynamic_xray(
    frames: np.ndarray,
    *,
    view: str,
    mask_method: str = "unsupervised",
    device: str | None = None,
    model_cache_dir: str | Path | None = None,
    mask_threshold: float = 0.5,
    signal_method: str = "motion",
    roi_fraction: float = 0.5,
    roi_min_mask_frequency: float = 0.6,
    lateral_smoothing_sigma: float = 0.0,
    min_component_area_fraction: float = 0.01,
) -> DynamicXrayResult:
    raw_frames = _ensure_frame_stack(frames)
    frames = _normalize_frames(raw_frames)
    intensity_frames = _normalize_frames_global(raw_frames)
    if mask_method == "torchxrayvision":
        masks = segment_lungs_torchxrayvision(
            frames,
            device=device,
            cache_dir=model_cache_dir,
            threshold=mask_threshold,
            min_component_area_fraction=min_component_area_fraction,
        )
        polarity = "torchxrayvision"
    elif mask_method == "unsupervised":
        if view == "lateral":
            masks = _segment_lateral_frames(frames, min_component_area_fraction, smoothing_sigma=lateral_smoothing_sigma)
            polarity = "lateral_unsupervised"
        else:
            mean_frame = frames.mean(axis=0)
            polarity, static_mask = _estimate_lung_mask(mean_frame, min_component_area_fraction)
            masks = np.stack(
                [
                    _segment_frame(frame, static_mask, polarity, min_component_area_fraction)
                    for frame in frames
                ]
            ).astype(bool)
    else:
        raise ValueError(f"Unsupported mask_method: {mask_method}")

    area = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float32)
    inferior = np.array([_inferior_boundary(mask) for mask in masks], dtype=np.float32)
    motion_signal = _zscore(area) + _zscore(inferior)
    roi_mask = make_lung_center_roi(
        masks,
        fraction=roi_fraction,
        min_mask_frequency=roi_min_mask_frequency,
        reference_image=intensity_frames.mean(axis=0),
        prefer_radiolucent=view == "lateral",
    )
    roi_mean = _roi_mean_intensity(intensity_frames, roi_mask)
    qc = _compute_roi_qc(masks, roi_mask, view)
    intensity_signal = -_zscore(roi_mean)
    if signal_method == "motion":
        signal = _smooth_signal(motion_signal)
    elif signal_method == "intensity-roi":
        signal = _smooth_signal(intensity_signal)
    elif signal_method == "combined":
        signal = _smooth_signal(_zscore(motion_signal) + _zscore(intensity_signal))
    else:
        raise ValueError(f"Unsupported signal_method: {signal_method}")
    phase = _phase_from_signal(signal)
    state = _respiratory_state(signal)

    return DynamicXrayResult(
        view=view,
        frames=frames,
        masks=masks,
        signal=signal,
        phase_rad=phase,
        phase_fraction=phase / (2.0 * np.pi),
        state=state,
        area_px=area,
        inferior_boundary_px=inferior,
        polarity=polarity,
        mask_method=mask_method,
        signal_method=signal_method,
        roi_mask=roi_mask,
        roi_mean_intensity=roi_mean,
        qc=qc,
    )


def segment_lungs_torchxrayvision(
    frames: np.ndarray,
    *,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    threshold: float = 0.5,
    min_component_area_fraction: float = 0.01,
) -> np.ndarray:
    """Segment left/right lungs with TorchXRayVision ChestX-Det PSPNet."""
    try:
        import torch
        import torch.nn.functional as F
        import torchxrayvision as xrv
    except ImportError as exc:
        raise ImportError(
            "TorchXRayVision lung segmentation requires torchxrayvision. "
            "Install it with `pip install torchxrayvision` or use "
            "`--mask-method unsupervised`."
        ) from exc

    frames = _normalize_frames(_ensure_frame_stack(frames))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(cache_dir) if cache_dir else None)
    model = model.to(device).eval()
    targets = list(model.targets)
    lung_indices = [targets.index("Left Lung"), targets.index("Right Lung")]
    masks = []

    with torch.no_grad():
        for frame in frames:
            padded, crop = _pad_to_square(frame)
            tensor = torch.from_numpy(padded[None, None].astype(np.float32) * 2048.0 - 1024.0).to(device)
            output = model(tensor)
            lung_logits = output[:, lung_indices].max(dim=1, keepdim=True).values
            if float(lung_logits.min()) < 0.0 or float(lung_logits.max()) > 1.0:
                lung_probs = torch.sigmoid(lung_logits)
            else:
                lung_probs = lung_logits
            lung_probs = F.interpolate(
                lung_probs,
                size=padded.shape,
                mode="bilinear",
                align_corners=False,
            )
            padded_mask = lung_probs.squeeze().detach().cpu().numpy() >= threshold
            y0, y1, x0, x1 = crop
            mask = padded_mask[y0:y1, x0:x1]
            cleaned = _clean_mask(mask, min_component_area_fraction)
            masks.append(cleaned if cleaned.any() else mask)

    return np.stack(masks).astype(bool)


def make_lung_center_roi(
    masks: np.ndarray,
    *,
    fraction: float = 0.5,
    min_mask_frequency: float = 0.6,
    reference_image: np.ndarray | None = None,
    prefer_radiolucent: bool = False,
) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (frames, height, width), got {masks.shape}")
    fraction = float(np.clip(fraction, 0.1, 1.0))
    min_mask_frequency = float(np.clip(min_mask_frequency, 0.0, 1.0))
    consensus = masks.mean(axis=0) >= min_mask_frequency
    if consensus.sum() < max(8, int(masks.shape[1] * masks.shape[2] * 0.002)):
        consensus = masks.mean(axis=0) > 0.0
    consensus = ndi.binary_erosion(consensus, iterations=2)
    if consensus.sum() == 0:
        consensus = masks.mean(axis=0) > 0.0
    if prefer_radiolucent and reference_image is not None:
        radiolucent = _radiolucent_lung_core(consensus, reference_image)
        if radiolucent.sum() >= max(8, int(consensus.sum() * 0.05)):
            consensus = radiolucent

    labels, count = ndi.label(consensus)
    roi = np.zeros_like(consensus, dtype=bool)
    if count == 0:
        return roi
    sizes = ndi.sum(consensus, labels, index=np.arange(1, count + 1))
    min_area = max(8, int(consensus.size * 0.002))
    keep = [i + 1 for i in np.argsort(sizes)[-2:] if sizes[i] >= min_area]
    for label in keep:
        component = labels == label
        parts = [component] if _is_lateral_view_masks(masks) else _split_wide_lung_component(component)
        for part in parts:
            roi |= _component_center_roi(part, fraction)
    if roi.sum() == 0:
        roi = consensus
    return roi


def _is_lateral_view_masks(masks: np.ndarray) -> bool:
    consensus = masks.mean(axis=0) > 0
    if consensus.sum() == 0:
        return False
    yy, xx = np.nonzero(consensus)
    width = float(xx.max() - xx.min() + 1)
    height = float(yy.max() - yy.min() + 1)
    return height > width * 1.15


def _radiolucent_lung_core(lung_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    if lung_mask.sum() == 0:
        return lung_mask
    image = np.asarray(image, dtype=np.float32)
    body_mask = _estimate_body_mask(image)
    body_distance = ndi.distance_transform_edt(body_mask)
    min_body_distance = max(3.0, min(image.shape) * 0.035)
    values = image[lung_mask]
    lower = float(np.percentile(values, 8))
    upper = float(np.percentile(values, 60))
    texture = _local_texture(image)
    texture_floor = float(np.percentile(texture[lung_mask], 35))
    candidate = (
        lung_mask
        & body_mask
        & (body_distance >= min_body_distance)
        & (image >= lower)
        & (image <= upper)
        & (texture >= texture_floor)
        & _central_lateral_roi(lung_mask.shape)
    )
    candidate = ndi.binary_opening(candidate, iterations=1)
    candidate = ndi.binary_closing(candidate, iterations=5)
    labels, count = ndi.label(candidate)
    if count == 0:
        return candidate

    h, w = candidate.shape
    scored = []
    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < max(8, int(lung_mask.sum() * 0.03)):
            continue
        yy, xx = np.nonzero(component)
        if _touches_border(component, margin_fraction=0.025):
            continue
        body_overlap = float((component & body_mask).sum()) / max(area, 1)
        if body_overlap < 0.98:
            continue
        if float(np.median(body_distance[component])) < min_body_distance:
            continue
        median_intensity = float(np.median(image[component]))
        if median_intensity <= float(np.percentile(image, 5)):
            continue
        median_texture = float(np.median(texture[component]))
        if median_texture < texture_floor:
            continue
        cx = float(xx.mean()) / max(w, 1)
        cy = float(yy.mean()) / max(h, 1)
        if cx < 0.14 or cx > 0.80 or cy < 0.12 or cy > 0.88:
            continue
        bbox_width = float(xx.max() - xx.min() + 1) / max(w, 1)
        bbox_height = float(yy.max() - yy.min() + 1) / max(h, 1)
        if bbox_width < 0.08 or bbox_height < 0.08:
            continue
        centrality = (1.0 - abs(cx - 0.42)) * (1.0 - 0.5 * abs(cy - 0.52))
        scored.append((area * centrality, label))
    if not scored:
        return candidate
    return ndi.binary_fill_holes(labels == max(scored)[1])


def _local_texture(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    image = image.astype(np.float32)
    mean = ndi.gaussian_filter(image, sigma=sigma)
    mean_sq = ndi.gaussian_filter(image * image, sigma=sigma)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance).astype(np.float32)


def _touches_border(mask: np.ndarray, margin_fraction: float) -> bool:
    if mask.sum() == 0:
        return False
    h, w = mask.shape
    margin_y = max(1, int(round(h * margin_fraction)))
    margin_x = max(1, int(round(w * margin_fraction)))
    border = np.zeros_like(mask, dtype=bool)
    border[:margin_y, :] = True
    border[-margin_y:, :] = True
    border[:, :margin_x] = True
    border[:, -margin_x:] = True
    return bool((mask & border).any())


def _split_wide_lung_component(component: np.ndarray) -> list[np.ndarray]:
    yy, xx = np.nonzero(component)
    if yy.size == 0:
        return []
    width_fraction = (float(xx.max() - xx.min() + 1) / max(component.shape[1], 1))
    if width_fraction < 0.45:
        return [component]
    mid_x = int(np.median(xx))
    left = component.copy()
    left[:, mid_x + 1 :] = False
    right = component.copy()
    right[:, : mid_x + 1] = False
    min_part = max(8, int(component.sum() * 0.15))
    parts = [part for part in (left, right) if int(part.sum()) >= min_part]
    return parts or [component]


def _component_center_roi(component: np.ndarray, fraction: float) -> np.ndarray:
    component_roi = np.zeros_like(component, dtype=bool)
    if component.sum() == 0:
        return component_roi
    yy, xx = np.nonzero(component)
    if yy.size == 0:
        return component_roi
    y0, y1 = _central_interval(int(yy.min()), int(yy.max()) + 1, fraction)
    x0, x1 = _central_interval(int(xx.min()), int(xx.max()) + 1, fraction)
    component_roi[y0:y1, x0:x1] = True
    return component & component_roi


def resolve_lateral_mask_method(mask_method: str, lateral_mask_method: str) -> str:
    if lateral_mask_method == "auto":
        return "unsupervised" if mask_method == "torchxrayvision" else mask_method
    return lateral_mask_method


def run_public_cxr_demo(
    out_dir: str | Path,
    *,
    mask_method: str = "torchxrayvision",
    lateral_mask_method: str = "auto",
    device: str | None = None,
    model_cache_dir: str | Path | None = None,
    mask_threshold: float = 0.5,
    signal_method: str = "intensity-roi",
    roi_fraction: float = 0.5,
    roi_min_mask_frequency: float = 0.6,
    lateral_smoothing_sigma: float = 0.0,
) -> dict[str, object]:
    out = Path(out_dir)
    image_dir = out / "public_demo_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = {
        view: _download_public_demo_image(view, image_dir)
        for view in ("ap", "lateral")
    }
    ap_result = analyze_dynamic_xray(
        read_frame_sequence(image_paths["ap"]),
        view="ap",
        mask_method=mask_method,
        device=device,
        model_cache_dir=model_cache_dir,
        mask_threshold=mask_threshold,
        signal_method=signal_method,
        roi_fraction=roi_fraction,
        roi_min_mask_frequency=roi_min_mask_frequency,
    )
    lateral_method = resolve_lateral_mask_method(mask_method, lateral_mask_method)
    lateral_result = analyze_dynamic_xray(
        read_frame_sequence(image_paths["lateral"]),
        view="lateral",
        mask_method=lateral_method,
        device=device,
        model_cache_dir=model_cache_dir,
        mask_threshold=mask_threshold,
        signal_method=signal_method,
        roi_fraction=roi_fraction,
        roi_min_mask_frequency=roi_min_mask_frequency,
        lateral_smoothing_sigma=lateral_smoothing_sigma,
    )
    summaries = {
        "ap": write_dynamic_xray_outputs(ap_result, out),
        "lateral": write_dynamic_xray_outputs(lateral_result, out),
        "public_images": PUBLIC_DEMO_IMAGES,
        "lateral_strategy": _lateral_strategy_note(mask_method, lateral_method),
    }
    _write_public_demo_figure(out / "public_cxr_mask_demo.png", ap_result, lateral_result)
    (out / "public_cxr_demo_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def run_lateral_smoothing_demo(
    out_dir: str | Path,
    *,
    cases: int = 10,
    sigmas: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0),
    roi_fraction: float = 0.5,
) -> dict[str, object]:
    out = Path(out_dir)
    image_dir = out / "wikimedia_lateral_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images = _download_wikimedia_lateral_images(image_dir, cases)
    rows = []
    for image_path in images:
        frames = _normalize_frames(read_frame_sequence(image_path))
        frame = frames[0]
        case = {"file": image_path.name, "masks": []}
        for sigma in sigmas:
            mask = _segment_lateral_frames(frames, 0.01, smoothing_sigma=sigma)[0]
            roi = make_lung_center_roi(
                mask[None],
                fraction=roi_fraction,
                reference_image=frame,
                prefer_radiolucent=True,
            )
            qc = _compute_roi_qc(mask[None], roi, "lateral")
            case["masks"].append(
                {
                    "sigma": float(sigma),
                    "mask_area_px": int(mask.sum()),
                    "roi_area_px": int(roi.sum()),
                    "qc_status": str(qc["status"]),
                    "qc_roi_area_fraction_of_lung": float(qc["roi_area_fraction_of_lung"]),
                    "qc_roi_centroid_x_fraction": float(qc["roi_centroid_x_fraction"]),
                    "qc_roi_centroid_y_fraction": float(qc["roi_centroid_y_fraction"]),
                }
            )
        rows.append(case)
    _write_lateral_smoothing_montage(out / "lateral_smoothing_grid.png", images, sigmas, roi_fraction)
    summary = {
        "source": "https://commons.wikimedia.org/wiki/Category:Lateral_x-rays_of_the_chest",
        "cases": len(images),
        "sigmas": [float(v) for v in sigmas],
        "results": rows,
    }
    (out / "lateral_smoothing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_dynamic_xray_outputs(result: DynamicXrayResult, out_dir: str | Path) -> dict[str, str | float | int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = result.view

    np.save(out / f"{prefix}_lung_masks.npy", result.masks.astype(np.uint8))
    np.save(out / f"{prefix}_roi_mask.npy", result.roi_mask.astype(np.uint8))
    write_png(out / f"{prefix}_mean_frame.png", result.frames.mean(axis=0))
    _write_mask_quicklook(out / f"{prefix}_mask_quicklook.png", result)
    _write_roi_quicklook(out / f"{prefix}_roi_quicklook.png", result)
    write_png(out / f"{prefix}_end_expiration_frame.png", result.frames[int(np.argmin(result.signal))])
    _write_phase_plot(out / f"{prefix}_respiratory_phase.png", result)
    _write_phase_csv(out / f"{prefix}_phase.csv", result)

    summary: dict[str, str | float | int] = {
        "view": result.view,
        "frames": int(result.frames.shape[0]),
        "height": int(result.frames.shape[1]),
        "width": int(result.frames.shape[2]),
        "mask_method": result.mask_method,
        "signal_method": result.signal_method,
        "polarity": result.polarity,
        "roi_area_px": int(result.roi_mask.sum()),
        "mean_roi_intensity": float(np.nanmean(result.roi_mean_intensity)),
        "mean_lung_area_px": float(np.mean(result.area_px)),
        "qc_status": str(result.qc["status"]),
        "qc_roi_lung_overlap_fraction": float(result.qc["roi_lung_overlap_fraction"]),
        "qc_roi_area_fraction_of_lung": float(result.qc["roi_area_fraction_of_lung"]),
        "qc_roi_centroid_x_fraction": float(result.qc["roi_centroid_x_fraction"]),
        "qc_roi_centroid_y_fraction": float(result.qc["roi_centroid_y_fraction"]),
        "signal_min": float(np.min(result.signal)),
        "signal_max": float(np.max(result.signal)),
        "end_inspiration_frame": int(np.argmax(result.signal)),
        "end_expiration_frame": int(np.argmin(result.signal)),
    }
    (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_combined_phase(
    ap: DynamicXrayResult | None,
    lateral: DynamicXrayResult | None,
    out_dir: str | Path,
) -> dict[str, str | int] | None:
    if ap is None or lateral is None:
        return None
    n = min(ap.signal.size, lateral.signal.size)
    if n < 2:
        return None
    signal = _smooth_signal(_zscore(ap.signal[:n]) + _zscore(lateral.signal[:n]))
    phase = _phase_from_signal(signal)
    state = _respiratory_state(signal)
    out = Path(out_dir)
    rows = [
        {
            "frame": i,
            "combined_signal": float(signal[i]),
            "phase_rad": float(phase[i]),
            "phase_fraction": float(phase[i] / (2.0 * np.pi)),
            "state": state[i],
        }
        for i in range(n)
    ]
    with (out / "combined_phase.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(signal, label="combined")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Respiratory signal")
    ax2 = ax.twinx()
    ax2.plot(phase / (2.0 * np.pi), color="tab:orange", alpha=0.7, label="phase")
    ax2.set_ylabel("Phase fraction")
    fig.tight_layout()
    fig.savefig(out / "combined_respiratory_phase.png", dpi=160)
    plt.close(fig)
    summary = {
        "views": "ap,lateral",
        "frames": n,
        "signal_method": f"{ap.signal_method}+{lateral.signal_method}",
        "signal_min": float(np.min(signal)),
        "signal_max": float(np.max(signal)),
        "end_inspiration_frame": int(np.argmax(signal)),
        "end_expiration_frame": int(np.argmin(signal)),
    }
    (out / "combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _ensure_frame_stack(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] <= 4:
        arr = _as_gray(arr)[None, ...]
    elif arr.ndim == 4 and arr.shape[-1] <= 4:
        arr = arr[..., :3].mean(axis=-1)
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D image or 3D frame stack, got shape {arr.shape}")
    return arr.astype(np.float32)


def _as_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.float32)


def _is_sitk_path(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return suffixes.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd", ".dcm"))


def _download_public_demo_image(view: str, out_dir: Path) -> Path:
    info = PUBLIC_DEMO_IMAGES[view]
    path = out_dir / str(info["filename"])
    if path.exists() and path.stat().st_size > 0:
        return path
    request = urllib.request.Request(
        str(info["url"]),
        headers={"User-Agent": "dvf-qa-public-cxr-demo/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def _download_wikimedia_lateral_images(out_dir: Path, cases: int) -> list[Path]:
    request = urllib.request.Request(
        WIKIMEDIA_LATERAL_CATEGORY_API,
        headers={"User-Agent": "dvf-qa-lateral-smoothing-demo/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = list(payload.get("query", {}).get("pages", {}).values())
    candidates = []
    for page in pages:
        imageinfo = page.get("imageinfo", [{}])[0]
        url = imageinfo.get("thumburl") or imageinfo.get("url")
        width = int(imageinfo.get("width", 0) or 0)
        height = int(imageinfo.get("height", 0) or 0)
        suffix = Path(urlparse(url).path).suffix.lower() if url else ""
        if not url or suffix not in IMAGE_SUFFIXES:
            continue
        if min(width, height) < 500:
            continue
        candidates.append((page.get("title", "image"), url))
    paths = []
    for index, (title, url) in enumerate(candidates[:cases], start=1):
        suffix = Path(urlparse(url).path).suffix.lower()
        safe_title = "".join(c if c.isalnum() else "_" for c in title.replace("File:", ""))[:80]
        path = out_dir / f"{index:02d}_{safe_title}{suffix}"
        if not path.exists() or path.stat().st_size == 0:
            image_request = urllib.request.Request(url, headers={"User-Agent": "dvf-qa-lateral-smoothing-demo/0.1"})
            last_error: Exception | None = None
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(image_request, timeout=120) as response:
                        path.write_bytes(response.read())
                    last_error = None
                    break
                except Exception as exc:  # pragma: no cover - network-dependent
                    last_error = exc
                    time.sleep(2.0 * (attempt + 1))
            if last_error is not None:
                continue
            time.sleep(0.5)
        paths.append(path)
        if len(paths) >= cases:
            return paths
    if len(paths) < cases:
        paths.extend(_download_hf_openi_lateral_images(out_dir, cases - len(paths), start_index=len(paths) + 1))
    if len(paths) < cases:
        raise ValueError(f"Only downloaded {len(paths)} usable lateral images; requested {cases}")
    return paths


def _download_hf_openi_lateral_images(out_dir: Path, cases: int, *, start_index: int) -> list[Path]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Need more lateral images after Wikimedia rate limiting, but `datasets` is not installed. "
            "Install it with `pip install datasets`."
        ) from exc
    paths = []
    dataset = load_dataset("ykumards/open-i", split="train", streaming=True)
    for row in dataset:
        lateral = row.get("img_lateral")
        if not lateral:
            continue
        uid = str(row.get("uid", "unknown"))
        path = out_dir / f"{start_index + len(paths):02d}_openi_{uid}_lateral.png"
        path.write_bytes(lateral)
        paths.append(path)
        if len(paths) >= cases:
            break
    return paths


def _normalize_frames(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float32)
    out = np.empty_like(frames)
    for i, frame in enumerate(frames):
        finite = np.nan_to_num(frame)
        lo, hi = np.percentile(finite, [1, 99])
        if hi <= lo:
            out[i] = 0.0
        else:
            out[i] = np.clip((finite - lo) / (hi - lo), 0.0, 1.0)
    return out


def _normalize_frames_global(frames: np.ndarray) -> np.ndarray:
    frames = np.nan_to_num(frames.astype(np.float32))
    lo, hi = np.percentile(frames, [1, 99])
    if hi <= lo:
        return np.zeros_like(frames, dtype=np.float32)
    return np.clip((frames - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _pad_to_square(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape
    side = max(height, width)
    pad_y = side - height
    pad_x = side - width
    top = pad_y // 2
    left = pad_x // 2
    padded = np.full((side, side), float(np.median(image)), dtype=np.float32)
    padded[top : top + height, left : left + width] = image
    return padded, (top, top + height, left, left + width)


def _central_interval(start: int, stop: int, fraction: float) -> tuple[int, int]:
    length = max(1, stop - start)
    inner = max(1, int(round(length * fraction)))
    offset = (length - inner) // 2
    return start + offset, start + offset + inner


def _roi_mean_intensity(frames: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    if roi_mask.sum() == 0:
        return np.full(frames.shape[0], np.nan, dtype=np.float32)
    return np.array([float(np.mean(frame[roi_mask])) for frame in frames], dtype=np.float32)


def _compute_roi_qc(masks: np.ndarray, roi_mask: np.ndarray, view: str) -> dict[str, float | str]:
    consensus = masks.mean(axis=0) > 0
    lung_area = max(1, int(consensus.sum()))
    roi_area = int(roi_mask.sum())
    if roi_area == 0:
        return {
            "status": "FAIL",
            "roi_lung_overlap_fraction": 0.0,
            "roi_area_fraction_of_lung": 0.0,
            "roi_centroid_x_fraction": 0.0,
            "roi_centroid_y_fraction": 0.0,
        }
    overlap = int((roi_mask & consensus).sum()) / max(roi_area, 1)
    area_fraction = roi_area / lung_area
    yy, xx = np.nonzero(roi_mask)
    cx = float(xx.mean()) / max(roi_mask.shape[1] - 1, 1)
    cy = float(yy.mean()) / max(roi_mask.shape[0] - 1, 1)
    status = "PASS"
    if overlap < 0.95:
        status = "FAIL"
    elif area_fraction < 0.05 or area_fraction > 0.55:
        status = "WARN"
    elif view == "lateral" and (cx < 0.18 or cx > 0.72 or cy < 0.18 or cy > 0.82):
        status = "WARN"
    elif view != "lateral" and (cy < 0.18 or cy > 0.82):
        status = "WARN"
    return {
        "status": status,
        "roi_lung_overlap_fraction": float(overlap),
        "roi_area_fraction_of_lung": float(area_fraction),
        "roi_centroid_x_fraction": float(cx),
        "roi_centroid_y_fraction": float(cy),
    }


def _estimate_lung_mask(mean_frame: np.ndarray, min_area_fraction: float) -> tuple[str, np.ndarray]:
    threshold = _otsu_threshold(mean_frame)
    dark = _clean_mask(mean_frame <= threshold, min_area_fraction)
    bright = _clean_mask(mean_frame >= threshold, min_area_fraction)
    dark_score = _mask_score(dark)
    bright_score = _mask_score(bright)
    if bright_score > dark_score:
        return "bright", bright
    return "dark", dark


def _segment_frame(frame: np.ndarray, static_mask: np.ndarray, polarity: str, min_area_fraction: float) -> np.ndarray:
    threshold = _otsu_threshold(frame[static_mask]) if static_mask.any() else _otsu_threshold(frame)
    raw = frame >= threshold if polarity == "bright" else frame <= threshold
    raw = ndi.binary_dilation(static_mask, iterations=10) & raw
    cleaned = _clean_mask(raw, min_area_fraction)
    if cleaned.sum() == 0:
        return static_mask
    return cleaned


def _segment_lateral_frames(frames: np.ndarray, min_area_fraction: float, *, smoothing_sigma: float = 0.0) -> np.ndarray:
    mean_frame = frames.mean(axis=0)
    body_mask = _estimate_body_mask(mean_frame)
    body_distance = ndi.distance_transform_edt(body_mask)
    body_interior = body_distance >= _body_boundary_clearance_px(mean_frame.shape)
    texture = _local_texture(mean_frame)
    texture_floor = float(np.percentile(texture[body_mask], 30)) if body_mask.any() else float(np.percentile(texture, 30))
    texture_mask = texture >= texture_floor
    dilated_body = ndi.binary_dilation(body_mask, iterations=3)
    masks = []
    for frame in frames:
        threshold = _otsu_threshold(frame[body_mask]) if body_mask.any() else _otsu_threshold(frame)
        raw = (frame <= threshold) & dilated_body & body_interior & texture_mask & _central_lateral_roi(frame.shape)
        raw = _smooth_binary_mask(raw, smoothing_sigma)
        cleaned = _clean_lateral_mask(raw, min_area_fraction, body_distance=body_distance)
        if cleaned.sum() == 0:
            relaxed = (frame <= threshold) & dilated_body & texture_mask & _central_lateral_roi(frame.shape)
            relaxed = _smooth_binary_mask(relaxed, smoothing_sigma)
            cleaned = _clean_lateral_mask(relaxed, max(min_area_fraction * 0.25, 0.002), body_distance=body_distance)
        masks.append(cleaned)
    return np.stack(masks).astype(bool)


def _smooth_binary_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return mask
    smoothed = ndi.gaussian_filter(mask.astype(np.float32), sigma=float(sigma))
    return smoothed >= 0.5


def _estimate_body_mask(frame: np.ndarray) -> np.ndarray:
    non_background = frame > max(0.03, float(np.percentile(frame, 5)))
    non_background = ndi.binary_closing(non_background, iterations=6)
    non_background = ndi.binary_fill_holes(non_background)
    labels, count = ndi.label(non_background)
    if count == 0:
        return np.ones_like(frame, dtype=bool)
    sizes = ndi.sum(non_background, labels, index=np.arange(1, count + 1))
    body = labels == (int(np.argmax(sizes)) + 1)
    body = ndi.binary_erosion(body, iterations=4)
    return body


def _central_lateral_roi(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return (yy > 0.06 * h) & (yy < 0.92 * h) & (xx > 0.04 * w) & (xx < 0.88 * w)


def _body_boundary_clearance_px(shape: tuple[int, int]) -> float:
    return max(4.0, min(shape) * 0.035)


def _clean_lateral_mask(
    mask: np.ndarray,
    min_area_fraction: float,
    *,
    body_distance: np.ndarray | None = None,
) -> np.ndarray:
    min_area = max(8, int(mask.size * min_area_fraction))
    cleaned = ndi.binary_opening(mask, iterations=1)
    cleaned = ndi.binary_closing(cleaned, iterations=2)
    cleaned = ndi.binary_fill_holes(cleaned)
    labels, count = ndi.label(cleaned)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    h, w = mask.shape
    candidates = []
    for label in range(1, count + 1):
        component = labels == label
        area = int(component.sum())
        if area < min_area:
            continue
        yy, xx = np.nonzero(component)
        centroid_x = float(xx.mean()) / max(w, 1)
        centroid_y = float(yy.mean()) / max(h, 1)
        if centroid_x > 0.78 or centroid_y < 0.12 or centroid_y > 0.86:
            continue
        if body_distance is not None:
            clearance = _body_boundary_clearance_px(mask.shape)
            near_boundary_fraction = float((body_distance[component] < clearance).mean())
            median_distance = float(np.median(body_distance[component]))
            if near_boundary_fraction > 0.20 or median_distance < clearance:
                continue
        center_score = 1.0 - abs(centroid_x - 0.45)
        candidates.append((area * center_score, label))
    if not candidates:
        return np.zeros_like(mask, dtype=bool)
    best_label = max(candidates)[1]
    return labels == best_label


def _clean_mask(mask: np.ndarray, min_area_fraction: float) -> np.ndarray:
    min_area = max(8, int(mask.size * min_area_fraction))
    cleaned = ndi.binary_opening(mask, iterations=1)
    cleaned = ndi.binary_closing(cleaned, iterations=3)
    cleaned = ndi.binary_fill_holes(cleaned)
    labels, count = ndi.label(cleaned)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(cleaned, labels, index=np.arange(1, count + 1))
    keep_labels = [i + 1 for i in np.argsort(sizes)[-2:] if sizes[i] >= min_area]
    return np.isin(labels, keep_labels)


def _mask_score(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return -1.0
    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    center_weight = 1.0 - abs(float(xx.mean()) - (w - 1) / 2.0) / max(w, 1)
    vertical_weight = 1.0 - abs(float(yy.mean()) - (h - 1) * 0.55) / max(h, 1)
    area_fraction = mask.mean()
    plausible_area = 1.0 if 0.08 <= area_fraction <= 0.65 else 0.25
    return center_weight + vertical_weight + plausible_area


def _otsu_threshold(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float32)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.5
    hist, edges = np.histogram(data, bins=256, range=(float(data.min()), float(data.max())))
    if hist.sum() == 0 or edges[0] == edges[-1]:
        return float(np.mean(data))
    prob = hist.astype(np.float64) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma = np.zeros_like(denom)
    valid = denom > 0
    sigma[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(centers[int(np.argmax(sigma))])


def _inferior_boundary(mask: np.ndarray) -> float:
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return np.nan
    return float(rows.max())


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.isnan(values).any():
        finite = np.isfinite(values)
        fill = np.nanmedian(values[finite]) if finite.any() else 0.0
        values = np.where(finite, values, fill)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / std).astype(np.float32)


def _smooth_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    n = signal.size
    if n < 5:
        return signal
    window = min(n if n % 2 else n - 1, 11)
    if window < 5:
        return signal
    return savgol_filter(signal, window_length=window, polyorder=2).astype(np.float32)


def _phase_from_signal(signal: np.ndarray) -> np.ndarray:
    centered = signal - float(np.mean(signal))
    if centered.size < 3 or float(np.std(centered)) < 1e-6:
        return np.zeros_like(centered, dtype=np.float32)
    phase = np.angle(hilbert(centered))
    phase = (phase - phase[0]) % (2.0 * np.pi)
    return phase.astype(np.float32)


def _respiratory_state(signal: np.ndarray) -> list[str]:
    if signal.size < 2:
        return ["static"] * int(signal.size)
    gradient = np.gradient(signal)
    return ["inspiration" if v >= 0 else "expiration" for v in gradient]


def _write_phase_csv(path: Path, result: DynamicXrayResult) -> None:
    rows = [
        {
            "frame": i,
            "area_px": float(result.area_px[i]),
            "inferior_boundary_px": float(result.inferior_boundary_px[i]),
            "roi_mean_intensity": float(result.roi_mean_intensity[i]),
            "respiratory_signal": float(result.signal[i]),
            "phase_rad": float(result.phase_rad[i]),
            "phase_fraction": float(result.phase_fraction[i]),
            "state": result.state[i],
        }
        for i in range(result.frames.shape[0])
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_mask_quicklook(path: Path, result: DynamicXrayResult) -> None:
    indices = np.linspace(0, result.frames.shape[0] - 1, min(6, result.frames.shape[0]), dtype=int)
    fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3), squeeze=False)
    for ax, idx in zip(axes.ravel(), indices, strict=True):
        ax.imshow(result.frames[idx], cmap="gray")
        ax.contour(result.masks[idx], colors="lime", linewidths=0.8)
        ax.set_title(f"{result.view} frame {idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_roi_quicklook(path: Path, result: DynamicXrayResult) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(result.frames.mean(axis=0), cmap="gray")
    if result.masks.any():
        display_mask = ndi.binary_fill_holes(result.masks.mean(axis=0) > 0)
        ax.contour(display_mask, colors="lime", linewidths=0.7)
    if result.roi_mask.any():
        overlay = np.ma.masked_where(~result.roi_mask, result.roi_mask)
        ax.imshow(overlay, cmap="autumn", alpha=0.45)
        ax.contour(result.roi_mask, colors="yellow", linewidths=1.0)
    ax.set_title(f"{result.view}: ROI for {result.signal_method}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_public_demo_figure(path: Path, ap: DynamicXrayResult, lateral: DynamicXrayResult) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), squeeze=False)
    for row, result in enumerate((ap, lateral)):
        image = result.frames[0]
        mask = result.masks[0]
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"{result.view}: input")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(image, cmap="gray")
        if mask.any():
            axes[row, 1].contour(mask, colors="lime", linewidths=1.0)
            overlay = np.ma.masked_where(~mask, mask)
            axes[row, 1].imshow(overlay, cmap="Greens", alpha=0.25)
        axes[row, 1].set_title(f"{result.view}: {result.mask_method} mask")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _write_lateral_smoothing_montage(
    path: Path,
    image_paths: list[Path],
    sigmas: tuple[float, ...],
    roi_fraction: float,
) -> None:
    nrows = len(image_paths)
    ncols = len(sigmas) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False)
    for row, image_path in enumerate(image_paths):
        frames = _normalize_frames(read_frame_sequence(image_path))
        frame = frames[0]
        axes[row, 0].imshow(frame, cmap="gray")
        axes[row, 0].set_title(f"case {row + 1}: input")
        axes[row, 0].axis("off")
        for col, sigma in enumerate(sigmas, start=1):
            mask = _segment_lateral_frames(frames, 0.01, smoothing_sigma=sigma)[0]
            roi = make_lung_center_roi(
                mask[None],
                fraction=roi_fraction,
                reference_image=frame,
                prefer_radiolucent=True,
            )
            qc = _compute_roi_qc(mask[None], roi, "lateral")
            ax = axes[row, col]
            ax.imshow(frame, cmap="gray")
            if mask.any():
                ax.contour(ndi.binary_fill_holes(mask), colors="lime", linewidths=0.6)
            if roi.any():
                overlay = np.ma.masked_where(~roi, roi)
                ax.imshow(overlay, cmap="autumn", alpha=0.35)
                ax.contour(roi, colors="yellow", linewidths=0.8)
            ax.set_title(f"sigma={sigma:g} {qc['status']}")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _lateral_strategy_note(mask_method: str, lateral_method: str) -> str:
    if mask_method == "torchxrayvision" and lateral_method == "unsupervised":
        return (
            "TorchXRayVision ChestX-Det is used for frontal/AP images. "
            "Lateral images default to the unsupervised radiolucent-region mask because "
            "the public TorchXRayVision segmentation model is not validated for lateral CXR."
        )
    return f"Lateral images use mask method: {lateral_method}."


def _write_phase_plot(path: Path, result: DynamicXrayResult) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(result.signal, label=f"{result.signal_method} signal")
    ax.scatter([int(np.argmax(result.signal))], [float(np.max(result.signal))], label="end inspiration")
    ax.scatter([int(np.argmin(result.signal))], [float(np.min(result.signal))], label="end expiration")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Signal")
    ax2 = ax.twinx()
    ax2.plot(result.phase_fraction, color="tab:orange", alpha=0.7, label="phase")
    if np.isfinite(result.roi_mean_intensity).any():
        roi_norm = _zscore(result.roi_mean_intensity)
        ax.plot(roi_norm, color="tab:green", alpha=0.35, label="ROI intensity z")
    ax2.set_ylabel("Phase fraction")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
