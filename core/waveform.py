import numpy as np
import cv2
import logging
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import pydicom
import torch.nn as nn
from typing import Tuple, List, Dict, Optional

from core.model import segment_lung_xray


def extract_waveform_centroid(dicom_ds: pydicom.Dataset) -> np.ndarray:
    """
    Extracts the respiratory waveform by calculating the centroid of the lower half of each frame.

    Args:
        dicom_ds (pydicom.Dataset): Loaded multi-frame DICOM dataset.

    Returns:
        np.ndarray: Extracted 1D waveform array.
    """
    wave = []
    for frame_2d in dicom_ds.pixel_array:
        img = frame_2d.astype(np.float32)
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = img_8bit.shape
        roi = img_8bit[h // 2 :, :]
        M = cv2.moments(roi)
        cy = (
            (M["m01"] / M["m00"] + (h // 2))
            if M["m00"] != 0
            else (wave[-1] if wave else h * 0.75)
        )
        wave.append(cy)
    logging.info("Extracted waveform using CENTROID method.")
    return np.array(wave)


def extract_waveform_roi_pca(
    dicom_ds: pydicom.Dataset, model: nn.Module
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the respiratory waveform using PCA over a lung ROI segmented from the middle frame.

    Args:
        dicom_ds (pydicom.Dataset): Loaded multi-frame DICOM dataset.
        model (nn.Module): The TorchXRayVision segmentation model.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Extracted 1D waveform array.
            - The binary ROI mask.
            - A visualization image of the ROI on the middle frame.
    """
    pixel_array_3d = dicom_ds.pixel_array
    num_frames, height, width = pixel_array_3d.shape

    middle_frame_idx = num_frames // 2
    middle_frame = pixel_array_3d[middle_frame_idx]
    roi_mask = segment_lung_xray(model, middle_frame)

    if np.sum(roi_mask) == 0:
        logging.warning(
            "Lung segmentation failed; ROI is empty. Falling back to global PCA."
        )
        roi_mask = np.ones((height, width), dtype=np.uint8)

    data_matrix = pixel_array_3d.reshape(num_frames, height * width)
    roi_pixels = data_matrix[:, roi_mask.flatten() > 0]

    if roi_pixels.shape[1] == 0:
        logging.error("Cannot perform PCA on an empty ROI. Returning a flat waveform.")
        middle_frame_8bit = cv2.normalize(
            pixel_array_3d[middle_frame_idx], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        roi_viz = cv2.cvtColor(middle_frame_8bit, cv2.COLOR_GRAY2BGR)
        return np.zeros(num_frames), np.zeros((height, width), dtype=np.uint8), roi_viz

    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(roi_pixels).flatten()

    middle_frame_8bit = cv2.normalize(
        middle_frame, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    roi_viz = cv2.cvtColor(middle_frame_8bit, cv2.COLOR_GRAY2BGR)
    roi_viz[roi_mask > 0, 1] = 255

    logging.info("Extracted waveform using ROI-PCA method.")
    return pca_result, roi_mask, roi_viz


def process_waveform(
    raw_wave: np.ndarray, frame_rate: float, cutoff_hz: float = 0.5
) -> np.ndarray:
    """
    Applies a low-pass filter to the raw waveform to remove noise.

    Args:
        raw_wave (np.ndarray): The raw 1D waveform.
        frame_rate (float): The frame rate of the original DICOM.
        cutoff_hz (float, optional): Cutoff frequency in Hz. Defaults to 0.5.

    Returns:
        np.ndarray: Filtered 1D waveform.
    """
    nyquist = 0.5 * frame_rate
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(4, normal_cutoff, btype="low", analog=False)
    filtered_wave = filtfilt(b, a, raw_wave)
    logging.info(f"Filtered waveform with a {cutoff_hz} Hz low-pass filter.")
    return filtered_wave


def find_all_cycles(wave: np.ndarray) -> List[Tuple[int, int, str, str]]:
    """
    Identifies all respiratory cycles (peak-to-peak and trough-to-trough) in a waveform.

    Args:
        wave (np.ndarray): The filtered 1D waveform.

    Returns:
        List[Tuple[int, int, str, str]]: A list of cycles, each represented as
            (start_idx, end_idx, cycle_name, cycle_type).
    """
    std_dev = np.std(wave)
    if std_dev == 0:
        return []
    peaks, _ = find_peaks(
        wave, prominence=std_dev * 0.3, distance=max(1, len(wave) // 10)
    )
    troughs, _ = find_peaks(
        -wave, prominence=std_dev * 0.3, distance=max(1, len(wave) // 10)
    )
    cycles = []
    for i in range(len(troughs) - 1):
        cycles.append(
            (troughs[i], troughs[i + 1], f"Trough-Trough_{i+1:02d}", "trough-to-trough")
        )
    for i in range(len(peaks) - 1):
        cycles.append((peaks[i], peaks[i + 1], f"Peak-Peak_{i+1:02d}", "peak-to-peak"))
    return cycles


def find_best_cycle_pair_and_log_all(
    ap_wave: np.ndarray, lat_wave: np.ndarray, csv_path: str
) -> Tuple[Optional[Tuple], Optional[Tuple], float]:
    """
    Finds the best matching pair of respiratory cycles between AP and LAT waveforms based on Pearson correlation.
    Logs all comparisons to a CSV file.

    Args:
        ap_wave (np.ndarray): Filtered AP waveform.
        lat_wave (np.ndarray): Filtered LAT waveform.
        csv_path (str): Path to save the CSV log.

    Returns:
        Tuple: Best AP cycle info, Best LAT cycle info, Maximum correlation coefficient.
    """
    import csv

    ap_cycles = find_all_cycles(ap_wave)
    lat_cycles = find_all_cycles(lat_wave)
    if not ap_cycles or not lat_cycles:
        return None, None, -1.0

    max_corr = -1.0
    best_pair = (None, None)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AP_Cycle_ID", "LAT_Cycle_ID", "Correlation"])
        for ap_start, ap_end, ap_name, ap_type in ap_cycles:
            ap_segment = ap_wave[ap_start:ap_end]
            for lat_start, lat_end, lat_name, lat_type in lat_cycles:
                lat_segment = lat_wave[lat_start:lat_end]
                if ap_segment.size < 2 or lat_segment.size < 2:
                    continue

                if len(ap_segment) > len(lat_segment):
                    lat_segment_resampled = resample(lat_segment, len(ap_segment))
                    ap_segment_resampled = ap_segment
                else:
                    ap_segment_resampled = resample(ap_segment, len(lat_segment))
                    lat_segment_resampled = lat_segment

                corr = np.corrcoef(ap_segment_resampled, lat_segment_resampled)[0, 1]
                writer.writerow([ap_name, lat_name, f"{corr:.4f}"])

                if corr > max_corr:
                    max_corr = corr
                    best_pair = (
                        (ap_start, ap_end, ap_name, ap_type),
                        (lat_start, lat_end, lat_name, lat_type),
                    )

    logging.info(
        f"Found best cycle pair with Pearson correlation: {max_corr:.4f}. Full results in {csv_path}"
    )
    return best_pair[0], best_pair[1], max_corr


def get_phased_frames(
    wave: np.ndarray, cycle_info: Tuple
) -> Tuple[Dict[int, int], List[int]]:
    """
    Maps percentages of the respiratory cycle (phases 0-90) to frame indices.

    Args:
        wave (np.ndarray): Filtered 1D waveform.
        cycle_info (Tuple): The chosen cycle information (start_idx, end_idx, name, type).

    Returns:
        Tuple[Dict[int, int], List[int]]:
            - A dictionary mapping phase percentage (0, 10, ... 90) to frame index.
            - A list showing the chronological order of phases to display.
    """
    start_f, end_f, cycle_name, cycle_type = cycle_info
    cycle_wave = wave[start_f:end_f]

    peak_inhalation_frame = start_f + np.argmax(cycle_wave)
    peak_exhalation_frame = start_f + np.argmin(cycle_wave)

    interp_inhale = interp1d(
        [0, 50],
        [peak_exhalation_frame, peak_inhalation_frame],
        bounds_error=False,
        fill_value="extrapolate",
    )
    interp_exhale = interp1d(
        [50, 100],
        [peak_inhalation_frame, peak_exhalation_frame],
        bounds_error=False,
        fill_value="extrapolate",
    )

    base_phases = np.arange(0, 100, 10)

    if cycle_type == "trough-to-trough":
        target_phases = np.roll(base_phases, 5).tolist()
        logging.info(
            f"Trough-to-trough cycle selected. Chronological phases: {target_phases}"
        )
    else:
        target_phases = base_phases.tolist()
        logging.info(
            f"Peak-to-peak cycle selected. Chronological phases: {target_phases}"
        )

    phased_frames = {}
    for phase in target_phases:
        if phase <= 50:
            frame_idx = interp_inhale(phase)
        else:
            frame_idx = interp_exhale(phase)
        phased_frames[phase] = int(np.round(frame_idx))

    for phase, frame in phased_frames.items():
        phased_frames[phase] = np.clip(frame, 0, len(wave) - 1)

    return phased_frames, target_phases
