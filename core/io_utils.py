import pydicom
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List


def create_output_directories(case_id: str) -> Dict[str, str]:
    """
    Creates necessary output directories for the given case ID.

    Args:
        case_id (str): The unique identifier for the patient case.

    Returns:
        Dict[str, str]: Dictionary mapping directory names to their paths.
    """
    base_path = Path("output")
    paths = {
        "png_ap": base_path / "PNG" / case_id / "AP",
        "png_lat": base_path / "PNG" / case_id / "LAT",
        "pt_ap": base_path / "pt" / case_id / "AP",
        "pt_lat": base_path / "pt" / case_id / "LAT",
        "report": base_path / "report",
        "video": base_path / "PNG" / case_id,
        "csv": base_path / "report" / "csv_details",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directories for case {case_id}")
    return {key: str(value) for key, value in paths.items()}


def load_multiframe_dicom(dicom_filepath: str) -> Tuple[pydicom.Dataset, float]:
    """
    Loads a multi-frame DICOM file and extracts its frame rate.

    Args:
        dicom_filepath (str): Path to the DICOM file.

    Returns:
        Tuple[pydicom.Dataset, float]: The loaded DICOM dataset and its calculated frame rate (FPS).

    Raises:
        FileNotFoundError: If the DICOM file does not exist.
        ValueError: If the file is not a valid multi-frame DICOM.
    """
    dicom_path = Path(dicom_filepath)
    if not dicom_path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {dicom_filepath}")
    ds = pydicom.dcmread(str(dicom_path))
    if not hasattr(ds, "NumberOfFrames") or ds.NumberOfFrames <= 1:
        raise ValueError(f"File {dicom_filepath} is not a valid multi-frame DICOM.")

    frame_rate = 30.0
    if "FrameTime" in ds and ds.FrameTime > 0:
        frame_rate = 1000.0 / ds.FrameTime
    elif "RecommendedDisplayFrameRate" in ds and ds.RecommendedDisplayFrameRate > 0:
        frame_rate = float(ds.RecommendedDisplayFrameRate)

    logging.info(
        f"Read {ds.NumberOfFrames} frames from {dicom_filepath} with a frame rate of {frame_rate:.2f} fps."
    )
    return ds, frame_rate


def save_outputs(
    dicom_ds: pydicom.Dataset, phased_frames: Dict[int, int], png_dir: str, pt_dir: str
):
    """
    Saves selected phased frames as PNG images and PyTorch tensors (.pt).

    Args:
        dicom_ds (pydicom.Dataset): The original DICOM dataset.
        phased_frames (Dict[int, int]): Mapping of phase percentages to frame indices.
        png_dir (str): Directory to save PNG images.
        pt_dir (str): Directory to save PyTorch tensors.
    """
    pixel_array_3d = dicom_ds.pixel_array
    for phase, frame_idx in phased_frames.items():
        if not (0 <= frame_idx < len(pixel_array_3d)):
            logging.warning(
                f"Skipping phase {phase} due to invalid frame index: {frame_idx}"
            )
            continue

        img = pixel_array_3d[frame_idx].astype(np.float32)
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_inverted = 255 - img_8bit

        cv2.imwrite(str(Path(png_dir) / f"phase_{phase:02d}.png"), img_inverted)
        torch.save(torch.from_numpy(img), Path(pt_dir) / f"phase_{phase:02d}.pt")

    logging.info(f"Saved PNGs to {png_dir} and PT files to {pt_dir}")


def create_comparison_video(
    case_id: str,
    png_ap_dir: str,
    png_lat_dir: str,
    video_path: str,
    phase_order: List[int],
    fps: int = 3,
):
    """
    Creates a side-by-side comparison video of AP and LAT view frames.

    Args:
        case_id (str): The case identifier.
        png_ap_dir (str): Directory containing AP PNG images.
        png_lat_dir (str): Directory containing LAT PNG images.
        video_path (str): File path to save the output MP4 video.
        phase_order (List[int]): Chronological order of phases.
        fps (int, optional): Frames per second of the output video. Defaults to 3.
    """
    ap_images = [Path(png_ap_dir) / f"phase_{p:02d}.png" for p in phase_order]
    ap_images_exist = [p for p in ap_images if p.exists()]

    if len(ap_images_exist) != len(phase_order):
        logging.warning("Could not create video: Some AP PNG files are missing.")
        if not ap_images_exist:
            logging.error("No frames available for video creation.")
            return

    frame1_ap = cv2.imread(str(ap_images_exist[0]), cv2.IMREAD_GRAYSCALE)
    lat_path_1 = Path(png_lat_dir) / ap_images_exist[0].name
    if not lat_path_1.exists():
        logging.error(f"Missing LAT image for {ap_images_exist[0].name}")
        return

    frame1_lat = cv2.imread(str(lat_path_1), cv2.IMREAD_GRAYSCALE)

    h_ap, w_ap = frame1_ap.shape
    h_lat, w_lat = frame1_lat.shape
    scale = h_ap / h_lat
    w_lat_new = int(w_lat * scale)

    height, width = h_ap, w_ap + w_lat_new
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        video_path, fourcc, fps, (width, height), isColor=False
    )

    for ap_path in ap_images_exist:
        lat_path = Path(png_lat_dir) / ap_path.name
        if not lat_path.exists():
            continue

        ap_img = cv2.imread(str(ap_path), cv2.IMREAD_GRAYSCALE)
        lat_img = cv2.imread(str(lat_path), cv2.IMREAD_GRAYSCALE)
        lat_resized = cv2.resize(
            lat_img, (w_lat_new, h_ap), interpolation=cv2.INTER_AREA
        )

        combined_frame = np.hstack((ap_img, lat_resized))
        video_writer.write(combined_frame)

    video_writer.release()
    logging.info(f"Created comparison video: {video_path}")
