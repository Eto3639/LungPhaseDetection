import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import sys
import pydicom
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fpdf import FPDF
from pathlib import Path
import warnings
import logging
import io
import time
import csv
from scipy.interpolate import interp1d
from PIL import Image

from xray_segmentation.models import PretrainedUNet
from torchvision.transforms import Compose, Resize, ToTensor

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=UserWarning, module='scipy.signal._peak_finding')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Model & Pre-processing ---
def load_xray_segmentation_model(model_path="xray_segmentation/unet_lung_segmentation.pth"):
    """Loads the pre-trained U-Net model for X-ray lung segmentation."""
    model = PretrainedUNet(in_channels=1, out_channels=1, batch_norm=True, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    logging.info("X-ray lung segmentation model loaded successfully.")
    return model

def segment_lung_xray(model, image_np):
    """Performs lung segmentation on a single X-ray image."""
    original_size = image_np.shape
    image_pil = Image.fromarray(image_np).convert('L')

    preprocess = Compose([
        Resize((512, 512)),
        ToTensor(),
    ])

    img_tensor = preprocess(image_pil)
    img_tensor = img_tensor - 0.5
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    mask = torch.sigmoid(output).squeeze().cpu().numpy() > 0.5
    mask = mask.astype(np.uint8)

    mask_pil = Image.fromarray(mask)
    mask_resized = mask_pil.resize((original_size[1], original_size[0]), Image.NEAREST)

    return np.array(mask_resized)

# --- Main Analysis Functions ---

def create_output_directories(case_id: str) -> dict:
    base_path = Path('output')
    paths = {
        "png_ap": base_path / "PNG" / case_id / "AP",
        "png_lat": base_path / "PNG" / case_id / "LAT",
        "pt_ap": base_path / "pt" / case_id / "AP",
        "pt_lat": base_path / "pt" / case_id / "LAT",
        "report": base_path / "report",
        "video": base_path / "PNG" / case_id,
        "csv": base_path / "report" / "csv_details"
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directories for case {case_id}")
    return {key: str(value) for key, value in paths.items()}

def load_multiframe_dicom(dicom_filepath: str) -> tuple[pydicom.Dataset, float]:
    dicom_path = Path(dicom_filepath)
    if not dicom_path.is_file():
        raise FileNotFoundError(f"DICOM file not found: {dicom_filepath}")
    ds = pydicom.dcmread(str(dicom_path))
    if not hasattr(ds, 'NumberOfFrames') or ds.NumberOfFrames <= 1:
        raise ValueError(f"File {dicom_filepath} is not a valid multi-frame DICOM.")
    frame_rate = 30
    if 'FrameTime' in ds and ds.FrameTime > 0:
        frame_rate = 1000 / ds.FrameTime
    elif 'RecommendedDisplayFrameRate' in ds and ds.RecommendedDisplayFrameRate > 0:
        frame_rate = ds.RecommendedDisplayFrameRate
    logging.info(f"Read {ds.NumberOfFrames} frames from {dicom_filepath} with a frame rate of {frame_rate:.2f} fps.")
    return ds, frame_rate

def extract_waveform_centroid(dicom_ds: pydicom.Dataset) -> np.ndarray:
    wave = []
    for frame_2d in dicom_ds.pixel_array:
        img = frame_2d.astype(np.float32)
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = img_8bit.shape
        roi = img_8bit[h // 2:, :]
        M = cv2.moments(roi)
        cy = (M["m01"] / M["m00"] + (h // 2)) if M["m00"] != 0 else (wave[-1] if wave else h * 0.75)
        wave.append(cy)
    logging.info(f"Extracted waveform using CENTROID method.")
    return np.array(wave)

def extract_waveform_roi_pca(dicom_ds: pydicom.Dataset, model: nn.Module) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel_array_3d = dicom_ds.pixel_array
    num_frames, height, width = pixel_array_3d.shape

    middle_frame_idx = num_frames // 2
    middle_frame = pixel_array_3d[middle_frame_idx]
    roi_mask = segment_lung_xray(model, middle_frame)

    if np.sum(roi_mask) == 0:
        logging.warning("Lung segmentation failed; ROI is empty. Falling back to global PCA.")
        roi_mask = np.ones((height, width), dtype=np.uint8)

    data_matrix = pixel_array_3d.reshape(num_frames, height * width)
    roi_pixels = data_matrix[:, roi_mask.flatten() > 0]

    if roi_pixels.shape[1] == 0:
        logging.error("Cannot perform PCA on an empty ROI. Returning a flat waveform.")
        middle_frame_8bit = cv2.normalize(pixel_array_3d[middle_frame_idx], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        roi_viz = cv2.cvtColor(middle_frame_8bit, cv2.COLOR_GRAY2BGR)
        return np.zeros(num_frames), np.zeros((height, width), dtype=np.uint8), roi_viz

    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(roi_pixels).flatten()

    middle_frame_8bit = cv2.normalize(middle_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi_viz = cv2.cvtColor(middle_frame_8bit, cv2.COLOR_GRAY2BGR)
    roi_viz[roi_mask > 0, 1] = 255

    logging.info(f"Extracted waveform using ROI-PCA method.")
    return pca_result, roi_mask, roi_viz

def process_waveform(raw_wave: np.ndarray, frame_rate: float, cutoff_hz: float = 0.5) -> np.ndarray:
    nyquist = 0.5 * frame_rate
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_wave = filtfilt(b, a, raw_wave)
    logging.info(f"Filtered waveform with a {cutoff_hz} Hz low-pass filter.")
    return filtered_wave

def find_all_cycles(wave: np.ndarray) -> list[tuple[int, int, str, str]]:
    std_dev = np.std(wave)
    if std_dev == 0: return []
    peaks, _ = find_peaks(wave, prominence=std_dev * 0.3, distance=len(wave)//10)
    troughs, _ = find_peaks(-wave, prominence=std_dev * 0.3, distance=len(wave)//10)
    cycles = []
    for i in range(len(troughs) - 1):
        cycles.append((troughs[i], troughs[i+1], f"Trough-Trough_{i+1:02d}", "trough-to-trough"))
    for i in range(len(peaks) - 1):
        cycles.append((peaks[i], peaks[i+1], f"Peak-Peak_{i+1:02d}", "peak-to-peak"))
    return cycles

def find_best_cycle_pair_and_log_all(ap_wave: np.ndarray, lat_wave: np.ndarray, csv_path: str) -> tuple:
    ap_cycles = find_all_cycles(ap_wave)
    lat_cycles = find_all_cycles(lat_wave)
    if not ap_cycles or not lat_cycles:
        return None, None, -1.0
    max_corr = -1.0
    best_pair = (None, None)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AP_Cycle_ID', 'LAT_Cycle_ID', 'Correlation'])
        for ap_start, ap_end, ap_name, ap_type in ap_cycles:
            ap_segment = ap_wave[ap_start:ap_end]
            for lat_start, lat_end, lat_name, lat_type in lat_cycles:
                lat_segment = lat_wave[lat_start:lat_end]
                if ap_segment.size < 2 or lat_segment.size < 2: continue
                if len(ap_segment) > len(lat_segment):
                    lat_segment_resampled = resample(lat_segment, len(ap_segment))
                    ap_segment_resampled = ap_segment
                else:
                    ap_segment_resampled = resample(ap_segment, len(lat_segment))
                    lat_segment_resampled = lat_segment
                corr = np.corrcoef(ap_segment_resampled, lat_segment_resampled)[0, 1]
                writer.writerow([ap_name, lat_name, f'{corr:.4f}'])
                if corr > max_corr:
                    max_corr = corr
                    best_pair = ((ap_start, ap_end, ap_name, ap_type), (lat_start, lat_end, lat_name, lat_type))
    logging.info(f"Found best cycle pair with Pearson correlation: {max_corr:.4f}. Full results in {csv_path}")
    return best_pair[0], best_pair[1], max_corr

def get_phased_frames(wave: np.ndarray, cycle_info: tuple) -> tuple[dict[int, int], list[int]]:
    start_f, end_f, cycle_name, cycle_type = cycle_info
    cycle_wave = wave[start_f:end_f]
    peak_inhalation_frame = start_f + np.argmax(cycle_wave)
    peak_exhalation_frame = start_f + np.argmin(cycle_wave)
    interp_inhale = interp1d([0, 50], [peak_exhalation_frame, peak_inhalation_frame], bounds_error=False, fill_value="extrapolate")
    interp_exhale = interp1d([50, 100], [peak_inhalation_frame, peak_exhalation_frame], bounds_error=False, fill_value="extrapolate")
    base_phases = np.arange(0, 100, 10)
    if cycle_type == 'trough-to-trough':
        target_phases = np.roll(base_phases, 5).tolist()
        logging.info("Trough-to-trough cycle selected. Chronological phases: %s", target_phases)
    else:
        target_phases = base_phases.tolist()
        logging.info("Peak-to-peak cycle selected. Chronological phases: %s", target_phases)
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

def save_outputs(dicom_ds: pydicom.Dataset, phased_frames: dict[int, int], png_dir: str, pt_dir: str):
    pixel_array_3d = dicom_ds.pixel_array
    for phase, frame_idx in phased_frames.items():
        if not (0 <= frame_idx < len(pixel_array_3d)):
            logging.warning(f"Skipping phase {phase} due to invalid frame index: {frame_idx}")
            continue
        img = pixel_array_3d[frame_idx].astype(np.float32)
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_inverted = 255 - img_8bit
        cv2.imwrite(str(Path(png_dir) / f"phase_{phase:02d}.png"), img_inverted)
        torch.save(torch.from_numpy(img), Path(pt_dir) / f"phase_{phase:02d}.pt")
    logging.info(f"Saved PNGs to {png_dir} and PT files to {pt_dir}")

def create_comparison_video(case_id: str, png_ap_dir: str, png_lat_dir: str, video_path: str, phase_order: list[int], fps: int = 3):
    ap_images = [Path(png_ap_dir) / f"phase_{p:02d}.png" for p in phase_order]
    ap_images_exist = [p for p in ap_images if p.exists()]
    if len(ap_images_exist) != len(phase_order):
        logging.warning("Could not create video: Some AP PNG files are missing.")
        if not ap_images_exist: logging.error("No frames available for video creation."); return
    frame1_ap = cv2.imread(str(ap_images_exist[0]), cv2.IMREAD_GRAYSCALE)
    lat_path_1 = Path(png_lat_dir) / ap_images_exist[0].name
    if not lat_path_1.exists(): logging.error(f"Missing LAT image for {ap_images_exist[0].name}"); return
    frame1_lat = cv2.imread(str(lat_path_1), cv2.IMREAD_GRAYSCALE)
    h_ap, w_ap = frame1_ap.shape; h_lat, w_lat = frame1_lat.shape
    scale = h_ap / h_lat; w_lat_new = int(w_lat * scale)
    height, width = h_ap, w_ap + w_lat_new
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)
    for ap_path in ap_images_exist:
        lat_path = Path(png_lat_dir) / ap_path.name
        if not lat_path.exists(): continue
        ap_img = cv2.imread(str(ap_path), cv2.IMREAD_GRAYSCALE)
        lat_img = cv2.imread(str(lat_path), cv2.IMREAD_GRAYSCALE)
        lat_resized = cv2.resize(lat_img, (w_lat_new, h_ap), interpolation=cv2.INTER_AREA)
        combined_frame = np.hstack((ap_img, lat_resized))
        video_writer.write(combined_frame)
    video_writer.release()
    logging.info(f"Created comparison video: {video_path}")

class PDF(FPDF):
    def header(self): self.set_font('Helvetica', 'B', 12); self.cell(0, 10, 'Respiratory Waveform Analysis Report', 0, 1, 'C')
    def footer(self): self.set_y(-15); self.set_font('Helvetica', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def chapter_title(self, title): self.set_font('Helvetica', 'B', 12); self.cell(0, 10, title, 0, 1, 'L'); self.ln(5)
    def chapter_body(self, body_text, image_data=None):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, body_text)
        self.ln()
        if image_data: self.image(image_data, x=10, w=self.w - 20); self.ln()

def create_report(case_id, paths, data):
    report_path = Path(paths["report"]) / f"{case_id}_report_v13.pdf"
    plot_data = {}
    winning_ap_method, winning_lat_method = data['winning_methods']
    for view in ['ap', 'lat']:
        buf = io.BytesIO()
        plt.figure(figsize=(8, 8)); plt.imshow(data[f'{view}_roi_viz']); plt.title(f'{view.upper()} Lung ROI'); plt.axis('off')
        plt.savefig(buf, format='png', bbox_inches='tight'); plt.close()
        buf.seek(0)
        plot_data[f"{view}_roi_viz"] = buf
    for view in ['ap', 'lat']:
        for method in ['centroid', 'roi_pca']:
            buf = io.BytesIO()
            plt.figure(figsize=(12, 4))
            plt.plot(data[f'{view}_{method}_raw'], label='Raw', alpha=0.5)
            plt.plot(data[f'{view}_{method}_filtered'], label='Filtered', linewidth=2)
            winning_method = winning_ap_method if view == 'ap' else winning_lat_method
            if method == winning_method:
                start, end, name, type = data[f'{view}_cycle']
                plt.axvspan(start, end, color='red', alpha=0.2, label=f'Best Cycle ({name})')
            plt.title(f'Case {case_id} - {view.upper()} Respiratory Waveform ({method.upper()})')
            plt.xlabel('Frame Number'); plt.ylabel('Signal Amplitude')
            plt.legend(); plt.grid(True)
            plt.savefig(buf, format='png', bbox_inches='tight'); plt.close()
            buf.seek(0)
            plot_data[f"{view}_{method}"] = buf
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title(f'Analysis Summary for Case: {case_id}')
    summary_text = (f"The best matching cycle pair was found between AP ({winning_ap_method.upper()}) and LAT ({winning_lat_method.upper()}) with a Pearson correlation of {data['max_corr']:.4f}.")
    pdf.chapter_body(summary_text)
    pdf.chapter_title('1. Lung Field Detection (ROI)')
    pdf.chapter_body('The following masks were generated by the U-Net model and used as the ROI for the ROI-PCA method.', image_data=plot_data['ap_roi_viz'])
    pdf.chapter_body('', image_data=plot_data['lat_roi_viz'])
    pdf.chapter_title('2. AP Waveform Analysis')
    pdf.chapter_body('2-A: CENTROID Method Waveform.', image_data=plot_data['ap_centroid'])
    pdf.chapter_body('2-B: ROI-PCA Method Waveform.', image_data=plot_data['ap_roi_pca'])
    pdf.chapter_title('3. LAT Waveform Analysis')
    pdf.chapter_body('3-A: CENTROID Method Waveform.', image_data=plot_data['lat_centroid'])
    pdf.chapter_body('3-B: ROI-PCA Method Waveform.', image_data=plot_data['lat_roi_pca'])
    pdf.chapter_title('4. Correlation Summary Table')
    pdf.set_font('Helvetica', 'B', 10); line_height = pdf.font_size * 1.5; col_widths = (pdf.epw * 0.3, pdf.epw * 0.3, pdf.epw * 0.4)
    pdf.cell(col_widths[0], line_height, 'AP Method', border=1, align='C'); pdf.cell(col_widths[1], line_height, 'LAT Method', border=1, align='C'); pdf.cell(col_widths[2], line_height, 'Highest Correlation', border=1, align='C'); pdf.ln(line_height)
    pdf.set_font('Helvetica', '', 10)
    for methods, corr in data['all_results'].items():
        pdf.cell(col_widths[0], line_height, methods[0].upper(), border=1); pdf.cell(col_widths[1], line_height, methods[1].upper(), border=1); pdf.cell(col_widths[2], line_height, f'{corr:.4f}', border=1, align='C'); pdf.ln(line_height)
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 10, f"Total script execution time: {data['execution_time']:.2f} seconds. Detailed CSV logs are in the report/csv_details directory.")
    pdf.output(str(report_path))
    logging.info(f"Created PDF report: {report_path}")

def main():
    start_time = time.time()
    if len(sys.argv) != 4:
        print("Usage: python respiratory_analysis.py <case_id> <ap_dicom_path> <lat_dicom_path>")
        sys.exit(1)
    case_id, ap_path, lat_path = sys.argv[1], sys.argv[2], sys.argv[3]
    logging.info(f"--- Starting Respiratory Analysis (v13) ---")
    try:
        paths = create_output_directories(case_id)
        model = load_xray_segmentation_model()
        ap_ds, ap_fps = load_multiframe_dicom(ap_path)
        lat_ds, lat_fps = load_multiframe_dicom(lat_path)
        waves = {
            'ap': {'fps': ap_fps},
            'lat': {'fps': lat_fps}
        }
        waves['ap']['centroid_raw'] = extract_waveform_centroid(ap_ds)
        waves['lat']['centroid_raw'] = extract_waveform_centroid(lat_ds)
        ap_roi_pca_raw, ap_roi_mask, ap_roi_viz = extract_waveform_roi_pca(ap_ds, model)
        lat_roi_pca_raw, lat_roi_mask, lat_roi_viz = extract_waveform_roi_pca(lat_ds, model)
        waves['ap']['roi_pca_raw'] = ap_roi_pca_raw
        waves['lat']['roi_pca_raw'] = lat_roi_pca_raw
        for view in ['ap', 'lat']:
            for method in ['centroid', 'roi_pca']:
                waves[view][f'{method}_filtered'] = process_waveform(waves[view][f'{method}_raw'], waves[view]['fps'])
        overall_max_corr = -1.0
        winning_config = {}
        all_results = {}
        method_combinations = [('centroid', 'centroid'), ('centroid', 'roi_pca'), ('roi_pca', 'centroid'), ('roi_pca', 'roi_pca')]
        for ap_method, lat_method in method_combinations:
            csv_path = Path(paths["csv"]) / f"{case_id}_AP_{ap_method.upper()}_vs_LAT_{lat_method.upper()}_all_correlations.csv"
            logging.info(f"--- Analyzing combination: AP({ap_method.upper()}) vs LAT({lat_method.upper()}) ---")
            ap_wave = waves['ap'][f'{ap_method}_filtered']
            lat_wave = waves['lat'][f'{lat_method}_filtered']
            ap_cycle, lat_cycle, max_corr = find_best_cycle_pair_and_log_all(ap_wave, lat_wave, str(csv_path))
            all_results[(ap_method, lat_method)] = max_corr
            if max_corr > overall_max_corr:
                overall_max_corr = max_corr
                winning_config = {'ap_method': ap_method, 'lat_method': lat_method, 'ap_cycle': ap_cycle, 'lat_cycle': lat_cycle}
        if not winning_config: raise ValueError("Failed to find any valid cycle pairs.")
        logging.info(f"--- Overall Best Result: AP({winning_config['ap_method'].upper()}) vs LAT({winning_config['lat_method'].upper()}) with Pearson corr {overall_max_corr:.4f} ---")
        winning_ap_wave = waves['ap'][f"{winning_config['ap_method']}_filtered"]
        winning_lat_wave = waves['lat'][f"{winning_config['lat_method']}_filtered"]
        ap_phased_frames, ap_phase_order = get_phased_frames(winning_ap_wave, winning_config['ap_cycle'])
        lat_phased_frames, lat_phase_order = get_phased_frames(winning_lat_wave, winning_config['lat_cycle'])
        save_outputs(ap_ds, ap_phased_frames, paths['png_ap'], paths['pt_ap'])
        save_outputs(lat_ds, lat_phased_frames, paths['png_lat'], paths['pt_lat'])
        create_comparison_video(case_id, paths['png_ap'], paths['png_lat'], str(Path(paths['video']) / 'ap_lat_compare.mp4'), ap_phase_order)
        end_time = time.time()
        execution_time = end_time - start_time
        report_data = {
            'ap_centroid_raw': waves['ap']['centroid_raw'], 'ap_centroid_filtered': waves['ap']['centroid_filtered'],
            'lat_centroid_raw': waves['lat']['centroid_raw'], 'lat_centroid_filtered': waves['lat']['centroid_filtered'],
            'ap_roi_pca_raw': waves['ap']['roi_pca_raw'], 'ap_roi_pca_filtered': waves['ap']['roi_pca_filtered'],
            'lat_roi_pca_raw': waves['lat']['roi_pca_raw'], 'lat_roi_pca_filtered': waves['lat']['roi_pca_filtered'],
            'ap_cycle': winning_config['ap_cycle'], 'lat_cycle': winning_config['lat_cycle'],
            'max_corr': overall_max_corr,
            'winning_methods': (winning_config['ap_method'], winning_config['lat_method']),
            'all_results': all_results,
            'execution_time': execution_time,
            'ap_roi_viz': ap_roi_viz, 'lat_roi_viz': lat_roi_viz
        }
        create_report(case_id, paths, report_data)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    logging.info(f"--- Analysis successfully completed in {execution_time:.2f} seconds! ---")

if __name__ == '__main__':
    main()
