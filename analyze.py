import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Fix for OpenMP duplicate lib issue on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from core.model import load_xray_segmentation_model
from core.waveform import (
    extract_waveform_centroid,
    extract_waveform_roi_pca,
    process_waveform,
    find_best_cycle_pair_and_log_all,
    get_phased_frames,
)
from core.io_utils import (
    create_output_directories,
    load_multiframe_dicom,
    save_outputs,
    create_comparison_video,
)
from core.reporting import create_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Respiratory Waveforms from multi-frame DICOM X-rays."
    )
    parser.add_argument(
        "case_id", type=str, help="Unique identifier for the patient case."
    )
    parser.add_argument(
        "ap_dicom", type=str, help="Path to the AP view multi-frame DICOM file."
    )
    parser.add_argument(
        "lat_dicom", type=str, help="Path to the LAT view multi-frame DICOM file."
    )
    return parser.parse_args()


def main():
    """Main execution function for the respiratory analysis script."""
    start_time = time.time()
    args = parse_args()

    case_id = args.case_id
    ap_path = args.ap_dicom
    lat_path = args.lat_dicom

    logging.info(f"--- Starting Respiratory Analysis for Case {case_id} ---")

    try:
        paths = create_output_directories(case_id)
        model = load_xray_segmentation_model()

        ap_ds, ap_fps = load_multiframe_dicom(ap_path)
        lat_ds, lat_fps = load_multiframe_dicom(lat_path)

        waves = {"ap": {"fps": ap_fps}, "lat": {"fps": lat_fps}}

        # Centroid Extraction
        waves["ap"]["centroid_raw"] = extract_waveform_centroid(ap_ds)
        waves["lat"]["centroid_raw"] = extract_waveform_centroid(lat_ds)

        # ROI-PCA Extraction
        ap_roi_pca_raw, ap_roi_mask, ap_roi_viz = extract_waveform_roi_pca(ap_ds, model)
        lat_roi_pca_raw, lat_roi_mask, lat_roi_viz = extract_waveform_roi_pca(
            lat_ds, model
        )
        waves["ap"]["roi_pca_raw"] = ap_roi_pca_raw
        waves["lat"]["roi_pca_raw"] = lat_roi_pca_raw

        # Filtering
        for view in ["ap", "lat"]:
            for method in ["centroid", "roi_pca"]:
                waves[view][f"{method}_filtered"] = process_waveform(
                    waves[view][f"{method}_raw"], waves[view]["fps"]
                )

        # Find best cycle match
        overall_max_corr = -1.0
        winning_config = {}
        all_results = {}

        method_combinations = [
            ("centroid", "centroid"),
            ("centroid", "roi_pca"),
            ("roi_pca", "centroid"),
            ("roi_pca", "roi_pca"),
        ]

        for ap_method, lat_method in method_combinations:
            csv_path = (
                Path(paths["csv"])
                / f"{case_id}_AP_{ap_method.upper()}_vs_LAT_{lat_method.upper()}_all_correlations.csv"
            )
            logging.info(
                f"--- Analyzing combination: AP({ap_method.upper()}) vs LAT({lat_method.upper()}) ---"
            )

            ap_wave = waves["ap"][f"{ap_method}_filtered"]
            lat_wave = waves["lat"][f"{lat_method}_filtered"]

            ap_cycle, lat_cycle, max_corr = find_best_cycle_pair_and_log_all(
                ap_wave, lat_wave, str(csv_path)
            )
            all_results[(ap_method, lat_method)] = max_corr

            if max_corr > overall_max_corr:
                overall_max_corr = max_corr
                winning_config = {
                    "ap_method": ap_method,
                    "lat_method": lat_method,
                    "ap_cycle": ap_cycle,
                    "lat_cycle": lat_cycle,
                }

        if not winning_config or winning_config.get("ap_cycle") is None:
            raise ValueError("Failed to find any valid cycle pairs.")

        logging.info(
            f"--- Overall Best Result: AP({winning_config['ap_method'].upper()}) "
            f"vs LAT({winning_config['lat_method'].upper()}) with Pearson corr {overall_max_corr:.4f} ---"
        )

        winning_ap_wave = waves["ap"][f"{winning_config['ap_method']}_filtered"]
        winning_lat_wave = waves["lat"][f"{winning_config['lat_method']}_filtered"]

        ap_phased_frames, ap_phase_order = get_phased_frames(
            winning_ap_wave, winning_config["ap_cycle"]
        )
        lat_phased_frames, lat_phase_order = get_phased_frames(
            winning_lat_wave, winning_config["lat_cycle"]
        )

        # Save Outputs
        save_outputs(ap_ds, ap_phased_frames, paths["png_ap"], paths["pt_ap"])
        save_outputs(lat_ds, lat_phased_frames, paths["png_lat"], paths["pt_lat"])
        create_comparison_video(
            case_id,
            paths["png_ap"],
            paths["png_lat"],
            str(Path(paths["video"]) / "ap_lat_compare.mp4"),
            ap_phase_order,
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Create Report
        report_data = {
            "ap_centroid_raw": waves["ap"]["centroid_raw"],
            "ap_centroid_filtered": waves["ap"]["centroid_filtered"],
            "lat_centroid_raw": waves["lat"]["centroid_raw"],
            "lat_centroid_filtered": waves["lat"]["centroid_filtered"],
            "ap_roi_pca_raw": waves["ap"]["roi_pca_raw"],
            "ap_roi_pca_filtered": waves["ap"]["roi_pca_filtered"],
            "lat_roi_pca_raw": waves["lat"]["roi_pca_raw"],
            "lat_roi_pca_filtered": waves["lat"]["roi_pca_filtered"],
            "ap_cycle": winning_config["ap_cycle"],
            "lat_cycle": winning_config["lat_cycle"],
            "max_corr": overall_max_corr,
            "winning_methods": (
                winning_config["ap_method"],
                winning_config["lat_method"],
            ),
            "all_results": all_results,
            "execution_time": execution_time,
            "ap_roi_viz": ap_roi_viz,
            "lat_roi_viz": lat_roi_viz,
        }
        create_report(case_id, paths, report_data)

        logging.info(
            f"--- Analysis successfully completed in {execution_time:.2f} seconds! ---"
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
