# Lung Phase Detection

A repository for analyzing respiratory waveforms from multi-frame DICOM X-ray sequences, applying lung segmentation using AI models, and identifying respiratory cycles to extract specific phased frames.

## Features
- **AI-based Lung Segmentation**: Uses `torchxrayvision` (PSPNet) to automatically segment the lung regions and calculate ROI based features.
- **Waveform Extraction**: Provides two methods for extracting respiratory signals:
  - `CENTROID`: Tracks the movement of the image centroid in the lower half of the frame.
  - `ROI-PCA`: Applies Principal Component Analysis (PCA) to pixel intensities within the segmented lung region.
- **Cycle Matching**: Automatically finds the best peak-to-peak or trough-to-trough respiratory cycles between AP (Anteroposterior) and LAT (Lateral) views based on Pearson correlation.
- **Phase Mapping**: Extracts exactly 10 phase frames (0% to 90%) representing one full respiratory cycle.
- **Reporting**: Generates a detailed PDF report with waveform plots and saves a side-by-side comparative MP4 video of AP and LAT view breathing phases.

## Directory Structure
```
.
├── analyze.py            # Main entry point script
├── core/                 # Core modular packages
│   ├── __init__.py
│   ├── io_utils.py       # DICOM loading, video/image generation
│   ├── model.py          # AI segmentation models
│   ├── reporting.py      # PDF reporting
│   └── waveform.py       # Waveform extraction and processing
├── requirements.txt      # Dependency list
├── setup.cfg             # Linter configs
└── tests/                # Unit tests
```

## Installation

1. Clone the repository.
2. It's recommended to create a virtual environment (`python -m venv venv` and `source venv/bin/activate`).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main analysis script providing a Case ID and the paths to both AP and LAT DICOM files:

```bash
python analyze.py <case_id> <path/to/ap.dcm> <path/to/lat.dcm>
```

### Output Directory
All results will be saved in the `output/` directory, structured by case ID:
- `output/PNG/<case_id>/`: Contains phased PNG images and the comparative video.
- `output/pt/<case_id>/`: Contains serialized PyTorch tensors of the images.
- `output/report/`: Contains the generated PDF report and CSV logs detailing correlation metrics.

## Development and Testing

Code formatting is enforced with `black` and linting with `flake8`.
Unit tests are written with `pytest`.

To run the test suite:
```bash
PYTHONPATH=. pytest tests/
```
