# lung-4d-pipeline

End-to-end QA toolkit for the 2D→4D lung-imaging pipeline. Three coordinated
components live in this monorepo:

| Subsystem | Purpose | Top-level entry points |
|---|---|---|
| **Phase detection** (upstream input prep) | Pair AP + lateral dynamic X-ray frames into one-cycle clips for the 2D-4D model. Amsterdam Shroud + cycle pairing, classical / AI-free. | `scripts/dirlab_demo.py`, `scripts/tcia_4d_lung_demo.py` |
| **DVF QA** (intermediate output) | Validate predicted displacement vector fields: Jacobian determinant, folding analysis, warped CT similarity. | `scripts/dvf_predict_qa.py` |
| **Generated CT QA** (downstream output) | Evaluate AI-generated synthetic CT: geometric integrity, dosimetric accuracy, temporal motion, robustness checks. | `generated_ct_qa/src/main.py` |

The original DVF QA module (`dvf_qa/`) still hosts the shared infrastructure
that all three subsystems reuse (image I/O, DRR rendering via DiffDRR,
HTML/PDF report builder, Notion / Slack / Pages automation).

## Legacy section: DVF QA (3D DVF validation)

QA utilities for 3D displacement vector fields estimated from fluoroscopic images and CT.

The main checks are:

- Jacobian determinant of `T(x) = x + u(x)`
- folding volume and connected components where `J <= 0`
- DVF magnitude and gradient/smoothness metrics
- warped CT similarity against a target CT
- DRR reprojection from generated CT compared with input fluoroscopy

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## GPU Docker

Build the CUDA/PyTorch image:

```bash
docker build --platform linux/amd64 -t dvf-qa:gpu .
```

Run on an NVIDIA GPU machine:

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data:ro \
  -v /path/to/outputs:/outputs \
  dvf-qa:gpu run \
    --dvf /data/predicted_dvf.nii.gz \
    --fixed /data/fixed_ct.nii.gz \
    --moving /data/moving_ct.nii.gz \
    --lung-mask /data/lung_mask.nii.gz \
    --generated-ct /data/generated_ct_or_dicom_series \
    --fluoro /data/input_fluoro.png \
    --geometry /data/geometry.json \
    --device cuda \
    --out /outputs/case_001
```

See `docs/deployment_and_productization.md` for productization, QA traceability, and regulatory-readiness notes.

## Minimal Jacobian QA

```bash
dvf-qa run \
  --dvf predicted_dvf.nii.gz \
  --fixed fixed_ct.nii.gz \
  --moving moving_ct.nii.gz \
  --lung-mask lung_mask.nii.gz \
  --out qa_out
```

The DVF is assumed to be a displacement field in millimeters with vector components `(ux, uy, uz)`.

## DRR Reprojection QA with DiffDRR

```bash
dvf-qa run \
  --generated-ct /path/to/dicom_series_or_ct.nii.gz \
  --fluoro input_fluoro.png \
  --geometry geometry.json \
  --out qa_out
```

Example `geometry.json`:

```json
{
  "sdd": 1020.0,
  "detector_shape": [512, 512],
  "pixel_spacing_mm": [0.8, 0.8],
  "rotation": [0.0, 0.0, 0.0],
  "translation": [0.0, 850.0, 0.0],
  "parameterization": "euler_angles",
  "convention": "ZXY",
  "orientation": "AP",
  "renderer": "siddon",
  "ct_value_mode": "hu"
}
```

DRR rendering uses `DiffDRR`. The CT input may be a DICOM series directory or an image file readable by TorchIO/SimpleITK. From Python, `dvf_qa.drr.project_drr` also accepts NumPy arrays and PyTorch tensors with CT shape `(z, y, x)` when `spacing_xyz` is provided.

If generated CT values are normalized instead of HU, set:

```json
{
  "ct_value_mode": "normalized_minus1_1_to_hu"
}
```

This clips values to `[-1, 1]` and maps them linearly to `[-1024, 1024] HU`.

## Dynamic Frontal/Lateral X-ray Lung Masks and Respiratory Phase

```bash
dvf-qa analyze-dynamic-xray \
  --ap /path/to/frontal_frames.npy \
  --lateral /path/to/lateral_frames.npy \
  --mask-method torchxrayvision \
  --signal-method intensity-roi \
  --out dynamic_xray_out
```

Inputs may be a directory of sorted image frames, a `.npy`/`.npz` stack with shape
`(frames, height, width)`, or a SimpleITK-readable 3D image. Outputs include:

- `ap_lung_masks.npy` / `lateral_lung_masks.npy`: binary lung masks per frame
- `ap_roi_mask.npy` / `lateral_roi_mask.npy`: fixed central lung ROI used for intensity-based phase analysis
- `ap_phase.csv` / `lateral_phase.csv`: lung area, inferior boundary, respiratory signal, phase, and inspiration/expiration state
- `combined_phase.csv`: fused respiratory phase when both AP and lateral sequences are provided
- quicklook PNGs for the mean frame, mask overlays, and respiratory phase plot

Use `--signal-method intensity-roi` when the diaphragm is not visible. This builds
a fixed central ROI from the segmented lung mask, tracks the ROI mean pixel value
over time with sequence-level intensity normalization, and treats high-density ROI
frames as expiration candidates. `--signal-method motion` keeps the older
area/inferior-boundary signal, while `--signal-method combined` fuses both.

For lateral images, `auto` mode uses one central ROI inside the lateral lung
candidate rather than separate anterior/posterior ROIs. The summary JSON reports
ROI QC fields:

- `qc_status`: `PASS`, `WARN`, or `FAIL`
- `qc_roi_lung_overlap_fraction`: ROI pixels that remain inside the lung mask
- `qc_roi_area_fraction_of_lung`: ROI size relative to the lung mask
- `qc_roi_centroid_x_fraction` / `qc_roi_centroid_y_fraction`: ROI centroid location

Recommended lateral workflow:

1. Generate masks and ROI quicklooks with `--signal-method intensity-roi`.
2. Accept automatic phase selection only when lateral `qc_status` is `PASS` and the quicklook ROI is visibly inside the central aerated lung region.
3. If QC warns or fails, reduce `--roi-fraction`, inspect the lateral mask, and fall back to AP-only phase if the lateral mask includes mediastinum, marker, or body edge artifacts.
4. For deployment, collect failed lateral cases and tune or replace the lateral mask step separately from the ROI intensity phase estimator.

Set `--mask-method torchxrayvision` to use the public TorchXRayVision ChestX-Det
PSPNet anatomical segmentation model. The pipeline combines the model's
`Left Lung` and `Right Lung` channels, resizes the mask back to the original frame
size, and then runs the same respiratory phase analysis. TorchXRayVision weights
are downloaded to the local model cache on first use. The model was trained for
chest radiographs, so lateral or fluoroscopic dynamic sequences still need
case-level overlay QC and, ideally, local validation.

For a quick qualitative check on public CC0 frontal/lateral CXR images:

```bash
dvf-qa demo-public-cxr \
  --mask-method torchxrayvision \
  --device cpu \
  --out public_cxr_demo_out
```

This downloads a Wikimedia Commons frontal/lateral pair, writes
`public_cxr_mask_demo.png`, per-view mask quicklooks, `.npy` masks, and JSON
summaries. In `auto` mode, lateral images use the unsupervised fallback because
the TorchXRayVision ChestX-Det segmentation model is frontal-CXR oriented rather
than validated for lateral CXR.

## 2D-4D Input Pipeline: 4DCT -> Paired AP/Lateral Cycles

End-to-end pipeline for preparing time-aligned paired AP+lateral dynamic
projection clips from a 4DCT, intended as input for a downstream 2D-4D
reconstruction model. The respiratory-cycle detection path is fully classical
(no learned segmentation), so it is usable in regulated contexts.

Pipeline stages:

1. `dvf_qa.synthetic_dynamic.simulate_from_4dct` — render per-phase AP and
   lateral DRRs via DiffDRR, then assemble a dynamic sequence by cycling the
   phase index over `n_cycles` breaths.
2. `dvf_qa.amsterdam_shroud.respiratory_signal` — Amsterdam Shroud diaphragm
   tracking with sub-pixel parabolic fit and zero-phase band-pass; no lung
   mask or learned model required.
3. `dvf_qa.cycle_pairing.pair_best_correlated_cycles` — pick the
   end-expiration-aligned AP/lateral cycle pair with the highest length-
   normalised NCC. Returns frame ranges in the original sequences.
4. `dvf_qa.pipeline_report.write_html_report` — self-contained HTML with
   embedded base64 figures covering 4DCT input, per-phase DRRs, dynamic
   samples, shroud + signal per view, NCC matrix, and selected pair overlay.

Run on a downloaded TCIA 4D-Lung case:

```bash
python scripts/tcia_4d_lung_demo.py \
  --phases-dir /path/to/4dct_phases \
  --out-dir outputs/tcia_demo \
  --fps 15 --n-cycles 3 --device cpu
```

`--phases-dir` is a directory with one subdirectory or file per respiratory
phase; subdirectory names sort alphabetically and that order becomes the
phase index. Outputs: `report.html`, `metrics.json`,
`ap_cycle_frames.npy`, `lateral_cycle_frames.npy`. See the script docstring
for TCIA download steps (NBIA Data Retriever or NBIA REST API).

## Sharing reports: GitHub Pages + Notion + Slack

`scripts/build_pages.py` stages every run directory that ships a top-level
`summary.json` (e.g. `outputs/dirlab_demo/`) under a `_site/` tree ready for
any static host. `.github/workflows/deploy-reports.yml` ties this to a full
CD loop: build the site, deploy to GitHub Pages, append per-case rows to a
Notion database, and post a Slack message.

### One-time setup

1. **Repository**: push this repo to your GitHub org. Keep it **private** —
   DIR-Lab derived images should not be on public Pages without explicit
   permission from the DIR-Lab authors (see [DIR-Lab reference data](https://www.dir-lab.com/ReferenceData.html)).
2. **GitHub Pages**: Settings → Pages → Source = "GitHub Actions". The
   workflow will configure and deploy on every push that changes a report
   file.
3. **Notion**:
   - Create an internal integration at <https://www.notion.so/profile/integrations>.
   - Create a database with properties listed in `scripts/notion_publish.py`'s
     docstring (Title, Run ID, Source, Best NCC, AP/Lat freq, Pages URL, Date).
   - Share the database with the integration.
   - Add GitHub secrets `NOTION_API_KEY` and `NOTION_DATABASE_ID`.
4. **Slack**: Create an incoming webhook for the channel you want notified
   and add the secret `SLACK_WEBHOOK_URL`.

Secrets are optional. The workflow skips Notion / Slack steps when their
secrets are absent.

### Day-to-day flow

```bash
# Run a pipeline (locally or on a runner)
python scripts/dirlab_demo.py --root outputs/dirlab_data --out-dir outputs/dirlab_demo ...

# Commit the produced reports (raw data stays gitignored)
git add outputs/dirlab_demo
git commit -m "DIR-Lab run YYYY-MM-DD"
git push

# Actions takes it from here:
#   build static site -> deploy to Pages -> notify Notion + Slack
```

Manual deploy with knobs:

```bash
gh workflow run deploy-reports.yml \
   -f notify_notion=true \
   -f notify_slack=false
```

### Visibility notes

- DIR-Lab derived images: keep behind authentication (private repo Pages on
  GitHub Team/Enterprise, Cloudflare Pages with Zero Trust, or an internal
  Emory / Yamaguchi U intranet host).
- TCIA 4D-Lung is CC BY 3.0 and may be shared with attribution.
- Patient or non-public data: do not deploy to GitHub Pages.
