# GPU Deployment and Productization Notes

This project is currently a research QA implementation for CT/DVF model outputs. If it is used to support clinical decisions, treatment planning, patient positioning, or release of generated CT/DVF data into a clinical workflow, treat it as potentially regulated medical device software / SaMD and establish the corresponding quality system before product claims are made.

This document is engineering guidance, not legal or regulatory advice.

## GPU Runtime

The provided Docker image uses a CUDA-enabled PyTorch base image and installs `diffdrr`, `torchio`, `SimpleITK`, and the local `dvf-qa` package.

Build:

```bash
docker build --platform linux/amd64 -t dvf-qa:gpu .
```

The CUDA image targets `linux/amd64`, which is the normal target for NVIDIA GPU servers. If you build it on Apple Silicon, Docker may warn about the host architecture; that is expected for a GPU-server image.

Check GPU visibility:

```bash
docker run --rm --gpus all dvf-qa:gpu python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("device_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Run QA:

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

Use `--device cpu` for deterministic debugging and `--device cuda` for GPU production runs. On Apple Silicon development machines, `--device mps` may work for parts of PyTorch, but CUDA should be the target for validated GPU deployment.

## Input Contracts

Supported CT inputs for DRR:

- DICOM series directory or image path readable by DiffDRR/TorchIO
- NumPy array via Python API, shape `(z, y, x)` plus `spacing_xyz=(dx, dy, dz)`
- PyTorch tensor via Python API, shape `(z, y, x)` or `(1, z, y, x)` plus spacing

CT value handling:

- `ct_value_mode="hu"` leaves values unchanged
- `ct_value_mode="normalized_minus1_1_to_hu"` clips to `[-1, 1]` and maps linearly to `[-1024, 1024] HU`
- `ct_value_mode="normalized_0_1_to_hu"` clips to `[0, 1]` and maps linearly to `[-1024, 1024] HU`

Supported DVF inputs:

- NIfTI/MHA/MHD/NRRD via SimpleITK
- `.npz` with keys `data`, optional `spacing_xyz`, `origin_xyz`, `direction`
- DVF array shape `(z, y, x, 3)` with components `(ux, uy, uz)` in millimeters

For product use, lock these contracts in an interface control document. Do not silently infer unknown coordinate systems.

## QA Outputs

Each case should preserve:

- `metrics.json`
- `jacobian.nii.gz`
- `folding_mask.nii.gz`
- `warped_moving.nii.gz` when fixed/moving CT are available
- `generated_ct_drr.png`
- `qa_report.png`
- software version, Docker image digest, model checkpoint hash, input file hashes, geometry JSON, and runtime device

The current code produces the image and metric artifacts. The audit metadata should be added before any regulated or customer-facing deployment.

## Acceptance Criteria

Recommended initial gates:

- Fail if any clinically relevant lung-region folding exists: `J <= 0`
- Fail if largest folding component exceeds the validated component-size threshold
- Warn/fail on extreme local compression: fraction of `J < 0.2`
- Warn/fail on extreme expansion: fraction of `J > 5`
- Track Jacobian percentiles: p01, p05, median, p95, p99
- Track DVF magnitude p95/max and bending energy mean
- Track warped CT similarity if fixed/moving CT are available
- Track generated CT DRR similarity against input fluoroscopy

The numeric thresholds must be validated on your own data distribution, stratified by phase pair, projection geometry, breathing amplitude, tumor location, and scanner/protocol.

## Productization Workstreams

1. Quality management

   Establish design controls, requirements traceability, verification/validation plans, change control, release records, supplier controls, and complaint/post-market processes. For a medical product, align early with ISO 13485 and IEC 62304 expectations.

2. Risk management

   Maintain an ISO 14971-style risk file. Hazards should include incorrect DVF, folding not detected, incorrect DRR geometry, DICOM orientation error, wrong patient/study, corrupted mask, unsupported scanner protocol, and model drift.

3. Clinical validation

   Separate algorithm development, locked validation, and external validation cohorts. Report performance by clinically meaningful strata, not only aggregate metrics.

4. Cybersecurity and privacy

   Treat DICOM and fluoroscopy as PHI. Minimize stored identifiers, use encrypted storage/transport, define retention, log access, and prepare cybersecurity documentation for networked deployments.

5. Reproducibility

   Pin Docker image digests, package versions, model checkpoint hashes, and test data hashes. Every QA result should be reproducible from immutable inputs.

6. Human factors

   Define whether the tool is advisory, blocking, or automated. For clinical use, the report must make failure modes visible and should avoid ambiguous pass/fail claims that users can over-trust.

7. Monitoring

   Add post-deployment monitoring for input distribution shift, Jacobian distribution drift, DRR similarity drift, runtime failures, and operator overrides.

## Regulatory Pointers

United States:

- FDA lists digital health guidance documents, including final guidance for predetermined change control plans for AI-enabled device software functions and draft guidance on AI-enabled device software lifecycle and marketing submissions.
- If intended use affects diagnosis, treatment planning, patient positioning, or clinical decision-making, assume FDA review may be relevant until regulatory counsel says otherwise.

European Union:

- The EU AI Act is risk-based. High-risk AI obligations include risk management, data governance, technical documentation, logging, human oversight, robustness, cybersecurity, and accuracy.
- The European Commission states high-risk rules apply in August 2026 and August 2027, with an extended transition for high-risk systems embedded into regulated products.
- AI-based medical software may also need MDR/IVDR conformity assessment.

Japan:

- PMDA provides SaMD information and consultation routes, including qualification consultation and clinical trial / clinical evaluation advice.
- Use PMDA/MHLW consultation early if the intended use could qualify as medical device software.

## Open Engineering Gaps Before Product Use

- Add audit metadata into `metrics.json`
- Add deterministic integration tests with fixed DICOM fixtures
- Add coordinate-system tests for DICOM orientation, spacing, gantry geometry, and projection pose
- Add model-output schema validation
- Add threshold configuration with versioned validation evidence
- Add report templates suitable for clinical review
- Add structured error codes instead of generic exceptions
- Add SBOM generation and vulnerability scanning in CI
- Add performance benchmarks on target GPU hardware
