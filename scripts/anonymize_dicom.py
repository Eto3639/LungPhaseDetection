"""Anonymize Konica Minolta DDR multi-frame DICOMs in DynamicChestXray/.

Implements a subset of the DICOM PS3.15 Annex E "Basic Application Confidentiality
Profile" (cleaning rules) tuned for our use case:

- Patient identifiers (name, ID, birth date, sex, ...) replaced with a stable
  anonymous label derived from the source directory name (``case01_rec1`` etc.).
- Institution / physician / operator names cleared.
- Dates and times cleared.
- All three UIDs (Study / Series / SOP) re-generated, but the mapping is
  preserved in ``anon_map.json`` so the audit trail is locally inspectable.
- Pixel data is left untouched (Konica DDR has no burned-in identifiers in
  these acquisitions; verify visually if in doubt).

The anonymized files live alongside the originals under a sibling directory
``DynamicChestXray_anon/``. Both raw and anonymized trees are kept out of git
via ``.gitignore``.

Usage::

    python scripts/anonymize_dicom.py --in DynamicChestXray --out DynamicChestXray_anon
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pydicom
from pydicom.uid import generate_uid


# Tags to clear or replace. Each entry: (group, element) -> replacement (None = delete)
# Source: DICOM PS3.15 Annex E, plus extras commonly seen in JP/KMI exports.
ANONYMIZE_TAGS: dict[str, object] = {
    "PatientName": "ANONYMOUS",
    "PatientID": None,            # set per-file from case+rec
    "PatientBirthDate": "",
    "PatientBirthTime": "",
    "PatientSex": "",
    "PatientAge": "",
    "PatientSize": "",
    "PatientWeight": "",
    "EthnicGroup": "",
    "Occupation": "",
    "AdditionalPatientHistory": "",
    "PatientComments": "",
    "OtherPatientIDs": "",
    "OtherPatientNames": "",
    "OtherPatientIDsSequence": None,
    "InstitutionName": "",
    "InstitutionAddress": "",
    "InstitutionalDepartmentName": "",
    "InstitutionCodeSequence": None,
    "ReferringPhysicianName": "",
    "ReferringPhysicianAddress": "",
    "ReferringPhysicianTelephoneNumbers": "",
    "PhysiciansOfRecord": "",
    "PerformingPhysicianName": "",
    "PerformingPhysicianIdentificationSequence": None,
    "NameOfPhysiciansReadingStudy": "",
    "OperatorsName": "",
    "OperatorIdentificationSequence": None,
    "StudyDate": "",
    "StudyTime": "",
    "SeriesDate": "",
    "SeriesTime": "",
    "AcquisitionDate": "",
    "AcquisitionTime": "",
    "ContentDate": "",
    "ContentTime": "",
    "AccessionNumber": "",
    "StudyID": "",
    "FillerOrderNumberImagingServiceRequest": "",
    "PlacerOrderNumberImagingServiceRequest": "",
    "DeviceSerialNumber": "",
    "RequestingPhysician": "",
    "RequestingService": "",
    "RequestedProcedureID": "",
    "RequestedProcedureDescription": "",
    "ScheduledProcedureStepID": "",
    "ScheduledProcedureStepDescription": "",
    "PerformedProcedureStepID": "",
    "PerformedProcedureStepDescription": "",
    "StudyDescription": "",
    "RequestAttributesSequence": None,
}


def anonymize_one(src: Path, dst: Path, *, anon_patient_id: str,
                  uid_map: dict[str, str]) -> dict:
    ds = pydicom.dcmread(str(src))

    # Capture *non-PHI* metadata of interest for audit
    audit = {
        "source": src.name,
        "modality": str(ds.get("Modality", "")),
        "series_description": str(ds.get("SeriesDescription", "")),
        "number_of_frames": int(ds.get("NumberOfFrames", 0) or 0),
        "frame_time_ms": float(ds.get("FrameTime", 0.0) or 0.0),
        "rows": int(ds.get("Rows", 0) or 0),
        "columns": int(ds.get("Columns", 0) or 0),
        "photometric_interpretation": str(ds.get("PhotometricInterpretation", "")),
    }

    # Strip / replace PHI tags
    for keyword, replacement in ANONYMIZE_TAGS.items():
        if keyword not in ds:
            continue
        if replacement is None:
            delattr(ds, keyword)
        else:
            ds.data_element(keyword).value = replacement

    # Set the anonymous PatientID
    ds.PatientID = anon_patient_id
    ds.PatientName = anon_patient_id

    # Re-map UIDs (preserve mapping for traceability)
    def remap(field: str) -> None:
        if field in ds:
            old = str(getattr(ds, field))
            new = uid_map.setdefault(old, generate_uid())
            setattr(ds, field, new)

    remap("StudyInstanceUID")
    remap("SeriesInstanceUID")
    remap("SOPInstanceUID")
    # The file meta SOPInstanceUID must match
    if hasattr(ds, "file_meta") and "MediaStorageSOPInstanceUID" in ds.file_meta:
        old = str(ds.file_meta.MediaStorageSOPInstanceUID)
        new = uid_map.get(old) or generate_uid()
        uid_map[old] = new
        ds.file_meta.MediaStorageSOPInstanceUID = new

    # Final write
    dst.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(dst), write_like_original=False)
    audit["anon_patient_id"] = anon_patient_id
    audit["output"] = dst.name
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", type=Path, required=True)
    parser.add_argument("--out", dest="dst", type=Path, required=True)
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"Not a directory: {args.src}", file=sys.stderr); return 2
    args.dst.mkdir(parents=True, exist_ok=True)

    uid_map: dict[str, str] = {}
    audit: list[dict] = []
    for case_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        case = case_dir.name
        files = sorted(p for p in case_dir.iterdir() if p.is_file())
        for rec_idx, src_file in enumerate(files, start=1):
            anon_id = f"case{case}_rec{rec_idx}"
            dst_file = args.dst / case / f"{anon_id}.dcm"
            try:
                entry = anonymize_one(src_file, dst_file,
                                      anon_patient_id=anon_id, uid_map=uid_map)
                entry["case"] = case
                entry["recording"] = rec_idx
                audit.append(entry)
                print(f"  {case} / rec{rec_idx}: {entry['number_of_frames']} frames "
                      f"@ {entry['frame_time_ms']:.0f}ms  -> {dst_file.name}")
            except Exception as exc:
                print(f"FAILED {case_dir.name}/{src_file.name}: {exc}", file=sys.stderr)

    (args.dst / "anon_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.dst / "uid_map.json").write_text(
        json.dumps(uid_map, indent=2), encoding="utf-8"
    )
    print(f"\nAnonymized {len(audit)} files into {args.dst}")
    print(f"Audit: {args.dst / 'anon_audit.json'}")
    print(f"UID map: {args.dst / 'uid_map.json'}  (keep this OUT of git)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
