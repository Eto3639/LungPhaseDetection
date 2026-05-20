"""DIR-Lab 4DCT loader.

DIR-Lab distributes thoracic 4DCT volumes as headerless int16 little-endian
raw files (``*.img``). The volume shape and voxel spacing are documented per
case on the project page (https://med.emory.edu/.../downloads-and-reference-data/4dct.html)
and must be supplied externally — this module bundles that metadata for the
10 standard cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DIRLabCaseSpec:
    case_id: str
    shape_zyx: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    image_pattern: str
    phase_tags: tuple[str, ...] = (
        "T00", "T10", "T20", "T30", "T40", "T50", "T60", "T70", "T80", "T90",
    )


# Reference: https://med.emory.edu/departments/radiation-oncology/research-laboratories/
#            deformable-image-registration/downloads-and-reference-data/4dct.html
DIRLAB_CASES: dict[str, DIRLabCaseSpec] = {
    "case1": DIRLabCaseSpec(
        case_id="case1",
        shape_zyx=(94, 256, 256),
        spacing_xyz_mm=(0.97, 0.97, 2.5),
        image_pattern="case1_{tag}_s.img",
    ),
    "case2": DIRLabCaseSpec(
        case_id="case2",
        shape_zyx=(112, 256, 256),
        spacing_xyz_mm=(1.16, 1.16, 2.5),
        image_pattern="case2_{tag}-ssm.img",
    ),
    "case3": DIRLabCaseSpec(
        case_id="case3",
        shape_zyx=(104, 256, 256),
        spacing_xyz_mm=(1.15, 1.15, 2.5),
        image_pattern="case3_{tag}-ssm.img",
    ),
    "case4": DIRLabCaseSpec(
        case_id="case4",
        shape_zyx=(99, 256, 256),
        spacing_xyz_mm=(1.13, 1.13, 2.5),
        image_pattern="case4_{tag}-ssm.img",
    ),
    "case5": DIRLabCaseSpec(
        case_id="case5",
        shape_zyx=(106, 256, 256),
        spacing_xyz_mm=(1.10, 1.10, 2.5),
        image_pattern="case5_{tag}-ssm.img",
    ),
    "case6": DIRLabCaseSpec(
        case_id="case6",
        shape_zyx=(128, 512, 512),
        spacing_xyz_mm=(0.97, 0.97, 2.5),
        image_pattern="case6_{tag}.img",
    ),
    "case7": DIRLabCaseSpec(
        case_id="case7",
        shape_zyx=(136, 512, 512),
        spacing_xyz_mm=(0.97, 0.97, 2.5),
        image_pattern="case7_{tag}.img",
    ),
    "case8": DIRLabCaseSpec(
        case_id="case8",
        shape_zyx=(128, 512, 512),
        spacing_xyz_mm=(0.97, 0.97, 2.5),
        image_pattern="case8_{tag}.img",
    ),
    "case9": DIRLabCaseSpec(
        case_id="case9",
        shape_zyx=(128, 512, 512),
        spacing_xyz_mm=(0.97, 0.97, 2.5),
        image_pattern="case9_{tag}.img",
    ),
    "case10": DIRLabCaseSpec(
        case_id="case10",
        shape_zyx=(120, 512, 512),
        spacing_xyz_mm=(0.97, 0.97, 2.5),
        image_pattern="case10_{tag}.img",
    ),
}


def detect_case_id(case_pack_dir: Path) -> str:
    """Infer the canonical case id (case1..case10) from a CaseN{Pack,Deploy}/Images directory."""
    case_pack_dir = Path(case_pack_dir)
    images_dir = case_pack_dir / "Images" if (case_pack_dir / "Images").is_dir() else case_pack_dir
    files = sorted(p.name for p in images_dir.iterdir() if p.suffix == ".img")
    if not files:
        raise ValueError(f"No .img files under {images_dir}")
    name = files[0].lower()
    # Longer keys first so e.g. "case10" wins over "case1" prefix match.
    for key in sorted(DIRLAB_CASES, key=len, reverse=True):
        if name.startswith(key + "_"):
            return key
    raise ValueError(f"Could not infer DIR-Lab case id from filename {files[0]}")


def load_case_phases(case_pack_dir: str | Path, case_id: str | None = None) -> tuple[list[np.ndarray], tuple[float, float, float], DIRLabCaseSpec]:
    """Load the 10-phase 4DCT for one DIR-Lab case.

    Returns
    -------
    volumes:
        List of 10 numpy arrays, each shape ``(z, y, x)`` int16->float32.
    spacing_xyz:
        Voxel spacing in mm as ``(dx, dy, dz)``.
    spec:
        The :class:`DIRLabCaseSpec` used for parsing.
    """
    case_pack_dir = Path(case_pack_dir)
    images_dir = case_pack_dir / "Images" if (case_pack_dir / "Images").is_dir() else case_pack_dir
    case_key = (case_id or detect_case_id(case_pack_dir)).lower()
    if case_key not in DIRLAB_CASES:
        raise ValueError(f"Unsupported DIR-Lab case_id {case_key!r}")
    spec = DIRLAB_CASES[case_key]
    n_voxels = int(np.prod(spec.shape_zyx))
    volumes: list[np.ndarray] = []
    for tag in spec.phase_tags:
        path = images_dir / spec.image_pattern.format(tag=tag)
        if not path.is_file():
            raise FileNotFoundError(f"Missing phase {tag} for {case_key}: {path}")
        raw = np.fromfile(path, dtype="<i2", count=n_voxels)
        if raw.size != n_voxels:
            raise ValueError(
                f"{path} size mismatch: expected {n_voxels} int16 voxels, got {raw.size}"
            )
        volumes.append(raw.reshape(spec.shape_zyx).astype(np.float32))
    return volumes, spec.spacing_xyz_mm, spec
