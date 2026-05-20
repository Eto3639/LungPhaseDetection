from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class Volume:
    data: np.ndarray
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]


def read_volume(path: str | Path) -> Volume:
    path = Path(path)
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise ValueError(f"No DICOM series found under directory: {path}")
        file_names = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(file_names)
        image = reader.Execute()
        data = sitk.GetArrayFromImage(image)
        return Volume(
            data=data,
            spacing_xyz=tuple(float(v) for v in image.GetSpacing()),
            origin_xyz=tuple(float(v) for v in image.GetOrigin()),
            direction=tuple(float(v) for v in image.GetDirection()),
        )
    if path.suffix == ".npy":
        return Volume(np.load(path), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), _identity_direction())
    if path.suffix == ".npz":
        npz = np.load(path)
        spacing = tuple(float(v) for v in npz.get("spacing_xyz", [1.0, 1.0, 1.0]))
        origin = tuple(float(v) for v in npz.get("origin_xyz", [0.0, 0.0, 0.0]))
        direction = tuple(float(v) for v in npz.get("direction", _identity_direction()))
        return Volume(npz["data"], spacing, origin, direction)

    image = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(image)
    return Volume(
        data=data,
        spacing_xyz=tuple(float(v) for v in image.GetSpacing()),
        origin_xyz=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def write_volume(path: str | Path, data: np.ndarray, reference: Volume | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, data)
        return
    image = sitk.GetImageFromArray(data)
    if reference is not None:
        image.SetSpacing(reference.spacing_xyz)
        image.SetOrigin(reference.origin_xyz)
        image.SetDirection(reference.direction)
    sitk.WriteImage(image, str(path))


def read_image(path: str | Path) -> np.ndarray:
    image = iio.imread(path).astype(np.float32)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    return image


def write_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finite = np.nan_to_num(image.astype(np.float32))
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((finite - lo) / (hi - lo), 0.0, 1.0)
    iio.imwrite(path, (scaled * 255).astype(np.uint8))


def _identity_direction() -> tuple[float, ...]:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

