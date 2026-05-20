from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchio as tio
from diffdrr.data import read as diffdrr_read
from diffdrr.drr import DRR
from torchio import Subject


ArrayLikeVolume = str | Path | np.ndarray | torch.Tensor | Subject | tio.ScalarImage


@dataclass(frozen=True)
class DiffDRRGeometry:
    """DiffDRR projection parameters.

    The pose follows DiffDRR's convention: rotation and translation define the
    C-arm pose. For Euler angles, set ``parameterization="euler_angles"`` and
    provide a convention such as ``"ZXY"``.
    """

    sdd: float
    detector_shape: tuple[int, int]
    pixel_spacing_mm: tuple[float, float]
    rotation: tuple[float, ...] = (0.0, 0.0, 0.0)
    translation: tuple[float, float, float] = (0.0, 850.0, 0.0)
    parameterization: str = "euler_angles"
    convention: str | None = "ZXY"
    degrees: bool = False
    orientation: str | None = "AP"
    center_volume: bool = True
    renderer: str = "siddon"
    reverse_x_axis: bool = True
    x0: float = 0.0
    y0: float = 0.0
    patch_size: int | None = None
    bone_attenuation_multiplier: float = 1.0
    resample_target: Any = None
    ct_value_mode: str = "hu"
    renderer_kwargs: dict[str, Any] | None = None

    @property
    def height(self) -> int:
        return int(self.detector_shape[0])

    @property
    def width(self) -> int:
        return int(self.detector_shape[1])

    @property
    def delx(self) -> float:
        return float(self.pixel_spacing_mm[0])

    @property
    def dely(self) -> float:
        return float(self.pixel_spacing_mm[1])

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DiffDRRGeometry":
        renderer_kwargs = data.get("renderer_kwargs")
        return DiffDRRGeometry(
            sdd=float(data["sdd"]),
            detector_shape=tuple(data["detector_shape"]),
            pixel_spacing_mm=tuple(data["pixel_spacing_mm"]),
            rotation=tuple(data.get("rotation", (0.0, 0.0, 0.0))),
            translation=tuple(data.get("translation", (0.0, 850.0, 0.0))),
            parameterization=data.get("parameterization", "euler_angles"),
            convention=data.get("convention", "ZXY"),
            degrees=bool(data.get("degrees", False)),
            orientation=data.get("orientation", "AP"),
            center_volume=bool(data.get("center_volume", True)),
            renderer=data.get("renderer", "siddon"),
            reverse_x_axis=bool(data.get("reverse_x_axis", True)),
            x0=float(data.get("x0", 0.0)),
            y0=float(data.get("y0", 0.0)),
            patch_size=data.get("patch_size"),
            bone_attenuation_multiplier=float(data.get("bone_attenuation_multiplier", 1.0)),
            resample_target=data.get("resample_target"),
            ct_value_mode=data.get("ct_value_mode", "hu"),
            renderer_kwargs=renderer_kwargs if renderer_kwargs is not None else None,
        )


# Backward-compatible public name used by the first CLI implementation.
DrrGeometry = DiffDRRGeometry


def project_drr(
    ct_hu: ArrayLikeVolume,
    spacing_xyz: tuple[float, float, float] | None,
    origin_xyz: tuple[float, float, float] | None,
    geometry: DiffDRRGeometry,
    *,
    device: str | torch.device | None = None,
    return_tensor: bool = False,
) -> np.ndarray | torch.Tensor:
    """Render a DRR using DiffDRR from DICOM/path, NumPy array, or torch tensor.

    Parameters
    ----------
    ct_hu:
        A DICOM directory / image path, a ``torchio.Subject``, a
        ``torchio.ScalarImage``, a NumPy array with shape ``(z, y, x)``, or a
        torch tensor with shape ``(z, y, x)``, ``(1, z, y, x)``, or TorchIO style
        ``(1, x, y, z)`` when ``tensor_layout="torchio"`` is used via
        :func:`make_diffdrr_subject`.
    spacing_xyz:
        Required for NumPy/tensor inputs. Physical spacing as ``(dx, dy, dz)``.
    origin_xyz:
        Optional for NumPy/tensor inputs. Physical origin as ``(ox, oy, oz)``.
    geometry:
        DiffDRR geometry and pose parameters.
    """
    subject = make_diffdrr_subject(
        ct_hu,
        spacing_xyz=spacing_xyz,
        origin_xyz=origin_xyz,
        orientation=geometry.orientation,
        center_volume=geometry.center_volume,
        bone_attenuation_multiplier=geometry.bone_attenuation_multiplier,
        resample_target=geometry.resample_target,
        ct_value_mode=geometry.ct_value_mode,
    )
    image = render_diffdrr_subject(subject, geometry, device=device)
    return image if return_tensor else image.detach().cpu().squeeze().numpy().astype(np.float32)


def project_drr_from_path(
    path: str | Path,
    geometry: DiffDRRGeometry,
    *,
    device: str | torch.device | None = None,
    return_tensor: bool = False,
) -> np.ndarray | torch.Tensor:
    """Render a DRR from a DICOM directory or image file using DiffDRR's reader."""
    return project_drr(path, None, None, geometry, device=device, return_tensor=return_tensor)


def make_diffdrr_subject(
    volume: ArrayLikeVolume,
    *,
    spacing_xyz: tuple[float, float, float] | None = None,
    origin_xyz: tuple[float, float, float] | None = None,
    orientation: str | None = "AP",
    center_volume: bool = True,
    bone_attenuation_multiplier: float = 1.0,
    resample_target: Any = None,
    ct_value_mode: str = "hu",
    tensor_layout: str = "zyx",
) -> Subject:
    """Create a DiffDRR ``Subject`` from DICOM/path, NumPy, torch, or TorchIO."""
    if isinstance(volume, Subject):
        return volume

    if isinstance(volume, (str, Path, tio.ScalarImage)):
        scalar = volume if ct_value_mode == "hu" else _scalar_image_from_path_with_value_mode(volume, ct_value_mode)
    else:
        if spacing_xyz is None:
            raise ValueError("spacing_xyz is required for NumPy array and torch tensor CT inputs")
        scalar = _scalar_image_from_arraylike(
            _apply_ct_value_mode(volume, ct_value_mode),
            spacing_xyz,
            origin_xyz,
            tensor_layout=tensor_layout,
        )

    return diffdrr_read(
        scalar,
        orientation=orientation,
        center_volume=center_volume,
        bone_attenuation_multiplier=bone_attenuation_multiplier,
        resample_target=resample_target,
    )


def render_diffdrr_subject(
    subject: Subject,
    geometry: DiffDRRGeometry,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Render a DiffDRR ``Subject`` and return a tensor with shape ``(1, 1, H, W)``."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    kwargs = geometry.renderer_kwargs or {}
    drr = DRR(
        subject,
        sdd=geometry.sdd,
        height=geometry.height,
        width=geometry.width,
        delx=geometry.delx,
        dely=geometry.dely,
        x0=geometry.x0,
        y0=geometry.y0,
        renderer=geometry.renderer,
        reverse_x_axis=geometry.reverse_x_axis,
        patch_size=geometry.patch_size,
        **kwargs,
    ).to(device)

    rotation = torch.as_tensor(geometry.rotation, dtype=torch.float32, device=device).unsqueeze(0)
    translation = torch.as_tensor(geometry.translation, dtype=torch.float32, device=device).unsqueeze(0)
    image = drr(
        rotation,
        translation,
        parameterization=geometry.parameterization,
        convention=geometry.convention,
        degrees=geometry.degrees,
    )
    return torch.nan_to_num(image)


def subject_volume_numpy(subject: Subject) -> np.ndarray:
    """Return subject CT volume as ``(z, y, x)`` NumPy for reporting."""
    data = subject.volume.data.detach().cpu()
    if data.ndim != 4 or data.shape[0] != 1:
        raise ValueError(f"Expected TorchIO volume tensor shape (1,x,y,z), got {tuple(data.shape)}")
    return data.squeeze(0).permute(2, 1, 0).numpy()


def _scalar_image_from_arraylike(
    volume: np.ndarray | torch.Tensor,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float] | None,
    *,
    tensor_layout: str,
) -> tio.ScalarImage:
    tensor = torch.as_tensor(volume)
    if tensor.ndim == 3:
        if tensor_layout != "zyx":
            raise ValueError("3D tensor_layout must be 'zyx'")
        tensor = tensor.permute(2, 1, 0).unsqueeze(0)
    elif tensor.ndim == 4 and tensor.shape[0] == 1:
        if tensor_layout == "zyx":
            tensor = tensor.squeeze(0).permute(2, 1, 0).unsqueeze(0)
        elif tensor_layout != "torchio":
            raise ValueError("4D tensor_layout must be 'zyx' or 'torchio'")
    else:
        raise ValueError(f"Expected CT shape (z,y,x), (1,z,y,x), or (1,x,y,z); got {tuple(tensor.shape)}")

    affine = _affine_from_spacing_origin(spacing_xyz, origin_xyz)
    return tio.ScalarImage(tensor=tensor.to(torch.float32), affine=affine)


def _scalar_image_from_path_with_value_mode(path_or_image: str | Path | tio.ScalarImage, ct_value_mode: str) -> tio.ScalarImage:
    image = path_or_image if isinstance(path_or_image, tio.ScalarImage) else tio.ScalarImage(path_or_image)
    tensor = _apply_ct_value_mode(image.data, ct_value_mode)
    return tio.ScalarImage(tensor=tensor.to(torch.float32), affine=image.affine)


def _apply_ct_value_mode(volume: np.ndarray | torch.Tensor, ct_value_mode: str) -> torch.Tensor:
    tensor = torch.as_tensor(volume, dtype=torch.float32)
    if ct_value_mode == "hu":
        return tensor
    if ct_value_mode == "normalized_minus1_1_to_hu":
        return torch.clamp(tensor, -1.0, 1.0) * 1024.0
    if ct_value_mode == "normalized_0_1_to_hu":
        return torch.clamp(tensor, 0.0, 1.0) * 2048.0 - 1024.0
    raise ValueError(
        "Unsupported ct_value_mode. Use 'hu', 'normalized_minus1_1_to_hu', "
        "or 'normalized_0_1_to_hu'."
    )


def _affine_from_spacing_origin(
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float] | None,
) -> np.ndarray:
    dx, dy, dz = spacing_xyz
    ox, oy, oz = origin_xyz or (0.0, 0.0, 0.0)
    return np.array(
        [
            [dx, 0.0, 0.0, ox],
            [0.0, dy, 0.0, oy],
            [0.0, 0.0, dz, oz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
