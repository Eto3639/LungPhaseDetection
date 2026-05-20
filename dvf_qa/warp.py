from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def warp_volume_moving_to_fixed(moving: np.ndarray, dvf: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Sample moving CT at fixed-grid coordinates plus displacement.

    The implementation assumes moving and fixed arrays share the same grid. For
    different grids, resample to a common physical reference before calling.
    """
    if moving.shape != dvf.shape[:3]:
        raise ValueError(f"moving shape {moving.shape} does not match DVF grid {dvf.shape[:3]}")

    dx, dy, dz = spacing_xyz
    z, y, x = np.meshgrid(
        np.arange(moving.shape[0]),
        np.arange(moving.shape[1]),
        np.arange(moving.shape[2]),
        indexing="ij",
    )
    coords = np.array(
        [
            z + dvf[..., 2] / dz,
            y + dvf[..., 1] / dy,
            x + dvf[..., 0] / dx,
        ]
    )
    return map_coordinates(moving, coords, order=1, mode="nearest")

