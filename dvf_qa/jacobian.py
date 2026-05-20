from __future__ import annotations

import numpy as np


def jacobian_determinant(dvf: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Return det(I + grad(u)) for a displacement vector field.

    Parameters
    ----------
    dvf:
        Array with shape ``(z, y, x, 3)``. Components are expected to be
        displacement in physical x, y, z directions, usually millimeters.
    spacing_xyz:
        Voxel spacing as ``(dx, dy, dz)`` in the same unit as ``dvf``.
    """
    if dvf.ndim != 4 or dvf.shape[-1] != 3:
        raise ValueError(f"Expected DVF shape (z, y, x, 3), got {dvf.shape}")

    dx, dy, dz = spacing_xyz
    ux = dvf[..., 0]
    uy = dvf[..., 1]
    uz = dvf[..., 2]

    dux_dz, dux_dy, dux_dx = np.gradient(ux, dz, dy, dx, edge_order=2)
    duy_dz, duy_dy, duy_dx = np.gradient(uy, dz, dy, dx, edge_order=2)
    duz_dz, duz_dy, duz_dx = np.gradient(uz, dz, dy, dx, edge_order=2)

    j11 = 1.0 + dux_dx
    j12 = dux_dy
    j13 = dux_dz
    j21 = duy_dx
    j22 = 1.0 + duy_dy
    j23 = duy_dz
    j31 = duz_dx
    j32 = duz_dy
    j33 = 1.0 + duz_dz

    return (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )


def gradient_norm(dvf: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Frobenius norm of the displacement gradient at each voxel."""
    dx, dy, dz = spacing_xyz
    parts = []
    for component in range(3):
        grads = np.gradient(dvf[..., component], dz, dy, dx, edge_order=2)
        parts.extend(grads)
    return np.sqrt(np.sum([g * g for g in parts], axis=0))


def bending_energy(dvf: np.ndarray, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    """Approximate voxelwise bending energy from second derivatives."""
    dx, dy, dz = spacing_xyz
    energy = np.zeros(dvf.shape[:3], dtype=np.float64)
    for component in range(3):
        first = np.gradient(dvf[..., component], dz, dy, dx, edge_order=2)
        for axis, (axis_grad, axis_spacing) in enumerate(zip(first, (dz, dy, dx), strict=True)):
            second = np.gradient(axis_grad, axis_spacing, axis=axis, edge_order=2)
            energy += second * second
    return energy
