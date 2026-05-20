import numpy as np
import torch

from dvf_qa.drr import DrrGeometry, project_drr


def test_diffdrr_projects_numpy_volume():
    ct = np.full((16, 16, 16), -500.0, dtype=np.float32)
    ct[4:12, 4:12, 4:12] = 100.0
    geom = DrrGeometry(
        sdd=1020.0,
        detector_shape=(16, 16),
        pixel_spacing_mm=(1.0, 1.0),
        rotation=(0.0, 0.0, 0.0),
        translation=(0.0, 850.0, 0.0),
        orientation=None,
        ct_value_mode="hu",
    )
    drr = project_drr(ct, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), geom, device="cpu")
    assert drr.shape == (16, 16)
    assert np.isfinite(drr).all()


def test_diffdrr_projects_torch_tensor():
    ct = torch.full((8, 8, 8), -500.0)
    ct[2:6, 2:6, 2:6] = 100.0
    geom = DrrGeometry(
        sdd=1020.0,
        detector_shape=(8, 8),
        pixel_spacing_mm=(1.0, 1.0),
        orientation=None,
    )
    drr = project_drr(ct, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), geom, device="cpu")
    assert drr.shape == (8, 8)
    assert np.isfinite(drr).all()


def test_diffdrr_projects_normalized_numpy_volume_as_hu():
    ct = np.full((8, 8, 8), -1.0, dtype=np.float32)
    ct[1:7, 1:7, 1:7] = 0.0
    ct[2:6, 2:6, 2:6] = 0.5
    geom = DrrGeometry(
        sdd=1020.0,
        detector_shape=(8, 8),
        pixel_spacing_mm=(1.0, 1.0),
        orientation=None,
        ct_value_mode="normalized_minus1_1_to_hu",
    )
    drr = project_drr(ct, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), geom, device="cpu")
    assert drr.shape == (8, 8)
    assert np.isfinite(drr).all()
