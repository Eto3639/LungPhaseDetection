import pytest
import numpy as np
import torch
from core.model import load_xray_segmentation_model, segment_lung_xray

@pytest.fixture
def model():
    return load_xray_segmentation_model()

def test_model_loading(model):
    assert model is not None
    assert hasattr(model, 'forward')

def test_segment_lung_xray(model):
    # Create a dummy image
    dummy_img = np.random.rand(256, 256).astype(np.float32)
    # The segment function handles scaling, but let's pass [0, 255] roughly
    dummy_img = dummy_img * 255

    mask = segment_lung_xray(model, dummy_img)

    assert mask.shape == (256, 256)
    # mask should be binary (0 or 1)
    assert set(np.unique(mask)).issubset({0, 1})
