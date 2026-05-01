import torch
import torch.nn as nn
import numpy as np
import logging
import cv2


try:
    import torchxrayvision as xrv
except ImportError:
    pass


def load_xray_segmentation_model() -> nn.Module:
    """
    Loads the pre-trained PSPNet model for X-ray segmentation using TorchXRayVision.
    This model can segment various parts including left/right lung.

    Returns:
        nn.Module: Loaded PSPNet segmentation model.
    """
    logging.info("Loading TorchXRayVision PSPNet model for lung segmentation...")
    model = xrv.baseline_models.chestx_det.PSPNet()
    model.eval()
    return model


def segment_lung_xray(model: nn.Module, image_np: np.ndarray) -> np.ndarray:
    """
    Performs lung segmentation on a single X-ray image using TorchXRayVision.
    Extracts the 'Left Lung' and 'Right Lung' masks and combines them.

    Args:
        model (nn.Module): The TorchXRayVision segmentation model.
        image_np (np.ndarray): The input X-ray image as a numpy array.

    Returns:
        np.ndarray: A binary mask (0 or 1) representing the combined lung region.
    """
    original_size = image_np.shape

    # Preprocess the image for torchxrayvision:
    # Scale to [-1024, 1024]
    img = image_np.astype(np.float32)
    # Basic normalization to [-1024, 1024] if originally e.g., [0, 4095]
    # For now, let's normalize to [0, 255] then to [-1024, 1024]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 2048) - 1024

    # Add color channel and batch dimension: (1, 1, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    # Output is a dict or tensor depending on the model, but for PSPNet it's usually a tensor or output obj
    # Let's inspect output shape later, assuming it returns segmentation masks
    # The targets are: 'Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
    # 'Left Lung' (idx 4), 'Right Lung' (idx 5), ...

    if isinstance(output, torch.Tensor):
        # shape is (batch, num_classes, H, W)
        masks = torch.sigmoid(output).squeeze(0).cpu().numpy()
        # Combine Left Lung (4) and Right Lung (5)
        left_lung = masks[4] > 0.5
        right_lung = masks[5] > 0.5
        mask = np.logical_or(left_lung, right_lung).astype(np.uint8)
    else:
        # Just in case
        mask = np.zeros(original_size, dtype=np.uint8)

    # Resize back if needed (PSPNet handles fixed or dynamic sizes, but usually output matches input)
    if mask.shape != original_size:
        mask = cv2.resize(
            mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST
        )

    return mask
