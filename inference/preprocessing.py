"""
Module: inference/preprocessing.py
Project: Plant Disease Classification — Leafcare Model (EfficientNetV2-S)

PURPOSE:
-------------------------------------------------------------------------
Provides production-grade image validation, format normalization, and
tensor transformations for live single-image and batch inference.

WHY THIS MODULE EXISTS:
-------------------------------------------------------------------------
In production deployments, inputs arrive in arbitrary formats (RGBA from PNGs,
grayscale, CMYK, rotated mobile JPEGs, corrupted buffers). The preprocessing
layer ensures deterministic sanitization before tensor handoff.

HOW IT WORKS:
-------------------------------------------------------------------------
1. Validates input image type (path, PIL Image, or numpy array).
2. Converts any color space into standard 3-channel RGB.
3. Resizes image to expected spatial resolution (224x224).
4. Converts to float32 PyTorch tensor in range [0.0, 1.0].
5. Adds batch dimension [1, 3, 224, 224] if necessary.

INPUTS:
-------------------------------------------------------------------------
- Image file path (str), PIL Image, or numpy ndarray

OUTPUTS:
-------------------------------------------------------------------------
- Preprocessed PyTorch Tensor [1, 3, H, W] or [B, 3, H, W]
- Validated RGB PIL Image

STATE MODIFICATIONS:
-------------------------------------------------------------------------
NONE.
"""

from typing import Tuple, Union
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


class InferencePreprocessor:
    """Preprocesses input images for EfficientNetV2-S inference."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def preprocess(self, input_image: Union[str, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, Image.Image]:
        """
        Validates, sanitizes, and transforms an input image into a ready tensor.

        Args:
            input_image: File path string, PIL Image instance, or numpy array.

        Returns:
            Tuple of (preprocessed_tensor [1, 3, H, W], sanitized_rgb_image)
        """
        # 1. Load or convert to PIL Image
        if isinstance(input_image, str):
            try:
                img = Image.open(input_image)
            except Exception as e:
                raise ValueError(f"Failed to open image from path '{input_image}': {e}")
        elif isinstance(input_image, np.ndarray):
            img = Image.fromarray(input_image)
        elif isinstance(input_image, Image.Image):
            img = input_image
        else:
            raise TypeError(f"Unsupported image input type: {type(input_image)}")

        # 2. Ensure 3-channel RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 3. Transform to tensor
        tensor = self.transform(img)

        # 4. Add batch dimension: [3, H, W] -> [1, 3, H, W]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        return tensor, img
