"""
Module: deployment/preprocessing.py
Project: Leafcare — Plant Disease Classification (FastAPI Deployment)

PURPOSE:
-------------------------------------------------------------------------
Transforms incoming multipart raw image bytes into sanitized PyTorch tensors
using the EXACT preprocessing pipeline established in data/dataset.py and
inference/preprocessing.py.

WHY THIS MODULE EXISTS:
-------------------------------------------------------------------------
Guarantees 100% deterministic feature consistency between training, offline
evaluation, and live production serving. Any divergence in color mode, spatial
resolution, or normalization scale would silently degrade diagnostic accuracy.

PREPROCESSING PIPELINE SPECIFICATION:
-------------------------------------------------------------------------
1. In-Memory Image Ingestion: Decodes raw byte buffer safely via PIL.Image.open(io.BytesIO).
2. Format & Integrity Validation: Ensures non-corrupted JPEG/PNG/WEBP stream.
3. Color Conversion: Converts arbitrary modes (RGBA, CMYK, Grayscale) to 3-channel RGB.
4. Deterministic Resize: Resizes input to spatial dimension (224, 224) using bilinear interpolation.
5. Tensor Normalization: Converts uint8 [0, 255] pixels to float32 [0.0, 1.0] via transforms.ToTensor().
6. Batch Dimension Expansion: Reshapes tensor to [1, 3, 224, 224].

INPUT:
-------------------------------------------------------------------------
- Raw image bytes (bytes) from FastAPI UploadFile

OUTPUT:
-------------------------------------------------------------------------
- torch.FloatTensor of shape [1, 3, 224, 224] in range [0.0, 1.0]

STATE MODIFICATIONS:
-------------------------------------------------------------------------
NONE. Pure mathematical transformations in memory.
"""

import io
from typing import Tuple
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

from deployment.exceptions import InvalidImageError, UnsupportedMediaFormatError
from deployment.config import settings


class DeploymentPreprocessor:
    """
    Production preprocessor adapting incoming HTTP payloads to evaluation-parity tensors.
    """

    def __init__(self, image_size: int = settings.IMAGE_SIZE):
        """
        Initializes torchvision transform pipeline matching data/dataset.py.
        """
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def preprocess_bytes(self, image_bytes: bytes, content_type: str = "") -> torch.Tensor:
        """
        Validates, decodes, and preprocesses raw image bytes into an inference-ready tensor.

        Args:
            image_bytes: Raw binary content of uploaded image file.
            content_type: Optional MIME type from HTTP multipart header.

        Returns:
            torch.FloatTensor of shape [1, 3, 224, 224].

        Raises:
            InvalidImageError: If the byte payload is empty or corrupted.
            UnsupportedMediaFormatError: If the image format is unsupported.
        """
        # 1. Byte payload validation
        if not image_bytes or len(image_bytes) == 0:
            raise InvalidImageError(detail="Uploaded image file is empty (0 bytes received).")

        if len(image_bytes) > settings.MAX_IMAGE_SIZE_BYTES:
            raise InvalidImageError(
                detail=f"Uploaded image exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB."
            )

        # 2. In-memory decode
        try:
            image_stream = io.BytesIO(image_bytes)
            img = Image.open(image_stream)
            img.load()  # Force reading entire raster stream to detect partial corruption
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise InvalidImageError(detail=f"Could not decode image file: {str(e)}")

        # 3. Format validation
        img_format = (img.format or "").upper()
        if img_format not in ["JPEG", "JPG", "PNG", "WEBP", "BMP"]:
            raise UnsupportedMediaFormatError(
                detail=f"Image format '{img_format}' is not supported. Please upload JPEG, PNG, or WEBP."
            )

        # 4. Color space normalization to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 5. Transform: Resize to 224x224 and convert to FloatTensor [0.0, 1.0]
        tensor = self.transform(img)

        # 6. Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        return tensor
