"""
Module: inference/predictor.py
Project: Plant Disease Classification — Leafcare Model (EfficientNetV2-S)

PURPOSE:
-------------------------------------------------------------------------
Provides a clean, production-ready inference interface (PlantDiseasePredictor)
for single-image disease diagnosis, confidence assessment, and top-K rankings.

WHY THIS MODULE EXISTS:
-------------------------------------------------------------------------
Decouples model execution from training code. Encapsulates model loading,
class mapping resolution, device management, and output structuring into
a single, robust, production-tested API.

HOW IT WORKS:
-------------------------------------------------------------------------
1. Loads model architecture and pretrained weights from checkpoint file.
2. Resolves class index to human-readable plant disease name mappings.
3. Automatically selects fastest available compute hardware (CUDA or CPU).
4. Sets model to evaluation mode (model.eval()).
5. Executes inference under torch.no_grad().
6. Computes softmax probability distribution.
7. Packages top-K predictions into structured response dict.

INPUTS:
-------------------------------------------------------------------------
- Image file path, PIL Image, or numpy array
- top_k: Number of highest-probability candidate classes to return (default 5)
- confidence_threshold: Acceptance threshold (default 0.70)

OUTPUTS:
-------------------------------------------------------------------------
- Structured dictionary with predicted class, confidence, top_K, and status.

STATE MODIFICATIONS:
-------------------------------------------------------------------------
NONE. Inference only.
"""

import os
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .preprocessing import InferencePreprocessor


class PlantDiseasePredictor:
    """
    Production-ready disease classification predictor for Leafcare model.
    """

    def __init__(
        self,
        checkpoint_path: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        device: Optional[str] = None,
        image_size: int = 224,
        confidence_threshold: float = 0.70
    ):
        """
        Initializes the predictor, loads model checkpoint, and maps class indices.

        Args:
            checkpoint_path: Path to trained PyTorch checkpoint (.pt)
            class_to_idx: Optional dictionary mapping class names to indices.
                          If None, loaded from checkpoint metadata.
            device: 'cuda', 'cpu', or None (auto-detect)
            image_size: Input spatial resolution (default 224)
            confidence_threshold: Acceptance threshold for predictions (default 0.70)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

        # Hardware selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.preprocessor = InferencePreprocessor(image_size=image_size)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load class mappings
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        elif "class_to_idx" in ckpt:
            self.class_to_idx = ckpt["class_to_idx"]
        else:
            raise KeyError("Class mapping 'class_to_idx' not found in checkpoint or arguments.")

        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        # Construct and load model
        self.model = self._build_model(ckpt)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, checkpoint: Dict[str, Any]) -> nn.Module:
        """Constructs EfficientNetV2-S model and loads state dict."""
        from models.efficientnetv2s import build_model
        model = build_model(num_classes=self.num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Performs disease diagnosis on input image and returns structured result.

        Args:
            image_input: File path, PIL Image, or numpy array.
            top_k: Number of candidate classes to return.

        Returns:
            Structured JSON-compatible dictionary.
        """
        t0 = time.perf_counter()

        try:
            tensor, _ = self.preprocessor.preprocess(image_input)
        except Exception as e:
            return {
                "predicted_class": None,
                "confidence": 0.0,
                "top_k": [],
                "status": "invalid_image",
                "error_message": str(e),
                "inference_time_ms": round((time.perf_counter() - t0) * 1000.0, 2)
            }

        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]

        top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, self.num_classes))
        
        top_k_list = []
        for p, idx in zip(top_k_probs.cpu().numpy(), top_k_indices.cpu().numpy()):
            top_k_list.append({
                "class": self.idx_to_class[int(idx)],
                "probability": round(float(p), 4)
            })

        top_pred_class = top_k_list[0]["class"]
        top_confidence = top_k_list[0]["probability"]

        status = "accepted" if top_confidence >= self.confidence_threshold else "low_confidence"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_class": top_pred_class,
            "confidence": top_confidence,
            "top_k": top_k_list,
            "status": status,
            "confidence_threshold": self.confidence_threshold,
            "inference_time_ms": round(elapsed_ms, 2)
        }
