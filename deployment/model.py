"""
Module: deployment/model.py
Project: Leafcare — Plant Disease Classification (FastAPI Deployment)

PURPOSE:
-------------------------------------------------------------------------
Loads the trained EfficientNetV2-S model checkpoint once at startup and
executes fast, deterministic inference under torch.inference_mode().
Supports CPU-only production containers and GPU acceleration.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from models.efficientnetv2s import build_model
from deployment.config import settings
from deployment.exceptions import ModelNotReadyError, InferenceExecutionError

logger = logging.getLogger("leafcare.deployment.model")


class EfficientNetPredictor:
    """
    Production inference engine for EfficientNetV2-S 38-class plant pathology.
    """

    def __init__(self, checkpoint_path: str = settings.CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device("cpu")
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.num_classes: int = 0
        self.is_loaded: bool = False

    def load(self) -> None:
        """
        Loads the trained model checkpoint once into application memory and executes warmup pass.
        """
        logger.info(f"Initiating model loading from: {self.checkpoint_path}")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Baseline checkpoint not found at: {self.checkpoint_path}")

        # 1. Device Selection: Check explicit config/env preference first, then auto-detect
        configured_device = getattr(settings, "DEVICE", "auto").lower()
        if configured_device == "cpu":
            self.device = torch.device("cpu")
            device_desc = "CPU (Configured via DEVICE=cpu)"
        elif configured_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_desc = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_desc = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            self.device = torch.device("cpu")
            device_desc = "CPU (Hardware fallback)"

        logger.info(f"Target Compute Device: {device_desc}")

        # 2. Checkpoint Ingestion (Read-only)
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file: {e}")

        # 3. Class Mapping Resolution
        if "class_to_idx" not in ckpt:
            raise KeyError("Mandatory 'class_to_idx' mapping not found in checkpoint.")

        self.class_to_idx = ckpt["class_to_idx"]
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

        if self.num_classes != settings.EXPECTED_CLASSES:
            logger.warning(
                f"Detected {self.num_classes} classes in checkpoint (expected {settings.EXPECTED_CLASSES})."
            )

        # 4. Construct Architecture & Ingest State Dict
        self.model = build_model(num_classes=self.num_classes, pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 5. Pre-heat / Warmup GPU/CPU Kernel to eliminate initial request latency
        try:
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE, device=self.device)
                _ = self.model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Warmup pass non-fatal warning: {e}")

        self.is_loaded = True
        logger.info(
            f"Model loaded successfully | Architecture: {settings.MODEL_NAME} | "
            f"Classes: {self.num_classes} | Device: {self.device.type}"
        )

    def predict(self, tensor: torch.Tensor, top_k: int = 5) -> Dict[str, Any]:
        """
        Performs forward pass under torch.inference_mode() and returns structured top-K predictions.

        Args:
            tensor: Preprocessed FloatTensor [1, 3, 224, 224]
            top_k: Number of highest probability candidates to extract (default: 5)

        Returns:
            Dictionary containing predicted_class, confidence, top_5_predictions, inference_time_ms.
        """
        if not self.is_loaded or self.model is None:
            raise ModelNotReadyError("Model is not loaded. Cannot execute inference.")

        t0 = time.perf_counter()

        try:
            tensor = tensor.to(self.device)

            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=-1)[0]

            top_k_count = min(top_k, self.num_classes)
            top_probs, top_indices = torch.topk(probabilities, k=top_k_count)

            top_probs_np = top_probs.cpu().numpy()
            top_indices_np = top_indices.cpu().numpy()

            top_predictions = []
            for prob, idx in zip(top_probs_np, top_indices_np):
                prob_float = float(prob)
                cls_name = self.idx_to_class[int(idx)]
                top_predictions.append({
                    "class": cls_name,
                    "confidence": round(prob_float, 4),
                    "confidence_percentage": round(prob_float * 100.0, 2)
                })

            top_1_class = top_predictions[0]["class"]
            top_1_conf = top_predictions[0]["confidence"]
            top_1_conf_pct = top_predictions[0]["confidence_percentage"]

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            inference_time = (time.perf_counter() - t0) * 1000.0

            return {
                "predicted_class": top_1_class,
                "confidence": top_1_conf,
                "confidence_percentage": top_1_conf_pct,
                "top_5_predictions": top_predictions,
                "inference_time_ms": round(inference_time, 2)
            }

        except Exception as e:
            logger.error(f"Inference forward-pass exception: {e}", exc_info=True)
            raise InferenceExecutionError(detail=f"Inference execution failed: {str(e)}")
