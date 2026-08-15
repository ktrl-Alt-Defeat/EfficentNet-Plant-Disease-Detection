"""
================================================================================
Models Package — Plant Disease Classification
================================================================================

This package exposes model architectures and factory constructors for transfer
learning experiments.

Exported Classes & Functions:
- EfficientNetV2SClassifier: Production-grade PyTorch nn.Module for EfficientNetV2-S.
- build_model: Factory function for constructing configured classifier instances.
- build_efficientnet_v2_s: Alias factory function for backward compatibility.
================================================================================
"""

from models.efficientnetv2s import (
    EfficientNetV2SClassifier,
    build_model,
    build_efficientnet_v2_s
)

__all__ = [
    "EfficientNetV2SClassifier",
    "build_model",
    "build_efficientnet_v2_s"
]
