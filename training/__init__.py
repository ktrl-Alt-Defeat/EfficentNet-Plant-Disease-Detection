"""
================================================================================
Training Package — Plant Disease Classification
================================================================================

Exposes:
- Trainer: Production-grade training and validation orchestrator.
- build_loss: Loss function factory.
- build_scheduler: Learning rate scheduler factory.
================================================================================
"""

from training.losses import build_loss
from training.scheduler import build_scheduler
from training.trainer import Trainer

__all__ = [
    "Trainer",
    "build_loss",
    "build_scheduler"
]
