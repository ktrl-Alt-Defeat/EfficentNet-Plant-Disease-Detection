"""
================================================================================
Loss Functions Module — Plant Disease Classification
Milestone 3 — Production-Grade Training Infrastructure
================================================================================

This module encapsulates loss function construction and configuration for
training deep learning models.

Educational Deep Dive:
----------------------
1. What is Cross-Entropy Loss?
   CrossEntropyLoss combines two mathematical operations:
     a) LogSoftmax: Converts unnormalized raw model logits into log-probabilities.
     b) Negative Log-Likelihood (NLL): Penalizes low predicted probabilities for
        the true target class.
   Formula for a single sample with true class y:
     Loss = -log(softmax(logits)[y]) = -logits[y] + log(sum(exp(logits)))

2. Why Label Smoothing Regularization?
   - Standard Cross-Entropy targets a "hard" one-hot distribution [0, 0, 1, 0...],
     encouraging the network to produce infinite logit values to reach 100% confidence.
   - Label Smoothing replaces the hard target 1.0 with (1.0 - epsilon) and distributes
     epsilon across all classes.
   - Benefit: Prevents overconfidence, reduces overfitting, and improves model calibration
     and generalization on noisy agricultural leaf disease images.
   - Configurable: Setting label_smoothing=0.0 reverts to standard hard cross-entropy.
================================================================================
"""

from typing import Dict, Any
import torch
import torch.nn as nn


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to construct the loss criterion from configuration.

    Args:
        config: Dictionary containing training/loss configuration.
                Expected structure: config['training']['loss'] or config['loss'].

    Returns:
        nn.Module: Configured PyTorch loss criterion.

    Raises:
        ValueError: If an unsupported loss function name is configured
                    or if label_smoothing is outside [0.0, 1.0).
    """
    # Extract loss configuration safely from nested 'training' or top-level 'loss'
    training_cfg = config.get("training", {})
    loss_cfg = training_cfg.get("loss", config.get("loss", {}))

    loss_name = loss_cfg.get("name", "cross_entropy").lower()
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    # Validate label smoothing bounds [0.0, 1.0)
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(
            f"label_smoothing must be in range [0.0, 1.0), got: {label_smoothing}. "
            "A value of 1.0 would mean no target class signal."
        )

    if loss_name in ("cross_entropy", "crossentropy", "ce"):
        # nn.CrossEntropyLoss expects raw unnormalized logits [B, num_classes]
        # and integer class indices [B] with dtype torch.int64 (torch.long).
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(
            f"Unsupported loss function: '{loss_name}'. "
            "Supported options in Milestone 3: 'cross_entropy'."
        )
