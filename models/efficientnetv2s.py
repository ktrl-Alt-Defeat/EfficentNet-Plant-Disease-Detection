"""
================================================================================
EfficientNetV2-S Transfer Learning Model Module
Plant Disease Classification Pipeline — Milestone 2
================================================================================

This module encapsulates the transfer learning architecture built on top of
torchvision's official pretrained EfficientNetV2-S network.

Key Design Decisions & Architecture Highlights:
------------------------------------------------
1. Pretrained Feature Extractor:
   - Uses official ImageNet-1K pretrained weights (`EfficientNet_V2_S_Weights.DEFAULT`).
   - Deep hierarchical convolutional features trained on 1.28M images are reused,
     providing fast convergence and high generalization on agricultural leaf images.

2. Programmatic Dimension Discovery:
   - Rather than hardcoding the 1280 feature dimension, the code inspects the
     original classification layer dynamically (`in_features = original_linear.in_features`).

3. Custom Classification Head:
   - Replaces the original 1000-class ImageNet linear projection with:
       Dropout(p=dropout_rate) -> Linear(1280 -> num_classes)
   - Dynamic class support: `num_classes` is derived at runtime from the dataset.

4. Two-Stage Transfer Learning Support:
   - Mode A (Stage 1: Head Training): Backbone is frozen (`requires_grad = False`).
     Only the custom classifier head is updated.
   - Mode B (Stage 2: Full Fine-Tuning): Backbone is unfrozen (`requires_grad = True`).
     Differential learning rates (`backbone_lr` and `classifier_lr`) are used
     to fine-tune deep features without destroying pretrained representations.
================================================================================
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_V2_S_Weights


class EfficientNetV2SClassifier(nn.Module):
    """
    EfficientNetV2-S Transfer-Learning Classifier for Plant Disease Classification.

    Attributes:
        num_classes (int): Number of target plant disease classes.
        pretrained (bool): Whether to load official ImageNet pretrained weights.
        dropout_rate (float): Dropout probability applied before final linear layer.
        is_backbone_frozen (bool): Tracks whether the feature backbone is currently frozen.
        in_features (int): Programmatically extracted input feature dimension (1280).
        model (nn.Module): The underlying torchvision EfficientNetV2-S model.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = True
    ) -> None:
        """
        Initialize the EfficientNetV2-S classifier.

        Args:
            num_classes: Positive integer representing the number of output disease classes.
            pretrained: If True, loads official torchvision ImageNet pretrained weights.
            dropout: Dropout probability between 0.0 and 1.0 (default: 0.3).
            freeze_backbone: If True, freezes backbone weights and only trains classifier head.

        Raises:
            ValueError: If num_classes <= 0 or dropout is not in [0.0, 1.0].
            AttributeError: If native torchvision classifier structure cannot be inspected.
        """
        super().__init__()

        # --- 1. Input Validation ---
        # Ensure class count is a positive integer (never hardcoded)
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive integer, got: {num_classes}")

        # Ensure dropout rate is within valid probability bounds [0.0, 1.0]
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0.0 and 1.0, got: {dropout}")

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout
        self.is_backbone_frozen = freeze_backbone

        # --- 2. Load Pretrained EfficientNetV2-S Backbone ---
        # Load official torchvision weights (DEFAULT maps to best available ImageNet-1K weights)
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.model = models.efficientnet_v2_s(weights=weights)

        # --- 3. Programmatically Inspect Native Classifier Dimension ---
        # EfficientNetV2-S classifier is an nn.Sequential: [Dropout(p=0.2), Linear(in=1280, out=1000)]
        if not hasattr(self.model, "classifier") or not isinstance(self.model.classifier, nn.Sequential):
            raise AttributeError("Loaded model does not have the expected nn.Sequential classifier attribute.")

        # Find the Linear layer to extract in_features programmatically without hardcoding
        original_linear = None
        for module in self.model.classifier:
            if isinstance(module, nn.Linear):
                original_linear = module
                break

        if original_linear is None:
            raise AttributeError("Could not find Linear layer in original EfficientNetV2-S classifier.")

        # In torchvision EfficientNetV2-S, in_features is 1280
        self.in_features: int = original_linear.in_features

        # --- 4. Construct Custom Disease Classification Head ---
        # Replace the 1000-class ImageNet output with our domain-specific num_classes output
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(in_features=self.in_features, out_features=self.num_classes, bias=True)
        )

        # --- 5. Initialize Weights for New Classification Head ---
        # Kaiming normal initialization for linear weights ensures healthy initial gradient flow
        nn.init.kaiming_normal_(self.model.classifier[1].weight, nonlinearity="linear")
        if self.model.classifier[1].bias is not None:
            nn.init.zeros_(self.model.classifier[1].bias)

        # --- 6. Apply Initial Freezing Strategy ---
        # If freeze_backbone is True: freeze features, train classifier only (Stage 1)
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the network.

        Data Flow:
            Input [B, 3, 224, 224]
              -> Backbone Feature Extraction (Conv + FusedMBConv + MBConv blocks)
              -> Adaptive Global Average Pooling (1280 channels)
              -> Flatten
              -> Dropout(p=dropout_rate)
              -> Linear(1280 -> num_classes)
              -> Class Logits [B, num_classes]

        Args:
            x: Input image tensor of shape [B, 3, H, W] (expected [B, 3, 224, 224]).

        Returns:
            torch.Tensor: Unnormalized class logits of shape [B, num_classes].

        Raises:
            ValueError: If input tensor does not have 4 dimensions or 3 channels.
        """
        # Validate batch tensor dimensions: must be 4D [Batch, Channels, Height, Width]
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got ndim={x.ndim} with shape {x.shape}")

        # Validate color channels: must be 3 (RGB)
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 color channels (RGB), got {x.shape[1]} channels in tensor of shape {x.shape}")

        # Pass through full EfficientNetV2-S model and return raw unnormalized logits
        return self.model(x)

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters in the EfficientNetV2-S feature extractor backbone.

        Used in Transfer Learning Stage 1:
        - Sets requires_grad = False for all parameters in model.features.
        - Ensures requires_grad = True for all parameters in model.classifier.
        - Prevents catastrophic forgetting of pretrained ImageNet visual representations.
        """
        # Freeze all backbone layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Keep custom classification head trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.is_backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all parameters across the entire network (backbone and classifier).

        Used in Transfer Learning Stage 2:
        - Sets requires_grad = True across all layers.
        - Enables end-to-end full fine-tuning with differential learning rates.
        """
        for param in self.model.parameters():
            param.requires_grad = True

        self.is_backbone_frozen = False

    def total_parameters(self) -> int:
        """
        Calculate total number of parameters in the network.
        
        Returns:
            int: Total parameter count.
        """
        return sum(p.numel() for p in self.parameters())

    def trainable_parameters(self) -> int:
        """
        Calculate number of currently trainable parameters (requires_grad = True).
        
        Returns:
            int: Trainable parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        """
        Calculate number of currently frozen parameters (requires_grad = False).
        
        Returns:
            int: Frozen parameter count.
        """
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def backbone_trainable_parameters(self) -> int:
        """
        Calculate number of trainable parameters in the backbone feature extractor.
        
        Returns:
            int: Backbone trainable parameter count (0 when frozen).
        """
        return sum(p.numel() for p in self.model.features.parameters() if p.requires_grad)

    def classifier_trainable_parameters(self) -> int:
        """
        Calculate number of trainable parameters in the custom classification head.
        
        Returns:
            int: Classifier trainable parameter count.
        """
        return sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)

    def parameter_groups(
        self,
        backbone_lr: float,
        classifier_lr: float,
        weight_decay: float = 1e-4
    ) -> List[Dict[str, Any]]:
        """
        Construct differential learning rate parameter groups for PyTorch optimizers.

        Differential Learning Rate Strategy:
        - The pretrained backbone uses a smaller learning rate (e.g. 1e-5) to preserve
          general visual features while adapting to leaf textures.
        - The randomly initialized classifier head uses a larger learning rate (e.g. 1e-4)
          for faster convergence on domain-specific disease labels.

        Args:
            backbone_lr: Learning rate for feature extractor parameters.
            classifier_lr: Learning rate for classification head parameters.
            weight_decay: L2 regularization coefficient.

        Returns:
            List of parameter group dictionaries compatible with torch.optim.AdamW / Adam.
        """
        backbone_params = [p for p in self.model.features.parameters() if p.requires_grad]
        classifier_params = [p for p in self.model.classifier.parameters() if p.requires_grad]

        groups: List[Dict[str, Any]] = []

        # Backbone parameter group (only included if backbone has trainable params)
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone"
            })

        # Classifier head parameter group (always included)
        if classifier_params:
            groups.append({
                "params": classifier_params,
                "lr": classifier_lr,
                "weight_decay": weight_decay,
                "name": "classifier"
            })

        return groups


def build_model(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = True
) -> EfficientNetV2SClassifier:
    """
    Factory function to construct an EfficientNetV2SClassifier instance.

    Args:
        num_classes: Number of disease classes discovered from dataset.
        pretrained: If True, loads official ImageNet pretrained weights.
        dropout: Dropout probability for classifier head (default: 0.3).
        freeze_backbone: If True, freezes backbone feature extractor.

    Returns:
        Configured EfficientNetV2SClassifier instance.
    """
    return EfficientNetV2SClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )


# Alias for compatibility across various naming conventions
build_efficientnet_v2_s = build_model
