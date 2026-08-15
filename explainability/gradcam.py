"""
Module: explainability/gradcam.py
Project: Plant Disease Classification — Leafcare Model (EfficientNetV2-S)

PURPOSE:
-------------------------------------------------------------------------
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
the spatial attention of the trained EfficientNetV2-S model on plant disease images.

WHY THIS MODULE EXISTS:
-------------------------------------------------------------------------
Visual explanations allow pathologists and ML engineers to verify whether the
network is focusing on true diagnostic disease lesions (e.g. necrotic rings,
chlorotic halos, fungal spot patterns) or spurious background artifacts
(soil, fingers, background lighting, shadows).

HOW IT WORKS:
-------------------------------------------------------------------------
1. Registers forward and backward hooks on the final convolutional layer of
   the EfficientNetV2-S backbone.
2. Runs a forward pass in eval mode to capture spatial feature activations A.
3. Computes the gradient of the target class logit with respect to feature maps:
     d(y_c) / d(A_k)
4. Computes channel importance weights by global average pooling gradients:
     alpha_k = (1/Z) * sum_i sum_j (d(y_c) / d(A_{i,j}^k))
5. Computes the weighted combination followed by ReLU:
     Heatmap = ReLU( sum_k alpha_k * A_k )
6. Normalizes the heatmap to [0, 1] and resizes it to match original image dimensions.
7. Overlays the colored heatmap onto the original RGB image.

INPUTS:
-------------------------------------------------------------------------
- Pretrained/Trained EfficientNetV2-S model in eval mode
- Input image tensor [1, 3, 224, 224]
- Target class index (defaults to predicted class)

OUTPUTS:
-------------------------------------------------------------------------
- 2D numpy heatmap [224, 224] in range [0, 1]
- 3D numpy overlay image [224, 224, 3]
- Annotated visual comparison panels

STATE MODIFICATIONS:
-------------------------------------------------------------------------
NONE. Model weights remain 100% frozen. Gradients are computed only through
feature maps for CAM synthesis and immediately discarded.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for EfficientNetV2-S architectures.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Initializes GradCAM by identifying the target convolutional layer and registering hooks.

        Args:
            model: PyTorch model (EfficientNetV2-S)
            target_layer: Specific layer to hook. If None, automatically discovers
                          the final convolutional layer in model backbone.
        """
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            self.target_layer = self._discover_target_layer()
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _discover_target_layer(self) -> nn.Module:
        """
        Automatically identifies the final convolutional layer of EfficientNetV2-S.
        In EfficientNetV2, features is a Sequential where features[-1] (or features[7])
        is the final 1280-channel Conv2dNormActivation stage.
        """
        # Unwrap if model is nested (e.g. model.model or model.backbone)
        base_model = self.model
        if hasattr(base_model, "model"):
            base_model = base_model.model

        if hasattr(base_model, "features"):
            # Target the final block of features
            target = base_model.features[-1]
            return target

        # Fallback: search backwards for the last Conv2d module
        for name, module in reversed(list(base_model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module

        raise RuntimeError("Could not automatically discover target convolutional layer.")

    def _register_hooks(self) -> None:
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input_t, output_t):
            self.activations = output_t.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        """Removes registered hooks from PyTorch model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates Grad-CAM activation heatmap for the given input tensor.

        Args:
            input_tensor: Image tensor of shape [1, 3, H, W]
            target_class_idx: Target class index to explain. If None, uses top predicted class.

        Returns:
            Tuple of:
                - heatmap: 2D numpy array [H, W] normalized to [0, 1]
                - predicted_class_idx: int
                - confidence: float
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass with gradient tracking enabled for CAM computation
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=-1)

        pred_idx = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred_idx].item())

        if target_class_idx is None:
            target_class_idx = pred_idx

        # Target score for backprop
        target_score = logits[0, target_class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Compute importance weights via global average pooling of gradients: [1, C, 1, 1]
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps: [1, C, H, W] * [1, C, 1, 1] -> [1, H, W]
        cam = torch.sum(self.activations * pooled_gradients, dim=1).squeeze(0)

        # Apply ReLU to retain only features with positive influence
        cam = torch.clamp(cam, min=0.0)

        cam_np = cam.cpu().numpy()
        max_val = np.max(cam_np)
        if max_val > 1e-8:
            cam_np = cam_np / max_val
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to original input spatial dimensions (e.g. 224x224)
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        return heatmap_resized, pred_idx, confidence


def apply_gradcam_overlay(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Applies colorized heatmap overlay onto the original RGB image.

    Args:
        original_rgb: RGB image numpy array [H, W, 3] in range [0, 255] or [0, 1]
        heatmap: 2D numpy array [H, W] in range [0, 1]
        alpha: Weight for heatmap blending (1-alpha for original image)
        colormap: OpenCV colormap enum

    Returns:
        Overlay image as RGB numpy array [H, W, 3] in range [0, 255] (uint8)
    """
    if original_rgb.dtype != np.uint8:
        if np.max(original_rgb) <= 1.0:
            original_rgb = (original_rgb * 255.0).astype(np.uint8)
        else:
            original_rgb = original_rgb.astype(np.uint8)

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
    return overlay


def save_gradcam_comparison_panel(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    overlay_rgb: np.ndarray,
    output_path: str,
    true_class: str,
    pred_class: str,
    confidence: float
) -> None:
    """Saves a 3-panel visualization: Original Image | Grad-CAM Heatmap | Overlay."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Original Image\nTrue: {true_class}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Activation Map", fontsize=10)
    axes[1].axis("off")

    is_correct = (true_class == pred_class)
    status_str = "CORRECT" if is_correct else "INCORRECT"
    color_title = "#16A34A" if is_correct else "#DC2626"
    
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(
        f"Overlay ({status_str})\nPred: {pred_class} ({confidence*100:.1f}%)",
        fontsize=10,
        color=color_title,
        fontweight="bold"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
