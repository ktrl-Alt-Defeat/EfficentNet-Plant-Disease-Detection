# Baseline Metadata Report — EXP-00 (Official Baseline)

**Project:** Plant Disease Classification — Leafcare Model
**Model Architecture:** EfficientNetV2-S
**Pretrained Weights:** ImageNet-1K (Torchvision `DEFAULT`)
**Backbone State:** Frozen (`requires_grad=False`)
**Classifier Head:** Linear(`in_features=1280`, `out_features=38`, `requires_grad=True`)
**Checkpoint Path:** `C:\cts\Efficientnet\Leafcare-model\checkpoints\baseline_38class_effnetv2s.pt`
**Experiment ID:** `EXP-00`

---

## 1. Baseline Model Parameters

| Parameter Category | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | 20,226,166 | Complete EfficientNetV2-S architecture |
| **Trainable Parameters** | 48,678 | Classification head weights (1280 × 38) + biases (38) |
| **Frozen Parameters** | 20,177,488 | Feature extraction backbone |
| **Trainable Ratio** | 0.241% | Percentage of parameters updated during baseline training |

---

## 2. Training Configuration Summary

- **Input Resolution:** 224 × 224 (RGB)
- **Batch Size:** 32
- **Optimizer:** AdamW (lr = 1e-3, weight_decay = 1e-2)
- **Scheduler:** CosineAnnealingLR (T_max = 15, eta_min = 1e-6)
- **Mixed Precision:** AMP (`torch.cuda.amp.autocast`) Enabled
- **Loss Function:** CrossEntropyLoss (Standard, unweighted)
- **Gradient Clipping:** Max norm = 1.0

---

## 3. Official Test Evaluation Metrics

| Metric | Baseline Value (EXP-00) | Priority for Comparison |
| :--- | :--- | :--- |
| **Top-1 Test Accuracy** | **93.32%** | Secondary |
| **Macro F1 Score** | **91.98%** | **PRIMARY DECISION METRIC** |
| **Weighted F1 Score** | **93.26%** | Secondary |
| **Macro Precision** | **91.56%** | Supporting |
| **Macro Recall** | **92.70%** | Supporting |
| **Weighted Precision** | **93.33%** | Supporting |
| **Weighted Recall** | **93.32%** | Supporting |
| **Test Loss** | **0.2212** | Supporting |

---

## 4. Preservation & Immutability Notice

> [!IMPORTANT]
> This baseline checkpoint (`checkpoints/baseline_38class_effnetv2s.pt`) is frozen as the official benchmark.
> Future experiments (e.g. `EXP-01` partial fine-tuning) must strictly improve upon **Macro F1 (91.98%)** without catastrophic degradation on any individual class.
