# Controlled Experiment Plan — EXP-01: Partial EfficientNetV2-S Fine-Tuning

**Project:** Plant Disease Classification — Leafcare Model
**Experiment ID:** `EXP-01`
**Control Baseline (EXP-00):** Top-1 Accuracy: **93.32%** | Macro F1: **91.98%**

---

## 1. Hypothesis & Motivation

### Hypothesis:
Fine-tuning the upper stages (Stages 5 and 6) of the EfficientNetV2-S backbone alongside the classification head using a **differential learning rate** will enable the network's high-level feature extractors to learn specialized phytopathological leaf texture representations (e.g. concentric fungal rings, chlorotic halos, bacterial spot margins), significantly reducing confusion among the weakest classes without overfitting or catastrophic forgetting.

### Why This Experiment is Justified:
1. **Diagnostic Evidence:** Diagnostic analysis demonstrated that the primary error mode is **fine-grained visual lesion ambiguity** (e.g. Tomato Early Blight vs Target Spot vs Late Blight), not class imbalance or sample scarcity.
2. **Current Constraint:** In `EXP-00`, 99.76% of parameters were frozen. The linear classifier had to operate exclusively on generic ImageNet features.
3. **Controlled Scope:** Rather than unfreezing the entire backbone (which risks overfitting on 38.5k samples and increases training instability), unfreezing only the top stages preserves low-level edge/texture filters while adapting task-specific semantic features.

---

## 2. Precise Changes vs Baseline

| Architectural / Training Element | Baseline (EXP-00) | Proposed Experiment (EXP-01) | Rationale |
| :--- | :--- | :--- | :--- |
| **Backbone Stages 0–4** | Frozen (`requires_grad=False`) | **Frozen** (`requires_grad=False`) | Retain generic low-level edge and color representations |
| **Backbone Stages 5–6 (Top)** | Frozen (`requires_grad=False`) | **Trainable** (`requires_grad=True`) | Adapt domain-specific disease lesion morphology |
| **Classifier Head** | Trainable (lr = 1e-3) | **Trainable** (lr = 1e-3) | Final 38-class classification mapping |
| **Backbone Learning Rate** | N/A (Frozen) | **1e-5** (100× smaller than head) | Prevent destroying pretrained ImageNet weights |
| **Optimizer** | AdamW (weight_decay = 1e-2) | AdamW (with differential parameter groups) | Standard decoupled weight decay |
| **Epochs** | 15 | 15 (with early stopping patience = 5) | Controlled convergence |
| **Loss Function** | Standard CrossEntropyLoss | Standard CrossEntropyLoss | Keep loss identical to isolate architecture change |
| **Resolution & Batch Size** | 224 × 224, Batch = 32 | 224 × 224, Batch = 32 | Maintain identical data pipeline |

---

## 3. Risk Assessment & Mitigation Strategies

| Potential Risk | Severity | Mitigation Strategy |
| :--- | :--- | :--- |
| **Backbone Weight Distortion** | High | Differential learning rate (1e-5 for backbone vs 1e-3 for head) |
| **Overfitting on Small Classes** | Medium | Maintain AdamW weight decay (0.01) + Cosine Annealing with warmup |
| **Training Instability** | Low | Gradient clipping (max_norm = 1.0) + AMP mixed precision |

---

## 4. Success and Decision Criteria

An experiment will be considered **SUCCESSFUL** and eligible to become the new benchmark if:
1. **Primary Metric:** Macro F1 strictly exceeds **91.98%** (Target: >= 93.5%).
2. **Secondary Metric:** Top-1 Accuracy strictly exceeds **93.32%** (Target: >= 94.5%).
3. **Worst-Class Robustness:** F1 score of the weakest baseline classes (`tomato___early_blight`, `tomato___target_spot`) improves by at least +3.0 percentage points.
4. **No Severe Degradation:** No individual class suffers an F1 drop > 2.0 percentage points compared to baseline.

> [!WARNING]
> If Top-1 Accuracy rises (e.g. to 93.6%) but Macro F1 drops (e.g. to 91.5%), the experiment **MUST BE REJECTED**.

---

## 5. Execution Protocol

> [!IMPORTANT]
> In accordance with user control rules, **NO TRAINING IS AUTOMATICALLY LAUNCHED**.
> The exact command provided below can be manually executed when approved by the user.

### Manual Command for User:
```powershell
python train.py --config config/config.yaml --experiment EXP-01 --fine-tune-backbone
```
