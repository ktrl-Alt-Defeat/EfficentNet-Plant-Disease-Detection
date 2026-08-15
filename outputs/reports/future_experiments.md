# Future Controlled Experiment Proposals (Post-Milestone 5)

**Project:** Plant Disease Classification — Leafcare Model
**Control Benchmark (EXP-00):** Top-1 Accuracy: **93.32%** | Macro F1: **91.98%**
**Status:** PROPOSALS ONLY — NO TRAINING EXECUTED

---

## Proposed Experiments Roadmap

### 1. EXP-01: Partial Backbone Fine-Tuning
- **Hypothesis:** Unfreezing upper backbone stages (Stages 5 & 6) with a differential learning rate (`1e-5` for backbone, `1e-3` for head) will resolve fine-grained lesion confusion among tomato diseases without overfitting.
- **Target Metric:** Macro F1 ≥ 93.5% (Baseline: 91.98%).

### 2. EXP-02: Post-Processing Temperature Scaling Calibration
- **Hypothesis:** Optimizing a single scalar temperature parameter on validation logits will reduce ECE from 3.8% to < 1.5% without modifying model weights.

### 3. EXP-03: Multi-Architecture Comparison
- **Candidates:** ResNet-50, ConvNeXt-T, Swin-T, DINOv2-B.
- **Evaluation Standard:** All candidates judged against the EXP-00 EfficientNetV2-S baseline.
