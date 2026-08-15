# Explainability & Grad-CAM Analysis Report

**Project:** Plant Disease Classification — Leafcare Model (EfficientNetV2-S)  
**Target Layer:** `EfficientNetV2-S.features[-1]` (1280 channels, 7×7 feature map)  
**Status:** COMPLETE (Diagnostic only — no weights modified)  

---

## 1. Visual Attention Observations

1. **Focal Lesion Localization:** On correctly classified diseases (e.g. `Tomato Early Blight`, `Apple Scab`), Grad-CAM heatmaps focus directly on the characteristic necrotic lesions and concentric rings on the leaf lamina.
2. **Background Invariance:** The model demonstrates strong background suppression, ignoring fingers, greenhouse soil, and mounting boards.
3. **High-Confidence Misclassifications:** In ambiguous cases (e.g. `Tomato Early Blight` vs `Target Spot`), the model activates on the general lesion area but relies on subtle textural distinctions that generic ImageNet features conflate.
