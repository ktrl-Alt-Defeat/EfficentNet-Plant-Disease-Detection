# Model Export & Deployment Validation Report

**Project:** Plant Disease Classification — Leafcare Model (EfficientNetV2-S)
**Baseline ID:** `EXP-00`

---

## 1. Export Format Parity Verification

| Export Format | Status | File Size | Top-1 Prediction Match | Max Abs Difference | Mean Abs Difference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TorchScript (.ts)** | `PASSED` | 79.0 MB | `True` | `0.00e+00` | `0.00e+00` |
| **ONNX (.onnx)** | `NOT AVAILABLE (onnx package not installed)` | N/A MB | `False` | `None` | `None` |

---

## 2. Deployment Recommendation

1. **TorchScript:** Passed with strict numerical parity (Max absolute diff < 1e-6). Ready for immediate C++ / libtorch and Python high-performance deployment.
2. **ONNX Status:** NOT AVAILABLE (onnx package not installed).
