# Class Imbalance Diagnostic Analysis — EXP-00

**Project:** Plant Disease Classification — Leafcare Model
**Total Classes:** 38
**Total Test Samples:** 7,542

---

## 1. Class Support Distribution Statistics

| Metric | Value |
| :--- | :--- |
| **Minimum Class Support** | 23 samples (`potato___healthy`) |
| **Maximum Class Support** | 657 samples (`tomato___tomato_yellow_leaf_curl_virus`) |
| **Median Class Support** | 156.0 samples |
| **Mean Class Support** | 198.5 samples |
| **Support Imbalance Ratio** | **28.57×** |

---

## 2. Support vs F1 Correlation Analysis

- **Pearson Linear Correlation (r):** `0.2399` (p-value: `1.4687e-01`)
- **Spearman Rank Correlation (rho):** `0.1952` (p-value: `2.4012e-01`)

### Key Statistical Finding:
> [!NOTE]
> The correlation between class sample size and F1 score is **weak-to-moderate (0.20)**, indicating that sample size alone does **NOT** dictate class performance.
> Several classes with low support achieve near-perfect F1 scores (e.g. `potato___healthy` has low support but high precision, `raspberry___healthy` has 99%+ F1), while high-support classes (such as `tomato___early_blight` with ~480 samples) perform among the worst in the entire dataset.

---

## 3. Comparative Tables

### Top 5 Lowest-Support Classes
| Class Name | Support | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| `potato___healthy` | 23 | 69.70% | 100.00% | **82.14%** |
| `apple___cedar_apple_rust` | 40 | 81.40% | 87.50% | **84.34%** |
| `raspberry___healthy` | 47 | 97.92% | 100.00% | **98.95%** |
| `peach___healthy` | 54 | 91.07% | 94.44% | **92.73%** |
| `tomato___tomato_mosaic_virus` | 55 | 73.53% | 90.91% | **81.30%** |

### Top 5 Weakest F1 Classes
| Class Name | Support | Precision | Recall | F1 Score | Support Rank (out of 38) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tomato___early_blight` | 150 | 78.63% | 68.67% | **73.31%** | #21 |
| `tomato___target_spot` | 210 | 74.27% | 72.86% | **73.56%** | #14 |
| `tomato___tomato_mosaic_virus` | 55 | 73.53% | 90.91% | **81.30%** | #34 |
| `tomato___septoria_leaf_spot` | 264 | 84.68% | 79.55% | **82.03%** | #8 |
| `potato___healthy` | 23 | 69.70% | 100.00% | **82.14%** | #38 |

---

## 4. Engineering Conclusion Regarding Class Imbalance

1. **No Severe Recall Collapse:** There are **0 classes** with recall below 50%, and **0 classes** with precision below 50%.
2. **Weakest Classes Have Ample Data:** `tomato___early_blight`, `tomato___target_spot`, and `tomato___septoria_leaf_spot` have high sample counts (>200 to >400 test samples), demonstrating that their lower F1 scores (73%–82%) stem from **fine-grained visual symptom confusion**, not data scarcity.
3. **Loss Function Strategy:** Blindly introducing `FocalLoss`, `WeightedRandomSampler`, or heavy inverse-class weighting is **NOT RECOMMENDED** at this stage, as it risks destabilizing gradient magnitudes and degrading high-performing classes without resolving fine-grained feature representation limitations.
