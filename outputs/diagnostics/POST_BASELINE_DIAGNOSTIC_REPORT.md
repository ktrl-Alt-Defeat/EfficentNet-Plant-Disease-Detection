# Master Post-Baseline Diagnostic & Error Analysis Report

**Project:** Plant Disease Classification — Leafcare Model
**Architecture:** EfficientNetV2-S (ImageNet Pretrained)
**Baseline ID:** `EXP-00`
**Status:** COMPLETE (Diagnostic Only — No Training Executed)

---

## 1. Executive Summary & Baseline Results

The baseline evaluation of **EfficientNetV2-S** with a frozen backbone and trained linear classifier head achieved strong initial performance across the 38-class plant disease test set:

- **Total Test Images:** 7,542
- **Total Classes:** 38
- **Top-1 Test Accuracy:** **93.32%**
- **Macro F1 Score:** **91.98%**
- **Weighted F1 Score:** **93.26%**
- **Macro Precision:** **91.56%**
- **Macro Recall:** **92.70%**
- **Test Loss:** **0.2212**

---

## 2. Per-Class Performance Breakdown

### Top 10 Weakest Classes (Primary Target for Improvement)
| Rank | Class Name | Support | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `tomato___early_blight` | 150 | 78.63% | 68.67% | **73.31%** |
| 2 | `tomato___target_spot` | 210 | 74.27% | 72.86% | **73.56%** |
| 3 | `tomato___tomato_mosaic_virus` | 55 | 73.53% | 90.91% | **81.30%** |
| 4 | `tomato___septoria_leaf_spot` | 264 | 84.68% | 79.55% | **82.03%** |
| 5 | `potato___healthy` | 23 | 69.70% | 100.00% | **82.14%** |
| 6 | `tomato___late_blight` | 271 | 82.77% | 81.55% | **82.16%** |
| 7 | `apple___cedar_apple_rust` | 40 | 81.40% | 87.50% | **84.34%** |
| 8 | `potato___late_blight` | 149 | 89.63% | 81.21% | **85.21%** |
| 9 | `corn_maize___cercospora_leaf_spot_gray_leaf_spot` | 77 | 88.89% | 83.12% | **85.91%** |
| 10 | `tomato___spider_mites_two_spotted_spider_mite` | 250 | 90.79% | 82.80% | **86.61%** |

### Top 10 Strongest Classes
| Rank | Class Name | Support | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `corn_maize___healthy` | 141 | 100.00% | 100.00% | **100.00%** |
| 2 | `corn_maize___common_rust` | 179 | 100.00% | 99.44% | **99.72%** |
| 3 | `strawberry___healthy` | 68 | 98.55% | 100.00% | **99.27%** |
| 4 | `orange___haunglongbing_citrus_greening` | 602 | 99.33% | 99.00% | **99.17%** |
| 5 | `strawberry___leaf_scorch` | 166 | 98.22% | 100.00% | **99.10%** |
| 6 | `raspberry___healthy` | 47 | 97.92% | 100.00% | **98.95%** |
| 7 | `squash___powdery_mildew` | 275 | 98.55% | 98.91% | **98.73%** |
| 8 | `soybean___healthy` | 637 | 97.53% | 99.37% | **98.44%** |
| 9 | `blueberry___healthy` | 223 | 97.35% | 98.65% | **98.00%** |
| 10 | `grape___leaf_blight_isariopsis_leaf_spot` | 162 | 96.97% | 98.77% | **97.86%** |

---

## 3. Confusion Pattern Analysis

The top confusion pairs across the dataset fall into distinct morphological categories:

| True Class | Predicted Class | Errors | % True Class | Error Category |
| :--- | :--- | :--- | :--- | :--- |
| `grape___esca_black_measles` | `grape___black_rot` | 18 | 8.70% | A. Same crop / Visually similar diseases |
| `tomato___spider_mites_two_spotted_spider_mite` | `tomato___target_spot` | 17 | 6.80% | A. Same crop / Visually similar diseases |
| `tomato___septoria_leaf_spot` | `tomato___bacterial_spot` | 13 | 4.92% | A. Same crop / Visually similar diseases |
| `corn_maize___cercospora_leaf_spot_gray_leaf_spot` | `corn_maize___northern_leaf_blight` | 12 | 15.58% | A. Same crop / Visually similar diseases |
| `tomato___target_spot` | `tomato___spider_mites_two_spotted_spider_mite` | 12 | 5.71% | A. Same crop / Visually similar diseases |
| `potato___late_blight` | `tomato___late_blight` | 11 | 7.38% | B. Cross-crop identical pathogen (Phytophthora infestans) |
| `tomato___early_blight` | `tomato___late_blight` | 11 | 7.33% | A. Same crop / Visually similar diseases |
| `tomato___late_blight` | `potato___late_blight` | 11 | 4.06% | B. Cross-crop identical pathogen (Phytophthora infestans) |
| `tomato___target_spot` | `tomato___healthy` | 10 | 4.76% | C. Healthy vs Diseased |
| `tomato___spider_mites_two_spotted_spider_mite` | `tomato___tomato_yellow_leaf_curl_virus` | 9 | 3.60% | A. Same crop / Visually similar diseases |

---

## 4. Class Imbalance & Tomato Subsystem Diagnosis

1. **Statistical Correlation:** Pearson r = 0.2399, Spearman rho = 0.1952. There is no strong linear dependence between class sample size and F1 score.
2. **Tomato Vulnerability:** Tomato classes account for 60% of the top 10 weakest classes despite comprising substantial training volume. The primary failure mode is **inter-disease visual lesion similarity** (Early Blight vs Target Spot vs Septoria vs Late Blight).
3. **No Severe Recall Deficits:** All 38 classes exhibit recall > 50%, confirming that aggressive class re-weighting or Focal Loss is unnecessary and could degrade overall performance.

---

## 5. High-Confidence Error & Data Quality Review Candidates

We identified **43 total review candidate images** where the baseline model made an incorrect prediction with high confidence:

- **Confidence >= 99%:** 4 images (High-priority candidates for potential label noise or extreme lesion ambiguity)
- **Confidence 95% <= p < 99%:** 17 images
- **Confidence 90% <= p < 95%:** 22 images

The full inventory is exported to:
`outputs/diagnostics/data_quality_candidates.csv`

---

## 6. Ranked Root Cause Bottlenecks

1. **Frozen Feature Extraction Limitation (Confidence: High / 90%)**
   - *Evidence:* The linear classifier is constrained to generic ImageNet representations. 99.76% of parameters were frozen. The weakest classes share microscopic spot/halo structures that generic ImageNet features cannot disentangle.
   - *Recommended Action:* Conduct `EXP-01` with partial fine-tuning of upper backbone stages at a low learning rate (`1e-5`).

2. **Fine-Grained Visual Symptom Overlap (Confidence: High / 85%)**
   - *Evidence:* Concentric rings in Early Blight vs Target Spot; identical fungal oomycete lesions in Tomato vs Potato Late Blight.
   - *Recommended Action:* Domain-adapted feature fine-tuning + spatial attention.

3. **Minor Label Ambiguity in Edge Cases (Confidence: Moderate / 60%)**
   - *Evidence:* 4 predictions with >99% model confidence opposing ground-truth annotations.
   - *Recommended Action:* Manual visual audit of candidate list in `data_quality_candidates.csv`.

---

## 7. Recommended Next Experiment: EXP-01

- **Experiment Name:** Partial EfficientNetV2-S Fine-Tuning
- **Strategy:** Freeze stages 0–4, unfreeze stages 5–6 + head, differential learning rate (backbone: 1e-5, head: 1e-3).
- **Primary Metric Target:** Macro F1 >= 93.5% (Baseline: **91.98%**).
- **Status:** **NOT EXECUTED** (Awaiting manual execution by user).

---

## 8. Verification Statement

> [!IMPORTANT]
> **NO TRAINING WAS EXECUTED.**
> All analysis was performed strictly through non-destructive post-hoc inspection of existing baseline checkpoints and evaluation artifacts.
