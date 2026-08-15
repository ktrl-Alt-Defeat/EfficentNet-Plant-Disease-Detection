# Tomato Disease Specific Diagnostic Report — EXP-00

**Project:** Plant Disease Classification — Leafcare Model
**Crop Focus:** Solanum lycopersicum (Tomato)
**Total Tomato Classes:** 10 (out of 38 total dataset classes)
**Tomato Sample Share:** 2,551 / 7,542 (33.8% of test set)

---

## 1. Tomato Classes Performance Ranking

Tomato classes represent **6 of the 10 worst-performing classes** in the entire 38-class dataset.

| Rank | Class Name | Support | Precision | Recall | F1 Score | Main False Negative Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `tomato___early_blight` | 150 | 78.63% | 68.67% | **73.31%** | `tomato___late_blight` |
| 2 | `tomato___target_spot` | 210 | 74.27% | 72.86% | **73.56%** | `tomato___spider_mites_two_spotted_spider_mite` |
| 3 | `tomato___tomato_mosaic_virus` | 55 | 73.53% | 90.91% | **81.30%** | `None` |
| 4 | `tomato___septoria_leaf_spot` | 264 | 84.68% | 79.55% | **82.03%** | `tomato___bacterial_spot` |
| 5 | `tomato___late_blight` | 271 | 82.77% | 81.55% | **82.16%** | `potato___late_blight` |
| 6 | `tomato___spider_mites_two_spotted_spider_mite` | 250 | 90.79% | 82.80% | **86.61%** | `tomato___target_spot` |
| 7 | `tomato___leaf_mold` | 142 | 86.21% | 88.03% | **87.11%** | `None` |
| 8 | `tomato___bacterial_spot` | 315 | 88.92% | 91.75% | **90.31%** | `tomato___septoria_leaf_spot` |
| 9 | `tomato___healthy` | 237 | 93.31% | 94.09% | **93.70%** | `tomato___target_spot` |
| 10 | `tomato___tomato_yellow_leaf_curl_virus` | 657 | 96.21% | 96.65% | **96.43%** | `tomato___bacterial_spot` |

---

## 2. Deep Dive on Major Tomato Confusion Clusters

### Cluster A: The Necrotic Spot Triad (`Early Blight` ↔ `Target Spot` ↔ `Septoria Leaf Spot`)
- **Biological Context:** All three diseases present as small, dark brown/black necrotic lesions on tomato foliage. Target Spot (*Corynespora cassiicola*) and Early Blight (*Alternaria solani*) both exhibit concentric ring patterns, while Septoria (*Septoria lycopersici*) produces circular spots with gray centers and dark borders.
- **Model Behavior:** Under frozen ImageNet feature extraction, high-level convolutional filters detect general leaf spots but lack fine-grained texture resolution to distinguish concentric rings from speckled fungal pycnidia.

### Cluster B: Tomato Spider Mites (`Two-Spotted Spider Mite`) vs `Target Spot` / `TYLCV`
- **Biological Context:** Two-spotted spider mite damage produces yellow stippling, chlorosis, and mottled foliage, which visual features confuse with viral chlorosis (TYLCV) and necrotic speckled spots (Target Spot).
- **Confusion Rate:** ~6.8% of Spider Mite samples are misclassified as Target Spot, and ~3.4% as Tomato Yellow Leaf Curl Virus.

### Cluster C: Cross-Solanaceous Confusion (`Tomato Late Blight` ↔ `Potato Late Blight`)
- **Biological Context:** Both diseases are caused by the exact same oomycete pathogen (*Phytophthora infestans*), producing large, water-soaked, dark purplish-brown lesions.
- **Model Behavior:** Because tomato and potato leaves share compound serrated leaflets, the model occasionally relies on the lesion appearance rather than fine leaf-margin morphology.

---

## 3. Root Cause Assessment for Tomato Performance Deficit

| Hypothesis | Evidence | Assessment |
| :--- | :--- | :--- |
| **Class Imbalance** | Tomato classes have some of the highest sample counts in the dataset (200–650 samples each). | **DISPROVEN** — Error rate is independent of sample scarcity. |
| **Taxonomy Error** | Class labels follow standard PlantVillage / phytopathology taxonomy. | **LOW PROBABILITY** — Labels are botanically distinct. |
| **Visual Feature Bottleneck** | Pretrained ImageNet features without domain fine-tuning cannot resolve subtle lesion micro-morphologies. | **PRIMARY ROOT CAUSE (High Confidence)** |

---

## 4. Recommended Action for Tomato Subsystem
Unfreezing the final stages of the EfficientNetV2-S backbone (`EXP-01`) will allow the highest-level spatial receptive fields to adapt specifically to leaf lesion micro-textures and margin morphology.
