# Plant Disease Classification — PyTorch Architecture Pipeline

Production-grade PyTorch computer vision project for plant disease image classification using **EfficientNetV2-S** transfer learning.

---

## Milestone 1: Dataset & DataLoader Infrastructure

- ImageFolder dataset layout: `train/`, `val/`, `test/`
- Class consistency verification and image integrity validation.
- Minimal deterministic transforms (224x224 RGB).

```bash
python scripts/verify_dataset.py --config config/sample_config.yaml
```

---

## Milestone 2: EfficientNetV2-S Transfer Learning Model

- Pretrained ImageNet-1K EfficientNetV2-S backbone.
- Dynamic classifier head replacement (`Dropout(p=0.3)` + `Linear(1280 → num_classes)`).
- Model architecture and forward/backward smoke testing.

```bash
python scripts/verify_model.py --config config/sample_config.yaml
```

---

## Milestone 3: Production-Grade Training Infrastructure

Milestone 3 establishes the complete baseline training pipeline with frozen backbone transfer learning.

### Training Pipeline Architecture

```text
Dataset (train_loader)
        ↓
  Batch [B, 3, 224, 224]
        ↓
  EfficientNetV2-S (Frozen Backbone)
        ↓
  Classification Head (Trainable)
        ↓
  Logits [B, num_classes]
        ↓
  CrossEntropyLoss (Label Smoothing)
        ↓
  AMP GradScaler Backward
        ↓
  Gradient Clipping (max_norm = 1.0)
        ↓
  AdamW Optimizer Step
        ↓
  CosineAnnealingLR Scheduler Step
        ↓
  Validation Loop (torch.no_grad, model.eval)
        ↓
  Checkpointing (Atomic write: best.pt, last.pt)
        ↓
  Early Stopping Check (patience = 7, monitor = val_loss)
        ↓
  Logging (Console, File, CSV, JSON, TensorBoard)
```

### Key Training Features
1. **Transfer Learning Baseline**: Feature extractor is frozen (`model.features`), only the custom classifier head is trained.
2. **Optimizer**: `AdamW` with learning rate `0.001` and weight decay `0.0001` applied strictly to trainable parameters.
3. **Scheduler**: `CosineAnnealingLR` decaying smoothly to `min_lr = 1e-6`.
4. **Mixed Precision (AMP)**: `torch.autocast` + `torch.amp.GradScaler` for 2x faster GPU training and 50% lower VRAM usage.
5. **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_` (max norm 1.0) prevents exploding gradients.
6. **Atomic Checkpointing**: Saves full training state dictionary (`model`, `optimizer`, `scheduler`, `scaler`, `epoch`, `best_metric`, `history`, `config`, `class_to_idx`) via atomic temporary file renaming.
7. **Early Stopping**: Prevents overfitting by halting training when validation loss fails to improve for `patience` consecutive epochs.
8. **Logging & Experiment Tracking**: CSV, JSON, and real-time TensorBoard scalar logging.

---

## Commands Reference

### 1. Training Infrastructure Dry-Run Verification
Verifies all components (model, optimizer, scheduler, loss, AMP, clipping, checkpoint serialization, resume recovery, 1-step smoke test) without performing a full training run:

```bash
python scripts/verify_training.py --config config/sample_config.yaml
```

### 2. Manual Training Commands (User Runs These)

**Start training from scratch on full dataset:**
```bash
python train.py --config config/config.yaml
```

**Fast training run on sample dataset:**
```bash
python train.py --config config/sample_config.yaml
```

**Resume training from a saved checkpoint:**
```bash
python train.py --config config/config.yaml --resume checkpoints/last.pt
```

### 3. Checkpoint Evaluation Command
Evaluate a saved checkpoint on the validation or test split:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --config config/config.yaml --split val
```

### 4. TensorBoard Experiment Visualization
Launch TensorBoard to visualize training and validation loss/accuracy curves in real time:

```bash
tensorboard --logdir outputs/logs
```
Then open `http://localhost:6006` in your web browser.

---

## Project Directory Structure

```text
plant-disease-classification/
│
├── config/
│   ├── config.yaml
│   └── sample_config.yaml
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── efficientnetv2s.py
│   └── __init__.py
│
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── scheduler.py
│   └── __init__.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluator.py
│   └── __init__.py
│
├── explainability/
│   ├── gradcam.py
│   └── __init__.py
│
├── benchmarking/
│   ├── benchmark.py
│   └── __init__.py
│
├── utils/
│   ├── seed.py
│   ├── checkpoint.py
│   ├── logger.py
│   └── __init__.py
│
├── scripts/
│   ├── verify_dataset.py
│   ├── verify_model.py
│   └── verify_training.py
│
├── checkpoints/
│   ├── best.pt
│   └── last.pt
│
├── outputs/
│   ├── metrics/
│   │   ├── training_history.csv
│   │   └── training_history.json
│   ├── plots/
│   ├── predictions/
│   └── logs/
│
├── train.py
├── evaluate.py
├── benchmark.py
├── requirements.txt
└── README.md
```
