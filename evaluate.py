"""
================================================================================
Diagnostic Model Evaluation Script — Plant Disease Classification
Milestone 3 / Diagnostic Evaluation Pipeline
================================================================================

This script executes a comprehensive diagnostic evaluation on trained checkpoints
(e.g., `checkpoints/best.pt`), producing:
1. Dataset-Wide Summary Metrics (Top-1 Accuracy, Macro F1, Weighted F1, Precision, Recall).
2. Granular Per-Class Performance Table (Support, Precision, Recall, F1 for all 122 classes).
3. Ranking of Top 20 and Bottom 20 disease classes by F1-Score.
4. Identification of failure modes:
   - Classes with severe false negatives (Low Recall < 20%)
   - Classes with severe false alarms (Low Precision < 20%)
5. Most frequent pairwise class confusions (which diseases are mistaken for each other).
6. Class imbalance and sample distribution analysis.
7. Publication-ready visualization plots (Confusion Matrix Heatmap, Top/Bottom F1 Bar Chart, Support vs F1).
8. Full CSV and JSON export under `outputs/metrics/`, `outputs/plots/`, and `outputs/predictions/`.

Usage:
------
1. Comprehensive Diagnostic Evaluation on Test Set:
   python evaluate.py --checkpoint checkpoints/best.pt --config config/config.yaml --split test

2. Diagnostic Evaluation on Validation Set:
   python evaluate.py --checkpoint checkpoints/best.pt --config config/config.yaml --split val
================================================================================
"""

import os
import sys
import argparse
from typing import Dict, Any
import yaml
import torch
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import create_datasets, create_dataloaders
from models.efficientnetv2s import build_model, EfficientNetV2SClassifier
from training.losses import build_loss
from evaluation.evaluator import Evaluator
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path: str, base_dir: str = PROJECT_ROOT) -> str:
    """Resolve relative path against project root base directory."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def print_table_header(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Plant Disease Classification — Diagnostic Evaluation Pipeline"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "best.pt"),
        help="Path to trained checkpoint file (e.g. checkpoints/best.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate: 'test', 'val', or 'train'"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target hardware device: 'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs"),
        help="Directory to save metric tables, plots, and prediction CSVs"
    )
    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})

    logger = setup_logger("diagnostic_evaluate")
    logger.info(f"Initializing Diagnostic Evaluation for checkpoint: {args.checkpoint}")

    # 2. Hardware Device Selection
    requested_device = args.device or runtime_cfg.get("device", "auto")
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Evaluation Hardware: {device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 3. Load Dataset & DataLoader
    train_dir = resolve_path(data_cfg["train_dir"])
    val_dir = resolve_path(data_cfg["val_dir"])
    test_dir = resolve_path(data_cfg["test_dir"])
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    logger.info(f"Loading {args.split} split dataset from: {test_dir if args.split == 'test' else val_dir}")
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=image_size
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    logger.info(f"Discovered {num_classes} plant disease classes")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    if args.split == "test":
        eval_loader = test_loader
        split_name = "Test"
    elif args.split == "val":
        eval_loader = val_loader
        split_name = "Validation"
    else:
        eval_loader = train_loader
        split_name = "Train"

    # 4. Build Model & Restore Checkpoint Weights
    model: EfficientNetV2SClassifier = build_model(
        num_classes=num_classes,
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.3)),
        freeze_backbone=False
    )
    model.to(device)

    logger.info(f"Restoring weights from checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        device=device
    )
    checkpoint_epoch = checkpoint.get("epoch", "N/A")
    logger.info(f"Checkpoint restored (Trained Epoch: {checkpoint_epoch})")

    # 5. Execute Deep Diagnostic Evaluation
    criterion = build_loss(config)
    evaluator = Evaluator(device=device)

    logger.info(f"Running full diagnostic evaluation across {len(eval_loader.dataset)} samples...")
    results = evaluator.evaluate_diagnostic(
        model=model,
        data_loader=eval_loader,
        criterion=criterion,
        class_names=class_names,
        output_dir=args.output_dir,
        split_name=args.split
    )

    summary = results["summary_metrics"]
    per_class_df: pd.DataFrame = results["per_class_df"]
    top_20_f1: pd.DataFrame = results["top_20_f1"]
    bottom_20_f1: pd.DataFrame = results["bottom_20_f1"]
    low_recall_df: pd.DataFrame = results["low_recall_df"]
    low_precision_df: pd.DataFrame = results["low_precision_df"]
    top_confusions_df: pd.DataFrame = results["top_confusions_df"]
    artifacts = results["saved_artifacts"]

    # -------------------------------------------------------------------------
    # 6. TERMINAL DIAGNOSTIC REPORT DISPLAY
    # -------------------------------------------------------------------------
    print_table_header(f"DIAGNOSTIC EVALUATION REPORT — {split_name.upper()} SET")
    print(f"Checkpoint File     : {args.checkpoint}")
    print(f"Checkpoint Epoch    : {checkpoint_epoch}")
    print(f"Total Test Samples  : {summary['total_samples']:,}")
    print(f"Number of Classes   : {summary['num_classes']}")
    print("-" * 80)
    print(f"Test Loss           : {summary['loss']:.4f}")
    print(f"Top-1 Accuracy      : {summary['top1_accuracy'] * 100.0:.2f}%")
    print(f"Macro F1-Score      : {summary['macro_f1'] * 100.0:.2f}%  (Unweighted across all 122 classes)")
    print(f"Weighted F1-Score   : {summary['weighted_f1'] * 100.0:.2f}%  (Sample-weighted by class support)")
    print(f"Macro Precision     : {summary['macro_precision'] * 100.0:.2f}%")
    print(f"Macro Recall        : {summary['macro_recall'] * 100.0:.2f}%")
    print(f"Weighted Precision  : {summary['weighted_precision'] * 100.0:.2f}%")
    print(f"Weighted Recall     : {summary['weighted_recall'] * 100.0:.2f}%")

    # Section A: Top 20 Best Performing Classes
    print_table_header("TOP 20 DISEASE CLASSES BY F1-SCORE")
    print(f"{'Rank':<5} {'Class Name':<42} {'Support':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    for idx, row in top_20_f1.iterrows():
        print(
            f"{idx+1:<5} {row['class_name']:<42} {row['support']:<8} "
            f"{row['precision']*100.0:6.2f}%   {row['recall']*100.0:6.2f}%   {row['f1_score']*100.0:6.2f}%"
        )

    # Section B: Bottom 20 Worst Performing Classes
    print_table_header("BOTTOM 20 DISEASE CLASSES BY F1-SCORE (CRITICAL BOTTLENECKS)")
    print(f"{'Rank':<5} {'Class Name':<42} {'Support':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    for idx, row in bottom_20_f1.iterrows():
        print(
            f"{idx+1:<5} {row['class_name']:<42} {row['support']:<8} "
            f"{row['precision']*100.0:6.2f}%   {row['recall']*100.0:6.2f}%   {row['f1_score']*100.0:6.2f}%"
        )

    # Section C: Severe Low Recall Warnings (< 20%)
    print_table_header(f"CLASSES WITH CRITICALLY LOW RECALL (< 20.0%) — [{len(low_recall_df)} Classes]")
    if not low_recall_df.empty:
        print(f"{'Class Name':<45} {'Support':<10} {'Recall':<12} {'False Negatives':<15}")
        print("-" * 80)
        for _, row in low_recall_df.iterrows():
            print(f"{row['class_name']:<45} {row['support']:<10} {row['recall']*100.0:6.2f}%      {row['false_negatives']:<15}")
    else:
        print("None! All classes achieved >= 20% recall.")

    # Section D: Severe Low Precision Warnings (< 20%)
    print_table_header(f"CLASSES WITH CRITICALLY LOW PRECISION (< 20.0%) — [{len(low_precision_df)} Classes]")
    if not low_precision_df.empty:
        print(f"{'Class Name':<45} {'Support':<10} {'Precision':<12} {'False Positives':<15}")
        print("-" * 80)
        for _, row in low_precision_df.iterrows():
            print(f"{row['class_name']:<45} {row['support']:<10} {row['precision']*100.0:6.2f}%      {row['false_positives']:<15}")
    else:
        print("None! All classes achieved >= 20% precision.")

    # Section E: Most Frequent Confusion Pairs
    print_table_header("MOST FREQUENT CLASS-TO-CLASS MISCLASSIFICATIONS")
    print(f"{'True Class':<35} -> {'Predicted As (Confused)':<35} {'Errors':<7} {'% of True':<10}")
    print("-" * 80)
    for _, row in top_confusions_df.head(15).iterrows():
        print(f"{row['true_class']:<35} -> {row['predicted_class']:<35} {row['error_count']:<7} {row['pct_of_true_class']:5.1f}%")

    # Section F: Class Imbalance Analysis
    print_table_header("CLASS IMBALANCE & SUPPORT DISTRIBUTION SUMMARY")
    support_series = per_class_df["support"]
    print(f"Min Support per Class    : {support_series.min()} images ({per_class_df.loc[support_series.idxmin(), 'class_name']})")
    print(f"Max Support per Class    : {support_series.max()} images ({per_class_df.loc[support_series.idxmax(), 'class_name']})")
    print(f"Median Support per Class : {support_series.median():.1f} images")
    print(f"25th - 75th Percentiles  : {support_series.quantile(0.25):.1f} - {support_series.quantile(0.75):.1f} images")
    print(f"Imbalance Ratio (Max/Min): {support_series.max() / max(support_series.min(), 1):.1f}x")

    # Section G: Exported Artifacts Summary
    print_table_header("SAVED DIAGNOSTIC ARTIFACTS")
    for key, path in artifacts.items():
        print(f"  [{key}] -> {path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
