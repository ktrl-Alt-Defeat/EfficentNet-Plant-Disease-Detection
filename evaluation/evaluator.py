"""
================================================================================
Model Evaluator & Diagnostic Reporting Module
Plant Disease Classification Pipeline
================================================================================

This module executes high-throughput inference evaluation and generates deep
pathology diagnostic reports, confusion matrices, error breakdowns, and publication-quality
visualizations.

Educational Deep Dive:
----------------------
1. Why do we need per-class and confusion diagnostics?
   - A single aggregate metric like "74.32% Top-1 Accuracy" hides critical domain-specific
     vulnerabilities:
       * Some high-frequency classes might achieve 98% accuracy while rare, high-consequence
         crop diseases score 0% recall.
       * A confusion matrix pinpoints EXACTLY which diseases the model is confusing with each
         other, guiding targeted data augmentation, resolution adjustments, or feature engineering.

2. Diagnostic Output Artifacts Generated:
   - `outputs/metrics/per_class_metrics.csv`: Full spreadsheet of Precision, Recall, F1, and Support per class.
   - `outputs/metrics/confusion_matrix.csv`: Full 122x122 pairwise prediction table.
   - `outputs/metrics/top_confusions.csv`: The 25 most frequent misclassification pairs.
   - `outputs/metrics/summary_metrics.json`: Top-1 Accuracy, Macro F1, Weighted F1, Macro Precision/Recall.
   - `outputs/predictions/test_predictions.csv`: Sample-by-sample record with ground truth, prediction, and confidence score.
   - `outputs/plots/confusion_matrix.png`: High-resolution visual heatmap.
   - `outputs/plots/top_bottom_f1.png`: Comparative bar chart of best vs worst performing disease classes.
   - `outputs/plots/class_support_vs_f1.png`: Scatter plot analyzing the correlation between sample count (imbalance) and model accuracy.
================================================================================
"""

import os
import json
from typing import Tuple, Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all servers/threads
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import (
    calculate_accuracy,
    calculate_confusion_matrix,
    calculate_per_class_metrics,
    find_most_frequent_confusions,
    calculate_summary_metrics
)


class Evaluator:
    """
    Modular evaluator for executing standard and deep diagnostic evaluation.
    """

    def __init__(self, device: torch.device):
        """
        Initialize Evaluator.

        Args:
            device: Target hardware device (cuda or cpu).
        """
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        num_classes: Optional[int] = None
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Execute standard evaluation pass across the provided DataLoader.

        Args:
            model: PyTorch classification model.
            data_loader: DataLoader providing validation or test batches.
            criterion: Loss function (e.g. CrossEntropyLoss).
            num_classes: Optional total number of classes.

        Returns:
            Tuple of (avg_loss, accuracy, metrics_dict).
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []

        for images, labels in data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        if total_samples == 0:
            return 0.0, 0.0, {"accuracy": 0.0}

        avg_loss = total_loss / total_samples
        all_preds_np = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)

        accuracy = calculate_accuracy(all_labels_np, all_preds_np)
        metrics_dict = {"accuracy": accuracy, "loss": avg_loss}

        if num_classes is not None:
            class_names = [f"Class_{i:03d}" for i in range(num_classes)]
            summary = calculate_summary_metrics(all_labels_np, all_preds_np, class_names, loss=avg_loss)
            metrics_dict.update(summary)

        return avg_loss, accuracy, metrics_dict

    @torch.no_grad()
    def evaluate_diagnostic(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        class_names: List[str],
        output_dir: str = "outputs",
        split_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Execute comprehensive diagnostic evaluation and generate all metrics, tables, and plots.

        Args:
            model: PyTorch model.
            data_loader: Evaluation DataLoader (sequential, un-shuffled).
            criterion: Loss function.
            class_names: List of class names matching class indices.
            output_dir: Base directory for output artifacts.
            split_name: Name of evaluated split (e.g. 'test' or 'val').

        Returns:
            Dict[str, Any]: Full structured diagnostic results dictionary.
        """
        model.eval()
        num_classes = len(class_names)

        metrics_dir = os.path.join(output_dir, "metrics")
        plots_dir = os.path.join(output_dir, "plots")
        preds_dir = os.path.join(output_dir, "predictions")

        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(preds_dir, exist_ok=True)

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        all_confs = []

        # ---------------------------------------------------------------------
        # 1. BATCH INFERENCE LOOP
        # ---------------------------------------------------------------------
        for images, labels in data_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Compute softmax probabilities to extract confidence score
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_confs.append(confs.detach().cpu().numpy())

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        confidences = np.concatenate(all_confs)

        # ---------------------------------------------------------------------
        # 2. METRIC CALCULATIONS
        # ---------------------------------------------------------------------
        # Summary metrics
        summary_metrics = calculate_summary_metrics(y_true, y_pred, class_names, loss=avg_loss)

        # Per-class metrics table
        per_class_df = calculate_per_class_metrics(y_true, y_pred, class_names)

        # Confusion Matrix
        cm = calculate_confusion_matrix(y_true, y_pred, num_classes)

        # Top confusions
        top_confusions_df = find_most_frequent_confusions(cm, class_names, top_k=25)

        # Identify Top 20 & Bottom 20 classes by F1
        sorted_by_f1 = per_class_df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
        top_20_f1 = sorted_by_f1.head(20)
        bottom_20_f1 = sorted_by_f1.tail(20).sort_values(by="f1_score", ascending=True).reset_index(drop=True)

        # Identify low recall classes (< 20%) and low precision classes (< 20%)
        low_recall_df = per_class_df[per_class_df["recall"] < 0.20].sort_values(by="recall").reset_index(drop=True)
        low_precision_df = per_class_df[per_class_df["precision"] < 0.20].sort_values(by="precision").reset_index(drop=True)

        # ---------------------------------------------------------------------
        # 3. SAVE CSV AND JSON ARTIFACTS
        # ---------------------------------------------------------------------
        # a) Summary metrics JSON
        summary_path = os.path.join(metrics_dir, f"{split_name}_summary_metrics.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        # b) Per-class metrics CSV & JSON
        per_class_csv = os.path.join(metrics_dir, f"{split_name}_per_class_metrics.csv")
        per_class_df.to_csv(per_class_csv, index=False)

        report_json = os.path.join(metrics_dir, f"{split_name}_classification_report.json")
        per_class_df.to_json(report_json, orient="records", indent=2)

        # c) Confusion matrix CSV
        cm_csv = os.path.join(metrics_dir, f"{split_name}_confusion_matrix.csv")
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(cm_csv)

        # d) Top confusions CSV
        confusions_csv = os.path.join(metrics_dir, f"{split_name}_top_confusions.csv")
        top_confusions_df.to_csv(confusions_csv, index=False)

        # e) Sample-by-sample predictions CSV
        sample_paths = []
        if hasattr(data_loader.dataset, "samples"):
            sample_paths = [s[0] for s in data_loader.dataset.samples]

        predictions_records = []
        for idx in range(total_samples):
            filepath = sample_paths[idx] if idx < len(sample_paths) else f"sample_{idx}"
            t_idx = int(y_true[idx])
            p_idx = int(y_pred[idx])
            predictions_records.append({
                "sample_idx": idx,
                "filepath": filepath,
                "true_class_idx": t_idx,
                "true_class": class_names[t_idx] if t_idx < num_classes else f"Class_{t_idx}",
                "predicted_class_idx": p_idx,
                "predicted_class": class_names[p_idx] if p_idx < num_classes else f"Class_{p_idx}",
                "is_correct": bool(t_idx == p_idx),
                "confidence": float(confidences[idx])
            })

        preds_csv = os.path.join(preds_dir, f"{split_name}_predictions.csv")
        pd.DataFrame(predictions_records).to_csv(preds_csv, index=False)

        # ---------------------------------------------------------------------
        # 4. GENERATE DIAGNOSTIC VISUALIZATIONS
        # ---------------------------------------------------------------------
        # Plot 1: Top 20 vs Bottom 20 F1-Score Bar Chart
        plot_top_bottom_path = os.path.join(plots_dir, f"{split_name}_top_bottom_f1.png")
        self._plot_top_bottom_f1(top_20_f1, bottom_20_f1, plot_top_bottom_path)

        # Plot 2: Class Support (Sample Count) vs F1-Score
        plot_support_f1_path = os.path.join(plots_dir, f"{split_name}_class_support_vs_f1.png")
        self._plot_support_vs_f1(per_class_df, plot_support_f1_path)

        # Plot 3: Confusion Matrix Heatmap (High Resolution)
        plot_cm_path = os.path.join(plots_dir, f"{split_name}_confusion_matrix.png")
        self._plot_confusion_matrix(cm, plot_cm_path)

        # ---------------------------------------------------------------------
        # 5. ASSEMBLE COMPLETE RESULTS DICTIONARY
        # ---------------------------------------------------------------------
        return {
            "summary_metrics": summary_metrics,
            "per_class_df": per_class_df,
            "confusion_matrix": cm,
            "top_confusions_df": top_confusions_df,
            "top_20_f1": top_20_f1,
            "bottom_20_f1": bottom_20_f1,
            "low_recall_df": low_recall_df,
            "low_precision_df": low_precision_df,
            "saved_artifacts": {
                "summary_json": summary_path,
                "per_class_csv": per_class_csv,
                "classification_report_json": report_json,
                "confusion_matrix_csv": cm_csv,
                "top_confusions_csv": confusions_csv,
                "predictions_csv": preds_csv,
                "top_bottom_f1_plot": plot_top_bottom_path,
                "support_vs_f1_plot": plot_support_f1_path,
                "confusion_matrix_plot": plot_cm_path
            }
        }

    def _plot_top_bottom_f1(
        self,
        top_df: pd.DataFrame,
        bottom_df: pd.DataFrame,
        save_path: str
    ) -> None:
        """Generate high-resolution comparative bar chart of Top 20 vs Bottom 20 classes by F1."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Top 20 Classes
        sns.barplot(
            data=top_df,
            x="f1_score",
            y="class_name",
            ax=axes[0],
            palette="Greens_r"
        )
        axes[0].set_title("Top 20 Classes by F1-Score", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("F1-Score", fontsize=12)
        axes[0].set_xlim(0, 1.05)
        axes[0].grid(axis="x", linestyle="--", alpha=0.6)

        # Add value labels
        for idx, row in top_df.iterrows():
            axes[0].text(row["f1_score"] + 0.01, idx, f"{row['f1_score']:.2f} (n={row['support']})", va="center", fontsize=9)

        # Bottom 20 Classes
        sns.barplot(
            data=bottom_df,
            x="f1_score",
            y="class_name",
            ax=axes[1],
            palette="Reds_r"
        )
        axes[1].set_title("Bottom 20 Classes by F1-Score", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("F1-Score", fontsize=12)
        axes[1].set_xlim(0, 1.05)
        axes[1].grid(axis="x", linestyle="--", alpha=0.6)

        for idx, row in bottom_df.iterrows():
            axes[1].text(row["f1_score"] + 0.01, idx, f"{row['f1_score']:.2f} (n={row['support']})", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_support_vs_f1(
        self,
        df: pd.DataFrame,
        save_path: str
    ) -> None:
        """Generate scatter plot analyzing correlation between class support (sample count) and F1."""
        fig, ax = plt.subplots(figsize=(12, 7))

        sns.scatterplot(
            data=df,
            x="support",
            y="f1_score",
            hue="recall",
            palette="viridis",
            size="support",
            sizes=(40, 250),
            alpha=0.85,
            ax=ax
        )

        ax.set_title("Class Imbalance Analysis: Test Sample Count vs F1-Score", fontsize=14, fontweight="bold")
        ax.set_xlabel("Test Class Support (Number of Test Images)", fontsize=12)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Median reference line
        median_support = df["support"].median()
        ax.axvline(median_support, color="gray", linestyle=":", label=f"Median Support ({median_support:.0f})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str
    ) -> None:
        """Generate high-resolution confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(16, 14))

        # Log scale visualization for better contrast across imbalanced classes
        cm_log = np.log1p(cm)

        sns.heatmap(
            cm_log,
            cmap="Blues",
            cbar_kws={"label": "log(1 + Counts)"},
            xticklabels=False,
            yticklabels=False,
            ax=ax
        )

        ax.set_title(f"Full {cm.shape[0]}x{cm.shape[0]} Confusion Matrix (Log-Scaled Counts)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Class Index (0..121)", fontsize=12)
        ax.set_ylabel("True Class Index (0..121)", fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
