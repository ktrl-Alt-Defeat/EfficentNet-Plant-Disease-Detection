"""
Module: evaluation/calibration.py
Project: Plant Disease Classification — Leafcare Model (EfficientNetV2-S)

PURPOSE:
-------------------------------------------------------------------------
Quantifies the calibration of model probability estimates and assesses
prediction confidence against empirical ground-truth correctness.

WHY THIS MODULE EXISTS:
-------------------------------------------------------------------------
In agricultural diagnostics, a model must know what it does not know.
A prediction with 99% confidence that is wrong is much more dangerous
in field deployment than a prediction with 55% confidence. Measuring
Expected Calibration Error (ECE) and Brier Score informs whether raw
softmax outputs can be trusted as true posterior probabilities.

HOW IT WORKS:
-------------------------------------------------------------------------
1. Groups predictions into M equal-width confidence bins (default M=10 or 15).
2. For each bin, computes average confidence and empirical accuracy.
3. Computes Expected Calibration Error (ECE) as weighted difference:
     ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
4. Computes Maximum Calibration Error (MCE) as worst-case gap:
     MCE = max_{m=1}^M |acc(B_m) - conf(B_m)|
5. Computes Brier Score: Mean squared error between one-hot labels and probabilities.
6. Generates Reliability Diagrams and Confidence Histograms.

INPUTS:
-------------------------------------------------------------------------
- confidences: 1D array of max softmax probabilities (float32, [0, 1])
- predictions: 1D array of predicted class indices (int64)
- targets: 1D array of ground-truth class indices (int64)
- Optional probabilities: 2D array of all class probabilities [N, C]

OUTPUTS:
-------------------------------------------------------------------------
- Dictionary of calibration metrics (ECE, MCE, Brier Score, Mean Confidences)
- Diagnostic plots (reliability diagram, confidence distribution)
- Formatted markdown calibration report

STATE MODIFICATIONS:
-------------------------------------------------------------------------
NONE. Strictly post-hoc evaluation on inference predictions. No weights modified.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
    probabilities: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive calibration metrics including ECE, MCE, and Brier Score.

    Args:
        confidences: 1D array of model confidence scores (max softmax prob)
        predictions: 1D array of predicted class indices
        targets: 1D array of ground truth class indices
        n_bins: Number of equal-width bins for reliability grouping
        probabilities: Optional 2D array of full softmax probability distributions [N, C]

    Returns:
        Dictionary containing ECE, MCE, Brier score, and bin statistics.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    is_correct = (predictions == targets)

    total_samples = len(confidences)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include lower boundary; include upper boundary on final bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_size = np.sum(in_bin)
        bin_sizes.append(int(bin_size))

        if bin_size > 0:
            bin_acc = np.mean(is_correct[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            gap = np.abs(bin_acc - bin_conf)
            
            bin_accs.append(float(bin_acc))
            bin_confs.append(float(bin_conf))
            
            ece += (bin_size / total_samples) * gap
            mce = max(mce, gap)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    # Compute Brier score
    brier_score = None
    if probabilities is not None:
        num_classes = probabilities.shape[1]
        one_hot_targets = np.zeros_like(probabilities)
        for idx, target in enumerate(targets):
            one_hot_targets[idx, target] = 1.0
        brier_score = float(np.mean(np.sum((probabilities - one_hot_targets) ** 2, axis=1)))
    else:
        # Simplified binary Brier score on max prediction
        brier_score = float(np.mean((confidences - is_correct.astype(np.float64)) ** 2))

    correct_confs = confidences[is_correct]
    incorrect_confs = confidences[~is_correct]

    metrics = {
        "total_samples": int(total_samples),
        "overall_accuracy": float(np.mean(is_correct)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_confidence_correct": float(np.mean(correct_confs)) if len(correct_confs) > 0 else 0.0,
        "mean_confidence_incorrect": float(np.mean(incorrect_confs)) if len(incorrect_confs) > 0 else 0.0,
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": float(brier_score),
        "n_bins": n_bins,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_sizes": bin_sizes,
        "high_confidence_correct_count": int(np.sum((confidences >= 0.90) & is_correct)),
        "high_confidence_incorrect_count": int(np.sum((confidences >= 0.90) & (~is_correct))),
        "low_confidence_correct_count": int(np.sum((confidences < 0.70) & is_correct)),
        "low_confidence_incorrect_count": int(np.sum((confidences < 0.70) & (~is_correct)))
    }
    return metrics


def plot_reliability_diagram(
    metrics: Dict[str, Any],
    output_path: str = "outputs/plots/reliability_diagram.png"
) -> None:
    """Generates and saves a publication-quality reliability diagram."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    n_bins = metrics["n_bins"]
    bin_confs = metrics["bin_confidences"]
    bin_accs = metrics["bin_accuracies"]
    bin_sizes = metrics["bin_sizes"]
    ece = metrics["expected_calibration_error"]
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    width = 1.0 / n_bins

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    bars = ax1.bar(
        bin_centers, bin_accs, width=width * 0.9, alpha=0.7, color="#2563EB", edgecolor="#1E40AF", label="Model Accuracy"
    )
    
    # Gap bars (red overlay where confidence exceeds accuracy)
    for c, a, w in zip(bin_centers, bin_accs, bin_confs):
        if w > a:
            ax1.bar(c, w - a, bottom=a, width=width * 0.9, color="#EF4444", alpha=0.4, edgecolor="#DC2626", label="Calibration Gap" if c == bin_centers[0] else "")

    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_ylim([0, 1.05])
    ax1.set_title(f"Reliability Diagram (ECE = {ece*100:.2f}%)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Sample counts per bin
    ax2.bar(bin_centers, bin_sizes, width=width * 0.9, color="#64748B", edgecolor="#475569")
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved reliability diagram to: {output_path}")


def plot_confidence_distribution(
    confidences: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str = "outputs/plots/confidence_distribution.png"
) -> None:
    """Plots histogram of confidence scores separated by prediction correctness."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    is_correct = (predictions == targets)
    correct_confs = confidences[is_correct]
    incorrect_confs = confidences[~is_correct]

    plt.figure(figsize=(8, 5))
    plt.hist(correct_confs, bins=30, alpha=0.6, color="#10B981", label=f"Correct (n={len(correct_confs)})", density=True)
    plt.hist(incorrect_confs, bins=30, alpha=0.6, color="#EF4444", label=f"Incorrect (n={len(incorrect_confs)})", density=True)
    plt.xlabel("Prediction Confidence", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Confidence Distribution by Prediction Correctness", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confidence distribution to: {output_path}")
