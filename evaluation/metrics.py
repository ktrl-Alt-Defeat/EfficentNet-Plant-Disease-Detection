"""
================================================================================
Classification Metrics & Diagnostic Analysis Module
Plant Disease Classification Pipeline
================================================================================

This module provides production-grade metric computation, per-class performance
auditing, confusion matrix analysis, and diagnostic identification of failure modes
(low recall, low precision, and frequent pairwise confusions).

Educational Deep Dive:
----------------------
1. Macro vs Weighted vs Top-1 Metrics:
   - Top-1 Accuracy:
       Overall percentage of correct predictions. In imbalanced datasets, it is biased
       toward majority classes.
   - Macro Average (Unweighted):
       Computes the metric independently for each of the N classes, then takes the
       unweighted arithmetic mean:
         Macro_Metric = (1 / N) * sum(Metric_c for c in 0..N-1)
       *Why it matters*: Gives equal weight to every plant disease. A severe disease
       with only 10 images counts just as much as a common disease with 500 images.
   - Weighted Average:
       Weights each class's metric by its support (number of true instances):
         Weighted_Metric = sum(Metric_c * Support_c for c in 0..N-1) / Total_Samples
       *Why it matters*: Reflects population-level field performance.

2. Precision vs Recall in Plant Pathology:
   - Precision (Purity):
       Precision = True_Positives / (True_Positives + False_Positives)
       *Meaning*: When the model predicts "Apple Scab", how often is it actually Apple Scab?
       *Low Precision*: The model is over-predicting this disease and producing false alarms.
   - Recall (Sensitivity):
       Recall = True_Positives / (True_Positives + False_Negatives)
       *Meaning*: Out of all actual "Apple Scab" leaves, how many did the model detect?
       *Low Recall*: The model is missing actual diseased leaves (dangerous false negatives).

3. Pairwise Confusion Matrix & Semantic Confusion:
   - Row i = Ground Truth Class, Column j = Model Prediction.
   - Off-diagonal entry (i, j) reveals how many times disease i was mistaken for disease j.
   - In plant pathology, high confusion typically occurs between visually similar symptoms
     (e.g., Early Blight vs Late Blight, or Cercospora Leaf Spot vs Septoria Leaf Spot).
================================================================================
"""

from typing import Dict, List, Any, Tuple, Union, Optional
import torch
import numpy as np
import pandas as pd


def calculate_accuracy(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Calculate overall top-1 classification accuracy.

    Args:
        y_true: 1D array/tensor of true ground-truth class integer indices.
        y_pred: 1D array/tensor of predicted class integer indices.

    Returns:
        float: Accuracy score in range [0.0, 1.0].
    """
    if len(y_true) == 0:
        return 0.0

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    return float(np.sum(y_true == y_pred) / len(y_true))


def calculate_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    num_classes: int
) -> np.ndarray:
    """
    Compute full N x N confusion matrix.

    Row i = True Class, Column j = Predicted Class.
    Entry (i, j) represents the number of samples with true class i predicted as j.

    Args:
        y_true: 1D array of ground truth labels [0..num_classes-1].
        y_pred: 1D array of predicted labels [0..num_classes-1].
        num_classes: Total number of target classes.

    Returns:
        np.ndarray: Confusion matrix of shape [num_classes, num_classes] with dtype int64.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    return cm


def calculate_per_class_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: List[str],
    epsilon: float = 1e-7
) -> pd.DataFrame:
    """
    Calculate granular per-class precision, recall, F1-score, and support.

    Args:
        y_true: 1D array of ground truth class indices.
        y_pred: 1D array of predicted class indices.
        class_names: List of human-readable class strings matching index order.
        epsilon: Small numerical constant to avoid division by zero.

    Returns:
        pd.DataFrame: DataFrame containing per-class metrics:
            - 'class_idx': Integer class index (0..N-1)
            - 'class_name': Name of the plant disease class
            - 'support': Number of true test samples for this class
            - 'precision': Precision score in [0.0, 1.0]
            - 'recall': Recall score in [0.0, 1.0]
            - 'f1_score': F1-score in [0.0, 1.0]
            - 'true_positives': Correctly predicted count
            - 'false_positives': False alarms count
            - 'false_negatives': Missed instances count
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    num_classes = len(class_names)
    records = []

    for c in range(num_classes):
        cls_name = class_names[c]

        # True Positives: true == c and pred == c
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        # False Positives: true != c and pred == c (model incorrectly claimed c)
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        # False Negatives: true == c and pred != c (model missed true c)
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        # Support: total true instances of class c in test split
        support = int(np.sum(y_true == c))

        # Precision, Recall, F1
        precision = float(tp / (tp + fp + epsilon)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn + epsilon)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall + epsilon)) if (precision + recall) > 0 else 0.0

        records.append({
            "class_idx": c,
            "class_name": cls_name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        })

    df = pd.DataFrame(records)
    return df


def find_most_frequent_confusions(
    cm: np.ndarray,
    class_names: List[str],
    top_k: int = 25
) -> pd.DataFrame:
    """
    Identify and rank the most frequent class-to-class misclassifications.

    Args:
        cm: Confusion matrix array of shape [N, N].
        class_names: List of class names.
        top_k: Number of top confusion pairs to return.

    Returns:
        pd.DataFrame: Table of top confusion pairs sorted by error count descending:
            - 'true_class': Actual disease class
            - 'predicted_class': Misclassified prediction
            - 'error_count': Number of misclassified images
            - 'pct_of_true_class': Percentage of true class samples confused as this
    """
    num_classes = len(class_names)
    confusions = []

    for i in range(num_classes):
        true_total = int(np.sum(cm[i, :]))
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                count = int(cm[i, j])
                pct = (count / true_total * 100.0) if true_total > 0 else 0.0
                confusions.append({
                    "true_class_idx": i,
                    "true_class": class_names[i],
                    "predicted_class_idx": j,
                    "predicted_class": class_names[j],
                    "error_count": count,
                    "true_class_total": true_total,
                    "pct_of_true_class": round(pct, 2)
                })

    df = pd.DataFrame(confusions)
    if not df.empty:
        df = df.sort_values(by=["error_count", "pct_of_true_class"], ascending=[False, False]).reset_index(drop=True)
        return df.head(top_k)
    return pd.DataFrame(columns=[
        "true_class_idx", "true_class", "predicted_class_idx",
        "predicted_class", "error_count", "true_class_total", "pct_of_true_class"
    ])


def calculate_summary_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: List[str],
    loss: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive dataset-wide summary classification metrics.

    Args:
        y_true: 1D ground-truth labels.
        y_pred: 1D predicted labels.
        class_names: List of class names.
        loss: Optional average loss value.

    Returns:
        Dict[str, Any]: Dictionary containing top-1 accuracy, macro metrics,
                        weighted metrics, and sample counts.
    """
    df = calculate_per_class_metrics(y_true, y_pred, class_names)

    total_samples = int(df["support"].sum())
    top1_accuracy = calculate_accuracy(y_true, y_pred)

    macro_precision = float(df["precision"].mean())
    macro_recall = float(df["recall"].mean())
    macro_f1 = float(df["f1_score"].mean())

    # Weighted F1: weighted by support of each class
    if total_samples > 0:
        weighted_f1 = float(np.sum(df["f1_score"] * df["support"]) / total_samples)
        weighted_precision = float(np.sum(df["precision"] * df["support"]) / total_samples)
        weighted_recall = float(np.sum(df["recall"] * df["support"]) / total_samples)
    else:
        weighted_f1 = 0.0
        weighted_precision = 0.0
        weighted_recall = 0.0

    summary = {
        "total_samples": total_samples,
        "num_classes": len(class_names),
        "top1_accuracy": top1_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }

    if loss is not None:
        summary["loss"] = float(loss)

    return summary
