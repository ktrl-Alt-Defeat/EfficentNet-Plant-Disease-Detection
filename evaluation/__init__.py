"""
================================================================================
Evaluation Package — Plant Disease Classification
================================================================================

Exposes:
- calculate_accuracy: Top-1 classification accuracy.
- calculate_confusion_matrix: Full N x N pairwise confusion matrix.
- calculate_per_class_metrics: Per-class Precision, Recall, F1, and Support DataFrame.
- find_most_frequent_confusions: Top pairwise confusion pairs table.
- calculate_summary_metrics: Comprehensive dataset-wide summary statistics.
- Evaluator: Modular evaluator supporting standard and deep diagnostic evaluation.
================================================================================
"""

from evaluation.metrics import (
    calculate_accuracy,
    calculate_confusion_matrix,
    calculate_per_class_metrics,
    find_most_frequent_confusions,
    calculate_summary_metrics
)
from evaluation.evaluator import Evaluator

__all__ = [
    "calculate_accuracy",
    "calculate_confusion_matrix",
    "calculate_per_class_metrics",
    "find_most_frequent_confusions",
    "calculate_summary_metrics",
    "Evaluator"
]
