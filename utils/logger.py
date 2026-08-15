"""
================================================================================
Logging & Experiment Tracking Module — Plant Disease Classification
Milestone 3 — Production-Grade Training Infrastructure
================================================================================

This module encapsulates structured console logging, file logging, CSV/JSON
training history serialization, and TensorBoard experiment tracking.

Educational Deep Dive:
----------------------
1. Structured Console & File Logging:
   - Separates informative terminal output from persistent audit logs on disk.
   - Outputs timestamps, log levels (INFO, WARNING, ERROR), and module identifiers.

2. Training History Persistence (CSV & JSON):
   - Preserves epoch-by-epoch loss, accuracy, and learning rate data for offline
     plotting, analysis, and metric auditing.
   - CSV format enables direct import into pandas, Excel, and scientific plotting tools.
   - JSON format provides structured schema for programmatic pipeline integrations.

3. TensorBoard Experiment Tracking:
   - PyTorch integrates with TensorBoard via `torch.utils.tensorboard.SummaryWriter`.
   - Logs scalar curves (train/loss, val/loss, train/acc, val/acc, learning_rate)
     in real-time during training.
   - Allows interactive visual inspection of overfitting, convergence rates,
     and learning rate decay dynamics in a web browser.
================================================================================
"""

import os
import sys
import json
import csv
import logging
from typing import Dict, Any, List, Optional, Tuple


def setup_logger(
    name: str = "leafcare",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a structured logger writing to console and optional log file.

    Args:
        name: Logger name identifier.
        log_file: Optional filepath for saving persistent log output (e.g. 'outputs/logs/train.log').
        level: Logging level threshold (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        # Formatter for structured log messages
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler (if log_file specified)
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class HistoryTracker:
    """
    Tracks epoch-level metrics and serializes training history to CSV and JSON files.
    """

    def __init__(self, output_dir: str = "outputs/metrics"):
        """
        Initialize HistoryTracker.

        Args:
            output_dir: Directory where CSV and JSON metrics files will be saved.
        """
        self.output_dir = output_dir
        self.history: List[Dict[str, Any]] = []

    def record_epoch(self, epoch_data: Dict[str, Any]) -> None:
        """
        Append metric records for a completed epoch.

        Args:
            epoch_data: Dictionary containing epoch metrics (epoch, train_loss, val_loss, etc.).
        """
        self.history.append(epoch_data)

    def save(
        self,
        csv_filename: str = "training_history.csv",
        json_filename: str = "training_history.json"
    ) -> Tuple[str, str]:
        """
        Save accumulated history to CSV and JSON files.

        Args:
            csv_filename: Name of the CSV output file.
            json_filename: Name of the JSON output file.

        Returns:
            Tuple[str, str]: Paths to saved CSV and JSON files.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, csv_filename)
        json_path = os.path.join(self.output_dir, json_filename)

        # 1. Save JSON representation
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        # 2. Save CSV representation
        if self.history:
            fieldnames = list(self.history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.history:
                    writer.writerow(row)

        return csv_path, json_path

    def load(self, history_list: List[Dict[str, Any]]) -> None:
        """
        Restore history records when resuming from a saved checkpoint.

        Args:
            history_list: List of historical epoch metric dictionaries.
        """
        self.history = list(history_list)


class TensorBoardLogger:
    """
    Wrapper around PyTorch TensorBoard SummaryWriter with graceful fallback
    if tensorboard package is not installed.
    """

    def __init__(self, log_dir: str = "outputs/logs", enabled: bool = True):
        """
        Initialize TensorBoardLogger.

        Args:
            log_dir: Directory where TensorBoard event files will be written.
            enabled: If False, disables all TensorBoard logging operations.
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.writer = None

        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception as e:
                # Gracefully disable if tensorboard dependencies are missing
                self.enabled = False
                self.writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to TensorBoard.

        Args:
            tag: Metric tag name (e.g. 'train/loss', 'val/accuracy', 'learning_rate').
            value: Float value of the metric.
            step: Global step or epoch index.
        """
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        learning_rate: float
    ) -> None:
        """
        Log standard training and validation metrics for an epoch.

        Args:
            epoch: Epoch index.
            train_loss: Average training loss.
            train_acc: Training accuracy.
            val_loss: Average validation loss.
            val_acc: Validation accuracy.
            learning_rate: Current learning rate.
        """
        if self.enabled and self.writer is not None:
            self.log_scalar("train/loss", train_loss, epoch)
            self.log_scalar("train/accuracy", train_acc, epoch)
            self.log_scalar("val/loss", val_loss, epoch)
            self.log_scalar("val/accuracy", val_acc, epoch)
            self.log_scalar("learning_rate", learning_rate, epoch)

    def close(self) -> None:
        """Flush and close the TensorBoard SummaryWriter."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
