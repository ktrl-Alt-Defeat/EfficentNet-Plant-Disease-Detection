"""
================================================================================
Trainer Module — Plant Disease Classification
Milestone 3 — Production-Grade Training Infrastructure
================================================================================

This module encapsulates the complete model training, validation, early stopping,
checkpointing, AMP mixed precision, and experiment tracking lifecycle.

Educational Deep Dive:
----------------------
1. The Training Step Lifecycle:
   For every mini-batch in the training dataset:
     a) `optimizer.zero_grad(set_to_none=True)`:
        - In PyTorch, gradients accumulate by default (`param.grad += new_grad`).
        - Setting grads to None (instead of allocating zeros) saves memory and skips
          unnecessary zero-fill tensor writes.
     b) `torch.autocast(device_type=..., enabled=...)`:
        - Runs matrix multiplications and convolutions in FP16 (Half Precision) on
          Tensor Cores, while keeping numerically sensitive ops (Softmax, Reductions) in FP32.
        - Result: 2x faster execution and 50% lower VRAM footprint.
     c) `GradScaler` (Loss Scaling):
        - In FP16, very small gradient values (< 2^-24) underflow to zero.
        - GradScaler multiplies loss by a scale factor (e.g. 65536) before backpropagation,
          scaling gradients up into representable FP16 range, and unscales them before the optimizer step.
     d) `scaler.unscale_(optimizer)` + `torch.nn.utils.clip_grad_norm_`:
        - Gradients MUST be unscaled to their true physical values before computing the L2 norm.
        - Gradient clipping caps the norm at `max_norm`, preventing exploding gradients from
          destabilizing weights.
     e) `scaler.step(optimizer)` + `scaler.update()`:
        - If any gradient contained Inf/NaN, the optimizer step is skipped, and the scaler
          dynamically reduces its scale factor for the next iteration.

2. Best Model Selection & Early Stopping:
   - Tracks validation loss (`val_loss`, mode `min`).
   - If validation loss does not improve for `patience` consecutive epochs, training terminates
     early to prevent overfitting to training images.
================================================================================
"""

import os
import time
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from evaluation.evaluator import Evaluator
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger, HistoryTracker, TensorBoardLogger


class Trainer:
    """
    Production-grade training orchestrator for transfer-learning classifiers.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        class_to_idx: Optional[Dict[str, int]] = None,
        logger: Optional[Any] = None,
        experiment_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model: PyTorch classification model (e.g. EfficientNetV2SClassifier).
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            criterion: Configured loss function.
            optimizer: Configured optimizer (e.g. AdamW).
            scheduler: Optional learning rate scheduler (e.g. CosineAnnealingLR).
            config: Full configuration dictionary.
            device: Hardware device (cuda or cpu).
            class_to_idx: Class name to index mapping dictionary.
            logger: Structured logging instance.
            experiment_dir: Root output directory for this experiment run.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.class_to_idx = class_to_idx or {}
        self.num_classes = len(self.class_to_idx) if self.class_to_idx else getattr(model, "num_classes", None)

        # Output and logging directories
        self.experiment_dir = experiment_dir or "outputs"
        self.checkpoint_dir = config.get("checkpoint", {}).get("directory", "checkpoints")
        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        self.log_dir = os.path.join(self.experiment_dir, "logs")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.logger = logger or setup_logger(
            name="trainer",
            log_file=os.path.join(self.log_dir, "train.log")
        )

        # 1. Evaluator instance
        self.evaluator = Evaluator(device=self.device)

        # 2. Training configuration parsing
        training_cfg = config.get("training", {})
        self.epochs = int(training_cfg.get("epochs", 30))

        # Automatic Mixed Precision (AMP) configuration
        amp_cfg = training_cfg.get("amp", {})
        # AMP is only active when enabled in config AND running on a CUDA GPU
        self.use_amp = bool(amp_cfg.get("enabled", True)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Gradient Clipping configuration
        grad_cfg = training_cfg.get("gradient", {})
        clip_cfg = grad_cfg.get("clipping", {})
        self.clip_enabled = bool(clip_cfg.get("enabled", True))
        self.clip_max_norm = float(clip_cfg.get("max_norm", 1.0))

        # Early Stopping configuration
        es_cfg = training_cfg.get("early_stopping", {})
        self.early_stopping_enabled = bool(es_cfg.get("enabled", True))
        self.early_stopping_patience = int(es_cfg.get("patience", 7))
        self.early_stopping_monitor = es_cfg.get("monitor", "val_loss")
        self.early_stopping_mode = es_cfg.get("mode", "min")
        self.early_stopping_counter = 0

        # Metric tracking
        self.best_metric_val = float("inf") if self.early_stopping_mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0

        # History tracking & TensorBoard
        logging_cfg = config.get("logging", {})
        self.history_tracker = HistoryTracker(output_dir=self.metrics_dir)
        self.tb_logger = TensorBoardLogger(
            log_dir=self.log_dir,
            enabled=bool(logging_cfg.get("tensorboard", True))
        )

    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Execute one complete training epoch across the training DataLoader.

        Args:
            epoch: Current epoch index (1-indexed).

        Returns:
            Tuple[float, float]: (average_training_loss, training_accuracy).
        """
        # Set model to training mode (activates dropout, enables batchnorm stats tracking)
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            # Step a: Zero out gradients from previous iteration
            self.optimizer.zero_grad(set_to_none=True)

            # Step b: Forward pass with Automatic Mixed Precision (AMP)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Assert numerical sanity: fail loudly if NaN/Inf occurs
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss is not finite (loss={loss.item()}). "
                    "Check learning rate or data scaling."
                )

            # Step c: Backward pass using GradScaler when AMP is active
            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Step d: Gradient clipping (requires unscaling before clipping)
                if self.clip_enabled:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_max_norm
                    )

                # Step e: Optimizer step and dynamic scale factor update
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_max_norm
                    )
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy

    def validate_one_epoch(self, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """
        Execute validation pass across the validation DataLoader.

        Args:
            epoch: Current epoch index.

        Returns:
            Tuple[float, float, Dict[str, float]]: (val_loss, val_accuracy, metrics_dict).
        """
        return self.evaluator.evaluate(
            model=self.model,
            data_loader=self.val_loader,
            criterion=self.criterion,
            num_classes=self.num_classes
        )

    def is_better_metric(self, current: float, best: float) -> bool:
        """
        Check if the current metric is an improvement over the best metric.

        Args:
            current: Current epoch's validation metric.
            best: Previous best validation metric.

        Returns:
            bool: True if current is strictly better than best.
        """
        if self.early_stopping_mode == "min":
            return current < best
        else:
            return current > best

    def fit(self, resume_checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute full multi-epoch training pipeline with checkpointing and early stopping.

        Args:
            resume_checkpoint_path: Optional path to checkpoint file to resume from.

        Returns:
            Dict[str, Any]: Summary dictionary of training results.
        """
        start_epoch = 1

        # ---------------------------------------------------------------------
        # 1. HANDLE CHECKPOINT RESUME
        # ---------------------------------------------------------------------
        if resume_checkpoint_path:
            self.logger.info(f"Resuming training from checkpoint: {resume_checkpoint_path}")
            checkpoint = load_checkpoint(
                checkpoint_path=resume_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler if self.use_amp else None,
                device=self.device
            )
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.best_metric_val = float(checkpoint.get("best_metric", self.best_metric_val))
            self.best_epoch = int(checkpoint.get("best_epoch", 0))
            if "history" in checkpoint:
                self.history_tracker.load(checkpoint["history"])
            self.logger.info(f"Resumed successfully. Starting from Epoch {start_epoch:03d}/{self.epochs:03d}")

        # ---------------------------------------------------------------------
        # 2. PRINT PRE-TRAINING BANNER
        # ---------------------------------------------------------------------
        cuda_avail = (self.device.type == "cuda")
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A (CPU Mode)"
        initial_lr = self.optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 60)
        print("EFFICIENTNETV2-S TRAINING")
        print("=" * 60)
        print(f"Device          : {self.device.type}")
        print(f"GPU             : {gpu_name}")
        print(f"Epochs          : {self.epochs}")
        print(f"Batch Size      : {self.train_loader.batch_size}")
        print(f"Optimizer       : {self.optimizer.__class__.__name__}")
        print(f"Learning Rate   : {initial_lr}")
        print(f"Scheduler       : {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        print(f"AMP             : {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Backbone        : {'Frozen' if getattr(self.model, 'is_backbone_frozen', True) else 'Unfrozen'}")
        print(f"Classifier      : Trainable")
        print("=" * 60 + "\n")

        # ---------------------------------------------------------------------
        # 3. EPOCH TRAINING LOOP
        # ---------------------------------------------------------------------
        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start_time = time.time()

            # Current learning rate from parameter group
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Run training phase
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Run validation phase
            val_loss, val_acc, val_metrics = self.validate_one_epoch(epoch)

            # Step the learning rate scheduler once per epoch
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time

            # Determine monitored metric value for best model tracking
            monitored_val = val_loss if self.early_stopping_monitor == "val_loss" else val_acc
            is_best = self.is_better_metric(monitored_val, self.best_metric_val)

            if is_best:
                self.best_metric_val = monitored_val
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # -----------------------------------------------------------------
            # 4. LOGGING & HISTORY RECORDING
            # -----------------------------------------------------------------
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc * 100.0, 2),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc * 100.0, 2),
                "learning_rate": round(current_lr, 8),
                "epoch_time_sec": round(epoch_time, 2)
            }
            self.history_tracker.record_epoch(epoch_record)
            self.history_tracker.save()

            # Log to TensorBoard
            self.tb_logger.log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                learning_rate=current_lr
            )

            # -----------------------------------------------------------------
            # 5. ATOMIC CHECKPOINT SAVING
            # -----------------------------------------------------------------
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "epoch": epoch,
                "best_epoch": self.best_epoch,
                "best_metric": self.best_metric_val,
                "history": self.history_tracker.history,
                "config": self.config,
                "class_to_idx": self.class_to_idx
            }

            save_checkpoint(
                state=state_dict,
                is_best=is_best,
                checkpoint_dir=self.checkpoint_dir,
                filename="last.pt",
                best_filename="best.pt"
            )

            # -----------------------------------------------------------------
            # 6. TERMINAL OUTPUT DISPLAY
            # -----------------------------------------------------------------
            print(f"Epoch {epoch:03d}/{self.epochs:03d}")
            print("-" * 60)
            print(f"Train Loss      : {train_loss:.4f}")
            print(f"Train Accuracy  : {train_acc * 100.0:.2f}%")
            print(f"Val Loss        : {val_loss:.4f}")
            print(f"Val Accuracy    : {val_acc * 100.0:.2f}%")
            print(f"Learning Rate   : {current_lr:.6f}")
            print(f"Epoch Time      : {epoch_time:.2f}s")

            if is_best:
                print("\nNEW BEST MODEL")
                print(f"Saved: {os.path.join(self.checkpoint_dir, 'best.pt')}")

            # GPU Memory diagnostic (if CUDA active)
            if cuda_avail:
                alloc_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
                res_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
                peak_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
                print(f"GPU VRAM        : Alloc {alloc_mb:.1f}MB | Res {res_mb:.1f}MB | Peak {peak_mb:.1f}MB")

            print("")

            # -----------------------------------------------------------------
            # 7. EARLY STOPPING EVALUATION
            # -----------------------------------------------------------------
            if self.early_stopping_enabled and self.early_stopping_counter >= self.early_stopping_patience:
                print("=" * 60)
                print(f"EARLY STOPPING TRIGGERED (No improvement for {self.early_stopping_patience} epochs)")
                print("=" * 60)
                break

        # ---------------------------------------------------------------------
        # 8. TRAINING COMPLETE SUMMARY
        # ---------------------------------------------------------------------
        self.tb_logger.close()

        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Epoch       : {self.best_epoch}")
        print(f"Best Val Loss    : {self.best_val_loss:.4f}")
        print(f"Best Val Accuracy: {self.best_val_accuracy * 100.0:.2f}%")
        print(f"\nBest Checkpoint  : {os.path.join(self.checkpoint_dir, 'best.pt')}")
        print(f"Last Checkpoint  : {os.path.join(self.checkpoint_dir, 'last.pt')}")
        print("=" * 60 + "\n")

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "total_epochs_trained": epoch
        }
