"""
================================================================================
Checkpoint Management Module — Plant Disease Classification
Milestone 3 — Production-Grade Training Infrastructure
================================================================================

This module handles serialization, atomic persistence, and full-state restoration
for deep learning models, optimizers, schedulers, and training metadata.

Educational Deep Dive:
----------------------
1. Why save more than model weights?
   - Saving only `model.state_dict()` allows inference, but is INSUFFICIENT for resuming
     training seamlessly:
       * Optimizer State: AdamW maintains 1st and 2nd momentum buffers (m_t and v_t)
         for every parameter. Resetting them mid-training spikes gradients and degrades
         convergence.
       * Scheduler State: Tracks the current epoch/step and learning rate multiplier.
       * AMP Scaler State: Tracks the dynamic loss scale factor (e.g. 65536.0) and consecutive
         unskipped steps.
       * Metadata: `epoch`, `best_metric`, `class_to_idx`, and `config` guarantee
         exact reproducibility upon resume.

2. Why Atomic Checkpoint Saving?
   - If a machine runs out of disk space, encounters an Out-Of-Memory (OOM) error, or experiences
     a sudden power failure while `torch.save()` is writing to disk, the destination checkpoint file
     becomes corrupted and unreadable.
   - Solution: Save to a temporary file (`checkpoint.pt.tmp`) first, flush to disk, and then perform
     an atomic rename (`os.replace`) to `checkpoint.pt`. If saving fails mid-stream, the previous
     healthy checkpoint remains intact.
================================================================================
"""

import os
import shutil
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    filename: str = "last.pt",
    best_filename: str = "best.pt"
) -> Tuple[str, Optional[str]]:
    """
    Save training state dictionary to disk atomically.

    Args:
        state: Full state dictionary containing model, optimizer, scheduler states,
               epoch number, best validation metric, class_to_idx mapping, and config.
        is_best: If True, also updates the best checkpoint file (e.g. 'best.pt').
        checkpoint_dir: Directory where checkpoint files will be written.
        filename: Filename for the most recent checkpoint (default: 'last.pt').
        best_filename: Filename for the best performing model (default: 'best.pt').

    Returns:
        Tuple[str, Optional[str]]: Paths to saved last checkpoint and best checkpoint (if updated).
    """
    # Ensure destination directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    last_path = os.path.join(checkpoint_dir, filename)
    last_tmp = f"{last_path}.tmp"

    # Step 1: Write state dictionary to temporary file
    torch.save(state, last_tmp)

    # Step 2: Atomically rename temporary file to destination path
    # os.replace is an atomic operation on POSIX and Windows (Python 3.3+)
    if os.path.exists(last_path):
        os.replace(last_tmp, last_path)
    else:
        os.rename(last_tmp, last_path)

    best_path = None
    # Step 3: If this epoch produced a new best validation metric, update best.pt
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        best_tmp = f"{best_path}.tmp"
        torch.save(state, best_tmp)
        if os.path.exists(best_path):
            os.replace(best_tmp, best_path)
        else:
            os.rename(best_tmp, best_path)

    return last_path, best_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    scaler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """
    Load saved checkpoint and restore model, optimizer, scheduler, and scaler states.

    Args:
        checkpoint_path: Path to checkpoint file (e.g. 'checkpoints/last.pt').
        model: PyTorch model instance whose weights will be restored.
        optimizer: Optional optimizer whose momentum buffers will be restored.
        scheduler: Optional learning rate scheduler whose state will be restored.
        scaler: Optional PyTorch AMP GradScaler whose state will be restored.
        device: Hardware device to map tensors to (e.g. 'cuda' or 'cpu').

    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary containing metadata (epoch, best metric, history).

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If required keys are missing in the checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: '{checkpoint_path}'")

    # Load checkpoint tensor dictionary with device mapping
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model parameter weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise KeyError("Checkpoint missing required key: 'model_state_dict'")

    # Restore optimizer state (momentum, step counts) if optimizer is supplied
    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state (milestones, last_epoch) if scheduler is supplied
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore AMP GradScaler scale factor if scaler is supplied
    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint
