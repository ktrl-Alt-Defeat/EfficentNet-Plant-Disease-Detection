"""
================================================================================
Learning Rate Scheduler Module — Plant Disease Classification
Milestone 3 — Production-Grade Training Infrastructure
================================================================================

This module encapsulates learning rate scheduler construction and lifecycle
management.

Educational Deep Dive:
----------------------
1. Why do we need Learning Rate Scheduling?
   - A constant learning rate throughout training is suboptimal:
     * Early training benefits from a larger learning rate to escape poor local minima
       and traverse flat error surfaces quickly.
     * Late training requires a smaller learning rate to settle into deep, sharp minima
       and avoid bouncing erratically around the optimal weights.

2. What is Cosine Annealing (Loshchilov & Hutter, 2016)?
   - Decays the learning rate following a cosine curve:
       lr(t) = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * t / T_max))
   - Unlike step decay (which causes abrupt drops), cosine annealing provides a smooth,
     continuous reduction in step size.
   - It maintains moderate exploration for longer before smoothly annealing to min_lr.

3. Why Learning Rate Warmup?
   - In the very first epochs, the newly initialized classification head has random weights
     that produce large gradients.
   - Warmup starts the learning rate at a fraction of initial_lr and ramps it up linearly
     over `warmup_epochs`.
   - Benefit: Prevents destabilizing the early optimization trajectory.

4. Granularity: Epoch-Level vs Batch-Level Stepping
   - This scheduler is designed for epoch-level stepping: `scheduler.step()` is called
     once per epoch after the training and validation loops complete.
================================================================================
"""

from typing import Dict, Any, Optional
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
    _LRScheduler
)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    total_epochs: Optional[int] = None
) -> Optional[_LRScheduler]:
    """
    Factory function to construct a learning rate scheduler from configuration.

    Args:
        optimizer: Configured PyTorch optimizer instance whose parameter groups will be scheduled.
        config: Full configuration dictionary containing training/scheduler settings.
        total_epochs: Total number of training epochs (overrides config if provided).

    Returns:
        Optional[_LRScheduler]: Configured PyTorch LR scheduler, or None if disabled.

    Raises:
        ValueError: If unsupported scheduler name or invalid parameters are provided.
    """
    training_cfg = config.get("training", {})
    sched_cfg = training_cfg.get("scheduler", config.get("scheduler", {}))

    sched_name = sched_cfg.get("name", "cosine").lower()

    # Determine total epochs for T_max
    if total_epochs is None:
        total_epochs = int(training_cfg.get("epochs", 30))

    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))

    if sched_name in ("none", "null", "disabled"):
        return None

    if sched_name in ("cosine", "cosine_annealing", "cosineannealinglr"):
        # If warmup is requested and total_epochs > warmup_epochs, chain LinearLR -> CosineAnnealingLR
        if warmup_epochs > 0 and total_epochs > warmup_epochs:
            # Phase 1: Warmup from (start_factor * lr) up to lr over warmup_epochs
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 1% of initial learning rate
                end_factor=1.0,     # Reach 100% of initial learning rate
                total_iters=warmup_epochs
            )

            # Phase 2: Cosine decay from lr down to min_lr over remaining epochs
            cosine_epochs = total_epochs - warmup_epochs
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=min_lr
            )

            # SequentialLR chains warmup and cosine decay seamlessly
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            return scheduler
        else:
            # Standard Cosine Annealing across the entire training duration
            return CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=min_lr
            )

    elif sched_name in ("step", "steplr"):
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    else:
        raise ValueError(
            f"Unsupported learning rate scheduler: '{sched_name}'. "
            "Supported options in Milestone 3: 'cosine', 'step', 'none'."
        )
