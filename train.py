"""
================================================================================
Training Orchestration Script — Plant Disease Classification
Milestone 3 — Production-Grade Training Infrastructure
================================================================================

This script orchestrates model training for plant disease classification.
It coordinates configuration loading, deterministic seeding, hardware device selection,
dataset loading, model instantiation, optimizer/scheduler setup, and launches
the training lifecycle via the modular Trainer class.

Usage:
------
1. Start Training from Scratch:
   python train.py --config config/config.yaml

2. Fast Verification Run on Sample Dataset:
   python train.py --config config/sample_config.yaml

3. Resume Interrupted Training from Checkpoint:
   python train.py --config config/config.yaml --resume checkpoints/last.pt

4. Force Device Selection:
   python train.py --config config/config.yaml --device cuda
================================================================================
"""

import os
import sys
import argparse
from typing import Dict, Any
import yaml
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.seed import seed_everything
from utils.logger import setup_logger
from data.dataset import create_datasets, create_dataloaders
from models.efficientnetv2s import build_model, EfficientNetV2SClassifier
from training.losses import build_loss
from training.scheduler import build_scheduler
from training.trainer import Trainer


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file with path existence validation.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file is not found.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path: str, base_dir: str = PROJECT_ROOT) -> str:
    """Resolve relative paths against the project root directory."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Construct the optimizer for training.

    Educational Note:
    - Only parameters with `requires_grad=True` are passed to the optimizer.
    - In Milestone 3, the EfficientNetV2-S feature extractor is frozen. Passing
      frozen parameters to AdamW would allocate unnecessary momentum buffers
      and waste GPU memory.

    Args:
        model: PyTorch model with trainable classification head.
        config: Configuration dictionary.

    Returns:
        torch.optim.Optimizer: Configured AdamW optimizer.
    """
    training_cfg = config.get("training", {})
    opt_cfg = training_cfg.get("optimizer", config.get("optimizer", {}))

    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("learning_rate", opt_cfg.get("classifier_lr", 1e-3)))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))

    # Filter for trainable parameters only (classifier head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found in model. Ensure classifier is unfrozen.")

    if opt_name in ("adamw", "adam_w"):
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    elif opt_name in ("adam",):
        return torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    elif opt_name in ("sgd",):
        momentum = float(opt_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: '{opt_name}'. Supported: 'adamw', 'adam', 'sgd'.")


def main():
    """
    Main training execution function.
    """
    parser = argparse.ArgumentParser(
        description="Plant Disease Classification — EfficientNetV2-S Training Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g. checkpoints/last.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override hardware device: 'cuda', 'cpu', or 'auto'"
    )
    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config)
    project_cfg = config.get("project", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})

    # 2. Setup Logging
    logger = setup_logger(
        name="plant_disease_train",
        log_file=os.path.join(PROJECT_ROOT, "outputs", "logs", "train.log")
    )
    logger.info("Initializing Plant Disease Classification Training Pipeline")
    logger.info(f"Loaded configuration from: {args.config}")

    # 3. Set Deterministic Random Seeds for Reproducibility
    seed = int(project_cfg.get("seed", 42))
    seed_everything(seed)
    logger.info(f"Global random seed set to: {seed}")

    # 4. Determine Hardware Device
    requested_device = args.device or runtime_cfg.get("device", "auto")
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA was requested but is unavailable. Falling back safely to CPU.")
            device = torch.device("cpu")
    else:  # "auto"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Selected device: {device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 5. Load Dataset and Create DataLoaders
    train_dir = resolve_path(data_cfg["train_dir"])
    val_dir = resolve_path(data_cfg["val_dir"])
    test_dir = resolve_path(data_cfg["test_dir"])
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))

    logger.info("Loading datasets from disk...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=image_size
    )

    # Dynamically extract number of classes from dataset
    num_classes = len(train_dataset.classes)
    class_to_idx = train_dataset.class_to_idx
    logger.info(f"Dynamically discovered {num_classes} plant disease classes from dataset")

    # Create DataLoaders
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    logger.info(f"DataLoaders initialized: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # 6. Build EfficientNetV2-S Model (Frozen Backbone + Trainable Head)
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.3))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))

    model: EfficientNetV2SClassifier = build_model(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    model.to(device)

    logger.info(
        f"Model built: EfficientNetV2-S | Total Params: {model.total_parameters():,} | "
        f"Trainable: {model.trainable_parameters():,} | Frozen: {model.frozen_parameters():,}"
    )

    # 7. Build Loss Criterion, Optimizer, and LR Scheduler
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    epochs = int(config.get("training", {}).get("epochs", 30))
    scheduler = build_scheduler(optimizer, config, total_epochs=epochs)

    # 8. Initialize Trainer and Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        class_to_idx=class_to_idx,
        logger=logger,
        experiment_dir=os.path.join(PROJECT_ROOT, "outputs")
    )

    # Execute training (fit will resume if args.resume is supplied)
    results = trainer.fit(resume_checkpoint_path=args.resume)
    logger.info(f"Training completed successfully: Best Val Loss = {results['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
