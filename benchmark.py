"""
================================================================================
Model Benchmarking Script — Plant Disease Classification
Milestone 3 / Future Milestone Preparation
================================================================================

This script measures inference latency, throughput (FPS), and peak GPU memory
for the EfficientNetV2-S classifier.

Usage:
------
python benchmark.py --config config/config.yaml --batch-size 1 --device cuda
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

from models.efficientnetv2s import build_model, EfficientNetV2SClassifier
from benchmarking.benchmark import benchmark_inference


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Benchmark EfficientNetV2-S Inference Performance")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: 1 for real-time latency)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device: 'cuda', 'cpu', or 'auto'"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of timed benchmark iterations"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    runtime_cfg = config.get("runtime", {})

    # Determine device
    requested_device = args.device or runtime_cfg.get("device", "auto")
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (3 classes placeholder or from config)
    model: EfficientNetV2SClassifier = build_model(
        num_classes=3,
        pretrained=True,
        dropout=0.3,
        freeze_backbone=True
    )

    print("\n" + "=" * 60)
    print("EFFICIENTNETV2-S INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Device               : {device.type} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Batch Size           : {args.batch_size}")
    print(f"Image Resolution     : 224 x 224 x 3")
    print(f"Iterations           : {args.iterations}")
    print("-" * 60)

    results = benchmark_inference(
        model=model,
        input_shape=(args.batch_size, 3, 224, 224),
        device=device,
        num_warmup=10,
        num_iterations=args.iterations
    )

    print(f"Avg Latency          : {results['avg_latency_ms']} ms / batch")
    print(f"Throughput           : {results['throughput_fps']} images / sec (FPS)")
    if device.type == "cuda":
        print(f"Peak GPU VRAM        : {results['peak_gpu_memory_mb']} MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
