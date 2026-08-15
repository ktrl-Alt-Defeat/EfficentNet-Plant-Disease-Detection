"""
================================================================================
Reproducibility Utilities Module
Plant Disease Classification Pipeline
================================================================================

This module establishes deterministic random seeding across all random number
generators in the environment (Python, NumPy, PyTorch CPU, and PyTorch CUDA).

Seeding Strategy:
-----------------
1. Python Random: Controls standard library random choices and shuffles.
2. NumPy: Controls synthetic array generation and preprocessing seeds.
3. PyTorch CPU: Controls tensor weight initializations and CPU DataLoader batching.
4. PyTorch CUDA: Controls CUDA kernel random states on all available GPUs.
5. cuDNN Determinism: Disables non-deterministic benchmarking heuristics
   (`torch.backends.cudnn.deterministic = True`, `benchmark = False`).
================================================================================
"""

import random
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set seeds across Python random, NumPy, PyTorch CPU, and PyTorch CUDA.
    
    Args:
        seed: Random integer seed value (default: 42).
    """
    # 1. Seed Python standard library random generator
    random.seed(seed)

    # 2. Seed NumPy numerical computing random generator
    np.random.seed(seed)

    # 3. Seed PyTorch CPU operations
    torch.manual_seed(seed)
    
    # 4. Seed PyTorch CUDA operations (if CUDA GPU is present)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic cuDNN algorithms are selected
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
