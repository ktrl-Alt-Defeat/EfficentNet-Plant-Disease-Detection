"""
Module: deployment/config.py
Project: Leafcare — Plant Disease Classification (FastAPI Deployment)

PURPOSE:
-------------------------------------------------------------------------
Centralized configuration management for the FastAPI deployment service.
Loads deployment environment variables with safe defaults.

WHY THIS MODULE EXISTS:
-------------------------------------------------------------------------
Decouples environment-specific configuration (ports, device preferences,
checkpoint paths, CORS origins) from application routing logic.
"""

import os
from typing import List


class DeploymentConfig:
    """Production deployment configuration container."""

    # API Metadata
    APP_NAME: str = "Leafcare Plant Disease Classification API"
    APP_VERSION: str = "1.0.0"
    MODEL_NAME: str = "EfficientNetV2-S"
    
    # Model Artifacts
    CHECKPOINT_PATH: str = os.getenv(
        "CHECKPOINT_PATH",
        os.path.join("checkpoints", "baseline_38class_effnetv2s.pt")
    )
    EXPECTED_CLASSES: int = 38
    IMAGE_SIZE: int = 224

    # Confidence Rejection Threshold (from Milestone 5 validation)
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

    # Security & Upload Limits
    MAX_IMAGE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB max upload
    ALLOWED_MIME_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ]

    # CORS Settings
    # NOTE: Never use allow_origins=["*"] in production with credentials/sensitive auth.
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
    ).split(",")


settings = DeploymentConfig()
