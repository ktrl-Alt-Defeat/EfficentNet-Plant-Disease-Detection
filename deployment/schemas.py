"""
Module: deployment/schemas.py
Project: Leafcare — Plant Disease Classification (FastAPI Deployment)

PURPOSE:
-------------------------------------------------------------------------
Pydantic data models for structured request/response validation and
automatic OpenAPI documentation generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RootResponse(BaseModel):
    """API metadata returned by GET /"""
    name: str = Field(..., example="Leafcare Plant Disease Classification API")
    model: str = Field(..., example="EfficientNetV2-S")
    classes: int = Field(..., example=38)
    version: str = Field(..., example="1.0.0")
    status: str = Field(..., example="running")


class HealthResponse(BaseModel):
    """Health check status returned by GET /health"""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    device: str = Field(..., example="cuda")
    model: str = Field(..., example="EfficientNetV2-S")
    classes: int = Field(..., example=38)


class TopPrediction(BaseModel):
    """Individual class prediction confidence entry."""
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(..., alias="class", example="tomato___target_spot")
    confidence: float = Field(..., example=0.9842, description="Softmax posterior probability [0.0 - 1.0]")
    confidence_percentage: float = Field(..., example=98.42, description="Confidence represented as percentage")


class PredictionResponse(BaseModel):
    """Structured response schema returned by POST /predict"""
    predicted_class: str = Field(..., example="tomato___target_spot", description="Top-1 predicted plant pathology class")
    confidence: float = Field(..., example=0.9842, description="Top-1 prediction confidence score")
    confidence_percentage: float = Field(..., example=98.42, description="Top-1 confidence percentage")
    top_5_predictions: List[TopPrediction] = Field(..., description="Top 5 candidate classes ranked by softmax probability")
    inference_time_ms: float = Field(..., example=19.2, description="Model forward-pass latency in milliseconds")


class ErrorResponse(BaseModel):
    """Standardized clean error response schema."""
    error: str = Field(..., example="Invalid Image")
    detail: str = Field(..., example="Uploaded file could not be decoded as a valid image.")
