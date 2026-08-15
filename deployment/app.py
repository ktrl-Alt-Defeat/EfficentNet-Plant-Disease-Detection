"""
Module: deployment/app.py
Project: Leafcare — Plant Disease Classification (FastAPI Deployment)

PURPOSE:
-------------------------------------------------------------------------
Main FastAPI application module defining HTTP routes, lifespan model caching,
CORS security middleware, and OpenAPI documentation endpoints.

WHY THIS MODULE EXISTS:
-------------------------------------------------------------------------
Serves the production HTTP interface for real-time mobile and edge plant disease
diagnosis, accepting image uploads and returning structured top-5 classifications.

LIFECYCLE & PERFORMANCE:
-------------------------------------------------------------------------
- EfficientNetV2-S model weights and class mappings are loaded ONCE at startup.
- All subsequent inference calls reuse in-memory tensors under torch.inference_mode().
- Uploaded image buffers are processed in memory and never persisted to disk.

ENDPOINTS:
-------------------------------------------------------------------------
- GET  /        : API metadata and operational status.
- GET  /health  : Health check and GPU/CPU device telemetry.
- POST /predict : Multipart image upload and disease prediction.
- GET  /docs    : Swagger interactive UI documentation.
- GET  /redoc   : ReDoc OpenAPI specification viewer.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI, File, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deployment.config import settings
from deployment.exceptions import APIError, api_error_handler, general_exception_handler
from deployment.model import EfficientNetPredictor
from deployment.preprocessing import DeploymentPreprocessor
from deployment.schemas import (
    RootResponse,
    HealthResponse,
    PredictionResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("leafcare.deployment.app")

# Initialize shared singleton instances
predictor = EfficientNetPredictor(checkpoint_path=settings.CHECKPOINT_PATH)
preprocessor = DeploymentPreprocessor(image_size=settings.IMAGE_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Loads the trained model checkpoint once before accepting incoming requests.
    """
    logger.info("=" * 60)
    logger.info("STARTING LEAFCARE FASTAPI INFERENCE SERVICE")
    logger.info("=" * 60)
    try:
        predictor.load()
        logger.info(f"Model initialization complete on device: {predictor.device.type}")
    except Exception as e:
        logger.error(f"Critical error during model startup loading: {e}", exc_info=True)
        # We allow startup so /health can report status: unhealthy
    yield
    logger.info("Shutting down Leafcare FastAPI service.")


# Instantiate FastAPI Application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# Leafcare Plant Disease Classification API 🌿

Production inference service powered by **EfficientNetV2-S** for automated
agricultural crop disease diagnosis across 38 pathology classes.

## Features
- **Deterministic Preprocessing:** Matches training and evaluation pipelines exactly.
- **High Throughput:** Real-time GPU-accelerated inference under `torch.inference_mode()`.
- **Top-5 Rankings:** Provides confidence scores and ranking distributions.
- **In-Memory Security:** Zero image retention on disk.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Register Exception Handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Metadata"],
    summary="Get API Metadata",
    description="Returns service name, architecture, class count, and running status."
)
async def get_root() -> RootResponse:
    """Returns basic API status and architecture configuration."""
    return RootResponse(
        name=settings.APP_NAME,
        model=settings.MODEL_NAME,
        classes=predictor.num_classes if predictor.is_loaded else settings.EXPECTED_CLASSES,
        version=settings.APP_VERSION,
        status="running"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health & Telemetry Check",
    description="Audits model readiness and reports compute hardware allocation (CUDA / CPU)."
)
async def get_health() -> HealthResponse:
    """Reports health check status and compute device."""
    is_healthy = predictor.is_loaded and predictor.model is not None
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=is_healthy,
        device=str(predictor.device.type),
        model=settings.MODEL_NAME,
        classes=predictor.num_classes
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or unreadable image file."},
        415: {"model": ErrorResponse, "description": "Unsupported media type / MIME format."},
        500: {"model": ErrorResponse, "description": "Inference forward-pass internal failure."},
        503: {"model": ErrorResponse, "description": "Model not loaded or unavailable."}
    },
    tags=["Inference"],
    summary="Classify Plant Disease from Leaf Image",
    description="""
Uploads an image file (`multipart/form-data`) and runs disease diagnosis.

### Workflow:
1. Validates image readability and decodes byte stream in memory.
2. Converts color space to 3-channel RGB.
3. Resizes to 224x224 and converts to normalized FloatTensor.
4. Executes EfficientNetV2-S forward pass under `torch.inference_mode()`.
5. Returns Top-1 prediction along with Top-5 candidate classes and confidence metrics.
    """
)
async def predict_disease(
    image: UploadFile = File(..., description="Crop leaf image file (JPEG, PNG, or WEBP format)")
) -> PredictionResponse:
    """
    Accepts uploaded image file, processes in-memory, and returns top-5 disease predictions.
    """
    # 1. Read binary image payload asynchronously into memory
    image_bytes = await image.read()

    # 2. Preprocess into [1, 3, 224, 224] FloatTensor
    content_type = image.content_type or ""
    tensor = preprocessor.preprocess_bytes(image_bytes, content_type=content_type)

    # 3. Execute inference and generate top-5 predictions
    prediction_result = predictor.predict(tensor, top_k=5)

    return PredictionResponse(**prediction_result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deployment.app:app", host="0.0.0.0", port=8000, reload=False)
