"""
Module: deployment/exceptions.py
Project: Leafcare — Plant Disease Classification (FastAPI Deployment)

PURPOSE:
-------------------------------------------------------------------------
Custom application exception classes and centralized HTTP error handlers
to guarantee clean JSON responses without leaking internal stack traces.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse


class APIError(Exception):
    """Base API Exception with HTTP status code and message."""
    def __init__(self, message: str, detail: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


class InvalidImageError(APIError):
    """Raised when an uploaded file is not a valid or readable image."""
    def __init__(self, detail: str = "Uploaded file could not be decoded as an image."):
        super().__init__(
            message="Invalid Image",
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST
        )


class UnsupportedMediaFormatError(APIError):
    """Raised when an uploaded file format/MIME type is unsupported."""
    def __init__(self, detail: str = "Unsupported media type. Supported formats: JPEG, PNG, WEBP."):
        super().__init__(
            message="Unsupported Media Format",
            detail=detail,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )


class ModelNotReadyError(APIError):
    """Raised when model weights or artifacts are unavailable."""
    def __init__(self, detail: str = "Inference model is not loaded or is unavailable."):
        super().__init__(
            message="Service Unavailable",
            detail=detail,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class InferenceExecutionError(APIError):
    """Raised when an unexpected error occurs during tensor forward pass."""
    def __init__(self, detail: str = "An internal error occurred during model inference."):
        super().__init__(
            message="Inference Failure",
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Centralized handler for all custom APIError subclasses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "detail": exc.detail
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler to prevent raw traceback leakage on unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred while processing the request."
        }
    )
