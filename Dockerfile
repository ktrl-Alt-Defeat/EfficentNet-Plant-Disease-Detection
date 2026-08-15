# ==============================================================================
# Leafcare Plant Disease Classification — Production CPU Dockerfile
# Optimized for Render.com and Cloud Container Deployments
# ==============================================================================

FROM python:3.11-slim AS runtime

# Set environment variables for performance and Python logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEVICE=cpu \
    PORT=8000

WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python production dependencies using CPU-only PyTorch index
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r /app/deployment/requirements.txt

# Copy ONLY necessary application code, model architecture, and checkpoint
COPY deployment /app/deployment
COPY models /app/models
COPY checkpoints/baseline_38class_effnetv2s.pt /app/checkpoints/baseline_38class_effnetv2s.pt

# Expose default port
EXPOSE 8000

# Start FastAPI application using dynamic $PORT injected by Render (defaulting to 8000)
CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
