# syntax=docker/dockerfile:1
# ==============================================================================
# FloraGuard AI — Production Dockerfile
# Optimized multi-stage build with CPU PyTorch, FastAPI, and Static Web Frontend
# ==============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_ENV=production

WORKDIR /app

# Install minimal OS dependencies for OpenCV and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels first (keeps Docker image lightweight ~700MB vs 4GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.5.0 \
    torchvision>=0.20.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install backend dependencies
COPY backend/requirements.txt ./backend_requirements.txt
RUN pip install --no-cache-dir -r ./backend_requirements.txt

# Copy application source code and pre-trained model artifacts
COPY backend ./backend
COPY frontend ./frontend
COPY training/reports ./training/reports
COPY database ./database
COPY uploads ./uploads

# Create directories and set permissions for non-root user
RUN mkdir -p uploads database backend/models/saved training/reports && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production server entrypoint
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
