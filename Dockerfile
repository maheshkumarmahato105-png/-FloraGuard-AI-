# syntax=docker/dockerfile:1
# ==============================================================================
# PlantDiseaseAI — Unified Production Multi-Stage Dockerfile
# Builds React SPA frontend and serves alongside FastAPI + PyTorch backend
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build React 19 Frontend
# ------------------------------------------------------------------------------
FROM node:20-alpine AS frontend-builder

WORKDIR /build/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# Build production bundle with relative API path
ENV VITE_API_BASE_URL=""
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Production Python Backend & Web Server
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_ENV=production \
    APP_VERSION=1.0.0

WORKDIR /app

# Install minimal OS runtime dependencies for OpenCV and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels (keeps container image ~800MB instead of 4GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.2.0 \
    torchvision>=0.17.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY backend/app/requirements.txt ./backend_requirements.txt
RUN pip install --no-cache-dir -r ./backend_requirements.txt

# Copy backend, ML configs/model checkpoint, and documentation
COPY backend ./backend
COPY ml/configs ./ml/configs
COPY ml/src ./ml/src
COPY ml/experiments/checkpoints ./ml/experiments/checkpoints
COPY docs ./docs

# Copy compiled React frontend distribution from Stage 1
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# Setup non-root execution user
RUN mkdir -p backend/runtime && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch production ASGI server
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
