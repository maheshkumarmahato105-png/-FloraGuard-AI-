"""FastAPI application factory and entrypoint for PlantDiseaseAI Backend (Phase 7).

Run locally from PlantDiseaseAI root:
    uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import asynccontextmanager
import time
from typing import Optional
import uuid

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import health, history, prediction, system
from .core.config import Settings, get_settings
from .core.exceptions import register_exception_handlers
from .core.logging import configure_logging
from .database.database import DatabaseManager
from .services.disease_information_service import DiseaseInformationService
from .services.explanation_service import ExplanationService
from .services.history_service import HistoryService
from .services.image_service import ImageService
from .services.model_service import ModelService
from .services.prediction_service import PredictionService
from .services.uncertainty_service import UncertaintyService


class _InMemoryRateLimiter:
    """Lightweight sliding-window per-IP rate limiter for API protection."""

    def __init__(self, max_requests_per_minute: int) -> None:
        self.max_requests = max(1, int(max_requests_per_minute))
        self.window_sec = 60.0
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        q = self._hits[client_key]
        while q and (now - q[0]) > self.window_sec:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Construct and wire the PlantDiseaseAI FastAPI application."""
    cfg = settings or get_settings()
    logger = configure_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.ensure_runtime_dirs()
        app.state.db_manager.init_db()
        # Initialize model once on startup; never crash if checkpoint is missing
        app.state.model_service.initialize()
        logger.info(
            "PlantDiseaseAI Backend startup complete",
            extra={"event": "startup"},
        )
        yield
        app.state.db_manager.dispose()
        logger.info(
            "PlantDiseaseAI Backend shutdown complete",
            extra={"event": "shutdown"},
        )

    app = FastAPI(
        title=cfg.app_name,
        description=(
            "Research-level REST API for PlantDiseaseAI integrating the custom AgriVision-CNN "
            "architecture, Phase 2 deterministic preprocessing, uncertainty rejection, "
            "approximate Grad-CAM attention visualization, and SQLite prediction history."
        ),
        version=cfg.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Wire services cleanly onto app.state (zero circular imports, easy dependency injection in tests)
    db_manager = DatabaseManager(cfg.database_url)
    model_service = ModelService(cfg)
    image_service = ImageService(cfg)
    uncertainty_service = UncertaintyService(cfg)
    explanation_service = ExplanationService(cfg, model_service)
    disease_info_service = DiseaseInformationService()
    history_service = HistoryService(image_service=image_service, explanation_service=explanation_service)
    prediction_service = PredictionService(
        settings=cfg,
        model_service=model_service,
        image_service=image_service,
        uncertainty_service=uncertainty_service,
        explanation_service=explanation_service,
        disease_info_service=disease_info_service,
        history_service=history_service,
    )

    app.state.settings = cfg
    app.state.db_manager = db_manager
    app.state.model_service = model_service
    app.state.image_service = image_service
    app.state.uncertainty_service = uncertainty_service
    app.state.explanation_service = explanation_service
    app.state.disease_info_service = disease_info_service
    app.state.history_service = history_service
    app.state.prediction_service = prediction_service

    # Explicit CORS origins + Vercel deployment preview regex
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_origin_regex=r"^https:\/\/.*\.vercel\.app$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rate_limiter = _InMemoryRateLimiter(cfg.rate_limit_per_minute)

    @app.middleware("http")
    async def request_context_and_rate_limit_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        client_ip = request.client.host if request.client else "unknown"

        # Apply rate limit to prediction endpoint
        if request.url.path.endswith("/predict") and not rate_limiter.allow(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "request_id": request_id,
                    "status": "error",
                    "error_code": "rate_limit_exceeded",
                    "detail": "Too many prediction requests. Please wait a moment and try again.",
                },
                headers={"X-Request-ID": request_id},
            )

        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "event": "http_request",
            },
        )
        return response

    register_exception_handlers(app)

    # Register versioned API routers under /api/v1
    prefix = cfg.api_v1_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(system.router, prefix=prefix)
    app.include_router(prediction.router, prefix=prefix)
    app.include_router(history.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    async def root_status():
        return {
            "status": "online",
            "app_name": cfg.app_name,
            "version": cfg.app_version,
            "endpoints": {
                "health": f"{prefix}/health",
                "predict": f"{prefix}/predict",
                "history": f"{prefix}/history",
                "system_info": f"{prefix}/system/info",
                "swagger_docs": "/docs",
            },
            "frontend_url": "https://floraguard-ai.vercel.app",
        }

    # Mount compiled React SPA production build if present
    from .core.config import PROJECT_ROOT
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.is_dir() and (frontend_dist / "index.html").is_file():
        assets_dir = frontend_dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="spa-assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa_frontend(full_path: str):
            if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
                return JSONResponse(status_code=404, content={"detail": "Not Found"})
            candidate = frontend_dist / full_path
            if candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(frontend_dist / "index.html"))

    return app


app = create_app()
