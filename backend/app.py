"""FastAPI application entrypoint for the AI Plant Disease Detection System.

Run locally from the project root:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIR, TRAINING_REPORTS_DIR, UPLOADS_DIR
from backend.database import init_db
from backend.routes import health, history, predict, upload
from backend.services.model_service import model_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize SQLite/PostgreSQL tables and load trained model on startup."""
    init_db()
    if not model_service.is_loaded:
        model_service.load_model()
    yield


app = FastAPI(
    title="FloraGuard AI — Plant Disease Detection API",
    description=(
        "End-to-end AI-powered Plant Disease Detection REST API using PyTorch "
        "transfer learning, OpenCV leaf preprocessing, and SQLite/PostgreSQL history."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for standalone frontend or mobile LAN development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(predict.router)
app.include_router(history.router)

# Serve uploaded leaf images and model evaluation plots
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/reports", StaticFiles(directory=str(TRAINING_REPORTS_DIR)), name="reports")

# Serve frontend static subdirectories
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
if (FRONTEND_DIR / "components").exists():
    app.mount(
        "/components",
        StaticFiles(directory=str(FRONTEND_DIR / "components")),
        name="components",
    )
if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/", include_in_schema=False)
def serve_frontend_index():
    """Serve the single-page frontend dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Plant Disease Detection API is running. Visit /docs for Swagger UI."
    }
