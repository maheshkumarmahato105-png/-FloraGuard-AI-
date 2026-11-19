# FloraGuard AI — Production Deployment Guide

This guide details how to deploy the FloraGuard AI Plant Disease Detection System locally, with Docker, or across free/managed cloud platforms.

---

## 1. Quick Local Production Run

To run the unified full-stack application directly on your machine:

```powershell
# Windows PowerShell
.\run_website.ps1
```

Or manually:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 2
```

- **Live Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Health Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 2. Docker Container Deployment

FloraGuard AI includes an optimized multi-stage `Dockerfile` and `docker-compose.yml`.

### Using Docker Compose (Recommended)
```bash
docker compose up -d --build
```

### Using Docker CLI
```bash
# 1. Build image
docker build -t floraguard-ai:latest .

# 2. Run container
docker run -d \
  -p 8000:8000 \
  -e PORT=8000 \
  -v floraguard_data:/app/database \
  -v floraguard_uploads:/app/uploads \
  --name floraguard_ai \
  floraguard-ai:latest
```

---

## 3. Free & Managed Cloud Deployment

### A. Render (render.com)
1. Push your repository to GitHub.
2. Sign in to [Render](https://render.com) and click **New > Blueprint**.
3. Select your GitHub repository (`-FloraGuard-AI-`).
4. Render will automatically detect [`render.yaml`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/render.yaml) and configure:
   - **Build Command:** `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r backend/requirements.txt`
   - **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path:** `/api/health`
5. Click **Apply**. Your app will be live on `https://floraguard-ai.onrender.com`.

### B. Railway (railway.app)
1. In Railway, click **New Project > Deploy from GitHub repo**.
2. Select your repository.
3. Railway automatically detects the [`Dockerfile`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/Dockerfile) and [`Procfile`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/Procfile).
4. Add environment variable `PORT=8000`.
5. Under Settings > Domains, generate a public Railway domain.

### C. Hugging Face Spaces (Docker Space)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Docker** as the SDK.
3. Push the repository to the Space remote.
4. The container builds and serves on port 7860 (set `ENV PORT=7860`).

---

## 4. Key Configuration & Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8000` | HTTP listening port |
| `CONFIDENCE_THRESHOLD` | `0.60` | Rejection threshold for uncertain / low-confidence scans |
| `DATABASE_URL` | `sqlite:///database/plant_disease.db` | Database connection string (supports PostgreSQL via `postgresql://...`) |
| `MODEL_WEIGHTS_PATH` | `backend/models/saved/plant_disease_model.pth` | Path to trained PyTorch weights |
