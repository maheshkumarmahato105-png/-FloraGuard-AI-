# PlantDiseaseAI — Web Server Deployment Guide

This guide details how to deploy the **PlantDiseaseAI** platform (React 19 Frontend + FastAPI + custom AgriVision-CNN Backend) to a web server.

---

## 1. Local Production Web Server (Unified Full-Stack)

The FastAPI backend now automatically compiles and serves the React 19 single-page application and its assets whenever `frontend/dist` exists.

### Build and Run
```powershell
# 1. Build the production React frontend bundle
npm --prefix frontend run build

# 2. Start the unified production web server
.\.venv\Scripts\uvicorn.exe backend.app.main:app --host 0.0.0.0 --port 8001 --workers 2
```

- **Web Dashboard:** [http://127.0.0.1:8001](http://127.0.0.1:8001)
- **API Health:** [http://127.0.0.1:8001/api/v1/health](http://127.0.0.1:8001/api/v1/health)
- **Swagger Documentation:** [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

## 2. Docker & Docker Compose (Container Deployment)

A multi-stage [`Dockerfile`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/PlantDiseaseAI/Dockerfile) builds the React frontend in Node 20, copies the bundle into a Python 3.11-slim container, installs lightweight CPU PyTorch, and launches the ASGI server.

### Run with Docker Compose
```bash
docker compose up -d --build
```

### Run with Docker CLI
```bash
docker build -t plantdiseaseai:latest .
docker run -d -p 8000:8000 --name plantdiseaseai_app plantdiseaseai:latest
```

The app will be available on `http://localhost:8000`.

---

## 3. Free Cloud Deployment (Render.com)

1. Push your repository to GitHub.
2. Sign in to [Render](https://render.com).
3. Click **New > Blueprint** and select your GitHub repository.
4. Render automatically detects [`render.yaml`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/PlantDiseaseAI/render.yaml):
   - **Build:** builds React frontend + installs PyTorch CPU & backend requirements.
   - **Start:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
   - **Health check:** `/api/v1/health`
5. Click **Apply**. Render will generate your public live URL (e.g. `https://plantdiseaseai-platform.onrender.com`).

---

## 4. Railway.app Deployment

1. Go to [Railway](https://railway.app).
2. Select **New Project > Deploy from GitHub repo**.
3. Railway automatically detects [`Dockerfile`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/PlantDiseaseAI/Dockerfile).
4. Under **Settings > Networking**, click **Generate Domain**.
5. Your platform is live with free automatic SSL.

---

## 5. Split Deployment (Vercel Frontend + Render Backend)

If you prefer deploying the React frontend on Vercel CDN and the backend on Render:

1. **Backend (Render / Railway):**
   - Deploy `PlantDiseaseAI/backend` with root directory set to `PlantDiseaseAI`.
   - Your backend will get a public URL: `https://my-api.onrender.com`.

2. **Frontend (Vercel):**
   - Import your repository in [Vercel](https://vercel.com).
   - Set **Root Directory** to `PlantDiseaseAI/frontend`.
   - Add environment variable:
     `VITE_API_BASE_URL=https://my-api.onrender.com`
   - Click **Deploy**. Vercel will deploy your React frontend to an edge CDN.
