# PlantDiseaseAI — Live Deployment Guide (Vercel + Render / Free AI Hosting)

This guide provides step-by-step instructions for deploying:
- **Frontend:** [Vercel](https://vercel.com) (Free Global Edge CDN)
- **Backend & AI Model:** [Render](https://render.com) (Free Web Service) or [Hugging Face Spaces](https://huggingface.co/spaces) (Free 16 GB RAM AI Container)

---

## Architecture Overview

```
┌─────────────────────────────────┐       HTTPS REST + JSON       ┌─────────────────────────────────────┐
│         Vercel (Frontend)       │ ────────────────────────────> │           Render / Hugging Face     │
│   React 19 + Vite + Tailwind    │                               │     FastAPI + PyTorch AgriVision    │
│   https://plantdiseaseai.vercel │ <──────────────────────────── │   https://plantdiseaseai.onrender   │
└─────────────────────────────────┘         Prediction & CAM      └─────────────────────────────────────┘
```

---

## 1. Deploy Frontend on Vercel

### Step 1: Push Repository to GitHub
Ensure your repository is pushed to your GitHub account:
`https://github.com/maheshkumarmahato105-png/-FloraGuard-AI-Plant-Disease-Detection-System.git`

### Step 2: Import Project in Vercel
1. Log in to [vercel.com](https://vercel.com) and click **Add New > Project**.
2. Select your repository: `-FloraGuard-AI-Plant-Disease-Detection-System`.
3. In the project configuration:
   - **Framework Preset:** `Vite`
   - **Root Directory:** click **Edit** and select `frontend` (or `PlantDiseaseAI/frontend`).
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`

### Step 3: Set Environment Variables in Vercel
Under **Environment Variables**, add:
- `VITE_API_BASE_URL`: `https://plantdiseaseai-backend.onrender.com` (replace with your actual Render or Hugging Face backend URL once deployed)
- `VITE_API_TIMEOUT_MS`: `30000`

### Step 4: Click Deploy
Vercel will build and deploy the React 19 app in ~30 seconds, generating your live domain (e.g. `https://plantdiseaseai.vercel.app`).
[`vercel.json`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/PlantDiseaseAI/frontend/vercel.json) handles all SPA route redirects and caching automatically.

---

## 2. Deploy Backend on Render (Free Tier)

### Step 1: Create Web Service on Render
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New > Web Service**.
3. Connect your GitHub repository.
4. If prompted to select a Blueprint, choose [`render.yaml`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/PlantDiseaseAI/render.yaml), or enter the following manual settings:
   - **Name:** `plantdiseaseai-backend`
   - **Region:** `Oregon (US West)` or `Frankfurt (EU)`
   - **Runtime:** `Python 3`
   - **Root Directory:** `PlantDiseaseAI` (or leave empty if repo root is the project)
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r backend/app/requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1
     ```
   - **Instance Type:** `Free`

### Step 2: Configure Environment Variables
In the Render dashboard under **Environment**:
- `PYTHON_VERSION`: `3.11.8`
- `APP_ENV`: `production`
- `INFERENCE_DEVICE`: `cpu`
- `CORS_ORIGINS`: `https://*.vercel.app,http://localhost:5173`

### Step 3: Deploy
Render will build and launch your backend. Once live, test the endpoint:
`https://<your-backend-app>.onrender.com/api/v1/health`

---

## 3. Alternative: Free AI Hosting on Hugging Face Spaces (16 GB RAM)

If Render's free tier (512 MB RAM) goes to sleep or hits memory limits, **Hugging Face Spaces** is the ideal free alternative:
- **100% Free**
- **2 vCPU + 16 GB RAM** (never runs out of memory for PyTorch models)
- Permanent HTTPS REST API

### How to Deploy to Hugging Face Spaces
1. Sign up at [huggingface.co](https://huggingface.co) and click **New Space**.
2. Select:
   - **Space SDK:** `Docker` (Blank)
   - **Space hardware:** `CPU basic · 2 vCPU · 16 GB RAM · Free`
3. Push your `PlantDiseaseAI` repository to the Hugging Face Space git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/plantdiseaseai
   git push space main
   ```
4. Hugging Face automatically builds [`Dockerfile`](file:///c:/Users/Mahesh%20Kumar%20Mahato/Desktop/Plant%20Disease%20Detection/PlantDiseaseAI/Dockerfile) and serves the API at:
   `https://<your-username>-plantdiseaseai.hf.space`

---

## 4. Connecting Frontend & Backend

Once your backend is live on Render or Hugging Face:
1. Copy the backend URL (e.g. `https://plantdiseaseai-backend.onrender.com`).
2. Go to your **Vercel Dashboard > Project Settings > Environment Variables**.
3. Update `VITE_API_BASE_URL` with your live backend URL.
4. Click **Redeploy** on Vercel.

Your live AI application is now completely online and operational!
