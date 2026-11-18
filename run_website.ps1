# FloraGuard AI — One-Command Local Website Launcher
# Usage: .\run_website.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

$uvicorn = Join-Path $PSScriptRoot ".venv\Scripts\uvicorn.exe"
if (-not (Test-Path $uvicorn)) {
    Write-Error "Virtual environment not found at .venv\. See README.md for setup instructions."
}

Write-Host "Starting FloraGuard AI Plant Disease Detection Website..." -ForegroundColor Green
Write-Host "  Web Dashboard: http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "  Swagger Docs:  http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray

& $uvicorn backend.app:app --host 127.0.0.1 --port 8000
