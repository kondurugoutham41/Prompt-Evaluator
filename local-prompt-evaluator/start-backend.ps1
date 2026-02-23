# Start Backend API
# Run this in Terminal 1

Write-Host "Starting Local Prompt Evaluator API..." -ForegroundColor Cyan
Write-Host ""

# Check if in virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠ No virtual environment detected. Activating .venv..." -ForegroundColor Yellow
    & ".\.venv\Scripts\Activate.ps1"
}

Write-Host ""
Write-Host "Starting FastAPI server on http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

python main.py api
