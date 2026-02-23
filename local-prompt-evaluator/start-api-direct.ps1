# Quick Fix - Start API Directly
# Use this if "python main.py api" doesn't work

Write-Host "Starting API with uvicorn directly..." -ForegroundColor Cyan
Write-Host ""

cd "D:\Prompt Engineering\local-prompt-evaluator"

# Run uvicorn directly
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
