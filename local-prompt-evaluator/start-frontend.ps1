# Start Frontend
# Run this in Terminal 2

Write-Host "Starting Local Prompt Evaluator Frontend..." -ForegroundColor Cyan
Write-Host ""

# Navigate to frontend directory
Set-Location -Path ".\frontend"

# Check if node_modules exists
if (Test-Path ".\node_modules") {
    Write-Host "✓ Dependencies already installed" -ForegroundColor Green
} else {
    Write-Host "⚠ Installing dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host ""
Write-Host "Starting Vite dev server on http://localhost:3000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

npm run dev
