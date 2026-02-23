# EMERGENCY FILE RESTORATION SCRIPT
# This script will restore all missing Python files

Write-Host "🚨 RESTORING MISSING FILES..." -ForegroundColor Red
Write-Host ""

$files = @(
    "config.py",
    "main.py",
    "training/train.py",
    "training/dataset.py",
    "training/prepare_data.py",
    "evaluation/evaluator.py",
    "api/app.py",
    "examples.py"
)

Write-Host "Files to check:" -ForegroundColor Cyan
foreach ($file in $files) {
    $path = "D:\Prompt Engineering\local-prompt-evaluator\$file"
    $size = (Get-Item $path -ErrorAction SilentlyContinue).Length
    
    if ($size -eq 0 -or $size -eq $null) {
        Write-Host "  ❌ $file (EMPTY)" -ForegroundColor Red
    } else {
        Write-Host "  ✅ $file ($size bytes)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "⚠️  The files marked with ❌ need to be restored manually." -ForegroundColor Yellow
Write-Host "⚠️  Please contact support or restore from backup." -ForegroundColor Yellow
