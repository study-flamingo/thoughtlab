# Research Connection Graph - Restart Script (PowerShell)
# Stops and then starts backend and frontend servers

$ProjectRoot = $PSScriptRoot

Write-Host "🔄 Restarting Research Connection Graph..." -ForegroundColor Cyan
Write-Host ""

# Stop servers first
& (Join-Path $ProjectRoot "stop.ps1")

Write-Host ""
Write-Host "🧹 Clearing Python bytecode cache..." -ForegroundColor Yellow
Set-Location (Join-Path $ProjectRoot "backend")
Remove-Item -Path "app\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "app" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Set-Location $ProjectRoot

Write-Host ""
Write-Host "⏳ Waiting for ports to be released..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Double-check that port 5173 is free
Write-Host "🔍 Verifying port 5173 is available..." -ForegroundColor Yellow
$port5173 = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($port5173) {
    Write-Host "⚠️  Port 5173 still in use. Forcing cleanup..." -ForegroundColor Yellow
    & (Join-Path $ProjectRoot "stop.ps1") | Out-Null
    Start-Sleep -Seconds 2
}

Write-Host ""

# Start servers
& (Join-Path $ProjectRoot "start.ps1") @args
