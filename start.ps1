# Research Connection Graph - Start Script (PowerShell)
# Starts both backend and frontend servers

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$RunDir = Join-Path $ProjectRoot ".run"
$LogDir = Join-Path $RunDir "logs"

# Create directories
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "🚀 Starting Research Connection Graph..." -ForegroundColor Cyan
Write-Host ""

# Check if Docker services are running
Write-Host "Checking Docker services..." -ForegroundColor Yellow
Set-Location $ProjectRoot
$dockerRunning = docker-compose ps 2>$null | Select-String "Up"
if (-not $dockerRunning) {
    Write-Host "⚠️  Docker services not running. Starting them..." -ForegroundColor Yellow
    docker-compose up -d
    Write-Host "⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
} else {
    Write-Host "✅ Docker services are running" -ForegroundColor Green
}

# Start backend
Write-Host ""
Write-Host "🐍 Starting backend server..." -ForegroundColor Cyan
Set-Location (Join-Path $ProjectRoot "backend")

if (-not (Test-Path ".venv")) {
    Write-Host "❌ Backend not set up. Run .\setup.sh first." -ForegroundColor Red
    exit 1
}

# Find Python in venv
$venvPython = $null
foreach ($path in @(".venv\Scripts\python.exe", ".venv\bin\python", ".venv\bin\python3")) {
    if (Test-Path $path) {
        $venvPython = $path
        break
    }
}

if (-not $venvPython) {
    Write-Host "❌ Could not find .venv Python" -ForegroundColor Red
    exit 1
}

# Check if uvicorn is installed
$uvicornCheck = & $venvPython -m uvicorn --help 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Backend dependencies not installed. uvicorn not found." -ForegroundColor Red
    Write-Host "   Please run .\setup.sh first to install dependencies." -ForegroundColor Red
    exit 1
}

# Clear Python bytecode cache
Write-Host "🧹 Clearing Python cache..." -ForegroundColor Yellow
Remove-Item -Path "app\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "app" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Start backend in background
$backendLog = Join-Path $LogDir "backend.log"
Write-Host "Backend starting at http://localhost:8000" -ForegroundColor Green
$backendProcess = Start-Process -FilePath $venvPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--reload-dir", "app", "--host", "0.0.0.0", "--port", "8000" `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError $backendLog `
    -PassThru `
    -NoNewWindow

$backendProcess.Id | Out-File (Join-Path $RunDir "backend.pid") -Encoding ASCII

# Give backend time to start
Start-Sleep -Seconds 3

# Check if backend started
if ($backendProcess.HasExited) {
    Write-Host "❌ Backend failed to start. Check logs:" -ForegroundColor Red
    Get-Content $backendLog
    exit 1
}
Write-Host "✅ Backend started (PID: $($backendProcess.Id))" -ForegroundColor Green

# Start frontend
Write-Host ""
Write-Host "📦 Starting frontend server..." -ForegroundColor Cyan
Set-Location (Join-Path $ProjectRoot "frontend")

if (-not (Test-Path "node_modules")) {
    Write-Host "❌ Frontend not set up. Run .\setup.sh first." -ForegroundColor Red
    Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

# Ensure port 5173 is free
Write-Host "🧹 Ensuring port 5173 is free..." -ForegroundColor Yellow
& (Join-Path $ProjectRoot "stop.ps1") | Out-Null
Start-Sleep -Seconds 1

# Start frontend in background
$frontendLog = Join-Path $LogDir "frontend.log"
Write-Host "Frontend starting at http://localhost:5173" -ForegroundColor Green
$frontendProcess = Start-Process -FilePath "npm" `
    -ArgumentList "run", "dev" `
    -RedirectStandardOutput $frontendLog `
    -RedirectStandardError $frontendLog `
    -PassThru `
    -NoNewWindow

$frontendProcess.Id | Out-File (Join-Path $RunDir "frontend.pid") -Encoding ASCII

# Give frontend time to start
Start-Sleep -Seconds 3

# Check if frontend started
if ($frontendProcess.HasExited) {
    Write-Host "❌ Frontend failed to start. Check logs:" -ForegroundColor Red
    Get-Content $frontendLog
    Stop-Process -Id $backendProcess.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "✅ Frontend started (PID: $($frontendProcess.Id))" -ForegroundColor Green

# Get actual frontend port from log
$frontendPort = 5173
$logContent = Get-Content $frontendLog -ErrorAction SilentlyContinue | Select-String "localhost:(\d+)" | Select-Object -First 1
if ($logContent -and $logContent.Matches.Groups.Count -gt 1) {
    $frontendPort = $logContent.Matches.Groups[1].Value
}

Write-Host ""
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "✅ All servers started successfully!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend: http://localhost:$frontendPort" -ForegroundColor White
Write-Host ""
Write-Host "  Logs: $LogDir" -ForegroundColor Gray
Write-Host ""
Write-Host "  To stop: .\stop.ps1" -ForegroundColor Yellow
Write-Host "  To view logs: Get-Content $LogDir\backend.log -Wait" -ForegroundColor Gray
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ Servers are running in the background" -ForegroundColor Green
Write-Host "   Use Task Manager or .\stop.ps1 to stop them" -ForegroundColor Gray
