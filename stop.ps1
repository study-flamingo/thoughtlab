# Research Connection Graph - Stop Script (PowerShell)
# Stops backend and frontend servers

Write-Host "🛑 Stopping Research Connection Graph servers..." -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = $PSScriptRoot
$RunDir = Join-Path $ProjectRoot ".run"

# Function to kill processes on a specific port
function Kill-Port {
    param(
        [int]$Port,
        [string]$Name
    )

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue

    if ($connections) {
        foreach ($conn in $connections) {
            $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "Stopping $Name (PID: $($process.Id)) on port $Port..." -ForegroundColor Yellow
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
        Write-Host "✅ Stopped $Name on port $Port" -ForegroundColor Green
        return $true
    }

    Write-Host "ℹ️  No $Name process found on port $Port" -ForegroundColor Gray
    return $false
}

# Function to kill all uvicorn/python processes
function Kill-UvicornProcesses {
    Write-Host "Searching for all uvicorn/python processes..." -ForegroundColor Yellow

    $found = $false

    # Find all python processes running uvicorn in the thoughtlab backend directory
    $pythonProcesses = Get-WmiObject Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and
        ($_.CommandLine -like '*uvicorn*' -or $_.CommandLine -like '*thoughtlab*backend*')
    }

    foreach ($proc in $pythonProcesses) {
        Write-Host "  Killing backend python process (PID: $($proc.ProcessId))..." -ForegroundColor Yellow
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        $found = $true
    }

    # Kill from PID file
    $pidFile = Join-Path $RunDir "backend.pid"
    if (Test-Path $pidFile) {
        $pid = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($pid) {
            Write-Host "  Killing backend from PID file (PID: $pid)..." -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            $found = $true
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }

    if ($found) {
        Write-Host "✅ Killed uvicorn processes" -ForegroundColor Green
        return $true
    }

    return $false
}

# Function to kill all vite/node processes
function Kill-ViteProcesses {
    Write-Host "Searching for all vite/frontend processes..." -ForegroundColor Yellow

    $found = $false

    # Find all node processes running vite in the thoughtlab frontend directory
    $nodeProcesses = Get-WmiObject Win32_Process | Where-Object {
        $_.Name -eq 'node.exe' -and
        ($_.CommandLine -like '*vite*' -or $_.CommandLine -like '*thoughtlab*frontend*')
    }

    foreach ($proc in $nodeProcesses) {
        Write-Host "  Killing frontend node process (PID: $($proc.ProcessId))..." -ForegroundColor Yellow
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        $found = $true
    }

    # Kill from PID file
    $pidFile = Join-Path $RunDir "frontend.pid"
    if (Test-Path $pidFile) {
        $pid = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($pid) {
            Write-Host "  Killing frontend from PID file (PID: $pid)..." -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            $found = $true
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }

    if ($found) {
        Write-Host "✅ Killed vite processes" -ForegroundColor Green
        return $true
    }

    return $false
}

# Stop backend
$backendStopped = $false
if (Kill-Port -Port 8000 -Name "Backend") { $backendStopped = $true }
if (Kill-UvicornProcesses) { $backendStopped = $true }

# Stop frontend on all possible ports
$frontendStopped = $false
if (Kill-Port -Port 5173 -Name "Frontend") { $frontendStopped = $true }
foreach ($port in 5174..5179) {
    if (Kill-Port -Port $port -Name "Frontend (port $port)") { $frontendStopped = $true }
}
if (Kill-ViteProcesses) { $frontendStopped = $true }

# Clean up PID files
Remove-Item (Join-Path $RunDir "backend.pid") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $RunDir "frontend.pid") -Force -ErrorAction SilentlyContinue

Write-Host ""
if ($backendStopped -or $frontendStopped) {
    Write-Host "✅ Servers stopped successfully" -ForegroundColor Green
} else {
    Write-Host "ℹ️  No servers were running" -ForegroundColor Gray
}
