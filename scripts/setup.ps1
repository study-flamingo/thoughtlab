# Research Connection Graph - Setup Script (PowerShell)
# This script automates the entire setup process for Windows

$ErrorActionPreference = "Stop"

Write-Host "🚀 Research Connection Graph - Setup" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

function Test-Command {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    if ($?) {
        Write-Host "✓ $Command installed" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ $Command not found" -ForegroundColor Red
        return $false
    }
}

$missingDeps = @()
if (-not (Test-Command "python")) { $missingDeps += "Python" }
if (-not (Test-Command "node")) { $missingDeps += "Node.js" }
if (-not (Test-Command "docker")) { $missingDeps += "Docker" }
if (-not (Test-Command "docker-compose")) { $missingDeps += "Docker Compose" }

# Check for uv, install if not present
if (-not (Test-Command "uv")) {
    Write-Host "⚠ uv not found. Installing uv..." -ForegroundColor Yellow
    try {
        # Install uv using PowerShell
        $uvInstallScript = Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -UseBasicParsing
        Invoke-Expression $uvInstallScript.Content
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } catch {
        Write-Host "❌ Failed to install uv automatically. Please install manually:" -ForegroundColor Red
        Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
        Write-Host "Or visit: https://github.com/astral-sh/uv" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Command "uv")) {
    Write-Host "❌ uv is required but not found. Please install it first." -ForegroundColor Red
    Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor Yellow
    exit 1
}

Test-Command "uv" | Out-Null
Write-Host "✓ uv installed" -ForegroundColor Green

if ($missingDeps.Count -gt 0) {
    Write-Host "❌ Missing required dependencies: $($missingDeps -join ', ')" -ForegroundColor Red
    Write-Host "Please install them first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🐳 Starting Docker services..." -ForegroundColor Yellow

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

docker-compose up -d

Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "🐍 Setting up backend..." -ForegroundColor Yellow

Set-Location backend

# Use uv to create venv and install dependencies
Write-Host "Installing Python dependencies with uv..."
Write-Host "  (uv is much faster than pip and has better dependency resolution)"

# Create venv with uv if it doesn't exist (uv defaults to .venv)
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment with uv..."
    uv venv
}

# Determine venv Python path for uv
if (Test-Path ".venv\Scripts\python.exe") {
    $venvPython = ".venv\Scripts\python.exe"
} elseif (Test-Path ".venv\bin\python.exe") {
    $venvPython = ".venv\bin\python.exe"
} elseif (Test-Path ".venv\bin\python") {
    $venvPython = ".venv\bin\python"
} else {
    Write-Host "❌ Could not find .venv Python" -ForegroundColor Red
    exit 1
}

# Install dependencies using uv (much faster and better resolution)
Write-Host "  Installing/updating packages..."
uv pip install --python $venvPython -r requirements.txt
Write-Host "✓ Python dependencies installed" -ForegroundColor Green

# Create .env if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..."
    Copy-Item .env.example .env
    
    # Generate SECRET_KEY using Python from venv
    $secretKey = & $venvPython -c "import secrets; print(secrets.token_urlsafe(32))"
    
    # Update SECRET_KEY in .env
    (Get-Content .env) -replace 'SECRET_KEY=.*', "SECRET_KEY=$secretKey" | Set-Content .env
    
    Write-Host "✓ Created .env with generated SECRET_KEY" -ForegroundColor Green
} else {
    Write-Host "✓ .env already exists" -ForegroundColor Green
}

Set-Location ..

Write-Host ""
Write-Host "📦 Setting up frontend..." -ForegroundColor Yellow

Set-Location frontend

# Install/update dependencies
Write-Host "Installing/updating Node dependencies (this may take a minute)..."
npm install
Write-Host "✓ Node dependencies installed/updated" -ForegroundColor Green

# Create .env if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..."
    Copy-Item .env.example .env
    Write-Host "✓ Created .env" -ForegroundColor Green
} else {
    Write-Host "✓ .env already exists" -ForegroundColor Green
}

Set-Location ..

Write-Host ""
Write-Host "🗄️  Initializing Neo4j..." -ForegroundColor Yellow

# Wait for Neo4j to be ready
$maxRetries = 30
$retryCount = 0
$neo4jReady = $false

while ($retryCount -lt $maxRetries) {
    try {
        docker exec research-graph-neo4j cypher-shell -u neo4j -p research_graph_password "RETURN 1" 2>$null | Out-Null
        $neo4jReady = $true
        break
    } catch {
        $retryCount++
        Start-Sleep -Seconds 2
    }
}

if ($neo4jReady) {
    Write-Host "Creating indexes and constraints..."
    $initScript = @"
CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (o:Observation) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT hypothesis_id IF NOT EXISTS FOR (h:Hypothesis) REQUIRE h.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE TEXT INDEX observation_text IF NOT EXISTS FOR (o:Observation) ON o.text;
CREATE TEXT INDEX hypothesis_claim IF NOT EXISTS FOR (h:Hypothesis) ON h.claim;
CREATE TEXT INDEX source_title IF NOT EXISTS FOR (s:Source) ON s.title;
CREATE INDEX observation_created IF NOT EXISTS FOR (o:Observation) ON o.created_at;
CREATE INDEX hypothesis_created IF NOT EXISTS FOR (h:Hypothesis) ON h.created_at;
CREATE INDEX source_created IF NOT EXISTS FOR (s:Source) ON s.created_at;
"@
    
    $initScript | docker exec -i research-graph-neo4j cypher-shell -u neo4j -p research_graph_password | Out-Null
    Write-Host "✓ Neo4j initialized" -ForegroundColor Green
} else {
    Write-Host "⚠ Neo4j not ready yet. You may need to initialize manually later." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Terminal 1 (Backend):" -ForegroundColor Yellow
Write-Host "    cd backend"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    uvicorn app.main:app --reload"
Write-Host ""
Write-Host "  Terminal 2 (Frontend):" -ForegroundColor Yellow
Write-Host "    cd frontend"
Write-Host "    npm run dev"
Write-Host ""
Write-Host "Then open http://localhost:5173 in your browser" -ForegroundColor Cyan
Write-Host ""
