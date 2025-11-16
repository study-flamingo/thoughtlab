#!/bin/bash
# Research Connection Graph - Setup Script
# This script automates the entire setup process

set -e  # Exit on error

echo "🚀 Research Connection Graph - Setup"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo "📋 Checking prerequisites..."

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

MISSING_DEPS=0
check_command python3 || MISSING_DEPS=1
check_command node || MISSING_DEPS=1
check_command docker || MISSING_DEPS=1
check_command docker-compose || MISSING_DEPS=1

# Check for uv, install if not present
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠ uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    # Reload shell config if available
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc" 2>/dev/null || true
    fi
    if [ -f "$HOME/.zshrc" ]; then
        source "$HOME/.zshrc" 2>/dev/null || true
    fi
    # Try to find uv in common locations
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ Failed to install uv. Please install manually:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

check_command uv && echo -e "${GREEN}✓${NC} uv installed"

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}❌ Missing required dependencies. Please install them first.${NC}"
    exit 1
fi

echo ""
echo "🐳 Checking Docker services..."
cd "$(dirname "$0")/.."

# Check if Docker services are running
DOCKER_STATUS=$(docker-compose ps 2>/dev/null | grep -q "Up" && echo "running" || echo "stopped")
if [ "$DOCKER_STATUS" != "running" ]; then
    echo -e "${YELLOW}⚠ Docker services not running.${NC}"
    echo "Docker services are needed for Neo4j initialization during setup."
    echo "They will be started temporarily for setup, but use './scripts/start.sh' to start them for normal operation."
    docker-compose up -d
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
    
    # Check if services are up
    if ! docker-compose ps | grep -q "healthy"; then
        echo -e "${YELLOW}⚠ Services are starting. This may take 30-60 seconds.${NC}"
        echo "Run 'docker-compose ps' to check status."
    fi
else
    echo -e "${GREEN}✓${NC} Docker services are running"
fi

echo ""
echo "🐍 Setting up backend..."

cd backend

# Use uv to create venv and install dependencies
echo "Installing Python dependencies with uv..."
echo "  (uv is much faster than pip and has better dependency resolution)"

# Create venv with uv if it doesn't exist (uv defaults to .venv)
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Determine venv Python path for uv
if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/bin/python3" ]; then
    VENV_PYTHON=".venv/bin/python3"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    echo "❌ Could not find .venv Python"
    exit 1
fi

# Install dependencies using uv (much faster and better resolution)
echo "  Installing/updating packages..."
uv pip install --python "$VENV_PYTHON" -r requirements.txt
echo -e "${GREEN}✓${NC} Python dependencies installed"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    
    # Generate SECRET_KEY using Python
    SECRET_KEY=$("$VENV_PYTHON" -c "import secrets; print(secrets.token_urlsafe(32))")
    
    # Update SECRET_KEY in .env (works on both Linux and Mac)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
    else
        # Linux
        sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
    fi
    
    echo -e "${GREEN}✓${NC} Created .env with generated SECRET_KEY"
else
    echo -e "${GREEN}✓${NC} .env already exists"
fi

cd ..

echo ""
echo "📦 Setting up frontend..."

cd frontend

# Install/update dependencies
echo "Installing/updating Node dependencies (this may take a minute)..."
npm install --no-progress
echo -e "${GREEN}✓${NC} Node dependencies installed/updated"

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env"
else
    echo -e "${GREEN}✓${NC} .env already exists"
fi

cd ..

echo ""
echo "🗄️  Initializing Neo4j..."

# Check if Neo4j is ready
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec research-graph-neo4j cypher-shell -u neo4j -p research_graph_password "RETURN 1" &> /dev/null; then
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${YELLOW}⚠ Neo4j not ready yet. You may need to initialize manually later.${NC}"
else
    # Initialize Neo4j indexes and constraints
    echo "Creating indexes and constraints..."
    docker exec research-graph-neo4j cypher-shell -u neo4j -p research_graph_password <<EOF > /dev/null 2>&1
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
EOF
    echo -e "${GREEN}✓${NC} Neo4j initialized"
fi

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To start the application:"
echo ""
echo "  Terminal 1 (Backend):"
echo "    cd backend"
echo "    source .venv/bin/activate"
echo "    uvicorn app.main:app --reload"
echo ""
echo "  Terminal 2 (Frontend):"
echo "    cd frontend"
echo "    npm run dev"
echo ""
echo "Then open http://localhost:5173 in your browser"
echo ""
