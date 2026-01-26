#!/bin/bash
# ThoughtLab - Enhanced Setup Script
# Supports both local development and containerized development

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 ThoughtLab - Enhanced Setup"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}━━${NC} ${BLUE}$1${NC} ${BLUE}━━${NC}"
}

# Function to check command existence
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

# Ask user for development mode
print_section "Development Mode Selection"
echo "Choose your preferred development approach:"
echo ""
echo "1) ${GREEN}Fully Containerized${NC} (Recommended) - All services in containers (Avoids port conflicts)"
echo "2) ${GREEN}Mixed${NC} - Frontend local, backend in containers (Faster hot reload)"
echo "3) ${GREEN}Fully Local${NC} - Everything runs locally (Most control)"
echo ""

read -p "Select mode [1-3] (default: 1): " MODE
MODE=${MODE:-1}

case $MODE in
    1)
        DEV_MODE="docker"
        echo -e "${GREEN}Selected: Containerized development${NC}"
        ;;
    2)
        DEV_MODE="mixed"
        echo -e "${GREEN}Selected: Local frontend + Containerized backend${NC}"
        ;;
    3)
        DEV_MODE="local"
        echo -e "${GREEN}Selected: Fully local development${NC}"
        ;;
    *)
        echo -e "${YELLOW}Invalid selection. Using default (Containerized)${NC}"
        DEV_MODE="docker"
        ;;
esac

print_section "Checking Prerequisites"

MISSING_DEPS=0

if [ "$DEV_MODE" != "docker" ]; then
    check_command python3 || MISSING_DEPS=1
    check_command node || MISSING_DEPS=1
fi

if [ "$DEV_MODE" == "docker" ] || [ "$DEV_MODE" == "mixed" ]; then
    check_command docker || MISSING_DEPS=1
    check_command docker-compose || MISSING_DEPS=1
fi

# Check for uv if doing local Python development
if [ "$DEV_MODE" != "docker" ]; then
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
    fi

    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ Failed to install uv. Please install manually:${NC}"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    check_command uv && echo -e "${GREEN}✓${NC} uv installed"
fi

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${RED}❌ Missing required dependencies. Please install them first.${NC}"
    exit 1
fi

# Docker Development Setup
if [ "$DEV_MODE" == "docker" ]; then
    print_section "Docker Development Setup"

    echo "🐳 Setting up Docker services..."

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi

    # Create backend .env if needed
    cd "$PROJECT_ROOT/backend"
    if [ ! -f ".env" ]; then
        echo "Creating backend .env file..."
        cp .env.example .env

        # Generate SECRET_KEY using Python if available
        if command -v python3 &> /dev/null; then
            SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
            else
                sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
            fi
            echo -e "${GREEN}✓${NC} Created backend .env with generated SECRET_KEY"
        else
            echo -e "${YELLOW}⚠ Could not generate SECRET_KEY (python3 not available). Please update manually.${NC}"
            cp .env.example .env
        fi
    else
        echo -e "${GREEN}✓${NC} Backend .env already exists"
    fi

    # Build Docker images
    print_section "Building Docker Images"
    echo "This may take a few minutes on first run..."
    docker-compose build

    print_section "Setup Complete (Docker Mode)"
    echo -e "${GREEN}✅ All setup complete!${NC}"
    echo ""
    echo "To start development:"
    echo "  ./scripts/start.sh"
    echo ""
    echo "To start with containerized frontend:"
    echo "  ./scripts/start-docker.sh"

# Mixed Development Setup (Local frontend + Containerized backend)
elif [ "$DEV_MODE" == "mixed" ]; then
    print_section "Mixed Development Setup"

    echo "🐍 Setting up backend (Docker)..."

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi

    # Create backend .env if needed
    cd "$PROJECT_ROOT/backend"
    if [ ! -f ".env" ]; then
        echo "Creating backend .env file..."
        cp .env.example .env

        # Generate SECRET_KEY using Python
        SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
        else
            sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
        fi
        echo -e "${GREEN}✓${NC} Created backend .env with generated SECRET_KEY"
    else
        echo -e "${GREEN}✓${NC} Backend .env already exists"
    fi

    echo ""
    echo "📦 Setting up frontend (Local)..."

    cd "$PROJECT_ROOT/frontend"

    # Install/update Node dependencies
    echo "Installing/updating Node dependencies..."
    npm install --no-progress
    echo -e "${GREEN}✓${NC} Node dependencies installed/updated"

    # Create frontend .env if needed
    if [ ! -f ".env" ]; then
        echo "Creating frontend .env file..."
        cp .env.example .env
        echo -e "${GREEN}✓${NC} Created frontend .env"
    else
        echo -e "${GREEN}✓${NC} Frontend .env already exists"
    fi

    # Build backend Docker image
    print_section "Building Backend Docker Image"
    cd "$PROJECT_ROOT"
    docker-compose build backend

    print_section "Setup Complete (Mixed Mode)"
    echo -e "${GREEN}✅ All setup complete!${NC}"
    echo ""
    echo "To start development:"
    echo "  ./scripts/start.sh"
    echo ""
    echo "This will start:"
    echo "  - Backend in Docker container"
    echo "  - Frontend locally with hot reload"

# Fully Local Development Setup
elif [ "$DEV_MODE" == "local" ]; then
    print_section "Fully Local Development Setup"

    echo "🐍 Setting up backend..."

    cd "$PROJECT_ROOT/backend"

    # Create venv with uv
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment with uv..."
        uv venv
    fi

    # Determine venv Python path
    if [ -f ".venv/bin/python" ]; then
        VENV_PYTHON=".venv/bin/python"
    elif [ -f ".venv/bin/python3" ]; then
        VENV_PYTHON=".venv/bin/python3"
    elif [ -f ".venv/Scripts/python.exe" ]; then
        VENV_PYTHON=".venv/Scripts/python.exe"
    else
        echo -e "${RED}❌ Could not find .venv Python${NC}"
        exit 1
    fi

    # Install dependencies
    echo "Installing Python dependencies..."
    uv pip install --python "$VENV_PYTHON" -r requirements.txt
    echo -e "${GREEN}✓${NC} Python dependencies installed"

    # Create .env if needed
    if [ ! -f ".env" ]; then
        echo "Creating backend .env file..."
        cp .env.example .env

        # Generate SECRET_KEY
        SECRET_KEY=$("$VENV_PYTHON" -c "import secrets; print(secrets.token_urlsafe(32))")
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
        else
            sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
        fi
        echo -e "${GREEN}✓${NC} Created backend .env with generated SECRET_KEY"
    else
        echo -e "${GREEN}✓${NC} Backend .env already exists"
    fi

    echo ""
    echo "📦 Setting up frontend..."

    cd "$PROJECT_ROOT/frontend"

    # Install/update Node dependencies
    echo "Installing/updating Node dependencies..."
    npm install --no-progress
    echo -e "${GREEN}✓${NC} Node dependencies installed/updated"

    # Create .env if needed
    if [ ! -f ".env" ]; then
        echo "Creating frontend .env file..."
        cp .env.example .env
        echo -e "${GREEN}✓${NC} Created frontend .env"
    else
        echo -e "${GREEN}✓${NC} Frontend .env already exists"
    fi

    # For fully local development, we need Neo4j running
    print_section "Starting Neo4j (Local development requires Neo4j)"
    cd "$PROJECT_ROOT"
    if ! docker-compose ps neo4j 2>/dev/null | grep -q "Up"; then
        echo "Starting Neo4j..."
        docker-compose up -d neo4j
        echo "⏳ Waiting for Neo4j to start..."
        sleep 10
    else
        echo -e "${GREEN}✓${NC} Neo4j is already running"
    fi

    print_section "Setup Complete (Local Mode)"
    echo -e "${GREEN}✅ All setup complete!${NC}"
    echo ""
    echo "To start development:"
    echo "  ./scripts/start.sh"
    echo ""
    echo "This will start:"
    echo "  - Backend locally with hot reload"
    echo "  - Frontend locally with hot reload"
    echo "  - Neo4j in Docker container"
fi

print_section "Additional Commands"

echo -e "${BLUE}Start Development:${NC}"
echo "  ./scripts/start.sh"
echo ""
echo -e "${BLUE}Start Containerized:${NC}"
echo "  ./scripts/start-docker.sh"
echo ""
echo -e "${BLUE}Stop All Services:${NC}"
echo "  ./scripts/stop.sh"
echo ""
echo -e "${BLUE}View Service Status:${NC}"
echo "  docker-compose ps"
echo ""
echo -e "${BLUE}View Logs:${NC}"
echo "  docker-compose logs -f"
echo ""

print_section "Documentation"
echo "For detailed documentation:"
echo "  - Local development: See DEVELOPMENT_GUIDE.md"
echo "  - Containerized: See frontend/DOCKER_DEV.md"