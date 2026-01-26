#!/bin/bash
# ThoughtLab - Enhanced Start Script
# Supports both local and containerized development

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 ThoughtLab - Start Development"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect available mode
detect_mode() {
    # Check if we should use Docker mode (check if frontend Dockerfile exists and backend is set up for Docker)
    if [ -f "frontend/Dockerfile.dev" ] && [ -f "backend/Dockerfile" ]; then
        # Check if user wants containerized frontend (preferred)
        if [ -f ".containerized-frontend" ]; then
            echo "docker-frontend"
        else
            # Ask user which mode they prefer for this session
            echo "ask"  # Will prompt user
        fi
    elif [ -f "backend/.venv" ] && [ -f "frontend/node_modules" ]; then
        echo "local"
    else
        echo "unknown"
    fi
}

# Ask user for mode if not detected
MODE=$(detect_mode)

if [ "$MODE" == "unknown" ] || [ "$MODE" == "ask" ]; then
    if [ "$MODE" == "ask" ]; then
        echo "Choose development mode for this session:"
        echo ""
        echo "1) ${GREEN}Fully Containerized${NC} (Recommended - avoids port conflicts)"
        echo "2) ${GREEN}Local frontend + Containerized backend${NC} (Faster hot reload)"
        echo "3) ${GREEN}Fully local${NC} (Everything runs locally)"
        echo ""
        read -p "Select mode [1-3] (default: 1): " USER_MODE
        USER_MODE=${USER_MODE:-1}
    else
        echo "No development setup found. Please run ./scripts/setup.sh first."
        exit 1
    fi

    case $USER_MODE in
        1) MODE="docker-frontend" ;;
        2) MODE="mixed" ;;
        3) MODE="local" ;;
        *) MODE="docker-frontend" ;;
    esac
fi

echo -e "${BLUE}Mode:${NC} $MODE"
echo ""

# Ensure Git Bash utilities are in PATH (Windows Git Bash fix)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]; then
    export PATH="/usr/bin:$PATH"
    export PATH="/mingw64/bin:/mingw32/bin:$PATH"
fi

# Create run directories
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

# Function to check if port is available
check_port() {
    local port=$1
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]; then
        # Windows
        if netstat -ano 2>/dev/null | grep -E ":$port\s" | grep -E "LISTENING|ESTABLISHED" > /dev/null; then
            return 1  # Port in use
        fi
    else
        # Unix
        if lsof -i :$port > /dev/null 2>&1; then
            return 1  # Port in use
        fi
    fi
    return 0  # Port available
}

# Ensure required services are running
ensure_docker_services() {
    local services=$1
    echo "🐳 Ensuring Docker services are running: $services"

    # Start services if not running
    for service in $services; do
        if ! docker-compose ps $service 2>/dev/null | grep -q "Up"; then
            echo "  Starting $service..."
            docker-compose up -d $service
            sleep 2
        else
            echo -e "  ${GREEN}✓${NC} $service is running"
        fi
    done

    # Wait for Neo4j to be healthy (if included)
    if echo "$services" | grep -q "neo4j"; then
        echo "  Waiting for Neo4j to be ready..."
        MAX_RETRIES=30
        RETRY_COUNT=0
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if docker exec research-graph-neo4j cypher-shell -u neo4j -p research_graph_password "RETURN 1" &> /dev/null; then
                break
            fi
            RETRY_COUNT=$((RETRY_COUNT + 1))
            sleep 2
        done
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo -e "  ${GREEN}✓${NC} Neo4j is ready"
        else
            echo -e "  ${YELLOW}⚠ Neo4j may not be fully ready yet${NC}"
        fi
    fi
}

# Start in Mixed Mode (Local frontend + Containerized backend)
start_mixed() {
    echo "🚀 Starting Mixed Development Mode"
    echo "==================================="
    echo ""

    # Ensure Docker services are running
    ensure_docker_services "neo4j redis backend"

    # Check if port 5173 is available
    if ! check_port 5173; then
        echo -e "${YELLOW}⚠ Port 5173 is in use. Stopping conflicting processes...${NC}"
        # Try to stop any existing frontend
        "$PROJECT_ROOT/scripts/stop.sh" frontend-only > /dev/null 2>&1 || true
        sleep 2
    fi

    # Start backend health check
    echo "🔍 Checking backend health..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Backend is healthy"
    else
        echo -e "${YELLOW}⚠ Backend may still be starting up...${NC}"
    fi

    # Start frontend locally
    echo ""
    echo "📦 Starting frontend locally..."
    cd "$PROJECT_ROOT/frontend"

    FRONTEND_LOG="$LOG_DIR/frontend.log"
    echo "Frontend starting at http://localhost:5173"
    npm run dev > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$RUN_DIR/frontend.pid"

    # Wait for frontend to start
    echo "⏳ Waiting for frontend to start..."
    sleep 5

    # Check if frontend started
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "${RED}❌ Frontend failed to start. Check logs:${NC}"
        cat "$FRONTEND_LOG"
        exit 1
    fi

    # Get actual port from log
    FRONTEND_PORT=$(grep -o "localhost:[0-9]*" "$FRONTEND_LOG" 2>/dev/null | head -1 | cut -d: -f2)
    FRONTEND_PORT=${FRONTEND_PORT:-5173}

    echo -e "${GREEN}✓${NC} Frontend started (PID: $FRONTEND_PID)"

    # Show summary
    show_summary_mixed $FRONTEND_PORT

    # Setup cleanup and log tailing
    setup_cleanup_mixed $FRONTEND_PID
}

# Start in Docker Frontend Mode (Everything containerized)
start_docker_frontend() {
    echo "🚀 Starting Containerized Development Mode"
    echo "==========================================="
    echo ""

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
        exit 1
    fi

    # Stop any existing local frontend process
    if [ -f "$RUN_DIR/frontend.pid" ]; then
        local pid=$(cat "$RUN_DIR/frontend.pid")
        if kill -0 $pid 2>/dev/null; then
            echo "Stopping local frontend process..."
            kill $pid 2>/dev/null || true
        fi
        rm -f "$RUN_DIR/frontend.pid"
    fi

    # Start all services
    echo "🐳 Starting all services in Docker..."
    docker-compose up -d

    # Wait for services to be healthy
    echo "⏳ Waiting for services to start..."
    sleep 10

    # Check service status
    echo ""
    echo "📊 Service Status:"
    docker-compose ps

    # Show summary
    show_summary_docker
}

# Start in Fully Local Mode
start_local() {
    echo "🚀 Starting Fully Local Development Mode"
    echo "========================================="
    echo ""

    # Ensure Neo4j is running (Docker)
    ensure_docker_services "neo4j"

    # Start backend locally
    echo "🐍 Starting backend locally..."
    cd "$PROJECT_ROOT/backend"

    if [ ! -d ".venv" ]; then
        echo -e "${RED}❌ Backend not set up. Run ./scripts/setup.sh first.${NC}"
        exit 1
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

    # Clear Python cache
    echo "🧹 Clearing Python cache..."
    rm -rf app/__pycache__ app/**/__pycache__ 2>/dev/null || true

    # Start backend
    BACKEND_LOG="$LOG_DIR/backend.log"
    echo "Backend starting at http://localhost:8000"
    "$VENV_PYTHON" -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000 > "$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$RUN_DIR/backend.pid"

    # Wait for backend to start
    sleep 3

    # Check if backend started
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${RED}❌ Backend failed to start. Check logs:${NC}"
        cat "$BACKEND_LOG"
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Backend started (PID: $BACKEND_PID)"

    # Start frontend
    echo ""
    echo "📦 Starting frontend locally..."
    cd "$PROJECT_ROOT/frontend"

    if [ ! -d "node_modules" ]; then
        echo -e "${RED}❌ Frontend not set up. Run ./scripts/setup.sh first.${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi

    # Check port
    if ! check_port 5173; then
        echo -e "${YELLOW}⚠ Port 5173 is in use. Stopping conflicting processes...${NC}"
        "$PROJECT_ROOT/scripts/stop.sh" frontend-only > /dev/null 2>&1 || true
        sleep 2
    fi

    FRONTEND_LOG="$LOG_DIR/frontend.log"
    echo "Frontend starting at http://localhost:5173"
    npm run dev > "$FRONTEND_LOG" 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$RUN_DIR/frontend.pid"

    # Wait for frontend to start
    sleep 5

    # Check if frontend started
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "${RED}❌ Frontend failed to start. Check logs:${NC}"
        cat "$FRONTEND_LOG"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Frontend started (PID: $FRONTEND_PID)"

    # Get actual frontend port
    FRONTEND_PORT=$(grep -o "localhost:[0-9]*" "$FRONTEND_LOG" 2>/dev/null | head -1 | cut -d: -f2)
    FRONTEND_PORT=${FRONTEND_PORT:-5173}

    # Show summary
    show_summary_local $FRONTEND_PORT

    # Setup cleanup and log tailing
    setup_cleanup_local $BACKEND_PID $FRONTEND_PID
}

# Show summary for mixed mode
show_summary_mixed() {
    local port=$1
    echo ""
    echo "═══════════════════════════════════════════="
    echo "✅ Development Servers Started!"
    echo "═══════════════════════════════════════════="
    echo ""
    echo "  Backend:  http://localhost:8000"
    echo "  Frontend: http://localhost:$port"
    echo ""
    echo "  Neo4j:    http://localhost:7474"
    echo "  Redis:    localhost:6379"
    echo ""
    echo "  Logs: $LOG_DIR/"
    echo ""
    echo "  Press Ctrl+C to stop frontend (backend stays running)"
    echo "═══════════════════════════════════════════="
    echo ""
}

# Show summary for Docker mode
show_summary_docker() {
    echo ""
    echo "═══════════════════════════════════════════="
    echo "✅ All Services Running in Docker!"
    echo "═══════════════════════════════════════════="
    echo ""
    echo "  Frontend: http://localhost:5173"
    echo "  Backend:  http://localhost:8000"
    echo "  Neo4j:    http://localhost:7474"
    echo "  Redis:    localhost:6379"
    echo ""
    echo "  To view logs: docker-compose logs -f"
    echo "  To stop: docker-compose down"
    echo "═══════════════════════════════════════════="
    echo ""
}

# Show summary for local mode
show_summary_local() {
    local port=$1
    echo ""
    echo "═══════════════════════════════════════════="
    echo "✅ All Services Started!"
    echo "═══════════════════════════════════════════="
    echo ""
    echo "  Backend:  http://localhost:8000"
    echo "  Frontend: http://localhost:$port"
    echo ""
    echo "  Neo4j:    http://localhost:7474 (Docker)"
    echo "  Redis:    localhost:6379"
    echo ""
    echo "  Logs: $LOG_DIR/"
    echo ""
    echo "  Press Ctrl+C to stop all servers"
    echo "═══════════════════════════════════════════="
    echo ""
}

# Setup cleanup for mixed mode
setup_cleanup_mixed() {
    local frontend_pid=$1

    cleanup() {
        echo ""
        echo "🛑 Stopping frontend..."
        kill $frontend_pid 2>/dev/null || true
        sleep 1
        kill -9 $frontend_pid 2>/dev/null || true
        rm -f "$RUN_DIR/frontend.pid"
        echo "✅ Frontend stopped (backend still running in Docker)"
        exit 0
    }

    trap cleanup SIGINT SIGTERM
    tail -f "$LOG_DIR/frontend.log" 2>/dev/null || wait
}

# Setup cleanup for local mode
setup_cleanup_local() {
    local backend_pid=$1
    local frontend_pid=$2

    cleanup() {
        echo ""
        echo "🛑 Stopping all servers..."
        kill $backend_pid $frontend_pid 2>/dev/null || true
        sleep 1
        kill -9 $backend_pid $frontend_pid 2>/dev/null || true
        rm -f "$RUN_DIR/backend.pid" "$RUN_DIR/frontend.pid"
        echo "✅ All servers stopped"
        exit 0
    }

    trap cleanup SIGINT SIGTERM
    tail -f "$LOG_DIR/backend.log" "$LOG_DIR/frontend.log" 2>/dev/null || wait
}

# Main execution
case $MODE in
    "mixed")
        start_mixed
        ;;
    "docker-frontend")
        start_docker_frontend
        ;;
    "local")
        start_local
        ;;
    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        echo "Please run ./scripts/setup.sh first"
        exit 1
        ;;
esac