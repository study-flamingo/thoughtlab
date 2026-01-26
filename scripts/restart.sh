#!/bin/bash
# ThoughtLab - Enhanced Restart Script
# Restarts all services with proper cleanup and port management

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔄 ThoughtLab - Restart Services"
echo "================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detect mode
MODE="local"
if [ -f ".containerized-frontend" ]; then
    MODE="docker"
fi

# Allow mode override
if [ -n "$1" ]; then
    case $1 in
        "docker") MODE="docker" ;;
        "local") MODE="local" ;;
        "mixed") MODE="mixed" ;;
    esac
fi

echo -e "${BLUE}Mode:${NC} $MODE"
echo ""

# Stop services first
echo "🛑 Stopping services..."
"$PROJECT_ROOT/scripts/stop.sh" > /dev/null 2>&1 || true

# Clear caches
echo ""
echo "🧹 Clearing caches..."

# Clear Python cache
if [ -d "$PROJECT_ROOT/backend" ]; then
    echo "  Clearing Python bytecode cache..."
    cd "$PROJECT_ROOT/backend"
    rm -rf app/__pycache__ app/**/__pycache__ 2>/dev/null || true
fi

# Clear frontend cache
if [ -d "$PROJECT_ROOT/frontend/.vite" ]; then
    echo "  Clearing Vite cache..."
    cd "$PROJECT_ROOT/frontend"
    rm -rf .vite 2>/dev/null || true
fi

echo -e "${GREEN}✓${NC} Caches cleared"

# Wait for ports to be released
echo ""
echo "⏳ Waiting for ports to be released..."
sleep 3

# Verify ports are free
echo "🔍 Verifying ports are free..."

# Check common ports
check_port() {
    local port=$1
    local name=$2
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]; then
        if netstat -ano 2>/dev/null | grep -E ":$port\s" | grep -E "LISTENING|ESTABLISHED" > /dev/null; then
            echo -e "${YELLOW}⚠ Port $port ($name) still in use${NC}"
            return 1
        fi
    else
        if lsof -i :$port > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠ Port $port ($name) still in use${NC}"
            return 1
        fi
    fi
    echo -e "${GREEN}✓${NC} Port $port ($name) is free"
    return 0
}

check_port 5173 "Frontend"
check_port 8000 "Backend"

# Additional ports for local development
if [ "$MODE" != "docker" ]; then
    check_port 7474 "Neo4j"
    check_port 6379 "Redis"
fi

# Start services based on mode
echo ""
case $MODE in
    "docker")
        echo "🚀 Starting all services in Docker..."
        "$PROJECT_ROOT/scripts/start-docker.sh"
        ;;

    "mixed")
        echo "🚀 Starting mixed development mode..."
        "$PROJECT_ROOT/scripts/start.sh"
        ;;

    "local")
        echo "🚀 Starting fully local development mode..."
        "$PROJECT_ROOT/scripts/start.sh"
        ;;

    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        exit 1
        ;;
esac