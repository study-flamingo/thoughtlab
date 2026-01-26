#!/bin/bash
# ThoughtLab - Enhanced Stop Script
# Stops all running services (both Docker and local processes)

set +e  # Don't exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🛑 ThoughtLab - Stopping Services"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

RUN_DIR="$PROJECT_ROOT/.run"
STOPPED_SERVICES=0

# Check mode
USE_DOCKER_FRONTEND=false
if [ -f ".containerized-frontend" ]; then
    USE_DOCKER_FRONTEND=true
fi

# Stop Docker services (always try, even if only backend is containerized)
stop_docker_services() {
    local service=$1
    echo "🐳 Checking Docker services: $service"

    if docker-compose ps $service 2>/dev/null | grep -q "Up"; then
        echo "  Stopping $service..."
        docker-compose stop $service
        STOPPED_SERVICES=$((STOPPED_SERVICES + 1))
        echo -e "  ${GREEN}✓${NC} Stopped $service"
    else
        echo -e "  ${GREEN}✓${NC} $service not running"
    fi
}

# Stop local processes by PID file
stop_local_process() {
    local process_name=$1
    local pid_file="$RUN_DIR/$process_name.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file" 2>/dev/null | tr -d '\r\n')
        if [ -n "$pid" ]; then
            if kill -0 $pid 2>/dev/null; then
                echo "  Stopping $process_name (PID: $pid)..."
                kill $pid 2>/dev/null || true
                sleep 1
                # Force kill if still running
                if kill -0 $pid 2>/dev/null; then
                    kill -9 $pid 2>/dev/null || true
                fi
                STOPPED_SERVICES=$((STOPPED_SERVICES + 1))
                echo -e "  ${GREEN}✓${NC} Stopped $process_name"
            else
                echo -e "  ${GREEN}✓${NC} $process_name not running (stale PID file)"
            fi
        fi
        rm -f "$pid_file"
    fi
}

# Stop local process by port
stop_process_by_port() {
    local port=$1
    local name=$2

    # Windows detection
    is_windows() {
        [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]
    }

    if is_windows; then
        # Windows: use netstat to find PIDs
        local pids=$(netstat -ano 2>/dev/null | grep -E ":$port\s" | grep -E "LISTENING|ESTABLISHED" | awk '{print $NF}' | sort -u || true)

        if [ -n "$pids" ]; then
            for pid in $pids; do
                if [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" != "0" ]; then
                    echo "  Stopping $name (PID: $pid) on port $port..."
                    taskkill //F //PID "$pid" //T 2>&1 | grep -v "ERROR: The process" || true
                    STOPPED_SERVICES=$((STOPPED_SERVICES + 1))
                fi
            done
            echo -e "  ${GREEN}✓${NC} Stopped $name"
            return 0
        fi
    else
        # Unix: try various methods
        local pids=""
        if command -v lsof > /dev/null 2>&1; then
            pids=$(lsof -ti :$port 2>/dev/null || true)
        elif command -v fuser > /dev/null 2>&1; then
            pids=$(fuser $port/tcp 2>/dev/null | awk '{print $1}' || true)
        elif command -v netstat > /dev/null 2>&1; then
            pids=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 || true)
        elif command -v ss > /dev/null 2>&1; then
            pids=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' || true)
        fi

        if [ -n "$pids" ]; then
            for pid in $pids; do
                echo "  Stopping $name (PID: $pid) on port $port..."
                kill $pid 2>/dev/null || true
            done
            sleep 1
            # Force kill if still running
            for pid in $pids; do
                if kill -0 $pid 2>/dev/null; then
                    kill -9 $pid 2>/dev/null || true
                fi
            done
            STOPPED_SERVICES=$((STOPPED_SERVICES + 1))
            echo -e "  ${GREEN}✓${NC} Stopped $name"
            return 0
        fi
    fi

    echo -e "  ${GREEN}✓${NC} No $name process found on port $port"
    return 1
}

# Check argument for specific mode
MODE=${1:-all}

case $MODE in
    "frontend-only")
        echo "🛑 Stopping frontend services only..."
        echo ""

        # First, try to stop via docker-compose (cleanest)
        if docker-compose ps frontend 2>/dev/null | grep -q "Up"; then
            echo "  Stopping Docker frontend container..."
            docker-compose stop frontend
            STOPPED_SERVICES=$((STOPPED_SERVICES + 1))
        fi

        # Stop local frontend process
        stop_local_process "frontend"

        # Be aggressive about port cleanup (check multiple times)
        echo "  Cleaning up any processes on frontend ports..."
        for i in {1..3}; do
            for port in 5173 5174 5175; do
                stop_process_by_port $port "Frontend (port $port)" > /dev/null 2>&1 || true
            done
            sleep 1
        done

        # Final verification
        for port in 5173 5174 5175; do
            stop_process_by_port $port "Frontend (port $port)"
        done
        ;;

    "backend-only")
        echo "🛑 Stopping backend services only..."
        echo ""

        # Stop local backend process
        stop_local_process "backend"

        # Stop containerized backend if it exists
        if docker-compose ps backend 2>/dev/null | grep -q "Up"; then
            stop_docker_services "backend"
        fi

        # Also check for any uvicorn processes on port 8000
        stop_process_by_port 8000 "Backend"
        ;;

    "all"|"")
        echo "🛑 Stopping all ThoughtLab services..."
        echo ""

        # Stop frontend services
        echo "📦 Frontend:"
        stop_local_process "frontend"
        if docker-compose ps frontend 2>/dev/null | grep -q "Up"; then
            stop_docker_services "frontend"
        fi

        # Stop any processes on frontend ports
        for port in 5173 5174 5175; do
            stop_process_by_port $port "Frontend (port $port)"
        done

        echo ""

        # Stop backend services
        echo "🐍 Backend:"
        stop_local_process "backend"
        if docker-compose ps backend 2>/dev/null | grep -q "Up"; then
            stop_docker_services "backend"
        fi

        # Stop any processes on backend ports
        stop_process_by_port 8000 "Backend"

        echo ""

        # Optional: Stop Docker infrastructure services
        echo "🐳 Docker Infrastructure:"
        echo "Do you want to stop Neo4j and Redis as well? (y/n): "
        read -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            stop_docker_services "neo4j"
            stop_docker_services "redis"
        else
            echo -e "  ${GREEN}✓${NC} Keeping Neo4j and Redis running"
        fi
        ;;

    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        echo "Usage: $0 [all|frontend-only|backend-only]"
        exit 1
        ;;
esac

echo ""
if [ $STOPPED_SERVICES -gt 0 ]; then
    echo -e "${GREEN}✅ Stopped $STOPPED_SERVICES service(s)${NC}"
else
    echo -e "${YELLOW}ℹ️  No services were running${NC}"
fi

# Clean up any stale PID files
if [ -d "$RUN_DIR" ]; then
    echo ""
    echo "🧹 Cleaning up PID files..."
    rm -f "$RUN_DIR"/*.pid 2>/dev/null || true
    echo -e "${GREEN}✓${NC} PID files cleaned up"
fi