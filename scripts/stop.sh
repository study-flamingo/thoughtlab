#!/bin/bash
# Research Connection Graph - Stop Script
# Stops backend and frontend servers

set +e  # Don't exit on error - processes might not be running

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🛑 Stopping Research Connection Graph servers..."
echo ""

# Function to find and kill process on a port
kill_port() {
    local port=$1
    local name=$2
    
    # Try different methods to find the process
    local pid=""
    
    if command -v lsof > /dev/null 2>&1; then
        pid=$(lsof -ti :$port 2>/dev/null || true)
    elif command -v fuser > /dev/null 2>&1; then
        pid=$(fuser $port/tcp 2>/dev/null | awk '{print $1}' || true)
    elif command -v netstat > /dev/null 2>&1; then
        pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | head -1 || true)
    elif command -v ss > /dev/null 2>&1; then
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    fi
    
    if [ -n "$pid" ] && [ "$pid" != "" ]; then
        echo "Stopping $name (PID: $pid) on port $port..."
        kill $pid 2>/dev/null || true
        sleep 1
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
        echo "✅ Stopped $name"
        return 0
    else
        echo "ℹ️  No $name process found on port $port"
        return 1
    fi
}

# Stop backend (port 8000)
BACKEND_STOPPED=0
if kill_port 8000 "Backend"; then
    BACKEND_STOPPED=1
fi

# Stop frontend (port 5173)
FRONTEND_STOPPED=0
if kill_port 5173 "Frontend"; then
    FRONTEND_STOPPED=1
fi

# Also try to find processes by name (as fallback)
if [ $BACKEND_STOPPED -eq 0 ]; then
    echo ""
    echo "Trying to find backend process by name..."
    BACKEND_PIDS=$(pgrep -f "uvicorn.*app.main:app" 2>/dev/null || true)
    if [ -n "$BACKEND_PIDS" ]; then
        for pid in $BACKEND_PIDS; do
            echo "Stopping backend process (PID: $pid)..."
            kill $pid 2>/dev/null || true
            sleep 1
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid 2>/dev/null || true
            fi
        done
        echo "✅ Stopped backend processes"
        BACKEND_STOPPED=1
    fi
fi

if [ $FRONTEND_STOPPED -eq 0 ]; then
    echo ""
    echo "Trying to find frontend process by name..."
    FRONTEND_PIDS=$(pgrep -f "vite.*dev" 2>/dev/null || true)
    if [ -n "$FRONTEND_PIDS" ]; then
        for pid in $FRONTEND_PIDS; do
            echo "Stopping frontend process (PID: $pid)..."
            kill $pid 2>/dev/null || true
            sleep 1
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid 2>/dev/null || true
            fi
        done
        echo "✅ Stopped frontend processes"
        FRONTEND_STOPPED=1
    fi
fi

echo ""
if [ $BACKEND_STOPPED -eq 1 ] || [ $FRONTEND_STOPPED -eq 1 ]; then
    echo "✅ Servers stopped successfully"
else
    echo "ℹ️  No servers were running"
fi
