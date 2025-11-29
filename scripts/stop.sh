#!/bin/bash
# Research Connection Graph - Stop Script
# Stops backend and frontend servers

set +e  # Don't exit on error - processes might not be running

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_DIR="$PROJECT_ROOT/.run"

echo "🛑 Stopping Research Connection Graph servers..."
echo ""

# Function to find and kill process on a port
kill_port() {
    local port=$1
    local name=$2
    
    # Try different methods to find the process
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
    
    if [ -n "$pids" ] && [ "$pids" != "" ]; then
        for pid in $pids; do
            echo "Stopping $name (PID: $pid) on port $port..."
            kill $pid 2>/dev/null || true
        done
        sleep 1
        # Force kill if still running
        for pid in $pids; do
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid 2>/dev/null || true
            fi
        done
        echo "✅ Stopped $name"
        return 0
    else
        echo "ℹ️  No $name process found on port $port"
        return 1
    fi
}

# Kill ALL uvicorn processes (parent and children) to ensure clean restart
kill_uvicorn_processes() {
    echo "Searching for all uvicorn processes..."
    local found=0
    
    # Method 1: pgrep for uvicorn
    if command -v pgrep > /dev/null 2>&1; then
        UVICORN_PIDS=$(pgrep -f "uvicorn" 2>/dev/null || true)
        if [ -n "$UVICORN_PIDS" ]; then
            for pid in $UVICORN_PIDS; do
                echo "  Killing uvicorn process (PID: $pid)..."
                kill -9 $pid 2>/dev/null || true
                found=1
            done
        fi
    fi
    
    # Method 2: On Windows (Git Bash), use taskkill
    if command -v taskkill > /dev/null 2>&1; then
        # Kill Python processes running uvicorn
        taskkill //F //FI "IMAGENAME eq python.exe" //FI "WINDOWTITLE eq *uvicorn*" 2>/dev/null || true
    fi
    
    # Method 3: Kill from PID file
    if [ -f "$RUN_DIR/backend.pid" ]; then
        local pid=$(cat "$RUN_DIR/backend.pid" 2>/dev/null)
        if [ -n "$pid" ]; then
            echo "  Killing backend from PID file (PID: $pid)..."
            kill -9 $pid 2>/dev/null || true
            found=1
        fi
        rm -f "$RUN_DIR/backend.pid"
    fi
    
    if [ $found -eq 1 ]; then
        echo "✅ Killed uvicorn processes"
        return 0
    fi
    return 1
}

# Kill ALL vite/node processes for frontend
kill_vite_processes() {
    echo "Searching for all vite/frontend processes..."
    local found=0
    
    if command -v pgrep > /dev/null 2>&1; then
        VITE_PIDS=$(pgrep -f "vite" 2>/dev/null || true)
        if [ -n "$VITE_PIDS" ]; then
            for pid in $VITE_PIDS; do
                echo "  Killing vite process (PID: $pid)..."
                kill -9 $pid 2>/dev/null || true
                found=1
            done
        fi
    fi
    
    # Method: Kill from PID file
    if [ -f "$RUN_DIR/frontend.pid" ]; then
        local pid=$(cat "$RUN_DIR/frontend.pid" 2>/dev/null)
        if [ -n "$pid" ]; then
            echo "  Killing frontend from PID file (PID: $pid)..."
            kill -9 $pid 2>/dev/null || true
            found=1
        fi
        rm -f "$RUN_DIR/frontend.pid"
    fi
    
    if [ $found -eq 1 ]; then
        echo "✅ Killed vite processes"
        return 0
    fi
    return 1
}

# Stop backend - try multiple methods
BACKEND_STOPPED=0
if kill_port 8000 "Backend"; then
    BACKEND_STOPPED=1
fi
if kill_uvicorn_processes; then
    BACKEND_STOPPED=1
fi

# Stop frontend - try multiple methods  
FRONTEND_STOPPED=0
if kill_port 5173 "Frontend"; then
    FRONTEND_STOPPED=1
fi
if kill_vite_processes; then
    FRONTEND_STOPPED=1
fi

# Clean up PID files
rm -f "$RUN_DIR/backend.pid" "$RUN_DIR/frontend.pid" 2>/dev/null

echo ""
if [ $BACKEND_STOPPED -eq 1 ] || [ $FRONTEND_STOPPED -eq 1 ]; then
    echo "✅ Servers stopped successfully"
else
    echo "ℹ️  No servers were running"
fi
