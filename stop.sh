#!/bin/bash
# Research Connection Graph - Stop Script
# Stops backend and frontend servers

set +e  # Don't exit on error - processes might not be running

# Ensure Git Bash utilities are in PATH (Windows Git Bash fix)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]; then
    export PATH="/usr/bin:$PATH"
    # Additional common Git Bash paths
    export PATH="/mingw64/bin:/mingw32/bin:$PATH"
fi

# Verify required commands are available
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ Error: Required command '$1' not found."
        echo "   Please ensure you're running this in Git Bash with Unix tools installed."
        echo "   You may need to reinstall Git for Windows with Unix tools enabled."
        exit 1
    fi
}

# Only check critical commands that might be missing
check_command dirname
check_command grep
check_command awk

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"

echo "🛑 Stopping Research Connection Graph servers..."
echo ""

# Detect if running on Windows (Git Bash, MSYS, Cygwin)
is_windows() {
    [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OS" == "Windows_NT" ]]
}

# Function to find and kill process on a port (Windows-aware)
kill_port() {
    local port=$1
    local name=$2
    local pids=""

    if is_windows; then
        # Windows: use netstat to find PIDs on the port
        # More robust parsing that handles different netstat output formats
        pids=$(netstat -ano 2>/dev/null | awk "/TCP.*:$port .*LISTENING/ {print \$5}" | sort -u || true)

        if [ -n "$pids" ]; then
            for pid in $pids; do
                # Skip if not a number or if 0
                if [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" != "0" ]; then
                    echo "Stopping $name (PID: $pid) on port $port..."
                    # Use taskkill on Windows - force kill the process tree
                    taskkill //F //PID "$pid" //T 2>&1 | grep -v "ERROR: The process" || true
                fi
            done
            echo "✅ Stopped $name on port $port"
            return 0
        fi
    else
        # Unix: try lsof, fuser, netstat, ss
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
        fi
    fi

    echo "ℹ️  No $name process found on port $port"
    return 1
}

# Kill ALL uvicorn/python processes (parent and children) to ensure clean restart
kill_uvicorn_processes() {
    echo "Searching for all uvicorn/python processes..."
    local found=0

    if is_windows; then
        # Method 1: Kill by process name and command line
        echo "  Checking for python.exe processes running uvicorn..."
        taskkill //F //FI "IMAGENAME eq python.exe" //FI "WINDOWTITLE eq *uvicorn*" //T 2>&1 | grep -v "ERROR: The process" || true

        # Method 2: Use PowerShell to find and kill python processes with uvicorn in command line
        powershell.exe -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { \$_.CommandLine -like '*uvicorn*' } | Stop-Process -Force" 2>/dev/null || true

        # Method 3: Kill all python.exe processes in the backend directory
        # This is aggressive but effective
        local backend_pids=$(powershell.exe -Command "Get-WmiObject Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*thoughtlab*backend*' } | Select-Object -ExpandProperty ProcessId" 2>/dev/null | tr -d '\r' || true)
        if [ -n "$backend_pids" ]; then
            for pid in $backend_pids; do
                pid=$(echo "$pid" | tr -d ' \r\n')
                if [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" != "0" ]; then
                    echo "  Killing backend python process (PID: $pid)..."
                    taskkill //F //PID "$pid" //T 2>&1 | grep -v "ERROR: The process" || true
                    found=1
                fi
            done
        fi
    else
        # Unix: pgrep for uvicorn
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
    fi

    # Kill from PID file (cross-platform)
    if [ -f "$RUN_DIR/backend.pid" ]; then
        local pid=$(cat "$RUN_DIR/backend.pid" 2>/dev/null | tr -d '\r\n')
        if [ -n "$pid" ]; then
            echo "  Killing backend from PID file (PID: $pid)..."
            if is_windows; then
                taskkill //F //PID "$pid" //T 2>&1 | grep -v "ERROR: The process" || true
            else
                kill -9 $pid 2>/dev/null || true
            fi
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

    if is_windows; then
        # Method 1: Kill by process name and command line filter
        echo "  Checking for node.exe processes running vite..."
        taskkill //F //FI "IMAGENAME eq node.exe" //FI "WINDOWTITLE eq *vite*" //T 2>&1 | grep -v "ERROR: The process" || true

        # Method 2: Use PowerShell to find and kill node processes with vite in command line
        powershell.exe -Command "Get-Process node -ErrorAction SilentlyContinue | Where-Object { \$_.CommandLine -like '*vite*' } | Stop-Process -Force" 2>/dev/null || true

        # Method 3: Kill all node.exe processes in the frontend directory
        local frontend_pids=$(powershell.exe -Command "Get-WmiObject Win32_Process | Where-Object { \$_.Name -eq 'node.exe' -and \$_.CommandLine -like '*thoughtlab*frontend*' } | Select-Object -ExpandProperty ProcessId" 2>/dev/null | tr -d '\r' || true)
        if [ -n "$frontend_pids" ]; then
            for pid in $frontend_pids; do
                pid=$(echo "$pid" | tr -d ' \r\n')
                if [[ "$pid" =~ ^[0-9]+$ ]] && [ "$pid" != "0" ]; then
                    echo "  Killing frontend node process (PID: $pid)..."
                    taskkill //F //PID "$pid" //T 2>&1 | grep -v "ERROR: The process" || true
                    found=1
                fi
            done
        fi
    else
        # Unix: pgrep for vite
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
    fi

    # Kill from PID file (cross-platform)
    if [ -f "$RUN_DIR/frontend.pid" ]; then
        local pid=$(cat "$RUN_DIR/frontend.pid" 2>/dev/null | tr -d '\r\n')
        if [ -n "$pid" ]; then
            echo "  Killing frontend from PID file (PID: $pid)..."
            if is_windows; then
                taskkill //F //PID "$pid" //T 2>&1 | grep -v "ERROR: The process" || true
            else
                kill -9 $pid 2>/dev/null || true
            fi
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

# Stop frontend - try multiple methods and ports
FRONTEND_STOPPED=0

# Kill port 5173 (primary frontend port)
if kill_port 5173 "Frontend"; then
    FRONTEND_STOPPED=1
fi

# Also check and kill any vite processes on nearby ports (5174-5179)
# in case vite auto-incremented the port
for port in 5174 5175 5176 5177 5178 5179; do
    if kill_port $port "Frontend (port $port)"; then
        FRONTEND_STOPPED=1
    fi
done

# Kill all vite processes by name
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
