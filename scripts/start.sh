#!/bin/bash
# Research Connection Graph - Start Script
# Starts both backend and frontend servers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_DIR="$PROJECT_ROOT/.run"
DETACHED=0

# Parse flags
for arg in "$@"; do
    case "$arg" in
        -d|--detached)
            DETACHED=1
            shift
            ;;
    esac
done

mkdir -p "$RUN_DIR"

echo "🚀 Starting Research Connection Graph..."
echo ""

# Check if Docker services are running
cd "$PROJECT_ROOT"
DOCKER_STATUS=$(docker-compose ps 2>/dev/null | grep -q "Up" && echo "running" || echo "stopped")
if [ "$DOCKER_STATUS" != "running" ]; then
    echo "⚠️  Docker services not running. Starting them..."
    docker-compose up -d
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
else
    echo "✅ Docker services are running"
fi

# Start backend
echo ""
echo "🐍 Starting backend server..."
cd "$PROJECT_ROOT/backend"

if [ ! -d ".venv" ]; then
    echo "❌ Backend not set up. Run ./scripts/setup.sh first."
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
    echo "❌ Could not find .venv Python"
    exit 1
fi

# Check if uvicorn is installed
if ! "$VENV_PYTHON" -m uvicorn --help > /dev/null 2>&1; then
    echo "❌ Backend dependencies not installed. uvicorn not found."
    echo "   Please run ./scripts/setup.sh first to install dependencies."
    exit 1
fi

# Clear Python bytecode cache to ensure fresh code is loaded
# This fixes issues where uvicorn --reload doesn't detect changes on Windows
echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Check if port 8000 is already in use
if command -v lsof > /dev/null 2>&1 && lsof -i :8000 > /dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use. Backend may already be running."
elif command -v netstat > /dev/null 2>&1 && netstat -an 2>/dev/null | grep -q ":8000.*LISTEN"; then
    echo "⚠️  Port 8000 is already in use. Backend may already be running."
elif command -v ss > /dev/null 2>&1 && ss -tuln 2>/dev/null | grep -q ":8000"; then
    echo "⚠️  Port 8000 is already in use. Backend may already be running."
fi

# Start backend (background; optionally detached)
# Use --reload-dir to explicitly watch the app directory
# Use watchfiles reloader which works better on Windows
echo "Backend starting at http://localhost:8000"
if [ $DETACHED -eq 1 ]; then
    nohup "$VENV_PYTHON" -m uvicorn app.main:app --reload --reload-dir app > /tmp/research-graph-backend.log 2>&1 &
else
    "$VENV_PYTHON" -m uvicorn app.main:app --reload --reload-dir app > /tmp/research-graph-backend.log 2>&1 &
fi
BACKEND_PID=$!
echo $BACKEND_PID > "$RUN_DIR/backend.pid"

# Give backend a moment to start
sleep 2

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check logs: tail -f /tmp/research-graph-backend.log"
    exit 1
fi

# Start frontend
echo ""
echo "📦 Starting frontend server..."
cd "$PROJECT_ROOT/frontend"

if [ ! -d "node_modules" ]; then
    echo "❌ Frontend not set up. Run ./scripts/setup.sh first."
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Check if port 5173 is already in use
if command -v lsof > /dev/null 2>&1 && lsof -i :5173 > /dev/null 2>&1; then
    echo "⚠️  Port 5173 is already in use. Frontend will try the next available port."
elif command -v netstat > /dev/null 2>&1 && netstat -an 2>/dev/null | grep -q ":5173.*LISTEN"; then
    echo "⚠️  Port 5173 is already in use. Frontend will try the next available port."
elif command -v ss > /dev/null 2>&1 && ss -tuln 2>/dev/null | grep -q ":5173"; then
    echo "⚠️  Port 5173 is already in use. Frontend will try the next available port."
fi

# Start frontend (background; optionally detached)
echo "Frontend starting at http://localhost:5173"
if [ $DETACHED -eq 1 ]; then
    nohup npm run dev > /tmp/research-graph-frontend.log 2>&1 &
else
    npm run dev > /tmp/research-graph-frontend.log 2>&1 &
fi
FRONTEND_PID=$!
echo $FRONTEND_PID > "$RUN_DIR/frontend.pid"

# Give frontend a moment to start
sleep 2

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start. Check logs: tail -f /tmp/research-graph-frontend.log"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

echo ""
echo "✅ Servers started successfully!"
echo ""
echo "Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo ""
echo "Logs:"
echo "  Backend:  tail -f /tmp/research-graph-backend.log"
echo "  Frontend: tail -f /tmp/research-graph-frontend.log"
echo ""

if [ $DETACHED -eq 1 ]; then
    echo "Running in detached mode. PID files:"
    echo "  $RUN_DIR/backend.pid"
    echo "  $RUN_DIR/frontend.pid"
    exit 0
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    # Wait a moment for processes to terminate
    sleep 1
    # Force kill if still running
    kill -9 $BACKEND_PID 2>/dev/null || true
    kill -9 $FRONTEND_PID 2>/dev/null || true
    echo "✅ Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes (this will block until interrupted)
wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || wait
