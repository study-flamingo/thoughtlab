#!/bin/bash
# Research Connection Graph - Start Script
# Starts both backend and frontend servers

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$RUN_DIR/logs"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "🚀 Starting Research Connection Graph..."
echo ""

# Check if Docker services are running
cd "$PROJECT_ROOT"
if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
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
    echo "❌ Backend not set up. Run ./setup.sh first."
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
    echo "   Please run ./setup.sh first to install dependencies."
    exit 1
fi

# Clear Python bytecode cache (helps with reload issues)
echo "🧹 Clearing Python cache..."
rm -rf app/__pycache__ app/**/__pycache__ 2>/dev/null || true

# Start backend in background
BACKEND_LOG="$LOG_DIR/backend.log"
echo "Backend starting at http://localhost:8000"
"$VENV_PYTHON" -m uvicorn app.main:app --reload --reload-dir app --host 0.0.0.0 --port 8000 > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$RUN_DIR/backend.pid"

# Give backend time to start
sleep 3

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check logs:"
    cat "$BACKEND_LOG"
    exit 1
fi
echo "✅ Backend started (PID: $BACKEND_PID)"

# Start frontend
echo ""
echo "📦 Starting frontend server..."
cd "$PROJECT_ROOT/frontend"

if [ ! -d "node_modules" ]; then
    echo "❌ Frontend not set up. Run ./setup.sh first."
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend in background
FRONTEND_LOG="$LOG_DIR/frontend.log"
echo "Frontend starting at http://localhost:5173"
npm run dev > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$RUN_DIR/frontend.pid"

# Give frontend time to start
sleep 3

# Check if frontend started
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start. Check logs:"
    cat "$FRONTEND_LOG"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi
echo "✅ Frontend started (PID: $FRONTEND_PID)"

# Get the actual frontend port from the log
FRONTEND_PORT=$(grep -o "localhost:[0-9]*" "$FRONTEND_LOG" 2>/dev/null | head -1 | cut -d: -f2)
FRONTEND_PORT=${FRONTEND_PORT:-5173}

echo ""
echo "════════════════════════════════════════════"
echo "✅ All servers started successfully!"
echo "════════════════════════════════════════════"
echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:$FRONTEND_PORT"
echo ""
echo "  Logs: $LOG_DIR/"
echo ""
echo "  Press Ctrl+C to stop all servers"
echo "════════════════════════════════════════════"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    sleep 1
    kill -9 $BACKEND_PID 2>/dev/null || true
    kill -9 $FRONTEND_PID 2>/dev/null || true
    rm -f "$RUN_DIR/backend.pid" "$RUN_DIR/frontend.pid"
    echo "✅ Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Tail logs to keep script running and show output
tail -f "$BACKEND_LOG" "$FRONTEND_LOG" 2>/dev/null || wait

