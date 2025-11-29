#!/bin/bash
# Research Connection Graph - Restart Script
# Stops and then starts backend and frontend servers
# Ensures clean restart by clearing Python cache

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔄 Restarting Research Connection Graph..."
echo ""

# Stop servers first
"$SCRIPT_DIR/stop.sh"

echo ""
echo "🧹 Clearing Python bytecode cache for clean restart..."
cd "$PROJECT_ROOT/backend"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
cd "$PROJECT_ROOT"

echo ""
echo "Waiting 3 seconds for ports to be released..."
sleep 3

# Verify ports are free
check_port() {
    local port=$1
    if command -v lsof > /dev/null 2>&1 && lsof -i :$port > /dev/null 2>&1; then
        return 1
    elif command -v netstat > /dev/null 2>&1 && netstat -an 2>/dev/null | grep -q ":$port.*LISTEN"; then
        return 1
    elif command -v ss > /dev/null 2>&1 && ss -tuln 2>/dev/null | grep -q ":$port"; then
        return 1
    fi
    return 0
}

# Wait for ports to be released (with timeout)
MAX_WAIT=10
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    PORT_8000_FREE=0
    PORT_5173_FREE=0
    
    if check_port 8000; then PORT_8000_FREE=1; fi
    if check_port 5173; then PORT_5173_FREE=1; fi
    
    if [ $PORT_8000_FREE -eq 1 ] && [ $PORT_5173_FREE -eq 1 ]; then
        echo "✅ Ports are free"
        break
    fi
    
    echo "⏳ Waiting for ports to be released... ($WAITED/$MAX_WAIT)"
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ $WAITED -eq $MAX_WAIT ]; then
    echo "⚠️  Ports may still be in use, attempting to start anyway..."
fi

echo ""

# Start servers (pass through any flags, e.g., --detached)
"$SCRIPT_DIR/start.sh" "$@"
