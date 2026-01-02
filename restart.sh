#!/bin/bash
# Research Connection Graph - Restart Script
# Stops and then starts backend and frontend servers

set -e

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
check_command sleep

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Restarting Research Connection Graph..."
echo ""

# Stop servers first
"$PROJECT_ROOT/stop.sh"

echo ""
echo "🧹 Clearing Python bytecode cache..."
cd "$PROJECT_ROOT/backend"
rm -rf app/__pycache__ app/**/__pycache__ 2>/dev/null || true
cd "$PROJECT_ROOT"

echo ""
echo "⏳ Waiting for ports to be released..."
sleep 2

# Double-check that port 5173 is free
echo "🔍 Verifying port 5173 is available..."
if lsof -i :5173 > /dev/null 2>&1 || netstat -ano 2>/dev/null | grep -E "LISTENING.*:5173\b" > /dev/null 2>&1; then
    echo "⚠️  Port 5173 still in use. Forcing cleanup..."
    "$PROJECT_ROOT/stop.sh" > /dev/null 2>&1 || true
    sleep 2
fi

echo ""

# Start servers (pass through any flags)
"$PROJECT_ROOT/start.sh" "$@"
