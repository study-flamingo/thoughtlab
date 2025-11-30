#!/bin/bash
# Research Connection Graph - Restart Script
# Stops and then starts backend and frontend servers

set -e

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
sleep 3

echo ""

# Start servers (pass through any flags)
"$PROJECT_ROOT/start.sh" "$@"

