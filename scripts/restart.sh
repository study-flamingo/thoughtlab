#!/bin/bash
# Research Connection Graph - Restart Script
# Stops and then starts backend and frontend servers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Restarting Research Connection Graph..."
echo ""

# Stop servers first
"$SCRIPT_DIR/stop.sh"

echo ""
echo "Waiting 2 seconds before starting..."
sleep 2

# Start servers
"$SCRIPT_DIR/start.sh"
