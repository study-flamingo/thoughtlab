#!/bin/bash
# Enter ThoughtLab development environment with all CLI tools

echo "🚀 Starting ThoughtLab dev environment..."
echo "This includes: git, gh (GitHub CLI), railway CLI, node, python, uv"
echo ""

# Check if running in OpenClaw/sandbox
if [ -n "$OPENCLAW_SANDBOX" ]; then
    echo "⚠️  Running in OpenClaw sandbox - some features may be limited"
    echo "For full access to git/gh/railway, run this script on the host machine."
    echo ""
fi

# Run the dev container
docker compose run --rm dev "$@"
