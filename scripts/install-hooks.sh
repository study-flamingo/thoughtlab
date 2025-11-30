#!/bin/bash
# Install git hooks for this repository

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔧 Installing git hooks..."

# Configure git to use our hooks directory
git config core.hooksPath .githooks

# Make hooks executable
chmod +x "$PROJECT_ROOT/.githooks/"* 2>/dev/null || true

echo "✅ Git hooks installed!"
echo ""
echo "The pre-commit hook will now check for:"
echo "  - .env files being committed"
echo "  - API keys (OpenAI, Anthropic, AWS, GitHub)"
echo "  - Private keys"
echo "  - Large files (>1MB)"

