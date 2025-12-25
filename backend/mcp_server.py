#!/usr/bin/env python
"""ThoughtLab MCP Server Entry Point.

This script starts the ThoughtLab MCP server, exposing knowledge graph
operations as MCP tools for Claude Desktop and other MCP clients.

Usage:
    python mcp_server.py

Environment Variables:
    THOUGHTLAB_API_BASE_URL: Base URL for backend API (default: http://localhost:8000/api/v1)
    THOUGHTLAB_API_TIMEOUT: API request timeout in seconds (default: 30.0)
    THOUGHTLAB_OPENAI_API_KEY: OpenAI API key (required for backend operations)

The server runs on stdio transport by default, suitable for Claude Desktop.
"""

import logging
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.mcp import create_mcp_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr)  # Log to stderr for MCP
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run the ThoughtLab MCP server."""
    logger.info("Starting ThoughtLab MCP Server...")

    # Create MCP server
    mcp = create_mcp_server("ThoughtLab")

    # Run the server (stdio transport by default)
    logger.info("MCP Server running on stdio transport")
    mcp.run()


if __name__ == "__main__":
    main()
