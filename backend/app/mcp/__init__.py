"""MCP server for ThoughtLab.

This module provides a Model Context Protocol (MCP) server that exposes
ThoughtLab's knowledge graph operations as MCP tools.

The MCP server calls ToolService directly (in-process) for:
- Sync tools: Immediate execution
- Async tools: Queue job, poll with check_job_status
- Dangerous tools: Gated by THOUGHTLAB_MCP_ADMIN_MODE

Usage:
    # Mount in FastAPI at /mcp
    from app.mcp import create_mcp_server

    mcp = create_mcp_server()
    app.mount("/mcp", mcp.http_app())
"""

from app.mcp.server import create_mcp_server
from app.mcp.mcp_tools import register_mcp_tools, ADMIN_MODE

__all__ = [
    "create_mcp_server",
    "register_mcp_tools",
    "ADMIN_MODE",
]
