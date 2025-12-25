"""MCP server for ThoughtLab.

This module provides a Model Context Protocol (MCP) server that exposes
ThoughtLab's knowledge graph operations as MCP tools.

The MCP server calls the backend API via HTTP, maintaining the same
architecture as the LangGraph agent layer.
"""

from app.mcp.server import create_mcp_server

__all__ = ["create_mcp_server"]
